//! Complete Incremental Axis-Factored Layout Engine (`IALE`) + Fenwick Variable-Height Virtualizer.
//!
//! Features:
//! - 128 B/node total dense SoA footprint (`TopoNode` 16B + `LayoutSpec` 64B + `ComputedBox` 32B + `Clip` 16B)
//! - Non-recursive iterative DFS slices (scales to arbitrary depth without call-stack overflow)
//! - Unbounded exact water-filling `Grow` & `Shrink` distribution (hybrid 64-child stack buffer + reusable scratch)
//! - 1D `Row`, `Col`, `WrapRow` (per-line `cross_align`, `main_align`, `scroll_y`, `snap_to_pixels`), and
//!   2D `Grid` (`Fixed`, `Fit`, `Auto`, `Percent`, `MinMax`, `Fr` + 2D occupancy cell spans + overflow clamping)
//! - `MainAlign` (`Start`, `Center`, `End`, `SpaceBetween`, `SpaceEvenly`) & `CrossAlign` (incl. `FirstBaseline`)
//! - Both 2D Relayout Boundaries AND 1D Inline-Only Relayout Boundaries (skips X-passes on Y-only updates!)
//! - Pass 4 Viewport-Flipped & Clamped Anchored Floating Overlays (`Top*`, `Bottom*`, `Left*`, `Right*`)
//! - Exact `[W_tight, W_next_unwrap)` Width-Interval Text Wrapping integration + Subpixel Error-Diffused Snapping
//! - Dynamic $O(\log N)$ Fenwick `VirtualListIndex` (`push`, `pop`, `resize`, `set_height_anchored`, `visible_range`)

use crate::text::{FontId, FontSystem, ShapedParagraph};
use core::range::Range as CopyRange;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const NONE: Self = Self(u32::MAX);
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }
    #[inline]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SizingMode {
    Fixed = 0,
    Fit = 1,
    Grow = 2,
    Percent = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Axis {
    Row = 0,
    Col = 1,
    WrapRow = 2,
    Grid = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MainAlign {
    Start = 0,
    Center = 1,
    End = 2,
    SpaceBetween = 3,
    SpaceEvenly = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Align {
    Start = 0,
    Center = 1,
    End = 2,
    Stretch = 3,
    FirstBaseline = 4,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TopoNode {
    pub parent: NodeId,
    pub first_child: NodeId,
    pub next_sibling: NodeId,
    pub flags: u32,
}

const _: () = assert!(core::mem::size_of::<TopoNode>() == 16);

pub const FLAG_DIRTY_LAYOUT: u32 = 1 << 0;
pub const FLAG_DIRTY_PAINT: u32 = 1 << 1;
pub const FLAG_RELAYOUT_BOUNDARY: u32 = 1 << 2;
pub const FLAG_INLINE_BOUNDARY: u32 = 1 << 3;
pub const FLAG_CLIPS_CHILDREN: u32 = 1 << 4;
pub const FLAG_HAS_TEXT: u32 = 1 << 5;
pub const FLAG_HAS_GRID: u32 = 1 << 6;
pub const FLAG_REPAIRED_SPEC: u32 = 1 << 7;
pub const FLAG_ALIVE: u32 = 1 << 8;
pub const FLAG_LAID_OUT: u32 = 1 << 9;
pub const FLAG_FLOATING_ANCHOR: u32 = 1 << 10;
pub const FLAG_DIRTY_INLINE: u32 = 1 << 11;
pub const FLAG_SCROLLABLE: u32 = 1 << 12;
pub const FLAG_DIRTY_SCROLL: u32 = 1 << 13;
const FLAG_SOLVED_THIS_FRAME: u32 = 1 << 14;
const FLAG_HAS_BASELINE: u32 = 1 << 15;

/// Hot layout specification: EXACTLY 64 bytes (1 cache line), zero heap pointers.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(64))]
pub struct LayoutSpec {
    pub val_w: f32,
    pub val_h: f32,
    pub min_w: f32,
    pub max_w: f32,
    pub min_h: f32,
    pub max_h: f32,
    pub pad_l: f32,
    pub pad_t: f32,
    pub pad_r: f32,
    pub pad_b: f32,
    pub gap: f32,
    pub aspect_ratio: f32,
    pub scroll_y: f32,
    pub aux_id: u32,
    pub shrink_q8: u16, // Fixed-point 8.8 shrink weight (`256 == 1.0`)
    pub mode_w: SizingMode,
    pub mode_h: SizingMode,
    pub axis: Axis,
    pub cross_align: Align,
    pub main_align: MainAlign,
    pub grid_spans: u8, // low nibble = col_span - 1, high nibble = row_span - 1
}

const _: () = assert!(core::mem::size_of::<LayoutSpec>() == 64);

impl Default for LayoutSpec {
    #[inline]
    fn default() -> Self {
        Self {
            val_w: 0.0,
            val_h: 0.0,
            min_w: 0.0,
            max_w: f32::INFINITY,
            min_h: 0.0,
            max_h: f32::INFINITY,
            pad_l: 0.0,
            pad_t: 0.0,
            pad_r: 0.0,
            pad_b: 0.0,
            gap: 0.0,
            aspect_ratio: 0.0,
            scroll_y: 0.0,
            aux_id: u32::MAX,
            shrink_q8: 256,
            mode_w: SizingMode::Fit,
            mode_h: SizingMode::Fit,
            axis: Axis::Col,
            cross_align: Align::Start,
            main_align: MainAlign::Start,
            grid_spans: 0,
        }
    }
}

const WIDE_SPAN_SENTINEL: u8 = 0xFF;
const WIDE_SPAN_AUX_MASK: u32 = 0xFFFF_0000;
const WIDE_SPAN_AUX_TAG: u32 = 0x7FFE_0000;

impl LayoutSpec {
    #[inline]
    pub fn row() -> Self {
        Self {
            axis: Axis::Row,
            ..Self::default()
        }
    }

    #[inline]
    pub fn col() -> Self {
        Self {
            axis: Axis::Col,
            ..Self::default()
        }
    }

    #[inline]
    pub fn wrap_row() -> Self {
        Self {
            axis: Axis::WrapRow,
            ..Self::default()
        }
    }

    #[inline]
    pub fn fixed(mut self, w: f32, h: f32) -> Self {
        self.mode_w = SizingMode::Fixed;
        self.val_w = w;
        self.mode_h = SizingMode::Fixed;
        self.val_h = h;
        self
    }

    #[inline]
    pub fn width_fixed(mut self, w: f32) -> Self {
        self.mode_w = SizingMode::Fixed;
        self.val_w = w;
        self
    }

    #[inline]
    pub fn height_fixed(mut self, h: f32) -> Self {
        self.mode_h = SizingMode::Fixed;
        self.val_h = h;
        self
    }

    #[inline]
    pub fn width_percent(mut self, frac: f32) -> Self {
        self.mode_w = SizingMode::Percent;
        self.val_w = frac;
        self
    }

    #[inline]
    pub fn height_percent(mut self, frac: f32) -> Self {
        self.mode_h = SizingMode::Percent;
        self.val_h = frac;
        self
    }

    #[inline]
    pub fn width_grow(mut self, weight: f32) -> Self {
        self.mode_w = SizingMode::Grow;
        self.val_w = weight;
        self
    }

    #[inline]
    pub fn height_grow(mut self, weight: f32) -> Self {
        self.mode_h = SizingMode::Grow;
        self.val_h = weight;
        self
    }

    #[inline]
    pub fn grow(self, weight: f32) -> Self {
        self.width_grow(weight).height_grow(weight)
    }

    #[inline]
    pub fn pad(mut self, p: f32) -> Self {
        self.pad_l = p;
        self.pad_t = p;
        self.pad_r = p;
        self.pad_b = p;
        self
    }

    #[inline]
    pub fn pad_xy(mut self, px: f32, py: f32) -> Self {
        self.pad_l = px;
        self.pad_r = px;
        self.pad_t = py;
        self.pad_b = py;
        self
    }

    #[inline]
    pub fn gap(mut self, g: f32) -> Self {
        self.gap = g;
        self
    }

    #[inline]
    pub fn align(mut self, a: Align) -> Self {
        self.cross_align = a;
        self
    }

    #[inline]
    pub fn justify(mut self, m: MainAlign) -> Self {
        self.main_align = m;
        self
    }

    #[inline]
    pub fn min_size(mut self, min_w: f32, min_h: f32) -> Self {
        self.min_w = min_w;
        self.min_h = min_h;
        self
    }

    #[inline]
    pub fn max_size(mut self, max_w: f32, max_h: f32) -> Self {
        self.max_w = max_w;
        self.max_h = max_h;
        self
    }

    #[inline]
    pub fn aspect(mut self, ratio_w_over_h: f32) -> Self {
        self.aspect_ratio = ratio_w_over_h;
        self
    }

    #[inline]
    pub fn grid_span(mut self, col_span: u8, row_span: u8) -> Self {
        let cs = col_span.max(1);
        let rs = row_span.max(1);
        if cs <= 15 && rs <= 15 {
            self.grid_spans = (cs - 1) | ((rs - 1) << 4);
            if (self.aux_id & WIDE_SPAN_AUX_MASK) == WIDE_SPAN_AUX_TAG {
                self.aux_id = u32::MAX;
            }
        } else {
            // Lossless wide-span encoding (`16..=255`) keeping `size_of::<LayoutSpec>() == 64`:
            self.grid_spans = WIDE_SPAN_SENTINEL;
            self.aux_id = WIDE_SPAN_AUX_TAG | (cs as u32) | ((rs as u32) << 8);
        }
        self
    }

    #[inline]
    const fn wide_grid_span_pair(&self) -> Option<(u8, u8)> {
        if self.grid_spans == WIDE_SPAN_SENTINEL
            && (self.aux_id & WIDE_SPAN_AUX_MASK) == WIDE_SPAN_AUX_TAG
        {
            let raw_c = (self.aux_id & 0xFF) as u8;
            let raw_r = ((self.aux_id >> 8) & 0xFF) as u8;
            Some((
                if raw_c == 0 { 1 } else { raw_c },
                if raw_r == 0 { 1 } else { raw_r },
            ))
        } else {
            None
        }
    }

    #[inline]
    pub const fn col_span(&self) -> usize {
        if let Some((cs, _)) = self.wide_grid_span_pair() {
            cs as usize
        } else {
            ((self.grid_spans & 0x0F) as usize) + 1
        }
    }

    #[inline]
    pub const fn row_span(&self) -> usize {
        if let Some((_, rs)) = self.wide_grid_span_pair() {
            rs as usize
        } else {
            ((self.grid_spans >> 4) as usize) + 1
        }
    }

    #[inline]
    pub fn shrink_weight(&self) -> f32 {
        (self.shrink_q8 as f32) * (1.0 / 256.0)
    }

    /// Normalize malformed constraints (`NaN`, `Inf` on fixed sizes, negative gaps/padding, `min > max`).
    #[inline]
    pub fn sanitize(&mut self) -> bool {
        let mut repaired = false;
        let fix = |v: &mut f32, rep: &mut bool| {
            if !v.is_finite() || *v < 0.0 {
                *v = 0.0;
                *rep = true;
            } else if *v == 0.0 {
                *v = 0.0; // Canonicalize `-0.0` to `+0.0`
            }
        };
        fix(&mut self.pad_l, &mut repaired);
        fix(&mut self.pad_t, &mut repaired);
        fix(&mut self.pad_r, &mut repaired);
        fix(&mut self.pad_b, &mut repaired);
        fix(&mut self.gap, &mut repaired);
        fix(&mut self.min_w, &mut repaired);
        fix(&mut self.min_h, &mut repaired);
        fix(&mut self.val_w, &mut repaired);
        fix(&mut self.val_h, &mut repaired);
        fix(&mut self.aspect_ratio, &mut repaired);

        if self.max_w.is_nan() || self.max_w < 0.0 {
            self.max_w = f32::INFINITY;
            repaired = true;
        } else if self.max_w == 0.0 {
            self.max_w = 0.0;
        }
        if self.max_h.is_nan() || self.max_h < 0.0 {
            self.max_h = f32::INFINITY;
            repaired = true;
        } else if self.max_h == 0.0 {
            self.max_h = 0.0;
        }
        if self.max_w < self.min_w {
            self.max_w = self.min_w;
            repaired = true;
        }
        if self.max_h < self.min_h {
            self.max_h = self.min_h;
            repaired = true;
        }
        if !self.scroll_y.is_finite() {
            self.scroll_y = 0.0;
            repaired = true;
        } else if self.scroll_y == 0.0 {
            self.scroll_y = 0.0;
        }
        repaired
    }
}

/// Hot computed geometry record: 32 bytes (2 nodes per cache line).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C, align(32))]
pub struct ComputedBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub fit_w: f32,
    pub fit_h: f32,
    pub min_content_w: f32,
    pub baseline: f32,
}

const _: () = assert!(core::mem::size_of::<ComputedBox>() == 32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackSize {
    Fixed(f32),
    Fit,
    Auto,
    Percent(f32),
    MinMax(f32, f32),
    Fr(f32),
}

impl TrackSize {
    #[inline]
    fn sanitized(self) -> Self {
        let clean_nonneg = |v: f32| if v.is_finite() && v >= 0.0 { v } else { 0.0 };
        match self {
            Self::Fixed(fx) => Self::Fixed(clean_nonneg(fx)),
            Self::Fit => Self::Fit,
            Self::Auto => Self::Auto,
            Self::Percent(p) => Self::Percent(clean_nonneg(p)),
            Self::MinMax(lo, hi) => {
                let lo_c = clean_nonneg(lo);
                let hi_c = if hi.is_nan() || hi < 0.0 {
                    f32::INFINITY
                } else {
                    hi.max(lo_c)
                };
                Self::MinMax(lo_c, hi_c)
            }
            Self::Fr(fr) => Self::Fr(clean_nonneg(fr)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GridPlacement {
    node: NodeId,
    col: u16,
    row: u16,
    col_span: u8,
    row_span: u8,
}

#[derive(Clone, Debug)]
pub struct GridSpec {
    pub cols: Vec<TrackSize>,
    pub rows: Vec<TrackSize>,
    pub resolved_cols: Vec<f32>,
    pub resolved_rows: Vec<f32>,
    pub col_offsets: Vec<f32>,
    pub row_offsets: Vec<f32>,
    placements: Vec<GridPlacement>,
    occupancy: Vec<bool>,
    placements_valid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorPlacement {
    BottomStart,
    BottomCenter,
    BottomEnd,
    TopStart,
    TopCenter,
    TopEnd,
    RightStart,
    RightCenter,
    RightEnd,
    LeftStart,
    LeftCenter,
    LeftEnd,
}

#[derive(Clone, Copy, Debug)]
pub struct AnchorSpec {
    pub overlay_node: NodeId,
    pub target_node: NodeId,
    pub placement: AnchorPlacement,
    pub offset_x: f32,
    pub offset_y: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LayoutMetrics {
    pub nodes_visited: u64,
    pub boundaries_solved: u64,
    pub inline_passes_skipped: u64,
    pub text_shaper_calls: u64,
    pub text_interval_hits: u64,
}

const INLINE_FLEX_CAP: usize = 64;

pub struct LayoutTree {
    pub topo: Vec<TopoNode>,
    pub specs: Vec<LayoutSpec>,
    pub boxes: Vec<ComputedBox>,
    pub clip_rects: Vec<[f32; 4]>,
    pub texts: Vec<ShapedParagraph>,
    pub grids: Vec<GridSpec>,
    pub anchors: Vec<AnchorSpec>,
    free_nodes: Vec<NodeId>,
    free_texts: Vec<u32>,
    free_grids: Vec<u32>,
    dfs_scratch: Vec<NodeId>,
    flex_scratch: Vec<u32>,
    pub solved_nodes_last_frame: Vec<NodeId>,
    stack_scratch: Vec<NodeId>,
    dirty_roots: Vec<NodeId>,
    dirty_scroll_roots: Vec<NodeId>,
    wide_spans: Vec<(NodeId, u8, u8)>,
    last_window_size: [f32; 2],
    last_snap_state: (bool, u32),
    pub root: NodeId,
    pub snap_to_pixels: bool,
    pub device_scale: f32,
    pub last_metrics: LayoutMetrics,
}

impl Default for LayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutTree {
    pub fn new() -> Self {
        Self {
            topo: Vec::new(),
            specs: Vec::new(),
            boxes: Vec::new(),
            clip_rects: Vec::new(),
            texts: Vec::new(),
            grids: Vec::new(),
            anchors: Vec::new(),
            free_nodes: Vec::new(),
            free_texts: Vec::new(),
            free_grids: Vec::new(),
            dfs_scratch: Vec::new(),
            flex_scratch: Vec::new(),
            solved_nodes_last_frame: Vec::new(),
            stack_scratch: Vec::new(),
            dirty_roots: Vec::new(),
            dirty_scroll_roots: Vec::new(),
            wide_spans: Vec::new(),
            last_window_size: [-1.0, -1.0],
            last_snap_state: (false, 0),
            root: NodeId::NONE,
            snap_to_pixels: false,
            device_scale: 1.0,
            last_metrics: LayoutMetrics::default(),
        }
    }

    /// Normalize an arbitrary display scale factor into a finite positive physical scale (`[1/64, 64.0]`, fallback `1.0`).
    #[inline]
    pub const fn sanitize_device_scale(scale: f32) -> f32 {
        if scale.is_finite() && scale > 0.0 {
            scale.clamp(1.0 / 64.0, 64.0)
        } else {
            1.0
        }
    }

    /// Set the physical device-pixel scale factor (`1.0`, `1.25`, `1.3333334`, `1.5`, `1.75`, `2.0`, etc.)
    /// used when `snap_to_pixels` is enabled, cleanly invalidating layout when the scale changes while active.
    pub fn set_device_scale(&mut self, scale: f32) {
        let clean = Self::sanitize_device_scale(scale);
        if self.device_scale.to_bits() == clean.to_bits() {
            return;
        }
        self.device_scale = clean;
        if self.snap_to_pixels {
            if !self.root.is_none() && (self.topo[self.root.idx()].flags & FLAG_ALIVE) != 0 {
                self.mark_layout_dirty_with_inline(self.root, true);
            }
            for a_i in 0..self.anchors.len() {
                let ov = self.anchors[a_i].overlay_node;
                if !ov.is_none() && (self.topo[ov.idx()].flags & FLAG_ALIVE) != 0 {
                    self.mark_layout_dirty_with_inline(ov, true);
                }
            }
        }
    }

    #[inline]
    fn snap_scale_pair(&self) -> (f32, f64) {
        let sf = Self::sanitize_device_scale(self.device_scale);
        if sf == 1.0 {
            (1.0, 1.0)
        } else {
            (sf, 1.0 / (sf as f64))
        }
    }

    #[inline]
    fn snap_px(v: f32, sf: f32, inv_sf: f64) -> f32 {
        let s = if sf == 1.0 {
            v.round()
        } else {
            (((v as f64) * (sf as f64)).round() * inv_sf) as f32
        };
        if s == 0.0 { 0.0 } else { s }
    }

    #[inline]
    fn phys_to_logical(phys: f64, sf: f32, inv_sf: f64) -> f32 {
        let s = if sf == 1.0 {
            phys as f32
        } else {
            (phys * inv_sf) as f32
        };
        if s == 0.0 { 0.0 } else { s }
    }

    #[inline]
    fn snap_ceil_px(v: f32, sf: f32, inv_sf: f64) -> f32 {
        if v <= 0.0 {
            return 0.0;
        }
        let phys = (v as f64) * (sf as f64);
        let snapped = (phys - 1e-4).ceil().max(0.0);
        let s = if sf == 1.0 {
            snapped as f32
        } else {
            (snapped * inv_sf) as f32
        };
        if s == 0.0 { 0.0 } else { s }
    }

    #[inline]
    pub fn retained_payload_bytes(&self) -> usize {
        self.topo.len() * core::mem::size_of::<TopoNode>()
            + self.specs.len() * core::mem::size_of::<LayoutSpec>()
            + self.boxes.len() * core::mem::size_of::<ComputedBox>()
            + self.clip_rects.len() * core::mem::size_of::<[f32; 4]>()
    }

    pub fn alloc_node(&mut self, mut spec: LayoutSpec) -> NodeId {
        let repaired = spec.sanitize();
        let mut flags = FLAG_ALIVE | FLAG_DIRTY_PAINT;
        if repaired {
            flags |= FLAG_REPAIRED_SPEC;
        }
        Self::update_boundary_flags(&spec, NodeId::NONE, &mut flags);

        let wide_pair = spec.wide_grid_span_pair();
        let id = if let Some(id) = self.free_nodes.pop() {
            let idx = id.idx();
            self.topo[idx] = TopoNode {
                parent: NodeId::NONE,
                first_child: NodeId::NONE,
                next_sibling: NodeId::NONE,
                flags,
            };
            self.specs[idx] = spec;
            self.boxes[idx] = ComputedBox::default();
            self.clip_rects[idx] = [0.0, 0.0, f32::INFINITY, f32::INFINITY];
            id
        } else {
            let id = NodeId(self.topo.len() as u32);
            self.topo.push(TopoNode {
                parent: NodeId::NONE,
                first_child: NodeId::NONE,
                next_sibling: NodeId::NONE,
                flags,
            });
            self.specs.push(spec);
            self.boxes.push(ComputedBox::default());
            self.clip_rects
                .push([0.0, 0.0, f32::INFINITY, f32::INFINITY]);
            id
        };
        self.sync_wide_span_table(id, spec.grid_spans, wide_pair);
        id
    }

    #[inline]
    fn sync_wide_span_table(&mut self, id: NodeId, grid_spans: u8, wide_pair: Option<(u8, u8)>) {
        if grid_spans == WIDE_SPAN_SENTINEL {
            if let Some((cs, rs)) = wide_pair {
                if let Some(entry) = self.wide_spans.iter_mut().find(|e| e.0 == id) {
                    entry.1 = cs;
                    entry.2 = rs;
                } else {
                    self.wide_spans.push((id, cs, rs));
                }
            }
        } else if !self.wide_spans.is_empty() {
            self.wide_spans.retain(|e| e.0 != id);
        }
    }

    #[inline]
    fn node_grid_spans(&self, id: NodeId) -> (usize, usize) {
        let spec = self.specs[id.idx()];
        if spec.grid_spans != WIDE_SPAN_SENTINEL {
            (
                ((spec.grid_spans & 0x0F) as usize) + 1,
                ((spec.grid_spans >> 4) as usize) + 1,
            )
        } else if let Some((cs, rs)) = spec.wide_grid_span_pair() {
            (cs as usize, rs as usize)
        } else if let Some(&(_, cs, rs)) = self.wide_spans.iter().find(|e| e.0 == id) {
            (cs as usize, rs as usize)
        } else {
            (16, 16)
        }
    }

    #[inline]
    fn update_boundary_flags(spec: &LayoutSpec, parent: NodeId, flags: &mut u32) {
        *flags &= !(FLAG_RELAYOUT_BOUNDARY | FLAG_INLINE_BOUNDARY);
        if (*flags & FLAG_FLOATING_ANCHOR) != 0 {
            *flags |= FLAG_RELAYOUT_BOUNDARY | FLAG_INLINE_BOUNDARY;
            return;
        }
        let inline_owned =
            matches!(spec.mode_w, SizingMode::Fixed | SizingMode::Percent) || parent.is_none();
        let block_owned =
            matches!(spec.mode_h, SizingMode::Fixed | SizingMode::Percent) || parent.is_none();
        if inline_owned && block_owned {
            *flags |= FLAG_RELAYOUT_BOUNDARY | FLAG_INLINE_BOUNDARY;
        } else if inline_owned {
            *flags |= FLAG_INLINE_BOUNDARY;
        }
    }

    /// Attach or update text using `FontSystem` (real OpenType `rustybuzz` shaping when `font_id != FALLBACK`,
    /// zero-allocation in-place fast path for `FontId::FALLBACK`).
    pub fn attach_shaped_text(
        &mut self,
        id: NodeId,
        text: &str,
        font_id: FontId,
        font_size: f32,
        line_h: f32,
        fonts: &FontSystem,
    ) {
        let idx = id.idx();
        let new_hash = crate::text::hash_text_style(text, font_id, font_size, line_h);
        if (self.topo[idx].flags & FLAG_HAS_TEXT) != 0 {
            let tid = self.specs[idx].aux_id as usize;
            if self.texts[tid].content_hash == new_hash && self.texts[tid].font_id == font_id {
                return;
            }
            if font_id == FontId::FALLBACK {
                self.texts[tid].sync_fallback_in_place(text, font_size, line_h);
            } else {
                self.texts[tid] = fonts.shape_paragraph(text, font_id, font_size, line_h);
            }
            self.mark_layout_dirty_with_inline(id, true);
        } else if let Some(tid) = self.free_texts.pop() {
            if font_id == FontId::FALLBACK {
                self.texts[tid as usize].content_hash = 0;
                self.texts[tid as usize].sync_fallback_in_place(text, font_size, line_h);
            } else {
                self.texts[tid as usize] = fonts.shape_paragraph(text, font_id, font_size, line_h);
            }
            self.specs[idx].aux_id = tid;
            self.topo[idx].flags |= FLAG_HAS_TEXT;
            self.mark_layout_dirty_with_inline(id, true);
        } else {
            let tid = self.texts.len() as u32;
            let para = fonts.shape_paragraph(text, font_id, font_size, line_h);
            self.texts.push(para);
            self.specs[idx].aux_id = tid;
            self.topo[idx].flags |= FLAG_HAS_TEXT;
            self.mark_layout_dirty_with_inline(id, true);
        }
    }

    pub fn attach_grid(&mut self, id: NodeId, mut cols: Vec<TrackSize>, mut rows: Vec<TrackSize>) {
        if cols.is_empty() {
            cols.push(TrackSize::Fr(1.0));
        } else {
            for t in &mut cols {
                *t = t.sanitized();
            }
        }
        if rows.is_empty() {
            rows.push(TrackSize::Fit);
        } else {
            for t in &mut rows {
                *t = t.sanitized();
            }
        }

        let idx = id.idx();
        self.specs[idx].axis = Axis::Grid;
        if (self.topo[idx].flags & FLAG_HAS_GRID) != 0 {
            let gid = self.specs[idx].aux_id as usize;
            if self.grids[gid].cols == cols && self.grids[gid].rows == rows {
                return;
            }
            let c_len = cols.len();
            let r_len = rows.len();
            let g = &mut self.grids[gid];
            g.cols = cols;
            g.rows = rows;
            g.resolved_cols.resize(c_len, 0.0);
            g.resolved_rows.resize(r_len, 0.0);
            g.col_offsets.resize(c_len, 0.0);
            g.row_offsets.resize(r_len, 0.0);
            g.placements_valid = false;
        } else if let Some(gid) = self.free_grids.pop() {
            let c_len = cols.len();
            let r_len = rows.len();
            let g = &mut self.grids[gid as usize];
            g.cols = cols;
            g.rows = rows;
            g.resolved_cols.resize(c_len, 0.0);
            g.resolved_rows.resize(r_len, 0.0);
            g.col_offsets.resize(c_len, 0.0);
            g.row_offsets.resize(r_len, 0.0);
            g.placements.clear();
            g.occupancy.clear();
            g.placements_valid = false;
            self.specs[idx].aux_id = gid;
            self.topo[idx].flags |= FLAG_HAS_GRID;
        } else {
            let c_len = cols.len();
            let r_len = rows.len();
            let gid = self.grids.len() as u32;
            self.grids.push(GridSpec {
                cols,
                rows,
                resolved_cols: vec![0.0; c_len],
                resolved_rows: vec![0.0; r_len],
                col_offsets: vec![0.0; c_len],
                row_offsets: vec![0.0; r_len],
                placements: Vec::new(),
                occupancy: Vec::new(),
                placements_valid: false,
            });
            self.specs[idx].aux_id = gid;
            self.topo[idx].flags |= FLAG_HAS_GRID;
        }
        self.mark_layout_dirty_with_inline(id, true);
    }

    pub fn set_floating_anchor(
        &mut self,
        overlay_node: NodeId,
        target_node: NodeId,
        placement: AnchorPlacement,
        offset_x: f32,
        offset_y: f32,
    ) {
        let offset_x = if offset_x.is_finite() { offset_x } else { 0.0 };
        let offset_y = if offset_y.is_finite() { offset_y } else { 0.0 };
        self.topo[overlay_node.idx()].flags |=
            FLAG_FLOATING_ANCHOR | FLAG_RELAYOUT_BOUNDARY | FLAG_INLINE_BOUNDARY;
        if let Some(existing) = self
            .anchors
            .iter_mut()
            .find(|a| a.overlay_node == overlay_node)
        {
            if existing.target_node == target_node
                && existing.placement == placement
                && (existing.offset_x - offset_x).abs() <= 1e-5
                && (existing.offset_y - offset_y).abs() <= 1e-5
            {
                return;
            }
            existing.target_node = target_node;
            existing.placement = placement;
            existing.offset_x = offset_x;
            existing.offset_y = offset_y;
        } else {
            self.anchors.push(AnchorSpec {
                overlay_node,
                target_node,
                placement,
                offset_x,
                offset_y,
            });
        }
        self.mark_layout_dirty_with_inline(overlay_node, true);
    }

    pub fn set_children(&mut self, parent: NodeId, children: &[NodeId]) {
        let p_idx = parent.idx();
        let mut cur = self.topo[p_idx].first_child;
        let mut same = true;
        for &ch in children {
            if cur != ch {
                same = false;
                break;
            }
            cur = self.topo[cur.idx()].next_sibling;
        }
        if same && cur.is_none() {
            return;
        }

        if (self.topo[p_idx].flags & FLAG_HAS_GRID) != 0 {
            let gid = self.specs[p_idx].aux_id as usize;
            self.grids[gid].placements_valid = false;
        }

        self.topo[p_idx].first_child = children.first().copied().unwrap_or(NodeId::NONE);
        for (i, &ch) in children.iter().enumerate() {
            let next = children.get(i + 1).copied().unwrap_or(NodeId::NONE);
            let c_idx = ch.idx();
            self.topo[c_idx].parent = parent;
            self.topo[c_idx].next_sibling = next;
            let spec = self.specs[c_idx];
            let mut flags = self.topo[c_idx].flags;
            Self::update_boundary_flags(&spec, parent, &mut flags);
            self.topo[c_idx].flags = flags;
        }
        self.mark_layout_dirty_with_inline(parent, true);
    }

    #[inline]
    pub fn set_scroll_y(&mut self, id: NodeId, scroll_y: f32) -> bool {
        let mut spec = self.specs[id.idx()];
        spec.scroll_y = scroll_y;
        self.update_spec(id, spec)
    }

    pub fn update_spec(&mut self, id: NodeId, mut new_spec: LayoutSpec) -> bool {
        let repaired = new_spec.sanitize();
        let idx = id.idx();
        if repaired {
            self.topo[idx].flags |= FLAG_REPAIRED_SPEC;
        }
        let old_spans = self.node_grid_spans(id);
        let incoming_wide = new_spec.wide_grid_span_pair();
        let new_spans = if new_spec.grid_spans != WIDE_SPAN_SENTINEL {
            (
                ((new_spec.grid_spans & 0x0F) as usize) + 1,
                ((new_spec.grid_spans >> 4) as usize) + 1,
            )
        } else if let Some((cs, rs)) = incoming_wide {
            (cs as usize, rs as usize)
        } else {
            old_spans
        };

        if (self.topo[idx].flags & (FLAG_HAS_TEXT | FLAG_HAS_GRID)) != 0
            && (new_spec.aux_id == u32::MAX
                || (new_spec.aux_id & WIDE_SPAN_AUX_MASK) == WIDE_SPAN_AUX_TAG)
        {
            new_spec.aux_id = self.specs[idx].aux_id;
        }
        if (self.topo[idx].flags & FLAG_HAS_GRID) != 0 {
            new_spec.axis = Axis::Grid;
        }
        let old = self.specs[idx];
        if old == new_spec && old_spans == new_spans {
            return false;
        }
        self.sync_wide_span_table(id, new_spec.grid_spans, incoming_wide);

        // Scroll-Only Fast Path (`FLAG_DIRTY_SCROLL`):
        // When ONLY `scroll_y` changed on an already-laid-out node that has no pending layout
        // invalidation, `w`, `h`, `fit_w`, `fit_h`, `min_content_w`, `baseline`, text wrapping,
        // and grid track sizes are 100% invariant—only descendant `(x, y)`, nested `clip_rects`,
        // and Pass 4 `AnchorPlacement` depend on `scroll_y`!
        let mut old_with_new_scroll = old;
        old_with_new_scroll.scroll_y = new_spec.scroll_y;
        if old_with_new_scroll == new_spec && old_spans == new_spans {
            self.specs[idx] = new_spec;
            let cur_flags = self.topo[idx].flags;
            if (cur_flags & FLAG_LAID_OUT) != 0 {
                if (cur_flags & FLAG_DIRTY_LAYOUT) == 0 && (cur_flags & FLAG_DIRTY_SCROLL) == 0 {
                    self.topo[idx].flags = cur_flags | FLAG_DIRTY_SCROLL;
                    self.dirty_scroll_roots.push(id);
                }
                return true;
            }
        }

        let parent = self.topo[idx].parent;
        if old_spans != new_spans
            && !parent.is_none()
            && (self.topo[parent.idx()].flags & FLAG_HAS_GRID) != 0
        {
            let gid = self.specs[parent.idx()].aux_id as usize;
            self.grids[gid].placements_valid = false;
        }

        // Detect whether ONLY vertical/block properties (`val_h`, `min_h`, `max_h`, `scroll_y`, `pad_t`, `pad_b`) changed.
        // If inline properties are identical inside an `INLINE_BOUNDARY`, Passes 1A & 1B can be skipped!
        let height_drives_inline = (old.aspect_ratio > 0.0 || new_spec.aspect_ratio > 0.0)
            && ((old.aspect_ratio - new_spec.aspect_ratio).abs() > 1e-6
                || old.mode_h != new_spec.mode_h
                || (old.val_h - new_spec.val_h).abs() > 1e-6);
        let inline_changed = old.mode_w != new_spec.mode_w
            || (old.val_w - new_spec.val_w).abs() > 1e-6
            || (old.min_w - new_spec.min_w).abs() > 1e-6
            || (old.max_w - new_spec.max_w).abs() > 1e-6
            || (old.pad_l - new_spec.pad_l).abs() > 1e-6
            || (old.pad_r - new_spec.pad_r).abs() > 1e-6
            || (old.gap - new_spec.gap).abs() > 1e-6
            || old.axis != new_spec.axis
            || old.shrink_q8 != new_spec.shrink_q8
            || old_spans != new_spans
            || (new_spec.axis == Axis::Col
                && (old.cross_align == Align::Stretch) != (new_spec.cross_align == Align::Stretch))
            || height_drives_inline;

        let old_flags = self.topo[idx].flags;
        self.specs[idx] = new_spec;
        let mut new_flags = old_flags;
        if repaired {
            new_flags |= FLAG_REPAIRED_SPEC;
        }
        Self::update_boundary_flags(&new_spec, parent, &mut new_flags);
        self.topo[idx].flags = new_flags;

        if !parent.is_none()
            && ((old_flags & FLAG_FLOATING_ANCHOR) == 0)
            && ((old_flags & FLAG_RELAYOUT_BOUNDARY) != 0
                || (new_flags & FLAG_RELAYOUT_BOUNDARY) == 0)
        {
            self.mark_layout_dirty_with_inline(parent, inline_changed);
        }
        self.mark_layout_dirty_with_inline(id, inline_changed);
        true
    }

    #[inline]
    pub fn mark_layout_dirty(&mut self, id: NodeId) {
        self.mark_layout_dirty_with_inline(id, true);
    }

    pub fn mark_layout_dirty_with_inline(&mut self, mut id: NodeId, inline_dirty: bool) {
        while !id.is_none() {
            let idx = id.idx();
            let had_inline = (self.topo[idx].flags & FLAG_DIRTY_INLINE) != 0;
            if inline_dirty {
                self.topo[idx].flags |= FLAG_DIRTY_INLINE;
            }
            if (self.topo[idx].flags & FLAG_DIRTY_LAYOUT) != 0 {
                // If already dirty for block-only and now inline became dirty, continue propagating `FLAG_DIRTY_INLINE` up to boundary!
                if inline_dirty
                    && !had_inline
                    && (self.topo[idx].flags & FLAG_RELAYOUT_BOUNDARY) == 0
                    && !self.topo[idx].parent.is_none()
                {
                    id = self.topo[idx].parent;
                    continue;
                }
                return;
            }
            self.topo[idx].flags |= FLAG_DIRTY_LAYOUT;
            if (self.topo[idx].flags & FLAG_RELAYOUT_BOUNDARY) != 0
                || self.topo[idx].parent.is_none()
            {
                self.dirty_roots.push(id);
                return;
            }
            id = self.topo[idx].parent;
        }
    }

    pub fn free_subtree(&mut self, root: NodeId) {
        if root.is_none() {
            return;
        }
        self.stack_scratch.clear();
        self.stack_scratch.push(root);
        while let Some(id) = self.stack_scratch.pop() {
            let idx = id.idx();
            if (self.topo[idx].flags & FLAG_ALIVE) == 0 {
                continue;
            }
            let mut ch = self.topo[idx].first_child;
            while !ch.is_none() {
                self.stack_scratch.push(ch);
                ch = self.topo[ch.idx()].next_sibling;
            }
            if (self.topo[idx].flags & FLAG_HAS_TEXT) != 0 {
                self.free_texts.push(self.specs[idx].aux_id);
            }
            if (self.topo[idx].flags & FLAG_HAS_GRID) != 0 {
                self.free_grids.push(self.specs[idx].aux_id);
            }
            if !self.anchors.is_empty() {
                self.anchors
                    .retain(|a| a.overlay_node != id && a.target_node != id);
            }
            if !self.wide_spans.is_empty() {
                self.wide_spans.retain(|e| e.0 != id);
            }
            if self.root == id {
                self.root = NodeId::NONE;
            }
            self.topo[idx].flags = 0;
            self.topo[idx].first_child = NodeId::NONE;
            self.topo[idx].next_sibling = NodeId::NONE;
            self.topo[idx].parent = NodeId::NONE;
            self.free_nodes.push(id);
        }
    }

    pub fn compute_layout(&mut self, window_w: f32, window_h: f32) {
        self.last_metrics = LayoutMetrics::default();
        for &prev_id in &self.solved_nodes_last_frame {
            if let Some(t) = self.topo.get_mut(prev_id.idx()) {
                t.flags &= !FLAG_SOLVED_THIS_FRAME;
            }
        }
        self.solved_nodes_last_frame.clear();
        if self.root.is_none() || (self.topo[self.root.idx()].flags & FLAG_ALIVE) == 0 {
            return;
        }
        let r_idx = self.root.idx();
        let win_w = if window_w.is_finite() && window_w >= 0.0 {
            window_w
        } else {
            0.0
        };
        let win_h = if window_h.is_finite() && window_h >= 0.0 {
            window_h
        } else {
            0.0
        };

        let (sf, inv_sf) = self.snap_scale_pair();
        let cur_snap_state = (
            self.snap_to_pixels,
            if self.snap_to_pixels { sf.to_bits() } else { 0 },
        );

        self.topo[r_idx].flags |= FLAG_RELAYOUT_BOUNDARY | FLAG_INLINE_BOUNDARY;
        if (self.topo[r_idx].flags & FLAG_LAID_OUT) == 0
            || (self.last_window_size[0] - win_w).abs() > 1e-5
            || (self.last_window_size[1] - win_h).abs() > 1e-5
            || self.last_snap_state != cur_snap_state
        {
            self.last_window_size = [win_w, win_h];
            self.last_snap_state = cur_snap_state;
            self.mark_layout_dirty_with_inline(self.root, true);
            for a_i in 0..self.anchors.len() {
                let ov = self.anchors[a_i].overlay_node;
                if (self.topo[ov.idx()].flags & FLAG_ALIVE) != 0 {
                    self.mark_layout_dirty_with_inline(ov, true);
                }
            }
        }

        let mut roots = core::mem::take(&mut self.dirty_roots);
        for &cand in &roots {
            let mut boundary = cand;
            let b_idx = boundary.idx();
            if (self.topo[b_idx].flags & FLAG_ALIVE) == 0
                || (self.topo[b_idx].flags & FLAG_DIRTY_LAYOUT) == 0
            {
                continue;
            }
            while !self.topo[boundary.idx()].parent.is_none()
                && (self.topo[boundary.idx()].flags & FLAG_RELAYOUT_BOUNDARY) == 0
            {
                boundary = self.topo[boundary.idx()].parent;
                self.topo[boundary.idx()].flags |= FLAG_DIRTY_LAYOUT | FLAG_DIRTY_INLINE;
            }

            // A floating anchor is an independent layout root (skipped by ancestor `solve_boundary` walks),
            // so dirty state above a `FLAG_FLOATING_ANCHOR` boundary never subsumes solving it!
            let mut ancestor_dirty = false;
            if (self.topo[boundary.idx()].flags & FLAG_FLOATING_ANCHOR) == 0 {
                let mut cur = boundary;
                while !self.topo[cur.idx()].parent.is_none() {
                    if (self.topo[cur.idx()].flags & FLAG_FLOATING_ANCHOR) != 0 {
                        break;
                    }
                    let anc = self.topo[cur.idx()].parent;
                    if (self.topo[anc.idx()].flags & FLAG_DIRTY_LAYOUT) != 0 {
                        ancestor_dirty = true;
                        break;
                    }
                    cur = anc;
                }
            }
            if !ancestor_dirty {
                self.solve_boundary(boundary);
            }
        }
        roots.clear();
        self.dirty_roots = roots;

        // =========================================================================
        // SCROLL-ONLY FAST PATH (`FLAG_DIRTY_SCROLL`):
        // Skips Passes 1A, 1B, 2, 3A, flex height water-filling, and grid track sizing,
        // recomputing exact Pass-3B `(x, y)` and `clip_rects` in a single top-down pass!
        // =========================================================================
        let mut scroll_roots = core::mem::take(&mut self.dirty_scroll_roots);
        for &cand in &scroll_roots {
            let c_idx = cand.idx();
            if (self.topo[c_idx].flags & FLAG_ALIVE) == 0
                || (self.topo[c_idx].flags & FLAG_DIRTY_SCROLL) == 0
            {
                continue;
            }
            let mut ancestor_scroll_dirty = false;
            if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                let mut cur = cand;
                while !self.topo[cur.idx()].parent.is_none() {
                    if (self.topo[cur.idx()].flags & FLAG_FLOATING_ANCHOR) != 0 {
                        break;
                    }
                    let anc = self.topo[cur.idx()].parent;
                    if (self.topo[anc.idx()].flags & FLAG_DIRTY_SCROLL) != 0 {
                        ancestor_scroll_dirty = true;
                        break;
                    }
                    cur = anc;
                }
            }
            if !ancestor_scroll_dirty {
                self.solve_scroll_only(cand);
            }
        }
        scroll_roots.clear();
        self.dirty_scroll_roots = scroll_roots;

        // =========================================================================
        // PASS 4: Viewport-Flipped & Clamped Anchored Floating Overlays
        // =========================================================================
        let win_clip = [0.0, 0.0, win_w, win_h];
        for a_i in 0..self.anchors.len() {
            let anchor = self.anchors[a_i];
            let ov = anchor.overlay_node;
            let tg = anchor.target_node;
            if (self.topo[ov.idx()].flags & FLAG_ALIVE) == 0
                || (self.topo[tg.idx()].flags & FLAG_ALIVE) == 0
            {
                continue;
            }
            if (self.topo[ov.idx()].flags & FLAG_LAID_OUT) == 0
                || (self.topo[ov.idx()].flags & FLAG_DIRTY_LAYOUT) != 0
            {
                self.solve_boundary(ov);
            } else if (self.topo[ov.idx()].flags & FLAG_DIRTY_SCROLL) != 0 {
                self.solve_scroll_only(ov);
            }
            let tb = self.boxes[tg.idx()];
            let ob = self.boxes[ov.idx()];
            let (mut ox, mut oy) = Self::compute_anchor_xy(anchor, tb, ob, win_w, win_h);
            let max_ox = (win_w - ob.w).max(0.0);
            let max_oy = (win_h - ob.h).max(0.0);
            ox = ox.clamp(0.0, max_ox);
            oy = oy.clamp(0.0, max_oy);
            if self.snap_to_pixels {
                if sf == 1.0 {
                    ox = ox.round().clamp(0.0, max_ox);
                    oy = oy.round().clamp(0.0, max_oy);
                } else {
                    let mut sx = Self::snap_px(ox, sf, inv_sf);
                    if sx > max_ox {
                        sx = Self::phys_to_logical(((max_ox as f64) * (sf as f64)).floor().max(0.0), sf, inv_sf);
                    }
                    let mut sy = Self::snap_px(oy, sf, inv_sf);
                    if sy > max_oy {
                        sy = Self::phys_to_logical(((max_oy as f64) * (sf as f64)).floor().max(0.0), sf, inv_sf);
                    }
                    ox = sx.clamp(0.0, max_ox);
                    oy = sy.clamp(0.0, max_oy);
                }
            }
            if ox == 0.0 {
                ox = 0.0;
            }
            if oy == 0.0 {
                oy = 0.0;
            }
            if ob.x.to_bits() != ox.to_bits()
                || ob.y.to_bits() != oy.to_bits()
                || self.clip_rects[ov.idx()] != win_clip
            {
                let already_counted = (self.topo[ov.idx()].flags & FLAG_SOLVED_THIS_FRAME) != 0;
                if !already_counted {
                    self.last_metrics.boundaries_solved += 1;
                }
                self.boxes[ov.idx()].x = ox;
                self.boxes[ov.idx()].y = oy;
                self.clip_rects[ov.idx()] = win_clip;
                let visited = self.reposition_subtree(ov);
                if !already_counted {
                    self.last_metrics.nodes_visited += visited;
                }
            }
        }
    }

    #[inline]
    fn compute_anchor_xy(
        anchor: AnchorSpec,
        tb: ComputedBox,
        ob: ComputedBox,
        win_w: f32,
        win_h: f32,
    ) -> (f32, f32) {
        let align_x_start = tb.x + anchor.offset_x;
        let align_x_center = tb.x + (tb.w - ob.w) * 0.5 + anchor.offset_x;
        let align_x_end = tb.x + tb.w - ob.w + anchor.offset_x;

        let align_y_start = tb.y + anchor.offset_y;
        let align_y_center = tb.y + (tb.h - ob.h) * 0.5 + anchor.offset_y;
        let align_y_end = tb.y + tb.h - ob.h + anchor.offset_y;

        let below_y = tb.y + tb.h + anchor.offset_y;
        let above_y = tb.y - ob.h - anchor.offset_y;
        let room_below = win_h - (tb.y + tb.h);
        let room_above = tb.y;

        let right_x = tb.x + tb.w + anchor.offset_x;
        let left_x = tb.x - ob.w - anchor.offset_x;
        let room_right = win_w - (tb.x + tb.w);
        let room_left = tb.x;

        let pick_bottom_or_flip = || {
            if below_y + ob.h > win_h && (above_y >= 0.0 || room_above > room_below) {
                above_y
            } else {
                below_y
            }
        };
        let pick_top_or_flip = || {
            if above_y < 0.0 && (below_y + ob.h <= win_h || room_below > room_above) {
                below_y
            } else {
                above_y
            }
        };
        let pick_right_or_flip = || {
            if right_x + ob.w > win_w && (left_x >= 0.0 || room_left > room_right) {
                left_x
            } else {
                right_x
            }
        };
        let pick_left_or_flip = || {
            if left_x < 0.0 && (right_x + ob.w <= win_w || room_right > room_left) {
                right_x
            } else {
                left_x
            }
        };

        match anchor.placement {
            AnchorPlacement::BottomStart => (align_x_start, pick_bottom_or_flip()),
            AnchorPlacement::BottomCenter => (align_x_center, pick_bottom_or_flip()),
            AnchorPlacement::BottomEnd => (align_x_end, pick_bottom_or_flip()),
            AnchorPlacement::TopStart => (align_x_start, pick_top_or_flip()),
            AnchorPlacement::TopCenter => (align_x_center, pick_top_or_flip()),
            AnchorPlacement::TopEnd => (align_x_end, pick_top_or_flip()),
            AnchorPlacement::RightStart => (pick_right_or_flip(), align_y_start),
            AnchorPlacement::RightCenter => (pick_right_or_flip(), align_y_center),
            AnchorPlacement::RightEnd => (pick_right_or_flip(), align_y_end),
            AnchorPlacement::LeftStart => (pick_left_or_flip(), align_y_start),
            AnchorPlacement::LeftCenter => (pick_left_or_flip(), align_y_center),
            AnchorPlacement::LeftEnd => (pick_left_or_flip(), align_y_end),
        }
    }

    fn solve_scroll_only(&mut self, scroll_root: NodeId) {
        self.last_metrics.boundaries_solved += 1;
        self.last_metrics.inline_passes_skipped += 1;
        let visited = self.reposition_subtree(scroll_root);
        self.last_metrics.nodes_visited += visited;
    }

    /// Exact memoryless top-down Pass-3B `(x, y)` and `clip_rects` solver for `subtree_root` and
    /// its non-floating descendants. Produces bit-identical `ComputedBox`es and `clip_rects` to a
    /// full from-scratch layout solve across nested clips, `snap_to_pixels`, `Row`, `Col`, `WrapRow`, and `Grid`.
    fn reposition_subtree(&mut self, subtree_root: NodeId) -> u64 {
        self.stack_scratch.clear();
        self.stack_scratch.push(subtree_root);
        let mut visited = 0u64;

        while let Some(id) = self.stack_scratch.pop() {
            visited += 1;
            let idx = id.idx();
            self.topo[idx].flags = (self.topo[idx].flags
                & !(FLAG_DIRTY_LAYOUT | FLAG_DIRTY_INLINE | FLAG_DIRTY_SCROLL))
                | FLAG_LAID_OUT;
            if (self.topo[idx].flags & FLAG_SOLVED_THIS_FRAME) == 0 {
                self.topo[idx].flags |= FLAG_SOLVED_THIS_FRAME;
                self.solved_nodes_last_frame.push(id);
            }

            let parent_clip = self.clip_rects[idx];
            let cur_clip = if (self.topo[idx].flags & FLAG_CLIPS_CHILDREN) != 0 {
                let b = self.boxes[idx];
                intersect_rect(parent_clip, [b.x, b.y, b.w, b.h])
            } else {
                parent_clip
            };

            let first = self.topo[idx].first_child;
            if first.is_none() {
                continue;
            }

            self.place_children_pass3b(idx, cur_clip, false);

            let mark = self.stack_scratch.len();
            let mut ch = first;
            while !ch.is_none() {
                let c_idx = ch.idx();
                if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                    self.stack_scratch.push(ch);
                }
                ch = self.topo[c_idx].next_sibling;
            }
            self.stack_scratch[mark..].reverse();
        }
        visited
    }

    fn solve_boundary(&mut self, boundary_root: NodeId) {
        self.last_metrics.boundaries_solved += 1;
        self.dfs_scratch.clear();
        self.stack_scratch.clear();
        self.stack_scratch.push(boundary_root);

        let mut any_inline_dirty = false;
        while let Some(id) = self.stack_scratch.pop() {
            let idx = id.idx();
            if (self.topo[idx].flags & FLAG_DIRTY_INLINE) != 0
                || (self.topo[idx].flags & FLAG_LAID_OUT) == 0
            {
                any_inline_dirty = true;
            }
            if (self.topo[idx].flags & FLAG_SOLVED_THIS_FRAME) == 0 {
                self.topo[idx].flags |= FLAG_SOLVED_THIS_FRAME;
                self.solved_nodes_last_frame.push(id);
            }
            self.dfs_scratch.push(id);
            let first = self.topo[idx].first_child;
            if !first.is_none() {
                let mark = self.stack_scratch.len();
                let mut ch = first;
                while !ch.is_none() {
                    // Floating overlays solve separately in Pass 4!
                    if (self.topo[ch.idx()].flags & FLAG_FLOATING_ANCHOR) == 0 {
                        self.stack_scratch.push(ch);
                    }
                    ch = self.topo[ch.idx()].next_sibling;
                }
                self.stack_scratch[mark..].reverse();
            }
        }

        let count = self.dfs_scratch.len();
        self.last_metrics.nodes_visited += count as u64;
        let span = CopyRange {
            start: 0usize,
            end: count,
        };

        let can_skip_inline = !any_inline_dirty
            && (self.topo[boundary_root.idx()].flags & FLAG_INLINE_BOUNDARY) != 0;

        let (sf, inv_sf) = self.snap_scale_pair();
        let snap = self.snap_to_pixels;

        if can_skip_inline {
            self.last_metrics.inline_passes_skipped += 1;
        } else {
            // =====================================================================
            // PASS 1A: Bottom-Up Inline (Width) Intrinsic Sizing
            // =====================================================================
            for pos in (span.start..span.end).rev() {
                let id = self.dfs_scratch[pos];
                let idx = id.idx();
                let spec = self.specs[idx];
                let flags = self.topo[idx].flags;
                let pad_x = spec.pad_l.algebraic_add(spec.pad_r);

                let (mut fit_w, mut min_cw) = if (flags & FLAG_HAS_TEXT) != 0 {
                    let t = &self.texts[spec.aux_id as usize];
                    (t.max_unwrapped_w + pad_x, t.min_word_w + pad_x)
                } else if spec.axis == Axis::Grid && (flags & FLAG_HAS_GRID) != 0 {
                    self.measure_grid_inline(idx, spec, pad_x)
                } else {
                    let mut sum_fit = 0.0_f32;
                    let mut max_fit = 0.0_f32;
                    let mut sum_min = 0.0_f32;
                    let mut max_min = 0.0_f32;
                    let mut n_children = 0u32;
                    let mut ch = self.topo[idx].first_child;
                    while !ch.is_none() {
                        if (self.topo[ch.idx()].flags & FLAG_FLOATING_ANCHOR) == 0 {
                            let cb = self.boxes[ch.idx()];
                            sum_fit = sum_fit.algebraic_add(cb.fit_w);
                            max_fit = max_fit.max(cb.fit_w);
                            sum_min = sum_min.algebraic_add(cb.min_content_w);
                            max_min = max_min.max(cb.min_content_w);
                            n_children += 1;
                        }
                        ch = self.topo[ch.idx()].next_sibling;
                    }
                    let gaps = (n_children.saturating_sub(1) as f32) * spec.gap;
                    match spec.axis {
                        Axis::Row => (sum_fit + gaps + pad_x, sum_min + gaps + pad_x),
                        Axis::WrapRow => (sum_fit + gaps + pad_x, max_min + pad_x),
                        Axis::Col | Axis::Grid => (max_fit + pad_x, max_min + pad_x),
                    }
                };

                if spec.mode_w == SizingMode::Fixed {
                    fit_w = spec.val_w;
                    min_cw = spec.val_w;
                } else if spec.aspect_ratio > 0.0 && spec.mode_h == SizingMode::Fixed {
                    // Height-driven aspect ratio with known fixed height!
                    fit_w = spec.val_h * spec.aspect_ratio;
                    min_cw = fit_w;
                } else if snap && spec.mode_w == SizingMode::Fit {
                    // Ceiling-snap intrinsic `Fit` widths to the physical pixel grid so `.round()` edge-snapping
                    // never rounds single-line buttons, badges, or labels <= 0.5 physical px narrower than `max_unwrapped_w`!
                    fit_w = Self::snap_ceil_px(fit_w, sf, inv_sf);
                }
                fit_w = fit_w.clamp(spec.min_w, spec.max_w);
                min_cw = min_cw.clamp(spec.min_w, spec.max_w);
                self.boxes[idx].fit_w = fit_w;
                self.boxes[idx].min_content_w = min_cw;
            }

            // Boundary root outer width:
            {
                let b_idx = boundary_root.idx();
                let b_spec = self.specs[b_idx];
                let b_flags = self.topo[b_idx].flags;
                let is_floating = (b_flags & FLAG_FLOATING_ANCHOR) != 0;
                if self.topo[b_idx].parent.is_none() && !is_floating {
                    let win_w = self.last_window_size[0].max(0.0);
                    let raw_w = match b_spec.mode_w {
                        SizingMode::Fixed => b_spec.val_w,
                        SizingMode::Grow => win_w,
                        SizingMode::Percent => win_w * b_spec.val_w,
                        SizingMode::Fit => self.boxes[b_idx].fit_w,
                    };
                    let w = raw_w.clamp(b_spec.min_w, b_spec.max_w);
                    self.boxes[b_idx].w = if snap { Self::snap_px(w, sf, inv_sf) } else { w };
                } else if is_floating {
                    let win_w = self.last_window_size[0].max(0.0);
                    let raw_w = match b_spec.mode_w {
                        SizingMode::Fixed => b_spec.val_w,
                        SizingMode::Percent => win_w * b_spec.val_w,
                        SizingMode::Fit | SizingMode::Grow => self.boxes[b_idx].fit_w,
                    };
                    let w = raw_w.clamp(b_spec.min_w, b_spec.max_w);
                    self.boxes[b_idx].w = if snap { Self::snap_px(w, sf, inv_sf) } else { w };
                } else if b_spec.mode_w == SizingMode::Fixed {
                    let w = b_spec.val_w.clamp(b_spec.min_w, b_spec.max_w);
                    let snapped_w = if snap { Self::snap_px(w, sf, inv_sf) } else { w };
                    let p_axis = self.specs[self.topo[b_idx].parent.idx()].axis;
                    let keep_diffused = snap
                        && (b_flags & FLAG_LAID_OUT) != 0
                        && p_axis == Axis::Row
                        && (self.boxes[b_idx].w - snapped_w).abs() * sf <= 1.05;
                    if !keep_diffused {
                        self.boxes[b_idx].w = snapped_w;
                    }
                } else if b_spec.mode_w == SizingMode::Percent {
                    let p_idx = self.topo[b_idx].parent.idx();
                    let p_spec = self.specs[p_idx];
                    if p_spec.axis != Axis::Grid || (b_flags & FLAG_LAID_OUT) == 0 {
                        let p_inner = (self.boxes[p_idx].w - p_spec.pad_l - p_spec.pad_r).max(0.0);
                        let w = (p_inner * b_spec.val_w).clamp(b_spec.min_w, b_spec.max_w);
                        let snapped_w = if snap { Self::snap_px(w, sf, inv_sf) } else { w };
                        let keep_diffused = snap
                            && (b_flags & FLAG_LAID_OUT) != 0
                            && p_spec.axis == Axis::Row
                            && (self.boxes[b_idx].w - snapped_w).abs() * sf <= 1.05;
                        if !keep_diffused {
                            self.boxes[b_idx].w = snapped_w;
                        }
                    }
                }
            }

            // =====================================================================
            // PASS 1B: Top-Down Inline (Width) Space Distribution
            // =====================================================================
            for pos in span.start..span.end {
                let id = self.dfs_scratch[pos];
                let idx = id.idx();
                let first = self.topo[idx].first_child;
                if first.is_none() {
                    continue;
                }
                let spec = self.specs[idx];
                let inner_w = (self.boxes[idx].w - spec.pad_l - spec.pad_r).max(0.0);

                match spec.axis {
                    Axis::Col => {
                        let mut ch = first;
                        while !ch.is_none() {
                            let c_idx = ch.idx();
                            if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                                let cs = self.specs[c_idx];
                                let raw_w = match cs.mode_w {
                                    SizingMode::Fixed => cs.val_w,
                                    SizingMode::Percent => inner_w * cs.val_w,
                                    SizingMode::Grow => inner_w,
                                    SizingMode::Fit => {
                                        if spec.cross_align == Align::Stretch {
                                            inner_w
                                        } else if cs.shrink_q8 > 0 {
                                            self.boxes[c_idx]
                                                .fit_w
                                                .min(inner_w)
                                                .max(self.boxes[c_idx].min_content_w)
                                        } else {
                                            self.boxes[c_idx].fit_w
                                        }
                                    }
                                };
                                let w = raw_w.clamp(cs.min_w, cs.max_w);
                                self.boxes[c_idx].w =
                                    if snap { Self::snap_px(w, sf, inv_sf) } else { w };
                            }
                            ch = self.topo[c_idx].next_sibling;
                        }
                    }
                    Axis::WrapRow => {
                        let mut ch = first;
                        while !ch.is_none() {
                            let c_idx = ch.idx();
                            if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                                let cs = self.specs[c_idx];
                                let raw_w = match cs.mode_w {
                                    SizingMode::Fixed => cs.val_w,
                                    SizingMode::Percent => inner_w * cs.val_w,
                                    SizingMode::Fit | SizingMode::Grow => {
                                        self.boxes[c_idx].fit_w.min(inner_w)
                                    }
                                };
                                let w = raw_w.clamp(cs.min_w, cs.max_w);
                                self.boxes[c_idx].w =
                                    if snap { Self::snap_px(w, sf, inv_sf) } else { w };
                            }
                            ch = self.topo[c_idx].next_sibling;
                        }
                    }
                    Axis::Row => {
                        self.distribute_row_widths(first, inner_w, spec.gap);
                    }
                    Axis::Grid => {
                        if (self.topo[idx].flags & FLAG_HAS_GRID) != 0 {
                            self.distribute_grid_cols(idx, inner_w, spec.gap);
                        }
                    }
                }
            }
        }

        // =========================================================================
        // PASS 2: Inline -> Block Coupling (Exact Width-Interval Text Wrapping & Aspect Ratio)
        // =========================================================================
        for pos in span.start..span.end {
            let id = self.dfs_scratch[pos];
            let idx = id.idx();
            let spec = self.specs[idx];
            if (self.topo[idx].flags & FLAG_HAS_TEXT) != 0 {
                let raw_inner_w = (self.boxes[idx].w - spec.pad_l - spec.pad_r).max(0.0);
                let tid = spec.aux_id as usize;
                let max_uw = self.texts[tid].max_unwrapped_w;
                let wrap_w = if snap
                    && raw_inner_w < max_uw
                    && (self.boxes[idx].fit_w - self.boxes[idx].w) * sf <= 1.05
                    && (max_uw - raw_inner_w) * sf <= 1.05
                {
                    max_uw
                } else {
                    raw_inner_w
                };
                let (wrapped_h, _tight_w, hit) = self.texts[tid].wrap_for_width(wrap_w);
                if hit {
                    self.last_metrics.text_interval_hits += 1;
                } else {
                    self.last_metrics.text_shaper_calls += 1;
                }
                let pad_y = spec.pad_t + spec.pad_b;
                self.boxes[idx].fit_h = (wrapped_h + pad_y).clamp(spec.min_h, spec.max_h);
                self.boxes[idx].baseline = spec.pad_t + self.texts[tid].ascent;
                self.topo[idx].flags |= FLAG_HAS_BASELINE;
            } else if spec.aspect_ratio > 0.0 && spec.mode_h != SizingMode::Fixed {
                let h_from_aspect = self.boxes[idx].w / spec.aspect_ratio;
                self.boxes[idx].fit_h = h_from_aspect.clamp(spec.min_h, spec.max_h);
                self.boxes[idx].baseline = self.boxes[idx].fit_h;
            }
        }

        // =========================================================================
        // PASS 3A: Bottom-Up Block (Height & Deep `FirstBaseline`) Intrinsic Sizing
        // =========================================================================
        for pos in (span.start..span.end).rev() {
            let id = self.dfs_scratch[pos];
            let idx = id.idx();
            let flags = self.topo[idx].flags;
            let spec = self.specs[idx];
            let pad_y = spec.pad_t.algebraic_add(spec.pad_b);

            if (flags & FLAG_HAS_TEXT) != 0 {
                if spec.mode_h == SizingMode::Fixed {
                    self.boxes[idx].fit_h = spec.val_h.clamp(spec.min_h, spec.max_h);
                }
                self.topo[idx].flags |= FLAG_HAS_BASELINE;
                continue;
            }

            if spec.aspect_ratio > 0.0 && self.topo[idx].first_child.is_none() {
                let raw_fh = if spec.mode_h == SizingMode::Fixed {
                    spec.val_h
                } else {
                    self.boxes[idx].w / spec.aspect_ratio
                };
                let fh = raw_fh.clamp(spec.min_h, spec.max_h);
                self.boxes[idx].fit_h = fh;
                self.boxes[idx].baseline = fh;
                self.topo[idx].flags &= !FLAG_HAS_BASELINE;
                continue;
            }

            if spec.axis == Axis::Grid && (flags & FLAG_HAS_GRID) != 0 {
                self.measure_grid_block(idx, spec, pad_y);
                continue;
            }

            if spec.axis == Axis::WrapRow {
                let inner_w = (self.boxes[idx].w - spec.pad_l - spec.pad_r).max(0.0);
                let mut line_w = 0.0_f32;
                let mut line_h = 0.0_f32;
                let mut line_ascent = 0.0_f32;
                let mut line_descent = 0.0_f32;
                let mut line_count = 0u32;
                let mut line_first_bl: Option<usize> = None;
                let mut total_h = 0.0_f32;
                let mut lines = 0u32;
                let mut natural_baseline: Option<f32> = None;
                let mut fallback_line_bl: Option<f32> = None;

                let resolve_wrap_line = |line_h: f32,
                                         line_ascent: f32,
                                         line_descent: f32,
                                         line_first_bl: Option<usize>,
                                         line_y_off: f32,
                                         natural_baseline: &mut Option<f32>,
                                         fallback_line_bl: &mut Option<f32>,
                                         boxes: &[ComputedBox],
                                         specs: &[LayoutSpec]|
                 -> f32 {
                    let resolved_line_h = if spec.cross_align == Align::FirstBaseline {
                        line_ascent + line_descent
                    } else {
                        line_h
                    };
                    if fallback_line_bl.is_none() && resolved_line_h > 0.0 {
                        *fallback_line_bl = Some(if spec.cross_align == Align::FirstBaseline {
                            line_y_off + line_ascent
                        } else {
                            line_y_off + resolved_line_h
                        });
                    }
                    if natural_baseline.is_none()
                        && let Some(fc_idx) = line_first_bl
                    {
                        if spec.cross_align == Align::FirstBaseline {
                            *natural_baseline = Some(line_y_off + line_ascent);
                        } else {
                            let cs = specs[fc_idx];
                            let cb = boxes[fc_idx];
                            let raw_ih = match cs.mode_h {
                                SizingMode::Fixed => cs.val_h,
                                SizingMode::Grow => resolved_line_h,
                                SizingMode::Percent => cb.fit_h,
                                SizingMode::Fit => {
                                    if spec.cross_align == Align::Stretch {
                                        resolved_line_h
                                    } else {
                                        cb.fit_h
                                    }
                                }
                            };
                            let ih = raw_ih.clamp(cs.min_h, cs.max_h);
                            let rel_y = match spec.cross_align {
                                Align::Start | Align::Stretch => 0.0,
                                Align::Center => (resolved_line_h - ih) * 0.5,
                                Align::End => resolved_line_h - ih,
                                Align::FirstBaseline => unreachable!(),
                            };
                            *natural_baseline = Some(line_y_off + rel_y + cb.baseline);
                        }
                    }
                    resolved_line_h
                };

                let mut ch = self.topo[idx].first_child;
                while !ch.is_none() {
                    let c_idx = ch.idx();
                    let c_flags = self.topo[c_idx].flags;
                    if (c_flags & FLAG_FLOATING_ANCHOR) == 0 {
                        let cb = self.boxes[c_idx];
                        let cs = self.specs[c_idx];
                        let item_h = if cs.mode_h == SizingMode::Fixed {
                            cs.val_h.clamp(cs.min_h, cs.max_h)
                        } else {
                            cb.fit_h.clamp(cs.min_h, cs.max_h)
                        };
                        let has_bl = (c_flags & FLAG_HAS_BASELINE) != 0;
                        let cand = if line_count > 0 {
                            line_w + spec.gap + cb.w
                        } else {
                            cb.w
                        };
                        if line_count > 0 && cand > inner_w + 1e-5 {
                            let line_y_off = total_h + (lines as f32) * spec.gap;
                            let resolved_line_h = resolve_wrap_line(
                                line_h,
                                line_ascent,
                                line_descent,
                                line_first_bl,
                                line_y_off,
                                &mut natural_baseline,
                                &mut fallback_line_bl,
                                &self.boxes,
                                &self.specs,
                            );
                            total_h += resolved_line_h;
                            lines += 1;
                            line_w = cb.w;
                            line_h = item_h;
                            line_ascent = cb.baseline;
                            line_descent = (item_h - cb.baseline).max(0.0);
                            line_first_bl = has_bl.then_some(c_idx);
                            line_count = 1;
                        } else {
                            line_w = cand;
                            line_h = line_h.max(item_h);
                            if has_bl || item_h > 0.0 {
                                line_ascent = line_ascent.max(cb.baseline);
                                line_descent = line_descent.max((item_h - cb.baseline).max(0.0));
                            }
                            if line_first_bl.is_none() && has_bl {
                                line_first_bl = Some(c_idx);
                            }
                            line_count += 1;
                        }
                    }
                    ch = self.topo[c_idx].next_sibling;
                }
                if line_count > 0 {
                    let line_y_off = total_h + (lines as f32) * spec.gap;
                    let resolved_line_h = resolve_wrap_line(
                        line_h,
                        line_ascent,
                        line_descent,
                        line_first_bl,
                        line_y_off,
                        &mut natural_baseline,
                        &mut fallback_line_bl,
                        &self.boxes,
                        &self.specs,
                    );
                    total_h += resolved_line_h + (lines as f32) * spec.gap;
                }
                let fh = if spec.mode_h == SizingMode::Fixed {
                    spec.val_h
                } else if (flags & FLAG_SCROLLABLE) != 0 && spec.mode_h != SizingMode::Fit {
                    pad_y
                } else if spec.aspect_ratio > 0.0 {
                    self.boxes[idx].w / spec.aspect_ratio
                } else {
                    total_h + pad_y
                };
                let fh_clamped = fh.clamp(spec.min_h, spec.max_h);
                self.boxes[idx].fit_h = fh_clamped;
                if let Some(nbl) = natural_baseline {
                    self.boxes[idx].baseline = spec.pad_t + nbl;
                    self.topo[idx].flags |= FLAG_HAS_BASELINE;
                } else {
                    self.topo[idx].flags &= !FLAG_HAS_BASELINE;
                    self.boxes[idx].baseline = if spec.cross_align == Align::FirstBaseline
                        && let Some(fbl) = fallback_line_bl
                    {
                        spec.pad_t + fbl
                    } else {
                        fh_clamped
                    };
                }
                continue;
            }

            let mut sum_fit = 0.0_f32;
            let mut max_fit = 0.0_f32;
            let mut max_ascent = 0.0_f32;
            let mut max_descent = 0.0_f32;
            let mut col_running_y = 0.0_f32;
            let mut col_bl_info: Option<(u32, f32)> = None; // (ordinal, y_before + child.baseline)
            let mut row_first_bl_child: Option<usize> = None;
            let mut n_children = 0u32;

            let mut ch = self.topo[idx].first_child;
            while !ch.is_none() {
                let c_idx = ch.idx();
                let c_flags = self.topo[c_idx].flags;
                if (c_flags & FLAG_FLOATING_ANCHOR) == 0 {
                    let cb = self.boxes[c_idx];
                    let has_bl = (c_flags & FLAG_HAS_BASELINE) != 0;
                    if col_bl_info.is_none() && has_bl {
                        col_bl_info = Some((n_children, col_running_y + cb.baseline));
                    }
                    if row_first_bl_child.is_none() && has_bl {
                        row_first_bl_child = Some(c_idx);
                    }
                    sum_fit = sum_fit.algebraic_add(cb.fit_h);
                    max_fit = max_fit.max(cb.fit_h);
                    if has_bl || cb.fit_h > 0.0 {
                        max_ascent = max_ascent.max(cb.baseline);
                        max_descent = max_descent.max((cb.fit_h - cb.baseline).max(0.0));
                    }
                    col_running_y += cb.fit_h + spec.gap;
                    n_children += 1;
                }
                ch = self.topo[c_idx].next_sibling;
            }
            let gaps = (n_children.saturating_sub(1) as f32) * spec.gap;

            match spec.axis {
                Axis::Col | Axis::Grid | Axis::WrapRow => {
                    let raw_fh = if spec.mode_h == SizingMode::Fixed {
                        spec.val_h
                    } else if (flags & FLAG_SCROLLABLE) != 0 && spec.mode_h != SizingMode::Fit {
                        pad_y
                    } else if spec.aspect_ratio > 0.0 {
                        self.boxes[idx].w / spec.aspect_ratio
                    } else {
                        sum_fit + gaps + pad_y
                    };
                    let fh_clamped = raw_fh.clamp(spec.min_h, spec.max_h);
                    self.boxes[idx].fit_h = fh_clamped;
                    if let Some((ord, base_rel)) = col_bl_info {
                        let outer_h = if snap {
                            Self::snap_px(fh_clamped, sf, inv_sf)
                        } else {
                            fh_clamped
                        };
                        let est_inner_h = (outer_h - pad_y).max(0.0);
                        let free = (est_inner_h - sum_fit - gaps).max(0.0);
                        let (lead_y, extra_gap) = match spec.main_align {
                            MainAlign::Start => (0.0, 0.0),
                            MainAlign::Center => (free * 0.5, 0.0),
                            MainAlign::End => (free, 0.0),
                            MainAlign::SpaceBetween => {
                                if n_children > 1 {
                                    (0.0, free / ((n_children - 1) as f32))
                                } else {
                                    (0.0, 0.0)
                                }
                            }
                            MainAlign::SpaceEvenly => {
                                let step = free / ((n_children + 1) as f32);
                                (step, step)
                            }
                        };
                        self.boxes[idx].baseline =
                            spec.pad_t + lead_y + base_rel + (ord as f32) * extra_gap;
                        self.topo[idx].flags |= FLAG_HAS_BASELINE;
                    } else {
                        self.topo[idx].flags &= !FLAG_HAS_BASELINE;
                        self.boxes[idx].baseline = fh_clamped;
                    }
                }
                Axis::Row => {
                    let content_h = if spec.cross_align == Align::FirstBaseline && n_children > 0 {
                        max_ascent + max_descent
                    } else {
                        max_fit
                    };
                    let raw_fh = if spec.mode_h == SizingMode::Fixed {
                        spec.val_h
                    } else if (flags & FLAG_SCROLLABLE) != 0 && spec.mode_h != SizingMode::Fit {
                        pad_y
                    } else if spec.aspect_ratio > 0.0 {
                        self.boxes[idx].w / spec.aspect_ratio
                    } else {
                        content_h + pad_y
                    };
                    let fh_clamped = raw_fh.clamp(spec.min_h, spec.max_h);
                    self.boxes[idx].fit_h = fh_clamped;

                    if spec.cross_align == Align::FirstBaseline {
                        if row_first_bl_child.is_some() {
                            self.boxes[idx].baseline = spec.pad_t + max_ascent;
                            self.topo[idx].flags |= FLAG_HAS_BASELINE;
                        } else if n_children > 0 && max_ascent > 0.0 {
                            self.boxes[idx].baseline = spec.pad_t + max_ascent;
                            self.topo[idx].flags &= !FLAG_HAS_BASELINE;
                        } else {
                            self.boxes[idx].baseline = fh_clamped;
                            self.topo[idx].flags &= !FLAG_HAS_BASELINE;
                        }
                    } else if let Some(fc_idx) = row_first_bl_child {
                        let outer_h = if snap {
                            Self::snap_px(fh_clamped, sf, inv_sf)
                        } else {
                            fh_clamped
                        };
                        let est_inner_h = (outer_h - pad_y).max(0.0);
                        let cs = self.specs[fc_idx];
                        let fc_raw_h = match cs.mode_h {
                            SizingMode::Fixed => cs.val_h,
                            SizingMode::Percent => est_inner_h * cs.val_h,
                            SizingMode::Grow => est_inner_h,
                            SizingMode::Fit => {
                                if spec.cross_align == Align::Stretch {
                                    est_inner_h
                                } else {
                                    self.boxes[fc_idx].fit_h
                                }
                            }
                        };
                        let fc_h = fc_raw_h.clamp(cs.min_h, cs.max_h);
                        let rel_y = match spec.cross_align {
                            Align::Start | Align::Stretch => 0.0,
                            Align::Center => (est_inner_h - fc_h) * 0.5,
                            Align::End => est_inner_h - fc_h,
                            Align::FirstBaseline => unreachable!(),
                        };
                        self.boxes[idx].baseline =
                            spec.pad_t + rel_y + self.boxes[fc_idx].baseline;
                        self.topo[idx].flags |= FLAG_HAS_BASELINE;
                    } else {
                        self.boxes[idx].baseline = fh_clamped;
                        self.topo[idx].flags &= !FLAG_HAS_BASELINE;
                    }
                }
            }
        }

        // Boundary root outer height & clip:
        {
            let b_idx = boundary_root.idx();
            let b_spec = self.specs[b_idx];
            let b_flags = self.topo[b_idx].flags;
            let is_floating = (b_flags & FLAG_FLOATING_ANCHOR) != 0;
            if self.topo[b_idx].parent.is_none() && !is_floating {
                let win_h = self.last_window_size[1].max(0.0);
                let raw_h = match b_spec.mode_h {
                    SizingMode::Fixed => b_spec.val_h,
                    SizingMode::Grow => win_h,
                    SizingMode::Percent => win_h * b_spec.val_h,
                    SizingMode::Fit => self.boxes[b_idx].fit_h,
                };
                let h = raw_h.clamp(b_spec.min_h, b_spec.max_h);
                self.boxes[b_idx].h = if snap { Self::snap_px(h, sf, inv_sf) } else { h };
                self.boxes[b_idx].x = 0.0;
                self.boxes[b_idx].y = 0.0;
                self.clip_rects[b_idx] = [0.0, 0.0, self.boxes[b_idx].w, self.boxes[b_idx].h];
            } else if is_floating {
                let win_w = self.last_window_size[0].max(0.0);
                let win_h = self.last_window_size[1].max(0.0);
                let raw_h = match b_spec.mode_h {
                    SizingMode::Fixed => b_spec.val_h,
                    SizingMode::Percent => win_h * b_spec.val_h,
                    SizingMode::Fit | SizingMode::Grow => self.boxes[b_idx].fit_h,
                };
                let h = raw_h.clamp(b_spec.min_h, b_spec.max_h);
                self.boxes[b_idx].h = if snap { Self::snap_px(h, sf, inv_sf) } else { h };
                self.clip_rects[b_idx] = [0.0, 0.0, win_w, win_h];
            } else if b_spec.mode_h == SizingMode::Fixed {
                let h = b_spec.val_h.clamp(b_spec.min_h, b_spec.max_h);
                let snapped_h = if snap { Self::snap_px(h, sf, inv_sf) } else { h };
                let p_axis = self.specs[self.topo[b_idx].parent.idx()].axis;
                let keep_diffused = snap
                    && (b_flags & FLAG_LAID_OUT) != 0
                    && p_axis == Axis::Col
                    && (self.boxes[b_idx].h - snapped_h).abs() * sf <= 1.05;
                if !keep_diffused {
                    self.boxes[b_idx].h = snapped_h;
                }
            } else if b_spec.mode_h == SizingMode::Percent {
                let p_idx = self.topo[b_idx].parent.idx();
                let p_spec = self.specs[p_idx];
                if p_spec.axis != Axis::Grid || (b_flags & FLAG_LAID_OUT) == 0 {
                    let p_inner = (self.boxes[p_idx].h - p_spec.pad_t - p_spec.pad_b).max(0.0);
                    let h = (p_inner * b_spec.val_h).clamp(b_spec.min_h, b_spec.max_h);
                    let snapped_h = if snap { Self::snap_px(h, sf, inv_sf) } else { h };
                    let keep_diffused = snap
                        && (b_flags & FLAG_LAID_OUT) != 0
                        && p_spec.axis == Axis::Col
                        && (self.boxes[b_idx].h - snapped_h).abs() * sf <= 1.05;
                    if !keep_diffused {
                        self.boxes[b_idx].h = snapped_h;
                    }
                }
            }
        }

        // =========================================================================
        // PASS 3B: Top-Down Block Distribution + Final (x, y) & Clip Resolution
        // =========================================================================
        for pos in span.start..span.end {
            let id = self.dfs_scratch[pos];
            let idx = id.idx();
            self.topo[idx].flags = (self.topo[idx].flags
                & !(FLAG_DIRTY_LAYOUT | FLAG_DIRTY_INLINE | FLAG_DIRTY_SCROLL))
                | FLAG_LAID_OUT;

            let parent_clip = self.clip_rects[idx];
            let cur_clip = if (self.topo[idx].flags & FLAG_CLIPS_CHILDREN) != 0 {
                let b = self.boxes[idx];
                intersect_rect(parent_clip, [b.x, b.y, b.w, b.h])
            } else {
                parent_clip
            };

            self.place_children_pass3b(idx, cur_clip, true);
        }
    }

    #[inline]
    fn place_children_pass3b(&mut self, idx: usize, cur_clip: [f32; 4], recompute_heights: bool) {
        let first = self.topo[idx].first_child;
        if first.is_none() {
            return;
        }
        let spec = self.specs[idx];
        let pb = self.boxes[idx];
        let inner_w = (pb.w - spec.pad_l - spec.pad_r).max(0.0);
        let inner_h = (pb.h - spec.pad_t - spec.pad_b).max(0.0);
        let origin_x = pb.x + spec.pad_l;
        let origin_y = pb.y + spec.pad_t - spec.scroll_y;

        let (sf, inv_sf) = self.snap_scale_pair();
        let snap = self.snap_to_pixels;

        match spec.axis {
            Axis::Row => {
                let (lead_x, extra_gap) =
                    self.compute_main_justify_row(first, inner_w, spec.gap, spec.main_align);
                let row_baseline_y = pb.baseline - spec.pad_t;
                let step_gap = spec.gap + extra_gap;
                let mut cursor_x = origin_x + lead_x;
                let mut phys_x = if snap && step_gap == 0.0 {
                    ((cursor_x as f64) * (sf as f64)).round()
                } else {
                    0.0
                };
                let mut ch = first;
                while !ch.is_none() {
                    let c_idx = ch.idx();
                    if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                        let cs = self.specs[c_idx];
                        let raw_h = match cs.mode_h {
                            SizingMode::Fixed => cs.val_h,
                            SizingMode::Percent => inner_h * cs.val_h,
                            SizingMode::Grow => inner_h,
                            SizingMode::Fit => {
                                if spec.cross_align == Align::Stretch {
                                    inner_h
                                } else {
                                    self.boxes[c_idx].fit_h
                                }
                            }
                        };
                        let ch_h = raw_h.clamp(cs.min_h, cs.max_h);
                        let ch_w = self.boxes[c_idx].w;
                        let ch_x = if snap {
                            if step_gap == 0.0 {
                                let x_log = Self::phys_to_logical(phys_x, sf, inv_sf);
                                phys_x += ((ch_w * sf).round().max(0.0)) as f64;
                                x_log
                            } else {
                                Self::snap_px(cursor_x, sf, inv_sf)
                            }
                        } else {
                            cursor_x
                        };

                        let ch_y = match spec.cross_align {
                            Align::Start | Align::Stretch => origin_y,
                            Align::Center => origin_y + (inner_h - ch_h) * 0.5,
                            Align::End => origin_y + (inner_h - ch_h),
                            Align::FirstBaseline => {
                                origin_y + (row_baseline_y - self.boxes[c_idx].baseline)
                            }
                        };

                        self.boxes[c_idx].x = ch_x;
                        self.boxes[c_idx].y = if snap {
                            Self::snap_px(ch_y, sf, inv_sf)
                        } else {
                            ch_y
                        };
                        if recompute_heights {
                            self.boxes[c_idx].h = if snap {
                                Self::snap_px(ch_h, sf, inv_sf)
                            } else {
                                ch_h
                            };
                        }
                        self.clip_rects[c_idx] = cur_clip;
                        cursor_x += ch_w + step_gap;
                    }
                    ch = self.topo[c_idx].next_sibling;
                }
            }
            Axis::WrapRow => {
                self.place_wrap_row_children(
                    first, origin_x, origin_y, inner_w, inner_h, spec, cur_clip,
                );
            }
            Axis::Col => {
                if recompute_heights {
                    let is_scrollable = (self.topo[idx].flags & FLAG_SCROLLABLE) != 0;
                    self.distribute_col_heights(first, inner_h, spec.gap, !is_scrollable);
                }
                let (lead_y, extra_gap) =
                    self.compute_main_justify_col(first, inner_h, spec.gap, spec.main_align);
                let step_gap = spec.gap + extra_gap;
                let mut cursor_y = origin_y + lead_y;
                let mut phys_y = if snap && step_gap == 0.0 {
                    ((cursor_y as f64) * (sf as f64)).round()
                } else {
                    0.0
                };
                let mut ch = first;
                while !ch.is_none() {
                    let c_idx = ch.idx();
                    if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                        let ch_h = self.boxes[c_idx].h;
                        let ch_w = self.boxes[c_idx].w;
                        let ch_y = if snap {
                            if step_gap == 0.0 {
                                let y_log = Self::phys_to_logical(phys_y, sf, inv_sf);
                                phys_y += ((ch_h * sf).round().max(0.0)) as f64;
                                y_log
                            } else {
                                Self::snap_px(cursor_y, sf, inv_sf)
                            }
                        } else {
                            cursor_y
                        };

                        let ch_x = match spec.cross_align {
                            Align::Start | Align::Stretch | Align::FirstBaseline => origin_x,
                            Align::Center => origin_x + (inner_w - ch_w) * 0.5,
                            Align::End => origin_x + (inner_w - ch_w),
                        };

                        self.boxes[c_idx].x = if snap {
                            Self::snap_px(ch_x, sf, inv_sf)
                        } else {
                            ch_x
                        };
                        self.boxes[c_idx].y = ch_y;
                        self.clip_rects[c_idx] = cur_clip;
                        cursor_y += ch_h + step_gap;
                    }
                    ch = self.topo[c_idx].next_sibling;
                }
            }
            Axis::Grid => {
                if (self.topo[idx].flags & FLAG_HAS_GRID) != 0 {
                    if recompute_heights {
                        self.distribute_grid_rows(idx, inner_h, spec.gap);
                    }
                    self.place_grid_children(idx, origin_x, origin_y, spec, cur_clip);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn place_wrap_row_children(
        &mut self,
        first: NodeId,
        origin_x: f32,
        origin_y: f32,
        inner_w: f32,
        inner_h: f32,
        spec: LayoutSpec,
        cur_clip: [f32; 4],
    ) {
        let (sf, inv_sf) = self.snap_scale_pair();
        let snap = self.snap_to_pixels;
        let mut cy = origin_y;
        let mut line_start = first;

        while !line_start.is_none() {
            // Advance `line_start` past any leading floating anchors:
            while !line_start.is_none()
                && (self.topo[line_start.idx()].flags & FLAG_FLOATING_ANCHOR) != 0
            {
                line_start = self.topo[line_start.idx()].next_sibling;
            }
            if line_start.is_none() {
                break;
            }

            // 1. Scan one wrapped line `[line_start .. line_end)` to determine exact `line_h`,
            //    `line_ascent`, `line_used_w`, and `line_count` with zero heap allocation!
            let mut line_w = 0.0_f32;
            let mut line_used_w = 0.0_f32;
            let mut line_h = 0.0_f32;
            let mut line_ascent = 0.0_f32;
            let mut line_descent = 0.0_f32;
            let mut line_count = 0u32;
            let mut cur = line_start;
            while !cur.is_none() {
                let c_idx = cur.idx();
                if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                    let cb = self.boxes[c_idx];
                    let cs = self.specs[c_idx];
                    let cand = if line_count > 0 {
                        line_w + spec.gap + cb.w
                    } else {
                        cb.w
                    };
                    if line_count > 0 && cand > inner_w + 1e-5 {
                        break;
                    }
                    let base_h = match cs.mode_h {
                        SizingMode::Fixed => cs.val_h,
                        SizingMode::Percent => inner_h * cs.val_h,
                        SizingMode::Fit | SizingMode::Grow => cb.fit_h,
                    }
                    .clamp(cs.min_h, cs.max_h);
                    line_w = cand;
                    line_used_w += cb.w;
                    line_h = line_h.max(base_h);
                    line_ascent = line_ascent.max(cb.baseline);
                    line_descent = line_descent.max((base_h - cb.baseline).max(0.0));
                    line_count += 1;
                }
                cur = self.topo[c_idx].next_sibling;
            }
            let line_end = cur;
            let resolved_line_h = if spec.cross_align == Align::FirstBaseline {
                line_ascent + line_descent
            } else {
                line_h
            };

            // 2. Per-line main-axis justification:
            let free = (inner_w
                - line_used_w
                - (line_count.saturating_sub(1) as f32) * spec.gap)
                .max(0.0);
            let (lead_x, extra_gap) = match spec.main_align {
                MainAlign::Start => (0.0, 0.0),
                MainAlign::Center => (free * 0.5, 0.0),
                MainAlign::End => (free, 0.0),
                MainAlign::SpaceBetween => {
                    if line_count > 1 {
                        (0.0, free / ((line_count - 1) as f32))
                    } else {
                        (0.0, 0.0)
                    }
                }
                MainAlign::SpaceEvenly => {
                    let step = free / ((line_count + 1) as f32);
                    (step, step)
                }
            };

            // 3. Position items along this wrapped line with per-line `cross_align`, `min_h`/`max_h`, and physical `snap_to_pixels`:
            let step_gap_x = spec.gap + extra_gap;
            let mut cx = origin_x + lead_x;
            let mut phys_cx = if snap && step_gap_x == 0.0 {
                ((cx as f64) * (sf as f64)).round()
            } else {
                0.0
            };
            let mut it = line_start;
            while it != line_end {
                let c_idx = it.idx();
                if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                    let cb = self.boxes[c_idx];
                    let cs = self.specs[c_idx];
                    let raw_h = match cs.mode_h {
                        SizingMode::Fixed => cs.val_h,
                        SizingMode::Percent => inner_h * cs.val_h,
                        SizingMode::Grow => resolved_line_h,
                        SizingMode::Fit => {
                            if spec.cross_align == Align::Stretch {
                                resolved_line_h
                            } else {
                                cb.fit_h
                            }
                        }
                    };
                    let ch_h = raw_h.clamp(cs.min_h, cs.max_h);
                    let ch_y = match spec.cross_align {
                        Align::Start | Align::Stretch => cy,
                        Align::Center => cy + (resolved_line_h - ch_h) * 0.5,
                        Align::End => cy + (resolved_line_h - ch_h),
                        Align::FirstBaseline => cy + (line_ascent - cb.baseline),
                    };

                    self.boxes[c_idx].x = if snap {
                        if step_gap_x == 0.0 {
                            let x_log = Self::phys_to_logical(phys_cx, sf, inv_sf);
                            phys_cx += ((cb.w * sf).round().max(0.0)) as f64;
                            x_log
                        } else {
                            Self::snap_px(cx, sf, inv_sf)
                        }
                    } else {
                        cx
                    };
                    self.boxes[c_idx].y = if snap {
                        Self::snap_px(ch_y, sf, inv_sf)
                    } else {
                        ch_y
                    };
                    self.boxes[c_idx].h = if snap {
                        Self::snap_px(ch_h, sf, inv_sf)
                    } else {
                        ch_h
                    };
                    self.clip_rects[c_idx] = cur_clip;
                    cx += cb.w + step_gap_x;
                }
                it = self.topo[c_idx].next_sibling;
            }

            cy += resolved_line_h + spec.gap;
            line_start = line_end;
        }
    }

    fn compute_main_justify_row(
        &self,
        first: NodeId,
        inner_w: f32,
        gap: f32,
        justify: MainAlign,
    ) -> (f32, f32) {
        if justify == MainAlign::Start {
            return (0.0, 0.0);
        }
        let mut n = 0u32;
        let mut used = 0.0_f32;
        let mut ch = first;
        while !ch.is_none() {
            if (self.topo[ch.idx()].flags & FLAG_FLOATING_ANCHOR) == 0 {
                used += self.boxes[ch.idx()].w;
                n += 1;
            }
            ch = self.topo[ch.idx()].next_sibling;
        }
        if n == 0 {
            return (0.0, 0.0);
        }
        let free = (inner_w - used - (n.saturating_sub(1) as f32) * gap).max(0.0);
        match justify {
            MainAlign::Start => (0.0, 0.0),
            MainAlign::Center => (free * 0.5, 0.0),
            MainAlign::End => (free, 0.0),
            MainAlign::SpaceBetween => {
                if n > 1 {
                    (0.0, free / ((n - 1) as f32))
                } else {
                    (0.0, 0.0)
                }
            }
            MainAlign::SpaceEvenly => {
                let step = free / ((n + 1) as f32);
                (step, step)
            }
        }
    }

    fn compute_main_justify_col(
        &self,
        first: NodeId,
        inner_h: f32,
        gap: f32,
        justify: MainAlign,
    ) -> (f32, f32) {
        if justify == MainAlign::Start {
            return (0.0, 0.0);
        }
        let mut n = 0u32;
        let mut used = 0.0_f32;
        let mut ch = first;
        while !ch.is_none() {
            if (self.topo[ch.idx()].flags & FLAG_FLOATING_ANCHOR) == 0 {
                used += self.boxes[ch.idx()].h;
                n += 1;
            }
            ch = self.topo[ch.idx()].next_sibling;
        }
        if n == 0 {
            return (0.0, 0.0);
        }
        let free = (inner_h - used - (n.saturating_sub(1) as f32) * gap).max(0.0);
        match justify {
            MainAlign::Start => (0.0, 0.0),
            MainAlign::Center => (free * 0.5, 0.0),
            MainAlign::End => (free, 0.0),
            MainAlign::SpaceBetween => {
                if n > 1 {
                    (0.0, free / ((n - 1) as f32))
                } else {
                    (0.0, 0.0)
                }
            }
            MainAlign::SpaceEvenly => {
                let step = free / ((n + 1) as f32);
                (step, step)
            }
        }
    }

    /// Exact constrained proportional `Row` width distribution (`Grow` & `Shrink` water-filling)
    /// with zero heap allocation for `<= 64` flex siblings and reusable scratch above `64`.
    fn distribute_row_widths(&mut self, first: NodeId, inner_w: f32, gap: f32) {
        let mut n = 0usize;
        let mut fixed_sum = 0.0_f32;
        let mut total_grow = 0.0_f64;
        let mut sum_grow_min = 0.0_f32;
        let mut sum_grow_max = 0.0_f32;
        let mut total_shrink = 0.0_f64;

        let mut inline_grow = [0u32; INLINE_FLEX_CAP];
        let mut n_grow = 0usize;

        let mut ch = first;
        while !ch.is_none() {
            let c_idx = ch.idx();
            if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                let cs = self.specs[c_idx];
                match cs.mode_w {
                    SizingMode::Fixed => {
                        let w = cs.val_w.clamp(cs.min_w, cs.max_w);
                        self.boxes[c_idx].w = w;
                        fixed_sum += w;
                    }
                    SizingMode::Percent => {
                        let w = (inner_w * cs.val_w).clamp(cs.min_w, cs.max_w);
                        self.boxes[c_idx].w = w;
                        fixed_sum += w;
                    }
                    SizingMode::Fit => {
                        let w = self.boxes[c_idx].fit_w;
                        self.boxes[c_idx].w = w;
                        fixed_sum += w;
                        total_shrink += cs.shrink_weight() as f64;
                    }
                    SizingMode::Grow => {
                        let gw = cs.val_w.max(0.0);
                        if gw > 0.0 {
                            self.boxes[c_idx].w = cs.min_w;
                            total_grow += gw as f64;
                            sum_grow_min += cs.min_w;
                            sum_grow_max += cs.max_w;
                            if n_grow < INLINE_FLEX_CAP {
                                inline_grow[n_grow] = c_idx as u32;
                            } else if n_grow == INLINE_FLEX_CAP {
                                self.flex_scratch.clear();
                                self.flex_scratch.extend_from_slice(&inline_grow);
                                self.flex_scratch.push(c_idx as u32);
                            } else {
                                self.flex_scratch.push(c_idx as u32);
                            }
                            n_grow += 1;
                        } else {
                            // Zero grow weight resolves immediately to `0.0.clamp(min_w, max_w) == min_w`.
                            self.boxes[c_idx].w = cs.min_w;
                            fixed_sum += cs.min_w;
                        }
                    }
                }
                n += 1;
            }
            ch = self.topo[c_idx].next_sibling;
        }

        let total_gaps = (n.saturating_sub(1) as f32) * gap;
        let rem = inner_w - fixed_sum - total_gaps;

        if rem > sum_grow_min && n_grow > 0 {
            if rem >= sum_grow_max {
                let slice: &[u32] = if n_grow <= INLINE_FLEX_CAP {
                    &inline_grow[..n_grow]
                } else {
                    &self.flex_scratch[..n_grow]
                };
                for &raw_idx in slice {
                    let c_idx = raw_idx as usize;
                    self.boxes[c_idx].w = self.specs[c_idx].max_w;
                }
            } else {
                let active: &mut [u32] = if n_grow <= INLINE_FLEX_CAP {
                    &mut inline_grow[..n_grow]
                } else {
                    &mut self.flex_scratch[..n_grow]
                };
                let mut active_len = n_grow;
                let mut active_grow = total_grow;
                let mut avail = rem as f64;

                loop {
                    if active_len == 0 || active_grow <= 0.0 {
                        break;
                    }
                    let inv_grow = avail / active_grow;
                    let mut clamped_sum = 0.0_f64;
                    let mut has_min_violation = false;
                    let mut has_max_violation = false;

                    for &raw_idx in &active[..active_len] {
                        let c_idx = raw_idx as usize;
                        let cs = self.specs[c_idx];
                        let share = ((cs.val_w.max(0.0) as f64) * inv_grow) as f32;
                        if share < cs.min_w {
                            has_min_violation = true;
                            self.boxes[c_idx].w = cs.min_w;
                            clamped_sum += cs.min_w as f64;
                        } else if share > cs.max_w {
                            has_max_violation = true;
                            self.boxes[c_idx].w = cs.max_w;
                            clamped_sum += cs.max_w as f64;
                        } else {
                            self.boxes[c_idx].w = share;
                            clamped_sum += share as f64;
                        }
                    }

                    let diff = clamped_sum - avail;
                    if (!has_min_violation && !has_max_violation) || diff.abs() <= 1e-4 {
                        break;
                    }

                    // CSS Flexbox §9.7 exact sign rule:
                    // - `diff > 0` (`clamped_sum > avail`) => true multiplier < `inv_grow` => freeze min violations!
                    // - `diff < 0` (`clamped_sum < avail`) => true multiplier > `inv_grow` => freeze max violations!
                    let freeze_min = diff > 0.0;
                    let mut keep = 0usize;
                    let mut next_grow = 0.0_f64;
                    for k in 0..active_len {
                        let c_idx = active[k] as usize;
                        let cs = self.specs[c_idx];
                        let w = cs.val_w.max(0.0) as f64;
                        let share = (w * inv_grow) as f32;
                        if freeze_min {
                            if share <= cs.min_w {
                                self.boxes[c_idx].w = cs.min_w;
                                avail = (avail - (cs.min_w as f64)).max(0.0);
                            } else {
                                active[keep] = c_idx as u32;
                                keep += 1;
                                next_grow += w;
                            }
                        } else if share >= cs.max_w {
                            self.boxes[c_idx].w = cs.max_w;
                            avail = (avail - (cs.max_w as f64)).max(0.0);
                        } else {
                            active[keep] = c_idx as u32;
                            keep += 1;
                            next_grow += w;
                        }
                    }
                    if keep == active_len || keep == 0 || next_grow <= 0.0 {
                        break;
                    }
                    active_len = keep;
                    active_grow = next_grow;
                }
            }
        } else {
            // Container has no surplus above `sum_grow_min`: `Grow` children stay at `cs.min_w`
            // (already initialized above) and `Fit` children absorb any deficit down to `cs.min_w`.
            let deficit = sum_grow_min - rem;
            if deficit > 0.0 && total_shrink > 0.0 {
                let mut inline_shrink = [0u32; INLINE_FLEX_CAP];
                let mut n_shrink = 0usize;
                let mut sum_max_cut = 0.0_f64;
                let mut cur = first;
                while !cur.is_none() {
                    let c_idx = cur.idx();
                    if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                        let cs = self.specs[c_idx];
                        if cs.mode_w == SizingMode::Fit && cs.shrink_q8 > 0 {
                            let max_cut = (self.boxes[c_idx].w - cs.min_w).max(0.0);
                            if max_cut > 0.0 {
                                sum_max_cut += max_cut as f64;
                                if n_shrink < INLINE_FLEX_CAP {
                                    inline_shrink[n_shrink] = c_idx as u32;
                                } else if n_shrink == INLINE_FLEX_CAP {
                                    self.flex_scratch.clear();
                                    self.flex_scratch.extend_from_slice(&inline_shrink);
                                    self.flex_scratch.push(c_idx as u32);
                                } else {
                                    self.flex_scratch.push(c_idx as u32);
                                }
                                n_shrink += 1;
                            } else {
                                self.boxes[c_idx].w = cs.min_w;
                            }
                        }
                    }
                    cur = self.topo[c_idx].next_sibling;
                }

                if n_shrink > 0 {
                    if (deficit as f64) >= sum_max_cut {
                        let slice: &[u32] = if n_shrink <= INLINE_FLEX_CAP {
                            &inline_shrink[..n_shrink]
                        } else {
                            &self.flex_scratch[..n_shrink]
                        };
                        for &raw_idx in slice {
                            let c_idx = raw_idx as usize;
                            self.boxes[c_idx].w = self.specs[c_idx].min_w;
                        }
                    } else {
                        let active: &mut [u32] = if n_shrink <= INLINE_FLEX_CAP {
                            &mut inline_shrink[..n_shrink]
                        } else {
                            &mut self.flex_scratch[..n_shrink]
                        };
                        let mut active_len = n_shrink;
                        let mut active_shrink = 0.0_f64;
                        for &raw_idx in &active[..active_len] {
                            active_shrink += self.specs[raw_idx as usize].shrink_weight() as f64;
                        }
                        let mut rem_deficit = deficit as f64;

                        loop {
                            if active_len == 0 || active_shrink <= 0.0 || rem_deficit <= 1e-5 {
                                break;
                            }
                            let inv_shrink = rem_deficit / active_shrink;
                            let mut any_floor = false;
                            let mut keep = 0usize;
                            let mut next_shrink = 0.0_f64;
                            for k in 0..active_len {
                                let c_idx = active[k] as usize;
                                let cs = self.specs[c_idx];
                                let sw = cs.shrink_weight() as f64;
                                let fit_w = self.boxes[c_idx].fit_w;
                                let cut = (sw * inv_shrink) as f32;
                                if fit_w - cut <= cs.min_w {
                                    self.boxes[c_idx].w = cs.min_w;
                                    rem_deficit = (rem_deficit - ((fit_w - cs.min_w).max(0.0) as f64)).max(0.0);
                                    any_floor = true;
                                } else {
                                    self.boxes[c_idx].w = fit_w - cut;
                                    active[keep] = c_idx as u32;
                                    keep += 1;
                                    next_shrink += sw;
                                }
                            }
                            if !any_floor || keep == active_len || keep == 0 {
                                break;
                            }
                            active_len = keep;
                            active_shrink = next_shrink;
                        }
                    }
                }
            }
        }

        if self.snap_to_pixels {
            let (sf, inv_sf) = self.snap_scale_pair();
            let mut snap_err = 0.0_f32;
            let mut cur = first;
            while !cur.is_none() {
                let c_idx = cur.idx();
                if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                    let ideal_phys = self.boxes[c_idx].w * sf + snap_err;
                    let snapped_phys = ideal_phys.round().max(0.0);
                    snap_err = ideal_phys - snapped_phys;
                    self.boxes[c_idx].w = Self::phys_to_logical(snapped_phys as f64, sf, inv_sf);
                }
                cur = self.topo[c_idx].next_sibling;
            }
        }
    }

    /// Exact constrained proportional `Col` height distribution (`Grow` & `Shrink` water-filling)
    /// with zero heap allocation for `<= 64` flex siblings and reusable scratch above `64`.
    fn distribute_col_heights(
        &mut self,
        first: NodeId,
        inner_h: f32,
        gap: f32,
        allow_shrink: bool,
    ) {
        let mut n = 0usize;
        let mut fixed_sum = 0.0_f32;
        let mut total_grow = 0.0_f64;
        let mut sum_grow_min = 0.0_f32;
        let mut sum_grow_max = 0.0_f32;
        let mut total_shrink = 0.0_f64;

        let mut inline_grow = [0u32; INLINE_FLEX_CAP];
        let mut n_grow = 0usize;

        let mut ch = first;
        while !ch.is_none() {
            let c_idx = ch.idx();
            if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                let cs = self.specs[c_idx];
                match cs.mode_h {
                    SizingMode::Fixed => {
                        let h = cs.val_h.clamp(cs.min_h, cs.max_h);
                        self.boxes[c_idx].h = h;
                        fixed_sum += h;
                    }
                    SizingMode::Percent => {
                        let h = (inner_h * cs.val_h).clamp(cs.min_h, cs.max_h);
                        self.boxes[c_idx].h = h;
                        fixed_sum += h;
                    }
                    SizingMode::Fit => {
                        let h = self.boxes[c_idx].fit_h;
                        self.boxes[c_idx].h = h;
                        fixed_sum += h;
                        total_shrink += cs.shrink_weight() as f64;
                    }
                    SizingMode::Grow => {
                        let gh = cs.val_h.max(0.0);
                        if gh > 0.0 {
                            self.boxes[c_idx].h = cs.min_h;
                            total_grow += gh as f64;
                            sum_grow_min += cs.min_h;
                            sum_grow_max += cs.max_h;
                            if n_grow < INLINE_FLEX_CAP {
                                inline_grow[n_grow] = c_idx as u32;
                            } else if n_grow == INLINE_FLEX_CAP {
                                self.flex_scratch.clear();
                                self.flex_scratch.extend_from_slice(&inline_grow);
                                self.flex_scratch.push(c_idx as u32);
                            } else {
                                self.flex_scratch.push(c_idx as u32);
                            }
                            n_grow += 1;
                        } else {
                            self.boxes[c_idx].h = cs.min_h;
                            fixed_sum += cs.min_h;
                        }
                    }
                }
                n += 1;
            }
            ch = self.topo[c_idx].next_sibling;
        }

        let total_gaps = (n.saturating_sub(1) as f32) * gap;
        let rem = inner_h - fixed_sum - total_gaps;

        if rem > sum_grow_min && n_grow > 0 {
            if rem >= sum_grow_max {
                let slice: &[u32] = if n_grow <= INLINE_FLEX_CAP {
                    &inline_grow[..n_grow]
                } else {
                    &self.flex_scratch[..n_grow]
                };
                for &raw_idx in slice {
                    let c_idx = raw_idx as usize;
                    self.boxes[c_idx].h = self.specs[c_idx].max_h;
                }
            } else {
                let active: &mut [u32] = if n_grow <= INLINE_FLEX_CAP {
                    &mut inline_grow[..n_grow]
                } else {
                    &mut self.flex_scratch[..n_grow]
                };
                let mut active_len = n_grow;
                let mut active_grow = total_grow;
                let mut avail = rem as f64;

                loop {
                    if active_len == 0 || active_grow <= 0.0 {
                        break;
                    }
                    let inv_grow = avail / active_grow;
                    let mut clamped_sum = 0.0_f64;
                    let mut has_min_violation = false;
                    let mut has_max_violation = false;

                    for &raw_idx in &active[..active_len] {
                        let c_idx = raw_idx as usize;
                        let cs = self.specs[c_idx];
                        let share = ((cs.val_h.max(0.0) as f64) * inv_grow) as f32;
                        if share < cs.min_h {
                            has_min_violation = true;
                            self.boxes[c_idx].h = cs.min_h;
                            clamped_sum += cs.min_h as f64;
                        } else if share > cs.max_h {
                            has_max_violation = true;
                            self.boxes[c_idx].h = cs.max_h;
                            clamped_sum += cs.max_h as f64;
                        } else {
                            self.boxes[c_idx].h = share;
                            clamped_sum += share as f64;
                        }
                    }

                    let diff = clamped_sum - avail;
                    if (!has_min_violation && !has_max_violation) || diff.abs() <= 1e-4 {
                        break;
                    }

                    let freeze_min = diff > 0.0;
                    let mut keep = 0usize;
                    let mut next_grow = 0.0_f64;
                    for k in 0..active_len {
                        let c_idx = active[k] as usize;
                        let cs = self.specs[c_idx];
                        let h_w = cs.val_h.max(0.0) as f64;
                        let share = (h_w * inv_grow) as f32;
                        if freeze_min {
                            if share <= cs.min_h {
                                self.boxes[c_idx].h = cs.min_h;
                                avail = (avail - (cs.min_h as f64)).max(0.0);
                            } else {
                                active[keep] = c_idx as u32;
                                keep += 1;
                                next_grow += h_w;
                            }
                        } else if share >= cs.max_h {
                            self.boxes[c_idx].h = cs.max_h;
                            avail = (avail - (cs.max_h as f64)).max(0.0);
                        } else {
                            active[keep] = c_idx as u32;
                            keep += 1;
                            next_grow += h_w;
                        }
                    }
                    if keep == active_len || keep == 0 || next_grow <= 0.0 {
                        break;
                    }
                    active_len = keep;
                    active_grow = next_grow;
                }
            }
        } else {
            let deficit = sum_grow_min - rem;
            if allow_shrink && deficit > 0.0 && total_shrink > 0.0 {
                let mut inline_shrink = [0u32; INLINE_FLEX_CAP];
                let mut n_shrink = 0usize;
                let mut sum_max_cut = 0.0_f64;
                let mut cur = first;
                while !cur.is_none() {
                    let c_idx = cur.idx();
                    if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                        let cs = self.specs[c_idx];
                        if cs.mode_h == SizingMode::Fit && cs.shrink_q8 > 0 {
                            let max_cut = (self.boxes[c_idx].h - cs.min_h).max(0.0);
                            if max_cut > 0.0 {
                                sum_max_cut += max_cut as f64;
                                if n_shrink < INLINE_FLEX_CAP {
                                    inline_shrink[n_shrink] = c_idx as u32;
                                } else if n_shrink == INLINE_FLEX_CAP {
                                    self.flex_scratch.clear();
                                    self.flex_scratch.extend_from_slice(&inline_shrink);
                                    self.flex_scratch.push(c_idx as u32);
                                } else {
                                    self.flex_scratch.push(c_idx as u32);
                                }
                                n_shrink += 1;
                            } else {
                                self.boxes[c_idx].h = cs.min_h;
                            }
                        }
                    }
                    cur = self.topo[c_idx].next_sibling;
                }

                if n_shrink > 0 {
                    if (deficit as f64) >= sum_max_cut {
                        let slice: &[u32] = if n_shrink <= INLINE_FLEX_CAP {
                            &inline_shrink[..n_shrink]
                        } else {
                            &self.flex_scratch[..n_shrink]
                        };
                        for &raw_idx in slice {
                            let c_idx = raw_idx as usize;
                            self.boxes[c_idx].h = self.specs[c_idx].min_h;
                        }
                    } else {
                        let active: &mut [u32] = if n_shrink <= INLINE_FLEX_CAP {
                            &mut inline_shrink[..n_shrink]
                        } else {
                            &mut self.flex_scratch[..n_shrink]
                        };
                        let mut active_len = n_shrink;
                        let mut active_shrink = 0.0_f64;
                        for &raw_idx in &active[..active_len] {
                            active_shrink += self.specs[raw_idx as usize].shrink_weight() as f64;
                        }
                        let mut rem_deficit = deficit as f64;

                        loop {
                            if active_len == 0 || active_shrink <= 0.0 || rem_deficit <= 1e-5 {
                                break;
                            }
                            let inv_shrink = rem_deficit / active_shrink;
                            let mut any_floor = false;
                            let mut keep = 0usize;
                            let mut next_shrink = 0.0_f64;
                            for k in 0..active_len {
                                let c_idx = active[k] as usize;
                                let cs = self.specs[c_idx];
                                let sw = cs.shrink_weight() as f64;
                                let fit_h = self.boxes[c_idx].fit_h;
                                let cut = (sw * inv_shrink) as f32;
                                if fit_h - cut <= cs.min_h {
                                    self.boxes[c_idx].h = cs.min_h;
                                    rem_deficit = (rem_deficit - ((fit_h - cs.min_h).max(0.0) as f64)).max(0.0);
                                    any_floor = true;
                                } else {
                                    self.boxes[c_idx].h = fit_h - cut;
                                    active[keep] = c_idx as u32;
                                    keep += 1;
                                    next_shrink += sw;
                                }
                            }
                            if !any_floor || keep == active_len || keep == 0 {
                                break;
                            }
                            active_len = keep;
                            active_shrink = next_shrink;
                        }
                    }
                }
            }
        }

        if self.snap_to_pixels {
            let (sf, inv_sf) = self.snap_scale_pair();
            let mut snap_err = 0.0_f32;
            let mut cur = first;
            while !cur.is_none() {
                let c_idx = cur.idx();
                if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                    let ideal_phys = self.boxes[c_idx].h * sf + snap_err;
                    let snapped_phys = ideal_phys.round().max(0.0);
                    snap_err = ideal_phys - snapped_phys;
                    self.boxes[c_idx].h = Self::phys_to_logical(snapped_phys as f64, sf, inv_sf);
                }
                cur = self.topo[c_idx].next_sibling;
            }
        }
    }

    /// Compute exact 2D `(col, row, col_span, row_span)` auto-placements for all non-floating children
    /// of a `Grid`, honoring multi-track `col_span`/`row_span` occupancy and clamping excess items to
    /// the final declared row (`n_rows - 1`) without out-of-bounds indexing.
    fn resolve_grid_placements(&mut self, grid_idx: usize) {
        let gid = self.specs[grid_idx].aux_id as usize;
        let n_cols = self.grids[gid].cols.len().max(1);
        let n_rows = self.grids[gid].rows.len().max(1);

        {
            let g = &mut self.grids[gid];
            g.placements.clear();
            let total_cells = n_cols.saturating_mul(n_rows);
            g.occupancy.resize(total_cells, false);
            g.occupancy.fill(false);
        }

        let mut row_cur = 0usize;
        let mut col_cur = 0usize;
        let mut overflow_col = 0usize;

        let mut ch = self.topo[grid_idx].first_child;
        while !ch.is_none() {
            let c_idx = ch.idx();
            if (self.topo[c_idx].flags & FLAG_FLOATING_ANCHOR) == 0 {
                let (raw_cs, raw_rs) = self.node_grid_spans(ch);
                let c_span = raw_cs.min(n_cols).min(u8::MAX as usize);

                while row_cur < n_rows {
                    if col_cur + c_span > n_cols {
                        col_cur = 0;
                        row_cur += 1;
                        continue;
                    }
                    let row_base = row_cur * n_cols;
                    let free = !self.grids[gid].occupancy[row_base + col_cur..row_base + col_cur + c_span]
                        .iter()
                        .any(|&occ| occ);
                    if free {
                        break;
                    }
                    col_cur += 1;
                }

                if row_cur < n_rows {
                    let r_span = raw_rs.min(n_rows - row_cur).min(u8::MAX as usize);
                    let g = &mut self.grids[gid];
                    for dy in 0..r_span {
                        let base = (row_cur + dy) * n_cols + col_cur;
                        for dx in 0..c_span {
                            g.occupancy[base + dx] = true;
                        }
                    }
                    g.placements.push(GridPlacement {
                        node: ch,
                        col: col_cur as u16,
                        row: row_cur as u16,
                        col_span: c_span as u8,
                        row_span: r_span as u8,
                    });
                    col_cur += c_span;
                    if col_cur >= n_cols {
                        col_cur = 0;
                        row_cur += 1;
                    }
                } else {
                    // Children exceed declared `rows`: clamp to last declared row `n_rows - 1`.
                    let last_row = n_rows - 1;
                    if overflow_col + c_span > n_cols {
                        overflow_col = 0;
                    }
                    self.grids[gid].placements.push(GridPlacement {
                        node: ch,
                        col: overflow_col as u16,
                        row: last_row as u16,
                        col_span: c_span as u8,
                        row_span: 1,
                    });
                    overflow_col = (overflow_col + c_span) % n_cols;
                }
            }
            ch = self.topo[c_idx].next_sibling;
        }
        self.grids[gid].placements_valid = true;
    }

    /// CSS Grid §11.5 Intrinsic Column Track Sizing:
    /// 1. Single-column (`col_span == 1`) items establish track bases.
    /// 2. Multi-column (`col_span > 1`) items are processed in strictly ascending span order
    ///    (`span = 2..=max_span`) using per-span-group `planned_increase` (`max` over items of equal
    ///    span, zero heap allocations via dead `col_offsets` scratch + `[f32; 16]` per-item stack buffer),
    ///    water-filling excess `fit_w` across `Fit`/`Auto`/`MinMax` tracks up to `hi` limits.
    fn measure_grid_inline(&mut self, idx: usize, spec: LayoutSpec, pad_x: f32) -> (f32, f32) {
        self.resolve_grid_placements(idx);
        let gid = spec.aux_id as usize;
        let n_cols = self.grids[gid].cols.len();
        for c in 0..n_cols {
            let init = match self.grids[gid].cols[c] {
                TrackSize::Fixed(fx) => fx,
                TrackSize::MinMax(lo, _) => lo,
                TrackSize::Fit | TrackSize::Auto | TrackSize::Percent(_) | TrackSize::Fr(_) => 0.0,
            };
            self.grids[gid].resolved_cols[c] = init;
            self.grids[gid].col_offsets[c] = init; // Accumulate `min_content` column minimums first
        }

        let p_len = self.grids[gid].placements.len();
        let mut max_span = 1usize;
        // 1. Single-column items (`col_span == 1`):
        for p_i in 0..p_len {
            let p = self.grids[gid].placements[p_i];
            let span = p.col_span as usize;
            if span == 1 {
                let c = p.col as usize;
                let cb = self.boxes[p.node.idx()];
                let cw = cb.fit_w;
                let mcw = cb.min_content_w;
                match self.grids[gid].cols[c] {
                    TrackSize::Fixed(fx) => {
                        self.grids[gid].resolved_cols[c] = fx;
                        self.grids[gid].col_offsets[c] = fx;
                    }
                    TrackSize::MinMax(lo, hi) => {
                        self.grids[gid].resolved_cols[c] =
                            self.grids[gid].resolved_cols[c].max(cw.clamp(lo, hi));
                        self.grids[gid].col_offsets[c] =
                            self.grids[gid].col_offsets[c].max(mcw.clamp(lo, hi));
                    }
                    TrackSize::Fit | TrackSize::Auto => {
                        self.grids[gid].resolved_cols[c] =
                            self.grids[gid].resolved_cols[c].max(cw);
                        self.grids[gid].col_offsets[c] =
                            self.grids[gid].col_offsets[c].max(mcw);
                    }
                    TrackSize::Percent(_) | TrackSize::Fr(_) => {
                        self.grids[gid].resolved_cols[c] =
                            self.grids[gid].resolved_cols[c].max(cw);
                    }
                }
            } else if span > max_span {
                max_span = span;
            }
        }

        let gaps = (n_cols.saturating_sub(1) as f32) * spec.gap;
        let mut min_cols_sum = 0.0_f32;
        for &mc in self.grids[gid].col_offsets.iter().take(n_cols) {
            min_cols_sum = min_cols_sum.algebraic_add(mc);
        }

        // 2. Multi-column spanned items (`col_span > 1`): ascending span order + simultaneous `planned_increase`.
        if max_span > 1 {
            for target_span in 2..=max_span {
                self.grids[gid].col_offsets[..n_cols].fill(0.0);
                let mut any_planned = false;

                for p_i in 0..p_len {
                    let p = self.grids[gid].placements[p_i];
                    let span = p.col_span as usize;
                    if span != target_span {
                        continue;
                    }
                    let c0 = p.col as usize;
                    let cw = self.boxes[p.node.idx()].fit_w;
                    let mut cur_w = ((span - 1) as f32) * spec.gap;
                    let mut total_fr = 0.0_f32;
                    let mut n_pct = 0u32;
                    for k in 0..span {
                        let c = c0 + k;
                        cur_w += self.grids[gid].resolved_cols[c];
                        match self.grids[gid].cols[c] {
                            TrackSize::Fr(fr) if fr > 0.0 => total_fr += fr,
                            TrackSize::Percent(_) | TrackSize::Fr(_) => n_pct += 1,
                            _ => {}
                        }
                    }
                    if cw <= cur_w + 1e-5 {
                        continue;
                    }

                    let mut rem_extra = cw - cur_w;
                    let mut item_inc = [0.0_f32; 256];
                    loop {
                        if rem_extra <= 1e-5 {
                            break;
                        }
                        let mut active_intrinsic = 0u32;
                        for (k, &inc_k) in item_inc.iter().enumerate().take(span) {
                            let c = c0 + k;
                            match self.grids[gid].cols[c] {
                                TrackSize::Fit | TrackSize::Auto => active_intrinsic += 1,
                                TrackSize::MinMax(_, hi)
                                    if self.grids[gid].resolved_cols[c] + inc_k + 1e-5 < hi =>
                                {
                                    active_intrinsic += 1;
                                }
                                _ => {}
                            }
                        }
                        if active_intrinsic == 0 {
                            break;
                        }
                        let add = rem_extra / (active_intrinsic as f32);
                        let mut gained = 0.0_f32;
                        for (k, inc_k) in item_inc.iter_mut().enumerate().take(span) {
                            let c = c0 + k;
                            match self.grids[gid].cols[c] {
                                TrackSize::Fit | TrackSize::Auto => {
                                    *inc_k += add;
                                    gained += add;
                                }
                                TrackSize::MinMax(_, hi) => {
                                    let cur_tot = self.grids[gid].resolved_cols[c] + *inc_k;
                                    if cur_tot + 1e-5 < hi {
                                        let step = add.min(hi - cur_tot);
                                        *inc_k += step;
                                        gained += step;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if gained <= 1e-6 {
                            break;
                        }
                        rem_extra = (rem_extra - gained).max(0.0);
                    }

                    if rem_extra > 1e-4 {
                        if total_fr > 0.0 {
                            for (k, inc_k) in item_inc.iter_mut().enumerate().take(span) {
                                if let TrackSize::Fr(fr) = self.grids[gid].cols[c0 + k]
                                    && fr > 0.0
                                {
                                    *inc_k += rem_extra * (fr / total_fr);
                                }
                            }
                        } else if n_pct > 0 {
                            let add = rem_extra / (n_pct as f32);
                            for (k, inc_k) in item_inc.iter_mut().enumerate().take(span) {
                                if matches!(
                                    self.grids[gid].cols[c0 + k],
                                    TrackSize::Percent(_) | TrackSize::Fr(_)
                                ) {
                                    *inc_k += add;
                                }
                            }
                        }
                    }

                    for (k, &inc) in item_inc.iter().enumerate().take(span) {
                        if inc > 0.0 {
                            any_planned = true;
                            let slot = &mut self.grids[gid].col_offsets[c0 + k];
                            *slot = (*slot).max(inc);
                        }
                    }
                }

                if any_planned {
                    for c in 0..n_cols {
                        self.grids[gid].resolved_cols[c] += self.grids[gid].col_offsets[c];
                    }
                }
            }
        }

        let mut total_cols = 0.0_f32;
        for &cw in self.grids[gid].resolved_cols.iter().take(n_cols) {
            total_cols = total_cols.algebraic_add(cw);
        }
        let fit_w = total_cols + gaps + pad_x;
        let min_cw = (min_cols_sum + gaps + pad_x).min(fit_w).max(spec.min_w.max(pad_x));
        (fit_w, min_cw)
    }

    fn distribute_grid_cols(&mut self, grid_idx: usize, inner_w: f32, gap: f32) {
        let gid = self.specs[grid_idx].aux_id as usize;
        if !self.grids[gid].placements_valid {
            self.resolve_grid_placements(grid_idx);
        }
        let n_cols = self.grids[gid].cols.len();
        let gaps = (n_cols.saturating_sub(1) as f32) * gap;
        let mut used = gaps;
        let mut total_fr = 0.0_f32;
        for i in 0..n_cols {
            match self.grids[gid].cols[i] {
                TrackSize::Fixed(fx) => {
                    self.grids[gid].resolved_cols[i] = fx;
                    used += fx;
                }
                TrackSize::Percent(p) => {
                    let w = inner_w * p;
                    self.grids[gid].resolved_cols[i] = w;
                    used += w;
                }
                TrackSize::Fit | TrackSize::Auto => {
                    used += self.grids[gid].resolved_cols[i];
                }
                TrackSize::MinMax(lo, hi) => {
                    let w = self.grids[gid].resolved_cols[i].clamp(lo, hi);
                    self.grids[gid].resolved_cols[i] = w;
                    used += w;
                }
                TrackSize::Fr(fr) => {
                    total_fr += fr;
                }
            }
        }

        // CSS Grid §11.6 Maximize `MinMax` tracks up to their growth limit `hi` before §11.7 `Fr` distribution:
        let mut rem = (inner_w - used).max(0.0);
        loop {
            if rem <= 1e-4 {
                break;
            }
            let mut expandable = 0u32;
            for i in 0..n_cols {
                if let TrackSize::MinMax(_, hi) = self.grids[gid].cols[i]
                    && (hi.is_finite() || total_fr <= 0.0)
                    && self.grids[gid].resolved_cols[i] + 1e-5 < hi
                {
                    expandable += 1;
                }
            }
            if expandable == 0 {
                break;
            }
            let add = rem / (expandable as f32);
            let mut gained = 0.0_f32;
            for i in 0..n_cols {
                if let TrackSize::MinMax(_, hi) = self.grids[gid].cols[i]
                    && (hi.is_finite() || total_fr <= 0.0)
                {
                    let cur = self.grids[gid].resolved_cols[i];
                    if cur + 1e-5 < hi {
                        let next = (cur + add).min(hi);
                        self.grids[gid].resolved_cols[i] = next;
                        gained += next - cur;
                    }
                }
            }
            if gained <= 1e-6 {
                break;
            }
            rem = (rem - gained).max(0.0);
        }

        let (sf, inv_sf) = self.snap_scale_pair();
        let snap = self.snap_to_pixels;
        let mut snap_err = 0.0_f32;
        let mut off = 0.0_f32;
        let mut phys_off = 0.0_f64;
        for i in 0..n_cols {
            if let TrackSize::Fr(fr) = self.grids[gid].cols[i] {
                self.grids[gid].resolved_cols[i] = if total_fr > 0.0 {
                    rem * (fr / total_fr)
                } else {
                    0.0
                };
            }
            if snap {
                let ideal_phys = self.grids[gid].resolved_cols[i] * sf + snap_err;
                let snapped_phys = ideal_phys.round().max(0.0);
                snap_err = ideal_phys - snapped_phys;
                self.grids[gid].resolved_cols[i] =
                    Self::phys_to_logical(snapped_phys as f64, sf, inv_sf);
                self.grids[gid].col_offsets[i] = if gap == 0.0 {
                    let c_log = Self::phys_to_logical(phys_off, sf, inv_sf);
                    phys_off += snapped_phys as f64;
                    c_log
                } else {
                    Self::snap_px(off, sf, inv_sf)
                };
            } else {
                self.grids[gid].col_offsets[i] = off;
            }
            off += self.grids[gid].resolved_cols[i] + gap;
        }

        let p_len = self.grids[gid].placements.len();
        for p_i in 0..p_len {
            let p = self.grids[gid].placements[p_i];
            let c_idx = p.node.idx();
            let cs = self.specs[c_idx];
            let c0 = p.col as usize;
            let span = p.col_span as usize;
            let mut spanned_w = ((span.saturating_sub(1)) as f32) * gap;
            for k in 0..span {
                spanned_w += self.grids[gid].resolved_cols[c0 + k];
            }
            let cw = match cs.mode_w {
                SizingMode::Fixed => cs.val_w,
                SizingMode::Percent => spanned_w * cs.val_w,
                SizingMode::Grow | SizingMode::Fit => spanned_w,
            }
            .clamp(cs.min_w, cs.max_w);
            self.boxes[c_idx].w = if snap {
                Self::snap_px(cw, sf, inv_sf)
            } else {
                cw
            };
        }
    }

    fn measure_grid_block(&mut self, idx: usize, spec: LayoutSpec, pad_y: f32) {
        let gid = spec.aux_id as usize;
        if !self.grids[gid].placements_valid {
            self.resolve_grid_placements(idx);
        }
        let n_rows = self.grids[gid].rows.len();
        for r in 0..n_rows {
            self.grids[gid].resolved_rows[r] = match self.grids[gid].rows[r] {
                TrackSize::Fixed(fx) => fx,
                TrackSize::MinMax(lo, _) => lo,
                TrackSize::Fit | TrackSize::Auto | TrackSize::Percent(_) | TrackSize::Fr(_) => 0.0,
            };
        }

        let p_len = self.grids[gid].placements.len();
        let mut max_span = 1usize;

        // If `FirstBaseline`, compute per-row `max_ascent + max_descent` for single-row items:
        if spec.cross_align == Align::FirstBaseline {
            self.grids[gid].row_offsets[..n_rows].fill(0.0); // row max_ascent
            for p_i in 0..p_len {
                let p = self.grids[gid].placements[p_i];
                if p.row_span == 1 {
                    let r = p.row as usize;
                    let cb = self.boxes[p.node.idx()];
                    let has_bl = (self.topo[p.node.idx()].flags & FLAG_HAS_BASELINE) != 0;
                    if has_bl || cb.fit_h > 0.0 {
                        self.grids[gid].row_offsets[r] =
                            self.grids[gid].row_offsets[r].max(cb.baseline);
                    }
                }
            }
        }

        // 1. Single-row items (`row_span == 1`):
        for p_i in 0..p_len {
            let p = self.grids[gid].placements[p_i];
            let span = p.row_span as usize;
            if span == 1 {
                let r = p.row as usize;
                let cb = self.boxes[p.node.idx()];
                let ch_h = if spec.cross_align == Align::FirstBaseline {
                    let row_asc = self.grids[gid].row_offsets[r];
                    let desc = (cb.fit_h - cb.baseline).max(0.0);
                    (row_asc + desc).max(cb.fit_h)
                } else {
                    cb.fit_h
                };
                match self.grids[gid].rows[r] {
                    TrackSize::Fixed(fx) => self.grids[gid].resolved_rows[r] = fx,
                    TrackSize::MinMax(lo, hi) => {
                        self.grids[gid].resolved_rows[r] =
                            self.grids[gid].resolved_rows[r].max(ch_h.clamp(lo, hi));
                    }
                    TrackSize::Fit | TrackSize::Auto | TrackSize::Percent(_) | TrackSize::Fr(_) => {
                        self.grids[gid].resolved_rows[r] =
                            self.grids[gid].resolved_rows[r].max(ch_h);
                    }
                }
            } else if span > max_span {
                max_span = span;
            }
        }

        // 2. Multi-row spanned items (`row_span > 1`): ascending span order + simultaneous `planned_increase`.
        if max_span > 1 {
            for target_span in 2..=max_span {
                self.grids[gid].row_offsets[..n_rows].fill(0.0);
                let mut any_planned = false;

                for p_i in 0..p_len {
                    let p = self.grids[gid].placements[p_i];
                    let span = p.row_span as usize;
                    if span != target_span {
                        continue;
                    }
                    let r0 = p.row as usize;
                    let ch_h = self.boxes[p.node.idx()].fit_h;
                    let mut cur_h = ((span - 1) as f32) * spec.gap;
                    let mut total_fr = 0.0_f32;
                    let mut n_pct = 0u32;
                    for k in 0..span {
                        let r = r0 + k;
                        cur_h += self.grids[gid].resolved_rows[r];
                        match self.grids[gid].rows[r] {
                            TrackSize::Fr(fr) if fr > 0.0 => total_fr += fr,
                            TrackSize::Percent(_) | TrackSize::Fr(_) => n_pct += 1,
                            _ => {}
                        }
                    }
                    if ch_h <= cur_h + 1e-5 {
                        continue;
                    }

                    let mut rem_extra = ch_h - cur_h;
                    let mut item_inc = [0.0_f32; 256];
                    loop {
                        if rem_extra <= 1e-5 {
                            break;
                        }
                        let mut active_intrinsic = 0u32;
                        for (k, &inc_k) in item_inc.iter().enumerate().take(span) {
                            let r = r0 + k;
                            match self.grids[gid].rows[r] {
                                TrackSize::Fit | TrackSize::Auto => active_intrinsic += 1,
                                TrackSize::MinMax(_, hi)
                                    if self.grids[gid].resolved_rows[r] + inc_k + 1e-5 < hi =>
                                {
                                    active_intrinsic += 1;
                                }
                                _ => {}
                            }
                        }
                        if active_intrinsic == 0 {
                            break;
                        }
                        let add = rem_extra / (active_intrinsic as f32);
                        let mut gained = 0.0_f32;
                        for (k, inc_k) in item_inc.iter_mut().enumerate().take(span) {
                            let r = r0 + k;
                            match self.grids[gid].rows[r] {
                                TrackSize::Fit | TrackSize::Auto => {
                                    *inc_k += add;
                                    gained += add;
                                }
                                TrackSize::MinMax(_, hi) => {
                                    let cur_tot = self.grids[gid].resolved_rows[r] + *inc_k;
                                    if cur_tot + 1e-5 < hi {
                                        let step = add.min(hi - cur_tot);
                                        *inc_k += step;
                                        gained += step;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if gained <= 1e-6 {
                            break;
                        }
                        rem_extra = (rem_extra - gained).max(0.0);
                    }

                    if rem_extra > 1e-4 {
                        if total_fr > 0.0 {
                            for (k, inc_k) in item_inc.iter_mut().enumerate().take(span) {
                                if let TrackSize::Fr(fr) = self.grids[gid].rows[r0 + k]
                                    && fr > 0.0
                                {
                                    *inc_k += rem_extra * (fr / total_fr);
                                }
                            }
                        } else if n_pct > 0 {
                            let add = rem_extra / (n_pct as f32);
                            for (k, inc_k) in item_inc.iter_mut().enumerate().take(span) {
                                if matches!(
                                    self.grids[gid].rows[r0 + k],
                                    TrackSize::Percent(_) | TrackSize::Fr(_)
                                ) {
                                    *inc_k += add;
                                }
                            }
                        }
                    }

                    for (k, &inc) in item_inc.iter().enumerate().take(span) {
                        if inc > 0.0 {
                            any_planned = true;
                            let slot = &mut self.grids[gid].row_offsets[r0 + k];
                            *slot = (*slot).max(inc);
                        }
                    }
                }

                if any_planned {
                    for r in 0..n_rows {
                        self.grids[gid].resolved_rows[r] += self.grids[gid].row_offsets[r];
                    }
                }
            }
        }

        let mut total_rows = 0.0_f32;
        for &rh in self.grids[gid].resolved_rows.iter().take(n_rows) {
            total_rows = total_rows.algebraic_add(rh);
        }
        let gaps = (n_rows.saturating_sub(1) as f32) * spec.gap;
        let fh = if spec.mode_h == SizingMode::Fixed {
            spec.val_h
        } else if (self.topo[idx].flags & FLAG_SCROLLABLE) != 0 && spec.mode_h != SizingMode::Fit {
            pad_y
        } else {
            total_rows + gaps + pad_y
        };
        let fh_clamped = fh.clamp(spec.min_h, spec.max_h);
        self.boxes[idx].fit_h = fh_clamped;

        // Propagate natural first baseline from the earliest grid row containing a baseline item:
        let mut grid_bl: Option<f32> = None;
        let mut best_row = usize::MAX;
        for p_i in 0..p_len {
            let p = self.grids[gid].placements[p_i];
            let r0 = p.row as usize;
            if r0 < best_row && (self.topo[p.node.idx()].flags & FLAG_HAS_BASELINE) != 0 {
                best_row = r0;
            }
        }
        if best_row < n_rows {
            let mut y_before = 0.0_f32;
            for &rh in self.grids[gid].resolved_rows.iter().take(best_row) {
                y_before += rh + spec.gap;
            }
            if spec.cross_align == Align::FirstBaseline {
                let mut row_asc = 0.0_f32;
                for p_i in 0..p_len {
                    let p = self.grids[gid].placements[p_i];
                    if p.row as usize == best_row && p.row_span == 1 {
                        row_asc = row_asc.max(self.boxes[p.node.idx()].baseline);
                    }
                }
                grid_bl = Some(y_before + row_asc);
            } else {
                for p_i in 0..p_len {
                    let p = self.grids[gid].placements[p_i];
                    let c_idx = p.node.idx();
                    if p.row as usize == best_row && (self.topo[c_idx].flags & FLAG_HAS_BASELINE) != 0 {
                        let r_span = p.row_span as usize;
                        let mut track_h = ((r_span.saturating_sub(1)) as f32) * spec.gap;
                        for k in 0..r_span {
                            track_h += self.grids[gid].resolved_rows[best_row + k];
                        }
                        let cs = self.specs[c_idx];
                        let ch_h = match cs.mode_h {
                            SizingMode::Fixed => cs.val_h,
                            SizingMode::Percent => track_h * cs.val_h,
                            SizingMode::Grow => track_h,
                            SizingMode::Fit => {
                                if spec.cross_align == Align::Stretch {
                                    track_h
                                } else {
                                    self.boxes[c_idx].fit_h
                                }
                            }
                        }
                        .clamp(cs.min_h, cs.max_h);
                        let rel_y = match spec.cross_align {
                            Align::Start | Align::Stretch => 0.0,
                            Align::Center => (track_h - ch_h) * 0.5,
                            Align::End => track_h - ch_h,
                            Align::FirstBaseline => unreachable!(),
                        };
                        grid_bl = Some(y_before + rel_y + self.boxes[c_idx].baseline);
                        break;
                    }
                }
            }
        }
        if let Some(gbl) = grid_bl {
            self.boxes[idx].baseline = spec.pad_t + gbl;
            self.topo[idx].flags |= FLAG_HAS_BASELINE;
        } else {
            self.boxes[idx].baseline = fh_clamped;
            self.topo[idx].flags &= !FLAG_HAS_BASELINE;
        }
    }

    fn distribute_grid_rows(&mut self, grid_idx: usize, inner_h: f32, gap: f32) {
        let gid = self.specs[grid_idx].aux_id as usize;
        if !self.grids[gid].placements_valid {
            self.resolve_grid_placements(grid_idx);
        }
        let n_rows = self.grids[gid].rows.len();
        let gaps = (n_rows.saturating_sub(1) as f32) * gap;
        let mut used = gaps;
        let mut total_fr = 0.0_f32;
        for i in 0..n_rows {
            match self.grids[gid].rows[i] {
                TrackSize::Fixed(fx) => {
                    self.grids[gid].resolved_rows[i] = fx;
                    used += fx;
                }
                TrackSize::Percent(p) => {
                    let h = inner_h * p;
                    self.grids[gid].resolved_rows[i] = h;
                    used += h;
                }
                TrackSize::Fit | TrackSize::Auto => {
                    used += self.grids[gid].resolved_rows[i];
                }
                TrackSize::MinMax(lo, hi) => {
                    let h = self.grids[gid].resolved_rows[i].clamp(lo, hi);
                    self.grids[gid].resolved_rows[i] = h;
                    used += h;
                }
                TrackSize::Fr(fr) => {
                    total_fr += fr;
                }
            }
        }

        // CSS Grid §11.6 Maximize `MinMax` rows up to `hi` before §11.7 `Fr` distribution:
        let mut rem = (inner_h - used).max(0.0);
        loop {
            if rem <= 1e-4 {
                break;
            }
            let mut expandable = 0u32;
            for i in 0..n_rows {
                if let TrackSize::MinMax(_, hi) = self.grids[gid].rows[i]
                    && (hi.is_finite() || total_fr <= 0.0)
                    && self.grids[gid].resolved_rows[i] + 1e-5 < hi
                {
                    expandable += 1;
                }
            }
            if expandable == 0 {
                break;
            }
            let add = rem / (expandable as f32);
            let mut gained = 0.0_f32;
            for i in 0..n_rows {
                if let TrackSize::MinMax(_, hi) = self.grids[gid].rows[i]
                    && (hi.is_finite() || total_fr <= 0.0)
                {
                    let cur = self.grids[gid].resolved_rows[i];
                    if cur + 1e-5 < hi {
                        let next = (cur + add).min(hi);
                        self.grids[gid].resolved_rows[i] = next;
                        gained += next - cur;
                    }
                }
            }
            if gained <= 1e-6 {
                break;
            }
            rem = (rem - gained).max(0.0);
        }

        let (sf, inv_sf) = self.snap_scale_pair();
        let snap = self.snap_to_pixels;
        let mut snap_err = 0.0_f32;
        let mut off_y = 0.0_f32;
        let mut phys_off_y = 0.0_f64;
        for i in 0..n_rows {
            if let TrackSize::Fr(fr) = self.grids[gid].rows[i] {
                self.grids[gid].resolved_rows[i] = if total_fr > 0.0 {
                    rem * (fr / total_fr)
                } else {
                    0.0
                };
            }
            if snap {
                let ideal_phys = self.grids[gid].resolved_rows[i] * sf + snap_err;
                let snapped_phys = ideal_phys.round().max(0.0);
                snap_err = ideal_phys - snapped_phys;
                self.grids[gid].resolved_rows[i] =
                    Self::phys_to_logical(snapped_phys as f64, sf, inv_sf);
                self.grids[gid].row_offsets[i] = if gap == 0.0 {
                    let r_log = Self::phys_to_logical(phys_off_y, sf, inv_sf);
                    phys_off_y += snapped_phys as f64;
                    r_log
                } else {
                    Self::snap_px(off_y, sf, inv_sf)
                };
            } else {
                self.grids[gid].row_offsets[i] = off_y;
            }
            off_y += self.grids[gid].resolved_rows[i] + gap;
        }
    }

    fn place_grid_children(
        &mut self,
        grid_idx: usize,
        origin_x: f32,
        origin_y: f32,
        grid_spec: LayoutSpec,
        cur_clip: [f32; 4],
    ) {
        let (sf, inv_sf) = self.snap_scale_pair();
        let snap = self.snap_to_pixels;
        let gap = grid_spec.gap;
        let base_origin_x = if snap && gap == 0.0 {
            Self::snap_px(origin_x, sf, inv_sf)
        } else {
            origin_x
        };
        let base_origin_y = if snap && gap == 0.0 {
            Self::snap_px(origin_y, sf, inv_sf)
        } else {
            origin_y
        };
        let gid = grid_spec.aux_id as usize;
        if !self.grids[gid].placements_valid {
            self.resolve_grid_placements(grid_idx);
        }
        let p_len = self.grids[gid].placements.len();

        for p_i in 0..p_len {
            let p = self.grids[gid].placements[p_i];
            let c_idx = p.node.idx();
            let cs = self.specs[c_idx];
            let c0 = p.col as usize;
            let c_span = p.col_span as usize;
            let r0 = p.row as usize;
            let r_span = p.row_span as usize;

            let mut spanned_w = ((c_span.saturating_sub(1)) as f32) * gap;
            for k in 0..c_span {
                spanned_w += self.grids[gid].resolved_cols[c0 + k];
            }
            let mut track_h = ((r_span.saturating_sub(1)) as f32) * gap;
            for k in 0..r_span {
                track_h += self.grids[gid].resolved_rows[r0 + k];
            }

            let ch_w = self.boxes[c_idx].w;
            let ch_h = match cs.mode_h {
                SizingMode::Fixed => cs.val_h,
                SizingMode::Percent => track_h * cs.val_h,
                SizingMode::Grow => track_h,
                SizingMode::Fit => {
                    if grid_spec.cross_align == Align::Stretch {
                        track_h
                    } else {
                        self.boxes[c_idx].fit_h
                    }
                }
            }
            .clamp(cs.min_h, cs.max_h);

            let cell_x = base_origin_x + self.grids[gid].col_offsets[c0];
            let ch_x = match grid_spec.main_align {
                MainAlign::Start | MainAlign::SpaceBetween => cell_x,
                MainAlign::Center | MainAlign::SpaceEvenly => cell_x + (spanned_w - ch_w) * 0.5,
                MainAlign::End => cell_x + (spanned_w - ch_w),
            };

            let cell_y = base_origin_y + self.grids[gid].row_offsets[r0];
            let ch_y = match grid_spec.cross_align {
                Align::Start | Align::Stretch => cell_y,
                Align::Center => cell_y + (track_h - ch_h) * 0.5,
                Align::End => cell_y + (track_h - ch_h),
                Align::FirstBaseline => {
                    if r_span == 1 {
                        let mut row_asc = self.boxes[c_idx].baseline;
                        for q_i in 0..p_len {
                            let q = self.grids[gid].placements[q_i];
                            if q.row as usize == r0 && q.row_span == 1 {
                                let qb = self.boxes[q.node.idx()];
                                if (self.topo[q.node.idx()].flags & FLAG_HAS_BASELINE) != 0
                                    || qb.fit_h > 0.0
                                {
                                    row_asc = row_asc.max(qb.baseline);
                                }
                            }
                        }
                        cell_y + (row_asc - self.boxes[c_idx].baseline)
                    } else {
                        cell_y
                    }
                }
            };

            self.boxes[c_idx].x = if snap {
                Self::snap_px(ch_x, sf, inv_sf)
            } else {
                ch_x
            };
            self.boxes[c_idx].y = if snap {
                Self::snap_px(ch_y, sf, inv_sf)
            } else {
                ch_y
            };
            self.boxes[c_idx].h = if snap {
                Self::snap_px(ch_h, sf, inv_sf)
            } else {
                ch_h
            };
            self.clip_rects[c_idx] = cur_clip;
        }
    }
}

#[inline]
pub fn intersect_rect(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = (a[0] + a[2]).min(b[0] + b[2]);
    let y1 = (a[1] + a[3]).min(b[1] + b[3]);
    [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)]
}

#[inline]
fn sanitize_item_height(h: f32) -> f32 {
    if h.is_finite() && h > 0.0 {
        h
    } else {
        0.0
    }
}

/// Knuth error-free transformation (`a + b == s + e` in exact real arithmetic).
/// Preserves 106-bit double-`f64` precision across millions of non-dyadic Fenwick mutations!
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Fenwick-Tree Prefix-Sum Variable-Height Virtualizer with 106-Bit Compensated Zero-Drift Precision,
/// Anchor-Locked Scroll Stabilization, Overscan Windowing, and $O(\log N)$ Dynamic Mutations.
pub struct VirtualListIndex {
    bit: Vec<f64>,
    bit_err: Vec<f64>,
    heights: Vec<f32>,
}

impl VirtualListIndex {
    pub fn new(count: usize, estimated_h: f32) -> Self {
        let h = sanitize_item_height(estimated_h);
        let heights = vec![h; count];
        let mut bit = vec![0.0_f64; count + 1];
        let bit_err = vec![0.0_f64; count + 1];
        let h_f64 = h as f64;
        for (idx, slot) in bit.iter_mut().enumerate().skip(1) {
            // `idx.isolate_lowest_one()` is an exact power of two `2^k`, so multiplying `h_f64`
            // only shifts the IEEE-754 exponent with zero mantissa rounding error (`bit_err == 0.0`).
            *slot = (idx.isolate_lowest_one() as f64) * h_f64;
        }
        Self {
            bit,
            bit_err,
            heights,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.heights.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heights.is_empty()
    }

    #[inline]
    pub fn height_at(&self, index: usize) -> f32 {
        self.heights.get(index).copied().unwrap_or(0.0)
    }

    /// Append a single item to the end of the virtual list in $O(\log N)$ worst-case ($O(1)$ amortized)
    /// without rebuilding the Fenwick tree.
    #[inline]
    pub fn push(&mut self, h: f32) {
        let clean = sanitize_item_height(h);
        self.heights.push(clean);
        let m = self.heights.len();
        let base = m & (m - 1); // m - m.isolate_lowest_one()
        let mut sum = clean as f64;
        let mut err = 0.0_f64;
        let mut j = m - 1;
        while j > base {
            let (s, e) = two_sum(sum, self.bit[j]);
            sum = s;
            err += e + self.bit_err[j];
            j &= j - 1;
        }
        let (hi, lo) = two_sum(sum, err);
        self.bit.push(hi);
        self.bit_err.push(lo);
    }

    /// Remove and return the last item's height in exact $O(1)$ time without rebuilding the Fenwick tree.
    #[inline]
    pub fn pop(&mut self) -> Option<f32> {
        let h = self.heights.pop()?;
        self.bit.pop();
        self.bit_err.pop();
        Some(h)
    }

    /// Resize the virtual list to `new_len` (truncating in $O(1)$ or extending with `default_h`
    /// without rebuilding `1..=old_len`).
    pub fn resize(&mut self, new_len: usize, default_h: f32) {
        let old_len = self.heights.len();
        if new_len == old_len {
            return;
        }
        if new_len < old_len {
            self.heights.truncate(new_len);
            self.bit.truncate(new_len + 1);
            self.bit_err.truncate(new_len + 1);
            return;
        }

        let clean = sanitize_item_height(default_h);
        let h_f64 = clean as f64;
        self.heights.resize(new_len, clean);
        self.bit.resize(new_len + 1, 0.0);
        self.bit_err.resize(new_len + 1, 0.0);

        for m in (old_len + 1)..=new_len {
            let lsb = m.isolate_lowest_one();
            let base = m - lsb;
            if base >= old_len {
                // Entire Fenwick range `(base .. m]` consists of newly appended `default_h` items (`lsb = 2^k`).
                self.bit[m] = (lsb as f64) * h_f64;
                self.bit_err[m] = 0.0;
            } else {
                // Straddles `old_len` (happens at most `ceil(log2(new_len))` times across the entire suffix!).
                let tail_cnt = m - old_len;
                let tail_hi = (tail_cnt as f64) * h_f64;
                let tail_lo = (tail_cnt as f64).mul_add(h_f64, -tail_hi);
                let mut sum = tail_hi;
                let mut err = tail_lo;
                let mut j = old_len;
                while j > base {
                    let (s, e) = two_sum(sum, self.bit[j]);
                    sum = s;
                    err += e + self.bit_err[j];
                    j &= j - 1;
                }
                let (hi, lo) = two_sum(sum, err);
                self.bit[m] = hi;
                self.bit_err[m] = lo;
            }
        }
    }

    #[inline]
    pub fn set_height(&mut self, index: usize, new_h: f32) {
        if index >= self.heights.len() {
            return;
        }
        let clean = sanitize_item_height(new_h);
        let old = self.heights[index];
        if old.to_bits() == clean.to_bits() {
            return;
        }
        self.heights[index] = clean;
        // Exact `f64` difference of two 24-bit `f32` values + 106-bit `two_sum` Fenwick propagation
        // eliminates cumulative rounding drift across arbitrary millions of non-dyadic updates!
        let (d_hi, d_lo) = two_sum(clean as f64, -(old as f64));
        let mut i = index + 1;
        while i < self.bit.len() {
            let (s, e) = two_sum(self.bit[i], d_hi);
            let err = self.bit_err[i] + (e + d_lo);
            let (hi, lo) = two_sum(s, err);
            self.bit[i] = hi;
            self.bit_err[i] = lo;
            i += i.isolate_lowest_one();
        }
    }

    /// Update item height while adjusting `scroll_y` if `index < anchor_index`,
    /// preventing visible viewport jump when off-screen items resize!
    #[inline]
    pub fn set_height_anchored(
        &mut self,
        index: usize,
        new_h: f32,
        anchor_index: usize,
        scroll_y: &mut f64,
    ) {
        if index >= self.heights.len() {
            return;
        }
        let clean = sanitize_item_height(new_h);
        let old = self.heights[index];
        if old.to_bits() == clean.to_bits() {
            return;
        }
        let delta = (clean as f64) - (old as f64);
        self.set_height(index, clean);
        if index < anchor_index {
            *scroll_y = (*scroll_y + delta).max(0.0);
        }
    }

    #[inline]
    pub fn offset_of(&self, count: usize) -> f64 {
        let mut sum = 0.0_f64;
        let mut err = 0.0_f64;
        let mut i = count.min(self.heights.len());
        while i > 0 {
            let (s, e) = two_sum(sum, self.bit[i]);
            sum = s;
            err += e + self.bit_err[i];
            i &= i - 1;
        }
        sum + err
    }

    #[inline]
    pub fn total_height(&self) -> f64 {
        self.offset_of(self.heights.len())
    }

    /// $O(\log N)$ binary lifting on the Fenwick tree to locate `[start..end]` overlapping `[scroll_y, scroll_y + viewport_h)`.
    ///
    /// Exact boundary invariants:
    /// - Empty list (`n == 0`) returns `(0, 0, 0.0)`.
    /// - `first_offset` ALWAYS equals `self.offset_of(start)` bit-for-bit.
    /// - When `scroll_y` lands right on an item boundary `offset_of(k)`, scrolled-past items `< k` with `h > 0`
    ///   are excluded while any `0.0`-height items sitting at `offset_of(k)` are preserved at `start`.
    /// - When `scroll_y >= total_height` (with `scroll_y > 0.0` or no 0-height items at `0.0`), returns `(n, n, total_height)`.
    #[inline]
    pub fn visible_range(&self, scroll_y: f64, viewport_h: f64) -> (usize, usize, f64) {
        self.visible_range_overscan(scroll_y, viewport_h, 0.0)
    }

    /// $O(\log N)$ binary lifting with symmetrical pixel overscan `[max(scroll_y - overscan_px, 0.0), scroll_y + viewport_h + overscan_px)`.
    ///
    /// Returns `(start, end, first_offset)` where `first_offset == self.offset_of(start)`.
    pub fn visible_range_overscan(
        &self,
        scroll_y: f64,
        viewport_h: f64,
        overscan_px: f64,
    ) -> (usize, usize, f64) {
        let n = self.heights.len();
        if n == 0 {
            return (0, 0, 0.0);
        }
        let sy = if scroll_y.is_finite() {
            scroll_y.max(0.0)
        } else {
            0.0
        };
        let view_h = if viewport_h.is_finite() {
            viewport_h.max(0.0)
        } else {
            0.0
        };
        let over = if overscan_px.is_finite() {
            overscan_px.max(0.0)
        } else {
            0.0
        };

        let target = (sy - over).max(0.0);
        let limit = sy + view_h + over;

        let mut idx = 0usize;
        let mut acc = 0.0_f64;
        let mut acc_err = 0.0_f64;
        if target > 0.0 {
            // Strict `< target` binary lifting locates the first index where cumulative height reaches `>= target`,
            // preserving any `0.0`-height items sitting right at `target`!
            let mut bit_mask = if n.is_power_of_two() {
                n
            } else {
                n.next_power_of_two() >> 1
            };
            while bit_mask > 0 {
                let next = idx + bit_mask;
                if next <= n {
                    let (cand_s, cand_e) = two_sum(acc, self.bit[next]);
                    let cand_err = acc_err + cand_e + self.bit_err[next];
                    if cand_s + cand_err < target {
                        idx = next;
                        acc = cand_s;
                        acc_err = cand_err;
                    }
                }
                bit_mask >>= 1;
            }
            // If `target` sits right on the bottom boundary of positive-height item `idx`, step past `idx`:
            while idx < n {
                let h = self.heights[idx] as f64;
                let (next_s, next_e) = two_sum(acc, h);
                let next_err = acc_err + next_e;
                if h > 0.0 && next_s + next_err <= target {
                    acc = next_s;
                    acc_err = next_err;
                    idx += 1;
                } else {
                    break;
                }
            }
        }

        let start = idx.min(n);
        let first_offset = self.offset_of(start);
        let mut end = start;
        let mut cur_s = first_offset;
        let mut cur_e = 0.0_f64;
        while end < n && (cur_s + cur_e) < limit {
            let (ns, ne) = two_sum(cur_s, self.heights[end] as f64);
            cur_s = ns;
            cur_e += ne;
            end += 1;
        }
        (start, end, first_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_and_col_unbounded_grow_clamping_beyond_64_children() {
        // 150 `Grow` children (> 64 inline cap!) mixing tight `min`, tight `max`, and unconstrained weights.
        let mut tree = LayoutTree::new();
        let root = tree.alloc_node(LayoutSpec::row().fixed(3000.0, 100.0).gap(0.0));
        tree.root = root;

        let mut children = Vec::with_capacity(150);
        for i in 0..150 {
            let spec = if i % 3 == 0 {
                // 50 items with `min_w = 30.0` (unclamped share would be 20 -> freezes at 30!)
                LayoutSpec::col().width_grow(1.0).min_size(30.0, 0.0)
            } else if i % 3 == 1 {
                // 50 items with `max_w = 10.0` (freezes at 10!)
                LayoutSpec::col().width_grow(1.0).max_size(10.0, 100.0)
            } else {
                // 50 unconstrained items: must absorb exact remaining `3000 - 50*30 - 50*10 = 1000` -> `20.0` each!
                LayoutSpec::col().width_grow(1.0)
            };
            children.push(tree.alloc_node(spec));
        }
        tree.set_children(root, &children);
        tree.compute_layout(3000.0, 100.0);

        let mut sum_w = 0.0_f32;
        for (i, &ch) in children.iter().enumerate() {
            let w = tree.boxes[ch.idx()].w;
            sum_w += w;
            if i % 3 == 0 {
                assert!((w - 30.0).abs() < 1e-4, "child {i} min clamp failed: {w}");
            } else if i % 3 == 1 {
                assert!((w - 10.0).abs() < 1e-4, "child {i} max clamp failed: {w}");
            } else {
                assert!((w - 20.0).abs() < 1e-4, "child {i} free share failed: {w}");
            }
        }
        assert!((sum_w - 3000.0).abs() < 1e-2);

        // Verify symmetric `Col` height clamping for 150 `Grow` children AND exact CSS §9.7 sign rule:
        // Three `Grow(1.0)` children in 90px: `[0..100]`, `[50..100]`, `[0..25]` -> exact water-filling is `20, 50, 20`!
        let mut ctree = LayoutTree::new();
        let croot = ctree.alloc_node(LayoutSpec::col().fixed(100.0, 90.0));
        ctree.root = croot;
        let a = ctree.alloc_node(LayoutSpec::row().height_grow(1.0).max_size(100.0, 100.0));
        let b = ctree.alloc_node(LayoutSpec::row().height_grow(1.0).min_size(0.0, 50.0));
        let c = ctree.alloc_node(LayoutSpec::row().height_grow(1.0).max_size(100.0, 25.0));
        ctree.set_children(croot, &[a, b, c]);
        ctree.compute_layout(100.0, 90.0);
        assert!((ctree.boxes[a.idx()].h - 20.0).abs() < 1e-4);
        assert!((ctree.boxes[b.idx()].h - 50.0).abs() < 1e-4);
        assert!((ctree.boxes[c.idx()].h - 20.0).abs() < 1e-4);
    }

    #[test]
    fn wrap_row_per_line_cross_align_scroll_snap_and_min_max_h() {
        for align in [Align::Start, Align::Center, Align::End, Align::Stretch] {
            let mut tree = LayoutTree::new();
            tree.snap_to_pixels = true;
            let mut wrap_spec = LayoutSpec::wrap_row()
                .width_fixed(200.0)
                .min_size(0.0, 90.0)
                .max_size(200.0, 110.0)
                .gap(10.0)
                .align(align);
            wrap_spec.scroll_y = 15.0;
            let root = tree.alloc_node(wrap_spec);
            tree.root = root;

            // Line 1: two items (width 90 + 10 + 90 = 190 <= 200) of heights 20 and 40 -> `line_h = 40`
            let i0 = tree.alloc_node(LayoutSpec::col().width_fixed(90.0).min_size(0.0, 20.0).max_size(90.0, 32.0));
            let i1 = tree.alloc_node(LayoutSpec::col().fixed(90.0, 40.0));
            // Line 2: one item (width 120) of height 30 -> `line_h = 30`, starts at `y = -15 + 40 + 10 = 35`
            let i2 = tree.alloc_node(LayoutSpec::col().fixed(120.0, 30.0));
            tree.set_children(root, &[i0, i1, i2]);
            tree.compute_layout(200.0, 300.0);

            // Intrinsic height `40 + 10 + 30 = 80` clamped up to `min_h = 90`:
            assert_eq!(tree.boxes[root.idx()].h, 90.0);
            assert_eq!(tree.boxes[i1.idx()].y, -15.0);
            assert_eq!(tree.boxes[i2.idx()].y, 35.0);

            let b0 = tree.boxes[i0.idx()];
            match align {
                Align::Start => {
                    assert_eq!(b0.y, -15.0);
                    assert_eq!(b0.h, 20.0);
                }
                Align::Center => {
                    assert_eq!(b0.y, -5.0); // -15 + (40 - 20) * 0.5
                    assert_eq!(b0.h, 20.0);
                }
                Align::End => {
                    assert_eq!(b0.y, 5.0); // -15 + (40 - 20)
                    assert_eq!(b0.h, 20.0);
                }
                Align::Stretch => {
                    // Stretches toward `line_h = 40`, clamped by `i0.max_h = 32`!
                    assert_eq!(b0.y, -15.0);
                    assert_eq!(b0.h, 32.0);
                }
                Align::FirstBaseline => unreachable!(),
            }
        }
    }

    #[test]
    fn grid_2d_spans_occupancy_row_overflow_percent_auto_scroll_and_snap() {
        let mut tree = LayoutTree::new();
        tree.snap_to_pixels = true;
        let mut gspec = LayoutSpec::col().fixed(400.0, 300.0).gap(10.0);
        gspec.scroll_y = 20.0;
        let grid = tree.alloc_node(gspec);
        // 3 columns: Percent(0.25) = 100, Auto (from 60px item) = 60, Fr(1.0) = 400 - 20 - 160 = 220
        // 2 declared rows: Fixed(50), Fr(1.0) = 300 - 10 - 50 = 240
        tree.attach_grid(
            grid,
            vec![TrackSize::Percent(0.25), TrackSize::Auto, TrackSize::Fr(1.0)],
            vec![TrackSize::Fixed(50.0), TrackSize::Fr(1.0)],
        );
        tree.root = grid;

        // Child 0 at (col=0, row=0) with `row_span = 2` -> occupies (0,0) AND (0,1)!
        let tall_sidebar = tree.alloc_node(LayoutSpec::col().grow(1.0).grid_span(1, 2));
        // Child 1 at (col=1, row=0): Auto width 60px
        let cell_1_0 = tree.alloc_node(LayoutSpec::col().fixed(60.0, 40.0));
        // Child 2 at (col=2, row=0)
        let cell_2_0 = tree.alloc_node(LayoutSpec::col().grow(1.0));
        // Child 3 MUST skip occupied (col=0, row=1) and land at (col=1, row=1) spanning 2 cols (`col_span = 2`)!
        let spanned_1_1 = tree.alloc_node(LayoutSpec::col().grow(1.0).grid_span(2, 1));
        // Child 4 exceeds declared 2 rows -> clamped safely to `row = 1` (`n_rows - 1`), `col = 0` without panic!
        let overflow_child = tree.alloc_node(LayoutSpec::col().fixed(50.0, 25.0));

        tree.set_children(
            grid,
            &[tall_sidebar, cell_1_0, cell_2_0, spanned_1_1, overflow_child],
        );
        tree.compute_layout(400.0, 300.0);

        let b_side = tree.boxes[tall_sidebar.idx()];
        assert_eq!(b_side.x, 0.0);
        assert_eq!(b_side.y, -20.0); // scroll_y = 20
        assert_eq!(b_side.w, 100.0);
        assert_eq!(b_side.h, 300.0); // 50 + 10 + 240

        let b_spanned = tree.boxes[spanned_1_1.idx()];
        assert_eq!(b_spanned.x, 110.0); // col 1 offset = 100 + 10
        assert_eq!(b_spanned.y, 40.0); // row 1 offset = 50 + 10 - 20(scroll)
        assert_eq!(b_spanned.w, 290.0); // Auto(60) + gap(10) + Fr(220)
        assert_eq!(b_spanned.h, 240.0);

        let b_over = tree.boxes[overflow_child.idx()];
        assert_eq!(b_over.x, 0.0);
        assert_eq!(b_over.y, 40.0);
        assert_eq!(b_over.h, 25.0);
    }

    #[test]
    fn pass4_anchor_placements_flip_clamp_and_establish_nested_clips() {
        let mut tree = LayoutTree::new();
        let root = tree.alloc_node(LayoutSpec::col().grow(1.0));
        tree.topo[root.idx()].flags |= FLAG_CLIPS_CHILDREN;
        tree.root = root;

        // Target near bottom-right corner (`x = 740, y = 560, w = 50, h = 30` in `800x600` window):
        let spacer = tree.alloc_node(LayoutSpec::col().fixed(10.0, 560.0));
        let row = tree.alloc_node(LayoutSpec::row().width_fixed(800.0).pad_xy(740.0, 0.0));
        let target = tree.alloc_node(LayoutSpec::col().fixed(50.0, 30.0));

        // Popover (`200x120`) with `BottomEnd`:
        // - Bottom (`560 + 30 + 4 = 594`, `+120 = 714 > 600`) flips ABOVE target: `560 - 120 - 4 = 436`!
        // - End X: `740 + 50 - 200 = 590` (within `[0..600]`)!
        let popover = tree.alloc_node(LayoutSpec::col().fixed(200.0, 120.0));
        tree.topo[popover.idx()].flags |= FLAG_CLIPS_CHILDREN;
        let pop_child = tree.alloc_node(LayoutSpec::col().fixed(200.0, 200.0));
        tree.set_floating_anchor(popover, target, AnchorPlacement::BottomEnd, 0.0, 4.0);

        // Second popover at `(0, 0)` target (`TopStart`) to verify flip-to-bottom AND clip establishment when `dx == 0, dy == 0`:
        let top_pop = tree.alloc_node(LayoutSpec::col().fixed(100.0, 80.0));
        tree.set_floating_anchor(top_pop, spacer, AnchorPlacement::TopStart, 0.0, 0.0);

        tree.set_children(row, &[target]);
        tree.set_children(popover, &[pop_child]);
        tree.set_children(root, &[spacer, row, popover, top_pop]);
        tree.compute_layout(800.0, 600.0);

        let pb = tree.boxes[popover.idx()];
        assert_eq!((pb.x, pb.y), (590.0, 436.0));
        assert_eq!(tree.clip_rects[popover.idx()], [0.0, 0.0, 800.0, 600.0]);
        // Child inside clipping popover must be clipped to the popover's translated bounds intersected with viewport!
        assert_eq!(tree.clip_rects[pop_child.idx()], [590.0, 436.0, 200.0, 120.0]);
        assert_eq!(tree.clip_rects[top_pop.idx()], [0.0, 0.0, 800.0, 600.0]);

        // Also verify `RightStart` -> left flip and `LeftEnd` -> right flip:
        tree.set_floating_anchor(popover, target, AnchorPlacement::RightStart, 6.0, 0.0);
        tree.compute_layout(800.0, 600.0);
        // `740 + 50 + 6 + 200 > 800` -> flips left to `740 - 200 - 6 = 534`, Y clamped from `560` to `600 - 120 = 480`!
        assert_eq!(
            (tree.boxes[popover.idx()].x, tree.boxes[popover.idx()].y),
            (534.0, 480.0)
        );
    }

    #[test]
    fn virtual_list_index_boundaries_zero_heights_and_dynamic_mutations() {
        // 1. Empty list:
        let mut vlist = VirtualListIndex::new(0, 24.0);
        assert!(vlist.is_empty());
        assert_eq!(vlist.visible_range(0.0, 100.0), (0, 0, 0.0));
        assert_eq!(vlist.pop(), None);

        // 2. Push, pop, and boundary-exact queries (including 0-height items!):
        for &h in &[20.0_f32, 0.0, 0.0, 30.0, 50.0, 0.0] {
            vlist.push(h);
        }
        assert_eq!(vlist.len(), 6);
        assert!((vlist.total_height() - 100.0).abs() < 1e-9);

        // `scroll_y = 20.0` sits right on the boundary after item 0 (`20px`):
        // Item 0 (`[0..20)`) must NOT be included; 0-height items 1 & 2 at `y = 20` + item 3 (`[20..50)`) MUST be included!
        let (st, en, off) = vlist.visible_range(20.0, 30.0);
        assert_eq!((st, en), (1, 4));
        assert_eq!(off.to_bits(), vlist.offset_of(st).to_bits());
        assert_eq!(off, 20.0);

        // Past end (`scroll_y = 120.0 > total_height = 100.0`):
        let (st_end, en_end, off_end) = vlist.visible_range(120.0, 50.0);
        assert_eq!((st_end, en_end), (6, 6));
        assert_eq!(off_end.to_bits(), vlist.offset_of(6).to_bits());

        // Pop trailing items and verify Fenwick sums stay exact:
        assert_eq!(vlist.pop(), Some(0.0));
        assert_eq!(vlist.pop(), Some(50.0));
        assert_eq!(vlist.len(), 4);
        assert!((vlist.total_height() - 50.0).abs() < 1e-9);

        // 3. `resize` grow (straddling binary Fenwick intervals) & shrink, verified against naive prefix sum at every index!
        vlist.set_height(1, 17.5);
        vlist.resize(257, 12.25);
        assert_eq!(vlist.len(), 257);
        let mut expected = 0.0_f64;
        for i in 0..=vlist.len() {
            let got = vlist.offset_of(i);
            assert!(
                (got - expected).abs() < 1e-6,
                "Fenwick mismatch at {i}: got {got}, expected {expected}"
            );
            if i < vlist.len() {
                expected += vlist.height_at(i) as f64;
            }
        }

        vlist.resize(3, 10.0);
        assert_eq!(vlist.len(), 3);
        assert!((vlist.total_height() - 37.5).abs() < 1e-9);

        // 4. All-zero-height list at `scroll_y = 0`:
        let zeros = VirtualListIndex::new(8, 0.0);
        assert_eq!(zeros.visible_range(0.0, 50.0), (0, 8, 0.0));
    }

    #[test]
    fn adversarial_empty_tree_zero_viewport_malformed_specs_and_140_shrink_children() {
        let mut tree = LayoutTree::new();
        // Empty tree on 0x0 and NaN viewports:
        tree.compute_layout(0.0, 0.0);
        tree.compute_layout(f32::NAN, f32::NEG_INFINITY);

        // Malformed spec (`NaN`, negative gaps, `min > max`): repaired with `FLAG_REPAIRED_SPEC`!
        let mut bad = LayoutSpec::row();
        bad.pad_l = -10.0;
        bad.gap = f32::NAN;
        bad.min_w = 200.0;
        bad.max_w = 50.0;
        bad.scroll_y = f32::INFINITY;
        let root = tree.alloc_node(bad);
        assert_ne!(tree.topo[root.idx()].flags & FLAG_REPAIRED_SPEC, 0);
        assert_eq!(tree.specs[root.idx()].max_w, 200.0);
        tree.root = root;

        // 140 `Fit` shrinkable children (> 64 inline cap!) inside a narrow 1400px `Row`:
        // - 70 children have `fit_w = 20, min_w = 15` (can shrink at most 5px -> freeze at 15px!)
        // - 70 children have `fit_w = 20, min_w = 0` (absorb remaining deficit down to 5px each!)
        // Total = `70 * 15 + 70 * 5 = 1050 + 350 = 1400px`!
        tree.update_spec(root, LayoutSpec::row().fixed(1400.0, 40.0));
        let mut kids = Vec::with_capacity(140);
        for i in 0..140 {
            let child = tree.alloc_node(LayoutSpec::col().min_size(if i % 2 == 0 { 15.0 } else { 0.0 }, 0.0));
            let inner = tree.alloc_node(LayoutSpec::col().fixed(20.0, 10.0));
            tree.set_children(child, &[inner]);
            kids.push(child);
        }
        tree.set_children(root, &kids);
        tree.compute_layout(1400.0, 40.0);

        for (i, &k) in kids.iter().enumerate() {
            let expected = if i % 2 == 0 { 15.0 } else { 5.0 };
            assert!(
                (tree.boxes[k.idx()].w - expected).abs() < 1e-3,
                "shrink child {i}: got {}, expected {expected}",
                tree.boxes[k.idx()].w
            );
        }

        // `0x0` viewport stability and `free_subtree(root)` hygiene:
        tree.compute_layout(0.0, 0.0);
        tree.free_subtree(root);
        assert!(tree.root.is_none());
        tree.compute_layout(800.0, 600.0);
    }

    fn build_complex_scroll_oracle_scene(
        snap_to_pixels: bool,
        outer_scroll_y: f32,
        inner_scroll_y: f32,
        wrap_scroll_y: f32,
        grid_scroll_y: f32,
        fonts: &FontSystem,
    ) -> (LayoutTree, NodeId, NodeId, NodeId, NodeId) {
        let mut tree = LayoutTree::new();
        tree.snap_to_pixels = snap_to_pixels;

        let root = tree.alloc_node(LayoutSpec::col().fixed(840.0, 620.0).pad(12.0).gap(8.0));
        tree.topo[root.idx()].flags |= FLAG_CLIPS_CHILDREN;
        tree.root = root;

        let mut outer_spec = LayoutSpec::col()
            .fixed(816.0, 520.0)
            .pad_xy(10.5, 8.25)
            .gap(6.5);
        outer_spec.scroll_y = outer_scroll_y;
        let outer_scroll = tree.alloc_node(outer_spec);
        tree.topo[outer_scroll.idx()].flags |= FLAG_CLIPS_CHILDREN | FLAG_SCROLLABLE;

        // 1. Row with `FirstBaseline`, `SpaceBetween`, mixed heights, and text:
        let row_bl = tree.alloc_node(
            LayoutSpec::row()
                .width_grow(1.0)
                .pad_xy(6.3, 4.7)
                .gap(7.0)
                .align(Align::FirstBaseline)
                .justify(MainAlign::SpaceBetween),
        );
        let r_txt = tree.alloc_node(LayoutSpec::col().width_fixed(180.0));
        tree.attach_shaped_text(
            r_txt,
            "Scroll fast-path oracle paragraph wrapping across lines",
            FontId::FALLBACK,
            15.0,
            20.0,
            fonts,
        );
        let r_target = tree.alloc_node(LayoutSpec::col().fixed(92.4, 34.6));
        let r_end = tree.alloc_node(LayoutSpec::col().fixed(64.2, 22.8));
        tree.set_children(row_bl, &[r_txt, r_target, r_end]);

        // 2. Nested scrollable `Col` with `Align::Center` and `MainAlign::Center`:
        let mut inner_spec = LayoutSpec::col()
            .fixed(760.0, 140.0)
            .pad_xy(8.4, 6.2)
            .gap(5.25)
            .align(Align::Center)
            .justify(MainAlign::Center);
        inner_spec.scroll_y = inner_scroll_y;
        let inner_scroll = tree.alloc_node(inner_spec);
        tree.topo[inner_scroll.idx()].flags |= FLAG_CLIPS_CHILDREN;
        let ic0 = tree.alloc_node(LayoutSpec::col().fixed(320.3, 42.4));
        let ic1 = tree.alloc_node(LayoutSpec::col().fixed(280.7, 56.8));
        tree.set_children(inner_scroll, &[ic0, ic1]);

        // 3. Scrollable `WrapRow` with `Align::Center` and `MainAlign::SpaceEvenly`:
        let mut wrap_spec = LayoutSpec::wrap_row()
            .fixed(420.0, 110.0)
            .pad_xy(5.5, 7.5)
            .gap(8.25)
            .align(Align::Center)
            .justify(MainAlign::SpaceEvenly);
        wrap_spec.scroll_y = wrap_scroll_y;
        let wrap_scroll = tree.alloc_node(wrap_spec);
        tree.topo[wrap_scroll.idx()].flags |= FLAG_CLIPS_CHILDREN;
        let wk0 = tree.alloc_node(LayoutSpec::col().fixed(150.2, 26.4));
        let wk1 = tree.alloc_node(LayoutSpec::col().fixed(160.6, 38.2));
        let wk2 = tree.alloc_node(LayoutSpec::col().fixed(190.4, 31.6));
        tree.set_children(wrap_scroll, &[wk0, wk1, wk2]);

        // 4. Scrollable 2D `Grid` with `col_span=2`, `row_span=2`, `MinMax`, `Auto`, `Fr`, and `Align::Center`:
        let mut grid_spec = LayoutSpec::col()
            .fixed(500.0, 150.0)
            .pad_xy(6.0, 5.5)
            .gap(7.5)
            .align(Align::Center)
            .justify(MainAlign::End);
        grid_spec.scroll_y = grid_scroll_y;
        let grid_scroll = tree.alloc_node(grid_spec);
        tree.topo[grid_scroll.idx()].flags |= FLAG_CLIPS_CHILDREN;
        tree.attach_grid(
            grid_scroll,
            vec![
                TrackSize::MinMax(60.0, 110.0),
                TrackSize::Auto,
                TrackSize::Fr(1.0),
            ],
            vec![TrackSize::Fit, TrackSize::Fixed(55.0)],
        );
        let gc0 = tree.alloc_node(LayoutSpec::col().fixed(150.0, 32.5).grid_span(2, 1));
        let gc1 = tree.alloc_node(LayoutSpec::col().fixed(70.0, 24.0));
        let gc2 = tree.alloc_node(LayoutSpec::col().fixed(65.0, 30.2));
        let gc3 = tree.alloc_node(LayoutSpec::col().grow(1.0).grid_span(2, 1));
        tree.set_children(grid_scroll, &[gc0, gc1, gc2, gc3]);

        // 5. Pass 4 Floating Anchor attached to `r_target` (flips BottomEnd <-> TopEnd as `outer_scroll` scrolls!):
        let popover = tree.alloc_node(LayoutSpec::col().fixed(180.0, 96.0).pad(6.5).gap(4.0));
        tree.topo[popover.idx()].flags |= FLAG_CLIPS_CHILDREN;
        let pop_inner = tree.alloc_node(LayoutSpec::row().fixed(160.0, 44.0).align(Align::End));
        let pop_leaf = tree.alloc_node(LayoutSpec::col().fixed(50.0, 18.5));
        tree.set_children(pop_inner, &[pop_leaf]);
        tree.set_children(popover, &[pop_inner]);
        tree.set_floating_anchor(popover, r_target, AnchorPlacement::BottomEnd, -3.5, 6.25);

        tree.set_children(
            outer_scroll,
            &[row_bl, inner_scroll, wrap_scroll, grid_scroll],
        );
        tree.set_children(root, &[outer_scroll, popover]);
        tree.compute_layout(840.0, 620.0);

        (tree, outer_scroll, inner_scroll, wrap_scroll, grid_scroll)
    }

    fn assert_trees_bit_identical(inc: &LayoutTree, fresh: &LayoutTree, ctx: &str) {
        assert_eq!(inc.boxes.len(), fresh.boxes.len(), "{ctx}: box count");
        for i in 0..inc.boxes.len() {
            let a = inc.boxes[i];
            let b = fresh.boxes[i];
            assert_eq!(
                (
                    a.x.to_bits(),
                    a.y.to_bits(),
                    a.w.to_bits(),
                    a.h.to_bits(),
                    a.fit_w.to_bits(),
                    a.fit_h.to_bits(),
                    a.min_content_w.to_bits(),
                    a.baseline.to_bits()
                ),
                (
                    b.x.to_bits(),
                    b.y.to_bits(),
                    b.w.to_bits(),
                    b.h.to_bits(),
                    b.fit_w.to_bits(),
                    b.fit_h.to_bits(),
                    b.min_content_w.to_bits(),
                    b.baseline.to_bits()
                ),
                "{ctx}: ComputedBox mismatch at node {i}: inc={a:?} vs fresh={b:?}"
            );
            let ca = inc.clip_rects[i];
            let cb = fresh.clip_rects[i];
            assert_eq!(
                ca.map(f32::to_bits),
                cb.map(f32::to_bits),
                "{ctx}: clip_rect mismatch at node {i}: inc={ca:?} vs fresh={cb:?}"
            );
        }
    }

    #[test]
    fn scroll_only_fast_path_bit_identical_to_full_solve_across_all_axes_clips_snap_and_anchors() {
        let fonts = FontSystem::new();
        let scroll_steps: &[(f32, f32, f32, f32)] = &[
            (0.4, 0.0, 0.0, 0.0),
            (13.7, 9.3, 0.0, 0.0),
            (-420.6, 25.8, 17.4, 11.9), // Forces `r_target` near bottom so Pass 4 `BottomEnd` flips above!
            (1_000_000.0, 500_000.0, 250_000.0, 125_000.0), // Extreme float jump (exposes naive `y += dy` ULP loss!)
            (27.35, 4.15, 19.85, 6.45), // Recover back to subpixel coordinates with zero precision loss!
            (0.0, 0.0, 0.0, 0.0),
        ];

        for snap in [false, true] {
            let (mut inc, outer_s, inner_s, wrap_s, grid_s) =
                build_complex_scroll_oracle_scene(snap, 0.0, 0.0, 0.0, 0.0, &fonts);

            for (step_idx, &(os, is, ws, gs)) in scroll_steps.iter().enumerate() {
                // Alternate inner-first vs outer-first mutation order to exercise nested ancestor subsumption!
                if step_idx % 2 == 0 {
                    inc.set_scroll_y(inner_s, is);
                    inc.set_scroll_y(wrap_s, ws);
                    inc.set_scroll_y(grid_s, gs);
                    inc.set_scroll_y(outer_s, os);
                } else {
                    inc.set_scroll_y(outer_s, os);
                    inc.set_scroll_y(grid_s, gs);
                    inc.set_scroll_y(wrap_s, ws);
                    inc.set_scroll_y(inner_s, is);
                }

                // Verify `FLAG_DIRTY_SCROLL` was set without escalating to `FLAG_DIRTY_LAYOUT`:
                assert_ne!(inc.topo[outer_s.idx()].flags & FLAG_DIRTY_SCROLL, 0);
                assert_eq!(inc.topo[outer_s.idx()].flags & FLAG_DIRTY_LAYOUT, 0);
                assert_eq!(inc.topo[inc.root.idx()].flags & FLAG_DIRTY_LAYOUT, 0);

                inc.compute_layout(840.0, 620.0);
                assert!(inc.last_metrics.boundaries_solved >= 1);
                assert!(inc.last_metrics.inline_passes_skipped >= 1);
                assert_eq!(inc.last_metrics.text_shaper_calls, 0);
                assert_eq!(inc.last_metrics.text_interval_hits, 0);
                assert!(!inc.solved_nodes_last_frame.is_empty());

                let (fresh, _, _, _, _) =
                    build_complex_scroll_oracle_scene(snap, os, is, ws, gs, &fonts);
                assert_trees_bit_identical(
                    &inc,
                    &fresh,
                    &format!("snap={snap} step={step_idx} ({os}, {is}, {ws}, {gs})"),
                );
            }
        }
    }

    #[test]
    fn grid_css_11_5_spanned_intrinsic_order_independence_minmax_and_cell_alignment() {
        // 1. Order-independence: `col_span = 3` (`fit_w = 90`) BEFORE `col_span = 2` (`fit_w = 80`)
        //    over `[Auto, Auto, Auto]` (`gap = 0`) must size tracks to `[40, 40, 10]` (total `90`), NOT `110`!
        let mut g1 = LayoutTree::new();
        let grid1 = g1.alloc_node(LayoutSpec::col()); // Fit width & height!
        g1.attach_grid(
            grid1,
            vec![TrackSize::Auto, TrackSize::Auto, TrackSize::Auto],
            vec![TrackSize::Fit, TrackSize::Fit],
        );
        g1.root = grid1;
        let span3_first = g1.alloc_node(LayoutSpec::col().fixed(90.0, 20.0).grid_span(3, 1));
        let span2_second = g1.alloc_node(LayoutSpec::col().fixed(80.0, 20.0).grid_span(2, 1));
        g1.set_children(grid1, &[span3_first, span2_second]);
        g1.compute_layout(800.0, 600.0);
        assert!(
            (g1.boxes[grid1.idx()].fit_w - 90.0).abs() < 1e-4,
            "CSS §11.5 ascending span order must yield fit_w=90.0, got {}",
            g1.boxes[grid1.idx()].fit_w
        );

        // 2. Simultaneous `planned_increase` symmetry: two overlapping `col_span = 2` items of width `100`
        //    at cols `0..2` (row 0) and cols `1..3` (row 1) over `[Auto, Auto, Auto]` must yield `[50, 50, 50]` (`150`)!
        let mut g2 = LayoutTree::new();
        let grid2 = g2.alloc_node(LayoutSpec::col());
        g2.attach_grid(
            grid2,
            vec![TrackSize::Auto, TrackSize::Auto, TrackSize::Auto],
            vec![TrackSize::Fit, TrackSize::Fit],
        );
        g2.root = grid2;
        let r0_span2 = g2.alloc_node(LayoutSpec::col().fixed(100.0, 20.0).grid_span(2, 1));
        let r0_pad = g2.alloc_node(LayoutSpec::col().fixed(0.0, 20.0));
        let r1_pad = g2.alloc_node(LayoutSpec::col().fixed(0.0, 20.0));
        let r1_span2 = g2.alloc_node(LayoutSpec::col().fixed(100.0, 20.0).grid_span(2, 1));
        g2.set_children(grid2, &[r0_span2, r0_pad, r1_pad, r1_span2]);
        g2.compute_layout(800.0, 600.0);
        assert_eq!(g2.grids[0].resolved_cols, vec![50.0, 50.0, 50.0]);
        assert!((g2.boxes[grid2.idx()].fit_w - 150.0).abs() < 1e-4);

        // 3. Spanned `col_span = 2` & `row_span = 2` over `[Fit, MinMax(30.0, 55.0)]` (`gap = 10`, child `120x120`):
        //    `MinMax` track caps at `55.0` and `Fit` track absorbs remaining `55.0` (`55 + 10 + 55 = 120`)!
        let mut g3 = LayoutTree::new();
        let grid3 = g3.alloc_node(LayoutSpec::col().gap(10.0));
        g3.attach_grid(
            grid3,
            vec![TrackSize::Fit, TrackSize::MinMax(30.0, 55.0)],
            vec![TrackSize::Fit, TrackSize::MinMax(30.0, 55.0)],
        );
        g3.root = grid3;
        let big_2x2 = g3.alloc_node(LayoutSpec::col().fixed(120.0, 120.0).grid_span(2, 2));
        g3.set_children(grid3, &[big_2x2]);
        g3.compute_layout(800.0, 600.0);
        assert!((g3.grids[0].resolved_cols[0] - 55.0).abs() < 1e-4);
        assert!((g3.grids[0].resolved_cols[1] - 55.0).abs() < 1e-4);
        assert!((g3.grids[0].resolved_rows[0] - 55.0).abs() < 1e-4);
        assert!((g3.grids[0].resolved_rows[1] - 55.0).abs() < 1e-4);
        assert!((g3.boxes[grid3.idx()].fit_w - 120.0).abs() < 1e-4);
        assert!((g3.boxes[grid3.idx()].fit_h - 120.0).abs() < 1e-4);

        // 4. Cell alignment (`Center`, `End`, `Stretch`, `MainAlign::Center`) inside Grid tracks:
        for align in [Align::Start, Align::Center, Align::End, Align::Stretch] {
            let mut ga = LayoutTree::new();
            let gr = ga.alloc_node(
                LayoutSpec::col()
                    .fixed(200.0, 100.0)
                    .align(align)
                    .justify(MainAlign::Center),
            );
            ga.attach_grid(gr, vec![TrackSize::Fixed(200.0)], vec![TrackSize::Fixed(100.0)]);
            ga.root = gr;
            let mut item_spec = LayoutSpec::col().width_fixed(80.0).min_size(0.0, 30.0).max_size(200.0, 70.0);
            if align != Align::Stretch {
                item_spec = item_spec.height_fixed(30.0);
            }
            let item = ga.alloc_node(item_spec);
            ga.set_children(gr, &[item]);
            ga.compute_layout(200.0, 100.0);
            let b = ga.boxes[item.idx()];
            assert!((b.x - 60.0).abs() < 1e-4); // (200 - 80) * 0.5
            match align {
                Align::Start => assert!((b.y - 0.0).abs() < 1e-4 && (b.h - 30.0).abs() < 1e-4),
                Align::Center => assert!((b.y - 35.0).abs() < 1e-4 && (b.h - 30.0).abs() < 1e-4),
                Align::End => assert!((b.y - 70.0).abs() < 1e-4 && (b.h - 30.0).abs() < 1e-4),
                Align::Stretch => assert!((b.y - 0.0).abs() < 1e-4 && (b.h - 70.0).abs() < 1e-4), // clamped to max_h=70
                Align::FirstBaseline => unreachable!(),
            }
        }

        // 5. Wide `u8` spans (`col_span = 20, row_span = 20`, beyond 4-bit nibble `15`):
        //    20 `Auto` columns & 20 `Fit` rows (`gap = 2.0`, so `19 * 2.0 = 38.0` gaps) spanned by a `438 x 238` item
        //    must distribute `400 / 20 = 20.0` per column and `200 / 20 = 10.0` per row with zero truncation!
        let mut gw = LayoutTree::new();
        let grid_w = gw.alloc_node(LayoutSpec::col().gap(2.0));
        gw.attach_grid(
            grid_w,
            vec![TrackSize::Auto; 20],
            vec![TrackSize::Fit; 20],
        );
        gw.root = grid_w;
        let span20_spec = LayoutSpec::col().fixed(438.0, 238.0).grid_span(20, 20);
        assert_eq!((span20_spec.col_span(), span20_spec.row_span()), (20, 20));
        let span20 = gw.alloc_node(span20_spec);
        gw.set_children(grid_w, &[span20]);
        gw.compute_layout(1000.0, 1000.0);
        assert!((gw.boxes[grid_w.idx()].fit_w - 438.0).abs() < 1e-3);
        assert!((gw.boxes[grid_w.idx()].fit_h - 238.0).abs() < 1e-3);
        assert!(gw.grids[0].resolved_cols.iter().all(|&c| (c - 20.0).abs() < 1e-3));
        assert!(gw.grids[0].resolved_rows.iter().all(|&r| (r - 10.0).abs() < 1e-3));
        assert!((gw.boxes[span20.idx()].w - 438.0).abs() < 1e-3);
        assert!((gw.boxes[span20.idx()].h - 238.0).abs() < 1e-3);
    }

    #[test]
    fn first_baseline_deep_propagation_with_padding_alignment_and_empty_children() {
        let fonts = FontSystem::new();
        let mut tree = LayoutTree::new();
        let root = tree.alloc_node(
            LayoutSpec::row()
                .width_fixed(900.0)
                .pad_xy(12.0, 14.0)
                .gap(16.0)
                .align(Align::FirstBaseline),
        );
        tree.root = root;

        // Branch 1: Direct large heading text (`font_size = 24, line_h = 32`):
        let direct_txt = tree.alloc_node(LayoutSpec::col().pad_xy(4.0, 6.0));
        tree.attach_shaped_text(direct_txt, "Heading", FontId::FALLBACK, 24.0, 32.0, &fonts);

        // Branch 2: Nested `Col` with padding, leading empty `Row {}`, leading empty `WrapRow {}`,
        //           decorative baselineless bar (`fixed(40, 6)`), and `Row(Align::Center)` with `[Icon(30x30), Text]`:
        let deep_col = tree.alloc_node(LayoutSpec::col().pad_xy(8.0, 9.0).gap(5.0));
        let empty_row = tree.alloc_node(LayoutSpec::row());
        let empty_wrap = tree.alloc_node(LayoutSpec::wrap_row());
        let deco_bar = tree.alloc_node(LayoutSpec::row().fixed(40.0, 6.0));
        let inner_centered_row = tree.alloc_node(
            LayoutSpec::row()
                .pad_xy(3.0, 4.0)
                .gap(6.0)
                .align(Align::Center),
        );
        let icon_box = tree.alloc_node(LayoutSpec::col().fixed(30.0, 30.0));
        let col_txt = tree.alloc_node(LayoutSpec::col().pad_xy(0.0, 2.0));
        tree.attach_shaped_text(col_txt, "Deep Col", FontId::FALLBACK, 13.0, 18.0, &fonts);
        tree.set_children(inner_centered_row, &[icon_box, col_txt]);
        tree.set_children(
            deep_col,
            &[empty_row, empty_wrap, deco_bar, inner_centered_row],
        );

        // Branch 3: `WrapRow` with `pad_t = 7.0`, leading empty `Col {}`, first-line text, and wrapped second line:
        let deep_wrap = tree.alloc_node(LayoutSpec::wrap_row().width_fixed(180.0).pad_xy(5.0, 7.0).gap(6.0));
        let empty_col = tree.alloc_node(LayoutSpec::col());
        let wrap_txt1 = tree.alloc_node(LayoutSpec::col().width_fixed(110.0));
        tree.attach_shaped_text(wrap_txt1, "Tag One", FontId::FALLBACK, 15.0, 20.0, &fonts);
        let wrap_txt2 = tree.alloc_node(LayoutSpec::col().width_fixed(110.0));
        tree.attach_shaped_text(wrap_txt2, "Tag Two", FontId::FALLBACK, 18.0, 24.0, &fonts);
        tree.set_children(deep_wrap, &[empty_col, wrap_txt1, wrap_txt2]);

        // Branch 4: Nested `Grid` with `pad_t = 11.0` and text in row 0:
        let deep_grid = tree.alloc_node(LayoutSpec::col().pad_xy(4.0, 11.0).gap(4.0));
        tree.attach_grid(deep_grid, vec![TrackSize::Fit], vec![TrackSize::Fit]);
        let grid_txt = tree.alloc_node(LayoutSpec::col());
        tree.attach_shaped_text(grid_txt, "Grid Cell", FontId::FALLBACK, 14.0, 19.0, &fonts);
        tree.set_children(deep_grid, &[grid_txt]);

        // Branch 5: Completely empty sibling container in the outer `FirstBaseline` row (must not corrupt baseline!):
        let outer_empty = tree.alloc_node(LayoutSpec::col());

        tree.set_children(
            root,
            &[outer_empty, direct_txt, deep_col, deep_wrap, deep_grid],
        );
        tree.compute_layout(900.0, 600.0);

        let screen_bl = |t: &LayoutTree, nid: NodeId| -> f32 {
            let b = t.boxes[nid.idx()];
            b.y + b.baseline
        };
        let target_bl = screen_bl(&tree, direct_txt);
        assert!(
            (screen_bl(&tree, col_txt) - target_bl).abs() < 1e-4,
            "deep_col text baseline ({}) != direct_txt baseline ({target_bl})",
            screen_bl(&tree, col_txt)
        );
        assert!(
            (screen_bl(&tree, wrap_txt1) - target_bl).abs() < 1e-4,
            "deep_wrap text baseline ({}) != direct_txt baseline ({target_bl})",
            screen_bl(&tree, wrap_txt1)
        );
        assert!(
            (screen_bl(&tree, grid_txt) - target_bl).abs() < 1e-4,
            "deep_grid text baseline ({}) != direct_txt baseline ({target_bl})",
            screen_bl(&tree, grid_txt)
        );
    }

    #[test]
    fn virtual_list_150k_non_dyadic_mutations_zero_drift_and_overscan() {
        // Mix wide-dynamic-range non-dyadic `f32` heights (`0.003`, `33.3`, `87.7`, `987.65`)
        // across 4,096 items over 150,000 mutations so naive uncompensated `f64` Fenwick trees drift > 1e-5,
        // while our 106-bit `two_sum` Fenwick maintains < 1e-11 precision!
        let mut vlist = VirtualListIndex::new(4096, 33.3);
        let palette = [0.003_f32, 33.3, 87.7, 19.125, 0.0, 987.65, 41.9];
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut anchor_scroll = 50_000.0_f64;

        for step in 0..150_000usize {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = ((seed >> 32) as usize) % vlist.len();
            let h = palette[((seed >> 16) as usize + step) % palette.len()];
            if step % 17 == 0 {
                vlist.set_height_anchored(idx, h, 2048, &mut anchor_scroll);
            } else {
                vlist.set_height(idx, h);
            }
            if step == 50_000 {
                vlist.push(73.15);
                vlist.push(0.007);
                assert_eq!(vlist.pop(), Some(0.007));
            } else if step == 100_000 {
                vlist.resize(4200, 29.4);
                vlist.resize(4096, 0.0);
            }
        }

        // Exact 106-bit oracle check at EVERY prefix index `0..=4096`:
        let mut exact_s = 0.0_f64;
        let mut exact_e = 0.0_f64;
        let mut max_drift = 0.0_f64;
        for i in 0..=vlist.len() {
            let expected = exact_s + exact_e;
            let got = vlist.offset_of(i);
            let drift = (got - expected).abs();
            max_drift = max_drift.max(drift);
            assert!(
                drift < 1e-9,
                "Fenwick drift at index {i}: got {got}, expected {expected}, drift={drift:e}"
            );
            if i < vlist.len() {
                let (ns, ne) = two_sum(exact_s, vlist.height_at(i) as f64);
                exact_s = ns;
                exact_e += ne;
            }
        }
        assert!(max_drift < 1e-9);

        // Verify `visible_range_overscan`:
        let ov = VirtualListIndex::new(10, 20.0); // [0..200], 10 items of 20px each
        // 1. Zero overscan matches `visible_range` bit-for-bit:
        assert_eq!(
            ov.visible_range_overscan(45.0, 50.0, 0.0),
            ov.visible_range(45.0, 50.0)
        );
        // 2. Near top (`scroll_y = 10, viewport_h = 40, overscan = 30`):
        //    Window is `[max(10 - 30, 0), 10 + 40 + 30) = [0, 80)` -> items `0..4` (NOT `0..5` from `[0, 100)`!):
        let (st_top, en_top, off_top) = ov.visible_range_overscan(10.0, 40.0, 30.0);
        assert_eq!((st_top, en_top, off_top), (0, 4, 0.0));
        // 3. Mid-list (`scroll_y = 80, viewport_h = 40, overscan = 20`):
        //    Window `[60, 140)` -> items `3..7` (`[60..140)`), `first_offset == 60.0`:
        let (st_mid, en_mid, off_mid) = ov.visible_range_overscan(80.0, 40.0, 20.0);
        assert_eq!((st_mid, en_mid, off_mid), (3, 7, 60.0));
        assert_eq!(off_mid.to_bits(), ov.offset_of(st_mid).to_bits());
        // 4. Negative / NaN overscan sanitizes to `0.0`:
        assert_eq!(
            ov.visible_range_overscan(80.0, 40.0, -50.0),
            ov.visible_range(80.0, 40.0)
        );
        assert_eq!(
            ov.visible_range_overscan(80.0, 40.0, f64::NAN),
            ov.visible_range(80.0, 40.0)
        );
    }

    #[test]
    fn fractional_device_scale_100_rows_zero_physical_seams_and_relayout_boundary_stability() {
        let scales: &[f32] = &[
            1.0,
            1.125,
            1.25,
            160.0 / 120.0,
            1.375,
            1.5,
            1.625,
            200.0 / 120.0,
            1.75,
            2.0,
            2.25,
        ];
        // Include exact `-0.5` physical-pixel scroll tie (`0.4 * 1.25 == 0.5`, `0.375 * (4/3) == 0.5`)!
        let scroll_offsets: &[f32] = &[0.0, 0.4, 0.375, 13.7, 99.15];
        let fonts = FontSystem::new();

        for &sf in scales {
            // 1. 100 contiguous `gap == 0.0` stacked rows (mixing fractional Fixed heights, Grow, and text leaves):
            for &scroll_y in scroll_offsets {
                let mut tree = LayoutTree::new();
                tree.snap_to_pixels = true;
                tree.set_device_scale(sf);

                let mut root_spec = LayoutSpec::col().fixed(420.0, 600.0).gap(0.0);
                root_spec.scroll_y = scroll_y;
                let root = tree.alloc_node(root_spec);
                tree.topo[root.idx()].flags |= FLAG_CLIPS_CHILDREN | FLAG_SCROLLABLE;
                tree.root = root;

                let mut rows = Vec::with_capacity(100);
                for i in 0..100usize {
                    let h = match i % 5 {
                        0 => 23.0,
                        1 => 19.35,
                        2 => 28.0,
                        3 => 31.7,
                        _ => 24.5,
                    };
                    let row = tree.alloc_node(LayoutSpec::row().fixed(420.0, h).gap(0.0));
                    let c0 = tree.alloc_node(LayoutSpec::col().width_grow(1.0).height_grow(1.0));
                    let c1 = tree.alloc_node(LayoutSpec::col().width_grow(2.0).height_grow(1.0));
                    let c2 = tree.alloc_node(LayoutSpec::col().width_fixed(73.4).height_grow(1.0));
                    if i == 2 {
                        tree.attach_shaped_text(c0, "Row 2 initial", FontId::FALLBACK, 13.0, 18.0, &fonts);
                    }
                    tree.set_children(row, &[c0, c1, c2]);
                    rows.push((row, [c0, c1, c2]));
                }
                let row_ids: Vec<NodeId> = rows.iter().map(|&(r, _)| r).collect();
                tree.set_children(root, &row_ids);
                tree.compute_layout(420.0, 600.0);

                for i in 0..100usize {
                    let rb = tree.boxes[rows[i].0.idx()];
                    let py0 = rb.y * sf;
                    let py1 = (rb.y + rb.h) * sf;
                    let px0 = rb.x * sf;
                    let px1 = (rb.x + rb.w) * sf;
                    // At coordinates `> 1024.0`, `1 ULP` in `f32` is `1.22e-4` (`2^-13`), so bound by `max(1e-4, 2 ULP)`:
                    let tol_y0 = 1e-4_f32.max(py0.abs() * 2.5e-7);
                    let tol_y1 = 1e-4_f32.max(py1.abs() * 2.5e-7);
                    assert!(
                        (py0 - py0.round()).abs() <= tol_y0 && (py1 - py1.round()).abs() <= tol_y1,
                        "Row {i} Y edges not on integer physical px at sf={sf}, scroll={scroll_y}: [{py0}, {py1}]"
                    );
                    assert!(
                        (px0 - px0.round()).abs() < 1e-4 && (px1 - px1.round()).abs() < 1e-4,
                        "Row {i} X edges not on integer physical px at sf={sf}: [{px0}, {px1}]"
                    );

                    // Contiguous `Row` columns inside each row must also have 0.0 physical seam:
                    let cols = rows[i].1;
                    for pair in cols.windows(2) {
                        let a = tree.boxes[pair[0].idx()];
                        let b = tree.boxes[pair[1].idx()];
                        let end_a = ((a.x + a.w) * sf).round();
                        let start_b = (b.x * sf).round();
                        assert_eq!(
                            end_a, start_b,
                            "Horizontal seam in row {i} at sf={sf}: {end_a} vs {start_b}"
                        );
                        assert!(((a.x + a.w) * sf - b.x * sf).abs() < 1e-4);
                    }

                    if i + 1 < 100 {
                        let next_rb = tree.boxes[rows[i + 1].0.idx()];
                        let prev_end = ((rb.y + rb.h) * sf).round();
                        let next_start = (next_rb.y * sf).round();
                        assert_eq!(
                            prev_end, next_start,
                            "Vertical physical seam between row {i} and {} at sf={sf}, scroll={scroll_y}: {prev_end} != {next_start}",
                            i + 1
                        );
                        let edge_tol = 1e-4_f32.max(prev_end.abs() * 2.5e-7);
                        assert!(
                            ((rb.y + rb.h) * sf - next_rb.y * sf).abs() <= edge_tol,
                            "Unrounded physical edge gap at row {i} (sf={sf}): {} vs {}",
                            (rb.y + rb.h) * sf,
                            next_rb.y * sf
                        );
                    }
                }

                // Single-row incremental `FLAG_RELAYOUT_BOUNDARY` update on `row[2]` MUST preserve
                // its error-diffused height without opening a 1px overlap or gap against `row[3]`!
                tree.attach_shaped_text(
                    rows[2].1[0],
                    "Row 2 updated incrementally inside boundary",
                    FontId::FALLBACK,
                    13.0,
                    18.0,
                    &fonts,
                );
                tree.compute_layout(420.0, 600.0);
                assert_eq!(tree.last_metrics.boundaries_solved, 1);
                assert!(tree.last_metrics.nodes_visited <= 4);
                let r1 = tree.boxes[rows[1].0.idx()];
                let r2 = tree.boxes[rows[2].0.idx()];
                let r3 = tree.boxes[rows[3].0.idx()];
                assert_eq!(((r1.y + r1.h) * sf).round(), (r2.y * sf).round());
                assert_eq!(
                    ((r2.y + r2.h) * sf).round(),
                    (r3.y * sf).round(),
                    "Single-row relayout boundary clobbered error-diffused height at sf={sf}!"
                );
            }

            // 2. Contiguous `gap == 0.0` 2D `Grid` (`4` Fr cols x `10` Fr rows) zero-seam check:
            let mut gtree = LayoutTree::new();
            gtree.snap_to_pixels = true;
            gtree.set_device_scale(sf);
            let mut gspec = LayoutSpec::col().fixed(333.0, 277.0).gap(0.0);
            gspec.scroll_y = 0.4;
            let grid = gtree.alloc_node(gspec);
            gtree.attach_grid(grid, vec![TrackSize::Fr(1.0); 4], vec![TrackSize::Fr(1.0); 10]);
            gtree.root = grid;
            let mut cells = Vec::with_capacity(40);
            for _ in 0..40 {
                cells.push(gtree.alloc_node(LayoutSpec::col().grow(1.0)));
            }
            gtree.set_children(grid, &cells);
            gtree.compute_layout(333.0, 277.0);
            for r in 0..10 {
                for c in 0..4 {
                    let cb = gtree.boxes[cells[r * 4 + c].idx()];
                    if c + 1 < 4 {
                        let right = gtree.boxes[cells[r * 4 + c + 1].idx()];
                        assert_eq!(((cb.x + cb.w) * sf).round(), (right.x * sf).round());
                    }
                    if r + 1 < 10 {
                        let below = gtree.boxes[cells[(r + 1) * 4 + c].idx()];
                        assert_eq!(((cb.y + cb.h) * sf).round(), (below.y * sf).round());
                    }
                }
            }
        }
    }

    #[test]
    fn fractional_device_scale_scroll_only_fast_path_bit_identity_and_invalidation() {
        let fonts = FontSystem::new();
        let scales: &[f32] = &[
            1.0,
            1.125,
            1.25,
            160.0 / 120.0,
            1.375,
            1.5,
            1.625,
            200.0 / 120.0,
            1.75,
            2.25,
        ];
        let scroll_steps: &[(f32, f32, f32, f32)] = &[
            (0.4, 0.0, 0.0, 0.0),
            (13.7, 9.3, 4.2, 2.8),
            (-420.6, 25.8, 17.4, 11.9),
            (1_000_000.0, 500_000.0, 250_000.0, 125_000.0),
            (27.35, 4.15, 19.85, 6.45),
            (0.0, 0.0, 0.0, 0.0),
        ];

        // 1. Verify `device_scale == 1.0` produces 100% bit-identical boxes to default `LayoutTree`:
        let (default_1x, _, _, _, _) =
            build_complex_scroll_oracle_scene(true, 13.7, 9.3, 4.2, 2.8, &fonts);
        let mut explicit_1x = default_1x_clone_via_scale(1.0, 13.7, 9.3, 4.2, 2.8, &fonts);
        assert_trees_bit_identical(&explicit_1x, &default_1x, "device_scale == 1.0 parity");

        // Verify steady-frame `set_device_scale(same)` does 0 work, while changing scale invalidates cleanly:
        explicit_1x.set_device_scale(1.0);
        explicit_1x.compute_layout(840.0, 620.0);
        assert_eq!(explicit_1x.last_metrics.nodes_visited, 0);

        explicit_1x.set_device_scale(1.25);
        assert_ne!(explicit_1x.topo[explicit_1x.root.idx()].flags & FLAG_DIRTY_LAYOUT, 0);
        explicit_1x.compute_layout(840.0, 620.0);
        assert!(explicit_1x.last_metrics.nodes_visited > 0);
        explicit_1x.compute_layout(840.0, 620.0);
        assert_eq!(explicit_1x.last_metrics.nodes_visited, 0);

        // Malformed scale inputs (`NaN`, `0.0`, `-3.0`, `INFINITY`) sanitize cleanly to `1.0`:
        for bad in [f32::NAN, 0.0, -3.0, f32::INFINITY, f32::NEG_INFINITY] {
            explicit_1x.set_device_scale(bad);
            assert_eq!(explicit_1x.device_scale, 1.0);
        }

        // When `snap_to_pixels == false`, `set_device_scale` does NOT needlessly dirty layout:
        let (mut unsnapped, _, _, _, _) =
            build_complex_scroll_oracle_scene(false, 0.0, 0.0, 0.0, 0.0, &fonts);
        unsnapped.set_device_scale(1.5);
        assert_eq!(unsnapped.topo[unsnapped.root.idx()].flags & FLAG_DIRTY_LAYOUT, 0);
        unsnapped.compute_layout(840.0, 620.0);
        assert_eq!(unsnapped.last_metrics.nodes_visited, 0);

        // 2. Verify `FLAG_DIRTY_SCROLL` is `f32::to_bits()` bit-identical to full from-scratch solve
        //    at EVERY fractional `device_scale`!
        for &sf in scales {
            let (mut inc, outer_s, inner_s, wrap_s, grid_s) =
                build_fractional_scroll_scene(sf, 0.0, 0.0, 0.0, 0.0, &fonts);

            for (step_idx, &(os, is, ws, gs)) in scroll_steps.iter().enumerate() {
                if step_idx % 2 == 0 {
                    inc.set_scroll_y(inner_s, is);
                    inc.set_scroll_y(wrap_s, ws);
                    inc.set_scroll_y(grid_s, gs);
                    inc.set_scroll_y(outer_s, os);
                } else {
                    inc.set_scroll_y(outer_s, os);
                    inc.set_scroll_y(grid_s, gs);
                    inc.set_scroll_y(wrap_s, ws);
                    inc.set_scroll_y(inner_s, is);
                }
                assert_ne!(inc.topo[outer_s.idx()].flags & FLAG_DIRTY_SCROLL, 0);
                assert_eq!(inc.topo[outer_s.idx()].flags & FLAG_DIRTY_LAYOUT, 0);

                inc.compute_layout(840.0, 620.0);
                assert_eq!(inc.last_metrics.text_shaper_calls, 0);

                let (fresh, _, _, _, _) =
                    build_fractional_scroll_scene(sf, os, is, ws, gs, &fonts);
                assert_trees_bit_identical(
                    &inc,
                    &fresh,
                    &format!("sf={sf} step={step_idx} ({os}, {is}, {ws}, {gs})"),
                );
            }
        }
    }

    fn default_1x_clone_via_scale(
        sf: f32,
        os: f32,
        is: f32,
        ws: f32,
        gs: f32,
        fonts: &FontSystem,
    ) -> LayoutTree {
        build_fractional_scroll_scene(sf, os, is, ws, gs, fonts).0
    }

    fn build_fractional_scroll_scene(
        sf: f32,
        outer_scroll_y: f32,
        inner_scroll_y: f32,
        wrap_scroll_y: f32,
        grid_scroll_y: f32,
        fonts: &FontSystem,
    ) -> (LayoutTree, NodeId, NodeId, NodeId, NodeId) {
        let (mut tree, o, i, w, g) = build_complex_scroll_oracle_scene(
            true,
            outer_scroll_y,
            inner_scroll_y,
            wrap_scroll_y,
            grid_scroll_y,
            fonts,
        );
        tree.set_device_scale(sf);
        tree.compute_layout(840.0, 620.0);
        (tree, o, i, w, g)
    }

    #[test]
    fn fit_buttons_badges_and_labels_never_spuriously_wrap_under_pixel_snapping() {
        let fonts = FontSystem::with_system_fonts();
        let scales: &[f32] = &[
            1.0,
            1.125,
            1.25,
            160.0 / 120.0,
            1.375,
            1.5,
            1.625,
            200.0 / 120.0,
            1.75,
            2.0,
            2.25,
        ];
        let labels = [
            "Close Inspector",
            "v1.0.0 STABLE",
            "1.25× Auto",
            "Skima Studio",
            "OKLCH 256°",
            "OS Fonts: 4",
            "Auto (150/120)",
        ];

        for &sf in scales {
            for &fid in &[FontId::FALLBACK, fonts.ui_font, fonts.mono_font] {
                let mut tree = LayoutTree::new();
                tree.snap_to_pixels = true;
                tree.set_device_scale(sf);

                let root = tree.alloc_node(LayoutSpec::col().fixed(1280.0, 800.0).pad(16.0).gap(8.0));
                tree.root = root;

                // Header `Row` mixing `Fit` labels, `Grow` spacer, and nested `Fit` buttons/badges with fractional padding:
                let header = tree.alloc_node(LayoutSpec::row().width_grow(1.0).gap(8.0));
                let wrap_bar = tree.alloc_node(LayoutSpec::wrap_row().width_grow(1.0).gap(6.0));
                let mut text_nodes = Vec::new();
                let mut header_kids = Vec::new();
                let mut wrap_kids = Vec::new();

                for (i, &txt) in labels.iter().enumerate() {
                    let pad_x = 8.0 + (i as f32) * 1.35;
                    let fs = 12.0 + (i % 4) as f32 * 1.25;
                    let lh = (fs * 1.35).ceil();

                    let btn = tree.alloc_node(LayoutSpec::row().pad_xy(pad_x, 5.0));
                    let lbl = tree.alloc_node(LayoutSpec::col());
                    tree.attach_shaped_text(lbl, txt, fid, fs, lh, &fonts);
                    tree.set_children(btn, &[lbl]);
                    text_nodes.push((txt, lbl));

                    if i % 2 == 0 {
                        header_kids.push(btn);
                        if i == 0 {
                            header_kids.push(tree.alloc_node(LayoutSpec::row().width_grow(1.0)));
                        }
                    } else {
                        wrap_kids.push(btn);
                    }
                }

                // Also verify genuinely constrained text (`max_w = 70.0`) STILL wraps onto >= 2 lines!
                let constrained = tree.alloc_node(LayoutSpec::col().max_size(70.0, 400.0));
                tree.attach_shaped_text(
                    constrained,
                    "Genuinely constrained narrow paragraph must still wrap across multiple lines cleanly",
                    fid,
                    14.0,
                    20.0,
                    &fonts,
                );

                tree.set_children(header, &header_kids);
                tree.set_children(wrap_bar, &wrap_kids);
                tree.set_children(root, &[header, wrap_bar, constrained]);
                tree.compute_layout(1280.0, 800.0);

                for &(txt, nid) in &text_nodes {
                    let tid = tree.specs[nid.idx()].aux_id as usize;
                    let line_cnt = tree.texts[tid].cached_lines.len();
                    assert_eq!(
                        line_cnt, 1,
                        "Unshrunk Fit label {txt:?} spuriously wrapped into {line_cnt} lines at sf={sf}, fid={fid:?} (w={}, fit_w={}, max_uw={})",
                        tree.boxes[nid.idx()].w,
                        tree.boxes[nid.idx()].fit_w,
                        tree.texts[tid].max_unwrapped_w,
                    );
                }

                let c_tid = tree.specs[constrained.idx()].aux_id as usize;
                assert!(
                    tree.texts[c_tid].cached_lines.len() >= 2,
                    "Constrained max_w=70 paragraph must wrap onto multiple lines at sf={sf}"
                );
            }
        }
    }
}
