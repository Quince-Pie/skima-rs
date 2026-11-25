//! Callsite-Keyed Retained `SlotTape`, Two-Lifetime `Ui<'sys, 's, S>` Cursor,
//! Perceptual OKLCH `Theme`, Analytical `Spring` Hooks, HiDPI Physical Scaling,
//! Spatial Hit-Testing, Pointer Capture, Keyboard Focus Ring, Typed Actions,
//! Incremental `GpuPrimitive` / `DamageRect` Emission, and Composable 1.0 Controls.

use crate::a11y::{
    A11yNodeId, A11yRole, Action, ActionData, ActionRequest, SemanticsMeta, TreeId, TreeUpdate,
    a11y_id_to_slot, apply_root_scale_transform, slot_to_a11y_id,
};
use crate::arena::FrameArena;
use crate::color::Color;
use crate::gpu::{
    CLIP_NONE, DamageRect, DamageTracker, GpuClip, GpuPrimitive, PrimKind, RenderBackend,
    RenderFrame, snap_physical_border_width,
};
use crate::layout::{
    Align, AnchorPlacement, ComputedBox, FLAG_CLIPS_CHILDREN, FLAG_DIRTY_PAINT,
    FLAG_FLOATING_ANCHOR, FLAG_HAS_TEXT, LayoutSpec, LayoutTree, NodeId, TrackSize,
    VirtualListIndex,
};
use crate::pool::{Key, ScopeId, SlotPool, Spring};
use crate::scale::{DisplayMetrics, ResolvedScale, ScalePolicy, zoom_step};
use crate::text::{AntialiasMode, AtlasPlane, FontId, FontSystem};
use crate::vector::{FillRule, Path2D, RasterScratch};
use core::fmt::NumBuffer;
use core::mem::DropGuard;
use core::ops::RangeInclusive;
use core::panic::Location;
use core::range::Range as CopyRange;
use std::any::{Any, TypeId};

/// Platform-agnostic mouse cursor icon requested by the hovered or captured UI element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CursorIcon {
    #[default]
    Default,
    Pointer,
    Text,
    Grab,
    Grabbing,
    EwResize,
    NsResize,
}

/// Perceptually uniform OKLCH design-token theme (`Theme::dark()`, `Theme::light()`, or live hue-shifted).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theme {
    pub is_dark: bool,
    pub accent_hue: f32,
    pub bg_canvas: Color,
    pub bg_surface: Color,
    pub bg_elevated: Color,
    pub bg_control: Color,
    pub bg_control_hover: Color,
    pub accent: Color,
    pub accent_hover: Color,
    pub accent_muted: Color,
    pub text_primary: Color,
    pub text_secondary: Color,
    pub text_muted: Color,
    pub text_on_accent: Color,
    pub border_subtle: Color,
    pub border_focus: Color,
    pub success: Color,
    pub warning: Color,
    pub danger: Color,
    pub radius_sm: f32,
    pub radius_md: f32,
    pub radius_lg: f32,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Crisp high-contrast OKLCH dark theme (default).
    pub fn dark() -> Self {
        Self::with_accent_hue(true, 258.0)
    }

    /// Clean perceptual OKLCH light theme.
    pub fn light() -> Self {
        Self::with_accent_hue(false, 258.0)
    }

    /// Generate a cohesive dark or light design palette around any OKLCH `accent_hue` (`0..360` deg).
    pub fn with_accent_hue(is_dark: bool, accent_hue: f32) -> Self {
        let hue = if accent_hue.is_finite() {
            accent_hue.rem_euclid(360.0)
        } else {
            258.0
        };
        if is_dark {
            let accent = Color::oklch(0.64, 0.19, hue, 1.0);
            let on_accent = if OklabLuma::of(accent) > 0.70 {
                Color::rgb8(16, 18, 26)
            } else {
                Color::WHITE
            };
            Self {
                is_dark: true,
                accent_hue: hue,
                bg_canvas: Color::oklch(0.165, 0.014, hue, 1.0),
                bg_surface: Color::oklch(0.21, 0.017, hue, 1.0),
                bg_elevated: Color::oklch(0.255, 0.021, hue, 1.0),
                bg_control: Color::oklch(0.265, 0.019, hue, 1.0),
                bg_control_hover: Color::oklch(0.315, 0.024, hue, 1.0),
                accent,
                accent_hover: accent.shift_lightness(0.06),
                accent_muted: Color::oklch(0.32, 0.08, hue, 1.0),
                text_primary: Color::rgb8(244, 246, 252),
                text_secondary: Color::rgb8(184, 190, 208),
                text_muted: Color::rgb8(134, 141, 162),
                text_on_accent: on_accent,
                border_subtle: Color::oklch(0.33, 0.020, hue, 1.0),
                border_focus: Color::oklch(0.74, 0.18, hue, 1.0),
                success: Color::oklch(0.69, 0.17, 148.0, 1.0),
                warning: Color::oklch(0.77, 0.16, 78.0, 1.0),
                danger: Color::oklch(0.64, 0.21, 26.0, 1.0),
                radius_sm: 5.0,
                radius_md: 8.0,
                radius_lg: 12.0,
            }
        } else {
            let accent = Color::oklch(0.54, 0.20, hue, 1.0);
            let on_accent = if OklabLuma::of(accent) > 0.70 {
                Color::rgb8(16, 18, 26)
            } else {
                Color::WHITE
            };
            Self {
                is_dark: false,
                accent_hue: hue,
                bg_canvas: Color::oklch(0.955, 0.006, hue, 1.0),
                bg_surface: Color::WHITE,
                bg_elevated: Color::oklch(0.985, 0.004, hue, 1.0),
                bg_control: Color::oklch(0.925, 0.010, hue, 1.0),
                bg_control_hover: Color::oklch(0.885, 0.015, hue, 1.0),
                accent,
                accent_hover: accent.shift_lightness(-0.06),
                accent_muted: Color::oklch(0.88, 0.055, hue, 1.0),
                text_primary: Color::rgb8(18, 21, 30),
                text_secondary: Color::rgb8(62, 68, 86),
                text_muted: Color::rgb8(102, 109, 128),
                text_on_accent: on_accent,
                border_subtle: Color::oklch(0.83, 0.012, hue, 1.0),
                border_focus: Color::oklch(0.56, 0.20, hue, 1.0),
                success: Color::oklch(0.55, 0.17, 148.0, 1.0),
                warning: Color::oklch(0.60, 0.16, 72.0, 1.0),
                danger: Color::oklch(0.54, 0.21, 26.0, 1.0),
                radius_sm: 5.0,
                radius_md: 8.0,
                radius_lg: 12.0,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowSpec {
    pub color: Color,
    pub blur_sigma: f32,
    pub offset: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientSpec {
    pub end_color: Color,
    pub angle_rad: f32,
}

/// Complete visual styling for any UI node (4-corner independent radii, OKLab gradients,
/// analytical Gaussian shadows, anti-aliased borders, and opacity).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintStyle {
    pub bg: Color,
    pub gradient: Option<GradientSpec>,
    pub border_color: Color,
    pub border_width: f32,
    pub radii: [f32; 4],
    pub shadow: Option<ShadowSpec>,
    pub opacity: f32,
}

impl Default for PaintStyle {
    #[inline]
    fn default() -> Self {
        Self {
            bg: Color::TRANSPARENT,
            gradient: None,
            border_color: Color::TRANSPARENT,
            border_width: 0.0,
            radii: [0.0; 4],
            shadow: None,
            opacity: 1.0,
        }
    }
}

impl PaintStyle {
    #[inline]
    pub const fn solid(bg: Color) -> Self {
        Self {
            bg,
            gradient: None,
            border_color: Color::TRANSPARENT,
            border_width: 0.0,
            radii: [0.0; 4],
            shadow: None,
            opacity: 1.0,
        }
    }

    #[inline]
    pub const fn rounded(mut self, r: f32) -> Self {
        self.radii = [r, r, r, r];
        self
    }

    #[inline]
    pub const fn corners(mut self, tl: f32, tr: f32, br: f32, bl: f32) -> Self {
        self.radii = [tl, tr, br, bl];
        self
    }

    #[inline]
    pub const fn border(mut self, width: f32, color: Color) -> Self {
        self.border_width = width;
        self.border_color = color;
        self
    }

    #[inline]
    pub const fn drop_shadow(mut self, blur_sigma: f32, offset: [f32; 2], color: Color) -> Self {
        self.shadow = Some(ShadowSpec {
            color,
            blur_sigma,
            offset,
        });
        self
    }

    #[inline]
    pub const fn oklab_gradient(start: Color, end: Color, angle_rad: f32) -> Self {
        Self {
            bg: start,
            gradient: Some(GradientSpec {
                end_color: end,
                angle_rad,
            }),
            border_color: Color::TRANSPARENT,
            border_width: 0.0,
            radii: [0.0; 4],
            shadow: None,
            opacity: 1.0,
        }
    }

    #[inline]
    pub const fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PointerEvent {
    /// Window-space logical X coordinate.
    pub x: f32,
    /// Window-space logical Y coordinate.
    pub y: f32,
    /// Element-local X coordinate (`x - element_box.x`), computed live from current layout.
    pub local_x: f32,
    /// Element-local Y coordinate (`y - element_box.y`), computed live from current layout.
    pub local_y: f32,
    /// Current logical width of the target element.
    pub target_w: f32,
    /// Current logical height of the target element.
    pub target_h: f32,
    pub delta_x: f32,
    pub delta_y: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyInput {
    Tab { shift: bool },
    Enter,
    Space,
    Escape,
    Backspace,
    CtrlBackspace,
    Delete,
    CtrlDelete,
    ArrowLeft,
    ShiftArrowLeft,
    CtrlArrowLeft,
    CtrlShiftArrowLeft,
    ArrowRight,
    ShiftArrowRight,
    CtrlArrowRight,
    CtrlShiftArrowRight,
    ArrowUp,
    ArrowDown,
    PageUp,
    PageDown,
    Home,
    ShiftHome,
    End,
    ShiftEnd,
    SelectAll,
    Copy,
    Cut,
    Paste(String),
    Text(String),
}

/// Rust 1.87.0 `+ use<...>` precise lifetime capturing in traits:
/// Stateless components borrow `'data` frame data without overcapturing `&'ui mut Ui`.
pub trait StatelessView<M = !> {
    fn mount<'ui, 's, 'data>(
        &'data self,
        ui: &'ui mut Ui<'_, 's, M>,
    ) -> impl Copy + use<'data, Self, M>;
}

type ErasedProjectFn = unsafe fn(*mut (), usize) -> Option<*mut ()>;

#[derive(Clone, Copy)]
struct LensStep {
    project: ErasedProjectFn,
    aux: usize,
}

type ErasedPointerFn = unsafe fn(
    *mut (),
    u32,
    &mut [Option<Box<dyn Any>>],
    &FontSystem,
    &LayoutTree,
    &PointerEvent,
);
type ErasedKeyFn =
    unsafe fn(*mut (), u32, &mut [Option<Box<dyn Any>>], &mut String, &KeyInput) -> bool;
type ErasedScrollFn = unsafe fn(*mut (), u32, &mut [Option<Box<dyn Any>>], f32) -> bool;
type ErasedActionFn = unsafe fn(*mut (), usize, &dyn Any) -> bool;

const BIND_CLICK: u8 = 1 << 0;
const BIND_DRAG: u8 = 1 << 1;
const BIND_KEY: u8 = 1 << 2;
const BIND_SCROLL: u8 = 1 << 3;
const BIND_ACTION: u8 = 1 << 4;

/// High bit in `SlotKey::user_key` distinguishing explicit `ui.keyed(...)` identity (`1 << 63`)
/// from automatic per-parent unkeyed callsite occurrence ordinals (`0 .. (1 << 63)`).
const USER_KEY_EXPLICIT_BIT: u64 = 1 << 63;

#[inline]
fn sync_lens_steps(dst: &mut Box<[LensStep]>, src: &[LensStep]) {
    if dst.len() == src.len() {
        dst.copy_from_slice(src);
    } else {
        *dst = src.into();
    }
}

struct PointerBinding {
    steps: Box<[LensStep]>,
    handler_slot: u32,
    invoke: ErasedPointerFn,
}

struct KeyBinding {
    steps: Box<[LensStep]>,
    handler_slot: u32,
    invoke: ErasedKeyFn,
}

struct ScrollBinding {
    steps: Box<[LensStep]>,
    handler_slot: u32,
    invoke: ErasedScrollFn,
}

struct ActionBinding {
    action_type: TypeId,
    steps: Box<[LensStep]>,
    handler_addr: usize,
    invoke: ErasedActionFn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SlotKey {
    callsite_addr: usize,
    user_key: u64,
}

impl SlotKey {
    #[inline]
    const fn is_explicitly_keyed(self) -> bool {
        (self.user_key & USER_KEY_EXPLICIT_BIT) != 0
    }
}

struct RetainedSlot {
    key: SlotKey,
    parent: Option<u32>,
    node_id: NodeId,
    paint: PaintStyle,
    committed_paint: PaintStyle,
    custom_dirty: bool,
    active_bindings: u8,
    text_color: Color,
    aa_mode: AntialiasMode,
    prim_range: CopyRange<u32>,
    custom_atlas_region: Option<crate::text::AtlasRegion>,
    custom_atlas_gen: u64,
    custom_upload_tag: u64,
    children: Vec<u32>,
    local_states: Vec<u32>,
    on_click: Option<PointerBinding>,
    on_drag: Option<PointerBinding>,
    on_key: Option<KeyBinding>,
    on_scroll: Option<ScrollBinding>,
    actions: Vec<ActionBinding>,
    scrollable_max_y: Option<f32>,
    focusable: bool,
    pointer_passthrough: bool,
    cursor: Option<CursorIcon>,
    semantics: Option<SemanticsMeta>,
    scope_id: Option<ScopeId>,
    scope_epoch: u64,
    unsub_hooks: Vec<Box<dyn FnOnce()>>,
}

#[derive(Clone, Copy)]
struct ScrollbarThumbSync {
    track_node: NodeId,
    thumb_node: NodeId,
    scroll_y: f64,
    viewport_h_override: Option<f64>,
    total_content_h: f64,
    pad: f32,
    min_thumb_h: f32,
}

#[derive(Clone, Copy, Default)]
struct ScrollbarDragState {
    grab_local_y: f32,
    grab_scroll_y: f64,
    skip_initial_down_drag: bool,
}

/// Retained UTF-8 caret, selection anchor, and horizontal scroll state for [`Ui::text_input`].
#[derive(Clone, Copy, Debug)]
pub struct TextEditState {
    /// Active caret byte offset (`0..=text.len()`).
    pub cursor: usize,
    /// Selection anchor byte offset (`anchor == cursor` when collapsed; `min..max` when selected).
    pub anchor: usize,
    /// Horizontal scroll offset in logical pixels when text exceeds the input viewport.
    pub scroll_x: f32,
    last_click_time: Option<std::time::Instant>,
    last_click_x: f32,
    last_was_double_click: bool,
    skip_initial_down_drag: bool,
    cached_text_node: NodeId,
}

impl Default for TextEditState {
    fn default() -> Self {
        Self::new_at(usize::MAX)
    }
}

impl TextEditState {
    #[inline]
    pub const fn new_at(byte_offset: usize) -> Self {
        Self {
            cursor: byte_offset,
            anchor: byte_offset,
            scroll_x: 0.0,
            last_click_time: None,
            last_click_x: -1e9,
            last_was_double_click: false,
            skip_initial_down_drag: false,
            cached_text_node: NodeId::NONE,
        }
    }

    /// Return the normalized UTF-8 byte range `start..end` (`start <= end`) clamped to valid
    /// character boundaries of `s`.
    #[inline]
    pub fn selection_range(&self, s: &str) -> core::ops::Range<usize> {
        let c = s.floor_char_boundary(self.cursor.min(s.len()));
        let a = s.floor_char_boundary(self.anchor.min(s.len()));
        if c <= a { c..a } else { a..c }
    }

    /// Returns `true` when a non-empty UTF-8 selection is active (`anchor != cursor` on `s`).
    #[inline]
    pub fn has_selection(&self, s: &str) -> bool {
        let r = self.selection_range(s);
        r.start < r.end
    }
}

#[derive(Clone, Copy)]
struct TextInputSync {
    input_node: NodeId,
    sel_node: NodeId,
    text_node: NodeId,
    caret_node: NodeId,
    state_arena_idx: u32,
    pad_x: f32,
    caret_x: f32,
    sel_x0: f32,
    sel_x1: f32,
    has_selection: bool,
    total_text_w: f32,
    is_empty: bool,
    focused: bool,
}

#[derive(Clone, Copy, Default)]
struct ScrollAreaMetrics {
    initialized: bool,
    content_h: f64,
    viewport_h: f64,
}

#[derive(Clone, Copy)]
struct ScrollAreaSync {
    viewport_node: NodeId,
    content_node: NodeId,
    scrollbar_slot: u32,
    metrics_arena_idx: u32,
    raw_scroll_y: f64,
    assumed_overflow: bool,
    applied_scroll_y: f32,
}

#[inline]
fn prev_word_boundary(s: &str, pos: usize) -> usize {
    let clamped = s.floor_char_boundary(pos.min(s.len()));
    if clamped == 0 {
        return 0;
    }
    let mut chars = s[..clamped].char_indices().rev().peekable();
    while let Some(&(_, ch)) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
    let Some((first_idx, first_ch)) = chars.next() else {
        return 0;
    };
    let mut start = first_idx;
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let want_word = is_word(first_ch);
    for (idx, ch) in chars {
        if ch.is_whitespace() || (is_word(ch) != want_word) {
            break;
        }
        start = idx;
    }
    start
}

#[inline]
fn next_word_boundary(s: &str, pos: usize) -> usize {
    let clamped = s.floor_char_boundary(pos.min(s.len()));
    if clamped >= s.len() {
        return s.len();
    }
    let tail = &s[clamped..];
    let mut chars = tail.char_indices().peekable();
    while let Some(&(_, ch)) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
    let Some((_, first_ch)) = chars.next() else {
        return s.len();
    };
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let want_word = is_word(first_ch);
    for (rel_idx, ch) in chars {
        if ch.is_whitespace() || (is_word(ch) != want_word) {
            return clamped + rel_idx;
        }
    }
    s.len()
}

#[inline]
fn word_range_at(s: &str, pos: usize) -> (usize, usize) {
    if s.is_empty() {
        return (0, 0);
    }
    let clamped = s.floor_char_boundary(pos.min(s.len()));
    let classify = |c: char| -> u8 {
        if c.is_whitespace() {
            0
        } else if c.is_alphanumeric() || c == '_' {
            1
        } else {
            2
        }
    };
    let (pivot_start, pivot_ch) = if clamped < s.len() {
        let ch = s[clamped..].chars().next().unwrap();
        (clamped, ch)
    } else {
        s.char_indices().next_back().unwrap()
    };
    let target_class = classify(pivot_ch);
    let mut start = pivot_start;
    for (idx, ch) in s[..pivot_start].char_indices().rev() {
        if classify(ch) == target_class {
            start = idx;
        } else {
            break;
        }
    }
    let pivot_end = pivot_start + pivot_ch.len_utf8();
    let mut end = pivot_end;
    for (rel_idx, ch) in s[pivot_end..].char_indices() {
        if classify(ch) == target_class {
            end = pivot_end + rel_idx + ch.len_utf8();
        } else {
            break;
        }
    }
    (start, end)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn hit_test_text_input_x(
    s: &str,
    text_node: NodeId,
    rel_x: f32,
    fid: FontId,
    font_size: f32,
    line_h: f32,
    fonts: &FontSystem,
    layout: &LayoutTree,
) -> usize {
    if !text_node.is_none()
        && let Some(topo) = layout.topo.get(text_node.idx())
        && (topo.flags & crate::layout::FLAG_ALIVE) != 0
        && (topo.flags & FLAG_HAS_TEXT) != 0
    {
        let tid = layout.specs[text_node.idx()].aux_id as usize;
        if let Some(para) = layout.texts.get(tid) {
            let want_hash = crate::text::hash_text_style(s, fid, font_size, line_h);
            if para.content_hash == want_hash && para.font_id == fid {
                return para.x_to_byte_offset(s, rel_x);
            }
        }
    }
    fonts.x_to_byte_offset(s, rel_x, fid, font_size)
}

#[inline]
fn live_atlas_uv(reg: crate::text::AtlasRegion, atlas_w: u16, atlas_h: u16) -> [f32; 4] {
    let inv_w = 1.0 / (atlas_w.max(1) as f32);
    let inv_h = 1.0 / (atlas_h.max(1) as f32);
    [
        (reg.x as f32) * inv_w,
        (reg.y as f32) * inv_h,
        ((reg.x as u32 + reg.w as u32) as f32) * inv_w,
        ((reg.y as u32 + reg.h as u32) as f32) * inv_h,
    ]
}

/// Snap a logical coordinate `v` onto the physical device pixel grid (`0.5 ULP` `f32` precision via `f64`).
#[inline]
fn snap_logical_to_device(v: f32, sf: f32) -> f32 {
    let s = if sf == 1.0 || !sf.is_finite() || sf <= 0.0 {
        v.round()
    } else {
        (((v as f64) * (sf as f64)).round() / (sf as f64)) as f32
    };
    if s == 0.0 { 0.0 } else { s }
}

/// Snap a non-zero logical extent `v` onto an integer physical device pixel count (`>= 1` physical px).
#[inline]
fn snap_extent_to_device_min1(v: f32, sf: f32) -> f32 {
    if sf == 1.0 || !sf.is_finite() || sf <= 0.0 {
        v.round().max(1.0)
    } else {
        (((v as f64) * (sf as f64)).round().max(1.0) / (sf as f64)) as f32
    }
}

/// Convert a logical [`ComputedBox`] into physical `[x, y, w, h]` pixels.
///
/// When `snap_to_pixels` is active, rounds the outer physical edges `x0 = (b.x * sf).round()` and
/// `x1 = ((b.x + b.w) * sf).round()` directly so non-dyadic scales (`4/3 = 1.3333334`) never land
/// `1 ULP` (`1e-6` px) off an integer physical pixel and contiguous `gap == 0` siblings share
/// `0.000000` physical-pixel boundaries.
#[inline]
fn physical_box_bounds(b: ComputedBox, sf: f32, snap_to_pixels: bool) -> [f32; 4] {
    if snap_to_pixels {
        let x0 = (b.x * sf).round();
        let y0 = (b.y * sf).round();
        let x1 = ((b.x + b.w) * sf).round();
        let y1 = ((b.y + b.h) * sf).round();
        [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)]
    } else {
        [b.x * sf, b.y * sf, (b.w * sf).max(0.0), (b.h * sf).max(0.0)]
    }
}

#[inline]
fn compute_vscrollbar_metrics(
    track_h: f32,
    viewport_h_override: Option<f64>,
    total_content_h: f64,
    scroll_y: f64,
    pad: f32,
    min_thumb_h: f32,
) -> (f64, f32, f32, f32, f32) {
    let viewport_h = viewport_h_override.unwrap_or(track_h as f64).max(1.0);
    let rail_h = (track_h - 2.0 * pad).max(1.0);
    let max_scroll = (total_content_h - viewport_h).max(0.0);
    if max_scroll <= 0.0 {
        return (0.0, rail_h, rail_h, 0.0, 0.0);
    }
    let ratio = (viewport_h / total_content_h.max(1.0)).clamp(0.0, 1.0);
    let eff_min_thumb = min_thumb_h.min(rail_h);
    let thumb_h = ((rail_h as f64 * ratio) as f32).clamp(eff_min_thumb, rail_h);
    let travel_h = (rail_h - thumb_h).max(0.0);
    let frac = (scroll_y / max_scroll).clamp(0.0, 1.0);
    let thumb_top = (frac * travel_h as f64) as f32;
    (max_scroll, rail_h, thumb_h, travel_h, thumb_top)
}

/// Complete Retained UI Engine:
/// Owns `LayoutTree`, `FontSystem`, `FrameArena`, `Theme`, retained `SlotTape`, `GpuPrimitive` buffer,
/// `DamageTracker`, focus/capture/hover state, analytical `Spring` clock, and `AccessKit` semantics.
pub struct GuiSystem {
    pub layout: LayoutTree,
    pub fonts: FontSystem,
    pub frame_arena: FrameArena,
    pub theme: Theme,
    pub clear_color: Color,
    pub scale_factor: f32,
    pub scale_policy: ScalePolicy,
    pub resolved_scale: Option<ResolvedScale>,
    pub display_metrics: Option<DisplayMetrics>,
    prev_scale_factor: f32,
    slots: Vec<Option<RetainedSlot>>,
    slot_pool_stash: Vec<RetainedSlot>,
    free_slots: Vec<u32>,
    state_arena: Vec<Option<Box<dyn Any>>>,
    free_states: Vec<u32>,
    active_springs: Vec<u32>,
    root_slot: Option<u32>,
    node_to_slot: Vec<u32>,
    pub primitives: Vec<GpuPrimitive>,
    pub clips: Vec<GpuClip>,
    pub dirty_prim_ranges: Vec<CopyRange<u32>>,
    dirty_paint_slots: Vec<u32>,
    pub damage_tracker: DamageTracker,
    pub current_frame_damage: Option<DamageRect>,
    scene_topology_changed: bool,
    dirty_scopes: Vec<bool>,
    next_scope_id: u32,
    // Interaction & Accessibility state:
    pub focused_slot: Option<u32>,
    pub captured_slot: Option<u32>,
    pub hovered_slot: Option<u32>,
    hovered_chain: Vec<u32>,
    pub active_cursor: CursorIcon,
    last_pointer_pos: [f32; 2],
    has_pointer: bool,
    pub a11y_enabled: bool,
    a11y_initialized: bool,
    dirty_a11y_slots: Vec<u32>,
    clipboard: String,
    // Reusable zero-alloc scratch buffers:
    lens_scratch: Vec<LensStep>,
    callsite_counts_scratch: Vec<(usize, u32)>,
    node_id_scratch: Vec<NodeId>,
    floating_slot_scratch: Vec<u32>,
    uncommitted_allocs: Vec<u32>,
    scrollbar_thumbs: Vec<ScrollbarThumbSync>,
    text_inputs: Vec<TextInputSync>,
    scroll_areas: Vec<ScrollAreaSync>,
    vector_path_scratch: Path2D,
    vector_raster_scratch: RasterScratch,
    vector_mask_scratch: Vec<u8>,
    current_logical_size: [f32; 2],
    prev_logical_size: [f32; 2],
    pub paint_updates_last_frame: u64,
    pub scopes_skipped_last_frame: u64,
}

impl Default for GuiSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl GuiSystem {
    /// Construct a lightweight `GuiSystem` (uses built-in crisp procedural font until `.ttf` or system fonts are loaded).
    pub fn new() -> Self {
        let theme = Theme::dark();
        let mut layout = LayoutTree::new();
        layout.snap_to_pixels = true;
        layout.set_device_scale(1.0);
        Self {
            layout,
            fonts: FontSystem::new(),
            frame_arena: FrameArena::with_capacity(256 * 1024),
            theme,
            clear_color: theme.bg_canvas,
            scale_factor: 1.0,
            scale_policy: ScalePolicy::Auto,
            resolved_scale: None,
            display_metrics: None,
            prev_scale_factor: 1.0,
            slots: Vec::new(),
            slot_pool_stash: Vec::new(),
            free_slots: Vec::new(),
            state_arena: Vec::new(),
            free_states: Vec::new(),
            active_springs: Vec::new(),
            root_slot: None,
            node_to_slot: Vec::new(),
            primitives: Vec::new(),
            clips: Vec::new(),
            dirty_prim_ranges: Vec::new(),
            dirty_paint_slots: Vec::new(),
            damage_tracker: DamageTracker::new(),
            current_frame_damage: None,
            scene_topology_changed: true,
            dirty_scopes: Vec::new(),
            next_scope_id: 0,
            focused_slot: None,
            captured_slot: None,
            hovered_slot: None,
            hovered_chain: Vec::with_capacity(32),
            active_cursor: CursorIcon::Default,
            last_pointer_pos: [0.0, 0.0],
            has_pointer: false,
            a11y_enabled: false,
            a11y_initialized: false,
            dirty_a11y_slots: Vec::new(),
            clipboard: String::new(),
            lens_scratch: Vec::with_capacity(16),
            callsite_counts_scratch: Vec::with_capacity(64),
            node_id_scratch: Vec::with_capacity(64),
            floating_slot_scratch: Vec::with_capacity(8),
            uncommitted_allocs: Vec::with_capacity(64),
            scrollbar_thumbs: Vec::with_capacity(4),
            text_inputs: Vec::with_capacity(4),
            scroll_areas: Vec::with_capacity(4),
            vector_path_scratch: Path2D::new(),
            vector_raster_scratch: RasterScratch::new(),
            vector_mask_scratch: Vec::new(),
            current_logical_size: [0.0, 0.0],
            prev_logical_size: [0.0, 0.0],
            paint_updates_last_frame: 0,
            scopes_skipped_last_frame: 0,
        }
    }

    /// Read the current text stored in `GuiSystem`'s clipboard buffer (populated by `KeyInput::Copy` / `KeyInput::Cut`
    /// or `set_clipboard_text`).
    #[inline]
    pub fn clipboard_text(&self) -> &str {
        &self.clipboard
    }

    /// Set the text stored in `GuiSystem`'s clipboard buffer.
    #[inline]
    pub fn set_clipboard_text(&mut self, text: impl Into<String>) {
        self.clipboard = text.into();
    }

    /// Paste the current `GuiSystem::clipboard_text()` into the focused widget.
    pub fn paste_clipboard<S: 'static>(&mut self, root_state: &mut S) -> bool {
        let clip = self.clipboard.clone();
        self.dispatch_key(&KeyInput::Paste(clip), root_state)
    }

    /// Inspect the retained [`TextEditState`] (`cursor`, `anchor`, `scroll_x`) of the currently focused `text_input`, if any.
    pub fn focused_text_edit_state(&self) -> Option<TextEditState> {
        let f_slot = self.focused_slot?;
        let slot = self.slots.get(f_slot as usize)?.as_ref()?;
        for &st_idx in &slot.local_states {
            if let Some(Some(any_box)) = self.state_arena.get(st_idx as usize)
                && let Some(st) = any_box.downcast_ref::<TextEditState>()
            {
                return Some(*st);
            }
        }
        None
    }

    /// Construct a `GuiSystem` pre-loaded with the host OS Sans-Serif, Monospace, and Emoji OpenType fonts.
    pub fn with_system_fonts() -> Self {
        let mut sys = Self::new();
        let _ = sys.fonts.load_system_fonts();
        sys
    }

    /// Update HiDPI display scale factor (`1.0` = 96 DPI 1:1, `1.25`/`1.5` = fractional HiDPI, `2.0` = Retina / 200% Wayland).
    /// Synchronizes `LayoutTree::set_device_scale` and automatically re-emits physical-pixel SDF & ClearType primitives on change!
    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        let sf = if scale_factor.is_finite() && scale_factor > 0.0 {
            scale_factor.clamp(0.5, 8.0)
        } else {
            1.0
        };
        self.layout.set_device_scale(sf);
        if (sf - self.scale_factor).abs() > 1e-6 {
            self.scale_factor = sf;
            self.prev_scale_factor = sf;
            self.scene_topology_changed = true;
            if self.a11y_enabled
                && self.a11y_initialized
                && let Some(rs) = self.root_slot
                && !self.dirty_a11y_slots.contains(&rs)
            {
                self.dirty_a11y_slots.push(rs);
            }
            if let Some(prev_res) = self.resolved_scale
                && (prev_res.scale_factor - sf).abs() > 1e-5
            {
                let base_metrics = self.display_metrics.unwrap_or_else(|| {
                    let [lw, lh] = self.current_logical_size;
                    DisplayMetrics::new(
                        (lw * sf).round().max(1.0) as u32,
                        (lh * sf).round().max(1.0) as u32,
                    )
                });
                self.resolved_scale = Some(ScalePolicy::Fixed(sf).resolve(&base_metrics));
            }
        }
    }

    /// Resolve the active fractional scale factor and display-physics telemetry from [`DisplayMetrics`]
    /// under `self.scale_policy`, storing [`Self::resolved_scale`] and syncing [`Self::set_scale_factor`].
    pub fn resolve_display_scale(&mut self, mut metrics: DisplayMetrics) -> ResolvedScale {
        if metrics.previous_120ths.is_none() {
            metrics.previous_120ths = self.resolved_scale.map(|r| r.wayland_120ths);
        }
        self.display_metrics = Some(metrics);
        let resolved = self.scale_policy.resolve(&metrics);
        self.resolved_scale = Some(resolved);
        self.set_scale_factor(resolved.scale_factor);
        resolved
    }

    /// Change the active [`ScalePolicy`] (`Auto`, `PurePhysicalAuto`, `System`, or `Fixed`)
    /// and immediately re-resolve against the latest [`DisplayMetrics`] (or headless viewport).
    pub fn set_scale_policy(&mut self, policy: ScalePolicy) -> ResolvedScale {
        self.scale_policy = policy;
        let metrics = self.display_metrics.unwrap_or_else(|| {
            let [lw, lh] = self.current_logical_size;
            let pw = if lw > 0.0 {
                (lw * self.scale_factor).round().max(1.0) as u32
            } else {
                1280
            };
            let ph = if lh > 0.0 {
                (lh * self.scale_factor).round().max(1.0) as u32
            } else {
                800
            };
            DisplayMetrics::new(pw, ph)
        });
        let resolved = policy.resolve(&metrics);
        self.resolved_scale = Some(resolved);
        self.set_scale_factor(resolved.scale_factor);
        resolved
    }

    /// Step cleanly along the Wayland `1/120` scale lattice (`+1` = zoom in, `-1` = zoom out),
    /// switching `self.scale_policy` to `ScalePolicy::Fixed(next_scale)`.
    pub fn zoom_scale(&mut self, delta_steps: i32) -> ResolvedScale {
        let next = zoom_step(self.scale_factor, delta_steps);
        self.set_scale_policy(ScalePolicy::Fixed(next))
    }

    /// Reset interactive zoom back to [`ScalePolicy::Auto`] (`Ctrl+0`).
    pub fn reset_scale_auto(&mut self) -> ResolvedScale {
        self.set_scale_policy(ScalePolicy::Auto)
    }

    #[inline]
    pub fn has_active_animations(&self) -> bool {
        !self.active_springs.is_empty()
    }

    /// Advance all active `Spring` animations analytically by `dt_sec` seconds.
    /// Returns `true` if any spring is still moving (requiring another VSync frame), or `false` once settled.
    pub fn step_animations(&mut self, dt_sec: f32) -> bool {
        if self.active_springs.is_empty() {
            return false;
        }
        let mut any_moving = false;
        let state_arena = &mut self.state_arena;
        self.active_springs.retain(|&arena_idx| {
            let Some(Some(any_box)) = state_arena.get_mut(arena_idx as usize) else {
                return false;
            };
            let Some(sp) = any_box.downcast_mut::<Spring>() else {
                return false;
            };
            if !dt_sec.is_finite() || dt_sec <= 0.0 {
                let moving = !sp.is_settled(1e-4);
                if moving {
                    any_moving = true;
                }
                return moving;
            }
            // Guard extreme large `dt_sec` (e.g. `1e38`) from `Inf * 0.0 = NaN` in raw `f32`:
            let z = sp.damping_ratio.max(0.0);
            let w = sp.angular_freq.max(0.1);
            let slow_rate = if z <= 1.0 {
                z * w
            } else {
                w / (z + (z * z - 1.0).sqrt())
            };
            if slow_rate > 0.0 && slow_rate * dt_sec > 88.0 {
                sp.value = sp.target;
                sp.velocity = 0.0;
                return false;
            }
            let moving = sp.step(dt_sec);
            if !sp.value.is_finite() || !sp.velocity.is_finite() {
                sp.value = sp.target;
                sp.velocity = 0.0;
                return false;
            }
            if moving {
                any_moving = true;
            }
            moving
        });
        any_moving
    }

    pub fn alloc_scope(&mut self) -> ScopeId {
        let id = ScopeId(self.next_scope_id);
        self.next_scope_id += 1;
        self.dirty_scopes.push(true);
        id
    }

    #[inline]
    pub fn mark_scope_dirty(&mut self, scope: ScopeId) {
        if let Some(flag) = self.dirty_scopes.get_mut(scope.0 as usize) {
            *flag = true;
        }
    }

    /// Reconcile the UI against `state: &'s S` in logical coordinates `(logical_w, logical_h)`,
    /// solve dirty `IALE` relayout boundaries, and emit physical-pixel `GpuPrimitive`s & `DamageRect`s!
    pub fn build_frame<'s, S: 'static>(
        &mut self,
        state: &'s S,
        logical_w: f32,
        logical_h: f32,
        build: impl FnOnce(&mut Ui<'_, 's, S>),
    ) {
        self.frame_arena.reset();
        self.paint_updates_last_frame = 0;
        self.scopes_skipped_last_frame = 0;
        self.lens_scratch.clear();
        self.callsite_counts_scratch.clear();
        self.uncommitted_allocs.clear();
        self.scrollbar_thumbs.clear();
        self.text_inputs.clear();
        self.scroll_areas.clear();
        self.dirty_prim_ranges.clear();
        self.current_frame_damage = None;
        self.prev_logical_size = self.current_logical_size;
        self.current_logical_size = [logical_w, logical_h];

        // Sanitize `scale_factor`, detect direct field mutations, and synchronize `LayoutTree::device_scale`:
        let clean_sf = if self.scale_factor.is_finite() && self.scale_factor > 0.0 {
            self.scale_factor.clamp(0.5, 8.0)
        } else {
            1.0
        };
        self.scale_factor = clean_sf;
        if (clean_sf - self.prev_scale_factor).abs() > 1e-6 {
            self.prev_scale_factor = clean_sf;
            self.scene_topology_changed = true;
            if self.a11y_enabled
                && self.a11y_initialized
                && let Some(rs) = self.root_slot
                && !self.dirty_a11y_slots.contains(&rs)
            {
                self.dirty_a11y_slots.push(rs);
            }
        }
        self.layout.set_device_scale(clean_sf);

        // Keep `resolved_scale` populated in headless sessions where `resolve_display_scale` was not called:
        if self.display_metrics.is_none() {
            let needs_headless_resolve = match self.resolved_scale {
                None => true,
                Some(rs) => {
                    (rs.scale_factor - clean_sf).abs() > 1e-5
                        || (logical_w.is_finite() && (rs.logical_width - logical_w).abs() > 0.5)
                        || (logical_h.is_finite() && (rs.logical_height - logical_h).abs() > 0.5)
                }
            };
            if needs_headless_resolve {
                let pw = if logical_w.is_finite() && logical_w > 0.0 {
                    (logical_w * clean_sf - 1e-4).ceil().max(1.0) as u32
                } else {
                    0
                };
                let ph = if logical_h.is_finite() && logical_h > 0.0 {
                    (logical_h * clean_sf - 1e-4).ceil().max(1.0) as u32
                } else {
                    0
                };
                let mut rs = ScalePolicy::Fixed(clean_sf).resolve(&DisplayMetrics::new(pw, ph));
                if logical_w.is_finite() && logical_w >= 0.0 {
                    rs.logical_width = logical_w;
                }
                if logical_h.is_finite() && logical_h >= 0.0 {
                    rs.logical_height = logical_h;
                }
                self.resolved_scale = Some(rs);
            }
        }

        {
            let mut ui = Ui {
                sys: self,
                state_ptr: (state as *const S).cast::<()>(),
                parent_slot: None,
                cursor_pos: 0,
                callsite_scope_base: 0,
                structure_changed: false,
                pending_user_key: 0,
                active_slot: None,
                state_cursor: 0,
                _marker: core::marker::PhantomData,
            };
            build(&mut ui);
        }

        if let Some(rs) = self.root_slot {
            self.layout.root = self.slots[rs as usize].as_ref().unwrap().node_id;
        }

        self.layout.compute_layout(logical_w, logical_h);
        let areas_changed = self.sync_scroll_areas(logical_w, logical_h);
        let thumbs_changed = self.sync_scrollbar_thumbs();
        let inputs_changed = self.sync_text_inputs();

        // Refresh hover chain & active cursor icon against post-layout boxes so stationary
        // pointers immediately reflect scrolled, animated, or newly toggled elements!
        if self.has_pointer {
            let _ = self.refresh_pointer_hover();
        }

        let sf = self.scale_factor;
        let phys_w = (logical_w * sf - 1e-4).ceil().max(1.0) as u32;
        let phys_h = (logical_h * sf - 1e-4).ceil().max(1.0) as u32;

        if self.scene_topology_changed {
            self.rebuild_ordered_gpu_scene(phys_w, phys_h);
            self.scene_topology_changed = false;
            self.dirty_paint_slots.clear();
        } else if self.layout.last_metrics.boundaries_solved > 0
            || !self.layout.solved_nodes_last_frame.is_empty()
            || areas_changed
            || thumbs_changed
            || inputs_changed
        {
            self.sync_solved_layout_nodes_gpu(phys_w, phys_h);
        } else if !self.dirty_paint_slots.is_empty() {
            for i in 0..self.dirty_paint_slots.len() {
                let s_idx = self.dirty_paint_slots[i];
                self.patch_slot_paint_in_place(s_idx, phys_w, phys_h);
            }
            self.dirty_paint_slots.clear();
        }

        self.coalesce_dirty_prim_ranges();
    }

    fn sync_scroll_areas(&mut self, logical_w: f32, logical_h: f32) -> bool {
        if self.scroll_areas.is_empty() {
            return false;
        }
        let mut need_relayout = false;
        for i in 0..self.scroll_areas.len() {
            let sa = self.scroll_areas[i];
            let view_h = self.layout.boxes[sa.viewport_node.idx()].h.max(1.0) as f64;
            let content_h = self.layout.boxes[sa.content_node.idx()].fit_h.max(0.0) as f64;

            if let Some(Some(any_box)) = self.state_arena.get_mut(sa.metrics_arena_idx as usize)
                && let Some(m) = any_box.downcast_mut::<ScrollAreaMetrics>()
            {
                m.initialized = true;
                m.content_h = content_h;
                m.viewport_h = view_h;
            }

            let actual_overflow = content_h > view_h + 0.5;
            let max_scroll = (content_h - view_h).max(0.0);
            let clamped_scroll = sa.raw_scroll_y.clamp(0.0, max_scroll) as f32;

            let sb_node = self
                .slots
                .get(sa.scrollbar_slot as usize)
                .and_then(|s| s.as_ref())
                .map(|s| s.node_id);

            if let Some(sb_nid) = sb_node {
                for thumb_sync in &mut self.scrollbar_thumbs {
                    if thumb_sync.track_node == sb_nid {
                        thumb_sync.total_content_h = content_h;
                        thumb_sync.viewport_h_override = Some(view_h);
                        thumb_sync.scroll_y = clamped_scroll as f64;
                    }
                }
            }

            if actual_overflow != sa.assumed_overflow
                || (clamped_scroll - sa.applied_scroll_y).abs() > 0.5
            {
                let mut v_spec = self.layout.specs[sa.viewport_node.idx()];
                v_spec.scroll_y = clamped_scroll;
                self.layout.update_spec(sa.viewport_node, v_spec);

                if let Some(sb_nid) = sb_node {
                    let mut sb_spec = self.layout.specs[sb_nid.idx()];
                    sb_spec.val_w = if actual_overflow { 14.0 } else { 0.0 };
                    self.layout.update_spec(sb_nid, sb_spec);

                    let sb_captured_or_hovered = self.captured_slot == Some(sa.scrollbar_slot)
                        || self.hovered_chain.contains(&sa.scrollbar_slot);
                    let want_thumb_w = if !actual_overflow {
                        5.0
                    } else if sb_captured_or_hovered {
                        10.0
                    } else {
                        8.0
                    };
                    let thumb_node_opt = if let Some(sb_slot) = self
                        .slots
                        .get_mut(sa.scrollbar_slot as usize)
                        .and_then(|s| s.as_mut())
                    {
                        sb_slot.paint.opacity = if actual_overflow { 1.0 } else { 0.0 };
                        sb_slot.focusable = actual_overflow;
                        sb_slot.cursor = actual_overflow.then_some(CursorIcon::Grab);
                        let thumb_opacity = if actual_overflow { 1.0 } else { 0.0 };
                        let thumb_slot_opt = sb_slot.children.first().copied();
                        if let Some(ts_idx) = thumb_slot_opt
                            && let Some(ts) =
                                self.slots.get_mut(ts_idx as usize).and_then(|s| s.as_mut())
                        {
                            ts.paint.opacity = thumb_opacity;
                            ts.paint.radii = [want_thumb_w * 0.5; 4];
                            Some(ts.node_id)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(ts_nid) = thumb_node_opt {
                        let mut ts_spec = self.layout.specs[ts_nid.idx()];
                        ts_spec.val_w = want_thumb_w;
                        self.layout.update_spec(ts_nid, ts_spec);
                    }
                }
                self.scene_topology_changed = true;
                need_relayout = true;
            }
        }

        if need_relayout {
            self.layout.compute_layout(logical_w, logical_h);
            for i in 0..self.scroll_areas.len() {
                let sa = self.scroll_areas[i];
                let view_h = self.layout.boxes[sa.viewport_node.idx()].h.max(1.0) as f64;
                let content_h = self.layout.boxes[sa.content_node.idx()].fit_h.max(0.0) as f64;
                let max_scroll = (content_h - view_h).max(0.0);
                let clamped_scroll = sa.raw_scroll_y.clamp(0.0, max_scroll);
                if let Some(Some(any_box)) = self.state_arena.get_mut(sa.metrics_arena_idx as usize)
                    && let Some(m) = any_box.downcast_mut::<ScrollAreaMetrics>()
                {
                    m.content_h = content_h;
                    m.viewport_h = view_h;
                }
                if let Some(sb_nid) = self
                    .slots
                    .get(sa.scrollbar_slot as usize)
                    .and_then(|s| s.as_ref())
                    .map(|s| s.node_id)
                {
                    for thumb_sync in &mut self.scrollbar_thumbs {
                        if thumb_sync.track_node == sb_nid {
                            thumb_sync.total_content_h = content_h;
                            thumb_sync.viewport_h_override = Some(view_h);
                            thumb_sync.scroll_y = clamped_scroll;
                        }
                    }
                }
            }
        }
        need_relayout
    }

    fn sync_text_inputs(&mut self) -> bool {
        let mut changed = false;
        let snap = self.layout.snap_to_pixels;
        let sf = self.scale_factor;
        let min_sel_w = if sf > 0.0 && sf.is_finite() {
            (1.0 / sf).min(1.0)
        } else {
            1.0
        };
        let want_sel_h = if snap {
            snap_extent_to_device_min1(18.0, sf)
        } else {
            18.0
        };
        let want_caret_w = if snap {
            snap_extent_to_device_min1(2.0, sf)
        } else {
            2.0
        };
        let want_caret_h = if snap {
            snap_extent_to_device_min1(16.0, sf)
        } else {
            16.0
        };
        for i in 0..self.text_inputs.len() {
            let ti = self.text_inputs[i];
            let ib = self.layout.boxes[ti.input_node.idx()];
            let inner_x = ib.x + ti.pad_x;
            let inner_w = (ib.w - 2.0 * ti.pad_x).max(0.0);
            let usable_w = (inner_w - 2.0).max(0.0);

            let prev_scroll_x = self
                .state_arena
                .get(ti.state_arena_idx as usize)
                .and_then(|b| b.as_ref())
                .and_then(|b| b.downcast_ref::<TextEditState>())
                .map_or(0.0, |st| st.scroll_x);

            let max_scroll_x = if ti.is_empty {
                0.0
            } else {
                (ti.total_text_w - usable_w).max(0.0)
            };
            let mut scroll_x = prev_scroll_x.clamp(0.0, max_scroll_x);
            if ti.focused && !ti.is_empty {
                if ti.caret_x < scroll_x {
                    scroll_x = ti.caret_x;
                } else if ti.caret_x > scroll_x + usable_w {
                    scroll_x = ti.caret_x - usable_w;
                }
                scroll_x = scroll_x.clamp(0.0, max_scroll_x);
            } else if ti.is_empty {
                scroll_x = 0.0;
            }

            if let Some(Some(any_box)) = self.state_arena.get_mut(ti.state_arena_idx as usize)
                && let Some(st) = any_box.downcast_mut::<TextEditState>()
            {
                st.scroll_x = scroll_x;
                st.cached_text_node = ti.text_node;
            }

            // 1. Selection highlight quad behind text glyphs (`[sel_x0..sel_x1]` clamped inside focus border):
            let sb = self.layout.boxes[ti.sel_node.idx()];
            let clip_x0 = ib.x + 2.0;
            let clip_x1 = (ib.x + ib.w - 2.0).max(clip_x0);
            let raw_sx0 = (inner_x + ti.sel_x0 - scroll_x).clamp(clip_x0, clip_x1);
            let raw_sx1 = (inner_x + ti.sel_x1 - scroll_x).clamp(clip_x0, clip_x1);
            let raw_sy = ib.y + (ib.h - want_sel_h) * 0.5;
            let want_sy = if snap {
                snap_logical_to_device(raw_sy, sf)
            } else {
                raw_sy
            };
            let (want_sx, want_sw) =
                if ti.focused && !ti.is_empty && ti.has_selection && raw_sx1 > raw_sx0 + 0.1 {
                    let sx0 = if snap {
                        snap_logical_to_device(raw_sx0, sf)
                    } else {
                        raw_sx0
                    };
                    let sx1 = if snap {
                        snap_logical_to_device(raw_sx1, sf)
                    } else {
                        raw_sx1
                    };
                    (sx0, (sx1 - sx0).max(min_sel_w))
                } else {
                    let sx0 = if snap {
                        snap_logical_to_device(inner_x, sf)
                    } else {
                        inner_x
                    };
                    (sx0, 0.0)
                };
            if (sb.x - want_sx).abs() > 1e-4
                || (sb.y - want_sy).abs() > 1e-4
                || (sb.w - want_sw).abs() > 1e-4
                || (sb.h - want_sel_h).abs() > 1e-4
            {
                self.layout.boxes[ti.sel_node.idx()].x = want_sx;
                self.layout.boxes[ti.sel_node.idx()].y = want_sy;
                self.layout.boxes[ti.sel_node.idx()].w = want_sw;
                self.layout.boxes[ti.sel_node.idx()].h = want_sel_h;
                if !self.layout.solved_nodes_last_frame.contains(&ti.sel_node) {
                    self.layout.solved_nodes_last_frame.push(ti.sel_node);
                }
                changed = true;
            }

            // 2. Unbroken single-line shaped text run:
            let tb = self.layout.boxes[ti.text_node.idx()];
            let raw_tx = if ti.is_empty {
                inner_x
            } else {
                inner_x - scroll_x
            };
            let raw_ty = ib.y + (ib.h - tb.h) * 0.5;
            let want_tx = if snap {
                snap_logical_to_device(raw_tx, sf)
            } else {
                raw_tx
            };
            let want_ty = if snap {
                snap_logical_to_device(raw_ty, sf)
            } else {
                raw_ty
            };
            let want_tw = tb.fit_w.max(ti.total_text_w + 2.0);
            if (tb.x - want_tx).abs() > 1e-4
                || (tb.y - want_ty).abs() > 1e-4
                || (tb.w - want_tw).abs() > 1e-4
            {
                self.layout.boxes[ti.text_node.idx()].x = want_tx;
                self.layout.boxes[ti.text_node.idx()].y = want_ty;
                self.layout.boxes[ti.text_node.idx()].w = want_tw;
                if !self.layout.solved_nodes_last_frame.contains(&ti.text_node) {
                    self.layout.solved_nodes_last_frame.push(ti.text_node);
                }
                changed = true;
            }

            // 3. Thin 2x16 active caret overlay:
            let cb = self.layout.boxes[ti.caret_node.idx()];
            let raw_cx = inner_x + if ti.is_empty { 0.0 } else { ti.caret_x - scroll_x };
            let raw_cy = ib.y + (ib.h - want_caret_h) * 0.5;
            let want_cx = if snap {
                snap_logical_to_device(raw_cx, sf)
            } else {
                raw_cx
            };
            let want_cy = if snap {
                snap_logical_to_device(raw_cy, sf)
            } else {
                raw_cy
            };
            if (cb.x - want_cx).abs() > 1e-4
                || (cb.y - want_cy).abs() > 1e-4
                || (cb.w - want_caret_w).abs() > 1e-4
                || (cb.h - want_caret_h).abs() > 1e-4
            {
                self.layout.boxes[ti.caret_node.idx()].x = want_cx;
                self.layout.boxes[ti.caret_node.idx()].y = want_cy;
                self.layout.boxes[ti.caret_node.idx()].w = want_caret_w;
                self.layout.boxes[ti.caret_node.idx()].h = want_caret_h;
                if !self.layout.solved_nodes_last_frame.contains(&ti.caret_node) {
                    self.layout.solved_nodes_last_frame.push(ti.caret_node);
                }
                changed = true;
            }
        }
        changed
    }

    fn sync_scrollbar_thumbs(&mut self) -> bool {
        let mut changed = false;
        let snap = self.layout.snap_to_pixels;
        let sf = self.scale_factor;
        for i in 0..self.scrollbar_thumbs.len() {
            let s = self.scrollbar_thumbs[i];
            let tb = self.layout.boxes[s.track_node.idx()];
            let (_max_s, _rail_h, thumb_h, _travel_h, thumb_top) = compute_vscrollbar_metrics(
                tb.h,
                s.viewport_h_override,
                s.total_content_h,
                s.scroll_y,
                s.pad,
                s.min_thumb_h,
            );
            let raw_y = tb.y + s.pad + thumb_top;
            let new_y = if snap {
                snap_logical_to_device(raw_y, sf)
            } else {
                raw_y
            };
            let new_h = if snap {
                snap_extent_to_device_min1(thumb_h, sf)
            } else {
                thumb_h
            };
            self.layout.specs[s.thumb_node.idx()].val_h = thumb_h;
            let old = self.layout.boxes[s.thumb_node.idx()];
            if (old.y - new_y).abs() > 1e-4
                || (old.h - new_h).abs() > 1e-4
                || (old.fit_h - new_h).abs() > 1e-4
            {
                self.layout.boxes[s.thumb_node.idx()].y = new_y;
                self.layout.boxes[s.thumb_node.idx()].h = new_h;
                self.layout.boxes[s.thumb_node.idx()].fit_h = new_h;
                if !self.layout.solved_nodes_last_frame.contains(&s.thumb_node) {
                    self.layout.solved_nodes_last_frame.push(s.thumb_node);
                }
                changed = true;
            }
        }
        changed
    }

    /// Build a raw `RenderFrame` with the current frame's uncoalesced `DamageRect` (used with `VulkanSwapchain::present_frame`).
    #[inline]
    pub fn raw_render_frame(&self, physical_w: u32, physical_h: u32) -> RenderFrame<'_> {
        RenderFrame {
            width: physical_w,
            height: physical_h,
            scale_factor: self.scale_factor,
            clear_color: self.clear_color,
            primitives: &self.primitives,
            dirty_prim_ranges: &self.dirty_prim_ranges,
            clips: &self.clips,
            damage: self.current_frame_damage,
        }
    }

    /// Present the prepared frame to a `VulkanSwapchain` using disjoint field borrows on `GuiSystem`.
    pub fn present_vulkan_swapchain(
        &mut self,
        vk: &mut crate::gpu::vulkan::VulkanRenderer,
        sc: &mut crate::gpu::vulkan::VulkanSwapchain,
    ) -> Result<bool, crate::gpu::vulkan::VulkanError> {
        let frame = RenderFrame {
            width: sc.extent.width,
            height: sc.extent.height,
            scale_factor: self.scale_factor,
            clear_color: self.clear_color,
            primitives: &self.primitives,
            dirty_prim_ranges: &self.dirty_prim_ranges,
            clips: &self.clips,
            damage: self.current_frame_damage,
        };
        vk.present_frame(sc, &mut self.damage_tracker, &frame, &mut self.fonts.atlas)
    }

    /// Render the prepared frame offscreen on Vulkan 1.3 and read back physical RGBA8 pixels.
    pub fn render_vulkan_offscreen(
        &mut self,
        vk: &mut crate::gpu::vulkan::VulkanRenderer,
        physical_w: u32,
        physical_h: u32,
    ) -> Result<Vec<u8>, crate::gpu::vulkan::VulkanError> {
        let frame = RenderFrame {
            width: physical_w,
            height: physical_h,
            scale_factor: self.scale_factor,
            clear_color: self.clear_color,
            primitives: &self.primitives,
            dirty_prim_ranges: &self.dirty_prim_ranges,
            clips: &self.clips,
            damage: self.current_frame_damage,
        };
        vk.render_offscreen_rgba8(&frame, &mut self.fonts.atlas)
    }

    /// Submit the prepared frame to any `RenderBackend` (`VulkanRenderer`, `CpuRenderer`, or `wgpu` adapter)
    /// with swapchain `buffer_age` damage-rect scissoring!
    pub fn render_to_backend<B: RenderBackend>(
        &mut self,
        backend: &mut B,
        physical_width: u32,
        physical_height: u32,
        buffer_age: u8,
    ) -> Result<(), B::Error> {
        let damage = self.damage_tracker.resolve_frame_damage(
            self.current_frame_damage,
            physical_width,
            physical_height,
            buffer_age,
        );
        let frame = RenderFrame {
            width: physical_width,
            height: physical_height,
            scale_factor: self.scale_factor,
            clear_color: self.clear_color,
            primitives: &self.primitives,
            dirty_prim_ranges: &self.dirty_prim_ranges,
            clips: &self.clips,
            damage,
        };
        backend.render(&frame, &mut self.fonts.atlas)
    }

    fn rebuild_ordered_gpu_scene(&mut self, phys_w: u32, phys_h: u32) {
        self.primitives.clear();
        self.clips.clear();
        self.dirty_prim_ranges.clear();
        self.floating_slot_scratch.clear();
        let Some(root) = self.root_slot else {
            return;
        };
        let updates_base = self.paint_updates_last_frame;
        for _attempt in 0..4 {
            let gen_before = self.fonts.atlas.generation();
            self.primitives.clear();
            self.clips.clear();
            self.floating_slot_scratch.clear();
            self.paint_updates_last_frame = updates_base;

            self.emit_slot_prims_dfs(root, CLIP_NONE);
            // Emit Pass-4 floating anchored overlays (`anchored_popover` & `tooltip`) in the topmost
            // overlay layer with `CLIP_NONE` so they escape clipped containers and render above siblings:
            let mut f_i = 0usize;
            while f_i < self.floating_slot_scratch.len() {
                let fl_slot = self.floating_slot_scratch[f_i];
                self.emit_slot_prims_dfs(fl_slot, CLIP_NONE);
                f_i += 1;
            }
            if self.fonts.atlas.generation() == gen_before {
                break;
            }
        }
        let count = self.primitives.len() as u32;
        if count > 0 {
            self.dirty_prim_ranges.push(CopyRange {
                start: 0,
                end: count,
            });
        }
        self.current_frame_damage = Some(DamageRect {
            x: 0,
            y: 0,
            w: phys_w,
            h: phys_h,
        });
    }

    fn emit_slot_prims_dfs(&mut self, slot_idx: u32, parent_clip: u32) {
        let Some(slot_ref) = self.slots[slot_idx as usize].as_ref() else {
            return;
        };
        let sf = self.scale_factor;
        let nid = slot_ref.node_id;
        let n_idx = nid.idx();
        let b = self.layout.boxes[n_idx];
        let pb = physical_box_bounds(b, sf, self.layout.snap_to_pixels);
        let paint = slot_ref.paint;
        let pradii = paint.radii.map(|r| r * sf);
        let phys_bw = snap_physical_border_width(paint.border_width, sf);
        let text_color = slot_ref.text_color;
        let aa_mode = slot_ref.aa_mode;
        let custom_region = slot_ref.custom_atlas_region;
        let clips_children = (self.layout.topo[n_idx].flags & FLAG_CLIPS_CHILDREN) != 0;

        let start_p = self.primitives.len() as u32;

        // 1. Optional Analytical Drop Shadow
        if let Some(sh) = paint.shadow
            && sh.color.channels()[3] > 0
        {
            self.primitives.push(GpuPrimitive::shadow(
                pb,
                pradii,
                sh.color,
                sh.blur_sigma * sf,
                [sh.offset[0] * sf, sh.offset[1] * sf],
                parent_clip,
                paint.opacity,
            ));
        }

        // 2. Main Quad / OKLab Gradient / Border (reserve 1 primary quad slot for O(1) in-place patches;
        //    invisible quads are culled in the vertex shader with zero fragment overdraw!)
        let primary_prim = if let Some(grad) = paint.gradient {
            GpuPrimitive::linear_gradient(
                pb,
                pradii,
                paint.bg,
                grad.end_color,
                grad.angle_rad,
                phys_bw,
                parent_clip,
                paint.opacity,
            )
        } else {
            GpuPrimitive::quad(
                pb,
                pradii,
                paint.bg,
                phys_bw,
                paint.border_color,
                parent_clip,
                paint.opacity,
            )
        };
        self.primitives.push(primary_prim);

        // 3. Optional Vector Path / Custom Atlas Image (compute live UVs against current atlas dimensions
        //    so texel-preserving `GlyphAtlas::try_grow` later in the same frame never leaves stale UVs!)
        if let Some(reg) = custom_region
            && reg.w > 0
            && reg.h > 0
        {
            let kind = match reg.plane {
                Some(AtlasPlane::R8) => PrimKind::GlyphGray,
                _ => PrimKind::ImageRgba,
            };
            let uv = live_atlas_uv(reg, self.fonts.atlas.width, self.fonts.atlas.height);
            self.primitives.push(GpuPrimitive::atlas_quad(
                kind,
                pb,
                uv,
                text_color,
                parent_clip,
                paint.opacity,
            ));
        }

        // 4. Shaped Text Glyphs (per-glyph font fallback + physical HiDPI rasterization + snapped baseline!)
        if (self.layout.topo[n_idx].flags & crate::layout::FLAG_HAS_TEXT) != 0
            && text_color.channels()[3] > 0
        {
            let spec = self.layout.specs[n_idx];
            let tid = spec.aux_id as usize;
            let origin_x = b.x + spec.pad_l;
            let origin_y = b.y + spec.pad_t;
            let font_size = self.layout.texts[tid].font_size;
            let phys_font_size = font_size * sf;
            let space_w = self.layout.texts[tid].space_w;
            let n_lines = self.layout.texts[tid].cached_lines.len();

            for l_i in 0..n_lines {
                let line = self.layout.texts[tid].cached_lines[l_i];
                let mut pen_x = origin_x;
                // Snap the horizontal line baseline to an exact integer physical Y BEFORE subtracting
                // `sg.y_offset * sf` so every glyph on the line shares a rock-solid baseline with 0px step jitter!
                let phys_base_y = ((origin_y + line.baseline_y) * sf).round();
                for w_i in line.words.start..line.words.end {
                    if w_i > line.words.start {
                        pen_x += space_w;
                    }
                    let wr = self.layout.texts[tid].words[w_i as usize];
                    for g_i in wr.glyphs.start..wr.glyphs.end {
                        let sg = self.layout.texts[tid].glyphs[g_i as usize];
                        let phys_gx = (pen_x + sg.x_offset) * sf;
                        let phys_gy = phys_base_y - sg.y_offset * sf;
                        let reg = self.fonts.rasterize_glyph(
                            sg.font_id,
                            sg.glyph_id,
                            phys_font_size,
                            phys_gx,
                            aa_mode,
                        );
                        if reg.w > 0 && reg.h > 0 {
                            let emoji_tint =
                                Color::rgba8(255, 255, 255, text_color.channels()[3]);
                            let (kind, glyph_tint) = match reg.plane {
                                Some(AtlasPlane::Rgba8) if reg.is_color => {
                                    (PrimKind::ImageRgba, emoji_tint)
                                }
                                Some(AtlasPlane::Rgba8) if aa_mode == AntialiasMode::SubpixelLcd => {
                                    (PrimKind::GlyphLcd, text_color)
                                }
                                Some(AtlasPlane::Rgba8) => (PrimKind::ImageRgba, emoji_tint),
                                _ => (PrimKind::GlyphGray, text_color),
                            };
                            self.primitives.push(GpuPrimitive::atlas_quad(
                                kind,
                                [
                                    phys_gx.floor() + (reg.bearing_x as f32),
                                    (phys_gy - reg.bearing_y as f32).round(),
                                    reg.w as f32,
                                    reg.h as f32,
                                ],
                                [reg.u0, reg.v0, reg.u1, reg.v1],
                                glyph_tint,
                                parent_clip,
                                paint.opacity,
                            ));
                        }
                        pen_x += sg.x_advance;
                    }
                }
            }
        }

        let end_p = self.primitives.len() as u32;
        {
            let slot_mut = self.slots[slot_idx as usize].as_mut().unwrap();
            slot_mut.prim_range = CopyRange {
                start: start_p,
                end: end_p,
            };
            slot_mut.committed_paint = paint;
            slot_mut.custom_dirty = false;
        }
        self.layout.topo[n_idx].flags &= !FLAG_DIRTY_PAINT;
        self.paint_updates_last_frame += (end_p - start_p) as u64;

        let child_clip = if clips_children {
            let cid = self.clips.len() as u32;
            self.clips.push(GpuClip {
                bounds: pb,
                radii: pradii,
                meta: [parent_clip, 0, 0, 0],
            });
            cid
        } else {
            parent_clip
        };

        let child_len = self.slots[slot_idx as usize]
            .as_ref()
            .unwrap()
            .children
            .len();
        for c_i in 0..child_len {
            let ch_slot = self.slots[slot_idx as usize].as_ref().unwrap().children[c_i];
            if let Some(ch_ref) = self.slots[ch_slot as usize].as_ref() {
                if (self.layout.topo[ch_ref.node_id.idx()].flags & FLAG_FLOATING_ANCHOR) != 0 {
                    self.floating_slot_scratch.push(ch_slot);
                } else {
                    self.emit_slot_prims_dfs(ch_slot, child_clip);
                }
            }
        }
    }

    /// Sort and merge adjacent or overlapping `dirty_prim_ranges` in place (`0 B` heap allocation)
    /// so `VulkanRenderer` issues minimal contiguous ReBAR `memcpy` spans.
    fn coalesce_dirty_prim_ranges(&mut self) {
        let len = self.dirty_prim_ranges.len();
        if len == 0 {
            return;
        }
        let max_prim = self.primitives.len() as u32;
        if len == 1 {
            let r = &mut self.dirty_prim_ranges[0];
            r.start = r.start.min(max_prim);
            r.end = r.end.min(max_prim);
            if r.end <= r.start {
                self.dirty_prim_ranges.clear();
            }
            return;
        }

        self.dirty_prim_ranges.sort_unstable_by_key(|r| r.start);
        let mut write = 0usize;
        let mut have_valid = false;
        for read in 0..len {
            let mut r = self.dirty_prim_ranges[read];
            r.start = r.start.min(max_prim);
            r.end = r.end.min(max_prim);
            if r.end <= r.start {
                continue;
            }
            if !have_valid {
                self.dirty_prim_ranges[0] = r;
                write = 0;
                have_valid = true;
            } else {
                let cur = &mut self.dirty_prim_ranges[write];
                if r.start <= cur.end {
                    if r.end > cur.end {
                        cur.end = r.end;
                    }
                } else {
                    write += 1;
                    self.dirty_prim_ranges[write] = r;
                }
            }
        }
        if !have_valid {
            self.dirty_prim_ranges.clear();
        } else {
            self.dirty_prim_ranges.truncate(write + 1);
        }
    }

    fn sync_solved_layout_nodes_gpu(&mut self, phys_w: u32, phys_h: u32) {
        let sf = self.scale_factor;
        let snap = self.layout.snap_to_pixels;
        let mut need_full_rebuild = false;
        for i in 0..self.layout.solved_nodes_last_frame.len() {
            let nid = self.layout.solved_nodes_last_frame[i];
            let n_idx = nid.idx();
            let Some(&s_idx) = self.node_to_slot.get(n_idx) else {
                continue;
            };
            if s_idx == u32::MAX {
                continue;
            };
            let Some(slot) = self.slots[s_idx as usize].as_ref() else {
                continue;
            };
            if ((self.layout.topo[n_idx].flags & crate::layout::FLAG_HAS_TEXT) != 0
                && slot.text_color.channels()[3] > 0)
                || (self.layout.topo[n_idx].flags & FLAG_CLIPS_CHILDREN) != 0
            {
                need_full_rebuild = true;
                break;
            }
            let r = slot.prim_range;
            if r.end > r.start {
                let b = self.layout.boxes[n_idx];
                let pb = physical_box_bounds(b, sf, snap);
                for p_idx in (r.start as usize)..(r.end as usize) {
                    if p_idx < self.primitives.len() {
                        let old_fp = self.primitives[p_idx].screen_footprint();
                        self.primitives[p_idx].bounds = pb;
                        let new_fp = self.primitives[p_idx].screen_footprint();
                        self.record_damage_bounds(old_fp, phys_w, phys_h);
                        self.record_damage_bounds(new_fp, phys_w, phys_h);
                    }
                }
                self.dirty_prim_ranges.push(r);
                self.paint_updates_last_frame += 1;
            }
        }
        if need_full_rebuild {
            self.rebuild_ordered_gpu_scene(phys_w, phys_h);
        } else {
            for i in 0..self.dirty_paint_slots.len() {
                let s_idx = self.dirty_paint_slots[i];
                self.patch_slot_paint_in_place(s_idx, phys_w, phys_h);
            }
        }
        self.dirty_paint_slots.clear();
    }

    #[inline]
    fn patch_slot_paint_in_place(&mut self, slot_idx: u32, phys_w: u32, phys_h: u32) {
        let Some((nid, r, paint, committed_paint, custom_dirty, text_color, custom_region)) = self
            .slots
            .get(slot_idx as usize)
            .and_then(|s| s.as_ref())
            .map(|s| {
                (
                    s.node_id,
                    s.prim_range,
                    s.paint,
                    s.committed_paint,
                    s.custom_dirty,
                    s.text_color,
                    s.custom_atlas_region,
                )
            })
        else {
            return;
        };
        let n_idx = nid.idx();
        if r.end <= r.start {
            return;
        }
        if paint == committed_paint && !custom_dirty {
            self.layout.topo[n_idx].flags &= !FLAG_DIRTY_PAINT;
            return;
        }
        let has_prev_shadow = self.primitives[r.start as usize].kind() == PrimKind::Shadow;
        let wants_shadow = paint.shadow.is_some_and(|s| s.color.channels()[3] > 0);
        if has_prev_shadow != wants_shadow {
            self.rebuild_ordered_gpu_scene(phys_w, phys_h);
            return;
        }
        let sf = self.scale_factor;
        let b = self.layout.boxes[n_idx];
        let pb = physical_box_bounds(b, sf, self.layout.snap_to_pixels);
        let pradii = paint.radii.map(|rad| rad * sf);
        let phys_bw = snap_physical_border_width(paint.border_width, sf);

        let mut p_cursor = r.start as usize;
        if let Some(sh) = paint.shadow
            && wants_shadow
        {
            let old_fp = self.primitives[p_cursor].screen_footprint();
            let clip_id = self.primitives[p_cursor].packed[2] & CLIP_NONE;
            self.primitives[p_cursor] = GpuPrimitive::shadow(
                pb,
                pradii,
                sh.color,
                sh.blur_sigma * sf,
                [sh.offset[0] * sf, sh.offset[1] * sf],
                clip_id,
                paint.opacity,
            );
            let new_fp = self.primitives[p_cursor].screen_footprint();
            self.record_damage_bounds(old_fp, phys_w, phys_h);
            self.record_damage_bounds(new_fp, phys_w, phys_h);
            p_cursor += 1;
        }
        if p_cursor < (r.end as usize) {
            let old_fp = self.primitives[p_cursor].screen_footprint();
            let clip_id = self.primitives[p_cursor].packed[2] & CLIP_NONE;
            self.primitives[p_cursor] = if let Some(grad) = paint.gradient {
                GpuPrimitive::linear_gradient(
                    pb,
                    pradii,
                    paint.bg,
                    grad.end_color,
                    grad.angle_rad,
                    phys_bw,
                    clip_id,
                    paint.opacity,
                )
            } else {
                GpuPrimitive::quad(
                    pb,
                    pradii,
                    paint.bg,
                    phys_bw,
                    paint.border_color,
                    clip_id,
                    paint.opacity,
                )
            };
            let new_fp = self.primitives[p_cursor].screen_footprint();
            self.record_damage_bounds(old_fp, phys_w, phys_h);
            self.record_damage_bounds(new_fp, phys_w, phys_h);
            p_cursor += 1;
        }
        if p_cursor < (r.end as usize)
            && let Some(reg) = custom_region
            && reg.w > 0
            && reg.h > 0
        {
            let old_fp = self.primitives[p_cursor].screen_footprint();
            let clip_id = self.primitives[p_cursor].packed[2] & CLIP_NONE;
            let kind = match reg.plane {
                Some(AtlasPlane::R8) => PrimKind::GlyphGray,
                _ => PrimKind::ImageRgba,
            };
            let uv = live_atlas_uv(reg, self.fonts.atlas.width, self.fonts.atlas.height);
            self.primitives[p_cursor] = GpuPrimitive::atlas_quad(
                kind,
                pb,
                uv,
                text_color,
                clip_id,
                paint.opacity,
            );
            let new_fp = self.primitives[p_cursor].screen_footprint();
            self.record_damage_bounds(old_fp, phys_w, phys_h);
            self.record_damage_bounds(new_fp, phys_w, phys_h);
        }
        if let Some(slot_mut) = self.slots.get_mut(slot_idx as usize).and_then(|s| s.as_mut()) {
            slot_mut.committed_paint = paint;
            slot_mut.custom_dirty = false;
        }
        self.dirty_prim_ranges.push(r);
        self.layout.topo[n_idx].flags &= !FLAG_DIRTY_PAINT;
        self.paint_updates_last_frame += 1;
    }

    #[inline]
    fn record_damage_bounds(&mut self, fb: [f32; 4], phys_w: u32, phys_h: u32) {
        if let Some(dr) = DamageRect::from_float_bounds(fb, phys_w, phys_h) {
            self.current_frame_damage =
                Some(self.current_frame_damage.map_or(dr, |cur| cur.union(dr)));
        }
    }

    // =========================================================================
    // Spatial Hit-Testing, Pointer Capture, Hover Chain, Scroll, Keyboard Focus, Actions
    // =========================================================================

    pub fn hit_test(&self, x: f32, y: f32) -> Option<u32> {
        self.hit_test_filtered(x, y, |s| {
            s.on_click.is_some() || s.on_drag.is_some() || s.focusable
        })
    }

    fn hit_test_filtered(
        &self,
        x: f32,
        y: f32,
        pred: impl Fn(&RetainedSlot) -> bool + Copy,
    ) -> Option<u32> {
        let root_s = self.root_slot?;
        // 1. Topmost overlay z-layer first: check Pass-4 floating anchors in reverse mount order
        //    (matches `rebuild_ordered_gpu_scene` overlay z-order; passive `pointer_passthrough` tooltips skip cleanly):
        for anchor in self.layout.anchors.iter().rev() {
            let ov_nid = anchor.overlay_node;
            if !ov_nid.is_none()
                && (self.layout.topo[ov_nid.idx()].flags & crate::layout::FLAG_ALIVE) != 0
                && let Some(&ov_slot) = self.node_to_slot.get(ov_nid.idx())
                && ov_slot != u32::MAX
                && let Some(hit) = self.hit_test_rec(ov_slot, x, y, pred, true)
            {
                return Some(hit);
            }
        }
        // 2. Main non-floating UI tree:
        self.hit_test_rec(root_s, x, y, pred, false)
    }

    fn hit_test_rec(
        &self,
        slot_idx: u32,
        x: f32,
        y: f32,
        pred: impl Fn(&RetainedSlot) -> bool + Copy,
        allow_floating_self: bool,
    ) -> Option<u32> {
        let slot = self.slots.get(slot_idx as usize)?.as_ref()?;
        if slot.pointer_passthrough {
            return None;
        }
        let n_idx = slot.node_id.idx();
        let is_floating = (self.layout.topo[n_idx].flags & FLAG_FLOATING_ANCHOR) != 0;
        if is_floating && !allow_floating_self {
            return None;
        }
        let b = self.layout.boxes[n_idx];
        let clip = self.layout.clip_rects[n_idx];
        if !is_floating
            && (x < clip[0] || x > clip[0] + clip[2] || y < clip[1] || y > clip[1] + clip[3])
        {
            return None;
        }
        let inside_self = x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h;
        if (self.layout.topo[n_idx].flags & FLAG_CLIPS_CHILDREN) != 0 && !inside_self {
            return None;
        }
        for &ch_slot in slot.children.iter().rev() {
            if let Some(hit) = self.hit_test_rec(ch_slot, x, y, pred, false) {
                return Some(hit);
            }
        }
        if inside_self && pred(slot) {
            Some(slot_idx)
        } else {
            None
        }
    }

    fn resolve_captured_cursor(&self) -> Option<CursorIcon> {
        let cap = self.captured_slot?;
        let slot = self.slots.get(cap as usize).and_then(|s| s.as_ref())?;
        if slot.on_drag.is_some() {
            match slot.cursor {
                Some(ic @ (CursorIcon::EwResize | CursorIcon::NsResize | CursorIcon::Text)) => {
                    Some(ic)
                }
                _ => Some(CursorIcon::Grabbing),
            }
        } else {
            slot.cursor
        }
    }

    fn refresh_hover_chain(&mut self, leaf_slot: Option<u32>) {
        self.hovered_slot = leaf_slot;
        self.hovered_chain.clear();
        let mut resolved_cursor = self.resolve_captured_cursor();

        let mut cur = leaf_slot;
        while let Some(s_idx) = cur {
            self.hovered_chain.push(s_idx);
            let Some(slot) = self.slots.get(s_idx as usize).and_then(|s| s.as_ref()) else {
                break;
            };
            if resolved_cursor.is_none()
                && let Some(ic) = slot.cursor
            {
                resolved_cursor = Some(ic);
            }
            cur = slot.parent;
        }
        self.active_cursor = resolved_cursor.unwrap_or(CursorIcon::Default);
    }

    /// Re-evaluate hovered element chain and `active_cursor` at the current stationary pointer coordinates.
    pub fn refresh_pointer_hover(&mut self) -> bool {
        let prev_hover = self.hovered_slot;
        let prev_cursor = self.active_cursor;
        if self.has_pointer {
            let [px, py] = self.last_pointer_pos;
            let new_hover = self.hit_test_filtered(px, py, |_| true);
            self.refresh_hover_chain(new_hover);
        } else {
            self.refresh_hover_chain(None);
        }
        prev_hover != self.hovered_slot || prev_cursor != self.active_cursor
    }

    /// Clear pointer hover state when the cursor leaves the window.
    pub fn pointer_leave(&mut self) -> bool {
        let had_hover = self.hovered_slot.is_some();
        self.has_pointer = false;
        self.refresh_hover_chain(None);
        had_hover
    }

    pub fn pointer_down<S: 'static>(&mut self, x: f32, y: f32, root_state: &mut S) -> bool {
        self.last_pointer_pos = [x, y];
        self.has_pointer = true;
        let prev_focus = self.focused_slot;
        let new_hover = self.hit_test_filtered(x, y, |_| true);
        let Some(hit_slot) = self.hit_test(x, y) else {
            self.focused_slot = None;
            self.refresh_hover_chain(new_hover);
            return prev_focus.is_some();
        };
        let (has_drag, has_click) = {
            let slot = self.slots[hit_slot as usize].as_ref().unwrap();
            (slot.on_drag.is_some(), slot.on_click.is_some())
        };
        // Walk up from `hit_slot` to find the nearest focusable element (e.g. clicking a row focuses its scroll view):
        let mut focus_target = None;
        let mut cur_f = Some(hit_slot);
        while let Some(s_i) = cur_f {
            let Some(s_ref) = self.slots.get(s_i as usize).and_then(|s| s.as_ref()) else {
                break;
            };
            if s_ref.focusable {
                focus_target = Some(s_i);
                break;
            }
            cur_f = s_ref.parent;
        }
        self.focused_slot = focus_target;
        if has_drag || has_click {
            self.captured_slot = Some(hit_slot);
        }
        self.refresh_hover_chain(new_hover);

        let ev = PointerEvent {
            x,
            y,
            ..PointerEvent::default()
        };
        if has_click {
            self.invoke_pointer_binding(hit_slot, true, root_state, &ev);
        }
        if has_drag {
            self.invoke_pointer_binding(hit_slot, false, root_state, &ev);
        }
        has_click || has_drag || focus_target.is_some() || prev_focus != self.focused_slot
    }

    /// Update pointer hover/drag state in logical coordinates `(x, y)`.
    /// Returns `true` if a drag handler fired OR hover state transitioned (so event-driven windows repaint!).
    pub fn pointer_move<S: 'static>(&mut self, x: f32, y: f32, root_state: &mut S) -> bool {
        let dx = if self.has_pointer {
            x - self.last_pointer_pos[0]
        } else {
            0.0
        };
        let dy = if self.has_pointer {
            y - self.last_pointer_pos[1]
        } else {
            0.0
        };
        self.last_pointer_pos = [x, y];
        self.has_pointer = true;

        let prev_hover = self.hovered_slot;
        let new_hover = self.hit_test_filtered(x, y, |_| true);
        self.refresh_hover_chain(new_hover);

        if let Some(cap) = self.captured_slot {
            let has_drag = self
                .slots
                .get(cap as usize)
                .and_then(|s| s.as_ref())
                .is_some_and(|s| s.on_drag.is_some());
            if has_drag {
                let ev = PointerEvent {
                    x,
                    y,
                    delta_x: dx,
                    delta_y: dy,
                    ..PointerEvent::default()
                };
                let _ = self.invoke_pointer_binding(cap, false, root_state, &ev);
                return true;
            }
        }
        prev_hover != new_hover
    }

    pub fn pointer_up(&mut self) -> bool {
        let was_cap = self.captured_slot.take().is_some();
        if was_cap {
            let _ = self.refresh_pointer_hover();
        }
        was_cap
    }

    /// Mouse wheel / touchpad scroll: bubbles from hovered leaf up to nearest `on_scroll` or `scroll_area` container!
    pub fn dispatch_scroll<S: 'static>(
        &mut self,
        x: f32,
        y: f32,
        wheel_dy: f32,
        root_state: &mut S,
    ) -> bool {
        self.last_pointer_pos = [x, y];
        self.has_pointer = true;
        let Some(start_slot) = self.hit_test_filtered(x, y, |_| true) else {
            self.refresh_hover_chain(None);
            return false;
        };
        let mut cur_node = self.slots[start_slot as usize].as_ref().unwrap().node_id;
        while !cur_node.is_none() {
            let n_idx = cur_node.idx();
            if let Some(&s_idx) = self.node_to_slot.get(n_idx)
                && s_idx != u32::MAX
                && let Some(slot) = self.slots[s_idx as usize].as_ref()
            {
                if let Some(sb) = &slot.on_scroll {
                    let mut target_ptr = (root_state as *mut S).cast::<()>();
                    let mut ok = true;
                    for step in sb.steps.iter() {
                        if let Some(next) = unsafe { (step.project)(target_ptr, step.aux) } {
                            target_ptr = next;
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        let invoke = sb.invoke;
                        let h_slot = sb.handler_slot;
                        if unsafe { invoke(target_ptr, h_slot, &mut self.state_arena, wheel_dy) } {
                            let _ = self.refresh_pointer_hover();
                            return true;
                        }
                    }
                }
                if let Some(max_scroll) = slot.scrollable_max_y {
                    let mut spec = self.layout.specs[n_idx];
                    let next_scroll = (spec.scroll_y - wheel_dy).clamp(0.0, max_scroll.max(0.0));
                    if (next_scroll - spec.scroll_y).abs() > 1e-4 {
                        spec.scroll_y = next_scroll;
                        self.layout.update_spec(cur_node, spec);
                        self.scene_topology_changed = true;
                        let [lw, lh] = self.current_logical_size;
                        if lw > 0.0 && lh > 0.0 {
                            self.layout.compute_layout(lw, lh);
                        }
                        let _ = self.refresh_pointer_hover();
                        return true;
                    }
                }
            }
            cur_node = self.layout.topo[n_idx].parent;
        }
        false
    }

    /// Dispatch keyboard input (`Tab` focus traversal + focused widget key handling + Enter/Space button activation!).
    pub fn dispatch_key<S: 'static>(&mut self, key: &KeyInput, root_state: &mut S) -> bool {
        if let KeyInput::Tab { shift } = *key {
            return self.advance_focus(shift);
        }
        if *key == KeyInput::Escape && self.focused_slot.is_some() {
            self.focused_slot = None;
            return true;
        }
        let Some(f_slot) = self.focused_slot else {
            return false;
        };
        // Bubble `on_key` from `focused_slot` up through ancestor slots until handled:
        let mut cur_s = Some(f_slot);
        while let Some(s_i) = cur_s {
            let Some(slot) = self.slots.get(s_i as usize).and_then(|s| s.as_ref()) else {
                break;
            };
            let next_parent = slot.parent;
            if let Some(kb) = &slot.on_key {
                let mut target_ptr = (root_state as *mut S).cast::<()>();
                let mut valid = true;
                for step in kb.steps.iter() {
                    if let Some(next) = unsafe { (step.project)(target_ptr, step.aux) } {
                        target_ptr = next;
                    } else {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    let invoke = kb.invoke;
                    let h_slot = kb.handler_slot;
                    if unsafe {
                        invoke(
                            target_ptr,
                            h_slot,
                            &mut self.state_arena,
                            &mut self.clipboard,
                            key,
                        )
                    } {
                        return true;
                    }
                }
            }
            cur_s = next_parent;
        }
        // Standard desktop keyboard accessibility: pressing Enter or Space on a focused clickable control activates it!
        if let Some(slot) = self.slots.get(f_slot as usize).and_then(|s| s.as_ref())
            && matches!(*key, KeyInput::Enter | KeyInput::Space)
            && slot.on_click.is_some()
        {
            let nid = slot.node_id;
            let b = self.layout.boxes[nid.idx()];
            let ev = PointerEvent {
                x: b.x + b.w * 0.5,
                y: b.y + b.h * 0.5,
                ..PointerEvent::default()
            };
            return self.invoke_pointer_binding(f_slot, true, root_state, &ev);
        }
        false
    }

    /// Bubble a typed `Action` from the currently focused element (or root) up the ancestor chain!
    pub fn dispatch_action<S: 'static, A: 'static>(&mut self, action: &A, root_state: &mut S) -> bool {
        let start_slot = self.focused_slot.or(self.root_slot);
        let Some(s0) = start_slot else {
            return false;
        };
        let mut cur_node = self.slots[s0 as usize].as_ref().unwrap().node_id;
        let want_tid = TypeId::of::<A>();

        while !cur_node.is_none() {
            let n_idx = cur_node.idx();
            if let Some(&s_idx) = self.node_to_slot.get(n_idx)
                && s_idx != u32::MAX
                && let Some(slot) = self.slots[s_idx as usize].as_ref()
            {
                for ab in &slot.actions {
                    if ab.action_type == want_tid {
                        let mut target_ptr = (root_state as *mut S).cast::<()>();
                        let mut valid = true;
                        for step in ab.steps.iter() {
                            if let Some(next) = unsafe { (step.project)(target_ptr, step.aux) } {
                                target_ptr = next;
                            } else {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            let invoke = ab.invoke;
                            let h_addr = ab.handler_addr;
                            if unsafe { invoke(target_ptr, h_addr, action as &dyn Any) } {
                                return true;
                            }
                        }
                    }
                }
            }
            cur_node = self.layout.topo[n_idx].parent;
        }
        false
    }

    fn advance_focus(&mut self, reverse: bool) -> bool {
        let mut order = Vec::new();
        if let Some(r) = self.root_slot {
            self.collect_focusable_dfs(r, &mut order);
        }
        if order.is_empty() {
            self.focused_slot = None;
            return false;
        }
        let next = match self
            .focused_slot
            .and_then(|cur| order.iter().position(|&s| s == cur))
        {
            Some(pos) => {
                if reverse {
                    order[(pos + order.len() - 1) % order.len()]
                } else {
                    order[(pos + 1) % order.len()]
                }
            }
            None => {
                if reverse {
                    *order.last().unwrap()
                } else {
                    order[0]
                }
            }
        };
        self.focused_slot = Some(next);
        true
    }

    fn collect_focusable_dfs(&self, slot_idx: u32, out: &mut Vec<u32>) {
        let Some(slot) = self.slots[slot_idx as usize].as_ref() else {
            return;
        };
        if slot.focusable {
            out.push(slot_idx);
        }
        for &ch in &slot.children {
            self.collect_focusable_dfs(ch, out);
        }
    }

    fn invoke_pointer_binding<S: 'static>(
        &mut self,
        slot_idx: u32,
        is_click: bool,
        root_state: &mut S,
        ev: &PointerEvent,
    ) -> bool {
        let Some(slot) = self.slots.get(slot_idx as usize).and_then(|s| s.as_ref()) else {
            return false;
        };
        let nid = slot.node_id;
        let b_box = self.layout.boxes[nid.idx()];
        let resolved_ev = PointerEvent {
            x: ev.x,
            y: ev.y,
            local_x: ev.x - b_box.x,
            local_y: ev.y - b_box.y,
            target_w: b_box.w,
            target_h: b_box.h,
            delta_x: ev.delta_x,
            delta_y: ev.delta_y,
        };
        let binding = if is_click {
            slot.on_click.as_ref()
        } else {
            slot.on_drag.as_ref()
        };
        let Some(b) = binding else {
            return false;
        };
        let mut target_ptr = (root_state as *mut S).cast::<()>();
        for step in b.steps.iter() {
            let Some(next) = (unsafe { (step.project)(target_ptr, step.aux) }) else {
                return false;
            };
            target_ptr = next;
        }
        let invoke = b.invoke;
        let h_slot = b.handler_slot;
        unsafe {
            invoke(
                target_ptr,
                h_slot,
                &mut self.state_arena,
                &self.fonts,
                &self.layout,
                &resolved_ev,
            )
        };
        true
    }

    // =========================================================================
    // Incremental AccessKit 0.24 TreeUpdate Export & Action Routing
    // =========================================================================

    pub fn build_accesskit_update(&mut self) -> Option<TreeUpdate> {
        if !self.a11y_enabled {
            self.dirty_a11y_slots.clear();
            return None;
        }
        let root_s = self.root_slot?;
        let root_a11y = slot_to_a11y_id(root_s);
        let focus_id = self.focused_slot.map_or(root_a11y, slot_to_a11y_id);

        if !self.a11y_initialized {
            self.a11y_initialized = true;
            self.dirty_a11y_slots.clear();
            let mut nodes = Vec::new();
            self.collect_a11y_subtree(root_s, &mut nodes);
            return Some(TreeUpdate {
                nodes,
                tree: Some(crate::a11y::A11yTree {
                    root: root_a11y,
                    toolkit_name: Some("skima".to_owned()),
                    toolkit_version: Some("1.0.0".to_owned()),
                }),
                tree_id: TreeId::ROOT,
                focus: focus_id,
            });
        }

        if self.dirty_a11y_slots.is_empty() {
            return None;
        }

        let mut nodes = Vec::with_capacity(self.dirty_a11y_slots.len());
        let dirty = core::mem::take(&mut self.dirty_a11y_slots);
        for s_idx in dirty {
            if let Some(slot) = self.slots.get(s_idx as usize).and_then(|s| s.as_ref()) {
                let bounds = self.layout.boxes[slot.node_id.idx()];
                let child_ids: Vec<A11yNodeId> =
                    slot.children.iter().map(|&c| slot_to_a11y_id(c)).collect();
                let default_sem = SemanticsMeta {
                    role: if s_idx == root_s {
                        A11yRole::Window
                    } else {
                        A11yRole::GenericContainer
                    },
                    ..SemanticsMeta::default()
                };
                let sem = slot.semantics.as_ref().unwrap_or(&default_sem);
                let mut a11y_node = sem.build_accesskit_node(bounds, &child_ids);
                if s_idx == root_s {
                    apply_root_scale_transform(&mut a11y_node, self.scale_factor);
                }
                nodes.push((slot_to_a11y_id(s_idx), a11y_node));
            }
        }
        Some(TreeUpdate {
            nodes,
            tree: None,
            tree_id: TreeId::ROOT,
            focus: focus_id,
        })
    }

    fn collect_a11y_subtree(
        &self,
        slot_idx: u32,
        out: &mut Vec<(A11yNodeId, crate::a11y::A11yNode)>,
    ) {
        let Some(slot) = self.slots[slot_idx as usize].as_ref() else {
            return;
        };
        let bounds = self.layout.boxes[slot.node_id.idx()];
        let child_ids: Vec<A11yNodeId> = slot.children.iter().map(|&c| slot_to_a11y_id(c)).collect();
        let is_root = Some(slot_idx) == self.root_slot;
        let default_sem = SemanticsMeta {
            role: if is_root {
                A11yRole::Window
            } else {
                A11yRole::GenericContainer
            },
            ..SemanticsMeta::default()
        };
        let sem = slot.semantics.as_ref().unwrap_or(&default_sem);
        let mut a11y_node = sem.build_accesskit_node(bounds, &child_ids);
        if is_root {
            apply_root_scale_transform(&mut a11y_node, self.scale_factor);
        }
        out.push((slot_to_a11y_id(slot_idx), a11y_node));
        for &ch in &slot.children {
            self.collect_a11y_subtree(ch, out);
        }
    }

    fn dispatch_key_from_slot<S: 'static>(
        &mut self,
        start_slot: u32,
        key: &KeyInput,
        root_state: &mut S,
    ) -> bool {
        let mut cur_s = Some(start_slot);
        while let Some(s_i) = cur_s {
            let Some(slot) = self.slots.get(s_i as usize).and_then(|s| s.as_ref()) else {
                break;
            };
            let next_parent = slot.parent;
            if let Some(kb) = &slot.on_key {
                let mut target_ptr = (root_state as *mut S).cast::<()>();
                let mut valid = true;
                for step in kb.steps.iter() {
                    if let Some(next) = unsafe { (step.project)(target_ptr, step.aux) } {
                        target_ptr = next;
                    } else {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    let invoke = kb.invoke;
                    let h_slot = kb.handler_slot;
                    if unsafe {
                        invoke(
                            target_ptr,
                            h_slot,
                            &mut self.state_arena,
                            &mut self.clipboard,
                            key,
                        )
                    } {
                        return true;
                    }
                }
            }
            cur_s = next_parent;
        }
        false
    }

    pub fn handle_accesskit_action<S: 'static>(
        &mut self,
        req: &ActionRequest,
        root_state: &mut S,
    ) -> bool {
        let Some(slot_idx) = a11y_id_to_slot(req.target_node) else {
            return false;
        };
        let Some(role) = self
            .slots
            .get(slot_idx as usize)
            .and_then(|s| s.as_ref())
            .map(|s| {
                s.semantics
                    .as_ref()
                    .map_or(A11yRole::GenericContainer, |sem| sem.role)
            })
        else {
            return false;
        };
        match req.action {
            Action::Focus => {
                self.focused_slot = Some(slot_idx);
                true
            }
            Action::Click => {
                let nid = self.slots[slot_idx as usize].as_ref().unwrap().node_id;
                let b = self.layout.boxes[nid.idx()];
                let ev = PointerEvent {
                    x: b.x + b.w * 0.5,
                    y: b.y + b.h * 0.5,
                    ..PointerEvent::default()
                };
                self.invoke_pointer_binding(slot_idx, true, root_state, &ev)
            }
            Action::Increment => {
                let step_key = if matches!(role, A11yRole::ScrollBar | A11yRole::ScrollView) {
                    KeyInput::ArrowDown
                } else {
                    KeyInput::ArrowRight
                };
                self.dispatch_key_from_slot(slot_idx, &step_key, root_state)
            }
            Action::Decrement => {
                let step_key = if matches!(role, A11yRole::ScrollBar | A11yRole::ScrollView) {
                    KeyInput::ArrowUp
                } else {
                    KeyInput::ArrowLeft
                };
                self.dispatch_key_from_slot(slot_idx, &step_key, root_state)
            }
            Action::ScrollDown => {
                self.dispatch_key_from_slot(slot_idx, &KeyInput::ArrowDown, root_state)
            }
            Action::ScrollUp => {
                self.dispatch_key_from_slot(slot_idx, &KeyInput::ArrowUp, root_state)
            }
            Action::SetValue => {
                if let Some(ActionData::Value(val)) = &req.data {
                    let cleared =
                        self.dispatch_key_from_slot(slot_idx, &KeyInput::SelectAll, root_state);
                    let inserted = self.dispatch_key_from_slot(
                        slot_idx,
                        &KeyInput::Text(val.to_string()),
                        root_state,
                    );
                    cleared || inserted
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn unmount_slot(&mut self, slot_idx: u32) {
        let Some(mut slot) = self.slots[slot_idx as usize].take() else {
            return;
        };
        if self.root_slot == Some(slot_idx) {
            self.root_slot = None;
            self.a11y_initialized = false;
        }
        if self.focused_slot == Some(slot_idx) {
            self.focused_slot = None;
        }
        if self.captured_slot == Some(slot_idx) {
            self.captured_slot = None;
        }
        if self.hovered_slot == Some(slot_idx) {
            self.hovered_slot = None;
        }
        self.hovered_chain.retain(|&s| s != slot_idx);
        self.dirty_paint_slots.retain(|&s| s != slot_idx);
        self.dirty_a11y_slots.retain(|&s| s != slot_idx);

        for ch in slot.children.drain(..).rev() {
            self.unmount_slot(ch);
        }
        for unsub in slot.unsub_hooks.drain(..).rev() {
            unsub();
        }
        for st_idx in slot.local_states.drain(..).rev() {
            self.active_springs.retain(|&sp_idx| sp_idx != st_idx);
            let _dropped = self.state_arena[st_idx as usize].take();
            self.free_states.push(st_idx);
        }
        if let Some(m) = self.node_to_slot.get_mut(slot.node_id.idx()) {
            *m = u32::MAX;
        }
        self.layout.free_subtree(slot.node_id);
        slot.actions.clear();
        slot.on_click = None;
        slot.on_drag = None;
        slot.on_key = None;
        slot.on_scroll = None;
        slot.active_bindings = 0;
        slot.pointer_passthrough = false;
        slot.cursor = None;
        slot.semantics = None;
        slot.custom_atlas_region = None;
        slot.custom_atlas_gen = 0;
        slot.custom_upload_tag = 0;
        slot.custom_dirty = false;
        self.slot_pool_stash.push(slot);
        self.free_slots.push(slot_idx);
        self.scene_topology_changed = true;
    }

    fn alloc_slot(
        &mut self,
        key: SlotKey,
        parent: Option<u32>,
        spec: LayoutSpec,
        paint: PaintStyle,
    ) -> u32 {
        let node_id = self.layout.alloc_node(spec);
        let slot_idx = if let Some(idx) = self.free_slots.pop() {
            idx
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(None);
            idx
        };
        if node_id.idx() >= self.node_to_slot.len() {
            self.node_to_slot.resize(node_id.idx() + 1, u32::MAX);
        }
        self.node_to_slot[node_id.idx()] = slot_idx;

        let recycled = if let Some(mut s) = self.slot_pool_stash.pop() {
            s.key = key;
            s.parent = parent;
            s.node_id = node_id;
            s.paint = paint;
            s.committed_paint = paint;
            s.custom_dirty = false;
            s.active_bindings = 0;
            s.text_color = Color::WHITE;
            s.aa_mode = AntialiasMode::Grayscale;
            s.prim_range = CopyRange { start: 0, end: 0 };
            s.custom_atlas_region = None;
            s.custom_atlas_gen = 0;
            s.custom_upload_tag = 0;
            s.children.clear();
            s.local_states.clear();
            s.on_click = None;
            s.on_drag = None;
            s.on_key = None;
            s.on_scroll = None;
            s.actions.clear();
            s.scrollable_max_y = None;
            s.focusable = false;
            s.pointer_passthrough = false;
            s.cursor = None;
            s.semantics = None;
            s.scope_id = None;
            s.scope_epoch = 0;
            s.unsub_hooks.clear();
            s
        } else {
            RetainedSlot {
                key,
                parent,
                node_id,
                paint,
                committed_paint: paint,
                custom_dirty: false,
                active_bindings: 0,
                text_color: Color::WHITE,
                aa_mode: AntialiasMode::Grayscale,
                prim_range: CopyRange { start: 0, end: 0 },
                custom_atlas_region: None,
                custom_atlas_gen: 0,
                custom_upload_tag: 0,
                children: Vec::new(),
                local_states: Vec::new(),
                on_click: None,
                on_drag: None,
                on_key: None,
                on_scroll: None,
                actions: Vec::new(),
                scrollable_max_y: None,
                focusable: false,
                pointer_passthrough: false,
                cursor: None,
                semantics: None,
                scope_id: None,
                scope_epoch: 0,
                unsub_hooks: Vec::new(),
            }
        };
        self.slots[slot_idx as usize] = Some(recycled);
        if self.a11y_enabled && self.a11y_initialized && !self.dirty_a11y_slots.contains(&slot_idx) {
            self.dirty_a11y_slots.push(slot_idx);
        }
        self.uncommitted_allocs.push(slot_idx);
        self.scene_topology_changed = true;
        slot_idx
    }
}

#[derive(Clone, Copy)]
pub struct LocalHandle<T> {
    arena_idx: u32,
    _marker: core::marker::PhantomData<fn() -> T>,
}

/// Two-lifetime retained UI cursor (`'sys` for `&mut GuiSystem`, `'s` for shared frame state `&'s S`).
pub struct Ui<'sys, 's, S: ?Sized> {
    pub sys: &'sys mut GuiSystem,
    state_ptr: *const (),
    parent_slot: Option<u32>,
    cursor_pos: usize,
    callsite_scope_base: usize,
    structure_changed: bool,
    pending_user_key: u64,
    active_slot: Option<u32>,
    state_cursor: usize,
    _marker: core::marker::PhantomData<&'s S>,
}

impl<'sys, 's, S: 'static> Ui<'sys, 's, S> {
    #[inline]
    pub fn state(&self) -> &'s S {
        unsafe { &*self.state_ptr.cast::<S>() }
    }

    #[inline]
    pub fn theme(&self) -> Theme {
        self.sys.theme
    }

    #[inline]
    pub fn is_focused(&self) -> bool {
        self.active_slot.is_some() && self.sys.focused_slot == self.active_slot
    }

    /// Returns `true` when the pointer hovers the active element OR any descendant inside it!
    #[inline]
    pub fn is_hovered(&self) -> bool {
        let Some(cur) = self.active_slot else {
            return false;
        };
        self.sys.hovered_chain.contains(&cur)
    }

    #[inline]
    pub fn is_captured(&self) -> bool {
        self.active_slot.is_some() && self.sys.captured_slot == self.active_slot
    }

    /// Previous-frame computed logical box `(x, y, w, h)` of the active element (useful for hit-testing sliders/carets).
    #[inline]
    pub fn active_box(&self) -> crate::layout::ComputedBox {
        let Some(s_idx) = self.active_slot else {
            return crate::layout::ComputedBox::default();
        };
        let nid = self.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
        self.sys.layout.boxes[nid.idx()]
    }

    #[inline]
    pub fn keyed<R>(&mut self, key: u64, f: impl FnOnce(&mut Ui<'_, 's, S>) -> R) -> R {
        let mixed = (key ^ 0x9e37_79b9_7f4a_7c15).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let prev_payload = self.pending_user_key & !USER_KEY_EXPLICIT_BIT;
        self.pending_user_key = USER_KEY_EXPLICIT_BIT | ((prev_payload ^ mixed) & !USER_KEY_EXPLICIT_BIT);
        let mut guard = DropGuard::new(self, |u| {
            u.pending_user_key = 0;
        });
        let res = f(&mut guard);
        let finished = DropGuard::dismiss(guard);
        finished.pending_user_key = 0;
        res
    }

    #[inline]
    fn resolve_slot_key(&mut self, caller_addr: usize) -> SlotKey {
        let pending = core::mem::replace(&mut self.pending_user_key, 0);
        if (pending & USER_KEY_EXPLICIT_BIT) != 0 {
            return SlotKey {
                callsite_addr: caller_addr,
                user_key: pending,
            };
        }
        let scope = &mut self.sys.callsite_counts_scratch[self.callsite_scope_base..];
        if let Some(last) = scope.last_mut()
            && last.0 == caller_addr
        {
            let ord = last.1 as u64;
            last.1 = last.1.saturating_add(1);
            return SlotKey {
                callsite_addr: caller_addr,
                user_key: ord,
            };
        }
        for entry in scope.iter_mut() {
            if entry.0 == caller_addr {
                let ord = entry.1 as u64;
                entry.1 = entry.1.saturating_add(1);
                return SlotKey {
                    callsite_addr: caller_addr,
                    user_key: ord,
                };
            }
        }
        self.sys.callsite_counts_scratch.push((caller_addr, 1));
        SlotKey {
            callsite_addr: caller_addr,
            user_key: 0,
        }
    }

    /// Zero-monomorphization, zero-allocation state projection (`&mut Ui<Parent>` -> `&mut Ui<Child>`).
    pub fn lens<Sub: 'static>(
        &mut self,
        project_ref: fn(&'s S) -> &'s Sub,
        project_mut: fn(&mut S) -> &mut Sub,
        child_fn: impl FnOnce(&mut Ui<'_, 's, Sub>),
    ) {
        unsafe fn trampoline<P: 'static, C: 'static>(ptr: *mut (), fn_addr: usize) -> Option<*mut ()> {
            let f: fn(&mut P) -> &mut C = unsafe { core::mem::transmute(fn_addr) };
            let p_mut = unsafe { &mut *ptr.cast::<P>() };
            Some((f(p_mut) as *mut C).cast::<()>())
        }

        let sub_ptr = (project_ref(self.state()) as *const Sub).cast::<()>();
        self.sys.lens_scratch.push(LensStep {
            project: trampoline::<S, Sub>,
            aux: project_mut as usize,
        });

        let cursor_pos = self.cursor_pos;
        let callsite_scope_base = self.callsite_scope_base;
        let structure_changed = self.structure_changed;
        let pending_user_key = self.pending_user_key;
        let active_slot = self.active_slot;
        let state_cursor = self.state_cursor;
        let parent_slot = self.parent_slot;

        let (new_cursor, new_struct_changed, new_state_cursor) = {
            let mut sub_ui = Ui::<Sub> {
                sys: &mut *self.sys,
                state_ptr: sub_ptr,
                parent_slot,
                cursor_pos,
                callsite_scope_base,
                structure_changed,
                pending_user_key,
                active_slot,
                state_cursor,
                _marker: core::marker::PhantomData,
            };
            let mut guard = DropGuard::new(&mut sub_ui, |u| {
                u.sys.lens_scratch.pop();
            });
            child_fn(&mut guard);
            let finished = DropGuard::dismiss(guard);
            finished.sys.lens_scratch.pop();
            (
                finished.cursor_pos,
                finished.structure_changed,
                finished.state_cursor,
            )
        };

        self.cursor_pos = new_cursor;
        self.structure_changed = new_struct_changed;
        self.pending_user_key = 0;
        self.state_cursor = new_state_cursor;
    }

    pub fn stateless<V: StatelessView<!>>(&mut self, view: &V) {
        let (new_cursor, new_struct_changed, new_state_cursor) = {
            let mut never_ui = Ui::<!> {
                sys: &mut *self.sys,
                state_ptr: core::ptr::dangling(),
                parent_slot: self.parent_slot,
                cursor_pos: self.cursor_pos,
                callsite_scope_base: self.callsite_scope_base,
                structure_changed: self.structure_changed,
                pending_user_key: self.pending_user_key,
                active_slot: self.active_slot,
                state_cursor: self.state_cursor,
                _marker: core::marker::PhantomData,
            };
            let _ = view.mount(&mut never_ui);
            (
                never_ui.cursor_pos,
                never_ui.structure_changed,
                never_ui.state_cursor,
            )
        };
        self.cursor_pos = new_cursor;
        self.structure_changed = new_struct_changed;
        self.pending_user_key = 0;
        self.state_cursor = new_state_cursor;
    }

    pub fn use_state<T: 'static>(&mut self, init: impl FnOnce() -> T) -> LocalHandle<T> {
        let slot_idx = self
            .active_slot
            .expect("use_state must be called inside an element scope");
        let pos = self.state_cursor;
        self.state_cursor += 1;

        let existing = {
            let slot = self.sys.slots[slot_idx as usize].as_ref().unwrap();
            slot.local_states.get(pos).copied()
        };
        let st_arena_idx = if let Some(idx) = existing {
            let matches_type = self.sys.state_arena[idx as usize]
                .as_ref()
                .is_some_and(|b| (**b).type_id() == TypeId::of::<T>());
            if !matches_type {
                if self.sys.state_arena[idx as usize]
                    .as_ref()
                    .is_some_and(|b| b.is::<Spring>())
                {
                    self.sys.active_springs.retain(|&sp_idx| sp_idx != idx);
                }
                self.sys.state_arena[idx as usize] = Some(Box::new(init()));
            }
            idx
        } else {
            let val: Box<dyn Any> = Box::new(init());
            let idx = if let Some(free) = self.sys.free_states.pop() {
                self.sys.state_arena[free as usize] = Some(val);
                free
            } else {
                let i = self.sys.state_arena.len() as u32;
                self.sys.state_arena.push(Some(val));
                i
            };
            self.sys.slots[slot_idx as usize]
                .as_mut()
                .unwrap()
                .local_states
                .push(idx);
            idx
        };
        LocalHandle {
            arena_idx: st_arena_idx,
            _marker: core::marker::PhantomData,
        }
    }

    /// Store `latest: T` in the element's retained state slot, updating it in-place on subsequent frames
    /// with **zero** heap allocations so closures always read current-frame captures!
    pub fn use_state_updated<T: 'static>(&mut self, latest: T) -> LocalHandle<T> {
        let slot_idx = self
            .active_slot
            .expect("use_state_updated must be called inside an element scope");
        let pos = self.state_cursor;
        self.state_cursor += 1;

        let existing = {
            let slot = self.sys.slots[slot_idx as usize].as_ref().unwrap();
            slot.local_states.get(pos).copied()
        };
        let st_arena_idx = if let Some(idx) = existing {
            if let Some(target) = self.sys.state_arena[idx as usize]
                .as_mut()
                .and_then(|b| b.downcast_mut::<T>())
            {
                *target = latest;
            } else {
                if self.sys.state_arena[idx as usize]
                    .as_ref()
                    .is_some_and(|b| b.is::<Spring>())
                {
                    self.sys.active_springs.retain(|&sp_idx| sp_idx != idx);
                }
                self.sys.state_arena[idx as usize] = Some(Box::new(latest));
            }
            idx
        } else {
            let val: Box<dyn Any> = Box::new(latest);
            let idx = if let Some(free) = self.sys.free_states.pop() {
                self.sys.state_arena[free as usize] = Some(val);
                free
            } else {
                let i = self.sys.state_arena.len() as u32;
                self.sys.state_arena.push(Some(val));
                i
            };
            self.sys.slots[slot_idx as usize]
                .as_mut()
                .unwrap()
                .local_states
                .push(idx);
            idx
        };
        LocalHandle {
            arena_idx: st_arena_idx,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline]
    pub fn read_local<T: 'static>(&self, handle: LocalHandle<T>) -> &T {
        self.sys.state_arena[handle.arena_idx as usize]
            .as_ref()
            .unwrap()
            .downcast_ref::<T>()
            .expect("type-matched local state")
    }

    /// Declarative critically-damped analytical spring hook!
    /// Automatically schedules frames on `GuiSystem` while moving and settles to 0% idle CPU.
    pub fn use_spring(&mut self, target: f32, response_hz: f32) -> f32 {
        self.use_spring_damped(target, response_hz, 1.0)
    }

    /// Declarative analytical spring hook with custom `damping_ratio` (`< 1.0` = bouncy, `1.0` = critically damped, `> 1.0` = overdamped).
    pub fn use_spring_damped(&mut self, target: f32, response_hz: f32, damping_ratio: f32) -> f32 {
        let safe_hz = if response_hz.is_finite() {
            response_hz.max(0.2)
        } else {
            1.0
        };
        let safe_z = if damping_ratio.is_finite() {
            damping_ratio.max(0.05)
        } else {
            1.0
        };
        let safe_target = if target.is_finite() { target } else { 0.0 };
        let h = self.use_state(|| Spring {
            value: safe_target,
            velocity: 0.0,
            target: safe_target,
            angular_freq: safe_hz * core::f32::consts::TAU,
            damping_ratio: safe_z,
        });
        let sp = self.sys.state_arena[h.arena_idx as usize]
            .as_mut()
            .unwrap()
            .downcast_mut::<Spring>()
            .unwrap();
        sp.angular_freq = safe_hz * core::f32::consts::TAU;
        sp.damping_ratio = safe_z;
        if target.is_finite() {
            sp.set_target(target);
        }
        let val = sp.value;
        let moving = !sp.is_settled(1e-4);
        if moving && !self.sys.active_springs.contains(&h.arena_idx) {
            self.sys.active_springs.push(h.arena_idx);
        }
        val
    }

    #[track_caller]
    pub fn memo_scope(
        &mut self,
        scope_id: ScopeId,
        input_epoch: u64,
        spec: LayoutSpec,
        paint: PaintStyle,
        f: impl FnOnce(&mut Ui<'_, 's, S>),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let key = self.resolve_slot_key(caller);

        if let Some((matched_slot, matched_pos)) = self.find_matching_child(key) {
            let scope_dirty = self
                .sys
                .dirty_scopes
                .get(scope_id.0 as usize)
                .copied()
                .unwrap_or(true);
            let same_epoch =
                self.sys.slots[matched_slot as usize].as_ref().unwrap().scope_epoch == input_epoch;
            if !scope_dirty && same_epoch {
                self.advance_to_matched_in_place(matched_pos, key.is_explicitly_keyed());
                self.sys.slots[matched_slot as usize].as_mut().unwrap().parent = self.parent_slot;
                self.reconcile_slot_props(matched_slot, spec, paint);
                self.sys.scopes_skipped_last_frame += 1;
                return self.sys.slots[matched_slot as usize]
                    .as_ref()
                    .unwrap()
                    .node_id;
            }
        }

        if let Some(flag) = self.sys.dirty_scopes.get_mut(scope_id.0 as usize) {
            *flag = false;
        }
        self.element_with_resolved_key(key, spec, paint, |ui| {
            let s_idx = ui.active_slot.unwrap();
            let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
            slot.scope_id = Some(scope_id);
            slot.scope_epoch = input_epoch;
            f(ui);
        })
    }

    #[track_caller]
    pub fn element(
        &mut self,
        spec: LayoutSpec,
        paint: PaintStyle,
        children_fn: impl FnOnce(&mut Ui<'_, 's, S>),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        self.element_internal(caller, spec, paint, children_fn)
    }

    #[track_caller]
    pub fn grid(
        &mut self,
        spec: LayoutSpec,
        cols: Vec<TrackSize>,
        rows: Vec<TrackSize>,
        paint: PaintStyle,
        children_fn: impl FnOnce(&mut Ui<'_, 's, S>),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        self.element_internal(caller, spec, paint, |ui| {
            let s_idx = ui.active_slot.unwrap();
            let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
            ui.sys.layout.attach_grid(nid, cols, rows);
            children_fn(ui);
        })
    }

    #[track_caller]
    pub fn anchored_popover(
        &mut self,
        target: NodeId,
        placement: AnchorPlacement,
        offset: [f32; 2],
        spec: LayoutSpec,
        paint: PaintStyle,
        children_fn: impl FnOnce(&mut Ui<'_, 's, S>),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        self.element_internal(caller, spec, paint, |ui| {
            let s_idx = ui.active_slot.unwrap();
            let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
            ui.sys
                .layout
                .set_floating_anchor(nid, target, placement, offset[0], offset[1]);
            children_fn(ui);
        })
    }

    fn element_internal(
        &mut self,
        caller_addr: usize,
        spec: LayoutSpec,
        paint: PaintStyle,
        children_fn: impl FnOnce(&mut Ui<'_, 's, S>),
    ) -> NodeId {
        let key = self.resolve_slot_key(caller_addr);
        self.element_with_resolved_key(key, spec, paint, children_fn)
    }

    fn element_with_resolved_key(
        &mut self,
        key: SlotKey,
        spec: LayoutSpec,
        paint: PaintStyle,
        children_fn: impl FnOnce(&mut Ui<'_, 's, S>),
    ) -> NodeId {
        let alloc_checkpoint = self.sys.uncommitted_allocs.len();
        let parent_for_child = self.parent_slot;

        let slot_idx = if parent_for_child.is_none() {
            if let Some(existing) = self.sys.root_slot {
                let existing_key = self.sys.slots[existing as usize].as_ref().unwrap().key;
                if existing_key == key {
                    self.reconcile_slot_props(existing, spec, paint);
                    existing
                } else {
                    self.sys.unmount_slot(existing);
                    let s = self.sys.alloc_slot(key, None, spec, paint);
                    self.sys.root_slot = Some(s);
                    s
                }
            } else {
                let s = self.sys.alloc_slot(key, None, spec, paint);
                self.sys.root_slot = Some(s);
                s
            }
        } else if let Some((matched_slot, matched_pos)) = self.find_matching_child(key) {
            self.advance_to_matched_in_place(matched_pos, key.is_explicitly_keyed());
            self.sys.slots[matched_slot as usize].as_mut().unwrap().parent = parent_for_child;
            self.reconcile_slot_props(matched_slot, spec, paint);
            matched_slot
        } else {
            let s = self.sys.alloc_slot(key, parent_for_child, spec, paint);
            if let Some(p_idx) = parent_for_child {
                let children = &mut self.sys.slots[p_idx as usize].as_mut().unwrap().children;
                children.insert(self.cursor_pos, s);
                self.cursor_pos += 1;
                self.structure_changed = true;
            }
            s
        };

        let node_id = self.sys.slots[slot_idx as usize].as_ref().unwrap().node_id;
        let was_scrollable = {
            let slot = self.sys.slots[slot_idx as usize].as_mut().unwrap();
            let prev_sc = slot.scrollable_max_y.is_some();
            slot.cursor = None;
            slot.focusable = false;
            slot.pointer_passthrough = false;
            slot.scrollable_max_y = None;
            slot.active_bindings = 0;
            prev_sc
        };

        let child_scope_base = self.sys.callsite_counts_scratch.len();
        let (final_cursor, mut struct_changed, final_state_cursor) = {
            let mut child_ui = Ui::<S> {
                sys: &mut *self.sys,
                state_ptr: self.state_ptr,
                parent_slot: Some(slot_idx),
                cursor_pos: 0,
                callsite_scope_base: child_scope_base,
                structure_changed: false,
                pending_user_key: 0,
                active_slot: Some(slot_idx),
                state_cursor: 0,
                _marker: core::marker::PhantomData,
            };

            let mut guard = DropGuard::new(&mut child_ui, |aborted_ui| {
                aborted_ui
                    .sys
                    .callsite_counts_scratch
                    .truncate(child_scope_base);
                while aborted_ui.sys.uncommitted_allocs.len() > alloc_checkpoint {
                    let s = aborted_ui.sys.uncommitted_allocs.pop().unwrap();
                    if let Some(p_idx) = aborted_ui.parent_slot
                        && let Some(p_slot) = aborted_ui.sys.slots[p_idx as usize].as_mut()
                    {
                        p_slot.children.retain(|&c| c != s);
                    }
                    if aborted_ui.sys.root_slot == Some(s) {
                        aborted_ui.sys.root_slot = None;
                    }
                    aborted_ui.sys.unmount_slot(s);
                }
            });

            children_fn(&mut guard);

            let completed_ui = DropGuard::dismiss(guard);
            completed_ui
                .sys
                .callsite_counts_scratch
                .truncate(child_scope_base);
            (
                completed_ui.cursor_pos,
                completed_ui.structure_changed,
                completed_ui.state_cursor,
            )
        };
        if self.parent_slot.is_none() {
            self.sys.uncommitted_allocs.truncate(alloc_checkpoint);
        }

        // Reconcile conditional local states, handlers, and focus hygiene:
        {
            let old_state_len = self.sys.slots[slot_idx as usize]
                .as_ref()
                .unwrap()
                .local_states
                .len();
            if final_state_cursor < old_state_len {
                for _ in (final_state_cursor..old_state_len).rev() {
                    let st_idx = self.sys.slots[slot_idx as usize]
                        .as_mut()
                        .unwrap()
                        .local_states
                        .pop()
                        .unwrap();
                    self.sys.active_springs.retain(|&sp_idx| sp_idx != st_idx);
                    let _dropped = self.sys.state_arena[st_idx as usize].take();
                    self.sys.free_states.push(st_idx);
                }
            }
            let (ab, is_focusable, now_scrollable) = {
                let slot = self.sys.slots[slot_idx as usize].as_mut().unwrap();
                let ab = slot.active_bindings;
                if (ab & BIND_CLICK) == 0 {
                    slot.on_click = None;
                }
                if (ab & BIND_DRAG) == 0 {
                    slot.on_drag = None;
                }
                if (ab & BIND_KEY) == 0 {
                    slot.on_key = None;
                }
                if (ab & BIND_SCROLL) == 0 {
                    slot.on_scroll = None;
                }
                if (ab & BIND_ACTION) == 0 {
                    slot.actions.clear();
                }
                (ab, slot.focusable, slot.scrollable_max_y.is_some())
            };
            if was_scrollable && !now_scrollable {
                self.sys.layout.topo[node_id.idx()].flags &= !crate::layout::FLAG_SCROLLABLE;
                let mut cur_sp = self.sys.layout.specs[node_id.idx()];
                cur_sp.scroll_y = spec.scroll_y;
                self.sys.layout.update_spec(node_id, cur_sp);
                self.sys.scene_topology_changed = true;
            }
            if self.sys.focused_slot == Some(slot_idx) && !is_focusable {
                self.sys.focused_slot = None;
            }
            if self.sys.captured_slot == Some(slot_idx) && (ab & (BIND_CLICK | BIND_DRAG)) == 0 {
                self.sys.captured_slot = None;
            }
        }

        let old_len = self.sys.slots[slot_idx as usize]
            .as_ref()
            .unwrap()
            .children
            .len();
        if final_cursor < old_len {
            struct_changed = true;
            for _ in (final_cursor..old_len).rev() {
                let stale = self.sys.slots[slot_idx as usize]
                    .as_mut()
                    .unwrap()
                    .children
                    .pop()
                    .unwrap();
                self.sys.unmount_slot(stale);
            }
        }

        if struct_changed {
            let mut tmp = core::mem::take(&mut self.sys.node_id_scratch);
            tmp.clear();
            for &cs in &self.sys.slots[slot_idx as usize].as_ref().unwrap().children {
                tmp.push(self.sys.slots[cs as usize].as_ref().unwrap().node_id);
            }
            self.sys.layout.set_children(node_id, &tmp);
            tmp.clear();
            self.sys.node_id_scratch = tmp;
            self.sys.scene_topology_changed = true;
            if self.sys.a11y_enabled
                && self.sys.a11y_initialized
                && !self.sys.dirty_a11y_slots.contains(&slot_idx)
            {
                self.sys.dirty_a11y_slots.push(slot_idx);
            }
        }

        node_id
    }

    /// Override `PaintStyle` inside an element closure (ideal for applying `is_hovered()` / `is_focused()` styles!).
    #[inline]
    pub fn set_paint(&mut self, paint: PaintStyle) {
        let s_idx = self.active_slot.expect("inside element");
        let nid = self.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        slot.paint = paint;
        if slot.paint != slot.committed_paint
            && (self.sys.layout.topo[nid.idx()].flags & FLAG_DIRTY_PAINT) == 0
        {
            self.sys.layout.topo[nid.idx()].flags |= FLAG_DIRTY_PAINT;
            self.sys.dirty_paint_slots.push(s_idx);
        }
    }

    #[inline]
    pub fn set_cursor(&mut self, cursor: CursorIcon) {
        let s_idx = self.active_slot.expect("inside element");
        self.sys.slots[s_idx as usize].as_mut().unwrap().cursor = Some(cursor);
    }

    #[track_caller]
    pub fn text(
        &mut self,
        content: &str,
        font_id: FontId,
        font_size: f32,
        line_h: f32,
        color: Color,
        aa_mode: AntialiasMode,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        self.element_internal(caller, LayoutSpec::default(), PaintStyle::default(), |ui| {
            let s_idx = ui.active_slot.unwrap();
            let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
            {
                let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
                if slot.text_color != color || slot.aa_mode != aa_mode {
                    slot.text_color = color;
                    slot.aa_mode = aa_mode;
                    ui.sys.scene_topology_changed = true;
                }
            }
            let prev_hash = (ui.sys.layout.topo[nid.idx()].flags & crate::layout::FLAG_HAS_TEXT)
                .ne(&0)
                .then(|| ui.sys.layout.texts[ui.sys.layout.specs[nid.idx()].aux_id as usize].content_hash);
            ui.sys
                .layout
                .attach_shaped_text(nid, content, font_id, font_size, line_h, &ui.sys.fonts);
            let new_hash =
                ui.sys.layout.texts[ui.sys.layout.specs[nid.idx()].aux_id as usize].content_hash;
            if prev_hash != Some(new_hash) && color.channels()[3] > 0 {
                ui.sys.scene_topology_changed = true;
            }
        })
    }

    /// Fast layout-measured label (`text_color = Color::TRANSPARENT` if purely structural, or visible `color`).
    #[track_caller]
    pub fn label(&mut self, content: &str, font_size: f32, line_h: f32, color: Color) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        self.element_internal(caller, LayoutSpec::default(), PaintStyle::default(), |ui| {
            let s_idx = ui.active_slot.unwrap();
            let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
            ui.sys.slots[s_idx as usize].as_mut().unwrap().text_color = color;
            ui.sys.layout.attach_shaped_text(
                nid,
                content,
                FontId::FALLBACK,
                font_size,
                line_h,
                &ui.sys.fonts,
            );
        })
    }

    /// Zero-allocation numeric label powered by Rust 1.98.0 `NumBuffer::format_into`!
    #[track_caller]
    pub fn label_usize(&mut self, value: usize, font_size: f32, line_h: f32, color: Color) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let mut buf = NumBuffer::new();
        let s: &str = value.format_into(&mut buf);
        self.element_internal(caller, LayoutSpec::default(), PaintStyle::default(), |ui| {
            let s_idx = ui.active_slot.unwrap();
            let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
            ui.sys.slots[s_idx as usize].as_mut().unwrap().text_color = color;
            ui.sys.layout.attach_shaped_text(
                nid,
                s,
                FontId::FALLBACK,
                font_size,
                line_h,
                &ui.sys.fonts,
            );
        })
    }

    /// Render crisp ClearType UI typography using `sys.fonts.ui_font` (auto-falling back if no TTF loaded).
    #[track_caller]
    pub fn ui_text(&mut self, content: &str, font_size: f32, color: Color) -> NodeId {
        let fid = self.sys.fonts.ui_font;
        let aa = if fid == FontId::FALLBACK {
            AntialiasMode::Grayscale
        } else {
            AntialiasMode::SubpixelLcd
        };
        self.text(content, fid, font_size, (font_size * 1.35).round(), color, aa)
    }

    /// Render crisp monospace code/telemetry typography using `sys.fonts.mono_font`.
    #[track_caller]
    pub fn mono_text(&mut self, content: &str, font_size: f32, color: Color) -> NodeId {
        let fid = self.sys.fonts.mono_font;
        let aa = if fid == FontId::FALLBACK {
            AntialiasMode::Grayscale
        } else {
            AntialiasMode::SubpixelLcd
        };
        self.text(content, fid, font_size, (font_size * 1.35).round(), color, aa)
    }

    #[track_caller]
    pub fn heading(&mut self, content: &str) -> NodeId {
        let col = self.sys.theme.text_primary;
        self.ui_text(content, 20.0, col)
    }

    /// Render a crisp cached analytical `Path2D` vector icon/shape!
    #[track_caller]
    pub fn vector_icon(
        &mut self,
        width: u16,
        height: u16,
        tint: Color,
        rule: FillRule,
        build_path: impl FnOnce(&mut Path2D),
    ) -> NodeId {
        self.vector_canvas(width, height, 1, tint, rule, build_path)
    }

    /// Render a dynamically updatable `Path2D` canvas (re-rasterizes in-place whenever `revision_tag` changes!).
    #[track_caller]
    pub fn vector_canvas(
        &mut self,
        width: u16,
        height: u16,
        revision_tag: u64,
        tint: Color,
        rule: FillRule,
        build_path: impl FnOnce(&mut Path2D),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let sf = self.sys.scale_factor;
        self.element_internal(
            caller,
            LayoutSpec::default().fixed(width as f32, height as f32),
            PaintStyle::default(),
            |ui| {
                let s_idx = ui.active_slot.unwrap();
                let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
                let pw = ((width as f32) * sf).round().max(1.0) as u16;
                let ph = ((height as f32) * sf).round().max(1.0) as u16;
                let cur_gen = ui.sys.fonts.atlas.generation();
                let tag_key = revision_tag ^ (sf.to_bits() as u64);
                let (prev_reg, prev_gen, prev_tag) = {
                    let slot = ui.sys.slots[s_idx as usize].as_ref().unwrap();
                    (slot.custom_atlas_region, slot.custom_atlas_gen, slot.custom_upload_tag)
                };
                let needs_upload =
                    prev_reg.is_none() || prev_gen != cur_gen || prev_tag != tag_key;
                if needs_upload {
                    {
                        let sys = &mut *ui.sys;
                        sys.vector_path_scratch.clear();
                        build_path(&mut sys.vector_path_scratch);
                        if (sf - 1.0).abs() > 1e-4 {
                            sys.vector_path_scratch.scale(sf, sf);
                        }
                        sys.vector_path_scratch.rasterize_into(
                            pw,
                            ph,
                            rule,
                            &mut sys.vector_raster_scratch,
                            &mut sys.vector_mask_scratch,
                        );
                    }
                    if let Some(reg) = prev_reg
                        && prev_gen == cur_gen
                        && reg.w == pw
                        && reg.h == ph
                    {
                        // Fast path: overwrite existing R8 atlas region in place (0 B new shelf usage, 0 B heap!)
                        {
                            let sys = &mut *ui.sys;
                            sys.fonts
                                .atlas
                                .update_r8_in_place(reg, &sys.vector_mask_scratch);
                        }
                        let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
                        slot.custom_upload_tag = tag_key;
                        slot.text_color = tint;
                        slot.custom_dirty = true;
                        if (ui.sys.layout.topo[nid.idx()].flags & FLAG_DIRTY_PAINT) == 0 {
                            ui.sys.layout.topo[nid.idx()].flags |= FLAG_DIRTY_PAINT;
                            ui.sys.dirty_paint_slots.push(s_idx);
                        }
                    } else {
                        let reg = {
                            let sys = &mut *ui.sys;
                            sys.fonts
                                .atlas
                                .upload_r8(pw, ph, 0, 0, &sys.vector_mask_scratch)
                        };
                        let new_gen = ui.sys.fonts.atlas.generation();
                        let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
                        slot.custom_atlas_region = reg;
                        slot.custom_atlas_gen = new_gen;
                        slot.custom_upload_tag = tag_key;
                        slot.text_color = tint;
                        ui.sys.scene_topology_changed = true;
                    }
                } else {
                    let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
                    if slot.text_color != tint {
                        slot.text_color = tint;
                        slot.custom_dirty = true;
                        if (ui.sys.layout.topo[nid.idx()].flags & FLAG_DIRTY_PAINT) == 0 {
                            ui.sys.layout.topo[nid.idx()].flags |= FLAG_DIRTY_PAINT;
                            ui.sys.dirty_paint_slots.push(s_idx);
                        }
                    }
                }
            },
        )
    }

    /// Render a cached RGBA8 bitmap image with optional rounded SDF corners!
    #[track_caller]
    pub fn image_rgba(
        &mut self,
        revision_tag: u64,
        width: u16,
        height: u16,
        rgba_pixels: &[u8],
        corner_radius: f32,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        self.element_internal(
            caller,
            LayoutSpec::default().fixed(width as f32, height as f32),
            PaintStyle::default().rounded(corner_radius),
            |ui| {
                if corner_radius > 0.0 {
                    ui.set_clips_children(true);
                }
                let s_idx = ui.active_slot.unwrap();
                let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
                let cur_gen = ui.sys.fonts.atlas.generation();
                let (prev_reg, prev_gen, prev_tag) = {
                    let slot = ui.sys.slots[s_idx as usize].as_ref().unwrap();
                    (slot.custom_atlas_region, slot.custom_atlas_gen, slot.custom_upload_tag)
                };
                let needs_upload =
                    prev_reg.is_none() || prev_gen != cur_gen || prev_tag != revision_tag;
                if needs_upload && rgba_pixels.len() >= (width as usize) * (height as usize) * 4 {
                    if let Some(reg) = prev_reg
                        && prev_gen == cur_gen
                        && reg.w == width
                        && reg.h == height
                    {
                        ui.sys.fonts.atlas.update_rgba8_in_place(reg, rgba_pixels);
                        let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
                        slot.custom_upload_tag = revision_tag;
                        slot.text_color = Color::WHITE;
                        slot.custom_dirty = true;
                        if (ui.sys.layout.topo[nid.idx()].flags & FLAG_DIRTY_PAINT) == 0 {
                            ui.sys.layout.topo[nid.idx()].flags |= FLAG_DIRTY_PAINT;
                            ui.sys.dirty_paint_slots.push(s_idx);
                        }
                    } else {
                        let reg = ui
                            .sys
                            .fonts
                            .atlas
                            .upload_rgba8(width, height, 0, 0, rgba_pixels);
                        let new_gen = ui.sys.fonts.atlas.generation();
                        let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
                        slot.custom_atlas_region = reg;
                        slot.custom_atlas_gen = new_gen;
                        slot.custom_upload_tag = revision_tag;
                        slot.text_color = Color::WHITE;
                        ui.sys.scene_topology_changed = true;
                    }
                }
            },
        )
    }

    pub fn set_clips_children(&mut self, clips: bool) {
        let s_idx = self.active_slot.expect("inside element");
        let nid = self.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
        let prev = (self.sys.layout.topo[nid.idx()].flags & FLAG_CLIPS_CHILDREN) != 0;
        if prev != clips {
            if clips {
                self.sys.layout.topo[nid.idx()].flags |= FLAG_CLIPS_CHILDREN;
            } else {
                self.sys.layout.topo[nid.idx()].flags &= !FLAG_CLIPS_CHILDREN;
            }
            self.sys.scene_topology_changed = true;
        }
    }

    pub fn set_scrollable(&mut self, max_scroll_y: f32) {
        let s_idx = self.active_slot.expect("inside element");
        let nid = self.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
        let safe_max = if max_scroll_y.is_finite() {
            max_scroll_y.max(0.0)
        } else {
            0.0
        };
        self.sys.layout.topo[nid.idx()].flags |= crate::layout::FLAG_SCROLLABLE;
        self.set_clips_children(true);
        self.sys.slots[s_idx as usize]
            .as_mut()
            .unwrap()
            .scrollable_max_y = Some(safe_max);
        let cur_sy = self.sys.layout.specs[nid.idx()].scroll_y;
        let clamped_sy = cur_sy.clamp(0.0, safe_max);
        if (clamped_sy - cur_sy).abs() > 1e-5 {
            let mut sp = self.sys.layout.specs[nid.idx()];
            sp.scroll_y = clamped_sy;
            self.sys.layout.update_spec(nid, sp);
            self.sys.scene_topology_changed = true;
        }
    }

    /// Attach a custom wheel/touchpad scroll handler (ideal for virtualized `VirtualListIndex` views!).
    pub fn on_scroll<F>(&mut self, handler: F)
    where
        F: Fn(&mut S, f32) -> bool + 'static,
    {
        unsafe fn invoke_scroll<S: 'static, F: Fn(&mut S, f32) -> bool + 'static>(
            state_ptr: *mut (),
            handler_slot: u32,
            arena: &mut [Option<Box<dyn Any>>],
            wheel_dy: f32,
        ) -> bool {
            let s = unsafe { &mut *state_ptr.cast::<S>() };
            let f = arena[handler_slot as usize]
                .as_ref()
                .unwrap()
                .downcast_ref::<F>()
                .unwrap();
            f(s, wheel_dy)
        }

        let h = self.use_state_updated(handler);
        let s_idx = self.active_slot.expect("inside element");
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        slot.active_bindings |= BIND_SCROLL;
        if let Some(existing) = &mut slot.on_scroll {
            sync_lens_steps(&mut existing.steps, &self.sys.lens_scratch);
            existing.handler_slot = h.arena_idx;
            existing.invoke = invoke_scroll::<S, F>;
        } else {
            slot.on_scroll = Some(ScrollBinding {
                steps: self.sys.lens_scratch.as_slice().into(),
                handler_slot: h.arena_idx,
                invoke: invoke_scroll::<S, F>,
            });
        }
    }

    pub fn set_focusable(&mut self, focusable: bool) {
        let s_idx = self.active_slot.expect("inside element");
        self.sys.slots[s_idx as usize].as_mut().unwrap().focusable = focusable;
    }

    pub fn set_semantics(&mut self, sem: SemanticsMeta) {
        let s_idx = self.active_slot.expect("inside element");
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        if slot.semantics.as_ref() != Some(&sem) {
            slot.semantics = Some(sem);
            if self.sys.a11y_enabled && !self.sys.dirty_a11y_slots.contains(&s_idx) {
                self.sys.dirty_a11y_slots.push(s_idx);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sync_semantics_in_place(
        &mut self,
        role: A11yRole,
        label: Option<&str>,
        value: Option<&str>,
        toggled: Option<crate::a11y::Toggled>,
        numeric_value: Option<f64>,
        min_numeric_value: Option<f64>,
        max_numeric_value: Option<f64>,
        numeric_step: Option<f64>,
        clickable: bool,
        focusable: bool,
        incrementable: bool,
    ) {
        let s_idx = self.active_slot.expect("inside element");
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        let changed = if let Some(sem) = &mut slot.semantics {
            let mut diff = false;
            if sem.role != role {
                sem.role = role;
                diff = true;
            }
            match (&mut sem.label, label) {
                (Some(dst), Some(src)) => {
                    if dst.as_str() != src {
                        dst.clear();
                        dst.push_str(src);
                        diff = true;
                    }
                }
                (None, None) => {}
                (dst, src) => {
                    *dst = src.map(str::to_owned);
                    diff = true;
                }
            }
            match (&mut sem.value, value) {
                (Some(dst), Some(src)) => {
                    if dst.as_str() != src {
                        dst.clear();
                        dst.push_str(src);
                        diff = true;
                    }
                }
                (None, None) => {}
                (dst, src) => {
                    *dst = src.map(str::to_owned);
                    diff = true;
                }
            }
            if sem.toggled != toggled {
                sem.toggled = toggled;
                diff = true;
            }
            if sem.numeric_value != numeric_value {
                sem.numeric_value = numeric_value;
                diff = true;
            }
            if sem.min_numeric_value != min_numeric_value {
                sem.min_numeric_value = min_numeric_value;
                diff = true;
            }
            if sem.max_numeric_value != max_numeric_value {
                sem.max_numeric_value = max_numeric_value;
                diff = true;
            }
            if sem.numeric_step != numeric_step {
                sem.numeric_step = numeric_step;
                diff = true;
            }
            if sem.clickable != clickable {
                sem.clickable = clickable;
                diff = true;
            }
            if sem.focusable != focusable {
                sem.focusable = focusable;
                diff = true;
            }
            if sem.incrementable != incrementable {
                sem.incrementable = incrementable;
                diff = true;
            }
            diff
        } else {
            slot.semantics = Some(SemanticsMeta {
                role,
                label: label.map(str::to_owned),
                value: value.map(str::to_owned),
                numeric_value,
                min_numeric_value,
                max_numeric_value,
                numeric_step,
                toggled,
                clickable,
                focusable,
                incrementable,
            });
            true
        };
        if changed && self.sys.a11y_enabled && !self.sys.dirty_a11y_slots.contains(&s_idx) {
            self.sys.dirty_a11y_slots.push(s_idx);
        }
    }

    pub fn on_click<F>(&mut self, handler: F)
    where
        F: Fn(&mut S, &PointerEvent) + 'static,
    {
        self.bind_pointer(true, handler);
    }

    pub fn on_drag<F>(&mut self, handler: F)
    where
        F: Fn(&mut S, &PointerEvent) + 'static,
    {
        self.bind_pointer(false, handler);
    }

    fn bind_pointer<F>(&mut self, is_click: bool, handler: F)
    where
        F: Fn(&mut S, &PointerEvent) + 'static,
    {
        unsafe fn invoke_zst<S: 'static, F: Fn(&mut S, &PointerEvent) + 'static>(
            state_ptr: *mut (),
            _handler_slot: u32,
            _arena: &mut [Option<Box<dyn Any>>],
            _fonts: &FontSystem,
            _layout: &LayoutTree,
            ev: &PointerEvent,
        ) {
            let s = unsafe { &mut *state_ptr.cast::<S>() };
            let f: F = unsafe { core::mem::conjure_zst::<F>() };
            f(s, ev);
        }

        unsafe fn invoke_captured<S: 'static, F: Fn(&mut S, &PointerEvent) + 'static>(
            state_ptr: *mut (),
            handler_slot: u32,
            arena: &mut [Option<Box<dyn Any>>],
            _fonts: &FontSystem,
            _layout: &LayoutTree,
            ev: &PointerEvent,
        ) {
            let s = unsafe { &mut *state_ptr.cast::<S>() };
            let f = arena[handler_slot as usize]
                .as_ref()
                .unwrap()
                .downcast_ref::<F>()
                .unwrap();
            f(s, ev);
        }

        let s_idx = self.active_slot.expect("inside element");
        let (h_idx, inv): (u32, ErasedPointerFn) = if core::mem::size_of::<F>() == 0 {
            (u32::MAX, invoke_zst::<S, F>)
        } else {
            let h = self.use_state_updated(handler);
            (h.arena_idx, invoke_captured::<S, F>)
        };

        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        slot.active_bindings |= if is_click { BIND_CLICK } else { BIND_DRAG };
        let target = if is_click {
            &mut slot.on_click
        } else {
            &mut slot.on_drag
        };
        if let Some(existing) = target {
            sync_lens_steps(&mut existing.steps, &self.sys.lens_scratch);
            existing.handler_slot = h_idx;
            existing.invoke = inv;
        } else {
            *target = Some(PointerBinding {
                steps: self.sys.lens_scratch.as_slice().into(),
                handler_slot: h_idx,
                invoke: inv,
            });
        }
    }

    pub fn on_click_with_local<L: 'static, F>(&mut self, local: LocalHandle<L>, handler: F)
    where
        F: Fn(&mut S, &mut L, &PointerEvent) + 'static,
    {
        self.bind_pointer_with_local(true, local, handler);
    }

    pub fn on_drag_with_local<L: 'static, F>(&mut self, local: LocalHandle<L>, handler: F)
    where
        F: Fn(&mut S, &mut L, &PointerEvent) + 'static,
    {
        self.bind_pointer_with_local(false, local, handler);
    }

    fn bind_pointer_with_local<L: 'static, F>(
        &mut self,
        is_click: bool,
        local: LocalHandle<L>,
        handler: F,
    ) where
        F: Fn(&mut S, &mut L, &PointerEvent) + 'static,
    {
        unsafe fn invoke_both<
            S: 'static,
            L: 'static,
            F: Fn(&mut S, &mut L, &PointerEvent) + 'static,
        >(
            state_ptr: *mut (),
            handler_slot: u32,
            arena: &mut [Option<Box<dyn Any>>],
            _fonts: &FontSystem,
            _layout: &LayoutTree,
            ev: &PointerEvent,
        ) {
            let s = unsafe { &mut *state_ptr.cast::<S>() };
            let local_idx = arena[handler_slot as usize]
                .as_ref()
                .unwrap()
                .downcast_ref::<(u32, F)>()
                .unwrap()
                .0;
            let [h_opt, l_opt] = arena
                .get_disjoint_mut([handler_slot as usize, local_idx as usize])
                .expect("disjoint handler and local state slots");
            let (_, f) = h_opt.as_ref().unwrap().downcast_ref::<(u32, F)>().unwrap();
            let l_mut = l_opt.as_mut().unwrap().downcast_mut::<L>().unwrap();
            f(s, l_mut, ev);
        }

        let pack = self.use_state_updated((local.arena_idx, handler));
        let s_idx = self.active_slot.expect("inside element");
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        slot.active_bindings |= if is_click { BIND_CLICK } else { BIND_DRAG };
        let target = if is_click {
            &mut slot.on_click
        } else {
            &mut slot.on_drag
        };
        if let Some(existing) = target {
            sync_lens_steps(&mut existing.steps, &self.sys.lens_scratch);
            existing.handler_slot = pack.arena_idx;
            existing.invoke = invoke_both::<S, L, F>;
        } else {
            *target = Some(PointerBinding {
                steps: self.sys.lens_scratch.as_slice().into(),
                handler_slot: pack.arena_idx,
                invoke: invoke_both::<S, L, F>,
            });
        }
    }

    fn bind_pointer_with_local_and_fonts<L: 'static, F>(
        &mut self,
        is_click: bool,
        local: LocalHandle<L>,
        handler: F,
    ) where
        F: Fn(&mut S, &mut L, &FontSystem, &LayoutTree, &PointerEvent) + 'static,
    {
        unsafe fn invoke_with_fonts<
            S: 'static,
            L: 'static,
            F: Fn(&mut S, &mut L, &FontSystem, &LayoutTree, &PointerEvent) + 'static,
        >(
            state_ptr: *mut (),
            handler_slot: u32,
            arena: &mut [Option<Box<dyn Any>>],
            fonts: &FontSystem,
            layout: &LayoutTree,
            ev: &PointerEvent,
        ) {
            let s = unsafe { &mut *state_ptr.cast::<S>() };
            let local_idx = arena[handler_slot as usize]
                .as_ref()
                .unwrap()
                .downcast_ref::<(u32, F)>()
                .unwrap()
                .0;
            let [h_opt, l_opt] = arena
                .get_disjoint_mut([handler_slot as usize, local_idx as usize])
                .expect("disjoint handler and local state slots");
            let (_, f) = h_opt.as_ref().unwrap().downcast_ref::<(u32, F)>().unwrap();
            let l_mut = l_opt.as_mut().unwrap().downcast_mut::<L>().unwrap();
            f(s, l_mut, fonts, layout, ev);
        }

        let pack = self.use_state_updated((local.arena_idx, handler));
        let s_idx = self.active_slot.expect("inside element");
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        slot.active_bindings |= if is_click { BIND_CLICK } else { BIND_DRAG };
        let target = if is_click {
            &mut slot.on_click
        } else {
            &mut slot.on_drag
        };
        if let Some(existing) = target {
            sync_lens_steps(&mut existing.steps, &self.sys.lens_scratch);
            existing.handler_slot = pack.arena_idx;
            existing.invoke = invoke_with_fonts::<S, L, F>;
        } else {
            *target = Some(PointerBinding {
                steps: self.sys.lens_scratch.as_slice().into(),
                handler_slot: pack.arena_idx,
                invoke: invoke_with_fonts::<S, L, F>,
            });
        }
    }

    pub fn on_key<F>(&mut self, handler: F)
    where
        F: Fn(&mut S, &KeyInput) -> bool + 'static,
    {
        let dummy = self.use_state(|| ());
        self.on_key_with_local(dummy, move |s, (), k| handler(s, k));
    }

    pub fn on_key_with_local<L: 'static, F>(&mut self, local: LocalHandle<L>, handler: F)
    where
        F: Fn(&mut S, &mut L, &KeyInput) -> bool + 'static,
    {
        self.bind_key_with_local_and_clipboard(local, move |s, l, _clip, k| handler(s, l, k));
    }

    fn bind_key_with_local_and_clipboard<L: 'static, F>(&mut self, local: LocalHandle<L>, handler: F)
    where
        F: Fn(&mut S, &mut L, &mut String, &KeyInput) -> bool + 'static,
    {
        unsafe fn invoke_key<
            S: 'static,
            L: 'static,
            F: Fn(&mut S, &mut L, &mut String, &KeyInput) -> bool + 'static,
        >(
            state_ptr: *mut (),
            handler_slot: u32,
            arena: &mut [Option<Box<dyn Any>>],
            clipboard: &mut String,
            key: &KeyInput,
        ) -> bool {
            let s = unsafe { &mut *state_ptr.cast::<S>() };
            let local_idx = arena[handler_slot as usize]
                .as_ref()
                .unwrap()
                .downcast_ref::<(u32, F)>()
                .unwrap()
                .0;
            let [h_opt, l_opt] = arena
                .get_disjoint_mut([handler_slot as usize, local_idx as usize])
                .unwrap();
            let (_, f) = h_opt.as_ref().unwrap().downcast_ref::<(u32, F)>().unwrap();
            let l_mut = l_opt.as_mut().unwrap().downcast_mut::<L>().unwrap();
            f(s, l_mut, clipboard, key)
        }

        let pack = self.use_state_updated((local.arena_idx, handler));
        let s_idx = self.active_slot.expect("inside element");
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        slot.active_bindings |= BIND_KEY;
        if let Some(existing) = &mut slot.on_key {
            sync_lens_steps(&mut existing.steps, &self.sys.lens_scratch);
            existing.handler_slot = pack.arena_idx;
            existing.invoke = invoke_key::<S, L, F>;
        } else {
            slot.on_key = Some(KeyBinding {
                steps: self.sys.lens_scratch.as_slice().into(),
                handler_slot: pack.arena_idx,
                invoke: invoke_key::<S, L, F>,
            });
        }
    }

    /// Register a typed `Action` handler that bubbles up from the focused widget through ancestors!
    pub fn on_action<A: 'static>(&mut self, handler: fn(&mut S, &A) -> bool) {
        unsafe fn invoke_act<S: 'static, A: 'static>(
            state_ptr: *mut (),
            fn_addr: usize,
            any_act: &dyn Any,
        ) -> bool {
            let s = unsafe { &mut *state_ptr.cast::<S>() };
            let f: fn(&mut S, &A) -> bool = unsafe { core::mem::transmute(fn_addr) };
            let act = any_act.downcast_ref::<A>().expect("matched Action TypeId");
            f(s, act)
        }

        let s_idx = self.active_slot.expect("inside element");
        let slot = self.sys.slots[s_idx as usize].as_mut().unwrap();
        if (slot.active_bindings & BIND_ACTION) == 0 {
            slot.actions.clear();
            slot.active_bindings |= BIND_ACTION;
        }
        let tid = TypeId::of::<A>();
        if let Some(existing) = slot.actions.iter_mut().find(|a| a.action_type == tid) {
            sync_lens_steps(&mut existing.steps, &self.sys.lens_scratch);
            existing.handler_addr = handler as usize;
            existing.invoke = invoke_act::<S, A>;
        } else {
            slot.actions.push(ActionBinding {
                action_type: tid,
                steps: self.sys.lens_scratch.as_slice().into(),
                handler_addr: handler as usize,
                invoke: invoke_act::<S, A>,
            });
        }
    }

    pub fn read_pool<'p, T: 'static>(
        &mut self,
        pool: &'p mut SlotPool<T>,
        key: Key<T>,
        scope_id: ScopeId,
    ) -> Option<&'p T> {
        pool.read_tracked(key, scope_id).map(|(v, _)| v)
    }

    // =========================================================================
    // Composable 1.0 Standard Controls (Button, Checkbox, ToggleSwitch, Slider, TextInput, Progress, Badge, ScrollArea, VirtualList)
    // =========================================================================

    #[track_caller]
    fn single_line_ui_text(&mut self, content: &str, font_size: f32, color: Color) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let fid = self.sys.fonts.ui_font;
        let aa = if fid == FontId::FALLBACK {
            AntialiasMode::Grayscale
        } else {
            AntialiasMode::SubpixelLcd
        };
        let line_h = (font_size * 1.35).round();
        let spec = LayoutSpec {
            shrink_q8: 0, // Never shrink or multi-line wrap inside single-line controls!
            ..LayoutSpec::default()
        };
        self.element_internal(caller, spec, PaintStyle::default(), |ui| {
            let s_idx = ui.active_slot.unwrap();
            let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
            {
                let slot = ui.sys.slots[s_idx as usize].as_mut().unwrap();
                if slot.text_color != color || slot.aa_mode != aa {
                    slot.text_color = color;
                    slot.aa_mode = aa;
                    ui.sys.scene_topology_changed = true;
                }
            }
            let prev_hash = (ui.sys.layout.topo[nid.idx()].flags & crate::layout::FLAG_HAS_TEXT)
                .ne(&0)
                .then(|| ui.sys.layout.texts[ui.sys.layout.specs[nid.idx()].aux_id as usize].content_hash);
            ui.sys
                .layout
                .attach_shaped_text(nid, content, fid, font_size, line_h, &ui.sys.fonts);
            let new_hash =
                ui.sys.layout.texts[ui.sys.layout.specs[nid.idx()].aux_id as usize].content_hash;
            if prev_hash != Some(new_hash) && color.channels()[3] > 0 {
                ui.sys.scene_topology_changed = true;
            }
        })
    }

    #[track_caller]
    pub fn button(
        &mut self,
        label_str: &str,
        bg: Color,
        on_press: impl Fn(&mut S, &PointerEvent) + 'static,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let th = self.sys.theme;
        let base_border = if OklabLuma::of(bg) > 0.78 {
            th.border_subtle.mix_oklab(bg.shift_lightness(-0.14), 0.45)
        } else {
            bg.shift_lightness(if th.is_dark { 0.10 } else { -0.08 })
        };
        self.element_internal(
            caller,
            LayoutSpec::row().pad_xy(14.0, 7.0).align(Align::Center),
            PaintStyle::solid(bg)
                .rounded(th.radius_sm)
                .border(1.0, base_border),
            |ui| {
                ui.set_focusable(true);
                ui.set_cursor(CursorIcon::Pointer);
                ui.sync_semantics_in_place(
                    A11yRole::Button,
                    Some(label_str),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    true,
                    true,
                    false,
                );
                ui.on_click(on_press);

                let captured = ui.is_captured();
                let hovered = ui.is_hovered();
                let focused = ui.is_focused();
                let eff_bg = if captured {
                    bg.shift_lightness(if th.is_dark { -0.05 } else { -0.09 })
                } else if hovered {
                    bg.shift_lightness(if th.is_dark { 0.06 } else { -0.05 })
                } else {
                    bg
                };
                let rest_border = if OklabLuma::of(eff_bg) > 0.78 {
                    th.border_subtle
                        .mix_oklab(eff_bg.shift_lightness(-0.14), 0.45)
                } else {
                    eff_bg.shift_lightness(if th.is_dark { 0.10 } else { -0.08 })
                };
                let (bw, bc) = if focused {
                    (2.0, th.border_focus)
                } else {
                    (1.0, rest_border)
                };
                ui.set_paint(PaintStyle::solid(eff_bg).rounded(th.radius_sm).border(bw, bc));
                let luma = OklabLuma::of(bg);
                let fg = if luma > 0.72 {
                    Color::rgb8(18, 20, 28)
                } else {
                    Color::WHITE
                };
                ui.ui_text(label_str, 13.5, fg);
            },
        )
    }

    #[track_caller]
    pub fn primary_button(
        &mut self,
        label_str: &str,
        on_press: impl Fn(&mut S, &PointerEvent) + 'static,
    ) -> NodeId {
        let accent = self.sys.theme.accent;
        self.button(label_str, accent, on_press)
    }

    #[track_caller]
    pub fn checkbox(
        &mut self,
        label_str: &str,
        checked: bool,
        on_toggle: impl Fn(&mut S, &PointerEvent) + 'static,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let th = self.sys.theme;
        self.element_internal(
            caller,
            LayoutSpec::row().gap(9.0).align(Align::Center),
            PaintStyle::default(),
            |ui| {
                ui.set_focusable(true);
                ui.set_cursor(CursorIcon::Pointer);
                ui.sync_semantics_in_place(
                    A11yRole::CheckBox,
                    Some(label_str),
                    None,
                    Some(if checked {
                        crate::a11y::Toggled::True
                    } else {
                        crate::a11y::Toggled::False
                    }),
                    None,
                    None,
                    None,
                    None,
                    true,
                    true,
                    false,
                );
                ui.on_click(on_toggle);
                let captured = ui.is_captured();
                let hovered = ui.is_hovered();
                let focused = ui.is_focused();

                let box_bg = if checked {
                    if captured {
                        th.accent.shift_lightness(-0.06)
                    } else if hovered {
                        th.accent_hover
                    } else {
                        th.accent
                    }
                } else if captured {
                    th.bg_control_hover
                        .shift_lightness(if th.is_dark { -0.03 } else { -0.05 })
                } else if hovered {
                    th.bg_control_hover
                } else {
                    th.bg_control
                };
                let border_c = if focused {
                    th.border_focus
                } else if checked {
                    box_bg.shift_lightness(if th.is_dark { 0.08 } else { -0.06 })
                } else {
                    th.border_subtle
                };

                ui.element(
                    LayoutSpec::col()
                        .fixed(18.0, 18.0)
                        .align(Align::Center)
                        .justify(crate::layout::MainAlign::Center),
                    PaintStyle::solid(box_bg)
                        .rounded(4.5)
                        .border(if focused { 2.0 } else { 1.0 }, border_c),
                    |ui| {
                        if checked {
                            // Single closed non-self-intersecting smooth Wang-Bézier checkmark
                            // with rounded endcaps & rounded knee join (0 seam artifacts at any HiDPI!):
                            ui.vector_icon(12, 12, th.text_on_accent, FillRule::NonZero, |p| {
                                p.move_to(2.85, 5.55)
                                    .line_to(4.78, 7.42)
                                    .line_to(9.20, 2.45)
                                    .quad_to(10.02, 1.72, 10.68, 2.38)
                                    .quad_to(11.24, 2.98, 10.68, 3.68)
                                    .line_to(5.54, 9.52)
                                    .quad_to(4.80, 10.28, 4.04, 9.54)
                                    .line_to(1.48, 7.00)
                                    .quad_to(0.86, 6.32, 1.50, 5.62)
                                    .quad_to(2.16, 4.94, 2.85, 5.55)
                                    .close();
                            });
                        }
                    },
                );
                ui.ui_text(label_str, 13.5, th.text_primary);
            },
        )
    }

    /// Animated pill toggle switch powered by `use_spring`!
    #[track_caller]
    pub fn toggle_switch(
        &mut self,
        label_str: &str,
        enabled: bool,
        on_toggle: impl Fn(&mut S, &PointerEvent) + 'static,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let th = self.sys.theme;
        self.element_internal(
            caller,
            LayoutSpec::row().gap(10.0).align(Align::Center),
            PaintStyle::default(),
            |ui| {
                ui.set_focusable(true);
                ui.set_cursor(CursorIcon::Pointer);
                ui.sync_semantics_in_place(
                    A11yRole::Switch,
                    Some(label_str),
                    None,
                    Some(if enabled {
                        crate::a11y::Toggled::True
                    } else {
                        crate::a11y::Toggled::False
                    }),
                    None,
                    None,
                    None,
                    None,
                    true,
                    true,
                    false,
                );
                ui.on_click(on_toggle);
                let focused = ui.is_focused();
                let hovered = ui.is_hovered();
                let captured = ui.is_captured();
                let raw_t = ui.use_spring(if enabled { 1.0 } else { 0.0 }, 5.5);
                let t = raw_t.clamp(0.0, 1.0);
                let off_col = if hovered {
                    th.bg_control_hover
                } else {
                    th.bg_control
                };
                let on_col = if hovered { th.accent_hover } else { th.accent };
                let mut track_col = off_col.mix_oklab(on_col, t);
                if captured {
                    track_col = track_col.shift_lightness(if th.is_dark { -0.04 } else { -0.06 });
                }
                let border_col = if focused {
                    th.border_focus
                } else if enabled {
                    on_col.shift_lightness(if th.is_dark { 0.06 } else { -0.08 })
                } else {
                    th.border_subtle
                };

                let mut track_spec = LayoutSpec::row().fixed(38.0, 21.0).align(Align::Center);
                track_spec.pad_l = (3.0 + raw_t * 17.0).clamp(2.0, 21.0);
                ui.element(
                    track_spec,
                    PaintStyle::solid(track_col)
                        .rounded(10.5)
                        .border(if focused { 2.0 } else { 1.0 }, border_col),
                    |ui| {
                        ui.element(
                            LayoutSpec::col().fixed(15.0, 15.0),
                            PaintStyle::solid(Color::WHITE)
                                .rounded(7.5)
                                .drop_shadow(2.0, [0.0, 1.0], Color::rgba8(0, 0, 0, 95)),
                            |_| {},
                        );
                    },
                );
                ui.ui_text(label_str, 13.5, th.text_primary);
            },
        )
    }

    /// Interactive horizontal value slider with pointer capture (`on_drag`), click-to-seek,
    /// Left/Right/PageUp/PageDown/Home/End keyboard stepping, and `AccessKit` `Role::Slider` (`Increment`/`Decrement`)!
    #[track_caller]
    pub fn slider(
        &mut self,
        label_for_a11y: &str,
        value: f32,
        range: RangeInclusive<f32>,
        track_width: f32,
        on_change: impl Fn(&mut S, f32) + Copy + 'static,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let lo = *range.start();
        let hi = (*range.end()).max(lo + 1e-6);
        let span = hi - lo;
        let clamped_val = value.clamp(lo, hi);
        let frac = ((clamped_val - lo) / span).clamp(0.0, 1.0);
        let th = self.sys.theme;
        let thumb_d = 14.0_f32;
        let thumb_r = thumb_d * 0.5; // 7.0px

        self.element_internal(
            caller,
            LayoutSpec::row().fixed(track_width, 24.0).align(Align::Center),
            PaintStyle::default(),
            |ui| {
                ui.set_focusable(true);
                ui.set_cursor(CursorIcon::Grab);
                ui.sync_semantics_in_place(
                    A11yRole::Slider,
                    Some(label_for_a11y),
                    None,
                    None,
                    Some(clamped_val as f64),
                    Some(lo as f64),
                    Some(hi as f64),
                    Some((span * 0.02) as f64),
                    false,
                    true,
                    true,
                );

                let live_val = ui.use_state_updated(clamped_val);

                ui.on_click_with_local(live_val, move |s, lv, ev| {
                    let usable_w = (ev.target_w - thumb_d).max(1.0);
                    let t = ((ev.local_x - thumb_r) / usable_w).clamp(0.0, 1.0);
                    let nv = lo + t * span;
                    *lv = nv;
                    on_change(s, nv);
                });
                ui.on_drag_with_local(live_val, move |s, lv, ev| {
                    let usable_w = (ev.target_w - thumb_d).max(1.0);
                    let t = ((ev.local_x - thumb_r) / usable_w).clamp(0.0, 1.0);
                    let nv = lo + t * span;
                    *lv = nv;
                    on_change(s, nv);
                });
                ui.on_key_with_local(live_val, move |s, lv, k| {
                    let cur = (*lv).clamp(lo, hi);
                    let nv = match k {
                        KeyInput::ArrowLeft | KeyInput::ArrowDown => (cur - span * 0.02).clamp(lo, hi),
                        KeyInput::ArrowRight | KeyInput::ArrowUp => (cur + span * 0.02).clamp(lo, hi),
                        KeyInput::PageDown => (cur - span * 0.10).clamp(lo, hi),
                        KeyInput::PageUp => (cur + span * 0.10).clamp(lo, hi),
                        KeyInput::Home => lo,
                        KeyInput::End => hi,
                        _ => return false,
                    };
                    *lv = nv;
                    on_change(s, nv);
                    true
                });

                let focused = ui.is_focused();
                let captured = ui.is_captured();
                let hovered = ui.is_hovered() || captured;

                // Background rail + exact thumb-center-aligned active fill (`thumb_center == 7 + frac * (W - 14)`):
                ui.element(
                    LayoutSpec::row()
                        .fixed(track_width, 6.0)
                        .align(Align::Center),
                    PaintStyle::solid(th.bg_control)
                        .rounded(3.0)
                        .border(1.0, th.border_subtle),
                    |ui| {
                        let usable_w = (track_width - thumb_d).max(0.0);
                        let fill_w = (thumb_d + frac * usable_w).min(track_width);
                        ui.element(
                            LayoutSpec::row()
                                .fixed(fill_w, 6.0)
                                .justify(crate::layout::MainAlign::End)
                                .align(Align::Center),
                            PaintStyle::solid(if hovered { th.accent_hover } else { th.accent })
                                .rounded(3.0),
                            |ui| {
                                let thumb_border_w = if focused || captured { 2.5 } else { 1.5 };
                                let thumb_border_c = if focused {
                                    th.border_focus
                                } else if captured {
                                    th.accent_hover
                                } else {
                                    th.accent
                                };
                                ui.element(
                                    LayoutSpec::col().fixed(thumb_d, thumb_d),
                                    PaintStyle::solid(Color::WHITE)
                                        .rounded(thumb_r)
                                        .border(thumb_border_w, thumb_border_c)
                                        .drop_shadow(
                                            if captured { 4.0 } else { 3.0 },
                                            [0.0, 1.5],
                                            Color::rgba8(0, 0, 0, if th.is_dark { 120 } else { 85 }),
                                        ),
                                    |_| {},
                                );
                            },
                        );
                    },
                );
            },
        )
    }

    #[track_caller]
    pub fn progress_bar(&mut self, fraction: f32, width: f32, fill: Color) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let th = self.sys.theme;
        let clamped_f = fraction.clamp(0.0, 1.0);
        let anim_f = self.use_spring(clamped_f, 5.0).clamp(0.0, 1.0);
        self.element_internal(
            caller,
            LayoutSpec::row().fixed(width, 8.0),
            PaintStyle::solid(th.bg_control)
                .rounded(4.0)
                .border(0.75, th.border_subtle),
            |ui| {
                ui.set_clips_children(true);
                ui.sync_semantics_in_place(
                    A11yRole::ProgressIndicator,
                    None,
                    None,
                    None,
                    Some(clamped_f as f64),
                    Some(0.0),
                    Some(1.0),
                    None,
                    false,
                    false,
                    false,
                );
                ui.element(
                    LayoutSpec::row().fixed((anim_f * width).max(0.0), 8.0),
                    PaintStyle::solid(fill).rounded(4.0),
                    |_| {},
                );
            },
        )
    }

    #[track_caller]
    pub fn badge(&mut self, text_str: &str, color: Color) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let is_dark = self.sys.theme.is_dark;
        let bg_alpha = if is_dark { 0.17 } else { 0.13 };
        let border_alpha = if is_dark { 0.44 } else { 0.36 };
        let fg = if is_dark {
            color.shift_lightness(0.14)
        } else {
            color.shift_lightness(-0.15)
        };
        self.element_internal(
            caller,
            LayoutSpec::row().pad_xy(8.0, 3.0).align(Align::Center),
            PaintStyle::solid(color.with_alpha(bg_alpha))
                .rounded(99.0)
                .border(1.0, color.with_alpha(border_alpha)),
            |ui| {
                ui.ui_text(text_str, 11.5, fg);
            },
        )
    }

    #[track_caller]
    pub fn divider(&mut self) -> NodeId {
        let col = self.sys.theme.border_subtle;
        self.element(
            LayoutSpec::row().width_grow(1.0).height_fixed(1.0),
            PaintStyle::solid(col),
            |_| {},
        )
    }

    /// Desktop-grade vertical scrollbar with `f64` sub-pixel precision across millions of pixels,
    /// proportional thumb sizing (`min_thumb_h = 28px`), **0.0 px** grab-offset-preserving thumb drag,
    /// track click-to-seek + continuous drag, keyboard paging, AccessKit `Increment`/`Decrement`,
    /// and zero-lag post-layout thumb placement!
    #[track_caller]
    pub fn vertical_scrollbar(
        &mut self,
        label_for_a11y: &str,
        get_scroll: impl Fn(&S) -> f64 + Copy + 'static,
        set_scroll: impl Fn(&mut S, f64) + Copy + 'static,
        viewport_h_override: Option<f64>,
        total_content_h: f64,
        mut track_spec: LayoutSpec,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let th = self.sys.theme;
        let cur_scroll = get_scroll(self.state()).max(0.0);
        let pad = 2.0_f32;
        let min_thumb_h = 28.0_f32;

        track_spec.axis = crate::layout::Axis::Col;
        track_spec.cross_align = Align::Center;
        let track_width = if track_spec.mode_w == crate::layout::SizingMode::Fixed {
            track_spec.val_w
        } else {
            14.0
        };
        let is_hidden = track_spec.mode_w == crate::layout::SizingMode::Fixed && track_spec.val_w <= 0.1;
        let fixed_track_h = (track_spec.mode_h == crate::layout::SizingMode::Fixed)
            .then_some(track_spec.val_h);
        let gutter_bg = th.bg_canvas.mix_oklab(th.bg_surface, 0.52);

        self.element_internal(
            caller,
            track_spec,
            PaintStyle::solid(gutter_bg)
                .border(1.0, th.border_subtle)
                .with_opacity(if is_hidden { 0.0 } else { 1.0 }),
            |ui| {
                let track_s_idx = ui.active_slot.unwrap();
                let track_nid = ui.sys.slots[track_s_idx as usize].as_ref().unwrap().node_id;
                ui.sys.layout.topo[track_nid.idx()].flags |= crate::layout::FLAG_SCROLLABLE;
                ui.set_clips_children(true);
                if !is_hidden {
                    ui.set_focusable(true);
                    ui.set_cursor(CursorIcon::Grab);
                }

                let prev_track_h = ui.active_box().h;
                let approx_track_h = if prev_track_h > 1.0 {
                    prev_track_h
                } else {
                    fixed_track_h
                        .or_else(|| viewport_h_override.map(|v| v as f32))
                        .unwrap_or(400.0)
                };
                let approx_view_h = viewport_h_override.unwrap_or(approx_track_h as f64).max(1.0);
                let approx_max_s = (total_content_h - approx_view_h).max(0.0);
                ui.sync_semantics_in_place(
                    A11yRole::ScrollBar,
                    Some(label_for_a11y),
                    None,
                    None,
                    Some(cur_scroll.min(approx_max_s)),
                    Some(0.0),
                    Some(approx_max_s.max(0.0)),
                    Some(38.0),
                    true,
                    true,
                    true,
                );

                let drag_st = ui.use_state(ScrollbarDragState::default);
                let live_metrics = ui.use_state_updated((total_content_h, approx_view_h));

                ui.on_click_with_local(
                    drag_st,
                    move |s: &mut S, st: &mut ScrollbarDragState, ev: &PointerEvent| {
                        let live_scroll = get_scroll(s).max(0.0);
                        let (max_scroll, _rail_h, thumb_h, travel_h, thumb_top) =
                            compute_vscrollbar_metrics(
                                ev.target_h,
                                viewport_h_override,
                                total_content_h,
                                live_scroll,
                                pad,
                                min_thumb_h,
                            );
                        if max_scroll <= 0.0 || travel_h <= 0.0 {
                            st.grab_local_y = ev.local_y;
                            st.grab_scroll_y = 0.0;
                            st.skip_initial_down_drag = true;
                            return;
                        }
                        let thumb_y0 = pad + thumb_top;
                        let thumb_y1 = thumb_y0 + thumb_h;
                        if ev.local_y >= thumb_y0 - 1.0 && ev.local_y <= thumb_y1 + 1.0 {
                            // Grabbed directly on the thumb: preserve exact pixel offset with 0.0 px jump!
                            st.grab_local_y = ev.local_y;
                            st.grab_scroll_y = live_scroll.clamp(0.0, max_scroll);
                            st.skip_initial_down_drag = true;
                        } else {
                            // Clicked on empty track: center thumb on cursor & smoothly transition to drag!
                            let centered_t = ((ev.local_y - pad - thumb_h * 0.5) as f64
                                / travel_h as f64)
                                .clamp(0.0, 1.0);
                            let jumped = centered_t * max_scroll;
                            st.grab_local_y = pad + (centered_t as f32) * travel_h + thumb_h * 0.5;
                            st.grab_scroll_y = jumped;
                            st.skip_initial_down_drag = true;
                            set_scroll(s, jumped);
                        }
                    },
                );

                ui.on_drag_with_local(
                    drag_st,
                    move |s: &mut S, st: &mut ScrollbarDragState, ev: &PointerEvent| {
                        if st.skip_initial_down_drag {
                            st.skip_initial_down_drag = false;
                            return;
                        }
                        let (max_scroll, _rail_h, _thumb_h, travel_h, _thumb_top) =
                            compute_vscrollbar_metrics(
                                ev.target_h,
                                viewport_h_override,
                                total_content_h,
                                st.grab_scroll_y,
                                pad,
                                min_thumb_h,
                            );
                        if max_scroll <= 0.0 || travel_h <= 0.0 {
                            return;
                        }
                        let dy_px = (ev.local_y - st.grab_local_y) as f64;
                        let next_scroll = (st.grab_scroll_y + (dy_px / travel_h as f64) * max_scroll)
                            .clamp(0.0, max_scroll);
                        set_scroll(s, next_scroll);
                    },
                );

                ui.on_scroll(move |s, wheel_dy| {
                    let view_h = viewport_h_override.unwrap_or(approx_track_h as f64).max(1.0);
                    let max_s = (total_content_h - view_h).max(0.0);
                    if max_s <= 0.0 {
                        return false;
                    }
                    let cur = get_scroll(s).clamp(0.0, max_s);
                    let next = (cur - wheel_dy as f64).clamp(0.0, max_s);
                    if (next - cur).abs() > 1e-4 {
                        set_scroll(s, next);
                        true
                    } else {
                        false
                    }
                });

                ui.on_key_with_local(live_metrics, move |s, &mut (tot_h, vw_h), k| {
                    let view_h = vw_h.max(34.0);
                    let max_s = (tot_h - view_h).max(0.0);
                    let cur = get_scroll(s).clamp(0.0, max_s);
                    let next = match k {
                        KeyInput::ArrowUp => (cur - 38.0).max(0.0),
                        KeyInput::ArrowDown => (cur + 38.0).min(max_s),
                        KeyInput::PageUp => (cur - (view_h * 0.9).max(38.0)).max(0.0),
                        KeyInput::PageDown => (cur + (view_h * 0.9).max(38.0)).min(max_s),
                        KeyInput::Home => 0.0,
                        KeyInput::End => max_s,
                        _ => return false,
                    };
                    if (next - cur).abs() > 1e-4 {
                        set_scroll(s, next);
                        true
                    } else {
                        false
                    }
                });

                let captured = ui.is_captured();
                let hovered = ui.is_hovered();
                let focused = ui.is_focused();
                let thumb_w = if captured || hovered {
                    (track_width - 4.0).max(6.0)
                } else {
                    (track_width - 6.0).max(5.0)
                };
                let thumb_col = if captured {
                    th.accent
                } else if hovered {
                    th.text_secondary
                } else {
                    th.text_muted.with_alpha(if th.is_dark { 0.74 } else { 0.65 })
                };

                let (_max_s, _rail_h, init_thumb_h, _travel_h, _init_top) =
                    compute_vscrollbar_metrics(
                        approx_track_h,
                        viewport_h_override,
                        total_content_h,
                        cur_scroll,
                        pad,
                        min_thumb_h,
                    );

                let thumb_nid = ui.element(
                    LayoutSpec::col().fixed(thumb_w, init_thumb_h),
                    PaintStyle::solid(thumb_col)
                        .rounded(thumb_w * 0.5)
                        .border(
                            if focused { 1.5 } else { 0.0 },
                            if focused { th.border_focus } else { Color::TRANSPARENT },
                        )
                        .with_opacity(if is_hidden { 0.0 } else { 1.0 }),
                    |_| {},
                );

                ui.sys.scrollbar_thumbs.push(ScrollbarThumbSync {
                    track_node: track_nid,
                    thumb_node: thumb_nid,
                    scroll_y: cur_scroll,
                    viewport_h_override,
                    total_content_h,
                    pad,
                    min_thumb_h,
                });
            },
        )
    }

    /// Turnkey general-purpose scrollable container for non-virtualized content:
    /// Automatically measures inner content `fit_h` against live viewport height (including Frame 0!),
    /// clamps `scroll_y`, handles mouse-wheel + `Up`/`Down`/`PageUp`/`PageDown`/`Home`/`End` + `AccessKit`,
    /// and pairs with an interactive `vertical_scrollbar` whenever content overflows!
    #[track_caller]
    pub fn scroll_area(
        &mut self,
        label_for_a11y: &str,
        get_scroll: impl Fn(&S) -> f64 + Copy + 'static,
        set_scroll: impl Fn(&mut S, f64) + Copy + 'static,
        spec: LayoutSpec,
        paint: PaintStyle,
        children_fn: impl FnOnce(&mut Ui<'_, 's, S>),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let mut outer_spec = LayoutSpec::row();
        outer_spec.mode_w = spec.mode_w;
        outer_spec.val_w = spec.val_w;
        outer_spec.min_w = spec.min_w;
        outer_spec.max_w = spec.max_w;
        outer_spec.mode_h = spec.mode_h;
        outer_spec.val_h = spec.val_h;
        outer_spec.min_h = spec.min_h;
        outer_spec.max_h = spec.max_h;
        outer_spec.aspect_ratio = spec.aspect_ratio;
        outer_spec.shrink_q8 = spec.shrink_q8;
        outer_spec.grid_spans = spec.grid_spans;

        let mut content_spec = LayoutSpec {
            axis: spec.axis,
            pad_l: spec.pad_l,
            pad_t: spec.pad_t,
            pad_r: spec.pad_r,
            pad_b: spec.pad_b,
            gap: spec.gap,
            cross_align: spec.cross_align,
            main_align: spec.main_align,
            shrink_q8: 0,
            ..LayoutSpec::default()
        };
        content_spec.mode_w = crate::layout::SizingMode::Grow;
        content_spec.val_w = 1.0;
        content_spec.mode_h = crate::layout::SizingMode::Fit;

        let raw_scroll = get_scroll(self.state()).max(0.0);

        self.element_internal(caller, outer_spec, paint, |ui| {
            let outer_s_idx = ui.active_slot.unwrap();
            let outer_nid = ui.sys.slots[outer_s_idx as usize].as_ref().unwrap().node_id;
            ui.sys.layout.topo[outer_nid.idx()].flags |= crate::layout::FLAG_SCROLLABLE;
            ui.set_clips_children(true);
            ui.set_focusable(true);

            let metrics_h = ui.use_state(ScrollAreaMetrics::default);
            let m_snap = *ui.read_local(metrics_h);

            let est_view_h = if m_snap.initialized && m_snap.viewport_h > 1.0 {
                m_snap.viewport_h
            } else if outer_spec.mode_h == crate::layout::SizingMode::Fixed {
                outer_spec.val_h.max(1.0) as f64
            } else {
                ui.sys.current_logical_size[1].max(300.0) as f64
            };
            let est_content_h = if m_snap.initialized {
                m_snap.content_h
            } else {
                0.0
            };
            let assumed_overflow = m_snap.initialized && est_content_h > est_view_h + 0.5;
            let est_max_scroll = if m_snap.initialized {
                (est_content_h - est_view_h).max(0.0)
            } else {
                raw_scroll
            };
            let applied_scroll = raw_scroll.clamp(0.0, est_max_scroll) as f32;

            ui.sync_semantics_in_place(
                A11yRole::ScrollView,
                Some(label_for_a11y),
                None,
                None,
                Some(applied_scroll as f64),
                Some(0.0),
                Some(est_max_scroll.max(0.0)),
                None,
                false,
                true,
                est_max_scroll > 0.0,
            );

            // Wheel & keyboard handlers read live post-layout `ScrollAreaMetrics` directly from `metrics_h`!
            ui.on_key_with_local(metrics_h, move |s, m, k| {
                let view_h = m.viewport_h.max(34.0);
                let limit = (m.content_h - m.viewport_h).max(0.0);
                let cur = get_scroll(s).clamp(0.0, limit);
                let next = match k {
                    KeyInput::ArrowUp => (cur - 38.0).max(0.0),
                    KeyInput::ArrowDown => (cur + 38.0).min(limit),
                    KeyInput::PageUp => (cur - (view_h * 0.9).max(38.0)).max(0.0),
                    KeyInput::PageDown => (cur + (view_h * 0.9).max(38.0)).min(limit),
                    KeyInput::Home => 0.0,
                    KeyInput::End => limit,
                    _ => return false,
                };
                if (next - cur).abs() > 1e-4 {
                    set_scroll(s, next);
                    true
                } else {
                    false
                }
            });

            {
                unsafe fn invoke_scroll_with_metrics<S: 'static, F>(
                    state_ptr: *mut (),
                    handler_slot: u32,
                    arena: &mut [Option<Box<dyn Any>>],
                    wheel_dy: f32,
                ) -> bool
                where
                    F: Fn(&mut S, ScrollAreaMetrics, f32) -> bool + 'static,
                {
                    let s = unsafe { &mut *state_ptr.cast::<S>() };
                    let m_idx = arena[handler_slot as usize]
                        .as_ref()
                        .unwrap()
                        .downcast_ref::<(u32, F)>()
                        .unwrap()
                        .0;
                    let m = arena[m_idx as usize]
                        .as_ref()
                        .unwrap()
                        .downcast_ref::<ScrollAreaMetrics>()
                        .copied()
                        .unwrap_or_default();
                    let f = &arena[handler_slot as usize]
                        .as_ref()
                        .unwrap()
                        .downcast_ref::<(u32, F)>()
                        .unwrap()
                        .1;
                    f(s, m, wheel_dy)
                }

                fn bind_sa_scroll<S: 'static, F>(
                    ui: &mut Ui<'_, '_, S>,
                    outer_s_idx: u32,
                    m_idx: u32,
                    handler: F,
                ) where
                    F: Fn(&mut S, ScrollAreaMetrics, f32) -> bool + 'static,
                {
                    let pack = ui.use_state_updated((m_idx, handler));
                    let slot = ui.sys.slots[outer_s_idx as usize].as_mut().unwrap();
                    slot.active_bindings |= BIND_SCROLL;
                    if let Some(existing) = &mut slot.on_scroll {
                        sync_lens_steps(&mut existing.steps, &ui.sys.lens_scratch);
                        existing.handler_slot = pack.arena_idx;
                        existing.invoke = invoke_scroll_with_metrics::<S, F>;
                    } else {
                        slot.on_scroll = Some(ScrollBinding {
                            steps: ui.sys.lens_scratch.as_slice().into(),
                            handler_slot: pack.arena_idx,
                            invoke: invoke_scroll_with_metrics::<S, F>,
                        });
                    }
                }

                bind_sa_scroll(
                    ui,
                    outer_s_idx,
                    metrics_h.arena_idx,
                    move |s, m, wheel_dy| {
                        let limit = (m.content_h - m.viewport_h).max(0.0);
                        if limit <= 0.0 {
                            return false;
                        }
                        let cur = get_scroll(s).clamp(0.0, limit);
                        let next = (cur - wheel_dy as f64).clamp(0.0, limit);
                        if (next - cur).abs() > 1e-4 {
                            set_scroll(s, next);
                            true
                        } else {
                            false
                        }
                    },
                );
            }

            // 1. Left column: hardware-scissored scrollable viewport + unshrunk `Fit` content wrapper
            let mut viewport_spec = LayoutSpec::col().width_grow(1.0);
            if outer_spec.mode_h == crate::layout::SizingMode::Fit {
                viewport_spec.mode_h = crate::layout::SizingMode::Fit;
                viewport_spec.max_h = outer_spec.max_h;
            } else {
                viewport_spec = viewport_spec.height_grow(1.0);
            }
            viewport_spec.scroll_y = applied_scroll;

            let mut content_nid = NodeId::NONE;
            let viewport_nid = ui.element(viewport_spec, PaintStyle::default(), |ui| {
                let vs_idx = ui.active_slot.unwrap();
                let v_nid = ui.sys.slots[vs_idx as usize].as_ref().unwrap().node_id;
                ui.sys.layout.topo[v_nid.idx()].flags |= crate::layout::FLAG_SCROLLABLE;
                ui.set_clips_children(true);
                content_nid = ui.element(content_spec, PaintStyle::default(), children_fn);
            });

            // 2. Right edge: auto-toggled `14px` vertical scrollbar (`0px` & hidden when content fits!)
            let sb_w = if assumed_overflow { 14.0 } else { 0.0 };
            let sb_nid = ui.vertical_scrollbar(
                label_for_a11y,
                get_scroll,
                set_scroll,
                Some(est_view_h),
                est_content_h,
                LayoutSpec::col().width_fixed(sb_w).height_grow(1.0),
            );
            let sb_slot = ui.sys.node_to_slot[sb_nid.idx()];

            ui.sys.scroll_areas.push(ScrollAreaSync {
                viewport_node: viewport_nid,
                content_node: content_nid,
                scrollbar_slot: sb_slot,
                metrics_arena_idx: metrics_h.arena_idx,
                raw_scroll_y: raw_scroll,
                assumed_overflow,
                applied_scroll_y: applied_scroll,
            });
        })
    }

    /// Turnkey $O(\log N)$ variable-height virtualized scroll container powered by `VirtualListIndex`!
    #[track_caller]
    pub fn virtual_list(
        &mut self,
        vlist: &VirtualListIndex,
        scroll_y: f64,
        width: f32,
        height: f32,
        mut render_item: impl FnMut(&mut Ui<'_, 's, S>, usize, f32),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let (start_idx, end_idx, first_top) = vlist.visible_range(scroll_y, height as f64);
        let mut spec = LayoutSpec::col().fixed(width, height);
        spec.scroll_y = (scroll_y - first_top).max(0.0) as f32;

        self.element_internal(caller, spec, PaintStyle::default(), |ui| {
            let s_idx = ui.active_slot.unwrap();
            let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
            ui.sys.layout.topo[nid.idx()].flags |= crate::layout::FLAG_SCROLLABLE;
            ui.set_clips_children(true);
            for idx in start_idx..end_idx {
                let row_h = vlist.height_at(idx);
                ui.keyed(idx as u64 + 1, |ui| {
                    render_item(ui, idx, row_h);
                });
            }
        })
    }

    /// Complete $O(\log N)$ variable-height virtualized scroll view with integrated right-hand
    /// `vertical_scrollbar`, mouse-wheel/touchpad pixel scrolling, keyboard `Up`/`Down`/`PageUp`/`PageDown`/`Home`/`End`,
    /// and dynamic `Grow` or `Fixed` viewport sizing!
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn virtual_scroll_view(
        &mut self,
        label_for_a11y: &str,
        vlist: &VirtualListIndex,
        get_scroll: impl Fn(&S) -> f64 + Copy + 'static,
        set_scroll: impl Fn(&mut S, f64) + Copy + 'static,
        mut spec: LayoutSpec,
        paint: PaintStyle,
        mut render_item: impl FnMut(&mut Ui<'_, 's, S>, usize, f32),
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        spec.axis = crate::layout::Axis::Row;
        let total_h = vlist.total_height();
        let raw_scroll = get_scroll(self.state()).max(0.0);

        self.element_internal(caller, spec, paint, |ui| {
            ui.set_clips_children(true);
            ui.set_focusable(true);

            let prev_box_h = ui.active_box().h;
            let cur_win_h = ui.sys.current_logical_size[1];
            let prev_win_h = ui.sys.prev_logical_size[1];
            let exact_view_h = if spec.mode_h == crate::layout::SizingMode::Fixed {
                spec.val_h.max(1.0)
            } else if prev_box_h > 1.0 {
                prev_box_h
            } else {
                cur_win_h.max(400.0)
            };
            // On sudden window maximize, query extra rows immediately so Frame 0 of resize never flashes blank space:
            let query_view_h = if spec.mode_h == crate::layout::SizingMode::Fixed {
                exact_view_h
            } else if prev_box_h <= 1.0 {
                cur_win_h.max(400.0)
            } else {
                prev_box_h + (cur_win_h - prev_win_h).max(0.0)
            };

            let max_scroll = (total_h - exact_view_h as f64).max(0.0);
            let clamped_scroll = raw_scroll.clamp(0.0, max_scroll);
            ui.sync_semantics_in_place(
                A11yRole::ScrollView,
                Some(label_for_a11y),
                None,
                None,
                Some(clamped_scroll),
                Some(0.0),
                Some(max_scroll.max(0.0)),
                None,
                false,
                true,
                max_scroll > 0.0,
            );
            let (start_idx, end_idx, first_top) =
                vlist.visible_range(clamped_scroll, query_view_h as f64);

            ui.on_scroll(move |s, wheel_dy| {
                let limit = (total_h - exact_view_h as f64).max(0.0);
                let cur = get_scroll(s).clamp(0.0, limit);
                let next = (cur - (wheel_dy as f64)).clamp(0.0, limit);
                if (next - cur).abs() > 1e-4 {
                    set_scroll(s, next);
                    true
                } else {
                    false
                }
            });

            ui.on_key(move |s, k| {
                let view_h = (exact_view_h as f64).max(34.0);
                let limit = (total_h - view_h).max(0.0);
                let cur = get_scroll(s).clamp(0.0, limit);
                let next = match k {
                    KeyInput::ArrowUp => (cur - 38.0).max(0.0),
                    KeyInput::ArrowDown => (cur + 38.0).min(limit),
                    KeyInput::PageUp => (cur - (view_h * 0.9).max(38.0)).max(0.0),
                    KeyInput::PageDown => (cur + (view_h * 0.9).max(38.0)).min(limit),
                    KeyInput::Home => 0.0,
                    KeyInput::End => limit,
                    _ => return false,
                };
                if (next - cur).abs() > 1e-4 {
                    set_scroll(s, next);
                    true
                } else {
                    false
                }
            });

            // 1. Left column: hardware-scissored virtualized rows viewport
            let mut viewport_spec = LayoutSpec::col().width_grow(1.0).height_grow(1.0);
            viewport_spec.scroll_y = (clamped_scroll - first_top).max(0.0) as f32;
            ui.element(viewport_spec, PaintStyle::default(), |ui| {
                let s_idx = ui.active_slot.unwrap();
                let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
                ui.sys.layout.topo[nid.idx()].flags |= crate::layout::FLAG_SCROLLABLE;
                ui.set_clips_children(true);
                for idx in start_idx..end_idx {
                    let row_h = vlist.height_at(idx);
                    ui.keyed(idx as u64 + 1, |ui| {
                        render_item(ui, idx, row_h);
                    });
                }
            });

            // 2. Right edge: integrated `f64` vertical scrollbar (`14px` gutter)
            ui.vertical_scrollbar(
                label_for_a11y,
                get_scroll,
                set_scroll,
                None,
                total_h,
                LayoutSpec::col().width_fixed(14.0).height_grow(1.0),
            );
        })
    }

    /// Attach a crisp, viewport-clamped hover tooltip to `target` using Pass 4 `AnchorPlacement::TopCenter`
    /// (automatically flipping below `target` when near the top viewport edge and escaping parent clips!).
    #[track_caller]
    pub fn tooltip(&mut self, target: NodeId, text: &str) -> NodeId {
        let saved_key = core::mem::replace(&mut self.pending_user_key, 0);
        if target.is_none() || text.is_empty() {
            return NodeId::NONE;
        }
        let Some(&target_slot) = self.sys.node_to_slot.get(target.idx()) else {
            return NodeId::NONE;
        };
        if target_slot == u32::MAX || !self.sys.hovered_chain.contains(&target_slot) {
            return NodeId::NONE;
        }

        let caller = Location::caller() as *const Location<'static> as usize;
        let th = self.sys.theme;
        let (bg, border_c, fg, shadow_c) = if th.is_dark {
            (
                th.bg_elevated.shift_lightness(0.035),
                th.accent.with_alpha(0.52),
                th.text_primary,
                Color::rgba8(0, 0, 0, 155),
            )
        } else {
            (
                Color::rgb8(24, 28, 38),
                Color::rgba8(255, 255, 255, 42),
                Color::rgb8(246, 248, 252),
                Color::rgba8(15, 23, 42, 68),
            )
        };

        let tip_mix = (target.0 as u64).wrapping_add(1).wrapping_mul(0x517c_c1b7_2722_0a95);
        let resolved_user_key =
            USER_KEY_EXPLICIT_BIT | ((saved_key ^ tip_mix) & !USER_KEY_EXPLICIT_BIT);
        self.pending_user_key = resolved_user_key;

        let avail_w = if self.sys.current_logical_size[0] > 32.0 {
            (self.sys.current_logical_size[0] - 16.0).clamp(160.0, 360.0)
        } else {
            360.0
        };
        let avail_h = if self.sys.current_logical_size[1] > 32.0 {
            (self.sys.current_logical_size[1] - 16.0).clamp(64.0, 180.0)
        } else {
            180.0
        };
        let max_inner_w = (avail_w - 20.0).max(1.0);
        let existing_pill_w = self
            .parent_slot
            .and_then(|p| {
                self.sys.slots[p as usize]
                    .as_ref()?
                    .children
                    .get(self.cursor_pos)
                    .copied()
            })
            .and_then(|s_idx| self.sys.slots[s_idx as usize].as_ref())
            .filter(|s| {
                s.key.callsite_addr == caller
                    && s.key.user_key == resolved_user_key
                    && self.sys.layout.specs[s.node_id.idx()].mode_w
                        == crate::layout::SizingMode::Fixed
            })
            .map(|s| self.sys.layout.specs[s.node_id.idx()].val_w.min(avail_w));

        let mut init_spec = LayoutSpec::row()
            .pad_xy(10.0, 6.0)
            .align(Align::Center)
            .max_size(avail_w, avail_h);
        if let Some(w) = existing_pill_w {
            init_spec = init_spec.width_fixed(w);
        }

        self.element_internal(
            caller,
            init_spec,
            PaintStyle::solid(bg)
                .rounded(th.radius_sm)
                .border(1.0, border_c)
                .drop_shadow(10.0, [0.0, 4.0], shadow_c),
            |ui| {
                ui.set_clips_children(true);
                let s_idx = ui.active_slot.unwrap();
                let nid = ui.sys.slots[s_idx as usize].as_ref().unwrap().node_id;
                ui.sys.slots[s_idx as usize]
                    .as_mut()
                    .unwrap()
                    .pointer_passthrough = true;
                ui.sync_semantics_in_place(
                    A11yRole::Tooltip,
                    Some(text),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                    false,
                    false,
                );
                ui.sys.layout.set_floating_anchor(
                    nid,
                    target,
                    AnchorPlacement::TopCenter,
                    0.0,
                    6.0,
                );
                let text_nid = ui.ui_text(text, 12.0, fg);
                let tid = ui.sys.layout.specs[text_nid.idx()].aux_id as usize;
                let unwrapped_w = ui.sys.layout.texts[tid].max_unwrapped_w;
                let content_w = if unwrapped_w <= max_inner_w {
                    unwrapped_w
                } else {
                    ui.sys.layout.texts[tid].wrap_for_width(max_inner_w).1
                };
                let exact_pill_w = (content_w.ceil() + 20.0).clamp(24.0, avail_w);
                let cur_spec = ui.sys.layout.specs[nid.idx()];
                if cur_spec.mode_w != crate::layout::SizingMode::Fixed
                    || (cur_spec.val_w - exact_pill_w).abs() > 1e-4
                {
                    ui.sys
                        .layout
                        .update_spec(nid, cur_spec.width_fixed(exact_pill_w));
                }
            },
        )
    }

    /// Build a target widget and attach a Pass-4 `AnchorPlacement::TopCenter` hover tooltip to it.
    #[track_caller]
    pub fn with_tooltip(
        &mut self,
        text: &str,
        build_target: impl FnOnce(&mut Ui<'_, 's, S>) -> NodeId,
    ) -> NodeId {
        let target = build_target(self);
        let _ = self.tooltip(target, text);
        target
    }

    /// Editable single-line UTF-8 `text_input` with full `TextEditState { cursor, anchor, scroll_x }` selection,
    /// `Ctrl+A` (`SelectAll`), `Shift+Arrows` / `Shift+Home` / `Shift+End` / `Ctrl+Shift+Arrows`,
    /// mouse click + drag selection + double-click word selection, `Copy` / `Cut` / `Paste` clipboard integration,
    /// crisp translucent `th.accent.with_alpha(0.28)` selection highlight quad behind `[x_start..x_end]`,
    /// kerning-preserving single-run typography, horizontal scroll auto-follow, and `AccessKit` `SetValue`!
    #[track_caller]
    pub fn text_input(
        &mut self,
        placeholder: &str,
        value_ref: fn(&'s S) -> &'s String,
        value_mut: fn(&mut S) -> &mut String,
    ) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let th = self.sys.theme;
        let pad_x = 10.0_f32;
        self.element_internal(
            caller,
            LayoutSpec::row()
                .width_grow(1.0)
                .height_fixed(34.0)
                .pad_xy(pad_x, 6.0)
                .align(Align::Center),
            PaintStyle::solid(th.bg_control)
                .rounded(th.radius_sm)
                .border(1.0, th.border_subtle),
            |ui| {
                let input_s_idx = ui.active_slot.unwrap();
                let input_nid = ui.sys.slots[input_s_idx as usize].as_ref().unwrap().node_id;
                ui.set_clips_children(true);
                ui.lens(value_ref, value_mut, |ui| {
                    ui.set_focusable(true);
                    ui.set_cursor(CursorIcon::Text);
                    let focused = ui.is_focused();
                    let hovered = ui.is_hovered();
                    ui.set_paint(
                        PaintStyle::solid(if focused {
                            th.bg_surface
                        } else if hovered {
                            th.bg_control_hover
                        } else {
                            th.bg_control
                        })
                        .rounded(th.radius_sm)
                        .border(
                            if focused { 2.0 } else { 1.0 },
                            if focused { th.border_focus } else { th.border_subtle },
                        ),
                    );

                    let cur_val = ui.state().as_str();
                    ui.sync_semantics_in_place(
                        A11yRole::TextInput,
                        Some(placeholder),
                        Some(cur_val),
                        None,
                        None,
                        None,
                        None,
                        None,
                        true,
                        true,
                        false,
                    );
                    let init_len = cur_val.len();
                    let ti_state = ui.use_state(|| TextEditState::new_at(init_len));
                    let fid = ui.sys.fonts.ui_font;

                    // Normalize `cursor` and `anchor` if external state mutated `String` shorter:
                    {
                        let st_mut = ui.sys.state_arena[ti_state.arena_idx as usize]
                            .as_mut()
                            .unwrap()
                            .downcast_mut::<TextEditState>()
                            .unwrap();
                        st_mut.cursor = cur_val.floor_char_boundary(st_mut.cursor.min(init_len));
                        st_mut.anchor = cur_val.floor_char_boundary(st_mut.anchor.min(init_len));
                    }

                    // Click: place collapsed caret (or select whole word on rapid double-click!)
                    // using zero-alloc `ShapedParagraph::x_to_byte_offset` when cached paragraph is current:
                    let line_h_14 = (14.0_f32 * 1.35).round();
                    ui.bind_pointer_with_local_and_fonts(
                        true,
                        ti_state,
                        move |s: &mut String, st: &mut TextEditState, fonts, layout, ev| {
                            let rel_x = (ev.local_x - pad_x + st.scroll_x).max(0.0);
                            let hit = hit_test_text_input_x(
                                s.as_str(),
                                st.cached_text_node,
                                rel_x,
                                fid,
                                14.0,
                                line_h_14,
                                fonts,
                                layout,
                            );
                            let now = std::time::Instant::now();
                            let is_double = !st.last_was_double_click
                                && (ev.local_x - st.last_click_x).abs() <= 4.0
                                && st.last_click_time.is_some_and(|prev| {
                                    now.duration_since(prev) <= std::time::Duration::from_millis(400)
                                });
                            if is_double {
                                let (w0, w1) = word_range_at(s.as_str(), hit);
                                st.anchor = w0;
                                st.cursor = w1;
                                st.last_was_double_click = true;
                            } else {
                                st.anchor = hit;
                                st.cursor = hit;
                                st.last_was_double_click = false;
                            }
                            st.last_click_time = Some(now);
                            st.last_click_x = ev.local_x;
                            st.skip_initial_down_drag = true;
                        },
                    );

                    // Drag: update `cursor` live while keeping `anchor` pinned at initial click offset!
                    ui.bind_pointer_with_local_and_fonts(
                        false,
                        ti_state,
                        move |s: &mut String, st: &mut TextEditState, fonts, layout, ev| {
                            if st.skip_initial_down_drag {
                                st.skip_initial_down_drag = false;
                                return;
                            }
                            let rel_x = (ev.local_x - pad_x + st.scroll_x).max(0.0);
                            st.cursor = hit_test_text_input_x(
                                s.as_str(),
                                st.cached_text_node,
                                rel_x,
                                fid,
                                14.0,
                                line_h_14,
                                fonts,
                                layout,
                            );
                        },
                    );

                    ui.bind_key_with_local_and_clipboard(
                        ti_state,
                        |s: &mut String,
                         st: &mut TextEditState,
                         clipboard: &mut String,
                         key: &KeyInput|
                         -> bool {
                            let cur = s.floor_char_boundary(st.cursor.min(s.len()));
                            let anc = s.floor_char_boundary(st.anchor.min(s.len()));
                            let (sel_min, sel_max) = if cur <= anc { (cur, anc) } else { (anc, cur) };
                            let has_sel = sel_min < sel_max;

                            match key {
                                KeyInput::Text(ins) | KeyInput::Paste(ins) => {
                                    if has_sel {
                                        s.replace_range(sel_min..sel_max, ins);
                                        let next = sel_min + ins.len();
                                        st.cursor = next;
                                        st.anchor = next;
                                    } else if !ins.is_empty() {
                                        s.insert_str(cur, ins);
                                        let next = cur + ins.len();
                                        st.cursor = next;
                                        st.anchor = next;
                                    } else {
                                        st.cursor = cur;
                                        st.anchor = cur;
                                    }
                                    true
                                }
                                KeyInput::Space => {
                                    if has_sel {
                                        s.replace_range(sel_min..sel_max, " ");
                                        st.cursor = sel_min + 1;
                                        st.anchor = sel_min + 1;
                                    } else {
                                        s.insert(cur, ' ');
                                        st.cursor = cur + 1;
                                        st.anchor = cur + 1;
                                    }
                                    true
                                }
                                KeyInput::SelectAll => {
                                    st.anchor = 0;
                                    st.cursor = s.len();
                                    true
                                }
                                KeyInput::Copy => {
                                    if has_sel {
                                        clipboard.clear();
                                        clipboard.push_str(&s[sel_min..sel_max]);
                                    }
                                    true
                                }
                                KeyInput::Cut => {
                                    if has_sel {
                                        clipboard.clear();
                                        clipboard.push_str(&s[sel_min..sel_max]);
                                        s.drain(sel_min..sel_max);
                                        st.cursor = sel_min;
                                        st.anchor = sel_min;
                                    }
                                    true
                                }
                                KeyInput::Backspace => {
                                    if has_sel {
                                        s.drain(sel_min..sel_max);
                                        st.cursor = sel_min;
                                        st.anchor = sel_min;
                                    } else if cur > 0 {
                                        let prev = s.floor_char_boundary(cur - 1);
                                        s.drain(prev..cur);
                                        st.cursor = prev;
                                        st.anchor = prev;
                                    }
                                    true
                                }
                                KeyInput::CtrlBackspace => {
                                    if has_sel {
                                        s.drain(sel_min..sel_max);
                                        st.cursor = sel_min;
                                        st.anchor = sel_min;
                                    } else if cur > 0 {
                                        let prev = prev_word_boundary(s, cur);
                                        s.drain(prev..cur);
                                        st.cursor = prev;
                                        st.anchor = prev;
                                    }
                                    true
                                }
                                KeyInput::Delete => {
                                    if has_sel {
                                        s.drain(sel_min..sel_max);
                                        st.cursor = sel_min;
                                        st.anchor = sel_min;
                                    } else if cur < s.len() {
                                        let next = s.ceil_char_boundary(cur + 1);
                                        s.drain(cur..next);
                                        st.cursor = cur;
                                        st.anchor = cur;
                                    }
                                    true
                                }
                                KeyInput::CtrlDelete => {
                                    if has_sel {
                                        s.drain(sel_min..sel_max);
                                        st.cursor = sel_min;
                                        st.anchor = sel_min;
                                    } else if cur < s.len() {
                                        let next = next_word_boundary(s, cur);
                                        s.drain(cur..next);
                                        st.cursor = cur;
                                        st.anchor = cur;
                                    }
                                    true
                                }
                                KeyInput::ArrowLeft => {
                                    let next = if has_sel {
                                        sel_min
                                    } else if cur > 0 {
                                        s.floor_char_boundary(cur - 1)
                                    } else {
                                        0
                                    };
                                    st.cursor = next;
                                    st.anchor = next;
                                    true
                                }
                                KeyInput::ShiftArrowLeft => {
                                    st.anchor = anc;
                                    st.cursor = if cur > 0 {
                                        s.floor_char_boundary(cur - 1)
                                    } else {
                                        0
                                    };
                                    true
                                }
                                KeyInput::CtrlArrowLeft => {
                                    let next = prev_word_boundary(s, cur);
                                    st.cursor = next;
                                    st.anchor = next;
                                    true
                                }
                                KeyInput::CtrlShiftArrowLeft => {
                                    st.anchor = anc;
                                    st.cursor = prev_word_boundary(s, cur);
                                    true
                                }
                                KeyInput::ArrowRight => {
                                    let next = if has_sel {
                                        sel_max
                                    } else if cur < s.len() {
                                        s.ceil_char_boundary(cur + 1)
                                    } else {
                                        s.len()
                                    };
                                    st.cursor = next;
                                    st.anchor = next;
                                    true
                                }
                                KeyInput::ShiftArrowRight => {
                                    st.anchor = anc;
                                    st.cursor = if cur < s.len() {
                                        s.ceil_char_boundary(cur + 1)
                                    } else {
                                        s.len()
                                    };
                                    true
                                }
                                KeyInput::CtrlArrowRight => {
                                    let next = next_word_boundary(s, cur);
                                    st.cursor = next;
                                    st.anchor = next;
                                    true
                                }
                                KeyInput::CtrlShiftArrowRight => {
                                    st.anchor = anc;
                                    st.cursor = next_word_boundary(s, cur);
                                    true
                                }
                                KeyInput::Home => {
                                    st.cursor = 0;
                                    st.anchor = 0;
                                    true
                                }
                                KeyInput::ShiftHome => {
                                    st.anchor = anc;
                                    st.cursor = 0;
                                    true
                                }
                                KeyInput::End => {
                                    st.cursor = s.len();
                                    st.anchor = s.len();
                                    true
                                }
                                KeyInput::ShiftEnd => {
                                    st.anchor = anc;
                                    st.cursor = s.len();
                                    true
                                }
                                _ => false,
                            }
                        },
                    );

                    let st_snap = *ui.read_local(ti_state);
                    let cur_pos = cur_val.floor_char_boundary(st_snap.cursor.min(cur_val.len()));
                    let anc_pos = cur_val.floor_char_boundary(st_snap.anchor.min(cur_val.len()));
                    let has_sel = cur_pos != anc_pos;
                    let is_empty = cur_val.is_empty();

                    // 1. Selection highlight quad (emitted FIRST so it paints cleanly behind text glyphs!):
                    //    Spec width is `0.0` during Pass 1 so `Axis::Row` places `text_node` at `inner_x`;
                    //    `sync_text_inputs` positions `[sel_x0..sel_x1]` post-layout with 0 layout churn!
                    let sel_col = if focused && has_sel && !is_empty {
                        th.accent.with_alpha(0.28)
                    } else {
                        Color::TRANSPARENT
                    };
                    let sel_nid = ui.element(
                        LayoutSpec::col().fixed(0.0, 18.0),
                        PaintStyle::solid(sel_col).rounded(2.0),
                        |_| {},
                    );

                    // 2. Render the ENTIRE string as ONE unbroken shaped text run so kerning & ligatures
                    //    never shift when typing, selecting, or moving the caret!
                    let text_nid = if is_empty {
                        ui.single_line_ui_text(placeholder, 14.0, th.text_muted)
                    } else {
                        ui.single_line_ui_text(cur_val, 14.0, th.text_primary)
                    };

                    let (caret_x, anc_x, total_text_w) = if is_empty {
                        (0.0, 0.0, 0.0)
                    } else if (ui.sys.layout.topo[text_nid.idx()].flags & FLAG_HAS_TEXT) != 0 {
                        // Zero-allocation `O(clusters)` caret & selection geometry directly from the shaped paragraph!
                        let para =
                            &ui.sys.layout.texts[ui.sys.layout.specs[text_nid.idx()].aux_id as usize];
                        let tot_w = para.max_unwrapped_w;
                        let cx = para.byte_offset_to_x(cur_val, cur_pos);
                        let ax = if !has_sel {
                            cx
                        } else {
                            para.byte_offset_to_x(cur_val, anc_pos)
                        };
                        (cx, ax, tot_w)
                    } else {
                        let tot_w = ui
                            .sys
                            .fonts
                            .byte_offset_to_x(cur_val, cur_val.len(), fid, 14.0);
                        let cx = ui.sys.fonts.byte_offset_to_x(cur_val, cur_pos, fid, 14.0);
                        let ax = if !has_sel {
                            cx
                        } else {
                            ui.sys.fonts.byte_offset_to_x(cur_val, anc_pos, fid, 14.0)
                        };
                        (cx, ax, tot_w)
                    };
                    let (sel_x0, sel_x1) = if caret_x <= anc_x {
                        (caret_x, anc_x)
                    } else {
                        (anc_x, caret_x)
                    };

                    // 3. Thin 2x16 overlay caret positioned post-layout at exact `inner_x + caret_x - scroll_x`:
                    let caret_col = if focused { th.accent } else { Color::TRANSPARENT };
                    let caret_nid = ui.element(
                        LayoutSpec::col().fixed(2.0, 16.0),
                        PaintStyle::solid(caret_col).rounded(1.0),
                        |_| {},
                    );

                    ui.sys.text_inputs.push(TextInputSync {
                        input_node: input_nid,
                        sel_node: sel_nid,
                        text_node: text_nid,
                        caret_node: caret_nid,
                        state_arena_idx: ti_state.arena_idx,
                        pad_x,
                        caret_x,
                        sel_x0,
                        sel_x1,
                        has_selection: has_sel,
                        total_text_w,
                        is_empty,
                        focused,
                    });
                });
            },
        )
    }

    #[inline]
    fn find_matching_child(&self, key: SlotKey) -> Option<(u32, usize)> {
        let p_idx = self.parent_slot?;
        let children = &self.sys.slots[p_idx as usize].as_ref()?.children;
        if let Some(&ch_slot) = children.get(self.cursor_pos)
            && let Some(ch) = self.sys.slots[ch_slot as usize].as_ref()
            && ch.key == key
        {
            return Some((ch_slot, self.cursor_pos));
        }
        for (pos, &ch_slot) in children.iter().enumerate().skip(self.cursor_pos + 1) {
            if let Some(ch) = self.sys.slots[ch_slot as usize].as_ref()
                && ch.key == key
            {
                return Some((ch_slot, pos));
            }
        }
        None
    }

    fn advance_to_matched_in_place(&mut self, matched_pos: usize, matched_is_keyed: bool) {
        if matched_pos == self.cursor_pos {
            self.cursor_pos += 1;
            return;
        }
        let Some(p_idx) = self.parent_slot else {
            return;
        };
        self.structure_changed = true;
        if matched_is_keyed {
            let children = &mut self.sys.slots[p_idx as usize].as_mut().unwrap().children;
            let matched_val = children.remove(matched_pos);
            children.insert(self.cursor_pos, matched_val);
            self.cursor_pos += 1;
            return;
        }
        let mut cur = self.cursor_pos;
        let mut target = matched_pos;
        while cur < target {
            let candidate = self.sys.slots[p_idx as usize].as_ref().unwrap().children[cur];
            let is_keyed = self.sys.slots[candidate as usize]
                .as_ref()
                .is_some_and(|s| s.key.is_explicitly_keyed());
            if is_keyed {
                cur += 1;
            } else {
                self.sys.slots[p_idx as usize]
                    .as_mut()
                    .unwrap()
                    .children
                    .remove(cur);
                self.sys.unmount_slot(candidate);
                target -= 1;
            }
        }
        if target > self.cursor_pos {
            let children = &mut self.sys.slots[p_idx as usize].as_mut().unwrap().children;
            let matched_val = children.remove(target);
            children.insert(self.cursor_pos, matched_val);
        }
        self.cursor_pos += 1;
    }

    #[inline]
    fn reconcile_slot_props(&mut self, slot_idx: u32, mut spec: LayoutSpec, paint: PaintStyle) {
        let (nid, has_builtin_scroll) = {
            let s = self.sys.slots[slot_idx as usize].as_ref().unwrap();
            (s.node_id, s.scrollable_max_y.is_some())
        };
        if has_builtin_scroll && spec.scroll_y == 0.0 {
            spec.scroll_y = self.sys.layout.specs[nid.idx()].scroll_y;
        }
        self.sys.layout.update_spec(nid, spec);
        let slot = self.sys.slots[slot_idx as usize].as_mut().unwrap();
        slot.paint = paint;
        if slot.paint != slot.committed_paint
            && (self.sys.layout.topo[nid.idx()].flags & FLAG_DIRTY_PAINT) == 0
        {
            self.sys.layout.topo[nid.idx()].flags |= FLAG_DIRTY_PAINT;
            self.sys.dirty_paint_slots.push(slot_idx);
        }
    }
}

struct OklabLuma;
impl OklabLuma {
    #[inline]
    fn of(c: Color) -> f32 {
        crate::color::Oklab::from_color(c).l
    }
}

impl<'sys, 's> Ui<'sys, 's, !> {
    #[inline]
    fn resolve_never_key(&mut self, caller_addr: usize) -> SlotKey {
        let pending = core::mem::replace(&mut self.pending_user_key, 0);
        if (pending & USER_KEY_EXPLICIT_BIT) != 0 {
            return SlotKey {
                callsite_addr: caller_addr,
                user_key: pending,
            };
        }
        let scope = &mut self.sys.callsite_counts_scratch[self.callsite_scope_base..];
        for entry in scope.iter_mut() {
            if entry.0 == caller_addr {
                let ord = entry.1 as u64;
                entry.1 = entry.1.saturating_add(1);
                return SlotKey {
                    callsite_addr: caller_addr,
                    user_key: ord,
                };
            }
        }
        self.sys.callsite_counts_scratch.push((caller_addr, 1));
        SlotKey {
            callsite_addr: caller_addr,
            user_key: 0,
        }
    }

    #[track_caller]
    pub fn passive_label(&mut self, text: &str, font_size: f32, line_h: f32, color: Color) -> NodeId {
        let caller = Location::caller() as *const Location<'static> as usize;
        let key = self.resolve_never_key(caller);
        let parent_for_child = self.parent_slot;
        let slot_idx = if let Some((matched, pos)) = self.find_matching_never(key) {
            self.advance_to_matched_never(pos, key.is_explicitly_keyed());
            self.sys.slots[matched as usize].as_mut().unwrap().parent = parent_for_child;
            let nid = self.sys.slots[matched as usize].as_ref().unwrap().node_id;
            if self.sys.slots[matched as usize].as_ref().unwrap().text_color != color {
                self.sys.slots[matched as usize].as_mut().unwrap().text_color = color;
                self.sys.scene_topology_changed = true;
            }
            let prev_hash = (self.sys.layout.topo[nid.idx()].flags & FLAG_HAS_TEXT)
                .ne(&0)
                .then(|| self.sys.layout.texts[self.sys.layout.specs[nid.idx()].aux_id as usize].content_hash);
            self.sys.layout.attach_shaped_text(
                nid,
                text,
                FontId::FALLBACK,
                font_size,
                line_h,
                &self.sys.fonts,
            );
            let new_hash =
                self.sys.layout.texts[self.sys.layout.specs[nid.idx()].aux_id as usize].content_hash;
            if prev_hash != Some(new_hash) && color.channels()[3] > 0 {
                self.sys.scene_topology_changed = true;
            }
            matched
        } else {
            let s = self
                .sys
                .alloc_slot(key, parent_for_child, LayoutSpec::default(), PaintStyle::default());
            let nid = self.sys.slots[s as usize].as_ref().unwrap().node_id;
            self.sys.slots[s as usize].as_mut().unwrap().text_color = color;
            self.sys.layout.attach_shaped_text(
                nid,
                text,
                FontId::FALLBACK,
                font_size,
                line_h,
                &self.sys.fonts,
            );
            if let Some(p_idx) = parent_for_child {
                self.sys.slots[p_idx as usize]
                    .as_mut()
                    .unwrap()
                    .children
                    .insert(self.cursor_pos, s);
                self.cursor_pos += 1;
                self.structure_changed = true;
            }
            s
        };
        self.sys.slots[slot_idx as usize].as_ref().unwrap().node_id
    }

    fn find_matching_never(&self, key: SlotKey) -> Option<(u32, usize)> {
        let p_idx = self.parent_slot?;
        let children = &self.sys.slots[p_idx as usize].as_ref()?.children;
        if let Some(&ch_slot) = children.get(self.cursor_pos)
            && let Some(ch) = self.sys.slots[ch_slot as usize].as_ref()
            && ch.key == key
        {
            return Some((ch_slot, self.cursor_pos));
        }
        for (pos, &ch_slot) in children.iter().enumerate().skip(self.cursor_pos + 1) {
            if let Some(ch) = self.sys.slots[ch_slot as usize].as_ref()
                && ch.key == key
            {
                return Some((ch_slot, pos));
            }
        }
        None
    }

    fn advance_to_matched_never(&mut self, matched_pos: usize, matched_is_keyed: bool) {
        if matched_pos == self.cursor_pos {
            self.cursor_pos += 1;
            return;
        }
        let Some(p_idx) = self.parent_slot else {
            return;
        };
        self.structure_changed = true;
        if matched_is_keyed {
            let children = &mut self.sys.slots[p_idx as usize].as_mut().unwrap().children;
            let matched_val = children.remove(matched_pos);
            children.insert(self.cursor_pos, matched_val);
            self.cursor_pos += 1;
            return;
        }
        let mut cur = self.cursor_pos;
        let mut target = matched_pos;
        while cur < target {
            let candidate = self.sys.slots[p_idx as usize].as_ref().unwrap().children[cur];
            let is_keyed = self.sys.slots[candidate as usize]
                .as_ref()
                .is_some_and(|s| s.key.is_explicitly_keyed());
            if is_keyed {
                cur += 1;
            } else {
                self.sys.slots[p_idx as usize]
                    .as_mut()
                    .unwrap()
                    .children
                    .remove(cur);
                self.sys.unmount_slot(candidate);
                target -= 1;
            }
        }
        if target > self.cursor_pos {
            let children = &mut self.sys.slots[p_idx as usize].as_mut().unwrap().children;
            let matched_val = children.remove(target);
            children.insert(self.cursor_pos, matched_val);
        }
        self.cursor_pos += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a11y::{Action, ActionData, ActionRequest, TreeId, slot_to_a11y_id};

    #[test]
    fn spring_all_three_damping_regimes_and_dt_edge_cases() {
        // 1. Underdamped (zeta = 0.35 < 1.0): frame-rate invariance + expected overshoot
        let mut under_coarse = Spring {
            value: 0.0,
            velocity: 0.0,
            target: 100.0,
            angular_freq: 20.0,
            damping_ratio: 0.35,
        };
        let mut under_fine = under_coarse;
        assert!(under_coarse.step(0.18));
        let mut saw_overshoot = false;
        for _ in 0..180 {
            under_fine.step(0.001);
            if under_fine.value > 100.0 {
                saw_overshoot = true;
            }
        }
        assert!(
            (under_coarse.value - under_fine.value).abs() < 1e-3
                && (under_coarse.velocity - under_fine.velocity).abs() < 1e-2,
            "Underdamped ODE mismatch: coarse={under_coarse:?} vs fine={under_fine:?}"
        );
        // Step further to peak to confirm underdamped ringing > target:
        for _ in 0..200 {
            under_fine.step(0.002);
            if under_fine.value > 101.0 {
                saw_overshoot = true;
            }
        }
        assert!(saw_overshoot, "Underdamped spring must overshoot target");

        // 2. Critically damped (zeta = 1.0): frame-rate invariance + zero overshoot
        let mut crit_coarse = Spring::critically_damped(0.0, 3.5);
        crit_coarse.set_target(50.0);
        let mut crit_fine = crit_coarse;
        crit_coarse.step(0.15);
        for _ in 0..150 {
            crit_fine.step(0.001);
            assert!(
                crit_fine.value <= 50.0 + 1e-5,
                "Critically damped spring from 0v0 must never overshoot target"
            );
        }
        assert!(
            (crit_coarse.value - crit_fine.value).abs() < 1e-3
                && (crit_coarse.velocity - crit_fine.velocity).abs() < 1e-2
        );

        // 3. Overdamped (zeta = 2.6 > 1.0): frame-rate invariance + monotone convergence
        let mut over_coarse = Spring {
            value: 0.0,
            velocity: 0.0,
            target: 80.0,
            angular_freq: 24.0,
            damping_ratio: 2.6,
        };
        let mut over_fine = over_coarse;
        over_coarse.step(0.20);
        let mut prev_v = 0.0_f32;
        for _ in 0..200 {
            over_fine.step(0.001);
            assert!(over_fine.value >= prev_v - 1e-5 && over_fine.value <= 80.0 + 1e-5);
            prev_v = over_fine.value;
        }
        assert!(
            (over_coarse.value - over_fine.value).abs() < 1e-3
                && (over_coarse.velocity - over_fine.velocity).abs() < 1e-2,
            "Overdamped ODE mismatch: coarse={over_coarse:?} vs fine={over_fine:?}"
        );

        // 4. dt <= 0.0, NaN, and extreme large dt stability via GuiSystem::step_animations:
        let mut sys = GuiSystem::new();
        let mut target = 0.0_f32;
        let build = |ui: &mut Ui<'_, '_, f32>| {
            let t = *ui.state();
            ui.element(LayoutSpec::col().fixed(100.0, 100.0), PaintStyle::default(), |ui| {
                let _ = ui.use_spring_damped(t, 4.0, 0.4);
                let _ = ui.use_spring_damped(t, 4.0, 1.0);
                let _ = ui.use_spring_damped(t, 4.0, 2.5);
            });
        };
        sys.build_frame(&target, 100.0, 100.0, build);
        target = 100.0;
        sys.build_frame(&target, 100.0, 100.0, build);
        assert_eq!(sys.active_springs.len(), 3);
        assert!(sys.step_animations(0.0));
        assert!(sys.step_animations(-0.5));
        assert!(sys.step_animations(f32::NAN));
        assert_eq!(sys.active_springs.len(), 3);

        // Huge dt settles all 3 regimes cleanly to target with zero NaN:
        assert!(!sys.step_animations(1e30));
        assert!(!sys.has_active_animations());
        sys.build_frame(&target, 100.0, 100.0, build);
        assert!(!sys.has_active_animations());
    }

    #[test]
    fn unmount_recycle_hygiene_and_deep_hover_chain_beyond_24_levels() {
        struct ToggleBranch {
            mount_first: bool,
            target: f32,
        }
        let mut st = ToggleBranch {
            mount_first: true,
            target: 0.0,
        };
        let mut sys = GuiSystem::new();
        sys.a11y_enabled = true;

        let build = |ui: &mut Ui<'_, '_, ToggleBranch>| {
            ui.element(LayoutSpec::col().fixed(300.0, 200.0), PaintStyle::default(), |ui| {
                if ui.state().mount_first {
                    ui.keyed(10, |ui| {
                        ui.element(
                            LayoutSpec::row().fixed(120.0, 40.0),
                            PaintStyle::solid(Color::rgb8(40, 60, 90)),
                            |ui| {
                                ui.set_focusable(true);
                                ui.set_cursor(CursorIcon::Pointer);
                                ui.set_semantics(SemanticsMeta::button("Transient"));
                                ui.on_drag(|_, _| {});
                                let _ = ui.use_spring(ui.state().target, 4.0);
                            },
                        );
                    });
                } else {
                    ui.keyed(20, |ui| {
                        ui.element(
                            LayoutSpec::row().fixed(120.0, 40.0),
                            PaintStyle::solid(Color::rgb8(90, 40, 60)),
                            |ui| {
                                let _ = ui.use_state(|| 999u64);
                            },
                        );
                    });
                }
            });
        };

        sys.build_frame(&st, 300.0, 200.0, build);
        let _ = sys.build_accesskit_update();

        // Trigger active spring, hover, focus, capture, dirty paint, and dirty a11y on keyed(10):
        st.target = 1.0;
        sys.pointer_move(20.0, 20.0, &mut st);
        sys.pointer_down(20.0, 20.0, &mut st);
        sys.build_frame(&st, 300.0, 200.0, build);
        assert!(sys.has_active_animations());
        assert!(sys.focused_slot.is_some());
        assert!(sys.captured_slot.is_some());
        assert_eq!(sys.active_cursor, CursorIcon::Grabbing);

        // Unmount keyed(10) and mount keyed(20) in the same frame (recycling slot & state indices):
        st.mount_first = false;
        sys.build_frame(&st, 300.0, 200.0, build);
        assert!(
            !sys.has_active_animations(),
            "Unmounted spring state index must be purged from active_springs before recycling"
        );
        assert!(sys.focused_slot.is_none());
        assert!(sys.captured_slot.is_none());
        assert_eq!(
            sys.active_cursor,
            CursorIcon::Default,
            "Cursor must reset after captured/hovered element unmounts under stationary pointer"
        );
        assert!(
            sys.dirty_paint_slots
                .iter()
                .all(|&s| sys.slots.get(s as usize).and_then(|x| x.as_ref()).is_some())
        );

        // Deep nesting test (depth = 40 > old 24 cap): every ancestor must report `is_hovered() == true`!
        fn nest(ui: &mut Ui<'_, '_, usize>, depth: usize, hovered_count: &mut usize) {
            ui.element(LayoutSpec::col().fixed(100.0, 100.0), PaintStyle::default(), |ui| {
                if ui.is_hovered() {
                    *hovered_count += 1;
                }
                if depth + 1 < 40 {
                    nest(ui, depth + 1, hovered_count);
                } else {
                    ui.set_cursor(CursorIcon::Pointer);
                }
            });
        }
        let mut deep_sys = GuiSystem::new();
        let mut dummy = 0usize;
        let mut count = 0usize;
        deep_sys.build_frame(&dummy, 100.0, 100.0, |ui| nest(ui, 0, &mut count));
        assert_eq!(count, 0, "Before pointer enters window, 0 nodes should be hovered");
        deep_sys.pointer_move(50.0, 50.0, &mut dummy);
        count = 0;
        deep_sys.build_frame(&dummy, 100.0, 100.0, |ui| nest(ui, 0, &mut count));
        assert_eq!(
            count, 40,
            "All 40 nested ancestors must report is_hovered() == true without depth truncation"
        );
        assert_eq!(deep_sys.active_cursor, CursorIcon::Pointer);
    }

    #[test]
    fn stationary_cursor_refresh_on_scroll_and_layout_shift() {
        struct ShiftState {
            spacer_h: f32,
            clicks: usize,
        }
        let mut st = ShiftState {
            spacer_h: 100.0,
            clicks: 0,
        };
        let mut sys = GuiSystem::new();

        let build = |ui: &mut Ui<'_, '_, ShiftState>| {
            ui.element(LayoutSpec::col().fixed(200.0, 160.0), PaintStyle::default(), |ui| {
                ui.set_scrollable(200.0);
                let sh = ui.state().spacer_h;
                ui.element(LayoutSpec::row().fixed(200.0, sh), PaintStyle::default(), |_| {});
                ui.primary_button("Target", |s, _| s.clicks += 1);
            });
        };

        sys.build_frame(&st, 200.0, 160.0, build);
        // Move cursor to (40, 20): currently over the 100px blank spacer -> Default cursor:
        sys.pointer_move(40.0, 20.0, &mut st);
        assert_eq!(sys.active_cursor, CursorIcon::Default);

        // Scroll down by 95px WITHOUT moving the mouse -> button [y=100..132] shifts to [y=5..37] under (40, 20)!
        assert!(sys.dispatch_scroll(40.0, 20.0, -95.0, &mut st));
        assert_eq!(
            sys.active_cursor,
            CursorIcon::Pointer,
            "dispatch_scroll must refresh active_cursor when button scrolls under stationary cursor"
        );

        // Scroll back to top -> Default cursor again:
        assert!(sys.dispatch_scroll(40.0, 20.0, 95.0, &mut st));
        assert_eq!(sys.active_cursor, CursorIcon::Default);

        // Collapse spacer via state update + build_frame WITHOUT pointer_move -> active_cursor becomes Pointer!
        st.spacer_h = 8.0;
        sys.build_frame(&st, 200.0, 160.0, build);
        assert_eq!(
            sys.active_cursor,
            CursorIcon::Pointer,
            "build_frame must refresh active_cursor when layout shifts button under stationary cursor"
        );
    }

    #[test]
    fn text_input_exact_x_hit_test_kerning_caret_word_delete_and_horizontal_scroll() {
        let mut text = String::from("alpha beta_gamma 🦀 delta");
        let mut sys = GuiSystem::new();

        let build = |ui: &mut Ui<'_, '_, String>| {
            ui.element(LayoutSpec::col().fixed(140.0, 50.0), PaintStyle::default(), |ui| {
                ui.text_input("Type here...", |s| s, |s| s);
            });
        };

        sys.build_frame(&text, 140.0, 50.0, build);

        // Click at exact x of byte 6 ('b' in "beta_gamma") at scroll_x = 0:
        let fid = sys.fonts.ui_font;
        let target_byte = 6usize; // start of "beta_gamma"
        let x_at_6 = sys.fonts.byte_offset_to_x(&text, target_byte, fid, 14.0);
        // Click slightly into 'b' (+1.5px past left edge of 'b', plus 10px input pad_l):
        assert!(sys.pointer_down(10.0 + x_at_6 + 1.5, 17.0, &mut text));
        sys.pointer_up();
        assert!(sys.dispatch_key(&KeyInput::Text("X".to_owned()), &mut text));
        assert_eq!(text, "alpha Xbeta_gamma 🦀 delta");

        // Verify single unbroken text node (only 1 FLAG_HAS_TEXT node in the entire tree!):
        sys.build_frame(&text, 140.0, 50.0, build);
        let text_nodes: Vec<_> = (0..sys.layout.topo.len())
            .filter(|&i| {
                (sys.layout.topo[i].flags & crate::layout::FLAG_ALIVE) != 0
                    && (sys.layout.topo[i].flags & crate::layout::FLAG_HAS_TEXT) != 0
            })
            .collect();
        assert_eq!(
            text_nodes.len(),
            1,
            "text_input must render a single un-split shaped text node so kerning never shifts"
        );

        // Jump to End -> narrow 140px box (usable ~118px) must horizontally scroll to keep caret visible!
        assert!(sys.dispatch_key(&KeyInput::End, &mut text));
        sys.build_frame(&text, 140.0, 50.0, build);
        let caret_box = sys
            .layout
            .boxes
            .iter()
            .find(|b| (b.w - 2.0).abs() < 1e-3 && (b.h - 16.0).abs() < 1e-3)
            .expect("2x16 caret node");
        assert!(
            caret_box.x >= 10.0 - 1e-3 && caret_box.x + caret_box.w <= 130.0 + 1e-3,
            "Caret ({}) must stay inside horizontal viewport [10..130] when text overflows!",
            caret_box.x
        );
        assert!(
            sys.layout.boxes[text_nodes[0]].x < 10.0,
            "Text node X ({}) must shift left (< 10.0) when scrolled horizontally",
            sys.layout.boxes[text_nodes[0]].x
        );

        // Test Ctrl+Backspace across words and 4-byte UTF-8 Emoji:
        // "alpha Xbeta_gamma 🦀 delta" -> deletes "delta" -> "alpha Xbeta_gamma 🦀 "
        assert!(sys.dispatch_key(&KeyInput::CtrlBackspace, &mut text));
        assert_eq!(text, "alpha Xbeta_gamma 🦀 ");
        // Next Ctrl+Backspace skips trailing space and deletes 4-byte "🦀":
        assert!(sys.dispatch_key(&KeyInput::CtrlBackspace, &mut text));
        assert_eq!(text, "alpha Xbeta_gamma ");
        // Next Ctrl+Backspace deletes identifier "Xbeta_gamma" (including '_'):
        assert!(sys.dispatch_key(&KeyInput::CtrlBackspace, &mut text));
        assert_eq!(text, "alpha ");
        // Home + Ctrl+Delete deletes "alpha":
        assert!(sys.dispatch_key(&KeyInput::Home, &mut text));
        assert!(sys.dispatch_key(&KeyInput::CtrlDelete, &mut text));
        assert_eq!(text, " ");
    }

    #[test]
    fn scroll_area_auto_overflow_and_accesskit_increment_decrement() {
        struct ScrollDemo {
            count: usize,
            scroll_y: f64,
            slider_v: f32,
            input_txt: String,
        }
        let mut st = ScrollDemo {
            count: 2,
            scroll_y: 0.0,
            slider_v: 0.50,
            input_txt: String::from("init"),
        };
        let mut sys = GuiSystem::new();
        sys.a11y_enabled = true;

        let build = |ui: &mut Ui<'_, '_, ScrollDemo>| {
            let mut slider_nid = NodeId::NONE;
            let _ = slider_nid;
            ui.element(LayoutSpec::col().fixed(300.0, 260.0).gap(8.0), PaintStyle::default(), |ui| {
                let sv = ui.state().slider_v;
                slider_nid = ui.slider("Gain", sv, 0.0..=1.0, 200.0, |s, v| s.slider_v = v);
                ui.text_input("Name", |s| &s.input_txt, |s| &mut s.input_txt);
                let n = ui.state().count;
                ui.scroll_area(
                    "Items Scroll",
                    |s| s.scroll_y,
                    |s, y| s.scroll_y = y,
                    LayoutSpec::col().width_grow(1.0).height_fixed(150.0).pad(6.0).gap(4.0),
                    PaintStyle::solid(Color::rgb8(25, 28, 36)),
                    |ui| {
                        for i in 0..n {
                            ui.keyed(i as u64 + 1, |ui| {
                                ui.element(
                                    LayoutSpec::row().width_grow(1.0).height_fixed(28.0),
                                    PaintStyle::solid(Color::rgb8(40, 45, 60)),
                                    |_| {},
                                );
                            });
                        }
                    },
                );
            });
        };

        // 1. With 2 items (content_h = 6 + 28 + 4 + 28 + 6 = 72 <= 150): no overflow, wheel is no-op!
        sys.build_frame(&st, 300.0, 260.0, build);
        assert!(!sys.dispatch_scroll(50.0, 150.0, -50.0, &mut st));
        assert_eq!(st.scroll_y, 0.0);

        // 2. Grow to 20 items (content_h = 12 + 20*28 + 19*4 = 648 > 150 -> max_scroll = 498.0):
        //    Verify Frame-0 of overflow already activates 14px scrollbar and exact scroll clamping!
        st.count = 20;
        sys.build_frame(&st, 300.0, 260.0, build);
        assert!(sys.dispatch_scroll(50.0, 150.0, -100.0, &mut st));
        assert!((st.scroll_y - 100.0).abs() < 1e-4);
        assert!(sys.dispatch_scroll(50.0, 150.0, -9999.0, &mut st));
        assert!(
            (st.scroll_y - 498.0).abs() < 1e-3,
            "scroll_area must clamp to exact content_h - viewport_h (498.0), got {}",
            st.scroll_y
        );

        // 3. Verify AccessKit Increment/Decrement on Slider & ScrollBar, plus SetValue on TextInput:
        sys.build_frame(&st, 300.0, 260.0, build);
        let tree_up = sys.build_accesskit_update().expect("AccessKit tree");
        let slider_a11y = tree_up
            .nodes
            .iter()
            .find(|(_, n)| n.role() == A11yRole::Slider)
            .map(|(id, _)| *id)
            .expect("Slider a11y node");
        let scrollbar_a11y = tree_up
            .nodes
            .iter()
            .find(|(_, n)| n.role() == A11yRole::ScrollBar)
            .map(|(id, _)| *id)
            .expect("ScrollBar a11y node");
        let input_a11y = tree_up
            .nodes
            .iter()
            .find(|(_, n)| n.role() == A11yRole::TextInput)
            .map(|(id, _)| *id)
            .expect("TextInput a11y node");

        assert!(sys.handle_accesskit_action(
            &ActionRequest {
                action: Action::Increment,
                target_tree: TreeId::ROOT,
                target_node: slider_a11y,
                data: None,
            },
            &mut st,
        ));
        assert!((st.slider_v - 0.52).abs() < 1e-4);

        assert!(sys.handle_accesskit_action(
            &ActionRequest {
                action: Action::Decrement,
                target_tree: TreeId::ROOT,
                target_node: scrollbar_a11y,
                data: None,
            },
            &mut st,
        ));
        assert!(
            (st.scroll_y - (498.0 - 38.0)).abs() < 1e-3,
            "AccessKit Decrement on ScrollBar must decrease scroll_y by 38px, got {}",
            st.scroll_y
        );
        assert!(sys.handle_accesskit_action(
            &ActionRequest {
                action: Action::Increment,
                target_tree: TreeId::ROOT,
                target_node: scrollbar_a11y,
                data: None,
            },
            &mut st,
        ));
        assert!((st.scroll_y - 498.0).abs() < 1e-3);

        assert!(sys.handle_accesskit_action(
            &ActionRequest {
                action: Action::SetValue,
                target_tree: TreeId::ROOT,
                target_node: input_a11y,
                data: Some(ActionData::Value("accessible 🦀".into())),
            },
            &mut st,
        ));
        assert_eq!(st.input_txt, "accessible 🦀");
        let _ = slot_to_a11y_id(0);
    }

    #[test]
    fn button_pressed_capture_steady_hover_zero_dirty_paint_and_theme_contrast() {
        let mut clicks = 0usize;
        let mut sys = GuiSystem::new();

        let build = |ui: &mut Ui<'_, '_, usize>| {
            ui.element(LayoutSpec::col().fixed(240.0, 120.0).pad(16.0).gap(8.0), PaintStyle::default(), |ui| {
                ui.primary_button("Run", |c, _| *c += 1);
                ui.badge("READY", ui.theme().success);
            });
        };

        sys.build_frame(&clicks, 240.0, 120.0, build);

        // Hover button -> 1 in-place paint update on transition frame, then 0 updates on steady-state hover!
        sys.pointer_move(30.0, 24.0, &mut clicks);
        sys.build_frame(&clicks, 240.0, 120.0, build);
        assert_eq!(sys.paint_updates_last_frame, 1);

        sys.build_frame(&clicks, 240.0, 120.0, build);
        assert_eq!(
            sys.paint_updates_last_frame, 0,
            "Steady-state hover frame must emit 0 redundant paint updates"
        );
        assert!(sys.dirty_prim_ranges.is_empty());

        // Press mouse button down -> captured_slot active (pressed state), cursor stays Pointer, release restores:
        assert!(sys.pointer_down(30.0, 24.0, &mut clicks));
        assert_eq!(clicks, 1);
        assert!(sys.captured_slot.is_some());
        assert_eq!(sys.active_cursor, CursorIcon::Pointer);
        sys.build_frame(&clicks, 240.0, 120.0, build);
        assert_eq!(sys.paint_updates_last_frame, 1, "Pressed state shifts button lightness");
        assert!(sys.pointer_up());
        assert!(sys.captured_slot.is_none());

        // Verify Light & Dark theme perceptual contrast across hues (including bright yellow 95 deg):
        for is_dark in [true, false] {
            for hue in [25.0, 95.0, 150.0, 258.0] {
                let th = Theme::with_accent_hue(is_dark, hue);
                let canvas_l = OklabLuma::of(th.bg_canvas);
                let text_l = OklabLuma::of(th.text_primary);
                assert!(
                    (canvas_l - text_l).abs() > 0.55,
                    "Primary text vs canvas OKLab L contrast too low for dark={is_dark}, hue={hue}"
                );
                let accent_l = OklabLuma::of(th.accent);
                let on_acc_l = OklabLuma::of(th.text_on_accent);
                assert!(
                    (accent_l - on_acc_l).abs() > 0.25,
                    "text_on_accent vs accent contrast too low for dark={is_dark}, hue={hue}"
                );
            }
        }

        // Verify inline color emojis inside SubpixelLcd `ui_text` emit `PrimKind::ImageRgba` with
        // untinted `Color::WHITE` rather than `PrimKind::GlyphLcd` multiplied by dark/light `text_color`:
        let mut emoji_sys = GuiSystem::with_system_fonts();
        if emoji_sys.fonts.font_count() >= 2 {
            let dark_ink = Color::rgba8(18, 21, 30, 140);
            emoji_sys.build_frame(&0usize, 240.0, 60.0, |ui| {
                ui.element(LayoutSpec::col().fixed(240.0, 60.0), PaintStyle::solid(Color::WHITE), |ui| {
                    ui.ui_text("Ship 🦀", 16.0, dark_ink);
                });
            });
            for prim in &emoji_sys.primitives {
                if prim.kind() == PrimKind::ImageRgba {
                    assert_eq!(
                        prim.packed[0],
                        Color::rgba8(255, 255, 255, 140).0,
                        "Inline color emoji primitive must preserve true white RGB with text_color alpha"
                    );
                }
            }
        }
    }

    #[test]
    fn shrinking_shadow_damage_rect_builtin_scroll_persistence_and_headless_ceil() {
        // 1. Shrinking shadow blur_sigma via set_paint must union `old_fp` into `current_frame_damage`:
        let mut sigma = 14.0_f32;
        let mut sys = GuiSystem::new();
        sys.clear_color = Color::rgb8(240, 240, 245);
        let build_sh = |ui: &mut Ui<'_, '_, f32>| {
            let s = *ui.state();
            ui.element(LayoutSpec::col().fixed(200.0, 200.0).pad(60.0), PaintStyle::default(), |ui| {
                ui.element(
                    LayoutSpec::col().fixed(60.0, 60.0),
                    PaintStyle::solid(Color::rgb8(40, 80, 200))
                        .rounded(8.0)
                        .drop_shadow(s, [0.0, 4.0], Color::rgba8(0, 0, 0, 200)),
                    |_| {},
                );
            });
        };
        sys.build_frame(&sigma, 200.0, 200.0, build_sh);
        let mut cpu = crate::gpu::cpu::CpuRenderer::new(200, 200);
        sys.render_to_backend(&mut cpu, 200, 200, 1).unwrap();
        // Outer shadow pixel at (90, 134) (10px below offset shadow edge y=124) is darkened when sigma == 14.0:
        assert!(cpu.pixel_at(90, 134)[0] < 225);

        // Shrink sigma from 14.0 -> 1.0 (paint-only patch): old_fp must be inside DamageRect!
        sigma = 1.0;
        sys.build_frame(&sigma, 200.0, 200.0, build_sh);
        let dmg = sys.current_frame_damage.expect("damage rect");
        assert!(
            dmg.y + dmg.h >= 165,
            "DamageRect ({dmg:?}) must cover old larger shadow footprint!"
        );
        sys.render_to_backend(&mut cpu, 200, 200, 1).unwrap();
        assert!(
            cpu.pixel_at(90, 134)[0] >= 239,
            "Shrunk shadow outer ring must be cleared back to clear_color without ghosting!"
        );

        // 2. Built-in `set_scrollable(max_scroll_y)` must persist `scroll_y` across `build_frame` and clamp on shrink:
        let mut max_s = 200.0_f32;
        let mut ssys = GuiSystem::new();
        let build_sc = |ui: &mut Ui<'_, '_, f32>| {
            let ms = *ui.state();
            ui.element(LayoutSpec::col().fixed(160.0, 80.0), PaintStyle::default(), |ui| {
                ui.set_scrollable(ms);
                ui.element(LayoutSpec::row().fixed(160.0, 300.0), PaintStyle::default(), |_| {});
            });
        };
        ssys.build_frame(&max_s, 160.0, 80.0, build_sc);
        assert!(ssys.dispatch_scroll(40.0, 40.0, -95.0, &mut max_s));
        // Rebuild frame -> scroll_y (95.0) must NOT be wiped back to 0.0!
        ssys.build_frame(&max_s, 160.0, 80.0, build_sc);
        assert!(
            (ssys.layout.boxes[1].y - (-95.0)).abs() < 1e-4,
            "Built-in set_scrollable must preserve scroll_y across build_frame, got child y={}",
            ssys.layout.boxes[1].y
        );
        // Shrink max_scroll_y to 40.0 -> scroll_y clamps to 40.0:
        max_s = 40.0;
        ssys.build_frame(&max_s, 160.0, 80.0, build_sc);
        assert!((ssys.layout.boxes[1].y - (-40.0)).abs() < 1e-4);

        // 3. `render_headless_rgba8` fractional HiDPI `.ceil()` parity (`101 * 1.25 = 126.25 -> 127`):
        let (pw, ph, rgba) = crate::app::render_headless_rgba8(101, 81, 1.25, false, &0usize, |ui| {
            ui.element(LayoutSpec::col().grow(1.0), PaintStyle::solid(Color::BLACK), |_| {});
        });
        assert_eq!((pw, ph), (127, 102));
        assert_eq!(rgba.len(), 127 * 102 * 4);
    }

    #[test]
    fn text_input_selection_highlight_clipboard_drag_and_double_click() {
        let mut text = String::from("café 🦀 rust_v2 end");
        let mut sys = GuiSystem::new();
        sys.a11y_enabled = true;

        let build = |ui: &mut Ui<'_, '_, String>| {
            ui.element(LayoutSpec::col().fixed(320.0, 54.0), PaintStyle::default(), |ui| {
                ui.text_input("Search...", |s| s, |s| s);
            });
        };

        sys.build_frame(&text, 320.0, 54.0, build);
        let a11y_init = sys.build_accesskit_update().unwrap();
        let input_a11y = a11y_init
            .nodes
            .iter()
            .find(|(_, n)| n.role() == A11yRole::TextInput)
            .map(|(id, _)| *id)
            .unwrap();

        // Focus via Tab -> initial collapsed caret at end of text:
        assert!(sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut text));
        let st0 = sys.focused_text_edit_state().expect("focused TextEditState");
        assert_eq!(st0.cursor, text.len());
        assert_eq!(st0.anchor, text.len());

        // 1. `Ctrl+A` (`SelectAll`) MUST select `[0..text.len()]` WITHOUT clearing `text`!
        assert!(sys.dispatch_key(&KeyInput::SelectAll, &mut text));
        assert_eq!(text, "café 🦀 rust_v2 end", "Ctrl+A must NEVER delete text!");
        let st_all = sys.focused_text_edit_state().unwrap();
        assert_eq!((st_all.anchor, st_all.cursor), (0, text.len()));

        // Rebuild and verify translucent accent highlight quad (`th.accent.with_alpha(0.28)`) is emitted BEHIND text glyphs:
        sys.build_frame(&text, 320.0, 54.0, build);
        let expected_sel_col = sys.theme.accent.with_alpha(0.28);
        let sel_prim_idx = sys
            .primitives
            .iter()
            .position(|p| p.kind() == PrimKind::Quad && p.packed[0] == expected_sel_col.0 && p.bounds[2] > 20.0)
            .expect("selection highlight quad primitive");
        let first_glyph_idx = sys
            .primitives
            .iter()
            .position(|p| matches!(p.kind(), PrimKind::GlyphGray | PrimKind::GlyphLcd | PrimKind::ImageRgba))
            .expect("text glyph primitive");
        assert!(
            sel_prim_idx < first_glyph_idx,
            "Selection highlight quad ({sel_prim_idx}) must paint behind text glyphs ({first_glyph_idx})"
        );

        // 2. `Copy` copies full selection; `ArrowLeft` collapses selection to `0` (`sel_min`);
        //    `Copy` with empty selection does NOT clobber existing clipboard!
        assert!(sys.dispatch_key(&KeyInput::Copy, &mut text));
        assert_eq!(sys.clipboard_text(), "café 🦀 rust_v2 end");
        assert!(sys.dispatch_key(&KeyInput::ArrowLeft, &mut text));
        assert_eq!(sys.focused_text_edit_state().unwrap().selection_range(&text), 0..0);
        sys.set_clipboard_text("preserved");
        assert!(sys.dispatch_key(&KeyInput::Copy, &mut text));
        assert_eq!(sys.clipboard_text(), "preserved", "Empty Copy must not clobber clipboard");

        // 3. `Ctrl+Shift+ArrowRight` twice selects `"café 🦀 "` (multi-byte `'é'` + 4-byte `'🦀'`), then `Cut`:
        assert!(sys.dispatch_key(&KeyInput::CtrlShiftArrowRight, &mut text));
        assert_eq!(&text[sys.focused_text_edit_state().unwrap().selection_range(&text)], "café");
        assert!(sys.dispatch_key(&KeyInput::ShiftArrowRight, &mut text)); // ' '
        assert!(sys.dispatch_key(&KeyInput::ShiftArrowRight, &mut text)); // '🦀' (4 bytes!)
        assert!(sys.dispatch_key(&KeyInput::ShiftArrowRight, &mut text)); // ' '
        assert_eq!(&text[sys.focused_text_edit_state().unwrap().selection_range(&text)], "café 🦀 ");
        assert!(sys.dispatch_key(&KeyInput::Cut, &mut text));
        assert_eq!(sys.clipboard_text(), "café 🦀 ");
        assert_eq!(text, "rust_v2 end");

        // 4. `End` -> `Shift+Home` -> `Paste` replaces entire selection; `Shift+End` + typing replaces tail:
        assert!(sys.dispatch_key(&KeyInput::End, &mut text));
        assert!(sys.paste_clipboard(&mut text));
        assert_eq!(text, "rust_v2 endcafé 🦀 ");
        assert!(sys.dispatch_key(&KeyInput::CtrlShiftArrowLeft, &mut text)); // selects `"🦀 "`
        assert!(sys.dispatch_key(&KeyInput::Text("🚀".to_owned()), &mut text));
        assert_eq!(text, "rust_v2 endcafé 🚀");

        // 5. Mouse click + drag selection (`anchor` pinned at click X, `cursor` tracks drag X!):
        sys.build_frame(&text, 320.0, 54.0, build);
        let fid = sys.fonts.ui_font;
        let x0 = sys.fonts.byte_offset_to_x(&text, 0, fid, 14.0);
        let x7 = sys.fonts.byte_offset_to_x(&text, 7, fid, 14.0); // end of "rust_v2"
        assert!(sys.pointer_down(10.0 + x0 + 0.2, 17.0, &mut text));
        assert!(sys.pointer_move(10.0 + x7 - 0.1, 17.0, &mut text));
        sys.pointer_up();
        assert_eq!(
            &text[sys.focused_text_edit_state().unwrap().selection_range(&text)],
            "rust_v2"
        );

        // 6. Double-click word selection on `"rust_v2"` (and verify initial down-drag does NOT clobber word range!):
        let x_mid = sys.fonts.byte_offset_to_x(&text, 3, fid, 14.0);
        assert!(sys.pointer_down(10.0 + x_mid + 1.0, 17.0, &mut text));
        sys.pointer_up();
        assert!(sys.pointer_down(10.0 + x_mid + 1.0, 17.0, &mut text));
        sys.pointer_up();
        assert_eq!(
            &text[sys.focused_text_edit_state().unwrap().selection_range(&text)],
            "rust_v2",
            "Double-click must select whole identifier word including '_'"
        );

        // 7. AccessKit `SetValue` with both non-empty and empty `""` string:
        assert!(sys.handle_accesskit_action(
            &ActionRequest {
                action: Action::SetValue,
                target_tree: TreeId::ROOT,
                target_node: input_a11y,
                data: Some(ActionData::Value("".into())),
            },
            &mut text,
        ));
        assert_eq!(text, "");
    }

    #[test]
    fn coalesced_dirty_prim_ranges_hover_tooltip_flip_and_randomized_slot_reconciliation() {
        // 1. Coalescing adjacent, overlapping, and out-of-order `dirty_prim_ranges`:
        struct MultiBtn(usize);
        let mut st = MultiBtn(0);
        let mut sys = GuiSystem::new();
        let build_btns = |ui: &mut Ui<'_, '_, MultiBtn>| {
            ui.element(LayoutSpec::row().fixed(400.0, 80.0).gap(4.0), PaintStyle::default(), |ui| {
                let flag = ui.state().0;
                for i in 0..4usize {
                    let c = if (flag & (1 << i)) != 0 {
                        Color::rgb8(200, 80, 90)
                    } else {
                        Color::rgb8(40, 50, 70)
                    };
                    ui.element(LayoutSpec::col().fixed(40.0, 30.0), PaintStyle::solid(c), |_| {});
                }
            });
        };
        sys.build_frame(&st, 400.0, 80.0, build_btns);
        // Mutate all 4 contiguous sibling quads simultaneously -> must coalesce into ONE single range!
        st.0 = 0b1111;
        sys.build_frame(&st, 400.0, 80.0, build_btns);
        assert_eq!(
            sys.dirty_prim_ranges.len(),
            1,
            "4 contiguous dirty sibling quads must coalesce into 1 ReBAR memcpy range: {:?}",
            sys.dirty_prim_ranges
        );
        assert_eq!(sys.dirty_prim_ranges[0], CopyRange { start: 1, end: 5 });

        // 2. `ui.tooltip` inside a clipped header near top edge (`y = 6`):
        //    Must auto-flip below button via Pass 4, escape parent `FLAG_CLIPS_CHILDREN`, and never flicker on hover!
        let mut tsys = GuiSystem::new();
        let mut clicks = 0usize;
        let build_tip = |ui: &mut Ui<'_, '_, usize>| {
            ui.element(LayoutSpec::col().fixed(300.0, 180.0), PaintStyle::solid(Color::BLACK), |ui| {
                ui.element(
                    LayoutSpec::row().fixed(300.0, 40.0).pad_xy(12.0, 4.0),
                    PaintStyle::solid(Color::rgb8(30, 34, 46)),
                    |ui| {
                        ui.set_clips_children(true);
                        let btn = ui.primary_button("Inspect", |c, _| *c += 1);
                        ui.tooltip(btn, "Flip below top edge without clip!");
                    },
                );
                ui.element(
                    LayoutSpec::col().width_grow(1.0).height_grow(1.0),
                    PaintStyle::solid(Color::rgb8(20, 22, 30)),
                    |_| {},
                );
            });
        };
        tsys.build_frame(&clicks, 300.0, 180.0, build_tip);
        assert!(tsys.layout.anchors.is_empty(), "Tooltip must not mount before target is hovered");

        // Hover over button `(30, 18)`:
        assert!(tsys.pointer_move(30.0, 18.0, &mut clicks));
        for _ in 0..4 {
            tsys.build_frame(&clicks, 300.0, 180.0, build_tip);
            assert_eq!(tsys.layout.anchors.len(), 1, "Tooltip must remain stably mounted with 0 hover flicker");
        }
        let ov_nid = tsys.layout.anchors[0].overlay_node;
        let ov_box = tsys.layout.boxes[ov_nid.idx()];
        assert!(
            ov_box.y >= 34.0,
            "TopCenter tooltip near y=4 must auto-flip below button via Pass 4, got y={}",
            ov_box.y
        );
        // Verify tooltip pill quad escapes parent header clip (`CLIP_NONE`), while inner glyphs
        // are hardware-clipped strictly to the tooltip's own rounded pill clip (`meta[0] == CLIP_NONE`):
        let ov_slot = tsys.node_to_slot[ov_nid.idx()];
        let ov_range = tsys.slots[ov_slot as usize].as_ref().unwrap().prim_range;
        assert_eq!(tsys.primitives[ov_range.start as usize].packed[2] & CLIP_NONE, CLIP_NONE);
        let last_prim = tsys.primitives.last().unwrap();
        let glyph_clip_id = last_prim.packed[2] & CLIP_NONE;
        assert_ne!(glyph_clip_id, CLIP_NONE, "Tooltip glyphs must be clipped to tooltip pill");
        assert_eq!(
            tsys.clips[glyph_clip_id as usize].meta[0],
            CLIP_NONE,
            "Tooltip pill clip must escape parent header clip"
        );
        assert_eq!(
            tsys.layout.last_metrics.boundaries_solved, 0,
            "Steady-state hover tooltip must solve 0 layout boundaries"
        );

        // Verify wide multi-word & unbroken tooltip strings wrap cleanly inside `max_w` without right-edge spill:
        let build_long_tip = |ui: &mut Ui<'_, '_, usize>| {
            ui.element(LayoutSpec::col().fixed(400.0, 200.0).pad(12.0), PaintStyle::solid(Color::BLACK), |ui| {
                let btn = ui.primary_button("Wide", |c, _| *c += 1);
                ui.tooltip(
                    btn,
                    "Mutate 2 tasks simultaneously via SlotPool::get_disjoint_mut without overflowing the right rounded tooltip border",
                );
            });
        };
        let mut wsys = GuiSystem::new();
        wsys.build_frame(&clicks, 400.0, 200.0, build_long_tip);
        let _ = wsys.pointer_move(30.0, 24.0, &mut clicks);
        wsys.build_frame(&clicks, 400.0, 200.0, build_long_tip);
        let w_ov = wsys.layout.anchors[0].overlay_node;
        let w_box = wsys.layout.boxes[w_ov.idx()];
        assert!(w_box.w <= 360.0 && w_box.h >= 40.0, "Long tooltip must wrap within max_w=360, got {w_box:?}");
        let w_txt_nid = wsys.layout.topo[w_ov.idx()].first_child;
        let w_txt_box = wsys.layout.boxes[w_txt_nid.idx()];
        assert!(
            w_txt_box.x + w_txt_box.w <= w_box.x + w_box.w - 9.5,
            "Tooltip text box {:?} must never spill past pill border {:?}",
            w_txt_box,
            w_box
        );

        // 3. Randomized stress test: unkeyed `for` loop + conditional `if` + `keyed` reordering + hook type toggles:
        struct StressState {
            mask: u32,
            order_rev: bool,
            loop_len: usize,
        }
        let mut sst = StressState {
            mask: 0b1010_0101,
            order_rev: false,
            loop_len: 4,
        };
        let mut rsys = GuiSystem::new();
        let build_stress = |ui: &mut Ui<'_, '_, StressState>| {
            let m = ui.state().mask;
            let rev = ui.state().order_rev;
            let n = ui.state().loop_len;
            ui.element(LayoutSpec::col().fixed(400.0, 400.0), PaintStyle::default(), |ui| {
                // Empty `ui.keyed` when bit 0 is clear (must NEVER leak pending_user_key to unkeyed loop!):
                ui.keyed(0x9e37_79b9_7f4a_7c15, |ui| {
                    if (m & 1) != 0 {
                        ui.badge("MAGIC_KEY", Color::WHITE);
                    }
                });

                let keys = if rev { [20u64, 10u64] } else { [10u64, 20u64] };
                for (k_pos, &k_id) in keys.iter().enumerate() {
                    ui.keyed(k_id, |ui| {
                        ui.element(LayoutSpec::row().fixed(50.0, 16.0), PaintStyle::default(), |ui| {
                            let h = ui.use_state(|| k_id);
                            assert_eq!(*ui.read_local(h), k_id, "Keyed state cross-wired!");
                            ui.label_usize(k_id as usize, 12.0, 16.0, Color::WHITE);
                        });
                    });

                    if k_pos == 0 {
                        // Unkeyed loop + interleaved flat conditional siblings + stateful unkeyed sibling
                        // sandwiched BETWEEN the two reordered keyed items!
                        for i in 0..n {
                            ui.element(LayoutSpec::row().fixed(120.0, 20.0), PaintStyle::default(), |ui| {
                                let iter_id = ui.use_state(|| i);
                                assert_eq!(
                                    *ui.read_local(iter_id),
                                    i,
                                    "Unkeyed loop item {i} cross-wired with iteration {}",
                                    *ui.read_local(iter_id)
                                );
                                if ((m >> (i + 1)) & 1) != 0 {
                                    let _sp = ui.use_spring(1.0, 4.0);
                                    let cap = i;
                                    ui.on_click(move |s, _| s.mask ^= 1 << cap);
                                } else {
                                    let h = ui.use_state(|| format!("str_{i}"));
                                    assert_eq!(ui.read_local(h).as_str(), format!("str_{i}"));
                                    let cap2 = (i, 99usize);
                                    ui.on_click(move |s, _| s.mask ^= (cap2.0 + cap2.1) as u32);
                                }
                                if ((m >> (i + 9)) & 1) != 0 {
                                    ui.element(LayoutSpec::row().fixed(40.0, 12.0), PaintStyle::default(), |ui| {
                                        let inner_tag = ui.use_state(|| 1000 + i);
                                        assert_eq!(*ui.read_local(inner_tag), 1000 + i);
                                    });
                                }
                            });
                            // Flat unkeyed conditional sibling directly interleaved between unkeyed loop items:
                            if ((m >> (i + 5)) & 1) != 0 {
                                ui.element(LayoutSpec::row().fixed(80.0, 16.0), PaintStyle::default(), |ui| {
                                    let flat_tag = ui.use_state(|| 5000usize + i);
                                    assert!(*ui.read_local(flat_tag) >= 5000);
                                });
                            }
                        }

                        ui.element(LayoutSpec::row().fixed(60.0, 18.0), PaintStyle::default(), |ui| {
                            let anchor_st = ui.use_state(|| 77777u64);
                            assert_eq!(*ui.read_local(anchor_st), 77777);
                        });
                    }
                }
            });
        };

        let mut rng = 0x1234_5678_9abc_def0_u64;
        for _ in 0..1000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            sst.mask = rng as u32;
            sst.order_rev = (rng & 0x10000) != 0;
            sst.loop_len = ((rng >> 20) as usize) % 6;
            rsys.build_frame(&sst, 400.0, 400.0, build_stress);
        }

        // 4. Zero-allocation steady-state `vector_canvas` scratch reuse:
        let mut vsys = GuiSystem::new();
        let pts = [[8.0, 28.0], [120.0, 10.0], [240.0, 48.0], [360.0, 18.0]];
        for rev in 1..=3u64 {
            vsys.build_frame(&rev, 400.0, 80.0, |ui| {
                let r = *ui.state();
                ui.vector_canvas(380, 56, r, Color::rgb8(90, 180, 255), FillRule::NonZero, |p| {
                    p.stroke_polyline(&pts, 2.5);
                });
            });
            vsys.fonts.atlas.dirty_rects.clear();
        }
        let mask_ptr_before = vsys.vector_mask_scratch.as_ptr();
        let mask_cap_before = vsys.vector_mask_scratch.capacity();
        for rev in 4..=64u64 {
            let dy = ((rev % 9) as f32) - 4.0;
            let anim_pts = [
                [8.0, 28.0 + dy],
                [120.0, 10.0 - dy],
                [240.0, 48.0 + dy],
                [360.0, 18.0 - dy],
            ];
            vsys.build_frame(&rev, 400.0, 80.0, |ui| {
                let r = *ui.state();
                ui.vector_canvas(380, 56, r, Color::rgb8(90, 180, 255), FillRule::NonZero, |p| {
                    p.stroke_polyline(&anim_pts, 2.5);
                });
            });
            vsys.fonts.atlas.dirty_rects.clear();
        }
        assert_eq!(vsys.vector_mask_scratch.as_ptr(), mask_ptr_before);
        assert_eq!(vsys.vector_mask_scratch.capacity(), mask_cap_before);
        assert_eq!(vsys.fonts.atlas.generation(), 1);

        // 5. Mid-frame `GlyphAtlas::try_grow` (`64 -> 128 -> 256`) UV coherence across earlier icons & glyphs:
        let mut gsys = GuiSystem::new();
        gsys.fonts.atlas = crate::text::GlyphAtlas::new(64, 64);
        gsys.build_frame(&0usize, 320.0, 200.0, |ui| {
            ui.element(LayoutSpec::col().fixed(320.0, 200.0).gap(4.0), PaintStyle::solid(Color::BLACK), |ui| {
                // Early icon uploaded while atlas is `64x64`:
                ui.vector_icon(24, 24, Color::WHITE, FillRule::NonZero, |p| {
                    p.move_to(0.0, 0.0)
                        .line_to(24.0, 0.0)
                        .line_to(24.0, 24.0)
                        .line_to(0.0, 24.0)
                        .close();
                });
                // Early glyphs rasterized before later icons/glyphs force `try_grow`:
                ui.label("EARLY GLYPHS", 22.0, 26.0, Color::WHITE);
                // Subsequent icons + large glyphs trigger `64 -> 128 -> 256` mid-frame growth!
                for k in 0..8u64 {
                    ui.keyed(k + 1, |ui| {
                        ui.vector_canvas(40, 40, k + 1, Color::WHITE, FillRule::NonZero, |p| {
                            p.circle(20.0, 20.0, 18.0);
                        });
                    });
                }
                ui.label("LATE BIG GLYPHS XYZ 12345", 28.0, 32.0, Color::WHITE);
            });
        });
        assert!(
            gsys.fonts.atlas.width > 64,
            "Expected mid-frame GlyphAtlas::try_grow past 64x64, got {}",
            gsys.fonts.atlas.width
        );
        let aw = gsys.fonts.atlas.width as f32;
        // Early 24x24 icon at `x=4, y=0` must have exact `28.0 / aw` `u1` (NOT stale `28.0 / 64.0`!):
        let early_icon_prim = gsys
            .primitives
            .iter()
            .find(|p| p.kind() == PrimKind::GlyphGray && (p.bounds[2] - 24.0).abs() < 1e-3 && (p.bounds[3] - 24.0).abs() < 1e-3)
            .expect("early 24x24 vector_icon primitive");
        assert!(
            ((early_icon_prim.uv_or_extra[2] - early_icon_prim.uv_or_extra[0]) - (24.0 / aw)).abs() < 1e-5,
            "Early vector_icon UV span ({}) must match post-growth 24.0/{aw}, not stale pre-growth atlas width!",
            early_icon_prim.uv_or_extra[2] - early_icon_prim.uv_or_extra[0]
        );
        // Render on CPU reference backend and confirm early solid 24x24 white icon is 100% solid white at `(12, 12)`:
        let mut gcpu = crate::gpu::cpu::CpuRenderer::new(320, 200);
        gsys.render_to_backend(&mut gcpu, 320, 200, 1).unwrap();
        assert_eq!(gcpu.pixel_at(12, 12), [255, 255, 255, 255]);
    }

    #[test]
    fn fractional_scale_physical_primitive_seams_border_snapping_and_incremental_patch_parity() {
        let scales: [f32; 10] = [
            1.0,
            1.125,
            1.25,
            160.0 / 120.0,
            1.375,
            1.5,
            200.0 / 120.0,
            1.75,
            2.0,
            2.25,
        ];

        for &sf in &scales {
            // 1. 50 contiguous `gap=0` rows in `Col` and `VirtualList`: 0.0 physical seam & 0 canary bleed!
            let canary = Color::rgb8(255, 0, 255);
            let row_fill = Color::rgb8(36, 44, 62);
            let mut sys = GuiSystem::new();
            sys.set_scale_factor(sf);
            sys.clear_color = canary;
            assert_eq!(
                sys.layout.device_scale.to_bits(),
                crate::layout::LayoutTree::sanitize_device_scale(sf).to_bits()
            );

            let vlist = VirtualListIndex::new(50, 19.0);
            sys.build_frame(&0usize, 240.0, 960.0, |ui| {
                ui.element(LayoutSpec::row().fixed(240.0, 960.0), PaintStyle::default(), |ui| {
                    // Left 120px: 50 contiguous Col rows (`height_fixed(19.0)`, `gap = 0.0`)
                    ui.element(LayoutSpec::col().fixed(120.0, 950.0), PaintStyle::default(), |ui| {
                        for i in 0..50usize {
                            ui.keyed(i as u64 + 1, |ui| {
                                ui.element(
                                    LayoutSpec::row().width_grow(1.0).height_fixed(19.0),
                                    PaintStyle::solid(row_fill),
                                    |_| {},
                                );
                            });
                        }
                    });
                    // Right 120px: 50 contiguous VirtualList rows (`height = 19.0`)
                    ui.virtual_list(&vlist, 0.0, 120.0, 950.0, |ui, _idx, rh| {
                        ui.element(
                            LayoutSpec::row().width_grow(1.0).height_fixed(rh),
                            PaintStyle::solid(row_fill),
                            |_| {},
                        );
                    });
                });
            });

            // Collect the two 50-row runs of solid `row_fill` quads:
            let row_quads: Vec<[f32; 4]> = sys
                .primitives
                .iter()
                .filter(|p| p.kind() == PrimKind::Quad && p.packed[0] == row_fill.0)
                .map(|p| p.bounds)
                .collect();
            assert_eq!(row_quads.len(), 100, "Expected 50 Col + 50 VirtualList row quads at sf={sf}");

            for group in [&row_quads[0..50], &row_quads[50..100]] {
                for r in group {
                    assert_eq!(r[0], r[0].round(), "Row x0 must be exact integer at sf={sf}");
                    assert_eq!(r[1], r[1].round(), "Row y0 must be exact integer at sf={sf}");
                    assert_eq!(r[2], r[2].round(), "Row w must be exact integer at sf={sf}");
                    assert_eq!(r[3], r[3].round(), "Row h must be exact integer at sf={sf}");
                }
                for pair in group.windows(2) {
                    let prev_end = pair[0][1] + pair[0][3];
                    let next_start = pair[1][1];
                    assert_eq!(
                        prev_end.to_bits(),
                        next_start.to_bits(),
                        "Zero physical seam violated at sf={sf}: prev_end={prev_end} vs next_start={next_start}"
                    );
                }
            }

            // Render onto bright magenta canary background and verify 0 bleed pixels across the entire covered stack:
            let pw = (240.0 * sf - 1e-4).ceil() as u32;
            let ph = (960.0 * sf - 1e-4).ceil() as u32;
            let mut cpu = crate::gpu::cpu::CpuRenderer::new(pw, ph);
            sys.render_to_backend(&mut cpu, pw, ph, 1).unwrap();
            let covered_x1 = (row_quads[50][0] + row_quads[50][2]) as u32;
            let covered_y1 = (row_quads[49][1] + row_quads[49][3]).min(row_quads[99][1] + row_quads[99][3]) as u32;
            for y in 1..covered_y1.saturating_sub(1) {
                for x in [10u32, (pw / 2) + 10].into_iter().filter(|&cx| cx < covered_x1) {
                    let px = cpu.pixel_at(x, y);
                    assert_ne!(
                        px,
                        [255, 0, 255, 255],
                        "Magenta canary background bled through row seam at ({x}, {y}) under sf={sf}!"
                    );
                }
            }

            // 2. Verify incremental `patch_slot_paint_in_place` and `sync_solved_layout_nodes_gpu` produce
            //    `to_bits()` bit-identical `GpuPrimitive`s to a full scene rebuild at fractional `sf`:
            let mut inc_sys = GuiSystem::new();
            inc_sys.set_scale_factor(sf);
            let mut st_hover = false;
            let build_inc = |ui: &mut Ui<'_, '_, bool>| {
                let hov = *ui.state();
                ui.element(LayoutSpec::col().fixed(180.0, 90.0).pad(11.0), PaintStyle::default(), |ui| {
                    let bg = if hov {
                        Color::rgb8(70, 120, 220)
                    } else {
                        Color::rgb8(45, 55, 75)
                    };
                    let bw = if hov { 2.0 } else { 1.0 };
                    ui.element(
                        LayoutSpec::row().fixed(133.0, 37.0),
                        PaintStyle::solid(bg)
                            .rounded(6.0)
                            .border(bw, Color::rgb8(180, 200, 240)),
                        |_| {},
                    );
                });
            };
            inc_sys.build_frame(&st_hover, 180.0, 90.0, build_inc);
            st_hover = true;
            inc_sys.build_frame(&st_hover, 180.0, 90.0, build_inc);
            assert_eq!(inc_sys.paint_updates_last_frame, 1);

            let mut fresh_sys = GuiSystem::new();
            fresh_sys.set_scale_factor(sf);
            fresh_sys.build_frame(&st_hover, 180.0, 90.0, build_inc);
            assert_eq!(inc_sys.primitives.len(), fresh_sys.primitives.len());
            for (a, b) in inc_sys.primitives.iter().zip(fresh_sys.primitives.iter()) {
                assert_eq!(
                    bytemuck::bytes_of(a),
                    bytemuck::bytes_of(b),
                    "In-place paint patch primitive diverged from fresh rebuild at sf={sf}: {a:?} vs {b:?}"
                );
            }
        }

        // 3. Zero Row-Divider Moiré check on 20 stacked `0.5px` hairline VirtualList rows at `1.25x` and `4/3`:
        for &sf in &[1.25_f32, 160.0 / 120.0] {
            let mut msys = GuiSystem::new();
            msys.set_scale_factor(sf);
            msys.clear_color = Color::rgb8(20, 22, 30);
            let vl = VirtualListIndex::new(20, 24.0);
            msys.build_frame(&0usize, 200.0, 480.0, |ui| {
                ui.virtual_list(&vl, 0.0, 200.0, 480.0, |ui, _i, rh| {
                    ui.element(
                        LayoutSpec::row().width_grow(1.0).height_fixed(rh),
                        PaintStyle::solid(Color::rgb8(28, 32, 44))
                            .border(0.5, Color::rgb8(150, 165, 195)),
                        |_| {},
                    );
                });
            });
            let pw = (200.0 * sf - 1e-4).ceil() as u32;
            let ph = (480.0 * sf - 1e-4).ceil() as u32;
            let mut cpu = crate::gpu::cpu::CpuRenderer::new(pw, ph);
            msys.render_to_backend(&mut cpu, pw, ph, 1).unwrap();

            let row_prims: Vec<[f32; 4]> = msys
                .primitives
                .iter()
                .filter(|p| p.kind() == PrimKind::Quad && p.uv_or_extra[0] > 0.0)
                .map(|p| p.bounds)
                .collect();
            assert_eq!(row_prims.len(), 20);
            let sample_x = pw / 2;
            let first_top_luma = cpu.pixel_at(sample_x, row_prims[0][1] as u32)[0] as i32;
            for (idx, rb) in row_prims.iter().enumerate() {
                let luma = cpu.pixel_at(sample_x, rb[1] as u32)[0] as i32;
                assert!(
                    (luma - first_top_luma).abs() <= 1,
                    "Row {idx} top divider luma ({luma}) diverged from row 0 ({first_top_luma}) at sf={sf} (Moiré!)"
                );
            }
        }
    }

    #[test]
    fn fractional_scale_horizontal_baseline_uniformity_and_scale_policy_integration() {
        // 1. Horizontal baseline uniformity across fractional scales:
        let mut sys = GuiSystem::with_system_fonts();
        for &sf in &[1.125_f32, 1.25, 160.0 / 120.0, 1.375, 1.5, 200.0 / 120.0, 1.75, 2.25] {
            sys.set_scale_factor(sf);
            sys.build_frame(&0usize, 360.0, 120.0, |ui| {
                ui.element(
                    LayoutSpec::col().fixed(360.0, 120.0).pad_xy(13.0, 11.0),
                    PaintStyle::solid(Color::BLACK),
                    |ui| {
                        ui.ui_text("AVATAR Waveform 123.45 Hz — Crisp!", 14.0, Color::WHITE);
                    },
                );
            });

            let text_nid = (0..sys.layout.topo.len())
                .find(|&i| {
                    (sys.layout.topo[i].flags & crate::layout::FLAG_ALIVE) != 0
                        && (sys.layout.topo[i].flags & FLAG_HAS_TEXT) != 0
                })
                .expect("shaped text node");
            let tb = sys.layout.boxes[text_nid];
            let tspec = sys.layout.specs[text_nid];
            let para = &sys.layout.texts[tspec.aux_id as usize];
            assert!(!para.cached_lines.is_empty());

            let mut checked_glyphs = 0usize;
            let mut prim_iter = sys
                .primitives
                .iter()
                .filter(|p| matches!(p.kind(), PrimKind::GlyphGray | PrimKind::GlyphLcd | PrimKind::ImageRgba));

            let n_lines = para.cached_lines.len();
            for l_i in 0..n_lines {
                let line = para.cached_lines[l_i];
                let expected_phys_base = ((tb.y + tspec.pad_t + line.baseline_y) * sf).round();
                let mut pen_x = tb.x + tspec.pad_l;
                for w_i in line.words.start..line.words.end {
                    if w_i > line.words.start {
                        pen_x += para.space_w;
                    }
                    let wr = para.words[w_i as usize];
                    for g_i in wr.glyphs.start..wr.glyphs.end {
                        let sg = para.glyphs[g_i as usize];
                        let phys_gx = (pen_x + sg.x_offset) * sf;
                        let reg = sys.fonts.rasterize_glyph(
                            sg.font_id,
                            sg.glyph_id,
                            para.font_size * sf,
                            phys_gx,
                            if sys.fonts.ui_font == FontId::FALLBACK {
                                AntialiasMode::Grayscale
                            } else {
                                AntialiasMode::SubpixelLcd
                            },
                        );
                        if reg.w > 0 && reg.h > 0 {
                            let gp = prim_iter.next().expect("emitted glyph quad");
                            assert_eq!(gp.bounds[0], gp.bounds[0].round(), "Glyph quad x must be integer");
                            assert_eq!(gp.bounds[1], gp.bounds[1].round(), "Glyph quad y must be integer");
                            if sg.y_offset == 0.0 {
                                let reconstructed_base = gp.bounds[1] + (reg.bearing_y as f32);
                                assert_eq!(
                                    reconstructed_base, expected_phys_base,
                                    "Glyph baseline jitter on line {l_i} at sf={sf}: got {reconstructed_base}, expected {expected_phys_base}"
                                );
                            }
                            checked_glyphs += 1;
                        }
                        pen_x += sg.x_advance;
                    }
                }
            }
            assert!(checked_glyphs >= 15, "Expected >= 15 visible glyphs checked at sf={sf}");
        }

        // 2. `ScalePolicy`, `DisplayMetrics`, `zoom_scale`, `reset_scale_auto`, & AccessKit root scale transform:
        assert_eq!(crate::app::WindowConfig::default().scale_policy, ScalePolicy::Auto);
        let mut ssys = GuiSystem::new();
        ssys.a11y_enabled = true;

        // 27" 4K (`3840x2160`, `597x336 mm`, unscaled X11 `os_scale = 1.0`) -> `1.50x` (`180/120`) Auto!
        let m_4k = DisplayMetrics::from_physical_mm(3840, 2160, 597, 336).with_os_scale(1.0);
        let r_4k = ssys.resolve_display_scale(m_4k);
        assert_eq!(r_4k.wayland_120ths, 180);
        assert_eq!(ssys.scale_factor, 1.5);
        assert_eq!(ssys.resolved_scale, Some(r_4k));

        ssys.build_frame(&0usize, 1280.0, 800.0, |ui| {
            ui.element(LayoutSpec::col().fixed(1280.0, 800.0), PaintStyle::default(), |ui| {
                ui.primary_button("Scale Test", |_, _| {});
            });
        });
        let a11y_150 = ssys.build_accesskit_update().expect("initial 1.5x AccessKit tree");
        let root_node_150 = &a11y_150
            .nodes
            .iter()
            .find(|(id, _)| *id == slot_to_a11y_id(ssys.root_slot.unwrap()))
            .unwrap()
            .1;
        assert_eq!(
            root_node_150.transform(),
            Some(&crate::a11y::A11yAffine::scale(1.5)),
            "Root AccessKit Window node must carry HiDPI Affine::scale(1.5)"
        );

        // Live `zoom_scale(+1)` (`180/120 -> 195/120 = 1.625x`), `zoom_scale(-1)` back to `1.50x`,
        // and `reset_scale_auto()` (`Ctrl+0`) back to `ScalePolicy::Auto`:
        let z_up = ssys.zoom_scale(1);
        assert_eq!(z_up.wayland_120ths, 195);
        assert_eq!(ssys.scale_factor, 1.625);
        assert_eq!(ssys.scale_policy, ScalePolicy::Fixed(1.625));
        let a11y_1625 = ssys.build_accesskit_update().expect("incremental AccessKit root scale update");
        assert_eq!(a11y_1625.nodes.len(), 1);

        let z_down = ssys.zoom_scale(-1);
        assert_eq!(z_down.wayland_120ths, 180);
        assert_eq!(ssys.scale_factor, 1.5);

        let z_auto = ssys.reset_scale_auto();
        assert_eq!(ssys.scale_policy, ScalePolicy::Auto);
        assert_eq!(z_auto.wayland_120ths, 180);
        assert!(z_auto.reason.is_auto());
    }
}
