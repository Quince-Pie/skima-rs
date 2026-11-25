//! Analytical SDF + Premultiplied OKLab + Gaussian `erf` Shadow + Subpixel/Grayscale Atlas CPU Reference Renderer.
//!
//! Architecture & Hot-Path Mechanisms:
//! 1. **Incremental 64-Primitive Hierarchical Bounding-Box Culling (`prim_boxes` + `block_boxes`):**
//!    Updates `[u16; 4]` pixel footprints in $O(|\text{dirty\_prim\_ranges}|)$ and skips 64 off-scissor
//!    primitives per 8-byte L1-cache block check (`~300 ns` across 50,000+ IDE primitives).
//! 2. **Pre-Resolved Clip Chains (`ResolvedClip`) & Rectilinear Safe-Interior Trivial Accept:**
//!    Computes exact ancestor clip outer bounds and corner-free safe spans once per frame ($O(|\text{clips}|)$),
//!    so clipped `VirtualList` / `scroll_area` glyphs and quads skip `eval_clip_chain` when strictly inside.
//! 3. **Scanline Active-Edge Sweep + Backward Opaque Occlusion + Interior Span Fast-Paths:**
//!    Buckets visible primitives by `py0` into an active scanline list, skips occluded background layers when
//!    an opaque solid quad covers `[x0..x1)`, fills solid interior spans without `sqrt`/`hypot` SDF evaluation,
//!    and samples 1:1 pixel-snapped glyphs directly from contiguous atlas rows with zero heap allocations.

use crate::color::Color;
use crate::gpu::{
    CLIP_NONE, DamageRect, GpuClip, GpuPrimitive, PrimKind, RenderBackend, RenderFrame,
};
use crate::text::GlyphAtlas;

const SRGB_TO_LINEAR_LUT: [f32; 256] = {
    let mut lut = [0.0_f32; 256];
    let mut i = 0usize;
    while i < 256 {
        let c = (i as f32) / 255.0;
        lut[i] = if c <= 0.04045 {
            c / 12.92
        } else {
            let base = (c + 0.055) / 1.055;
            const_pow_2_4(base)
        };
        i += 1;
    }
    lut
};

const fn const_pow_2_4(x: f32) -> f32 {
    // ln(x) for x in (0.04..=1.0] via artanh series `ln(x) = 2 * sum_{k=0..63} ((x-1)/(x+1))^(2k+1) / (2k+1)`
    let u = (x as f64 - 1.0) / (x as f64 + 1.0);
    let u2 = u * u;
    let mut term = u;
    let mut ln_x = 0.0_f64;
    let mut k = 0usize;
    while k < 64 {
        ln_x += term / ((2 * k + 1) as f64);
        term *= u2;
        k += 1;
    }
    let y = 2.4_f64 * (2.0 * ln_x);
    // exp(y) via Taylor series around 0 (`y` in `[-7.7, 0.0]`): scale `y / 8` then raise to 8th power!
    let ys = y * 0.125;
    let mut exp_s = 1.0_f64;
    let mut p = 1.0_f64;
    let mut n = 1usize;
    while n <= 24 {
        p *= ys / (n as f64);
        exp_s += p;
        n += 1;
    }
    let e2 = exp_s * exp_s;
    let e4 = e2 * e2;
    (e4 * e4) as f32
}

const LINEAR_BITS_BASE: u32 = 0x3900_0000; // 2^-13 = 0.0001220703125 (< 0.5 / (255 * 12.92))
const LINEAR_BITS_ONE: u32 = 0x3f80_0000;  // 1.0_f32
const LINEAR_LUT_SHIFT: u32 = 14;          // 9 mantissa bits per octave -> 13 * 512 = 6,656 entries (6.5 KiB in L1!)
const LINEAR_LUT_LEN: usize = ((LINEAR_BITS_ONE - LINEAR_BITS_BASE) >> LINEAR_LUT_SHIFT) as usize;

const LINEAR_BITS_TO_SRGB_LUT: [u8; LINEAR_LUT_LEN] = {
    let mut lut = [0u8; LINEAR_LUT_LEN];
    let mut i = 0usize;
    while i < LINEAR_LUT_LEN {
        // Sample exact bin midpoint in IEEE-754 bit-space for minimum quantization error (<= 0.11 LSB everywhere)
        let mid_bits = LINEAR_BITS_BASE + ((i as u32) << LINEAR_LUT_SHIFT) + (1u32 << (LINEAR_LUT_SHIFT - 1));
        let x = f32::from_bits(mid_bits) as f64;
        lut[i] = const_linear_to_srgb_u8(x);
        i += 1;
    }
    lut
};

const fn const_linear_to_srgb_u8(x: f64) -> u8 {
    let srgb = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        let mut m = x;
        let mut exp2 = 0i32;
        while m < 0.5 {
            m *= 2.0;
            exp2 += 1;
        }
        let u = (m - 1.0) / (m + 1.0);
        let u2 = u * u;
        let mut term = u;
        let mut sum = 0.0_f64;
        let mut k = 0usize;
        while k < 22 {
            sum += term / ((2 * k + 1) as f64);
            term *= u2;
            k += 1;
        }
        let ln_x = 2.0 * sum - (exp2 as f64) * core::f64::consts::LN_2;
        let ys = (ln_x / 2.4) * 0.25;
        let mut exp_s = 1.0_f64;
        let mut p = 1.0_f64;
        let mut n = 1usize;
        while n <= 20 {
            p *= ys / (n as f64);
            exp_s += p;
            n += 1;
        }
        let e2 = exp_s * exp_s;
        1.055 * (e2 * e2) - 0.055
    };
    let scaled = srgb * 255.0 + 0.5;
    if scaled <= 0.0 {
        0
    } else if scaled >= 255.0 {
        255
    } else {
        scaled as u8
    }
}

#[inline]
fn srgb_to_linear_chan(c: f32) -> f32 {
    if c <= 0.04045 {
        c.max(0.0) / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Ultra-fast IEEE-754 logarithmic bit-space linear `[0..1]` -> sRGB `u8` converter
/// (~2 CPU cycles, zero `sqrt`/`powf`, 6.5 KiB L1 `const` LUT, max error <= 0.11 LSB).
#[inline(always)]
fn fast_linear_to_srgb_u8(c: f32) -> u8 {
    let bits = c.to_bits();
    if bits >= LINEAR_BITS_ONE {
        return if (bits as i32) < 0 || c.is_nan() { 0 } else { 255 };
    }
    if bits < LINEAR_BITS_BASE {
        return 0;
    }
    LINEAR_BITS_TO_SRGB_LUT[((bits - LINEAR_BITS_BASE) >> LINEAR_LUT_SHIFT) as usize]
}

#[inline]
fn encode_rgba8_from_linear(dst: [f32; 4]) -> [u8; 4] {
    [
        fast_linear_to_srgb_u8(dst[0]),
        fast_linear_to_srgb_u8(dst[1]),
        fast_linear_to_srgb_u8(dst[2]),
        (dst[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    ]
}

#[inline]
fn unpack_linear_lut(c: Color) -> [f32; 4] {
    let [r, g, b, a] = c.channels();
    [
        SRGB_TO_LINEAR_LUT[r as usize],
        SRGB_TO_LINEAR_LUT[g as usize],
        SRGB_TO_LINEAR_LUT[b as usize],
        (a as f32) * (1.0 / 255.0),
    ]
}

#[inline]
fn cbrt_pos(x: f32) -> f32 {
    if x <= 1e-12 {
        0.0
    } else {
        x.cbrt()
    }
}

#[inline]
fn linear_to_oklab(c: [f32; 3]) -> [f32; 3] {
    let l = cbrt_pos(0.412_221_47 * c[0] + 0.536_332_54 * c[1] + 0.051_445_993 * c[2]);
    let m = cbrt_pos(0.211_903_5 * c[0] + 0.680_699_5 * c[1] + 0.107_396_96 * c[2]);
    let s = cbrt_pos(0.088_302_46 * c[0] + 0.281_718_84 * c[1] + 0.629_978_7 * c[2]);
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

#[inline]
fn oklab_to_linear(lab: [f32; 3]) -> [f32; 3] {
    let l_ = lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2];
    let m_ = lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2];
    let s_ = lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2];
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    [
        (4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s).clamp(0.0, 1.0),
        (-1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s).clamp(0.0, 1.0),
        (-0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s).clamp(0.0, 1.0),
    ]
}

const BLOCK_SHIFT: usize = 6;
const BLOCK_SIZE: usize = 1 << BLOCK_SHIFT;
const SNAP_NONE: i32 = i32::MIN;

/// Pre-resolved clip chain summary for $O(1)$ outer culling and rectilinear inner-safe trivial acceptance.
#[derive(Clone, Copy)]
struct ResolvedClip {
    /// Axis-aligned outer intersection `[x0, y0, x1, y1]` across all ancestors in the clip chain.
    outer: [f32; 4],
    /// Corner-free horizontal band `[nx0, nx1]` (`max_i(bx_i + mr_i) + 1.0 .. min_i(bx_i + bw_i - mr_i) - 1.0`).
    no_corner_x: [f32; 2],
    /// Corner-free vertical band `[ny0, ny1]` (`max_i(by_i + mr_i) + 1.0 .. min_i(by_i + bh_i - mr_i) - 1.0`).
    no_corner_y: [f32; 2],
    has_corners: bool,
    is_empty: bool,
}

/// Auxiliary precomputed parameters allocated ONLY for `PrimKind::Quad` and `PrimKind::LinearGradient`,
/// keeping hot per-glyph `PreparedPrim` records at a dense 56 bytes instead of 256 bytes!
#[derive(Clone, Copy)]
struct QuadExtra {
    fill_premul: [f32; 4],
    ring_premul: [f32; 4],
    interior_premul: [f32; 4],
    max_inner_r: f32,
    solid_box: [f32; 4],
    solid_no_corner_x: [f32; 2],
    solid_no_corner_y: [f32; 2],
    plab0: [f32; 4],
    plab1: [f32; 4],
    grad_inv_span: f32,
    opaque_srgb: [u8; 4],
    has_solid_interior: bool,
    is_opaque_quad: bool,
}

/// Compact 56-byte (`< 1` cache line) hot scanline record shared across all primitive kinds.
#[derive(Clone, Copy)]
struct PreparedPrim {
    prim_idx: u32,
    clip_id: u32,
    px0: u32,
    py0: u32,
    px1: u32,
    py1: u32,
    c0_lin: [f32; 4],
    opacity: f32,
    /// Kind-specific scalar: `stem_k` for glyphs, `inv_sigma_sqrt2` for shadows.
    aux_f32: f32,
    /// For `Quad`/`LinearGradient`: index into `quad_extras`; for snapped glyphs/images: `tex_x0 - dst_x0` (`SNAP_NONE` if unsnapped).
    aux_or_snap_dx: i32,
    /// For snapped glyphs/images: `tex_y0 - dst_y0`.
    snap_dy: i32,
    kind: PrimKind,
    is_opaque_quad: bool,
}

const _: () = assert!(core::mem::size_of::<PreparedPrim>() <= 64);

pub struct CpuRenderer {
    pub width: u32,
    pub height: u32,
    pub framebuffer: Vec<u8>,
    cached_prims: Vec<GpuPrimitive>,
    prim_boxes: Vec<[u16; 4]>,
    block_boxes: Vec<[u16; 4]>,
    resolved_clips: Vec<ResolvedClip>,
    prepared: Vec<PreparedPrim>,
    quad_extras: Vec<QuadExtra>,
    start_heads: Vec<u32>,
    next_in_start: Vec<u32>,
    active: Vec<u32>,
    scanline_lin: Vec<[f32; 4]>,
    initialized: bool,
    scene_synced: bool,
}

impl Default for CpuRenderer {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

impl CpuRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            framebuffer: vec![0u8; (width as usize) * (height as usize) * 4],
            cached_prims: Vec::new(),
            prim_boxes: Vec::new(),
            block_boxes: Vec::new(),
            resolved_clips: Vec::with_capacity(64),
            prepared: Vec::with_capacity(256),
            quad_extras: Vec::with_capacity(64),
            start_heads: Vec::new(),
            next_in_start: Vec::with_capacity(256),
            active: Vec::with_capacity(128),
            scanline_lin: vec![[0.0; 4]; width as usize],
            initialized: false,
            scene_synced: false,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.framebuffer
                .resize((width as usize) * (height as usize) * 4, 0);
            self.scanline_lin.resize(width as usize, [0.0; 4]);
            self.initialized = false;
        }
    }

    #[inline]
    pub fn pixel_at(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * self.width + x) as usize) * 4;
        [
            self.framebuffer[idx],
            self.framebuffer[idx + 1],
            self.framebuffer[idx + 2],
            self.framebuffer[idx + 3],
        ]
    }

    fn resolve_clips(&mut self, clips: &[GpuClip]) {
        self.resolved_clips.clear();
        let n_clips = clips.len();
        if n_clips == 0 {
            return;
        }
        self.resolved_clips.reserve(n_clips);
        for start in 0..n_clips {
            let mut ox0 = f32::NEG_INFINITY;
            let mut oy0 = f32::NEG_INFINITY;
            let mut ox1 = f32::INFINITY;
            let mut oy1 = f32::INFINITY;
            let mut ncx0 = f32::NEG_INFINITY;
            let mut ncy0 = f32::NEG_INFINITY;
            let mut ncx1 = f32::INFINITY;
            let mut ncy1 = f32::INFINITY;
            let mut has_corners = false;
            let mut is_empty = false;

            let mut cid = start as u32;
            let mut steps = 0usize;
            while cid != CLIP_NONE && (cid as usize) < n_clips && steps < n_clips {
                let c = &clips[cid as usize];
                let bw = c.bounds[2];
                let bh = c.bounds[3];
                if bw <= 0.0 || bh <= 0.0 || !bw.is_finite() || !bh.is_finite() {
                    is_empty = true;
                    break;
                }
                let bx0 = c.bounds[0];
                let by0 = c.bounds[1];
                let bx1 = bx0 + bw;
                let by1 = by0 + bh;
                ox0 = ox0.max(bx0);
                oy0 = oy0.max(by0);
                ox1 = ox1.min(bx1);
                oy1 = oy1.min(by1);
                if ox1 <= ox0 || oy1 <= oy0 {
                    is_empty = true;
                    break;
                }
                let max_c = (bw * 0.5).min(bh * 0.5).max(0.0);
                let mr = c.radii[0]
                    .max(c.radii[1])
                    .max(c.radii[2])
                    .max(c.radii[3])
                    .clamp(0.0, max_c);
                if mr > 0.0 {
                    has_corners = true;
                    ncx0 = ncx0.max(bx0 + mr + 1.0);
                    ncx1 = ncx1.min(bx1 - mr - 1.0);
                    ncy0 = ncy0.max(by0 + mr + 1.0);
                    ncy1 = ncy1.min(by1 - mr - 1.0);
                } else {
                    ncx0 = ncx0.max(bx0);
                    ncx1 = ncx1.min(bx1);
                    ncy0 = ncy0.max(by0);
                    ncy1 = ncy1.min(by1);
                }
                cid = c.meta[0];
                steps += 1;
            }
            self.resolved_clips.push(ResolvedClip {
                outer: [ox0, oy0, ox1, oy1],
                no_corner_x: [ncx0, ncx1],
                no_corner_y: [ncy0, ncy1],
                has_corners,
                is_empty,
            });
        }
    }

    #[inline]
    fn compute_prim_box16(prim: &GpuPrimitive) -> [u16; 4] {
        if prim.is_invisible() {
            return [u16::MAX, u16::MAX, 0, 0];
        }
        let fp = prim.screen_footprint();
        let x1_f = fp[0] + fp[2];
        let y1_f = fp[1] + fp[3];
        if fp[2] <= 0.0
            || fp[3] <= 0.0
            || x1_f < 0.5
            || y1_f < 0.5
            || !fp[0].is_finite()
            || !fp[1].is_finite()
            || !x1_f.is_finite()
            || !y1_f.is_finite()
        {
            return [u16::MAX, u16::MAX, 0, 0];
        }
        let px0 = ((fp[0] - 0.5).ceil().max(0.0) as u32).min(u16::MAX as u32) as u16;
        let py0 = ((fp[1] - 0.5).ceil().max(0.0) as u32).min(u16::MAX as u32) as u16;
        let px1 = (((x1_f - 0.5).floor().max(0.0) as u32).saturating_add(1)).min(u16::MAX as u32)
            as u16;
        let py1 = (((y1_f - 0.5).floor().max(0.0) as u32).saturating_add(1)).min(u16::MAX as u32)
            as u16;
        if px1 > px0 && py1 > py0 {
            [px0, py0, px1, py1]
        } else {
            [u16::MAX, u16::MAX, 0, 0]
        }
    }

    #[inline]
    fn recompute_block(&mut self, block_idx: usize) {
        let start = block_idx << BLOCK_SHIFT;
        let end = (start + BLOCK_SIZE).min(self.prim_boxes.len());
        let mut bx0 = u16::MAX;
        let mut by0 = u16::MAX;
        let mut bx1 = 0u16;
        let mut by1 = 0u16;
        for b in &self.prim_boxes[start..end] {
            if b[2] > b[0] {
                bx0 = bx0.min(b[0]);
                by0 = by0.min(b[1]);
                bx1 = bx1.max(b[2]);
                by1 = by1.max(b[3]);
            }
        }
        self.block_boxes[block_idx] = [bx0, by0, bx1, by1];
    }

    fn sync_scene_index(&mut self, frame: &RenderFrame<'_>, force_full: bool) {
        let n = frame.primitives.len();
        let n_blocks = n.div_ceil(BLOCK_SIZE);
        if force_full
            || !self.scene_synced
            || frame.dirty_prim_ranges.is_empty()
            || self.cached_prims.len() != n
        {
            self.cached_prims.clear();
            self.cached_prims.extend_from_slice(frame.primitives);
            self.prim_boxes.resize(n, [u16::MAX, u16::MAX, 0, 0]);
            self.block_boxes.resize(n_blocks, [u16::MAX, u16::MAX, 0, 0]);
            for (i, p) in frame.primitives.iter().enumerate() {
                self.prim_boxes[i] = Self::compute_prim_box16(p);
            }
            for b in 0..n_blocks {
                self.recompute_block(b);
            }
            self.scene_synced = true;
        } else {
            for span in frame.dirty_prim_ranges {
                let s = (span.start as usize).min(n);
                let e = (span.end as usize).min(n);
                if e <= s {
                    continue;
                }
                self.cached_prims[s..e].copy_from_slice(&frame.primitives[s..e]);
                for i in s..e {
                    self.prim_boxes[i] = Self::compute_prim_box16(&frame.primitives[i]);
                }
                let b_start = s >> BLOCK_SHIFT;
                let b_end = (e - 1) >> BLOCK_SHIFT;
                for b in b_start..=b_end {
                    self.recompute_block(b);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn build_prepared_prim(
        prim_idx: u32,
        prim: &GpuPrimitive,
        resolved_clips: &[ResolvedClip],
        atlas: &GlyphAtlas,
        quad_extras: &mut Vec<QuadExtra>,
        sc_x0: u32,
        sc_y0: u32,
        sc_x1: u32,
        sc_y1: u32,
    ) -> Option<PreparedPrim> {
        if prim.is_invisible() {
            return None;
        }
        let fp = prim.screen_footprint();
        let mut fx0 = fp[0];
        let mut fy0 = fp[1];
        let mut fx1 = fp[0] + fp[2];
        let mut fy1 = fp[1] + fp[3];

        let clip_id = prim.packed[2] & CLIP_NONE;
        if clip_id != CLIP_NONE && (clip_id as usize) < resolved_clips.len() {
            let rc = &resolved_clips[clip_id as usize];
            if rc.is_empty {
                return None;
            }
            fx0 = fx0.max(rc.outer[0]);
            fy0 = fy0.max(rc.outer[1]);
            fx1 = fx1.min(rc.outer[2]);
            fy1 = fy1.min(rc.outer[3]);
        }

        if fx1 <= fx0
            || fy1 <= fy0
            || fx1 < 0.5
            || fy1 < 0.5
            || !fx0.is_finite()
            || !fy0.is_finite()
            || !fx1.is_finite()
            || !fy1.is_finite()
        {
            return None;
        }
        let px0 = ((fx0 - 0.5).ceil().max(0.0) as u32).max(sc_x0);
        let py0 = ((fy0 - 0.5).ceil().max(0.0) as u32).max(sc_y0);
        let px1 = (((fx1 - 0.5).floor().max(0.0) as u32).saturating_add(1)).min(sc_x1);
        let py1 = (((fy1 - 0.5).floor().max(0.0) as u32).saturating_add(1)).min(sc_y1);
        if px1 <= px0 || py1 <= py0 {
            return None;
        }

        let kind = prim.kind();
        let c0_color = Color(prim.packed[0]);
        let c0_lin = unpack_linear_lut(c0_color);
        let op_u16 = prim.packed[3] & 0xFFFF;
        let opacity = (op_u16 as f32) * (1.0 / 65535.0);

        match kind {
            PrimKind::Quad | PrimKind::LinearGradient => {
                let c1_lin = unpack_linear_lut(Color(prim.packed[1]));
                let bx = prim.bounds[0];
                let by = prim.bounds[1];
                let bw = prim.bounds[2];
                let bh = prim.bounds[3];
                let hx = bw * 0.5;
                let hy = bh * 0.5;
                let border_w = prim.uv_or_extra[0].max(0.0);
                let max_r = hx.min(hy).max(0.0);
                let max_outer_r = prim.radii[0]
                    .max(prim.radii[1])
                    .max(prim.radii[2])
                    .max(prim.radii[3])
                    .clamp(0.0, max_r);
                let max_inner_r = if border_w > 0.0 {
                    (max_outer_r - border_w).max(0.0)
                } else {
                    max_outer_r
                };

                let fill_premul = [
                    c0_lin[0] * c0_lin[3],
                    c0_lin[1] * c0_lin[3],
                    c0_lin[2] * c0_lin[3],
                    c0_lin[3],
                ];
                let ring_premul = if border_w > 0.0 {
                    let b_a = c1_lin[3];
                    let one_minus_ba = 1.0 - b_a;
                    [
                        c1_lin[0] * b_a + fill_premul[0] * one_minus_ba,
                        c1_lin[1] * b_a + fill_premul[1] * one_minus_ba,
                        c1_lin[2] * b_a + fill_premul[2] * one_minus_ba,
                        b_a + fill_premul[3] * one_minus_ba,
                    ]
                } else {
                    [0.0; 4]
                };

                let mut interior_premul = [0.0_f32; 4];
                let mut opaque_srgb = c0_color.channels();
                let mut solid_box = [0.0_f32; 4];
                let mut solid_no_corner_x = [0.0_f32; 2];
                let mut solid_no_corner_y = [0.0_f32; 2];
                let mut has_solid_interior = false;
                let mut is_opaque_quad = false;
                let mut plab0 = [0.0_f32; 4];
                let mut plab1 = [0.0_f32; 4];
                let mut grad_inv_span = 1.0_f32;

                if kind == PrimKind::Quad {
                    interior_premul = [
                        fill_premul[0] * opacity,
                        fill_premul[1] * opacity,
                        fill_premul[2] * opacity,
                        fill_premul[3] * opacity,
                    ];
                    let in_x0 = bx + border_w;
                    let in_y0 = by + border_w;
                    let in_x1 = bx + bw - border_w;
                    let in_y1 = by + bh - border_w;
                    let pad = if border_w > 0.0 { 1.0 } else { 0.0 };
                    if in_x1 - in_x0 >= 1.0 + 2.0 * pad && in_y1 - in_y0 >= 1.0 + 2.0 * pad {
                        solid_box = [in_x0 + pad, in_y0 + pad, in_x1 - pad, in_y1 - pad];
                        solid_no_corner_x = [in_x0 + max_inner_r + 1.0, in_x1 - max_inner_r - 1.0];
                        solid_no_corner_y = [in_y0 + max_inner_r + 1.0, in_y1 - max_inner_r - 1.0];
                        has_solid_interior = true;
                        if opaque_srgb[3] == 255 && op_u16 == 65535 {
                            is_opaque_quad = true;
                            opaque_srgb[3] = 255;
                        }
                    }
                } else {
                    let l0 = linear_to_oklab([c0_lin[0], c0_lin[1], c0_lin[2]]);
                    let l1 = linear_to_oklab([c1_lin[0], c1_lin[1], c1_lin[2]]);
                    let a0 = c0_lin[3];
                    let a1 = c1_lin[3];
                    plab0 = [l0[0] * a0, l0[1] * a0, l0[2] * a0, a0];
                    plab1 = [l1[0] * a1, l1[1] * a1, l1[2] * a1, a1];
                    let dir = [prim.uv_or_extra[2], prim.uv_or_extra[3]];
                    grad_inv_span = 1.0 / (dir[0].abs() * bw + dir[1].abs() * bh).max(1e-4);
                }

                let extra_idx = quad_extras.len() as i32;
                quad_extras.push(QuadExtra {
                    fill_premul,
                    ring_premul,
                    interior_premul,
                    max_inner_r,
                    solid_box,
                    solid_no_corner_x,
                    solid_no_corner_y,
                    plab0,
                    plab1,
                    grad_inv_span,
                    opaque_srgb,
                    has_solid_interior,
                    is_opaque_quad,
                });

                Some(PreparedPrim {
                    prim_idx,
                    clip_id,
                    px0,
                    py0,
                    px1,
                    py1,
                    c0_lin,
                    opacity,
                    aux_f32: 0.0,
                    aux_or_snap_dx: extra_idx,
                    snap_dy: 0,
                    kind,
                    is_opaque_quad,
                })
            }
            PrimKind::Shadow => {
                let sigma = prim.uv_or_extra[1].max(0.5);
                let inv_sigma_sqrt2 = 1.0 / (sigma * core::f32::consts::SQRT_2);
                Some(PreparedPrim {
                    prim_idx,
                    clip_id,
                    px0,
                    py0,
                    px1,
                    py1,
                    c0_lin,
                    opacity,
                    aux_f32: inv_sigma_sqrt2,
                    aux_or_snap_dx: SNAP_NONE,
                    snap_dy: 0,
                    kind,
                    is_opaque_quad: false,
                })
            }
            PrimKind::GlyphGray | PrimKind::GlyphLcd | PrimKind::ImageRgba => {
                let lum = 0.2126 * c0_lin[0] + 0.7152 * c0_lin[1] + 0.0722 * c0_lin[2];
                let stem_k = if kind == PrimKind::GlyphLcd {
                    0.44 - 0.56 * lum
                } else {
                    0.46 - 0.58 * lum
                };

                let mut snap_dx = SNAP_NONE;
                let mut snap_dy = 0i32;
                let aw = atlas.width as usize;
                let ah = atlas.height as usize;
                let bx = prim.bounds[0];
                let by = prim.bounds[1];
                let bw = prim.bounds[2];
                let bh = prim.bounds[3];
                if aw > 0
                    && ah > 0
                    && bx.abs() <= 1.0e6
                    && by.abs() <= 1.0e6
                    && bw >= 1.0
                    && bh >= 1.0
                    && bw <= 65536.0
                    && bh <= 65536.0
                    && (bx - bx.round()).abs() < 1e-3
                    && (by - by.round()).abs() < 1e-3
                    && (bw - bw.round()).abs() < 1e-3
                    && (bh - bh.round()).abs() < 1e-3
                {
                    let aw_f = aw as f32;
                    let ah_f = ah as f32;
                    let tx0_f = prim.uv_or_extra[0] * aw_f;
                    let ty0_f = prim.uv_or_extra[1] * ah_f;
                    let tw_f = (prim.uv_or_extra[2] - prim.uv_or_extra[0]) * aw_f;
                    let th_f = (prim.uv_or_extra[3] - prim.uv_or_extra[1]) * ah_f;
                    let tx0_r = tx0_f.round();
                    let ty0_r = ty0_f.round();
                    if tx0_r >= 0.0
                        && ty0_r >= 0.0
                        && (tx0_f - tx0_r).abs() < 1e-3
                        && (ty0_f - ty0_r).abs() < 1e-3
                        && (tw_f - bw).abs() < 1e-3
                        && (th_f - bh).abs() < 1e-3
                    {
                        let ix0 = tx0_r as usize;
                        let iy0 = ty0_r as usize;
                        let iw = bw.round() as usize;
                        let ih = bh.round() as usize;
                        let plane_ok = if kind == PrimKind::GlyphGray {
                            atlas.r8_data.len() >= aw * ah
                        } else {
                            atlas.rgba8_data.len() >= aw * ah * 4
                        };
                        let dst_x = bx.round() as i32;
                        let dst_y = by.round() as i32;
                        if plane_ok
                            && ix0 + iw <= aw
                            && iy0 + ih <= ah
                            && (px0 as i32) >= dst_x
                            && (py0 as i32) >= dst_y
                            && (px1 as i32) <= dst_x.saturating_add(iw as i32)
                            && (py1 as i32) <= dst_y.saturating_add(ih as i32)
                        {
                            snap_dx = (ix0 as i32) - dst_x;
                            snap_dy = (iy0 as i32) - dst_y;
                        }
                    }
                }

                Some(PreparedPrim {
                    prim_idx,
                    clip_id,
                    px0,
                    py0,
                    px1,
                    py1,
                    c0_lin,
                    opacity,
                    aux_f32: stem_k,
                    aux_or_snap_dx: snap_dx,
                    snap_dy,
                    kind,
                    is_opaque_quad: false,
                })
            }
        }
    }
}

/// Compute the contiguous pixel span `[sx0, sx1)` inside `[x0, x1)` on row `py` where `clip_alpha == 1.0` is guaranteed.
#[inline]
fn clip_full_span_on_row(
    resolved_clips: &[ResolvedClip],
    clip_id: u32,
    py: u32,
    fy: f32,
    x0: u32,
    x1: u32,
) -> (u32, u32) {
    if clip_id == CLIP_NONE || (clip_id as usize) >= resolved_clips.len() {
        return (x0, x1);
    }
    let rc = &resolved_clips[clip_id as usize];
    if rc.is_empty || (py as f32) < rc.outer[1] || ((py + 1) as f32) > rc.outer[3] {
        return (0, 0);
    }
    let mut cx0 = (rc.outer[0].ceil().max(0.0) as u32).max(x0);
    let mut cx1 = if rc.outer[2] >= 1.0 {
        (rc.outer[2].floor() as u32).min(x1)
    } else {
        0
    };
    if cx1 <= cx0 {
        return (0, 0);
    }
    if !rc.has_corners || (fy >= rc.no_corner_y[0] && fy <= rc.no_corner_y[1]) {
        return (cx0, cx1);
    }
    cx0 = cx0.max((rc.no_corner_x[0] - 0.5).ceil().max(0.0) as u32);
    let ncx1 = if rc.no_corner_x[1] >= 0.5 {
        ((rc.no_corner_x[1] - 0.5).floor() as u32).saturating_add(1)
    } else {
        0
    };
    cx1 = cx1.min(ncx1);
    (cx0, cx1.max(cx0))
}

/// Compute the contiguous pixel span `[qx0, qx1)` inside `[px0, px1)` on row `py` where both shape & clip have `cov == 1.0`.
#[inline]
fn quad_fast_span_on_row(
    prep: &PreparedPrim,
    qe: &QuadExtra,
    resolved_clips: &[ResolvedClip],
    py: u32,
    fy: f32,
    x0: u32,
    x1: u32,
) -> (u32, u32) {
    if !qe.has_solid_interior
        || (py as f32) < qe.solid_box[1]
        || ((py + 1) as f32) > qe.solid_box[3]
    {
        return (0, 0);
    }
    let (csx0, csx1) = clip_full_span_on_row(resolved_clips, prep.clip_id, py, fy, x0, x1);
    if csx1 <= csx0 {
        return (0, 0);
    }
    let mut qx0 = (qe.solid_box[0].ceil().max(0.0) as u32)
        .max(prep.px0)
        .max(csx0);
    let mut qx1 = if qe.solid_box[2] >= 1.0 {
        (qe.solid_box[2].floor() as u32).min(prep.px1).min(csx1)
    } else {
        0
    };
    if qx1 <= qx0 {
        return (0, 0);
    }
    if qe.max_inner_r > 0.0 && (fy < qe.solid_no_corner_y[0] || fy > qe.solid_no_corner_y[1]) {
        qx0 = qx0.max((qe.solid_no_corner_x[0] - 0.5).ceil().max(0.0) as u32);
        let ncx1 = if qe.solid_no_corner_x[1] >= 0.5 {
            ((qe.solid_no_corner_x[1] - 0.5).floor() as u32).saturating_add(1)
        } else {
            0
        };
        qx1 = qx1.min(ncx1);
    }
    (qx0, qx1.max(qx0))
}

impl RenderBackend for CpuRenderer {
    type Error = core::convert::Infallible;

    fn render(&mut self, frame: &RenderFrame<'_>, atlas: &mut GlyphAtlas) -> Result<(), Self::Error> {
        atlas.dirty_rects.clear();
        self.resize(frame.width, frame.height);
        if frame.width == 0 || frame.height == 0 {
            return Ok(());
        }

        let was_initialized = self.initialized;
        if was_initialized && frame.damage.is_none() {
            // Still sync incremental primitive bounds if caller passed dirty ranges before returning:
            if !frame.dirty_prim_ranges.is_empty() && frame.width <= u16::MAX as u32 && frame.height <= u16::MAX as u32 {
                self.sync_scene_index(frame, false);
            }
            return Ok(());
        }

        let scissor = if !was_initialized {
            self.initialized = true;
            DamageRect {
                x: 0,
                y: 0,
                w: frame.width,
                h: frame.height,
            }
        } else {
            frame.damage.unwrap()
        };

        let x0 = scissor.x.min(frame.width);
        let y0 = scissor.y.min(frame.height);
        let x1 = scissor.x.saturating_add(scissor.w).min(frame.width);
        let y1 = scissor.y.saturating_add(scissor.h).min(frame.height);
        if x1 <= x0 || y1 <= y0 {
            return Ok(());
        }

        // 1. Pre-resolve all clip chains in O(clips.len()) once per frame:
        self.resolve_clips(frame.clips);

        // 2. Gather & pre-decode ONLY primitives intersecting both `scissor` and their `ResolvedClip::outer`:
        self.prepared.clear();
        self.quad_extras.clear();
        let prim_slice: &[GpuPrimitive] = if frame.width <= u16::MAX as u32 && frame.height <= u16::MAX as u32 {
            self.sync_scene_index(frame, !was_initialized);
            let sx0_16 = x0 as u16;
            let sy0_16 = y0 as u16;
            let sx1_16 = x1 as u16;
            let sy1_16 = y1 as u16;
            let total_prims = self.prim_boxes.len();
            for (b_idx, bb) in self.block_boxes.iter().enumerate() {
                if bb[2] <= sx0_16 || bb[0] >= sx1_16 || bb[3] <= sy0_16 || bb[1] >= sy1_16 {
                    continue;
                }
                let start = b_idx << BLOCK_SHIFT;
                let end = (start + BLOCK_SIZE).min(total_prims);
                for i in start..end {
                    let pb = self.prim_boxes[i];
                    if pb[2] <= sx0_16 || pb[0] >= sx1_16 || pb[3] <= sy0_16 || pb[1] >= sy1_16 {
                        continue;
                    }
                    if let Some(prep) = Self::build_prepared_prim(
                        i as u32,
                        &self.cached_prims[i],
                        &self.resolved_clips,
                        atlas,
                        &mut self.quad_extras,
                        x0,
                        y0,
                        x1,
                        y1,
                    ) {
                        self.prepared.push(prep);
                    }
                }
            }
            &self.cached_prims
        } else {
            self.scene_synced = false;
            for (i, prim) in frame.primitives.iter().enumerate() {
                if let Some(prep) = Self::build_prepared_prim(
                    i as u32,
                    prim,
                    &self.resolved_clips,
                    atlas,
                    &mut self.quad_extras,
                    x0,
                    y0,
                    x1,
                    y1,
                ) {
                    self.prepared.push(prep);
                }
            }
            frame.primitives
        };

        // 3. Bucket prepared primitives by starting scanline `py0 - y0` in ascending painters' Z-order:
        let sc_h = (y1 - y0) as usize;
        let span_w = (x1 - x0) as usize;
        self.start_heads.clear();
        self.start_heads.resize(sc_h, u32::MAX);
        self.next_in_start.resize(self.prepared.len(), u32::MAX);
        for idx in (0..self.prepared.len()).rev() {
            let row = (self.prepared[idx].py0 - y0) as usize;
            self.next_in_start[idx] = self.start_heads[row];
            self.start_heads[row] = idx as u32;
        }
        self.active.clear();

        let raw_clear = unpack_linear_lut(frame.clear_color);
        let clear_lin = [
            raw_clear[0] * raw_clear[3],
            raw_clear[1] * raw_clear[3],
            raw_clear[2] * raw_clear[3],
            raw_clear[3],
        ];
        let clear_srgb = encode_rgba8_from_linear(clear_lin);

        // 4. Scanline active-primitive sweep with backward opaque occlusion & direct sRGB span fills:
        for py in y0..y1 {
            let fy = (py as f32) + 0.5;
            self.active.retain(|&idx| self.prepared[idx as usize].py1 > py);
            let mut head = self.start_heads[(py - y0) as usize];
            if head != u32::MAX {
                let was_non_empty = !self.active.is_empty();
                while head != u32::MAX {
                    self.active.push(head);
                    head = self.next_in_start[head as usize];
                }
                if was_non_empty {
                    self.active.sort_unstable();
                }
            }

            let dst_byte_start = (((py * self.width) + x0) as usize) * 4;
            let dst_row_bytes = &mut self.framebuffer[dst_byte_start..dst_byte_start + span_w * 4];

            if self.active.is_empty() {
                fill_rgba8_row(dst_row_bytes, clear_srgb);
                continue;
            }

            // Backward search for the topmost opaque solid quad whose interior covers all of `[x0..x1)`:
            let mut opaque_cover_pos: Option<usize> = None;
            for pos in (0..self.active.len()).rev() {
                let prep = &self.prepared[self.active[pos] as usize];
                if prep.is_opaque_quad {
                    let qe = &self.quad_extras[prep.aux_or_snap_dx as usize];
                    let (qx0, qx1) =
                        quad_fast_span_on_row(prep, qe, &self.resolved_clips, py, fy, x0, x1);
                    if qx0 <= x0 && qx1 >= x1 {
                        opaque_cover_pos = Some(pos);
                        break;
                    }
                }
            }

            // Fast-path: topmost primitive on this scanline is an opaque solid quad covering `[x0..x1)` ->
            // write sRGB bytes directly into `self.framebuffer` (skipping `scanline_lin` completely!).
            if let Some(pos) = opaque_cover_pos
                && pos + 1 == self.active.len()
            {
                let prep = &self.prepared[self.active[pos] as usize];
                let srgb = self.quad_extras[prep.aux_or_snap_dx as usize].opaque_srgb;
                fill_rgba8_row(dst_row_bytes, srgb);
                continue;
            }

            let scan_buf = &mut self.scanline_lin[..span_w];
            let start_pos = if let Some(pos) = opaque_cover_pos {
                let c = self.prepared[self.active[pos] as usize].c0_lin;
                scan_buf.fill([c[0], c[1], c[2], 1.0]);
                pos + 1
            } else {
                scan_buf.fill(clear_lin);
                0
            };

            for &prep_idx in &self.active[start_pos..] {
                let prep = &self.prepared[prep_idx as usize];
                let prim = &prim_slice[prep.prim_idx as usize];
                composite_prim_on_scanline(
                    prep,
                    prim,
                    &self.quad_extras,
                    frame.clips,
                    &self.resolved_clips,
                    atlas,
                    py,
                    fy,
                    x0,
                    x1,
                    scan_buf,
                );
            }

            // Convert `scan_buf` linear premultiplied RGBA -> sRGB `u8` via 6.5 KiB IEEE-754 bit-space LUT:
            for (lin, out_px) in scan_buf.iter().zip(dst_row_bytes.as_chunks_mut::<4>().0) {
                *out_px = encode_rgba8_from_linear(*lin);
            }
        }

        Ok(())
    }
}

#[inline]
fn fill_rgba8_row(dst_row_bytes: &mut [u8], rgba: [u8; 4]) {
    for px in dst_row_bytes.as_chunks_mut::<4>().0 {
        *px = rgba;
    }
}

#[inline]
fn blend_premul(dst: &mut [f32; 4], src_rgb: [f32; 3], mask: [f32; 4]) {
    dst[0] = src_rgb[0] + dst[0] * (1.0 - mask[0]);
    dst[1] = src_rgb[1] + dst[1] * (1.0 - mask[1]);
    dst[2] = src_rgb[2] + dst[2] * (1.0 - mask[2]);
    dst[3] = mask[3] + dst[3] * (1.0 - mask[3]);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn composite_prim_on_scanline(
    prep: &PreparedPrim,
    prim: &GpuPrimitive,
    quad_extras: &[QuadExtra],
    clips: &[GpuClip],
    resolved_clips: &[ResolvedClip],
    atlas: &GlyphAtlas,
    py: u32,
    fy: f32,
    x0: u32,
    x1: u32,
    scan_buf: &mut [[f32; 4]],
) {
    let clip_id = prep.clip_id;
    let px0 = prep.px0;
    let px1 = prep.px1;

    match prep.kind {
        PrimKind::Quad => {
            let qe = &quad_extras[prep.aux_or_snap_dx as usize];
            let (qx0, qx1) = quad_fast_span_on_row(prep, qe, resolved_clips, py, fy, x0, x1);
            if qx1 > qx0 {
                // Left edge / border / corner before interior span:
                for px in px0..qx0 {
                    let fx = (px as f32) + 0.5;
                    if let Some((src_rgb, mask)) =
                        eval_quad_or_grad_pixel(prep, prim, qe, clips, resolved_clips, [fx, fy], false)
                    {
                        blend_premul(&mut scan_buf[(px - x0) as usize], src_rgb, mask);
                    }
                }
                // Interior constant-coverage span (zero SDF, zero clip walk!):
                let s_off = (qx0 - x0) as usize;
                let e_off = (qx1 - x0) as usize;
                if qe.is_opaque_quad {
                    let c = prep.c0_lin;
                    scan_buf[s_off..e_off].fill([c[0], c[1], c[2], 1.0]);
                } else {
                    let ip = qe.interior_premul;
                    if ip[3] > 0.0005 {
                        let inv_a = 1.0 - ip[3];
                        for dst in &mut scan_buf[s_off..e_off] {
                            dst[0] = ip[0] + dst[0] * inv_a;
                            dst[1] = ip[1] + dst[1] * inv_a;
                            dst[2] = ip[2] + dst[2] * inv_a;
                            dst[3] = ip[3] + dst[3] * inv_a;
                        }
                    }
                    // When `ip[3] <= 0.0005` (e.g. hollow border quad with `bg = TRANSPARENT`), interior is a 0-op skip!
                }
                // Right edge / border / corner after interior span:
                for px in qx1..px1 {
                    let fx = (px as f32) + 0.5;
                    if let Some((src_rgb, mask)) =
                        eval_quad_or_grad_pixel(prep, prim, qe, clips, resolved_clips, [fx, fy], false)
                    {
                        blend_premul(&mut scan_buf[(px - x0) as usize], src_rgb, mask);
                    }
                }
            } else {
                let (csx0, csx1) = clip_full_span_on_row(resolved_clips, clip_id, py, fy, x0, x1);
                for px in px0..px1 {
                    let fx = (px as f32) + 0.5;
                    let clip_one = px >= csx0 && px < csx1;
                    if let Some((src_rgb, mask)) =
                        eval_quad_or_grad_pixel(prep, prim, qe, clips, resolved_clips, [fx, fy], clip_one)
                    {
                        blend_premul(&mut scan_buf[(px - x0) as usize], src_rgb, mask);
                    }
                }
            }
        }
        PrimKind::LinearGradient => {
            let qe = &quad_extras[prep.aux_or_snap_dx as usize];
            let (csx0, csx1) = clip_full_span_on_row(resolved_clips, clip_id, py, fy, x0, x1);
            for px in px0..px1 {
                let fx = (px as f32) + 0.5;
                let clip_one = px >= csx0 && px < csx1;
                if let Some((src_rgb, mask)) =
                    eval_quad_or_grad_pixel(prep, prim, qe, clips, resolved_clips, [fx, fy], clip_one)
                {
                    blend_premul(&mut scan_buf[(px - x0) as usize], src_rgb, mask);
                }
            }
        }
        PrimKind::Shadow => {
            let hx = prim.bounds[2] * 0.5;
            let hy = prim.bounds[3] * 0.5;
            let cx = prim.bounds[0] + hx;
            let cy = prim.bounds[1] + hy;
            let sp_y = (fy - cy) - prim.uv_or_extra[3];
            let inv_s = prep.aux_f32;
            let iy = erf_scalar((hy - sp_y) * inv_s) + erf_scalar((hy + sp_y) * inv_s);
            if iy <= 0.0 {
                return;
            }
            let max_r = hx.min(hy).max(0.0);
            let has_r = prim.radii[0]
                .max(prim.radii[1])
                .max(prim.radii[2])
                .max(prim.radii[3])
                .min(max_r)
                > 0.0;
            let (csx0, csx1) = clip_full_span_on_row(resolved_clips, clip_id, py, fy, x0, x1);
            let base_scale = prep.c0_lin[3] * prep.opacity;
            let off_x = prim.uv_or_extra[2];

            for px in px0..px1 {
                let fx = (px as f32) + 0.5;
                let clip_alpha = if px >= csx0 && px < csx1 {
                    1.0
                } else {
                    let ca = eval_clip_resolved([fx, fy], clips, resolved_clips, clip_id);
                    if ca <= 0.0005 {
                        continue;
                    }
                    ca
                };
                let sp_x = (fx - cx) - off_x;
                let ix = erf_scalar((hx - sp_x) * inv_s) + erf_scalar((hx + sp_x) * inv_s);
                let rect_cov = (0.25 * ix * iy).clamp(0.0, 1.0);
                let cov = if has_r {
                    let cr = select_corner_radius([sp_x, sp_y], prim.radii).clamp(0.0, max_r);
                    if cr > 0.0 {
                        let d = sd_rounded_box([sp_x, sp_y], [hx, hy], prim.radii);
                        let sdf_cov = (0.5 * (1.0 - erf_scalar(d * inv_s))).clamp(0.0, 1.0);
                        rect_cov.min(sdf_cov)
                    } else {
                        rect_cov
                    }
                } else {
                    rect_cov
                };
                let final_a = base_scale * cov * clip_alpha;
                if final_a > 0.0005 {
                    blend_premul(
                        &mut scan_buf[(px - x0) as usize],
                        [
                            prep.c0_lin[0] * final_a,
                            prep.c0_lin[1] * final_a,
                            prep.c0_lin[2] * final_a,
                        ],
                        [final_a, final_a, final_a, final_a],
                    );
                }
            }
        }
        PrimKind::GlyphLcd => {
            let (csx0, csx1) = clip_full_span_on_row(resolved_clips, clip_id, py, fy, x0, x1);
            let lin = prep.c0_lin;
            let k = prep.aux_f32;
            let base_scale = lin[3] * prep.opacity;

            if prep.aux_or_snap_dx != SNAP_NONE {
                let aw = atlas.width as usize;
                let row_tex_y = ((py as i32) + prep.snap_dy) as usize;
                let snap_dx = prep.aux_or_snap_dx;
                let row_base = row_tex_y * aw * 4;

                for px in px0..px1 {
                    let clip_alpha = if px >= csx0 && px < csx1 {
                        1.0
                    } else {
                        let ca = eval_clip_resolved(
                            [(px as f32) + 0.5, fy],
                            clips,
                            resolved_clips,
                            clip_id,
                        );
                        if ca <= 0.0005 {
                            continue;
                        }
                        ca
                    };
                    let t_off = row_base + (((px as i32) + snap_dx) as usize) * 4;
                    let r_u8 = atlas.rgba8_data[t_off];
                    let g_u8 = atlas.rgba8_data[t_off + 1];
                    let b_u8 = atlas.rgba8_data[t_off + 2];
                    if (r_u8 | g_u8 | b_u8) == 0 {
                        continue;
                    }
                    let mr = (r_u8 as f32) * (1.0 / 255.0);
                    let mg = (g_u8 as f32) * (1.0 / 255.0);
                    let mb = (b_u8 as f32) * (1.0 / 255.0);
                    let scale = base_scale * clip_alpha;
                    let er = (mr + k * mr * (1.0 - mr)).clamp(0.0, 1.0) * scale;
                    let eg = (mg + k * mg * (1.0 - mg)).clamp(0.0, 1.0) * scale;
                    let eb = (mb + k * mb * (1.0 - mb)).clamp(0.0, 1.0) * scale;
                    let avg_a = 0.299 * er + 0.587 * eg + 0.114 * eb;
                    if er.max(eg).max(eb).max(avg_a) > 0.0005 {
                        blend_premul(
                            &mut scan_buf[(px - x0) as usize],
                            [lin[0] * er, lin[1] * eg, lin[2] * eb],
                            [er, eg, eb, avg_a],
                        );
                    }
                }
            } else {
                let ah = atlas.height as usize;
                let aw = atlas.width as usize;
                let v = atlas_axis_uv(
                    fy,
                    prim.bounds[1],
                    prim.bounds[3],
                    prim.uv_or_extra[1],
                    prim.uv_or_extra[3],
                    ah,
                );

                for px in px0..px1 {
                    let fx = (px as f32) + 0.5;
                    let clip_alpha = if px >= csx0 && px < csx1 {
                        1.0
                    } else {
                        let ca = eval_clip_resolved([fx, fy], clips, resolved_clips, clip_id);
                        if ca <= 0.0005 {
                            continue;
                        }
                        ca
                    };
                    let u = atlas_axis_uv(
                        fx,
                        prim.bounds[0],
                        prim.bounds[2],
                        prim.uv_or_extra[0],
                        prim.uv_or_extra[2],
                        aw,
                    );
                    let (rgba, _) = sample_atlas_rgba8(atlas, u, v);
                    let mr = rgba[0].clamp(0.0, 1.0);
                    let mg = rgba[1].clamp(0.0, 1.0);
                    let mb = rgba[2].clamp(0.0, 1.0);
                    let scale = base_scale * clip_alpha;
                    let er = (mr + k * mr * (1.0 - mr)).clamp(0.0, 1.0) * scale;
                    let eg = (mg + k * mg * (1.0 - mg)).clamp(0.0, 1.0) * scale;
                    let eb = (mb + k * mb * (1.0 - mb)).clamp(0.0, 1.0) * scale;
                    let avg_a = 0.299 * er + 0.587 * eg + 0.114 * eb;
                    if er.max(eg).max(eb).max(avg_a) > 0.0005 {
                        blend_premul(
                            &mut scan_buf[(px - x0) as usize],
                            [lin[0] * er, lin[1] * eg, lin[2] * eb],
                            [er, eg, eb, avg_a],
                        );
                    }
                }
            }
        }
        PrimKind::GlyphGray => {
            let (csx0, csx1) = clip_full_span_on_row(resolved_clips, clip_id, py, fy, x0, x1);
            let lin = prep.c0_lin;
            let k = prep.aux_f32;
            let base_scale = lin[3] * prep.opacity;

            if prep.aux_or_snap_dx != SNAP_NONE {
                let aw = atlas.width as usize;
                let row_tex_y = ((py as i32) + prep.snap_dy) as usize;
                let snap_dx = prep.aux_or_snap_dx;
                let row_base = row_tex_y * aw;

                for px in px0..px1 {
                    let clip_alpha = if px >= csx0 && px < csx1 {
                        1.0
                    } else {
                        let ca = eval_clip_resolved(
                            [(px as f32) + 0.5, fy],
                            clips,
                            resolved_clips,
                            clip_id,
                        );
                        if ca <= 0.0005 {
                            continue;
                        }
                        ca
                    };
                    let raw_u8 = atlas.r8_data[row_base + (((px as i32) + snap_dx) as usize)];
                    if raw_u8 == 0 {
                        continue;
                    }
                    let raw_cov = (raw_u8 as f32) * (1.0 / 255.0);
                    let cov = (raw_cov + k * raw_cov * (1.0 - raw_cov)).clamp(0.0, 1.0);
                    let final_a = base_scale * cov * clip_alpha;
                    if final_a > 0.0005 {
                        blend_premul(
                            &mut scan_buf[(px - x0) as usize],
                            [lin[0] * final_a, lin[1] * final_a, lin[2] * final_a],
                            [final_a, final_a, final_a, final_a],
                        );
                    }
                }
            } else {
                let ah = atlas.height as usize;
                let aw = atlas.width as usize;
                let v = atlas_axis_uv(
                    fy,
                    prim.bounds[1],
                    prim.bounds[3],
                    prim.uv_or_extra[1],
                    prim.uv_or_extra[3],
                    ah,
                );

                for px in px0..px1 {
                    let fx = (px as f32) + 0.5;
                    let clip_alpha = if px >= csx0 && px < csx1 {
                        1.0
                    } else {
                        let ca = eval_clip_resolved([fx, fy], clips, resolved_clips, clip_id);
                        if ca <= 0.0005 {
                            continue;
                        }
                        ca
                    };
                    let u = atlas_axis_uv(
                        fx,
                        prim.bounds[0],
                        prim.bounds[2],
                        prim.uv_or_extra[0],
                        prim.uv_or_extra[2],
                        aw,
                    );
                    let raw_cov = sample_atlas_r8(atlas, u, v).clamp(0.0, 1.0);
                    let cov = (raw_cov + k * raw_cov * (1.0 - raw_cov)).clamp(0.0, 1.0);
                    let final_a = base_scale * cov * clip_alpha;
                    if final_a > 0.0005 {
                        blend_premul(
                            &mut scan_buf[(px - x0) as usize],
                            [lin[0] * final_a, lin[1] * final_a, lin[2] * final_a],
                            [final_a, final_a, final_a, final_a],
                        );
                    }
                }
            }
        }
        PrimKind::ImageRgba => {
            let (csx0, csx1) = clip_full_span_on_row(resolved_clips, clip_id, py, fy, x0, x1);
            let tint = prep.c0_lin;
            let base_scale = tint[3] * prep.opacity;

            if prep.aux_or_snap_dx != SNAP_NONE {
                let aw = atlas.width as usize;
                let row_tex_y = ((py as i32) + prep.snap_dy) as usize;
                let snap_dx = prep.aux_or_snap_dx;
                let row_base = row_tex_y * aw * 4;

                for px in px0..px1 {
                    let clip_alpha = if px >= csx0 && px < csx1 {
                        1.0
                    } else {
                        let ca = eval_clip_resolved(
                            [(px as f32) + 0.5, fy],
                            clips,
                            resolved_clips,
                            clip_id,
                        );
                        if ca <= 0.0005 {
                            continue;
                        }
                        ca
                    };
                    let t_off = row_base + (((px as i32) + snap_dx) as usize) * 4;
                    let raw_a = atlas.rgba8_data[t_off + 3];
                    if raw_a == 0 {
                        continue;
                    }
                    let final_a = (raw_a as f32) * (1.0 / 255.0) * base_scale * clip_alpha;
                    if final_a > 0.0005 {
                        let tr = SRGB_TO_LINEAR_LUT[atlas.rgba8_data[t_off] as usize];
                        let tg = SRGB_TO_LINEAR_LUT[atlas.rgba8_data[t_off + 1] as usize];
                        let tb = SRGB_TO_LINEAR_LUT[atlas.rgba8_data[t_off + 2] as usize];
                        blend_premul(
                            &mut scan_buf[(px - x0) as usize],
                            [tr * tint[0] * final_a, tg * tint[1] * final_a, tb * tint[2] * final_a],
                            [final_a, final_a, final_a, final_a],
                        );
                    }
                }
            } else {
                let ah = atlas.height as usize;
                let aw = atlas.width as usize;
                let v = atlas_axis_uv(
                    fy,
                    prim.bounds[1],
                    prim.bounds[3],
                    prim.uv_or_extra[1],
                    prim.uv_or_extra[3],
                    ah,
                );

                for px in px0..px1 {
                    let fx = (px as f32) + 0.5;
                    let clip_alpha = if px >= csx0 && px < csx1 {
                        1.0
                    } else {
                        let ca = eval_clip_resolved([fx, fy], clips, resolved_clips, clip_id);
                        if ca <= 0.0005 {
                            continue;
                        }
                        ca
                    };
                    let u = atlas_axis_uv(
                        fx,
                        prim.bounds[0],
                        prim.bounds[2],
                        prim.uv_or_extra[0],
                        prim.uv_or_extra[2],
                        aw,
                    );
                    let (rgba, exact_u8) = sample_atlas_rgba8(atlas, u, v);
                    let final_a = rgba[3] * base_scale * clip_alpha;
                    if final_a > 0.0005 {
                        let (tr, tg, tb) = if let Some(raw) = exact_u8 {
                            (
                                SRGB_TO_LINEAR_LUT[raw[0] as usize],
                                SRGB_TO_LINEAR_LUT[raw[1] as usize],
                                SRGB_TO_LINEAR_LUT[raw[2] as usize],
                            )
                        } else {
                            (
                                srgb_to_linear_chan(rgba[0]),
                                srgb_to_linear_chan(rgba[1]),
                                srgb_to_linear_chan(rgba[2]),
                            )
                        };
                        blend_premul(
                            &mut scan_buf[(px - x0) as usize],
                            [tr * tint[0] * final_a, tg * tint[1] * final_a, tb * tint[2] * final_a],
                            [final_a, final_a, final_a, final_a],
                        );
                    }
                }
            }
        }
    }
}

#[inline]
fn eval_quad_or_grad_pixel(
    prep: &PreparedPrim,
    prim: &GpuPrimitive,
    qe: &QuadExtra,
    clips: &[GpuClip],
    resolved_clips: &[ResolvedClip],
    px: [f32; 2],
    clip_known_one: bool,
) -> Option<([f32; 3], [f32; 4])> {
    let clip_alpha = if clip_known_one {
        1.0
    } else {
        let ca = eval_clip_resolved(px, clips, resolved_clips, prep.clip_id);
        if ca <= 0.0005 {
            return None;
        }
        ca
    };

    let bx = prim.bounds[0];
    let by = prim.bounds[1];
    let bw = prim.bounds[2];
    let bh = prim.bounds[3];
    let dl = [px[0] - bx, px[1] - by];
    let dr = [(bx + bw) - px[0], (by + bh) - px[1]];
    let hx = bw * 0.5;
    let hy = bh * 0.5;
    let rel = [px[0] - (bx + hx), px[1] - (by + hy)];
    let raw_cr = select_corner_radius(rel, prim.radii);
    let outer_cov = rounded_box_cov_edges(dl, dr, [bw, bh], raw_cr);
    if outer_cov <= 0.0005 {
        return None;
    }

    let fill_premul = if prep.kind == PrimKind::LinearGradient {
        let dir = [prim.uv_or_extra[2], prim.uv_or_extra[3]];
        let t = ((rel[0] * dir[0] + rel[1] * dir[1]) * qe.grad_inv_span + 0.5).clamp(0.0, 1.0);
        let p0 = qe.plab0;
        let p1 = qe.plab1;
        let fill_a = p0[3] + (p1[3] - p0[3]) * t;
        if fill_a > 1e-6 {
            let inv_a = 1.0 / fill_a;
            let lab = [
                (p0[0] + (p1[0] - p0[0]) * t) * inv_a,
                (p0[1] + (p1[1] - p0[1]) * t) * inv_a,
                (p0[2] + (p1[2] - p0[2]) * t) * inv_a,
            ];
            let rgb = oklab_to_linear(lab);
            [rgb[0] * fill_a, rgb[1] * fill_a, rgb[2] * fill_a, fill_a]
        } else {
            [0.0; 4]
        }
    } else {
        qe.fill_premul
    };

    let border_w = prim.uv_or_extra[0];
    let shape_premul = if border_w > 0.0 {
        let max_r = hx.min(hy).max(0.0);
        let eff_outer_cr = raw_cr.clamp(0.0, max_r);
        let inner_cr = (eff_outer_cr - border_w).max(0.0);
        let inner_cov = rounded_box_cov_edges(
            [dl[0] - border_w, dl[1] - border_w],
            [dr[0] - border_w, dr[1] - border_w],
            [bw - 2.0 * border_w, bh - 2.0 * border_w],
            inner_cr,
        )
        .min(outer_cov);
        let border_cov = (outer_cov - inner_cov).max(0.0);
        let ring_premul = if prep.kind == PrimKind::LinearGradient {
            let c1 = unpack_linear_lut(Color(prim.packed[1]));
            let b_a = c1[3];
            let omb = 1.0 - b_a;
            [
                c1[0] * b_a + fill_premul[0] * omb,
                c1[1] * b_a + fill_premul[1] * omb,
                c1[2] * b_a + fill_premul[2] * omb,
                b_a + fill_premul[3] * omb,
            ]
        } else {
            qe.ring_premul
        };
        [
            fill_premul[0] * inner_cov + ring_premul[0] * border_cov,
            fill_premul[1] * inner_cov + ring_premul[1] * border_cov,
            fill_premul[2] * inner_cov + ring_premul[2] * border_cov,
            fill_premul[3] * inner_cov + ring_premul[3] * border_cov,
        ]
    } else {
        [
            fill_premul[0] * outer_cov,
            fill_premul[1] * outer_cov,
            fill_premul[2] * outer_cov,
            fill_premul[3] * outer_cov,
        ]
    };

    let scale = prep.opacity * clip_alpha;
    let final_a = shape_premul[3] * scale;
    if final_a <= 0.0005 {
        None
    } else {
        Some((
            [
                shape_premul[0] * scale,
                shape_premul[1] * scale,
                shape_premul[2] * scale,
            ],
            [final_a, final_a, final_a, final_a],
        ))
    }
}

#[inline]
fn select_corner_radius(p: [f32; 2], radii: [f32; 4]) -> f32 {
    if p[0] > 0.0 {
        if p[1] > 0.0 { radii[2] } else { radii[1] }
    } else if p[1] > 0.0 {
        radii[3]
    } else {
        radii[0]
    }
}

#[inline]
fn sd_rounded_box(p: [f32; 2], half: [f32; 2], radii: [f32; 4]) -> f32 {
    let cr = select_corner_radius(p, radii).clamp(0.0, half[0].min(half[1]).max(0.0));
    let qx = p[0].abs() - half[0] + cr;
    let qy = p[1].abs() - half[1] + cr;
    if qx > 0.0 && qy > 0.0 {
        (qx * qx + qy * qy).sqrt() - cr
    } else {
        qx.max(qy) - cr
    }
}

#[inline]
fn rounded_box_cov_edges(dl: [f32; 2], dr: [f32; 2], size: [f32; 2], raw_cr: f32) -> f32 {
    if size[0] <= 0.0 || size[1] <= 0.0 {
        return 0.0;
    }
    let edx = dl[0].min(dr[0]);
    let edy = dl[1].min(dr[1]);
    let max_cx = size[0].clamp(0.0, 1.0);
    let max_cy = size[1].clamp(0.0, 1.0);
    let rx = (edx + 0.5).clamp(0.0, max_cx);
    let ry = (edy + 0.5).clamp(0.0, max_cy);
    let box_area = rx * ry;
    if box_area <= 0.0 {
        return 0.0;
    }
    let cr = raw_cr.clamp(0.0, 0.5 * size[0].min(size[1]));
    if cr <= 0.0 {
        return box_area;
    }
    let qx = cr - edx;
    let qy = cr - edy;
    if qx > 0.0 && qy > 0.0 {
        let d = (qx * qx + qy * qy).sqrt() - cr;
        let sdf_cov = (0.5 - d).clamp(0.0, 1.0);
        let w = cr.clamp(0.0, 1.0);
        return box_area + (box_area.min(sdf_cov) - box_area) * w;
    }
    box_area
}

#[inline]
fn erf_scalar(x: f32) -> f32 {
    let s = x.signum();
    let a = x.abs();
    let p = 1.0 + (0.278_393 + (0.230_389 + 0.078_108 * (a * a)) * a) * a;
    let p4 = p * p * p * p;
    s - s / p4
}

#[inline]
fn eval_clip_resolved(
    px: [f32; 2],
    clips: &[GpuClip],
    resolved_clips: &[ResolvedClip],
    start_clip: u32,
) -> f32 {
    if start_clip == CLIP_NONE || (start_clip as usize) >= resolved_clips.len() {
        return 1.0;
    }
    let rc = &resolved_clips[start_clip as usize];
    if rc.is_empty {
        return 0.0;
    }
    let ix0 = (px[0] - 0.5).max(rc.outer[0]);
    let iy0 = (px[1] - 0.5).max(rc.outer[1]);
    let ix1 = (px[0] + 0.5).min(rc.outer[2]);
    let iy1 = (px[1] + 0.5).min(rc.outer[3]);
    if ix1 <= ix0 || iy1 <= iy0 {
        return 0.0;
    }
    let box_cov = (ix1 - ix0) * (iy1 - iy0);
    if !rc.has_corners
        || (px[1] >= rc.no_corner_y[0] && px[1] <= rc.no_corner_y[1])
        || (px[0] >= rc.no_corner_x[0] && px[0] <= rc.no_corner_x[1])
    {
        return box_cov;
    }
    eval_clip_chain(px, clips, start_clip)
}

#[inline]
fn eval_clip_chain(px: [f32; 2], clips: &[GpuClip], start_clip: u32) -> f32 {
    if start_clip == CLIP_NONE {
        return 1.0;
    }
    let n_clips = clips.len();
    let mut ix0 = px[0] - 0.5;
    let mut iy0 = px[1] - 0.5;
    let mut ix1 = px[0] + 0.5;
    let mut iy1 = px[1] + 0.5;
    let mut min_arc_cov = 1.0_f32;
    let mut cid = start_clip;
    let mut steps = 0usize;

    while cid != CLIP_NONE && (cid as usize) < n_clips && steps < n_clips {
        let clip = &clips[cid as usize];
        if clip.bounds[2] <= 0.0 || clip.bounds[3] <= 0.0 {
            return 0.0;
        }
        ix0 = ix0.max(clip.bounds[0]);
        iy0 = iy0.max(clip.bounds[1]);
        ix1 = ix1.min(clip.bounds[0] + clip.bounds[2]);
        iy1 = iy1.min(clip.bounds[1] + clip.bounds[3]);
        if ix1 <= ix0 || iy1 <= iy0 {
            return 0.0;
        }
        let cur_box_cov = (ix1 - ix0) * (iy1 - iy0);
        let hx = clip.bounds[2] * 0.5;
        let hy = clip.bounds[3] * 0.5;
        let rel = [px[0] - (clip.bounds[0] + hx), px[1] - (clip.bounds[1] + hy)];
        let cr = select_corner_radius(rel, clip.radii).clamp(0.0, hx.min(hy));
        if cr > 0.0 {
            let edx = (px[0] - clip.bounds[0]).min((clip.bounds[0] + clip.bounds[2]) - px[0]);
            let edy = (px[1] - clip.bounds[1]).min((clip.bounds[1] + clip.bounds[3]) - px[1]);
            let qx = cr - edx;
            let qy = cr - edy;
            if qx > 0.0 && qy > 0.0 {
                let d = (qx * qx + qy * qy).sqrt() - cr;
                let sdf_cov = (0.5 - d).clamp(0.0, 1.0);
                let w = cr.clamp(0.0, 1.0);
                let corner_cov = cur_box_cov + (cur_box_cov.min(sdf_cov) - cur_box_cov) * w;
                min_arc_cov = min_arc_cov.min(corner_cov);
                if min_arc_cov <= 0.0005 {
                    return 0.0;
                }
            }
        }
        cid = clip.meta[0];
        steps += 1;
    }
    let box_cov = (ix1 - ix0) * (iy1 - iy0);
    box_cov.min(min_arc_cov)
}

/// Per-axis exact `1:1` texel-center UV helper (prevents horizontal ClearType subpixel bleed even when
/// vertically smooth-scrolled at fractional Y, matching `sample_atlas_*_prim` in `shaders/uber.wgsl`).
#[inline]
fn atlas_axis_uv(coord: f32, b0: f32, bw: f32, u0: f32, u1: f32, dim: usize) -> f32 {
    if dim > 0 && bw >= 1.0 {
        let dim_f = dim as f32;
        let b0_r = b0.round();
        let bw_r = bw.round();
        let t_min = u0 * dim_f;
        let t_min_r = t_min.round();
        let t_sz = (u1 - u0) * dim_f;
        if t_min_r >= 0.0
            && (t_min_r + bw_r) <= dim_f
            && (b0 - b0_r).abs() < 1e-3
            && (bw - bw_r).abs() < 1e-3
            && (t_min - t_min_r).abs() < 1e-3
            && (t_sz - bw_r).abs() < 1e-3
        {
            let rel = (coord.floor() - b0_r).clamp(0.0, (bw_r - 1.0).max(0.0));
            return ((t_min_r + rel).clamp(0.0, (dim_f - 1.0).max(0.0)) + 0.5) / dim_f;
        }
    }
    let t = ((coord - b0) / bw.max(1e-5)).clamp(0.0, 1.0);
    u0 + (u1 - u0) * t
}

/// Vulkan `VK_FILTER_LINEAR` + `CLAMP_TO_EDGE` sampler for `R8_UNORM` atlas plane.
#[inline]
fn sample_atlas_r8(atlas: &GlyphAtlas, u: f32, v: f32) -> f32 {
    let w = atlas.width as usize;
    let h = atlas.height as usize;
    if w == 0 || h == 0 || atlas.r8_data.len() < w * h {
        return 0.0;
    }
    let fx = u * (w as f32) - 0.5;
    let fy = v * (h as f32) - 0.5;
    let ix0 = fx.floor();
    let iy0 = fy.floor();
    let mut tx = fx - ix0;
    let mut ty = fy - iy0;
    let max_x = (w - 1) as isize;
    let max_y = (h - 1) as isize;
    let mut x0 = (ix0 as isize).clamp(0, max_x) as usize;
    let mut y0 = (iy0 as isize).clamp(0, max_y) as usize;
    if tx < 1e-4 {
        tx = 0.0;
    } else if tx > 1.0 - 1e-4 {
        x0 = ((ix0 as isize) + 1).clamp(0, max_x) as usize;
        tx = 0.0;
    }
    if ty < 1e-4 {
        ty = 0.0;
    } else if ty > 1.0 - 1e-4 {
        y0 = ((iy0 as isize) + 1).clamp(0, max_y) as usize;
        ty = 0.0;
    }

    if tx == 0.0 && ty == 0.0 {
        return atlas.r8_data[y0 * w + x0] as f32 * (1.0 / 255.0);
    }

    let x1 = ((ix0 as isize) + 1).clamp(0, max_x) as usize;
    let y1 = ((iy0 as isize) + 1).clamp(0, max_y) as usize;
    let c00 = atlas.r8_data[y0 * w + x0] as f32;
    let c10 = atlas.r8_data[y0 * w + x1] as f32;
    let c01 = atlas.r8_data[y1 * w + x0] as f32;
    let c11 = atlas.r8_data[y1 * w + x1] as f32;
    let top = c00 + (c10 - c00) * tx;
    let bot = c01 + (c11 - c01) * tx;
    (top + (bot - top) * ty) * (1.0 / 255.0)
}

/// Vulkan `VK_FILTER_LINEAR` + `CLAMP_TO_EDGE` sampler for `R8G8B8A8_UNORM` atlas plane.
#[inline]
fn sample_atlas_rgba8(atlas: &GlyphAtlas, u: f32, v: f32) -> ([f32; 4], Option<[u8; 4]>) {
    let w = atlas.width as usize;
    let h = atlas.height as usize;
    if w == 0 || h == 0 || atlas.rgba8_data.len() < w * h * 4 {
        return ([0.0; 4], None);
    }
    let fx = u * (w as f32) - 0.5;
    let fy = v * (h as f32) - 0.5;
    let ix0 = fx.floor();
    let iy0 = fy.floor();
    let mut tx = fx - ix0;
    let mut ty = fy - iy0;
    let max_x = (w - 1) as isize;
    let max_y = (h - 1) as isize;
    let mut x0 = (ix0 as isize).clamp(0, max_x) as usize;
    let mut y0 = (iy0 as isize).clamp(0, max_y) as usize;
    if tx < 1e-4 {
        tx = 0.0;
    } else if tx > 1.0 - 1e-4 {
        x0 = ((ix0 as isize) + 1).clamp(0, max_x) as usize;
        tx = 0.0;
    }
    if ty < 1e-4 {
        ty = 0.0;
    } else if ty > 1.0 - 1e-4 {
        y0 = ((iy0 as isize) + 1).clamp(0, max_y) as usize;
        ty = 0.0;
    }

    if tx == 0.0 && ty == 0.0 {
        let base = (y0 * w + x0) * 4;
        let raw = [
            atlas.rgba8_data[base],
            atlas.rgba8_data[base + 1],
            atlas.rgba8_data[base + 2],
            atlas.rgba8_data[base + 3],
        ];
        return (
            [
                raw[0] as f32 * (1.0 / 255.0),
                raw[1] as f32 * (1.0 / 255.0),
                raw[2] as f32 * (1.0 / 255.0),
                raw[3] as f32 * (1.0 / 255.0),
            ],
            Some(raw),
        );
    }

    let x1 = ((ix0 as isize) + 1).clamp(0, max_x) as usize;
    let y1 = ((iy0 as isize) + 1).clamp(0, max_y) as usize;
    let b00 = (y0 * w + x0) * 4;
    let b10 = (y0 * w + x1) * 4;
    let b01 = (y1 * w + x0) * 4;
    let b11 = (y1 * w + x1) * 4;
    let out = core::array::from_fn(|c| {
        let c00 = atlas.rgba8_data[b00 + c] as f32;
        let c10 = atlas.rgba8_data[b10 + c] as f32;
        let c01 = atlas.rgba8_data[b01 + c] as f32;
        let c11 = atlas.rgba8_data[b11 + c] as f32;
        let top = c00 + (c10 - c00) * tx;
        let bot = c01 + (c11 - c01) * tx;
        (top + (bot - top) * ty) * (1.0 / 255.0)
    });
    (out, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_lut_and_fast_encoder_exact_roundtrip() {
        for i in 0..=255u8 {
            let lin = SRGB_TO_LINEAR_LUT[i as usize];
            let exact = crate::color::srgb_to_linear((i as f32) / 255.0);
            assert!(
                (lin - exact).abs() < 1e-6,
                "LUT mismatch at {i}: lut={lin} vs exact={exact}"
            );
            let encoded = fast_linear_to_srgb_u8(lin);
            assert_eq!(
                encoded, i,
                "sRGB u8 round-trip failed at {i}: got {encoded}"
            );
        }
    }

    #[test]
    fn cpu_zero_viewport_partial_damage_and_cyclic_clip_safety() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let mut cpu = CpuRenderer::new(32, 32);

        // 0x0 viewport must succeed cleanly and produce 0-byte framebuffer:
        let empty_frame = RenderFrame {
            width: 0,
            height: 0,
            scale_factor: 1.0,
            clear_color: Color::rgb8(10, 20, 30),
            primitives: &[],
            dirty_prim_ranges: &[],
            clips: &[],
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: u32::MAX,
                h: u32::MAX,
            }),
        };
        cpu.render(&empty_frame, &mut atlas).unwrap();
        assert_eq!(cpu.framebuffer.len(), 0);

        // Cyclic GpuClip graph (`0 -> 1 -> 0`) must terminate cleanly without hanging:
        let clips = [
            GpuClip {
                bounds: [4.0, 4.0, 24.0, 24.0],
                radii: [6.0; 4],
                meta: [1, 0, 0, 0],
            },
            GpuClip {
                bounds: [4.0, 4.0, 24.0, 24.0],
                radii: [6.0; 4],
                meta: [0, 0, 0, 0],
            },
        ];
        let prims = [GpuPrimitive::quad(
            [0.0, 0.0, 32.0, 32.0],
            [0.0; 4],
            Color::rgb8(220, 60, 80),
            0.0,
            Color::TRANSPARENT,
            0,
            1.0,
        )];
        // Partial damage on uninitialized resized target must still initialize full backdrop:
        let frame = RenderFrame {
            width: 32,
            height: 32,
            scale_factor: 1.0,
            clear_color: Color::rgb8(12, 16, 24),
            primitives: &prims,
            dirty_prim_ranges: &[],
            clips: &clips,
            damage: Some(DamageRect {
                x: 8,
                y: 8,
                w: 8,
                h: 8,
            }),
        };
        cpu.render(&frame, &mut atlas).unwrap();
        assert_eq!(cpu.pixel_at(1, 1), [12, 16, 24, 255]);
        assert_eq!(cpu.pixel_at(16, 16), [220, 60, 80, 255]);
    }

    #[test]
    fn subpixel_hairline_energy_conservation_and_large_coord_sterbenz_invariance() {
        // 1. Exact analytical coverage on integer-aligned boxes at `y0 = 4.0` vs `y0 = 4096.0` on a `2048px`
        //    tall box with fractional `border_w = 0.625` (`0.5 * 1.25`) and `0.6666667` (`0.5 * 4/3`):
        for &bw in &[0.5_f32, 0.5625, 0.625, 0.5 * (160.0 / 120.0), 0.75, 0.875] {
            let dl_near = [50.5 - 10.0, 4.5 - 4.0];
            let dr_near = [210.0 - 50.5, 42.0 - 4.5];
            let outer_near = rounded_box_cov_edges(dl_near, dr_near, [200.0, 38.0], 8.0);
            let inner_near = rounded_box_cov_edges(
                [dl_near[0] - bw, dl_near[1] - bw],
                [dr_near[0] - bw, dr_near[1] - bw],
                [200.0 - 2.0 * bw, 38.0 - 2.0 * bw],
                (8.0 - bw).max(0.0),
            );
            let bcov_near = outer_near - inner_near;
            assert!(
                (bcov_near - bw).abs() < 1e-6,
                "Expected exact {bw} boundary coverage, got {bcov_near}"
            );

            // Far coordinate `y0 = 8192.0, h = 4096.0` with rounded `cr = 12.0` on straight top edge:
            let dl_far = [50.5 - 10.0, 8192.5 - 8192.0];
            let dr_far = [210.0 - 50.5, (8192.0 + 4096.0) - 8192.5];
            let outer_far = rounded_box_cov_edges(dl_far, dr_far, [200.0, 4096.0], 12.0);
            let inner_far = rounded_box_cov_edges(
                [dl_far[0] - bw, dl_far[1] - bw],
                [dr_far[0] - bw, dr_far[1] - bw],
                [200.0 - 2.0 * bw, 4096.0 - 2.0 * bw],
                (12.0 - bw).max(0.0),
            );
            let bcov_far = outer_far - inner_far;
            assert_eq!(
                bcov_near.to_bits(),
                bcov_far.to_bits(),
                "Sterbenz edge-distance border coverage must be bit-identical at y=4 vs y=8192 for bw={bw}"
            );

            // Second pixel inward (`fy = y0 + 1.5`) must have 0.0000000 inner leak when `bw <= 1.0`:
            let dl_in = [50.5 - 10.0, 5.5 - 4.0];
            let dr_in = [210.0 - 50.5, 42.0 - 5.5];
            let outer_in = rounded_box_cov_edges(dl_in, dr_in, [200.0, 38.0], 8.0);
            let inner_in = rounded_box_cov_edges(
                [dl_in[0] - bw, dl_in[1] - bw],
                [dr_in[0] - bw, dr_in[1] - bw],
                [200.0 - 2.0 * bw, 38.0 - 2.0 * bw],
                (8.0 - bw).max(0.0),
            );
            assert_eq!(outer_in, 1.0);
            assert_eq!(inner_in, 1.0);
        }

        // 2. Sub-pixel phase sweep (`y0 = 10.0 + phase / 16.0` for `phase = 0..16`):
        //    Sum of linear border energy across straddled boundary pixels must conserve `bw = 0.625`!
        let mut atlas = GlyphAtlas::new(64, 64);
        let mut cpu = CpuRenderer::new(32, 32);
        let bw = 0.625_f32;
        for phase in 0..16u32 {
            let y0 = 8.0 + (phase as f32) * 0.0625;
            let prims = [GpuPrimitive::quad(
                [4.0, y0, 24.0, 14.0],
                [0.0; 4],
                Color::BLACK,
                bw,
                Color::WHITE,
                CLIP_NONE,
                1.0,
            )];
            let frame = RenderFrame {
                width: 32,
                height: 32,
                scale_factor: 1.25,
                clear_color: Color::BLACK,
                primitives: &prims,
                dirty_prim_ranges: &[],
                clips: &[],
                damage: Some(DamageRect {
                    x: 0,
                    y: 0,
                    w: 32,
                    h: 32,
                }),
            };
            cpu.render(&frame, &mut atlas).unwrap();
            let mut lin_sum = 0.0_f32;
            for py in 7..12u32 {
                let srgb_u8 = cpu.pixel_at(16, py)[0];
                lin_sum += SRGB_TO_LINEAR_LUT[srgb_u8 as usize];
            }
            assert!(
                (lin_sum - bw).abs() <= 0.012,
                "Sub-pixel hairline energy not conserved at phase {phase}/16 (y0={y0}): integrated={lin_sum} vs expected={bw}"
            );
        }
    }
}
