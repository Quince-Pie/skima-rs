//! Unified GPU Scene Representation, Damage-Rect / Buffer-Age Tracker, Precompiled SPIR-V,
//! and Zero-Overhead `RenderBackend` Trait (implemented by Native Vulkan 1.3 and CPU Reference,
//! and directly compatible with optional WebGPU / `wgpu` pipelines).

pub mod cpu;
pub mod vulkan;

use crate::color::Color;
use crate::text::GlyphAtlas;
use bytemuck::{Pod, Zeroable};
use core::range::Range as CopyRange;

/// Canonical WGSL shader source (used to build precompiled SPIR-V at compile time via `build.rs`,
/// and exported directly for optional WebGPU / `wgpu` integration).
pub const UBER_SHADER_WGSL: &str = include_str!("../../shaders/uber.wgsl");

include!(concat!(env!("OUT_DIR"), "/uber_spv.rs"));

pub const CLIP_NONE: u32 = 0x00FF_FFFF;

/// Round a non-negative finite `f32` half-away-from-zero (`0.5 -> 1.0`, `1.4999999 -> 1.0`, `1.5 -> 2.0`),
/// bit-identical to `f32::round()` across all `x >= 0.0` while remaining usable in `const` contexts.
#[inline]
const fn const_round_nonneg_f32(x: f32) -> f32 {
    if x >= 8_388_608.0 {
        x
    } else {
        ((x as f64 + 0.5) as u32) as f32
    }
}

/// Compute a crisp, Moiré-free physical-pixel border width for `logical_bw` at `scale_factor`:
///
/// - **Zero / non-positive / non-finite (`<= 0.0`, `NaN`, `Inf`):** returns `0.0`.
/// - **Sub-pixel logical hairlines (`0.0 < logical_bw < 0.75`, e.g. `0.5px` VirtualList dividers):**
///   preserves exact single-pixel analytical energy conservation `logical_bw * scale_factor` while
///   `< 1.0` physical px (touching strictly 1 physical boundary pixel on integer-snapped boxes with
///   `0.0` inner leak and `0` LSB row-to-row luminance variance), and snaps to integer physical pixels
///   once `>= 1.0` (e.g. `1.0` physical px at `2.0x..2.75x`, preventing a `1.125px` 2-pixel fringe at `2.25x`).
/// - **Standard & heavy UI borders (`logical_bw >= 0.75`, e.g. `1.0px` card/button/input borders,
///   `1.5px` accents, `2.0px` focus rings):** snaps to `(logical_bw * scale_factor).round().max(1.0)`
///   integer physical pixels (`1.0` px at `1.0x..1.375x`, `2.0` px at `1.5x..2.375x` for `1px` borders),
///   preserving strict `0.5px < 1.0px < 2.0px` visual hierarchy with zero 2-pixel blur.
#[inline]
pub const fn snap_physical_border_width(logical_bw: f32, scale_factor: f32) -> f32 {
    if !logical_bw.is_finite()
        || logical_bw <= 0.0
        || !scale_factor.is_finite()
        || scale_factor <= 0.0
    {
        return 0.0;
    }
    let raw = logical_bw * scale_factor;
    if !raw.is_finite() {
        return 0.0;
    }
    if logical_bw < 0.75 {
        if raw < 1.0 {
            raw
        } else {
            const_round_nonneg_f32(raw)
        }
    } else {
        let snapped = const_round_nonneg_f32(raw);
        if snapped < 1.0 { 1.0 } else { snapped }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PrimKind {
    Quad = 0,
    Shadow = 1,
    LinearGradient = 2,
    GlyphGray = 3,
    GlyphLcd = 4,
    ImageRgba = 5,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct ViewportUniform {
    pub transform: [f32; 4], // [2.0 / width, 2.0 / height, scale_factor, 0.0]
    pub scissor: [f32; 4],   // [sx0, sy0, sx1, sy1] in physical pixels
}

const _: () = assert!(core::mem::size_of::<ViewportUniform>() == 32);

/// 64-byte (1 exact cache line) GPU primitive matching `struct GpuPrimitive` in `uber.wgsl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct GpuPrimitive {
    pub bounds: [f32; 4],      // [x, y, w, h]
    pub radii: [f32; 4],       // [tl, tr, br, bl]
    pub uv_or_extra: [f32; 4], // [border_w, shadow_sigma, dx_or_cos, dy_or_sin] OR [u0, v0, u1, v1]
    pub packed: [u32; 4],      // [color0, color1, (kind << 24) | clip_idx, opacity_u16]
}

const _: () = assert!(core::mem::size_of::<GpuPrimitive>() == 64);

impl GpuPrimitive {
    #[inline]
    pub fn quad(
        bounds: [f32; 4],
        radii: [f32; 4],
        bg: Color,
        border_w: f32,
        border_color: Color,
        clip_idx: u32,
        opacity: f32,
    ) -> Self {
        let op = (opacity.clamp(0.0, 1.0) * 65535.0).round() as u32;
        Self {
            bounds,
            radii,
            uv_or_extra: [border_w, 0.0, 0.0, 0.0],
            packed: [
                bg.0,
                border_color.0,
                ((PrimKind::Quad as u32) << 24) | (clip_idx & CLIP_NONE),
                op,
            ],
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_gradient(
        bounds: [f32; 4],
        radii: [f32; 4],
        c0: Color,
        c1: Color,
        angle_rad: f32,
        border_w: f32,
        clip_idx: u32,
        opacity: f32,
    ) -> Self {
        let op = (opacity.clamp(0.0, 1.0) * 65535.0).round() as u32;
        Self {
            bounds,
            radii,
            uv_or_extra: [border_w, 0.0, angle_rad.cos(), angle_rad.sin()],
            packed: [
                c0.0,
                c1.0,
                ((PrimKind::LinearGradient as u32) << 24) | (clip_idx & CLIP_NONE),
                op,
            ],
        }
    }

    #[inline]
    pub fn shadow(
        bounds: [f32; 4],
        radii: [f32; 4],
        color: Color,
        blur_sigma: f32,
        offset: [f32; 2],
        clip_idx: u32,
        opacity: f32,
    ) -> Self {
        let op = (opacity.clamp(0.0, 1.0) * 65535.0).round() as u32;
        Self {
            bounds,
            radii,
            uv_or_extra: [0.0, blur_sigma.max(0.5), offset[0], offset[1]],
            packed: [
                color.0,
                0,
                ((PrimKind::Shadow as u32) << 24) | (clip_idx & CLIP_NONE),
                op,
            ],
        }
    }

    #[inline]
    pub fn atlas_quad(
        kind: PrimKind,
        bounds: [f32; 4],
        uv: [f32; 4],
        tint: Color,
        clip_idx: u32,
        opacity: f32,
    ) -> Self {
        let op = (opacity.clamp(0.0, 1.0) * 65535.0).round() as u32;
        Self {
            bounds,
            radii: [0.0; 4],
            uv_or_extra: uv,
            packed: [
                tint.0,
                0,
                ((kind as u32) << 24) | (clip_idx & CLIP_NONE),
                op,
            ],
        }
    }

    #[inline]
    pub fn kind(&self) -> PrimKind {
        match (self.packed[2] >> 24) as u8 {
            0 => PrimKind::Quad,
            1 => PrimKind::Shadow,
            2 => PrimKind::LinearGradient,
            3 => PrimKind::GlyphGray,
            4 => PrimKind::GlyphLcd,
            _ => PrimKind::ImageRgba,
        }
    }

    /// Fast check for invisible layout-wrapper primitives (zero alpha & zero border).
    #[inline]
    pub fn is_invisible(&self) -> bool {
        if (self.packed[3] & 0xFFFF) == 0 || self.bounds[2] <= 0.0 || self.bounds[3] <= 0.0 {
            return true;
        }
        match self.kind() {
            PrimKind::Quad => {
                (self.packed[0] & 0xFF) == 0
                    && (self.uv_or_extra[0] <= 0.0 || (self.packed[1] & 0xFF) == 0)
            }
            PrimKind::LinearGradient => {
                (self.packed[0] & 0xFF) == 0 && (self.packed[1] & 0xFF) == 0
            }
            PrimKind::Shadow | PrimKind::GlyphGray | PrimKind::GlyphLcd | PrimKind::ImageRgba => {
                (self.packed[0] & 0xFF) == 0
            }
        }
    }

    /// Screen-space conservative bounding box including analytical shadow penumbra / AA margin
    /// (matches `vs_main` in `shaders/uber.wgsl` identically).
    #[inline]
    pub fn screen_footprint(&self) -> [f32; 4] {
        let kind = self.kind();
        let (margin, shift_x, shift_y) = match kind {
            PrimKind::Shadow => {
                let sigma = self.uv_or_extra[1].max(0.5);
                (sigma * 3.5 + 2.0, self.uv_or_extra[2], self.uv_or_extra[3])
            }
            PrimKind::Quad | PrimKind::LinearGradient => (1.0, 0.0, 0.0),
            PrimKind::GlyphGray | PrimKind::GlyphLcd | PrimKind::ImageRgba => (0.0, 0.0, 0.0),
        };
        let x0 = self.bounds[0] + shift_x - margin;
        let y0 = self.bounds[1] + shift_y - margin;
        let x1 = self.bounds[0] + self.bounds[2] + shift_x + margin;
        let y1 = self.bounds[1] + self.bounds[3] + shift_y + margin;
        [x0, y0, x1 - x0, y1 - y0]
    }
}

/// 48-byte GPU clip node supporting nested rounded-rectangle clip chains without stencil buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct GpuClip {
    pub bounds: [f32; 4],
    pub radii: [f32; 4],
    pub meta: [u32; 4], // [parent_clip_idx, 0, 0, 0]
}

const _: () = assert!(core::mem::size_of::<GpuClip>() == 48);

/// Integer pixel scissor/damage rectangle `[x, y, width, height]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DamageRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl DamageRect {
    pub fn from_float_bounds(b: [f32; 4], max_w: u32, max_h: u32) -> Option<Self> {
        if !(b[2] > 0.0 && b[3] > 0.0)
            || !b[0].is_finite()
            || !b[1].is_finite()
            || !b[2].is_finite()
            || !b[3].is_finite()
        {
            return None;
        }
        let x0 = (b[0].floor().max(0.0) as u32).min(max_w);
        let y0 = (b[1].floor().max(0.0) as u32).min(max_h);
        let x1 = ((b[0] + b[2]).ceil().max(0.0) as u32).min(max_w);
        let y1 = ((b[1] + b[3]).ceil().max(0.0) as u32).min(max_h);
        if x1 > x0 && y1 > y0 {
            Some(Self {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            })
        } else {
            None
        }
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = self.x.saturating_add(self.w).max(other.x.saturating_add(other.w));
        let y1 = self.y.saturating_add(self.h).max(other.y.saturating_add(other.h));
        Self {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    #[inline]
    pub fn intersects_float(&self, fb: [f32; 4]) -> bool {
        if !(fb[2] > 0.0 && fb[3] > 0.0) {
            return false;
        }
        let sx0 = self.x as f32;
        let sy0 = self.y as f32;
        let sx1 = self.x.saturating_add(self.w) as f32;
        let sy1 = self.y.saturating_add(self.h) as f32;
        fb[0] < sx1 && (fb[0] + fb[2]) > sx0 && fb[1] < sy1 && (fb[1] + fb[3]) > sy0
    }
}

/// Swapchain-aware Damage Tracker: records dirty primitive spans and coalesces screen-space `DamageRect`s
/// across swapchain buffer ages (up to 4 triple/quad-buffered images).
pub struct DamageTracker {
    history: [Option<DamageRect>; 4],
    head: usize,
    last_size: (u32, u32),
}

impl Default for DamageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DamageTracker {
    pub const fn new() -> Self {
        Self {
            history: [None; 4],
            head: 0,
            last_size: (0, 0),
        }
    }

    /// Record current frame's damage and return the coalesced scissor rectangle needed for a backbuffer
    /// of `buffer_age` (`1` = single-buffered / offscreen target, `2` = double-buffered, `3` = triple-buffered).
    pub fn resolve_frame_damage(
        &mut self,
        frame_damage: Option<DamageRect>,
        width: u32,
        height: u32,
        buffer_age: u8,
    ) -> Option<DamageRect> {
        if width == 0 || height == 0 {
            self.last_size = (0, 0);
            self.history = [None; 4];
            return None;
        }
        let full = DamageRect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
        if self.last_size != (width, height) {
            self.last_size = (width, height);
            self.history = [Some(full); 4];
            return Some(full);
        }
        self.history[self.head] = frame_damage;
        let age = (buffer_age as usize).clamp(1, 4);
        let mut acc: Option<DamageRect> = None;
        for step in 0..age {
            let idx = (self.head + 4 - step) & 3;
            if let Some(d) = self.history[idx] {
                acc = Some(acc.map_or(d, |cur| cur.union(d)));
            }
        }
        self.head = (self.head + 1) & 3;
        acc
    }
}

/// Complete prepared GPU scene frame passed to any `RenderBackend`.
pub struct RenderFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub scale_factor: f32,
    pub clear_color: Color,
    pub primitives: &'a [GpuPrimitive],
    pub dirty_prim_ranges: &'a [CopyRange<u32>],
    pub clips: &'a [GpuClip],
    pub damage: Option<DamageRect>,
}

/// Graphics-API-agnostic zero-overhead rendering backend contract.
pub trait RenderBackend {
    type Error: std::fmt::Debug;

    /// Render `frame` and upload any dirty `atlas` regions.
    fn render(&mut self, frame: &RenderFrame<'_>, atlas: &mut GlyphAtlas) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::cpu::CpuRenderer;
    use crate::gpu::vulkan::VulkanRenderer;
    use ash::vk;

    fn assert_gpu_cpu_parity(
        width: u32,
        height: u32,
        clear_color: Color,
        prims: &[GpuPrimitive],
        clips: &[GpuClip],
        atlas: &mut GlyphAtlas,
        max_lsb_tol: i32,
    ) -> (CpuRenderer, Vec<u8>) {
        assert_gpu_cpu_parity_at_scale(
            width,
            height,
            1.0,
            clear_color,
            prims,
            clips,
            atlas,
            max_lsb_tol,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_gpu_cpu_parity_at_scale(
        width: u32,
        height: u32,
        scale_factor: f32,
        clear_color: Color,
        prims: &[GpuPrimitive],
        clips: &[GpuClip],
        atlas: &mut GlyphAtlas,
        max_lsb_tol: i32,
    ) -> (CpuRenderer, Vec<u8>) {
        let frame = RenderFrame {
            width,
            height,
            scale_factor,
            clear_color,
            primitives: prims,
            dirty_prim_ranges: &[],
            clips,
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: width,
                h: height,
            }),
        };

        let mut vk = VulkanRenderer::new_headless(prims.len().max(256), clips.len().max(64))
            .expect("Vulkan 1.3 headless renderer");
        let vk_rgba = vk
            .render_offscreen_rgba8(&frame, atlas)
            .expect("Vulkan offscreen render");

        let mut cpu = CpuRenderer::new(width, height);
        cpu.render(&frame, atlas).unwrap();

        assert_eq!(vk_rgba.len(), cpu.framebuffer.len());
        let mut max_diff = 0i32;
        let mut worst = (0u32, 0u32, [0u8; 4], [0u8; 4]);
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) as usize) * 4;
                let vp = [vk_rgba[i], vk_rgba[i + 1], vk_rgba[i + 2], vk_rgba[i + 3]];
                let cp = cpu.pixel_at(x, y);
                for c in 0..4 {
                    let diff = (vp[c] as i32 - cp[c] as i32).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        worst = (x, y, vp, cp);
                    }
                }
            }
        }
        assert!(
            max_diff <= max_lsb_tol,
            "GPU vs CPU pixel parity exceeded {max_lsb_tol} LSB (max_diff={max_diff} at ({}, {}): vk={:?} vs cpu={:?})",
            worst.0,
            worst.1,
            worst.2,
            worst.3
        );
        (cpu, vk_rgba)
    }

    #[test]
    fn parity_thick_borders_per_corner_radii_and_hollow_border_no_halo() {
        let mut atlas = GlyphAtlas::new(64, 64);

        // 1. Hollow white border (`bg = TRANSPARENT`, `border = 4px WHITE`) over pure WHITE backdrop:
        //    Must have ZERO dark inner/outer premultiplication fringe anywhere!
        let hollow = [GpuPrimitive::quad(
            [12.0, 12.0, 72.0, 56.0],
            [14.0, 0.0, 6.0, 20.0],
            Color::TRANSPARENT,
            4.0,
            Color::WHITE,
            CLIP_NONE,
            1.0,
        )];
        let (cpu_hollow, vk_hollow) =
            assert_gpu_cpu_parity(96, 80, Color::WHITE, &hollow, &[], &mut atlas, 2);
        for px in cpu_hollow.framebuffer.as_chunks::<4>().0 {
            assert!(
                px[0] >= 254 && px[1] >= 254 && px[2] >= 254,
                "Dark halo detected on hollow white border over white backdrop (CPU): {px:?}"
            );
        }
        for px in vk_hollow.as_chunks::<4>().0 {
            assert!(
                px[0] >= 254 && px[1] >= 254 && px[2] >= 254,
                "Dark halo detected on hollow white border over white backdrop (Vulkan): {px:?}"
            );
        }

        // 2. Thick borders (`border_w >= corner_radius` on `[20, 0, 6, 12]` with `border_w = 10`,
        //    plus degenerate `border_w >= half_height` with `h = 18, border_w = 12`):
        let prims = [
            GpuPrimitive::quad(
                [10.0, 10.0, 80.0, 60.0],
                [20.0, 0.0, 6.0, 12.0],
                Color::rgb8(35, 130, 235),
                10.0,
                Color::rgb8(245, 200, 60),
                CLIP_NONE,
                1.0,
            ),
            // `border_w = 12.0 > h * 0.5 = 9.0`: inner hole completely closed -> 0% red fill bleed!
            GpuPrimitive::quad(
                [10.0, 80.0, 80.0, 18.0],
                [6.0; 4],
                Color::rgb8(255, 0, 0),
                12.0,
                Color::rgb8(40, 210, 120),
                CLIP_NONE,
                1.0,
            ),
        ];
        let (cpu_thick, _) =
            assert_gpu_cpu_parity(104, 110, Color::rgb8(18, 20, 28), &prims, &[], &mut atlas, 3);
        // Centerline of second quad (`x = 50, y = 89`) must be pure green border `(40, 210, 120)` with ZERO red bleed:
        let center_solid_border = cpu_thick.pixel_at(50, 89);
        assert!(
            center_solid_border[0] <= 42 && center_solid_border[1] >= 208,
            "Unexpected fill bleed in solid thick border: {center_solid_border:?}"
        );
    }

    #[test]
    fn parity_arbitrary_depth_nested_rounded_clips_and_smooth_corner_aa() {
        let mut atlas = GlyphAtlas::new(64, 64);

        // Build a 28-level deep clip chain (> 16 old convenience cap!):
        // Levels 0..27 are coincident rounded `[10, 10, 80, 80]` with `r = 18.0`,
        // and root level 0 ALSO clips `x >= 20.0` to prove the deepest ancestor (step #28) is honored!
        let mut clips = Vec::with_capacity(28);
        clips.push(GpuClip {
            bounds: [20.0, 10.0, 70.0, 80.0],
            radii: [0.0, 18.0, 18.0, 0.0],
            meta: [CLIP_NONE, 0, 0, 0],
        });
        for i in 1..28u32 {
            clips.push(GpuClip {
                bounds: [10.0, 10.0, 80.0, 80.0],
                radii: [18.0; 4],
                meta: [i - 1, 0, 0, 0],
            });
        }

        // Also test a collapsed 0-height clip (`clips[28]`) -> must clip 100% of child!
        clips.push(GpuClip {
            bounds: [10.0, 50.0, 80.0, 0.0],
            radii: [8.0; 4],
            meta: [CLIP_NONE, 0, 0, 0],
        });

        let prims = [
            GpuPrimitive::quad(
                [0.0, 0.0, 100.0, 100.0],
                [0.0; 4],
                Color::WHITE,
                0.0,
                Color::TRANSPARENT,
                27, // Starts at depth 28 (`27 -> 26 -> ... -> 0`)
                1.0,
            ),
            GpuPrimitive::quad(
                [0.0, 0.0, 100.0, 100.0],
                [0.0; 4],
                Color::rgb8(255, 0, 0),
                0.0,
                Color::TRANSPARENT,
                28, // Collapsed 0-height clip -> must be 100% invisible!
                1.0,
            ),
        ];

        let (cpu, _) =
            assert_gpu_cpu_parity(100, 100, Color::BLACK, &prims, &clips, &mut atlas, 3);

        // `x = 15, y = 50` is inside `clips[1..27]` (`x >= 10`) but outside `clips[0]` (`x >= 20`) at depth 28:
        assert_eq!(
            cpu.pixel_at(15, 50),
            [0, 0, 0, 255],
            "Ancestor clip at depth 28 must clip x=15!"
        );
        // `x = 50, y = 50` is inside all 28 clips (and 0-height red quad must not leak):
        assert_eq!(cpu.pixel_at(50, 50), [255, 255, 255, 255]);

        // Verify smooth non-binary SDF antialiasing along top-right rounded corner `(center = (72, 28), r = 18)`
        // despite 28 coincident rounded clips (no 0.5^28 erosion!):
        let mut aa_pixels = 0usize;
        for y in 10..28 {
            for x in 72..90 {
                let v = cpu.pixel_at(x, y)[0];
                if v > 16 && v < 240 {
                    aa_pixels += 1;
                }
            }
        }
        assert!(
            aa_pixels >= 12,
            "Expected smooth 1px AA ramp on 28-deep rounded clip corner, got only {aa_pixels} intermediate pixels"
        );
    }

    #[test]
    fn parity_premultiplied_oklab_gradient_and_rounded_erf_shadow() {
        let mut atlas = GlyphAtlas::new(64, 64);

        let prims = [
            // 1. Premultiplied OKLab gradient from vibrant Blue to `Color::TRANSPARENT` over WHITE:
            GpuPrimitive::linear_gradient(
                [10.0, 10.0, 100.0, 36.0],
                [8.0; 4],
                Color::rgb8(30, 110, 250),
                Color::TRANSPARENT,
                0.0,
                0.0,
                CLIP_NONE,
                1.0,
            ),
            // 2. Pure Black -> Pure White OKLab gradient (verifies `cbrt_pos(0.0)` has zero NaNs on SPIR-V):
            GpuPrimitive::linear_gradient(
                [10.0, 54.0, 100.0, 24.0],
                [4.0; 4],
                Color::BLACK,
                Color::WHITE,
                0.0,
                0.0,
                CLIP_NONE,
                1.0,
            ),
            // 3. Circular pill (`60x60`, `r = 30`) with tight Gaussian shadow (`sigma = 4`, `offset = [0, 4]`):
            GpuPrimitive::shadow(
                [130.0, 18.0, 60.0, 60.0],
                [30.0; 4],
                Color::rgba8(0, 0, 0, 200),
                4.0,
                [0.0, 4.0],
                CLIP_NONE,
                1.0,
            ),
        ];

        let (cpu, _) =
            assert_gpu_cpu_parity(210, 96, Color::WHITE, &prims, &[], &mut atlas, 3);

        // Midpoint of Blue -> Transparent over White `(60, 28)` must be bright clean sky-blue, NOT muddy gray!
        let mid = cpu.pixel_at(60, 28);
        assert!(
            mid[2] >= 245 && mid[1] >= 165 && mid[0] >= 130,
            "Premultiplied OKLab gradient to TRANSPARENT darkened midpoint: {mid:?}"
        );

        // Shadow right below bottom of 60x60 circle (`center = (160, 48)`, shifted bottom = `82`):
        // At `y = 83.5` (`+1.5px` outside circle), `cov = 0.353`, `alpha = 0.277`, linear `0.723` -> sRGB `221`:
        let sh_bottom = cpu.pixel_at(160, 83);
        assert!(
            sh_bottom[0] <= 225,
            "Circular pill drop shadow must extend cleanly beyond pill edge, got {sh_bottom:?}"
        );
        // Outside 45° bounding-box corner `(189, 81)` must be much lighter (`~255`) than bottom `(160, 83)` because contour is circular:
        let sh_corner = cpu.pixel_at(189, 81);
        assert!(
            sh_corner[0] >= sh_bottom[0] + 25,
            "Rounded drop shadow must follow circular corner contour: corner={sh_corner:?} vs bottom={sh_bottom:?}"
        );
    }

    #[test]
    fn parity_cleartype_lcd_dark_and_light_and_single_src_fallback() {
        let mut atlas = GlyphAtlas::new(64, 64);
        // Upload a 3x4 synthetic ClearType subpixel stem:
        // col 0: [80, 140, 210, 255], col 1: [255, 255, 255, 255], col 2: [210, 140, 80, 255]
        let mut glyph_rgba = Vec::with_capacity(3 * 4 * 4);
        for _ in 0..4 {
            glyph_rgba.extend_from_slice(&[80, 140, 210, 255]);
            glyph_rgba.extend_from_slice(&[255, 255, 255, 255]);
            glyph_rgba.extend_from_slice(&[210, 140, 80, 255]);
        }
        let reg = atlas.upload_rgba8(3, 4, 0, 4, &glyph_rgba).unwrap();
        let uv = [reg.u0, reg.v0, reg.u1, reg.v1];

        // 1. Dark text on Light background:
        let dark_on_light = [GpuPrimitive::atlas_quad(
            PrimKind::GlyphLcd,
            [10.0, 10.0, 3.0, 4.0],
            uv,
            Color::rgb8(18, 20, 28),
            CLIP_NONE,
            1.0,
        )];
        let (cpu_light, _) =
            assert_gpu_cpu_parity(32, 24, Color::WHITE, &dark_on_light, &[], &mut atlas, 3);
        let left_edge = cpu_light.pixel_at(10, 11);
        let stem_core = cpu_light.pixel_at(11, 11);
        assert!(
            left_edge[0] > left_edge[1] && left_edge[1] > left_edge[2],
            "Expected per-channel ClearType subpixel ramp on left stem edge: {left_edge:?}"
        );
        assert!(
            left_edge[1] < 165 && stem_core[1] <= 22,
            "Expected high-contrast stem-darkened dark text on light background: edge={left_edge:?}, core={stem_core:?}"
        );

        // 2. Light text on Dark background:
        atlas.generation += 1;
        let light_on_dark = [GpuPrimitive::atlas_quad(
            PrimKind::GlyphLcd,
            [10.0, 10.0, 3.0, 4.0],
            uv,
            Color::WHITE,
            CLIP_NONE,
            1.0,
        )];
        let _ = assert_gpu_cpu_parity(
            32,
            24,
            Color::rgb8(16, 18, 26),
            &light_on_dark,
            &[],
            &mut atlas,
            3,
        );

        // 3. Single-target `UBER_FRAG_SPV` fallback pipeline (`use_dual_src = false`) on real Vulkan 1.3:
        //    Drawing WHITE ClearType text over WHITE background must NEVER produce rainbow color fringes!
        let mut vk = VulkanRenderer::new_headless(256, 64).unwrap();
        let fallback_pipe = VulkanRenderer::build_graphics_pipeline(
            &vk.device,
            vk.pipeline_layout,
            vk::Format::R8G8B8A8_SRGB,
            false,
        )
        .expect("Single-source fallback pipeline must compile cleanly");
        atlas.generation += 1;
        let white_on_white_frame = RenderFrame {
            width: 32,
            height: 24,
            scale_factor: 1.0,
            clear_color: Color::WHITE,
            primitives: &light_on_dark,
            dirty_prim_ranges: &[],
            clips: &[],
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: 32,
                h: 24,
            }),
        };
        vk.render_internal_with_pipeline(
            &white_on_white_frame,
            &mut atlas,
            true,
            Some(fallback_pipe),
        )
        .unwrap();
        let rb = vk.readback_buf.as_ref().unwrap();
        let mut fb_bytes = vec![0u8; 32 * 24 * 4];
        unsafe {
            core::ptr::copy_nonoverlapping(rb.mapped, fb_bytes.as_mut_ptr(), fb_bytes.len());
            vk.device.destroy_pipeline(fallback_pipe, None);
        }
        for px in fb_bytes.as_chunks::<4>().0 {
            assert!(
                px[0] >= 254 && px[1] >= 254 && px[2] >= 254,
                "Single-target GlyphLcd fallback produced color fringe on white-over-white: {px:?}"
            );
        }
    }

    #[test]
    fn vulkan_dynamic_staging_growth_atlas_resize_non_coherent_flush_and_zero_viewport() {
        let mut vk = VulkanRenderer::new_headless(256, 64).unwrap();

        // Force tiny 256-byte initial `staging_buf` AND simulate a strict non-HOST_COHERENT device
        // with `nonCoherentAtomSize = 256` across ALL mapped buffers!
        unsafe {
            vk.device.device_wait_idle().unwrap();
            let mem_props = vk
                .instance
                .get_physical_device_memory_properties(vk.physical_device);
            let (mut tiny_staging, _) = VulkanRenderer::create_mapped_buffer(
                &vk.device,
                &mem_props,
                256,
                vk::BufferUsageFlags::TRANSFER_SRC,
                false,
                false,
            )
            .unwrap();
            tiny_staging.is_coherent = false;
            let old_s = core::mem::replace(&mut vk.staging_buf, tiny_staging);
            VulkanRenderer::destroy_mapped_buffer(&vk.device, old_s);
        }
        vk.non_coherent_atom_size = 256;
        vk.uniform_buf.is_coherent = false;
        vk.prim_buf.is_coherent = false;
        vk.clip_buf.is_coherent = false;

        // 1. 0x0 viewport safety on Vulkan:
        let mut custom_atlas = GlyphAtlas::new(256, 256);
        let empty = RenderFrame {
            width: 0,
            height: 0,
            scale_factor: 1.0,
            clear_color: Color::BLACK,
            primitives: &[],
            dirty_prim_ranges: &[],
            clips: &[],
            damage: None,
        };
        vk.render(&empty, &mut custom_atlas).unwrap();
        assert!(vk.render_offscreen_rgba8(&empty, &mut custom_atlas).unwrap().is_empty());

        // 2. Sync initial 256x256 custom atlas (grows staging_buf past 256B and resizes GPU atlas from 1024->256):
        let init_frame = RenderFrame {
            width: 64,
            height: 64,
            scale_factor: 1.0,
            clear_color: Color::BLACK,
            primitives: &[],
            dirty_prim_ranges: &[],
            clips: &[],
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: 64,
                h: 64,
            }),
        };
        let _ = vk.render_offscreen_rgba8(&init_frame, &mut custom_atlas).unwrap();
        vk.readback_buf.as_mut().unwrap().is_coherent = false;

        // Shrink `staging_buf` back to 256B NOW that `atlas_gen_uploaded == custom_atlas.generation`,
        // to test incremental `dirty_rects` dynamic staging growth without dropping ANY dirty rects!
        unsafe {
            vk.device.device_wait_idle().unwrap();
            let mem_props = vk
                .instance
                .get_physical_device_memory_properties(vk.physical_device);
            let (mut tiny_staging, _) = VulkanRenderer::create_mapped_buffer(
                &vk.device,
                &mem_props,
                256,
                vk::BufferUsageFlags::TRANSFER_SRC,
                false,
                false,
            )
            .unwrap();
            tiny_staging.is_coherent = false;
            let old_s = core::mem::replace(&mut vk.staging_buf, tiny_staging);
            VulkanRenderer::destroy_mapped_buffer(&vk.device, old_s);
        }
        assert_eq!(vk.staging_buf.size, 256);

        // Upload 12 incremental 32x32 RGBA8 blocks (`12 * 4096 = 49,152 B >> 256 B`!) WITHOUT changing `generation`:
        let gen_before = custom_atlas.generation;
        let green_tile = vec![255u8; 32 * 32 * 4];
        let mut last_reg = None;
        for _ in 0..12 {
            last_reg = custom_atlas.upload_rgba8(32, 32, 0, 0, &green_tile);
        }
        assert_eq!(
            custom_atlas.generation, gen_before,
            "256x256 atlas fits twelve 32x32 tiles without saturation reset"
        );
        let reg = last_reg.expect("12th incremental atlas upload");

        // Render the 12th tile (which previously would have been DROPPED by `break` when staging_buf filled!):
        let prims = [GpuPrimitive::atlas_quad(
            PrimKind::ImageRgba,
            [8.0, 8.0, 32.0, 32.0],
            [reg.u0, reg.v0, reg.u1, reg.v1],
            Color::rgb8(40, 230, 90),
            CLIP_NONE,
            1.0,
        )];
        let dirty_ranges = [CopyRange { start: 0, end: 1 }];
        let test_frame = RenderFrame {
            width: 64,
            height: 64,
            scale_factor: 1.0,
            clear_color: Color::rgb8(10, 12, 18),
            primitives: &prims,
            dirty_prim_ranges: &dirty_ranges,
            clips: &[],
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: 64,
                h: 64,
            }),
        };
        let out = vk
            .render_offscreen_rgba8(&test_frame, &mut custom_atlas)
            .expect("Non-coherent flush/invalidate + dynamic staging growth must succeed");
        assert!(
            vk.staging_buf.size >= 49_152,
            "staging_buf must have grown dynamically from 256B, got {}",
            vk.staging_buf.size
        );
        assert!(custom_atlas.dirty_rects.is_empty());

        let px_idx = (20 * 64 + 20) * 4;
        let rendered_px = [out[px_idx], out[px_idx + 1], out[px_idx + 2], out[px_idx + 3]];
        assert!(
            (rendered_px[0] as i32 - 40).abs() <= 2
                && (rendered_px[1] as i32 - 230).abs() <= 2
                && (rendered_px[2] as i32 - 90).abs() <= 2,
            "12th incremental dirty_rect must upload & render accurately (not dropped!), got {rendered_px:?}"
        );
    }

    #[test]
    fn parity_partial_damage_moving_and_translucent_primitives_no_ghosting_or_alpha_accumulation() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let clear = Color::rgb8(18, 22, 34);
        let mut vk = VulkanRenderer::new_headless(256, 64).expect("Vulkan 1.3 headless");
        let mut cpu = CpuRenderer::new(96, 64);

        // Frame 1:
        // - prim[0]: static green badge at [6, 6, 18, 18] (outside Frame 2's partial damage rect)
        // - prim[1]: moving coral quad at [36, 10, 20, 16] (over raw `clear_color`, no background quad!)
        // - prim[2]: translucent cyan quad (alpha = 128) at [64, 14, 22, 22] (inside Frame 2's damage rect)
        let mut prims = [
            GpuPrimitive::quad(
                [6.0, 6.0, 18.0, 18.0],
                [4.0; 4],
                Color::rgb8(50, 210, 110),
                0.0,
                Color::TRANSPARENT,
                CLIP_NONE,
                1.0,
            ),
            GpuPrimitive::quad(
                [36.0, 10.0, 20.0, 16.0],
                [4.0; 4],
                Color::rgb8(240, 80, 70),
                0.0,
                Color::TRANSPARENT,
                CLIP_NONE,
                1.0,
            ),
            GpuPrimitive::quad(
                [64.0, 14.0, 22.0, 22.0],
                [6.0; 4],
                Color::rgba8(40, 200, 240, 128),
                0.0,
                Color::TRANSPARENT,
                CLIP_NONE,
                1.0,
            ),
        ];

        let frame1 = RenderFrame {
            width: 96,
            height: 64,
            scale_factor: 1.0,
            clear_color: clear,
            primitives: &prims,
            dirty_prim_ranges: &[],
            clips: &[],
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: 96,
                h: 64,
            }),
        };
        let vk_f1 = vk.render_offscreen_rgba8(&frame1, &mut atlas).unwrap();
        cpu.render(&frame1, &mut atlas).unwrap();
        let trans_f1_vk = {
            let i = (25 * 96 + 75) * 4;
            [vk_f1[i], vk_f1[i + 1], vk_f1[i + 2], vk_f1[i + 3]]
        };

        // Frame 2: move `prim[1]` down to `[36, 36, 20, 16]` and redraw ONLY partial `DamageRect [32..92, 6..58]`!
        prims[1] = GpuPrimitive::quad(
            [36.0, 36.0, 20.0, 16.0],
            [4.0; 4],
            Color::rgb8(240, 80, 70),
            0.0,
            Color::TRANSPARENT,
            CLIP_NONE,
            1.0,
        );
        let dirty_spans = [CopyRange { start: 1, end: 2 }];
        let frame2 = RenderFrame {
            width: 96,
            height: 64,
            scale_factor: 1.0,
            clear_color: clear,
            primitives: &prims,
            dirty_prim_ranges: &dirty_spans,
            clips: &[],
            damage: Some(DamageRect {
                x: 32,
                y: 6,
                w: 60,
                h: 52,
            }),
        };
        let vk_f2 = vk.render_offscreen_rgba8(&frame2, &mut atlas).unwrap();
        cpu.render(&frame2, &mut atlas).unwrap();

        let vk_px = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 96 + x) as usize) * 4;
            [vk_f2[i], vk_f2[i + 1], vk_f2[i + 2], vk_f2[i + 3]]
        };

        // 1. Untouched static green badge outside partial damage `(14, 14)` must remain intact:
        assert!(
            (vk_px(14, 14)[1] as i32 - 210).abs() <= 2,
            "Static primitive outside partial DamageRect must be preserved: {:?}",
            vk_px(14, 14)
        );
        // 2. Old position `(46, 18)` of moved coral quad must be exact `clear_color` `(18, 22, 34)` (ZERO ghosting!):
        let vacated = vk_px(46, 18);
        assert!(
            (vacated[0] as i32 - 18).abs() <= 2
                && (vacated[1] as i32 - 22).abs() <= 2
                && (vacated[2] as i32 - 34).abs() <= 2,
            "Ghost trail left at vacated partial-damage pixel: {vacated:?}"
        );
        // 3. New position `(46, 44)` must be coral `(240, 80, 70)`:
        assert!((vk_px(46, 44)[0] as i32 - 240).abs() <= 2);
        // 4. Translucent quad `(75, 25)` inside partial damage must NOT double-accumulate alpha on Frame 2:
        let trans_f2_vk = vk_px(75, 25);
        for c in 0..4 {
            assert!(
                (trans_f2_vk[c] as i32 - trans_f1_vk[c] as i32).abs() <= 1,
                "Translucent quad accumulated alpha on partial redraw: f1={trans_f1_vk:?} vs f2={trans_f2_vk:?}"
            );
        }
        // 5. Full 96x64 Frame-2 GPU vs CPU parity:
        for y in 0..64 {
            for x in 0..96 {
                let vp = vk_px(x, y);
                let cp = cpu.pixel_at(x, y);
                for c in 0..4 {
                    assert!(
                        (vp[c] as i32 - cp[c] as i32).abs() <= 2,
                        "Frame-2 partial-damage GPU vs CPU mismatch at ({x}, {y}): vk={vp:?} vs cpu={cp:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn parity_opaque_interior_spans_clipped_glyph_viewport_and_vertex_clip_culling() {
        let mut atlas = GlyphAtlas::new(128, 128);

        // Upload a 6x10 synthetic ClearType LCD glyph and a 6x10 R8 grayscale glyph:
        let mut lcd_bytes = Vec::with_capacity(6 * 10 * 4);
        let mut r8_bytes = Vec::with_capacity(6 * 10);
        for y in 0..10u8 {
            for x in 0..6u8 {
                let edge = x == 0 || x == 5 || y == 0 || y == 9;
                if edge {
                    lcd_bytes.extend_from_slice(&[90, 155, 220, 255]);
                    r8_bytes.push(145);
                } else {
                    lcd_bytes.extend_from_slice(&[255, 255, 255, 255]);
                    r8_bytes.push(255);
                }
            }
        }
        let lcd_reg = atlas.upload_rgba8(6, 10, 0, 8, &lcd_bytes).unwrap();
        let r8_reg = atlas.upload_r8(6, 10, 0, 8, &r8_bytes).unwrap();
        let lcd_uv = [lcd_reg.u0, lcd_reg.v0, lcd_reg.u1, lcd_reg.v1];
        let r8_uv = [r8_reg.u0, r8_reg.v0, r8_reg.u1, r8_reg.v1];

        // Nested rounded scroll viewport clip (`[16, 14, 128, 92]`, `r = 16.0`) inside outer clip (`[12, 10, 136, 100]`):
        // Also add a disjoint nested clip (`clips[2]` + `clips[3]`) to verify `vs_main` & `ResolvedClip` disjoint culling!
        let clips = [
            GpuClip {
                bounds: [12.0, 10.0, 136.0, 100.0],
                radii: [0.0; 4],
                meta: [CLIP_NONE, 0, 0, 0],
            },
            GpuClip {
                bounds: [16.0, 14.0, 128.0, 92.0],
                radii: [16.0; 4],
                meta: [0, 0, 0, 0],
            },
            GpuClip {
                bounds: [10.0, 10.0, 30.0, 30.0],
                radii: [4.0; 4],
                meta: [CLIP_NONE, 0, 0, 0],
            },
            GpuClip {
                bounds: [60.0, 60.0, 30.0, 30.0],
                radii: [4.0; 4],
                meta: [2, 0, 0, 0], // Disjoint from parent clip 2!
            },
        ];

        let mut prims = Vec::with_capacity(128);
        // 1. Occluded bright-magenta full-bleed quad (`[0, 0, 160, 120]`) completely covered by dark-slate root quad:
        //    Tests backward opaque occlusion skip on scanlines!
        prims.push(GpuPrimitive::quad(
            [0.0, 0.0, 160.0, 120.0],
            [0.0; 4],
            Color::rgb8(255, 0, 255),
            0.0,
            Color::TRANSPARENT,
            CLIP_NONE,
            1.0,
        ));
        prims.push(GpuPrimitive::quad(
            [0.0, 0.0, 160.0, 120.0],
            [0.0; 4],
            Color::rgb8(20, 24, 34),
            0.0,
            Color::TRANSPARENT,
            CLIP_NONE,
            1.0,
        ));

        // 2. Oversized scrolled container background (`y = -200..400`) inside rounded `clip = 1` + bordered card:
        prims.push(GpuPrimitive::quad(
            [16.0, -200.0, 128.0, 600.0],
            [12.0; 4],
            Color::rgb8(32, 38, 52),
            2.0,
            Color::rgb8(90, 120, 190),
            1,
            1.0,
        ));

        // 3. Translucent selection highlight quad (`alpha < 255`) + hollow border quad inside `clip = 1`:
        prims.push(GpuPrimitive::quad(
            [22.0, 38.0, 116.0, 24.0],
            [6.0; 4],
            Color::rgba8(70, 140, 250, 96),
            0.0,
            Color::TRANSPARENT,
            1,
            1.0,
        ));
        prims.push(GpuPrimitive::quad(
            [20.0, 36.0, 120.0, 28.0],
            [7.0; 4],
            Color::TRANSPARENT,
            1.5,
            Color::rgba8(160, 200, 255, 200),
            1,
            1.0,
        ));

        // 4. Grid of 1:1 snapped LCD & Grayscale glyphs spanning:
        //    - off-clip top (`y = 0`),
        //    - rounded corner AA zone (`x = 18..32, y = 16`),
        //    - straight-edge margin (`x = 18 < 16 + 16 = 32, y = 45`),
        //    - inner safe core (`x = 40..120, y = 45..85`),
        //    - off-clip bottom (`y = 112`).
        for row_y in [0.0_f32, 16.0, 45.0, 68.0, 98.0, 112.0] {
            for (col_i, col_x) in (14..144).step_by(8).enumerate() {
                let is_lcd = (col_i & 1) == 0;
                prims.push(GpuPrimitive::atlas_quad(
                    if is_lcd {
                        PrimKind::GlyphLcd
                    } else {
                        PrimKind::GlyphGray
                    },
                    [col_x as f32, row_y, 6.0, 10.0],
                    if is_lcd { lcd_uv } else { r8_uv },
                    Color::rgb8(240, 244, 252),
                    1,
                    1.0,
                ));
            }
        }

        // 5. Subpixel-scaled non-snapped glyph (`bounds = [42.3, 78.6, 9.4, 12.2]`) + disjoint-clip quad (`clip = 3`):
        prims.push(GpuPrimitive::atlas_quad(
            PrimKind::GlyphLcd,
            [42.3, 78.6, 9.4, 12.2],
            lcd_uv,
            Color::rgb8(255, 220, 120),
            1,
            0.85,
        ));
        prims.push(GpuPrimitive::quad(
            [0.0, 0.0, 160.0, 120.0],
            [0.0; 4],
            Color::rgb8(255, 0, 0),
            0.0,
            Color::TRANSPARENT,
            3, // Disjoint nested clip chain -> 0% visible!
            1.0,
        ));

        let (cpu, _) = assert_gpu_cpu_parity(
            160,
            120,
            Color::rgb8(10, 12, 18),
            &prims,
            &clips,
            &mut atlas,
            3,
        );

        // Verify magenta background (`prims[0]`) and red disjoint-clip quad (`last`) never leaked anywhere,
        // while white glyph stem at `(72, 70)` rendered cleanly inside `clip = 1`:
        assert_eq!(cpu.pixel_at(4, 4), [20, 24, 34, 255]);
        assert_eq!(cpu.pixel_at(69, 70), [32, 38, 52, 255]);
        assert!(cpu.pixel_at(72, 70)[1] >= 235);
    }

    #[test]
    fn vulkan_and_cpu_coalesced_dirty_prim_ranges_and_first_frame_partial_sync() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let clear = Color::rgb8(14, 16, 24);

        let mut prims = Vec::with_capacity(160);
        // Root background:
        prims.push(GpuPrimitive::quad(
            [0.0, 0.0, 128.0, 96.0],
            [0.0; 4],
            Color::rgb8(22, 26, 36),
            0.0,
            Color::TRANSPARENT,
            CLIP_NONE,
            1.0,
        ));
        for i in 0..150usize {
            let x = ((i % 15) * 8 + 4) as f32;
            let y = ((i / 15) * 8 + 6) as f32;
            prims.push(GpuPrimitive::quad(
                [x, y, 6.0, 6.0],
                [2.0; 4],
                Color::rgb8(45, 70, 120),
                0.0,
                Color::TRANSPARENT,
                CLIP_NONE,
                1.0,
            ));
        }

        // 1. First-frame safety: initialize BOTH `VulkanRenderer` (with `max_primitives = 512 >= 151` and simulated
        //    non-coherent `atom = 256`!) and `CpuRenderer`, and pass a NON-EMPTY partial `dirty_prim_ranges = &[5..6]`
        //    on the VERY FIRST render! Both backends must detect unsynced initial SSBO state and sync ALL 151 prims!
        let mut vk = VulkanRenderer::new_headless(512, 64).expect("Vulkan headless");
        vk.non_coherent_atom_size = 256;
        vk.uniform_buf.is_coherent = false;
        vk.prim_buf.is_coherent = false;
        vk.clip_buf.is_coherent = false;

        let mut cpu = CpuRenderer::new(128, 96);
        let initial_partial_ranges = [CopyRange { start: 5, end: 6 }];
        let frame1 = RenderFrame {
            width: 128,
            height: 96,
            scale_factor: 1.0,
            clear_color: clear,
            primitives: &prims,
            dirty_prim_ranges: &initial_partial_ranges,
            clips: &[],
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: 128,
                h: 96,
            }),
        };
        let vk_f1 = vk.render_offscreen_rgba8(&frame1, &mut atlas).unwrap();
        cpu.render(&frame1, &mut atlas).unwrap();
        assert_eq!(vk.prim_synced_count, prims.len());
        // Tile #140 (outside `5..6`!) must still have rendered on both Vulkan and CPU:
        let i140 = ((78 + 3) * 128 + (44 + 3)) * 4;
        assert!((vk_f1[i140 + 2] as i32 - 120).abs() <= 2);
        assert!((cpu.pixel_at(47, 81)[2] as i32 - 120).abs() <= 2);

        // 2. Frame 2: Mutate unsorted, overlapping, adjacent, and sparse far-apart indices
        //    (`145`, `2`, `3`, `4` via unsorted overlapping ranges `[3..5, 145..146, 2..4]`)!
        for &idx in &[2usize, 3, 4, 145] {
            let b = prims[idx].bounds;
            prims[idx] = GpuPrimitive::quad(
                b,
                [2.0; 4],
                Color::rgb8(245, 180, 40),
                0.0,
                Color::TRANSPARENT,
                CLIP_NONE,
                1.0,
            );
        }
        let messy_ranges = [
            CopyRange { start: 3, end: 5 },
            CopyRange { start: 145, end: 146 },
            CopyRange { start: 2, end: 4 },
        ];
        let frame2 = RenderFrame {
            width: 128,
            height: 96,
            scale_factor: 1.0,
            clear_color: clear,
            primitives: &prims,
            dirty_prim_ranges: &messy_ranges,
            clips: &[],
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: 128,
                h: 96,
            }),
        };
        let vk_f2 = vk.render_offscreen_rgba8(&frame2, &mut atlas).unwrap();
        cpu.render(&frame2, &mut atlas).unwrap();

        for y in 0..96u32 {
            for x in 0..128u32 {
                let i = ((y * 128 + x) as usize) * 4;
                let vp = [vk_f2[i], vk_f2[i + 1], vk_f2[i + 2], vk_f2[i + 3]];
                let cp = cpu.pixel_at(x, y);
                for c in 0..4 {
                    assert!(
                        (vp[c] as i32 - cp[c] as i32).abs() <= 2,
                        "Coalesced dirty_prim_ranges parity mismatch at ({x}, {y}): vk={vp:?} vs cpu={cp:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn snap_physical_border_width_hierarchy_const_and_adversarial_inputs() {
        const CONST_CHECK_125: f32 = snap_physical_border_width(1.0, 1.25);
        const CONST_CHECK_150: f32 = snap_physical_border_width(1.0, 1.5);
        const CONST_HAIR_125: f32 = snap_physical_border_width(0.5, 1.25);
        const CONST_HAIR_225: f32 = snap_physical_border_width(0.5, 2.25);
        assert_eq!(CONST_CHECK_125, 1.0);
        assert_eq!(CONST_CHECK_150, 2.0);
        assert_eq!(CONST_HAIR_125, 0.625);
        assert_eq!(CONST_HAIR_225, 1.0);

        // Adversarial NaN / Inf / negative / zero inputs:
        for bad in [0.0_f32, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(snap_physical_border_width(bad, 1.25), 0.0);
            if bad <= 0.0 || !bad.is_finite() {
                assert_eq!(snap_physical_border_width(1.0, bad), 0.0);
            }
        }
        assert_eq!(snap_physical_border_width(1e30, 1e20), 0.0);

        // Verify `const_round_nonneg_f32` matches `f32::round()` bit-for-bit across half-way & ULP boundaries:
        for &probe in &[
            0.0_f32,
            f32::from_bits(0x3eff_ffff), // 0.49999997 -> must round to 0.0, NOT 1.0!
            0.5,
            f32::from_bits(0x3fbf_ffff), // 1.4999999 -> 1.0
            1.5,
            f32::from_bits(0x401f_ffff), // 2.4999998 -> 2.0
            2.5,
            8_388_607.5,
            8_388_608.0,
            16_777_216.0,
        ] {
            assert_eq!(
                const_round_nonneg_f32(probe).to_bits(),
                probe.round().to_bits(),
                "const_round_nonneg_f32 mismatch at {probe} (0x{:08x})",
                probe.to_bits()
            );
        }

        // Across all fractional scales, verify strict `0 < 0.5px < 1.0px < 2.0px` visual hierarchy
        // AND zero multi-pixel fractional blur (`snapped <= 1.0` or exact integer):
        let scales = [
            1.0_f32,
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
            let b0 = snap_physical_border_width(0.0, sf);
            let b05 = snap_physical_border_width(0.5, sf);
            let b10 = snap_physical_border_width(1.0, sf);
            let b20 = snap_physical_border_width(2.0, sf);
            assert_eq!(b0, 0.0);
            assert!(
                b05 > 0.0 && b05 < b10 && b10 < b20,
                "Broken border hierarchy at sf={sf}: 0.5->{b05}, 1.0->{b10}, 2.0->{b20}"
            );
            for &lbw in &[0.5_f32, 0.75, 1.0, 1.5, 2.0, 2.5] {
                let pbw = snap_physical_border_width(lbw, sf);
                assert!(
                    pbw <= 1.0 || (pbw - pbw.round()).abs() == 0.0,
                    "Multi-pixel fractional blur at lbw={lbw}, sf={sf}: pbw={pbw}"
                );
            }
        }
        // Zoomed-out `1px` border never vanishes to `0px`:
        assert_eq!(snap_physical_border_width(1.0, 0.625), 1.0);
    }

    #[test]
    fn parity_fractional_stacked_row_dividers_zero_moire_and_zero_bleed() {
        let mut atlas = GlyphAtlas::new(64, 64);
        let row_bg = Color::rgb8(33, 38, 50);
        let border_col = Color::rgb8(82, 92, 118);
        let seam_canary = Color::rgb8(255, 0, 220);

        for &sf in &[1.125_f32, 1.25, 160.0 / 120.0, 1.5, 200.0 / 120.0, 1.75, 2.25] {
            let n_rows = 20usize;
            let log_row_h = 28.0_f32;
            let total_h = ((n_rows as f32 * log_row_h) * sf).round() as u32;
            let col_w = (76.0 * sf).round();
            let gap_x = (8.0 * sf).round();
            let phys_w = (col_w * 2.0 + gap_x + 8.0) as u32;

            let bw_hair = snap_physical_border_width(0.5, sf);
            let bw_crisp = snap_physical_border_width(1.0, sf);

            let mut prims = Vec::with_capacity(n_rows * 2 + 1);
            // Bright magenta canary behind all rows: ANY 1px physical gap between adjacent rows leaks `G=0, R=255`!
            prims.push(GpuPrimitive::quad(
                [0.0, 0.0, phys_w as f32, total_h as f32],
                [0.0; 4],
                seam_canary,
                0.0,
                Color::TRANSPARENT,
                CLIP_NONE,
                1.0,
            ));

            let mut boundaries = Vec::with_capacity(n_rows + 1);
            boundaries.push(0.0_f32);
            for i in 0..n_rows {
                let y0 = ((i as f32 * log_row_h) * sf).round();
                let y1 = (((i + 1) as f32 * log_row_h) * sf).round();
                boundaries.push(y1);
                let rh = y1 - y0;
                // Left column: `0.5px` logical hairline dividers
                prims.push(GpuPrimitive::quad(
                    [4.0, y0, col_w, rh],
                    [0.0; 4],
                    row_bg,
                    bw_hair,
                    border_col,
                    CLIP_NONE,
                    1.0,
                ));
                // Right column: `1.0px` logical crisp borders
                prims.push(GpuPrimitive::quad(
                    [4.0 + col_w + gap_x, y0, col_w, rh],
                    [0.0; 4],
                    row_bg,
                    bw_crisp,
                    border_col,
                    CLIP_NONE,
                    1.0,
                ));
            }

            let (cpu, vk_rgba) = assert_gpu_cpu_parity_at_scale(
                phys_w,
                total_h,
                sf,
                seam_canary,
                &prims,
                &[],
                &mut atlas,
                2,
            );

            let probe_x_hair = (4.0 + col_w * 0.5) as u32;
            let probe_x_crisp = (4.0 + col_w + gap_x + col_w * 0.5) as u32;
            let vk_px = |x: u32, y: u32| -> [u8; 4] {
                let idx = ((y * phys_w + x) as usize) * 4;
                [vk_rgba[idx], vk_rgba[idx + 1], vk_rgba[idx + 2], vk_rgba[idx + 3]]
            };

            // 1. Zero background canary leak anywhere inside either 20-row column:
            for py in 0..total_h {
                for &px in &[probe_x_hair, probe_x_crisp] {
                    let cp = cpu.pixel_at(px, py);
                    let vp = vk_px(px, py);
                    assert!(
                        cp[0] < 150 && vp[0] < 150,
                        "Physical seam canary leaked at sf={sf}, ({px}, {py}): cpu={cp:?}, vk={vp:?}"
                    );
                }
            }

            // 2. Zero periodic Moiré striping across all 19 interior seams (`i = 1..n_rows`):
            let ref_y = boundaries[1] as u32;
            let ref_cpu_hair_top = cpu.pixel_at(probe_x_hair, ref_y);
            let ref_cpu_hair_bot = cpu.pixel_at(probe_x_hair, ref_y - 1);
            let ref_vk_hair_top = vk_px(probe_x_hair, ref_y);
            let ref_vk_hair_bot = vk_px(probe_x_hair, ref_y - 1);
            let ref_vk_crisp_top = vk_px(probe_x_crisp, ref_y);

            for &yb_f in &boundaries[1..n_rows] {
                let yb = yb_f as u32;
                assert_eq!(
                    cpu.pixel_at(probe_x_hair, yb),
                    ref_cpu_hair_top,
                    "CPU Moiré variance on top divider pixel at sf={sf}, yb={yb}"
                );
                assert_eq!(
                    cpu.pixel_at(probe_x_hair, yb - 1),
                    ref_cpu_hair_bot,
                    "CPU Moiré variance on bottom divider pixel at sf={sf}, yb={yb}"
                );
                // When `bw_hair <= 1.0`, pixel `yb + 1` must be pure `row_bg` (`0` inner leak!):
                if bw_hair <= 1.0 {
                    assert_eq!(cpu.pixel_at(probe_x_hair, yb + 1), [33, 38, 50, 255]);
                }
                for c in 0..3 {
                    assert!(
                        (vk_px(probe_x_hair, yb)[c] as i32 - ref_vk_hair_top[c] as i32).abs() <= 1
                            && (vk_px(probe_x_hair, yb - 1)[c] as i32 - ref_vk_hair_bot[c] as i32)
                                .abs()
                                <= 1
                            && (vk_px(probe_x_crisp, yb)[c] as i32 - ref_vk_crisp_top[c] as i32)
                                .abs()
                                <= 1,
                        "Vulkan Moiré striping across stacked rows at sf={sf}, yb={yb}"
                    );
                }
            }
        }
    }

    #[test]
    fn parity_atlas_1to1_texel_exactness_1024_2048_4096_no_subpixel_bleed() {
        // Build an isolated 5x6 ClearType & Grayscale stripe pattern surrounded by `255` poison guard columns:
        // Inside the 5x6 glyph (`x = 1..6` of a 7x6 upload):
        //   col 0 (poison left), col 1 = [0,0,0], col 2 = [255,0,0], col 3 = [0,0,0],
        //   col 4 = [0,255,0], col 5 = [0,0,0], col 6 (poison right)!
        // ANY `1e-6` or `1/256` horizontal bilinear bleed turns the `[0, 0, 0]` zero columns non-zero!
        let gw = 7u16;
        let gh = 6u16;
        let mut lcd_tex = Vec::with_capacity((gw as usize) * (gh as usize) * 4);
        let mut r8_tex = Vec::with_capacity((gw as usize) * (gh as usize));
        for _y in 0..gh {
            for x in 0..gw {
                let (rgba, gray) = match x {
                    0 | 6 => ([255u8, 255, 255, 255], 255u8), // Poison neighbors right next to inner 5x6 slice!
                    2 => ([255, 0, 0, 255], 255),
                    4 => ([0, 255, 0, 255], 200),
                    _ => ([0, 0, 0, 0], 0), // Zero columns (x = 1, 3, 5)
                };
                lcd_tex.extend_from_slice(&rgba);
                r8_tex.push(gray);
            }
        }

        let mut atlas = GlyphAtlas::new(1024, 1024);
        let mut vk = VulkanRenderer::new_headless(256, 64).expect("Vulkan 1.3 headless");

        for target_dim in [1024u16, 2048, 4096] {
            while atlas.width < target_dim {
                assert!(atlas.try_grow());
            }
            assert_eq!(atlas.width, target_dim);

            // Advance shelf near the far right/bottom of the `target_dim` atlas so `u0 > 0.98` tests worst-case `f32` UV ULP!
            let pad_x = target_dim - 16;
            let pad_y = target_dim - 16;
            let row_stride = target_dim as usize;
            for row in 0..(gh as usize) {
                let r8_dst = (pad_y as usize + row) * row_stride + (pad_x as usize);
                atlas.r8_data[r8_dst..r8_dst + (gw as usize)]
                    .copy_from_slice(&r8_tex[row * (gw as usize)..(row + 1) * (gw as usize)]);
                let rgba_dst = r8_dst * 4;
                atlas.rgba8_data[rgba_dst..rgba_dst + (gw as usize) * 4].copy_from_slice(
                    &lcd_tex[row * (gw as usize) * 4..(row + 1) * (gw as usize) * 4],
                );
            }
            atlas.generation = atlas.generation.wrapping_add(1).max(1);

            let inv_d = 1.0 / (target_dim as f32);
            // Slice ONLY the inner `5x6` `[pad_x + 1 .. pad_x + 6]` right next to the `255` poison texels at `pad_x` and `pad_x + 6`!
            let inner_x = (pad_x + 1) as u32;
            let inner_y = pad_y as u32;
            let uv = [
                (inner_x as f32) * inv_d,
                (inner_y as f32) * inv_d,
                ((inner_x + 5) as f32) * inv_d,
                ((inner_y + 6) as f32) * inv_d,
            ];

            // Odd non-power-of-two fractional-DPI viewport `213 x 97` (`2.0 / 213.0` & `2.0 / 97.0` non-dyadic NDC!):
            let vw = 213u32;
            let vh = 97u32;
            let prims = [
                // 1:1 LCD glyph near right edge `(x = 196, y = 78)`
                GpuPrimitive::atlas_quad(
                    PrimKind::GlyphLcd,
                    [196.0, 78.0, 5.0, 6.0],
                    uv,
                    Color::WHITE,
                    CLIP_NONE,
                    1.0,
                ),
                // 1:1 Grayscale glyph near right edge `(x = 184, y = 78)`
                GpuPrimitive::atlas_quad(
                    PrimKind::GlyphGray,
                    [184.0, 78.0, 5.0, 6.0],
                    uv,
                    Color::WHITE,
                    CLIP_NONE,
                    1.0,
                ),
                // Partially scrolled negative-origin 1:1 LCD glyph `(x = -1, y = -2)`: visible cols `1..5` -> screen `x = 0..4`
                GpuPrimitive::atlas_quad(
                    PrimKind::GlyphLcd,
                    [-1.0, -2.0, 5.0, 6.0],
                    uv,
                    Color::WHITE,
                    CLIP_NONE,
                    1.0,
                ),
                // Vertically smooth-scrolled (`y = 40.35`), horizontally integer-snapped (`x = 60.0`) LCD glyph:
                // Must still have ZERO horizontal subpixel bleed into zero columns `60, 62, 64`!
                GpuPrimitive::atlas_quad(
                    PrimKind::GlyphLcd,
                    [60.0, 40.35, 5.0, 6.0],
                    uv,
                    Color::WHITE,
                    CLIP_NONE,
                    1.0,
                ),
            ];

            let frame = RenderFrame {
                width: vw,
                height: vh,
                scale_factor: 160.0 / 120.0,
                clear_color: Color::BLACK,
                primitives: &prims,
                dirty_prim_ranges: &[],
                clips: &[],
                damage: Some(DamageRect {
                    x: 0,
                    y: 0,
                    w: vw,
                    h: vh,
                }),
            };
            let vk_rgba = vk.render_offscreen_rgba8(&frame, &mut atlas).unwrap();
            let mut cpu = CpuRenderer::new(vw, vh);
            cpu.render(&frame, &mut atlas).unwrap();

            let vk_px = |x: u32, y: u32| -> [u8; 4] {
                let i = ((y * vw + x) as usize) * 4;
                [vk_rgba[i], vk_rgba[i + 1], vk_rgba[i + 2], vk_rgba[i + 3]]
            };

            // Verify zero neighbor bleed on ALL zero columns (`col = 0, 2, 4` -> `x = 196, 198, 200`, `184, 186, 188`, `60, 62, 64`, and `x = 1, 3` for `-1` shifted):
            for &zero_x in &[196u32, 198, 200, 184, 186, 188] {
                assert_eq!(
                    vk_px(zero_x, 80),
                    [0, 0, 0, 255],
                    "Vulkan bilinear neighbor bleed on 1:1 glyph at atlas_dim={target_dim}, x={zero_x}"
                );
                assert_eq!(
                    cpu.pixel_at(zero_x, 80),
                    [0, 0, 0, 255],
                    "CPU neighbor bleed on 1:1 glyph at atlas_dim={target_dim}, x={zero_x}"
                );
            }
            // Active R-only column (`x = 197`) and G-only column (`x = 199`) must stay pure single-channel:
            assert_eq!(vk_px(197, 80), [255, 0, 0, 255]);
            assert_eq!(vk_px(199, 80), [0, 255, 0, 255]);
            // Negative-origin glyph (`x = -1` -> `col 2` at `x = 1` is `[0,0,0]`, `col 1` at `x = 0` is `[255,0,0]`):
            assert_eq!(vk_px(0, 1), [255, 0, 0, 255]);
            assert_eq!(vk_px(1, 1), [0, 0, 0, 255]);
            assert_eq!(cpu.pixel_at(0, 1), [255, 0, 0, 255]);
            assert_eq!(cpu.pixel_at(1, 1), [0, 0, 0, 255]);
            // Vertically smooth-scrolled glyph (`y = 40.35`): zero horizontal bleed at `x = 60, 62, 64`!
            for &zx in &[60u32, 62, 64] {
                assert_eq!(vk_px(zx, 43), [0, 0, 0, 255]);
                assert_eq!(cpu.pixel_at(zx, 43), [0, 0, 0, 255]);
            }

            // Full-viewport strict `<= 2 LSB` GPU vs CPU parity at `1024`, `2048`, and `4096`:
            for y in 0..vh {
                for x in 0..vw {
                    let vp = vk_px(x, y);
                    let cp = cpu.pixel_at(x, y);
                    for c in 0..4 {
                        assert!(
                            (vp[c] as i32 - cp[c] as i32).abs() <= 2,
                            "Atlas {target_dim} parity exceeded 2 LSB at ({x}, {y}): vk={vp:?} vs cpu={cp:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn parity_all_fractional_scales_composite_scene_strict_3_lsb() {
        let mut atlas = GlyphAtlas::new(128, 128);
        let mut lcd_glyph = Vec::with_capacity(6 * 9 * 4);
        let mut gray_glyph = Vec::with_capacity(6 * 9);
        let mut icon_rgba = Vec::with_capacity(8 * 8 * 4);
        for y in 0..9u8 {
            for x in 0..6u8 {
                let on_stem = x == 1 || x == 4 || y == 4;
                if on_stem {
                    lcd_glyph.extend_from_slice(&[240, 255, 220, 255]);
                    gray_glyph.push(245);
                } else {
                    lcd_glyph.extend_from_slice(&[75, 130, 195, 255]);
                    gray_glyph.push(110);
                }
            }
        }
        for y in 0..8u8 {
            for x in 0..8u8 {
                icon_rgba.extend_from_slice(&[30 + x * 24, 180 - y * 14, 235, 220]);
            }
        }
        let lcd_reg = atlas.upload_rgba8(6, 9, 0, 7, &lcd_glyph).unwrap();
        let gray_reg = atlas.upload_r8(6, 9, 0, 7, &gray_glyph).unwrap();
        let icon_reg = atlas.upload_rgba8(8, 8, 0, 8, &icon_rgba).unwrap();
        let lcd_uv = [lcd_reg.u0, lcd_reg.v0, lcd_reg.u1, lcd_reg.v1];
        let gray_uv = [gray_reg.u0, gray_reg.v0, gray_reg.u1, gray_reg.v1];
        let icon_uv = [icon_reg.u0, icon_reg.v0, icon_reg.u1, icon_reg.v1];

        let snap_rect = |lx: f32, ly: f32, lw: f32, lh: f32, sf: f32| -> [f32; 4] {
            let x0 = (lx * sf).round();
            let y0 = (ly * sf).round();
            let x1 = ((lx + lw) * sf).round();
            let y1 = ((ly + lh) * sf).round();
            [x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0)]
        };

        for &sf in &[1.125_f32, 1.25, 160.0 / 120.0, 1.5, 200.0 / 120.0, 1.75, 2.25] {
            atlas.generation = atlas.generation.wrapping_add(1).max(1);
            let vw = (132.0 * sf).round() as u32;
            let vh = (98.0 * sf).round() as u32;

            let card_b = snap_rect(8.0, 8.0, 116.0, 82.0, sf);
            let card_r = [8.0 * sf, 4.0 * sf, 12.0 * sf, 6.0 * sf];
            let clips = [GpuClip {
                bounds: card_b,
                radii: card_r,
                meta: [CLIP_NONE, 0, 0, 0],
            }];

            let prims = [
                // 1. Fractional-sigma Gaussian shadow under card:
                GpuPrimitive::shadow(
                    card_b,
                    card_r,
                    Color::rgba8(0, 0, 0, 165),
                    3.5 * sf,
                    [0.0, 1.5 * sf],
                    CLIP_NONE,
                    1.0,
                ),
                // 2. Card surface with snapped `1.0px` logical border:
                GpuPrimitive::quad(
                    card_b,
                    card_r,
                    Color::rgb8(28, 33, 46),
                    snap_physical_border_width(1.0, sf),
                    Color::rgb8(76, 88, 118),
                    CLIP_NONE,
                    1.0,
                ),
                // 3. Clipped OKLab gradient banner with raw fractional border (`1.0 * sf`, e.g. `1.25px`, `1.3333px`):
                GpuPrimitive::linear_gradient(
                    snap_rect(12.0, 12.0, 108.0, 22.0, sf),
                    [5.0 * sf; 4],
                    Color::rgb8(48, 118, 246),
                    Color::rgba8(140, 70, 240, 90),
                    0.35,
                    1.0 * sf,
                    0,
                    1.0,
                ),
                // 4. Focused input box with snapped `2.0px` logical focus border:
                GpuPrimitive::quad(
                    snap_rect(12.0, 38.0, 108.0, 20.0, sf),
                    [4.0 * sf; 4],
                    Color::rgb8(20, 24, 34),
                    snap_physical_border_width(2.0, sf),
                    Color::rgb8(96, 165, 250),
                    0,
                    1.0,
                ),
                // 5. Sub-pixel `0.5px` hairline row inside card:
                GpuPrimitive::quad(
                    snap_rect(12.0, 62.0, 108.0, 22.0, sf),
                    [0.0; 4],
                    Color::rgb8(34, 40, 56),
                    snap_physical_border_width(0.5, sf),
                    Color::rgb8(86, 98, 128),
                    0,
                    1.0,
                ),
                // 6. Snapped 1:1 ClearType LCD glyph, Grayscale glyph, and RGBA8 icon inside rounded clip:
                GpuPrimitive::atlas_quad(
                    PrimKind::GlyphLcd,
                    [(18.0 * sf).floor(), (42.0 * sf).round(), 6.0, 9.0],
                    lcd_uv,
                    Color::rgb8(242, 246, 252),
                    0,
                    1.0,
                ),
                GpuPrimitive::atlas_quad(
                    PrimKind::GlyphGray,
                    [(30.0 * sf).floor(), (42.0 * sf).round(), 6.0, 9.0],
                    gray_uv,
                    Color::rgb8(184, 194, 214),
                    0,
                    1.0,
                ),
                GpuPrimitive::atlas_quad(
                    PrimKind::ImageRgba,
                    [(44.0 * sf).floor(), (42.0 * sf).round(), 8.0, 8.0],
                    icon_uv,
                    Color::WHITE,
                    0,
                    0.92,
                ),
            ];

            let _ = assert_gpu_cpu_parity_at_scale(
                vw,
                vh,
                sf,
                Color::rgb8(16, 19, 27),
                &prims,
                &clips,
                &mut atlas,
                3,
            );
        }
    }
}
