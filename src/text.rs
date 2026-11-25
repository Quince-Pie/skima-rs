//! OpenType text shaping (`rustybuzz` + `swash`), zero-C system font discovery, zero-padding
//! per-glyph multi-font fallback (`Latin` + `Symbols` + `Color Emoji`), 5-tap ClearType FIR
//! subpixel/grayscale shelf-packed `GlyphAtlas` with bounds & saturation safety, and Exact
//! `[W_tight, W_next_unwrap)` Width-Interval Line-Break Cache with explicit `\n` support.

use core::range::Range as CopyRange;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use swash::FontRef;
use swash::scale::image::{Content, Image};
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::zeno::{Format, Transform, Vector};

/// Handle to a loaded OpenType font face (or `FontId::FALLBACK` built-in crisp bitmap font).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontId(pub u16);

impl FontId {
    /// Built-in crisp procedural fallback font (always available even with zero `.ttf` files loaded).
    pub const FALLBACK: Self = Self(u16::MAX);
}

/// Glyph rasterization antialiasing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AntialiasMode {
    Grayscale,
    SubpixelLcd,
}

/// Texture plane inside `GlyphAtlas`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtlasPlane {
    R8,
    Rgba8,
}

/// UV coordinates and pixel metrics of a cached glyph or custom icon/image inside `GlyphAtlas`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtlasRegion {
    pub plane: Option<AtlasPlane>,
    /// True when `Rgba8` holds a full-color glyph/image (`PrimKind::ImageRgba`, untinted `Color::WHITE`),
    /// false when `Rgba8` holds a per-channel ClearType subpixel coverage mask (`PrimKind::GlyphLcd`).
    pub is_color: bool,
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
    pub bearing_x: i16,
    pub bearing_y: i16,
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
}

const _: () = assert!(core::mem::size_of::<AtlasRegion>() == 32);

/// Incremental dirty upload rectangle recorded whenever new glyphs/icons enter `GlyphAtlas`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasDirtyRect {
    pub plane: AtlasPlane,
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

/// Deterministic `GlyphAtlas` cache key using FreeType/HarfBuzz `F26Dot6` (`26.6` fixed-point = `1/64`
/// physical pixel = `0.015625 px`) font-size quantization so fractional-DPI `1 ULP` float differences
/// (e.g. `14.0 * (150.0 / 120.0)` vs `17.5`) hit the exact same cached glyph deterministically!
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Physical font size quantized to `F26Dot6` (`(phys_font_size * 64.0).round() as u32`).
    pub size_f26dot6: u32,
    pub font_id: FontId,
    pub glyph_id: u16,
    /// Horizontal subpixel phase bin `0..4` (`0.00`, `0.25`, `0.50`, `0.75` physical px).
    pub subpx_bin: u8,
    pub aa_mode: AntialiasMode,
}

pub type GlyphRasterKey = GlyphKey;

const _: () = assert!(core::mem::size_of::<GlyphKey>() == 12);

impl GlyphKey {
    /// Maximum clamped physical font size (`4096.0 px`, matching [`GlyphAtlas::MAX_DIMENSION`]).
    pub const MAX_PHYSICAL_FONT_SIZE: f32 = GlyphAtlas::MAX_DIMENSION as f32;

    /// Quantize a physical-pixel font size to `F26Dot6` (`1/64` physical px = `(phys_font_size * 64.0).round() as u32`).
    /// Returns `0` for non-finite or non-positive sizes.
    #[inline]
    pub fn quantize_f26dot6(physical_font_size: f32) -> u32 {
        if !physical_font_size.is_finite() || physical_font_size <= 0.0 {
            return 0;
        }
        let clamped = physical_font_size.clamp(1.0, Self::MAX_PHYSICAL_FONT_SIZE);
        (clamped * 64.0).round() as u32
    }

    /// Convert a quantized `F26Dot6` size back to an exact binary `f32` (`1/64 == 2^-6` has zero IEEE-754 mantissa error!).
    #[inline]
    pub const fn f26dot6_to_px(size_f26dot6: u32) -> f32 {
        (size_f26dot6 as f32) * (1.0 / 64.0)
    }

    /// Quantize a physical X coordinate's fractional phase into one of 4 horizontal subpixel bins (`0..4`).
    #[inline]
    pub fn quantize_subpx_bin(physical_subpx_x: f32) -> u8 {
        if physical_subpx_x.is_finite() {
            (((physical_subpx_x.fract() + 1.0).fract()) * 4.0).floor() as u8 & 3
        } else {
            0
        }
    }

    /// Construct the canonical `GlyphKey` for `(font_id, glyph_id, physical_font_size, physical_subpx_x, aa_mode)`.
    #[inline]
    pub fn new(
        font_id: FontId,
        glyph_id: u16,
        physical_font_size: f32,
        physical_subpx_x: f32,
        aa_mode: AntialiasMode,
    ) -> Self {
        Self {
            size_f26dot6: Self::quantize_f26dot6(physical_font_size),
            font_id,
            glyph_id,
            subpx_bin: if font_id == FontId::FALLBACK {
                0
            } else {
                Self::quantize_subpx_bin(physical_subpx_x)
            },
            aa_mode,
        }
    }
}

struct ShelfPacker {
    width: u16,
    height: u16,
    cursor_x: u16,
    cursor_y: u16,
    row_h: u16,
}

impl ShelfPacker {
    fn new(width: u16, height: u16) -> Self {
        let reserve = width.min(height).min(4);
        Self {
            width,
            height,
            cursor_x: reserve,
            cursor_y: 0,
            row_h: reserve,
        }
    }

    fn reset(&mut self) {
        let reserve = self.width.min(self.height).min(4);
        self.cursor_x = reserve;
        self.cursor_y = 0;
        self.row_h = reserve;
    }

    /// True if `(w, h)` can fit into a freshly reset `ShelfPacker` (prevents pointless saturation flushes on oversized regions).
    #[inline]
    fn can_ever_fit(&self, w: u16, h: u16) -> bool {
        if w == 0 || h == 0 {
            return true;
        }
        if w > self.width || h > self.height {
            return false;
        }
        let reserve = self.width.min(self.height).min(4) as usize;
        let fits_first_row =
            reserve + (w as usize) <= (self.width as usize) && (h as usize) <= (self.height as usize);
        let fits_next_row =
            (w as usize) <= (self.width as usize) && reserve + (h as usize) <= (self.height as usize);
        fits_first_row || fits_next_row
    }

    fn alloc(&mut self, w: u16, h: u16) -> Option<(u16, u16)> {
        if w == 0 || h == 0 {
            return Some((0, 0));
        }
        if w > self.width || h > self.height {
            return None;
        }
        // Require `cursor_x + w <= width` (edge-clamped at right border, 1px gutter between neighbors).
        // Compute candidate shelf coordinates without mutating packer state until vertical fit is confirmed!
        let (nx, ny, n_row_h) = if (self.cursor_x as usize) + (w as usize) > (self.width as usize) {
            (0u16, self.cursor_y.saturating_add(self.row_h), 0u16)
        } else {
            (self.cursor_x, self.cursor_y, self.row_h)
        };
        if (ny as usize) + (h as usize) > (self.height as usize) {
            return None;
        }
        let padded_w = w.saturating_add(1);
        let padded_h = h.saturating_add(1);
        self.cursor_x = nx.saturating_add(padded_w).min(self.width);
        self.cursor_y = ny;
        self.row_h = n_row_h.max(padded_h);
        Some((nx, ny))
    }
}

/// Dual-plane (`R8` grayscale + `RGBA8` subpixel/color) shelf-packed texture atlas with incremental
/// dirty-rect tracking, strict bounds validation, dynamic texel-preserving `1024 -> 2048 -> 4096` growth,
/// and fallback saturation eviction at maximum dimension.
pub struct GlyphAtlas {
    pub width: u16,
    pub height: u16,
    pub r8_data: Vec<u8>,
    pub rgba8_data: Vec<u8>,
    r8_packer: ShelfPacker,
    rgba_packer: ShelfPacker,
    cache: HashMap<GlyphRasterKey, AtlasRegion>,
    pub dirty_rects: Vec<AtlasDirtyRect>,
    pub generation: u64,
}

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self::new(1024, 1024)
    }
}

impl GlyphAtlas {
    /// Maximum automatic atlas dimension (`4096 x 4096`, guaranteed by Vulkan 1.0+ `maxImageDimension2D`).
    pub const MAX_DIMENSION: u16 = 4096;

    pub fn new(width: u16, height: u16) -> Self {
        let len = (width as usize) * (height as usize);
        let mut r8_data = vec![0u8; len];
        let mut rgba8_data = vec![0u8; len * 4];
        let ww = width.min(2) as usize;
        let wh = height.min(2) as usize;
        for y in 0..wh {
            for x in 0..ww {
                let idx = y * (width as usize) + x;
                r8_data[idx] = 255;
                rgba8_data[idx * 4..idx * 4 + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let mut dirty_rects = Vec::new();
        if ww > 0 && wh > 0 {
            dirty_rects.push(AtlasDirtyRect {
                plane: AtlasPlane::R8,
                x: 0,
                y: 0,
                w: ww as u16,
                h: wh as u16,
            });
            dirty_rects.push(AtlasDirtyRect {
                plane: AtlasPlane::Rgba8,
                x: 0,
                y: 0,
                w: ww as u16,
                h: wh as u16,
            });
        }
        Self {
            width,
            height,
            r8_data,
            rgba8_data,
            r8_packer: ShelfPacker::new(width, height),
            rgba_packer: ShelfPacker::new(width, height),
            cache: HashMap::with_capacity(512),
            dirty_rects,
            generation: 1,
        }
    }

    #[inline]
    pub fn cached_glyph_count(&self) -> usize {
        self.cache.len()
    }

    /// Look up a pre-rasterized [`AtlasRegion`] directly by its [`GlyphKey`].
    #[inline]
    pub fn get_cached(&self, key: &GlyphKey) -> Option<AtlasRegion> {
        self.cache.get(key).copied()
    }

    /// Returns `true` if `key` is already resident in `GlyphAtlas`.
    #[inline]
    pub fn contains_glyph_key(&self, key: &GlyphKey) -> bool {
        self.cache.contains_key(key)
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// True if `(w, h)` fits either the current atlas dimensions or (for `>= 1024x1024` elastic atlases)
    /// [`Self::MAX_DIMENSION`] after [`Self::try_grow`] expansion, without mutating small fixed-bounds test atlases.
    #[inline]
    fn can_fit_after_growth(&self, is_rgba: bool, w: u16, h: u16) -> bool {
        let packer = if is_rgba {
            &self.rgba_packer
        } else {
            &self.r8_packer
        };
        if packer.can_ever_fit(w, h) {
            return true;
        }
        if self.width >= 1024 && self.height >= 1024 {
            let max_w = self.width.max(Self::MAX_DIMENSION);
            let max_h = self.height.max(Self::MAX_DIMENSION);
            return ShelfPacker::new(max_w, max_h).can_ever_fit(w, h);
        }
        false
    }

    /// Dynamically double `GlyphAtlas` dimensions up to [`Self::MAX_DIMENSION`] (`1024 -> 2048 -> 4096`),
    /// preserving all existing packed `R8`/`RGBA8` texels and shelf cursors while rescaling cached `AtlasRegion` UVs!
    /// Returns `false` if already at or above `MAX_DIMENSION` (or `0x0`).
    pub fn try_grow(&mut self) -> bool {
        if self.width == 0
            || self.height == 0
            || (self.width >= Self::MAX_DIMENSION && self.height >= Self::MAX_DIMENSION)
        {
            return false;
        }
        let new_w = self
            .width
            .saturating_mul(2)
            .min(Self::MAX_DIMENSION)
            .max(self.width);
        let new_h = self
            .height
            .saturating_mul(2)
            .min(Self::MAX_DIMENSION)
            .max(self.height);
        if new_w == self.width && new_h == self.height {
            return false;
        }

        let old_w = self.width as usize;
        let old_h = self.height as usize;
        let nw = new_w as usize;
        let nh = new_h as usize;

        let mut new_r8 = vec![0u8; nw * nh];
        let mut new_rgba = vec![0u8; nw * nh * 4];

        if self.r8_data.len() >= old_w * old_h {
            for y in 0..old_h {
                new_r8[y * nw..y * nw + old_w]
                    .copy_from_slice(&self.r8_data[y * old_w..(y + 1) * old_w]);
            }
        }
        if self.rgba8_data.len() >= old_w * old_h * 4 {
            let old_stride = old_w * 4;
            let new_stride = nw * 4;
            for y in 0..old_h {
                new_rgba[y * new_stride..y * new_stride + old_stride]
                    .copy_from_slice(&self.rgba8_data[y * old_stride..(y + 1) * old_stride]);
            }
        }

        self.width = new_w;
        self.height = new_h;
        self.r8_data = new_r8;
        self.rgba8_data = new_rgba;
        self.r8_packer.width = new_w;
        self.r8_packer.height = new_h;
        self.rgba_packer.width = new_w;
        self.rgba_packer.height = new_h;

        let inv_w = 1.0 / (new_w as f32);
        let inv_h = 1.0 / (new_h as f32);
        for reg in self.cache.values_mut() {
            if reg.plane.is_some() && reg.w > 0 && reg.h > 0 {
                reg.u0 = (reg.x as f32) * inv_w;
                reg.v0 = (reg.y as f32) * inv_h;
                reg.u1 = ((reg.x as u32 + reg.w as u32) as f32) * inv_w;
                reg.v1 = ((reg.y as u32 + reg.h as u32) as f32) * inv_h;
            }
        }

        self.generation = self.generation.wrapping_add(1).max(1);
        self.dirty_rects.clear();
        self.dirty_rects.push(AtlasDirtyRect {
            plane: AtlasPlane::R8,
            x: 0,
            y: 0,
            w: new_w,
            h: new_h,
        });
        self.dirty_rects.push(AtlasDirtyRect {
            plane: AtlasPlane::Rgba8,
            x: 0,
            y: 0,
            w: new_w,
            h: new_h,
        });
        true
    }

    pub fn clear_on_saturation(&mut self) {
        self.r8_data.fill(0);
        self.rgba8_data.fill(0);
        let ww = self.width.min(2) as usize;
        let wh = self.height.min(2) as usize;
        for y in 0..wh {
            for x in 0..ww {
                let idx = y * (self.width as usize) + x;
                if idx < self.r8_data.len() {
                    self.r8_data[idx] = 255;
                }
                if (idx + 1) * 4 <= self.rgba8_data.len() {
                    self.rgba8_data[idx * 4..idx * 4 + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
        self.r8_packer.reset();
        self.rgba_packer.reset();
        self.cache.clear();
        self.generation = self.generation.wrapping_add(1).max(1);
        self.dirty_rects.clear();
        if self.width > 0 && self.height > 0 {
            self.dirty_rects.push(AtlasDirtyRect {
                plane: AtlasPlane::R8,
                x: 0,
                y: 0,
                w: self.width,
                h: self.height,
            });
            self.dirty_rects.push(AtlasDirtyRect {
                plane: AtlasPlane::Rgba8,
                x: 0,
                y: 0,
                w: self.width,
                h: self.height,
            });
        }
    }

    /// Upload custom `R8` mask (e.g., rasterized vector icon or procedural glyph) into the atlas.
    /// Gracefully returns `None` without panicking or evicting cached glyphs if `(w, h)` exceeds atlas bounds
    /// or `pixels` is shorter than `w * h`.
    pub fn upload_r8(
        &mut self,
        w: u16,
        h: u16,
        bearing_x: i16,
        bearing_y: i16,
        pixels: &[u8],
    ) -> Option<AtlasRegion> {
        if w == 0 || h == 0 {
            return Some(AtlasRegion::default());
        }
        let needed_bytes = (w as usize).checked_mul(h as usize)?;
        if pixels.len() < needed_bytes || !self.can_fit_after_growth(false, w, h) {
            return None;
        }
        let total_atlas = (self.width as usize).checked_mul(self.height as usize)?;
        if self.r8_data.len() < total_atlas {
            return None;
        }

        let mut full_dirty = false;
        let (ax, ay) = match self.r8_packer.alloc(w, h) {
            Some(pos) => pos,
            None => loop {
                full_dirty = true;
                if !self.try_grow() {
                    self.clear_on_saturation();
                    break self.r8_packer.alloc(w, h)?;
                }
                if let Some(pos) = self.r8_packer.alloc(w, h) {
                    break pos;
                }
            },
        };
        let stride = self.width as usize;
        let row_w = w as usize;
        for row in 0..(h as usize) {
            let dst_start = (ay as usize + row) * stride + (ax as usize);
            let src_start = row * row_w;
            self.r8_data[dst_start..dst_start + row_w]
                .copy_from_slice(&pixels[src_start..src_start + row_w]);
        }
        if !full_dirty {
            self.dirty_rects.push(AtlasDirtyRect {
                plane: AtlasPlane::R8,
                x: ax,
                y: ay,
                w,
                h,
            });
        }
        let inv_w = 1.0 / (self.width as f32);
        let inv_h = 1.0 / (self.height as f32);
        Some(AtlasRegion {
            plane: Some(AtlasPlane::R8),
            is_color: false,
            x: ax,
            y: ay,
            w,
            h,
            bearing_x,
            bearing_y,
            u0: (ax as f32) * inv_w,
            v0: (ay as f32) * inv_h,
            u1: ((ax as u32 + w as u32) as f32) * inv_w,
            v1: ((ay as u32 + h as u32) as f32) * inv_h,
        })
    }

    /// Overwrite an existing `R8` `AtlasRegion` in place (0 B new shelf usage, 0 atlas evictions!).
    /// Safely ignores out-of-bounds regions or undersized `pixels` buffers without panicking.
    pub fn update_r8_in_place(&mut self, reg: AtlasRegion, pixels: &[u8]) {
        if reg.w == 0 || reg.h == 0 {
            return;
        }
        if reg.plane.is_some_and(|p| p != AtlasPlane::R8) {
            return;
        }
        let x_end = (reg.x as usize).saturating_add(reg.w as usize);
        let y_end = (reg.y as usize).saturating_add(reg.h as usize);
        let stride = self.width as usize;
        let total_atlas = stride.saturating_mul(self.height as usize);
        let Some(needed_src) = (reg.w as usize).checked_mul(reg.h as usize) else {
            return;
        };
        if x_end > stride
            || y_end > (self.height as usize)
            || pixels.len() < needed_src
            || self.r8_data.len() < total_atlas
        {
            return;
        }
        let w = reg.w as usize;
        for row in 0..(reg.h as usize) {
            let dst_start = (reg.y as usize + row) * stride + (reg.x as usize);
            let src_start = row * w;
            self.r8_data[dst_start..dst_start + w]
                .copy_from_slice(&pixels[src_start..src_start + w]);
        }
        self.dirty_rects.push(AtlasDirtyRect {
            plane: AtlasPlane::R8,
            x: reg.x,
            y: reg.y,
            w: reg.w,
            h: reg.h,
        });
    }

    /// Upload `RGBA8` image / subpixel LCD mask / color emoji into the atlas.
    /// Gracefully returns `None` without panicking or evicting cached glyphs if `(w, h)` exceeds atlas bounds
    /// or `pixels` is shorter than `w * h * 4`.
    pub fn upload_rgba8(
        &mut self,
        w: u16,
        h: u16,
        bearing_x: i16,
        bearing_y: i16,
        pixels: &[u8],
    ) -> Option<AtlasRegion> {
        if w == 0 || h == 0 {
            return Some(AtlasRegion::default());
        }
        let needed_bytes = (w as usize)
            .checked_mul(h as usize)
            .and_then(|px| px.checked_mul(4))?;
        if pixels.len() < needed_bytes || !self.can_fit_after_growth(true, w, h) {
            return None;
        }
        let total_atlas = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|px| px.checked_mul(4))?;
        if self.rgba8_data.len() < total_atlas {
            return None;
        }

        let mut full_dirty = false;
        let (ax, ay) = match self.rgba_packer.alloc(w, h) {
            Some(pos) => pos,
            None => loop {
                full_dirty = true;
                if !self.try_grow() {
                    self.clear_on_saturation();
                    break self.rgba_packer.alloc(w, h)?;
                }
                if let Some(pos) = self.rgba_packer.alloc(w, h) {
                    break pos;
                }
            },
        };
        let stride = (self.width as usize) * 4;
        let row_bytes = (w as usize) * 4;
        for row in 0..(h as usize) {
            let dst_start = (ay as usize + row) * stride + (ax as usize) * 4;
            let src_start = row * row_bytes;
            self.rgba8_data[dst_start..dst_start + row_bytes]
                .copy_from_slice(&pixels[src_start..src_start + row_bytes]);
        }
        if !full_dirty {
            self.dirty_rects.push(AtlasDirtyRect {
                plane: AtlasPlane::Rgba8,
                x: ax,
                y: ay,
                w,
                h,
            });
        }
        let inv_w = 1.0 / (self.width as f32);
        let inv_h = 1.0 / (self.height as f32);
        Some(AtlasRegion {
            plane: Some(AtlasPlane::Rgba8),
            is_color: true,
            x: ax,
            y: ay,
            w,
            h,
            bearing_x,
            bearing_y,
            u0: (ax as f32) * inv_w,
            v0: (ay as f32) * inv_h,
            u1: ((ax as u32 + w as u32) as f32) * inv_w,
            v1: ((ay as u32 + h as u32) as f32) * inv_h,
        })
    }

    /// Overwrite an existing `RGBA8` `AtlasRegion` in place (0 B new shelf usage, 0 atlas evictions!).
    /// Safely ignores out-of-bounds regions or undersized `pixels` buffers without panicking.
    pub fn update_rgba8_in_place(&mut self, reg: AtlasRegion, pixels: &[u8]) {
        if reg.w == 0 || reg.h == 0 {
            return;
        }
        if reg.plane.is_some_and(|p| p != AtlasPlane::Rgba8) {
            return;
        }
        let x_end = (reg.x as usize).saturating_add(reg.w as usize);
        let y_end = (reg.y as usize).saturating_add(reg.h as usize);
        let width_px = self.width as usize;
        let stride = width_px.saturating_mul(4);
        let total_atlas = stride.saturating_mul(self.height as usize);
        let Some(needed_src) = (reg.w as usize)
            .checked_mul(reg.h as usize)
            .and_then(|px| px.checked_mul(4))
        else {
            return;
        };
        if x_end > width_px
            || y_end > (self.height as usize)
            || pixels.len() < needed_src
            || self.rgba8_data.len() < total_atlas
        {
            return;
        }
        let row_bytes = (reg.w as usize) * 4;
        for row in 0..(reg.h as usize) {
            let dst_start = (reg.y as usize + row) * stride + (reg.x as usize) * 4;
            let src_start = row * row_bytes;
            self.rgba8_data[dst_start..dst_start + row_bytes]
                .copy_from_slice(&pixels[src_start..src_start + row_bytes]);
        }
        self.dirty_rects.push(AtlasDirtyRect {
            plane: AtlasPlane::Rgba8,
            x: reg.x,
            y: reg.y,
            w: reg.w,
            h: reg.h,
        });
    }

    #[inline]
    pub fn drain_dirty_rects(&mut self, out: &mut Vec<AtlasDirtyRect>) {
        out.append(&mut self.dirty_rects);
    }
}

/// Apply normalized 5-tap ClearType FIR subpixel filter `[1, 2, 3, 2, 1] / 9` across a 3x-oversampled
/// horizontal alpha scanline `sub_row` (`sub_w` samples starting at global subpixel `sub_left`),
/// writing balanced `[R, G, B, A]` pixels for physical columns `p_min .. p_min + out_w`.
///
/// Energy-balance invariant: taps at `0 mod 3`, `1 mod 3`, and `2 mod 3` each sum to `3/9 == 1/3`,
/// guaranteeing unity DC response (`255 -> [255, 255, 255]`) and zero integrated RGB color fringing!
#[inline]
pub fn filter_cleartype_5tap_row(
    sub_row: &[u8],
    sub_left: i32,
    p_min: i32,
    out_w: usize,
    out_rgba_row: &mut [u8],
) {
    let sub_w = sub_row.len() as i32;
    let sample = |k: i32| -> u32 {
        let rel = k - sub_left;
        if rel >= 0 && rel < sub_w {
            sub_row[rel as usize] as u32
        } else {
            0
        }
    };

    for px in 0..out_w {
        let p = p_min + (px as i32);
        let c0 = 3 * p; // Red center (`p + 1/6`), Green (`3*p + 1`), Blue (`3*p + 2`)
        let s_m2 = sample(c0 - 2);
        let s_m1 = sample(c0 - 1);
        let s_0 = sample(c0);
        let s_p1 = sample(c0 + 1);
        let s_p2 = sample(c0 + 2);
        let s_p3 = sample(c0 + 3);
        let s_p4 = sample(c0 + 4);

        let r = (s_m2 + 2 * s_m1 + 3 * s_0 + 2 * s_p1 + s_p2 + 4) / 9;
        let g = (s_m1 + 2 * s_0 + 3 * s_p1 + 2 * s_p2 + s_p3 + 4) / 9;
        let b = (s_0 + 2 * s_p1 + 3 * s_p2 + 2 * s_p3 + s_p4 + 4) / 9;
        let a = (r + g + b + 1) / 3;

        let dst = px * 4;
        out_rgba_row[dst] = r as u8;
        out_rgba_row[dst + 1] = g as u8;
        out_rgba_row[dst + 2] = b as u8;
        out_rgba_row[dst + 3] = a as u8;
    }
}

#[derive(Clone, Copy)]
enum WordSegment<'a> {
    Control(usize, char),
    Subrun(FontId, usize, &'a str),
}

struct FontEntry {
    data: Arc<[u8]>,
    face_index: u32,
    has_color_tables: bool,
}

#[inline]
fn sfnt_has_color_glyphs(data: &[u8], face_index: u32) -> bool {
    let table_dir_offset = if data.get(0..4) == Some(b"ttcf") {
        let Some(nf_bytes) = data.get(8..12) else {
            return false;
        };
        let num_fonts = u32::from_be_bytes([nf_bytes[0], nf_bytes[1], nf_bytes[2], nf_bytes[3]]);
        if face_index >= num_fonts {
            return false;
        }
        let off_pos = 12usize.saturating_add((face_index as usize).saturating_mul(4));
        let Some(fb) = data.get(off_pos..off_pos + 4) else {
            return false;
        };
        u32::from_be_bytes([fb[0], fb[1], fb[2], fb[3]]) as usize
    } else {
        if face_index != 0 {
            return false;
        }
        0usize
    };
    let Some(nt_bytes) = data.get(table_dir_offset + 4..table_dir_offset + 6) else {
        return false;
    };
    let num_tables = u16::from_be_bytes([nt_bytes[0], nt_bytes[1]]) as usize;
    let records_base = table_dir_offset.saturating_add(12);
    for i in 0..num_tables {
        let rec_off = records_base.saturating_add(i.saturating_mul(16));
        let Some(tag) = data.get(rec_off..rec_off + 4) else {
            break;
        };
        if matches!(tag, b"COLR" | b"CBDT" | b"sbix" | b"SVG ") {
            return true;
        }
    }
    false
}

/// Complete OpenType Font, System Discovery, Multi-Font Fallback, & Shaping System (`rustybuzz` + `swash`).
pub struct FontSystem {
    fonts: Vec<FontEntry>,
    scale_cx: ScaleContext,
    raw_glyph_img: Image,
    lcd_filter_buf: Vec<u8>,
    pub atlas: GlyphAtlas,
    pub ui_font: FontId,
    pub mono_font: FontId,
    pub fallback_chain: Vec<FontId>,
}

impl Default for FontSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl FontSystem {
    pub fn new() -> Self {
        Self {
            fonts: Vec::new(),
            scale_cx: ScaleContext::new(),
            raw_glyph_img: Image::new(),
            lcd_filter_buf: Vec::new(),
            atlas: GlyphAtlas::default(),
            ui_font: FontId::FALLBACK,
            mono_font: FontId::FALLBACK,
            fallback_chain: Vec::new(),
        }
    }

    /// Create a `FontSystem` pre-populated with best available OS Sans-Serif, Monospace, and Emoji fonts.
    pub fn with_system_fonts() -> Self {
        let mut fs = Self::new();
        let _ = fs.load_system_fonts();
        fs
    }

    pub fn load_font_bytes(&mut self, bytes: impl Into<Arc<[u8]>>, face_index: u32) -> Option<FontId> {
        if self.fonts.len() >= (u16::MAX as usize) {
            return None;
        }
        let data: Arc<[u8]> = bytes.into();
        let _rb = rustybuzz::Face::from_slice(&data, face_index)?;
        let _sw = FontRef::from_index(&data, face_index as usize)?;
        let has_color_tables = sfnt_has_color_glyphs(&data, face_index);
        let id = FontId(self.fonts.len() as u16);
        self.fonts.push(FontEntry {
            data,
            face_index,
            has_color_tables,
        });
        if self.ui_font == FontId::FALLBACK {
            self.ui_font = id;
        }
        if self.mono_font == FontId::FALLBACK {
            self.mono_font = id;
        }
        if !self.fallback_chain.contains(&id) {
            self.fallback_chain.push(id);
        }
        Some(id)
    }

    #[inline]
    pub fn font_count(&self) -> usize {
        self.fonts.len()
    }

    /// Zero-C system font discovery across Linux/NixOS (`/etc/fonts`), macOS, and Windows.
    /// Loads only the top-ranked Sans-Serif UI font, Monospace code font, and Unicode/Emoji fallback
    /// (typically 2-4 files in `< 2 ms`, avoiding scanning or reading hundreds of unused font weights).
    pub fn load_system_fonts(&mut self) -> usize {
        let roots = collect_system_font_roots();
        let mut best_sans: Option<(i32, PathBuf)> = None;
        let mut best_mono: Option<(i32, PathBuf)> = None;
        let mut best_emoji: Option<(i32, PathBuf)> = None;
        let mut best_symbols: Option<(i32, PathBuf)> = None;

        for root in &roots {
            scan_font_dir_shallow(root, 0, &mut |path| {
                let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
                    return;
                };
                let lower = fname.to_ascii_lowercase();
                if !(lower.ends_with(".ttf") || lower.ends_with(".otf") || lower.ends_with(".ttc")) {
                    return;
                }
                let sans_score = match lower.as_str() {
                    "inter-regular.ttf" | "sfns.ttf" | "segoeui.ttf" => 100,
                    "notosans-regular.ttf" => 95,
                    "dejavusans.ttf" => 90,
                    "liberationsans-regular.ttf" => 85,
                    "freesans.ttf" | "arial.ttf" => 75,
                    _ => 0,
                };
                if sans_score > best_sans.as_ref().map_or(0, |(s, _)| *s) {
                    best_sans = Some((sans_score, path.to_path_buf()));
                }

                let mono_score = match lower.as_str() {
                    "jetbrainsmono-regular.ttf" | "intelonemono-regular.ttf" => 100,
                    "firacode-regular.ttf" | "cascadiamono.ttf" => 95,
                    "dejavusansmono.ttf" => 90,
                    "liberationmono-regular.ttf" | "menlo.ttc" | "consola.ttf" => 85,
                    _ if lower.contains("blexmono") && lower.contains("regular") => 88,
                    _ => 0,
                };
                if mono_score > best_mono.as_ref().map_or(0, |(s, _)| *s) {
                    best_mono = Some((mono_score, path.to_path_buf()));
                }

                let emoji_score = match lower.as_str() {
                    "notocoloremoji.ttf" => 100,
                    "apple color emoji.ttc" | "seguiemj.ttf" => 95,
                    _ if lower.contains("twemoji") || lower.contains("coloremoji") => 85,
                    _ => 0,
                };
                if emoji_score > best_emoji.as_ref().map_or(0, |(s, _)| *s) {
                    best_emoji = Some((emoji_score, path.to_path_buf()));
                }

                let sym_score = match lower.as_str() {
                    "notosanssymbols2-regular.otf" | "notosanssymbols2-regular.ttf" => 100,
                    "seguisym.ttf" | "apple symbols.ttf" => 95,
                    "notosanssymbols-regular.ttf" | "notosanssymbols.ttf" => 85,
                    _ => 0,
                };
                if sym_score > best_symbols.as_ref().map_or(0, |(s, _)| *s) {
                    best_symbols = Some((sym_score, path.to_path_buf()));
                }
            });
        }

        let mut loaded = 0usize;
        if let Some((_, p)) = best_sans
            && let Ok(bytes) = std::fs::read(&p)
            && let Some(id) = self.load_font_bytes(bytes, 0)
        {
            self.ui_font = id;
            loaded += 1;
        }
        if let Some((_, p)) = best_mono
            && let Ok(bytes) = std::fs::read(&p)
            && let Some(id) = self.load_font_bytes(bytes, 0)
        {
            self.mono_font = id;
            loaded += 1;
        }
        if let Some((_, p)) = best_emoji
            && let Ok(bytes) = std::fs::read(&p)
            && self.load_font_bytes(bytes, 0).is_some()
        {
            loaded += 1;
        }
        if let Some((_, p)) = best_symbols
            && let Ok(bytes) = std::fs::read(&p)
            && self.load_font_bytes(bytes, 0).is_some()
        {
            loaded += 1;
        }
        loaded
    }

    #[inline]
    fn font_covers_char(&self, fid: FontId, ch: char) -> bool {
        fid != FontId::FALLBACK
            && self
                .fonts
                .get(fid.0 as usize)
                .and_then(|ent| rustybuzz::Face::from_slice(&ent.data, ent.face_index))
                .is_some_and(|face| face.glyph_index(ch).is_some())
    }

    fn select_font_for_char(&self, preferred: FontId, ch: char, prefer_color_emoji: bool) -> FontId {
        if prefer_color_emoji {
            if preferred != FontId::FALLBACK
                && let Some(ent) = self.fonts.get(preferred.0 as usize)
                && ent.has_color_tables
                && let Some(face) = rustybuzz::Face::from_slice(&ent.data, ent.face_index)
                && face.glyph_index(ch).is_some()
            {
                return preferred;
            }
            for &fb in &self.fallback_chain {
                if let Some(ent) = self.fonts.get(fb.0 as usize)
                    && ent.has_color_tables
                    && let Some(face) = rustybuzz::Face::from_slice(&ent.data, ent.face_index)
                    && face.glyph_index(ch).is_some()
                {
                    return fb;
                }
            }
        }
        if self.font_covers_char(preferred, ch) {
            return preferred;
        }
        for &fb in &self.fallback_chain {
            if fb == preferred {
                continue;
            }
            if self.font_covers_char(fb, ch) {
                return fb;
            }
        }
        FontId::FALLBACK
    }

    #[inline]
    fn space_advance_for(&self, face: Option<&rustybuzz::Face<'_>>, font_size: f32) -> (f32, rustybuzz::UnicodeBuffer) {
        let buf = rustybuzz::UnicodeBuffer::new();
        let Some(face) = face else {
            return ((font_size * 0.56).max(1.0), buf);
        };
        let upem = (face.units_per_em() as f32).max(1.0);
        let scale = font_size / upem;
        let mut sb = buf;
        sb.push_str(" ");
        let out = rustybuzz::shape(face, &[], sb);
        let adv = out
            .glyph_positions()
            .iter()
            .map(|p| (p.x_advance as f32) * scale)
            .sum::<f32>()
            .max(font_size * 0.25);
        (adv, out.clear())
    }

    /// Shape a string into cluster-preserved `ShapedParagraph` using OpenType (`rustybuzz`) with
    /// automatic zero-padding per-glyph fallback across `self.fallback_chain`, explicit `\n` hard
    /// line breaks, preserved column/input spacing, and Exact Width-Interval Line-Break Memoization!
    pub fn shape_paragraph(
        &self,
        text: &str,
        font_id: FontId,
        font_size: f32,
        line_height: f32,
    ) -> ShapedParagraph {
        let safe_fs = if font_size.is_finite() { font_size.max(1.0) } else { 14.0 };
        let safe_lh = if line_height.is_finite() { line_height.max(0.0) } else { safe_fs };
        let content_hash = hash_text_style(text, font_id, font_size, line_height);
        if font_id == FontId::FALLBACK || (font_id.0 as usize) >= self.fonts.len() {
            return ShapedParagraph::shape_fallback(text, safe_fs, safe_lh, content_hash);
        }

        let entry = &self.fonts[font_id.0 as usize];
        let face = rustybuzz::Face::from_slice(&entry.data, entry.face_index)
            .expect("validated font face");
        let upem = (face.units_per_em() as f32).max(1.0);
        let scale = safe_fs / upem;
        let ascent = ((face.ascender() as f32) * scale).clamp(safe_fs * 0.72, safe_lh.max(safe_fs) * 0.92);
        let (space_advance, rb_buf_init) = self.space_advance_for(Some(&face), safe_fs);
        let mut rb_buf = Some(rb_buf_init);

        let mut glyphs = Vec::with_capacity(text.len());
        let mut words = Vec::new();
        let mut clusters = Vec::with_capacity(text.len() + 1);
        let has_newlines = text.contains('\n');
        let mut hard_lines = Vec::new();
        let mut hard_line_clusters = Vec::new();
        let mut max_unwrapped_w = 0.0_f32;
        let mut min_word_w = 0.0_f32;

        if !text.is_empty() {
            let mut line_base_byte = 0usize;
            for raw_line in text.split('\n') {
                let line_str = raw_line.strip_suffix('\r').unwrap_or(raw_line);
                let hl_start = words.len() as u32;
                let cl_start = clusters.len() as u32;
                let line_unwrapped_w = shape_hard_line_words(
                    line_str,
                    line_base_byte,
                    space_advance,
                    &mut glyphs,
                    &mut words,
                    &mut clusters,
                    &mut min_word_w,
                    |word_str, word_base_byte, out_glyphs, out_clusters| {
                        let all_primary = word_str
                            .chars()
                            .all(|c| !is_ignorable_control_char(c) && face.glyph_index(c).is_some())
                            && !(!entry.has_color_tables
                                && (word_str.contains('\u{FE0F}') || word_str.contains('\u{200D}')));
                        if all_primary {
                            let (w, next_b) = shape_run_into(
                                &face,
                                font_id,
                                safe_fs,
                                word_str,
                                word_base_byte,
                                rb_buf.take(),
                                out_glyphs,
                                out_clusters,
                            );
                            rb_buf = Some(next_b);
                            w
                        } else {
                            self.shape_mixed_word(
                                font_id,
                                safe_fs,
                                word_str,
                                word_base_byte,
                                &mut rb_buf,
                                out_glyphs,
                                out_clusters,
                            )
                        }
                    },
                );
                let cl_end = (clusters.len() - 1) as u32;
                max_unwrapped_w = max_unwrapped_w.max(line_unwrapped_w);
                if has_newlines {
                    hard_lines.push(CopyRange {
                        start: hl_start,
                        end: words.len() as u32,
                    });
                    hard_line_clusters.push(CopyRange {
                        start: cl_start,
                        end: cl_end,
                    });
                }
                line_base_byte += raw_line.len() + 1;
            }
        }

        ShapedParagraph {
            content_hash,
            font_id,
            font_size: safe_fs,
            line_height: safe_lh,
            ascent,
            space_w: space_advance,
            max_unwrapped_w,
            min_word_w,
            glyphs,
            words,
            clusters,
            hard_lines,
            hard_line_clusters,
            valid_lo_w: f32::INFINITY,
            valid_hi_w: -1.0,
            cached_wrapped_h: safe_lh,
            cached_tight_w: max_unwrapped_w,
            cached_lines: Vec::new(),
        }
    }

    fn for_each_word_subrun<'a>(
        &self,
        preferred: FontId,
        word_str: &'a str,
        mut on_seg: impl FnMut(WordSegment<'a>) -> bool,
    ) -> bool {
        let mut run_start = 0usize;
        let mut cur_font: Option<FontId> = None;
        let mut after_zwj = false;
        let mut iter = word_str.char_indices().peekable();

        while let Some((rel_b, ch)) = iter.next() {
            if is_ignorable_control_char(ch) {
                if let Some(prev_f) = cur_font.take()
                    && run_start < rel_b
                    && !on_seg(WordSegment::Subrun(prev_f, run_start, &word_str[run_start..rel_b]))
                {
                    return false;
                }
                if !on_seg(WordSegment::Control(rel_b, ch)) {
                    return false;
                }
                run_start = rel_b + ch.len_utf8();
                after_zwj = false;
                continue;
            }

            let next_ch = iter.peek().map(|&(_, nc)| nc);
            let chosen = if ch == '\u{200D}' {
                after_zwj = true;
                cur_font.unwrap_or_else(|| self.select_font_for_char(preferred, ch, true))
            } else if is_cluster_continuation(ch) {
                cur_font.unwrap_or_else(|| self.select_font_for_char(preferred, ch, false))
            } else if after_zwj {
                after_zwj = false;
                if let Some(cf) = cur_font
                    && self.font_covers_char(cf, ch)
                {
                    cf
                } else {
                    self.select_font_for_char(preferred, ch, true)
                }
            } else {
                let prefer_color = matches!(
                    next_ch,
                    Some('\u{FE0F}' | '\u{200D}' | '\u{1F3FB}'..='\u{1F3FF}' | '\u{20E3}')
                );
                self.select_font_for_char(preferred, ch, prefer_color)
            };

            if let Some(prev_f) = cur_font
                && prev_f != chosen
            {
                if run_start < rel_b
                    && !on_seg(WordSegment::Subrun(prev_f, run_start, &word_str[run_start..rel_b]))
                {
                    return false;
                }
                run_start = rel_b;
            }
            cur_font = Some(chosen);
        }

        if let Some(last_f) = cur_font
            && run_start < word_str.len()
        {
            return on_seg(WordSegment::Subrun(last_f, run_start, &word_str[run_start..]));
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn shape_mixed_word(
        &self,
        preferred: FontId,
        font_size: f32,
        word_str: &str,
        word_base_byte: usize,
        rb_buf: &mut Option<rustybuzz::UnicodeBuffer>,
        out_glyphs: &mut Vec<ShapedGlyph>,
        out_clusters: &mut Vec<ShapedCluster>,
    ) -> f32 {
        let mut word_w = 0.0_f32;
        self.for_each_word_subrun(preferred, word_str, |seg| match seg {
            WordSegment::Control(rel_b, _ch) => {
                out_clusters.push(ShapedCluster {
                    byte_start: (word_base_byte + rel_b) as u32,
                    advance: 0.0,
                });
                true
            }
            WordSegment::Subrun(fid, run_start, sub) => {
                word_w += self.shape_subrun(
                    fid,
                    font_size,
                    sub,
                    word_base_byte + run_start,
                    rb_buf,
                    out_glyphs,
                    out_clusters,
                );
                true
            }
        });
        word_w
    }

    #[allow(clippy::too_many_arguments)]
    fn shape_subrun(
        &self,
        fid: FontId,
        font_size: f32,
        slice: &str,
        base_byte: usize,
        rb_buf: &mut Option<rustybuzz::UnicodeBuffer>,
        out_glyphs: &mut Vec<ShapedGlyph>,
        out_clusters: &mut Vec<ShapedCluster>,
    ) -> f32 {
        if fid == FontId::FALLBACK || (fid.0 as usize) >= self.fonts.len() {
            let char_adv = (font_size * 0.56).max(1.0);
            return shape_fallback_slice(slice, base_byte, char_adv, out_glyphs, out_clusters);
        }
        let ent = &self.fonts[fid.0 as usize];
        let face = rustybuzz::Face::from_slice(&ent.data, ent.face_index).unwrap();
        let (w, next_b) = shape_run_into(
            &face,
            fid,
            font_size,
            slice,
            base_byte,
            rb_buf.take(),
            out_glyphs,
            out_clusters,
        );
        *rb_buf = Some(next_b);
        w
    }

    /// Walk UTF-8 character cells `(char_byte_start, char_byte_end, cell_advance)` across single-line `line_text`
    /// using exact OpenType `rustybuzz` word/subrun shaping (preserving kerning, ligatures, combining marks,
    /// ZWJ/VS16 emojis, multi-font fallback, control-char zero width, and space/tab advances identical to `shape_paragraph`).
    fn walk_line_char_advances(
        &self,
        line_text: &str,
        font_id: FontId,
        font_size: f32,
        mut visit: impl FnMut(usize, usize, f32) -> bool,
    ) {
        if line_text.is_empty() {
            return;
        }
        let safe_fs = if font_size.is_finite() { font_size.max(1.0) } else { 14.0 };
        if font_id == FontId::FALLBACK || (font_id.0 as usize) >= self.fonts.len() {
            let char_adv = (safe_fs * 0.56).max(1.0);
            let _ = visit_fallback_char_cells(line_text, 0, char_adv, &mut visit);
            return;
        }

        let entry = &self.fonts[font_id.0 as usize];
        let face = rustybuzz::Face::from_slice(&entry.data, entry.face_index)
            .expect("validated font face");
        let (space_adv, rb_buf_init) = self.space_advance_for(Some(&face), safe_fs);
        let mut rb_buf = Some(rb_buf_init);

        let bytes = line_text.as_bytes();
        let len = bytes.len();
        let mut pos = 0usize;

        while pos < len {
            let b = bytes[pos];
            if b == b' ' {
                if !visit(pos, pos + 1, space_adv) {
                    return;
                }
                pos += 1;
                continue;
            }
            if b == b'\t' {
                if !visit(pos, pos + 1, space_adv * 4.0) {
                    return;
                }
                pos += 1;
                continue;
            }
            if b == b'\r' || b == b'\n' {
                if !visit(pos, pos + 1, 0.0) {
                    return;
                }
                pos += 1;
                continue;
            }

            let word_start = pos;
            while pos < len && !matches!(bytes[pos], b' ' | b'\t' | b'\r' | b'\n') {
                pos += 1;
            }
            let word_str = &line_text[word_start..pos];

            let all_primary = word_str
                .chars()
                .all(|c| !is_ignorable_control_char(c) && face.glyph_index(c).is_some())
                && !(!entry.has_color_tables
                    && (word_str.contains('\u{FE0F}') || word_str.contains('\u{200D}')));
            if all_primary {
                let (cont, next_b) = visit_shaped_subrun(
                    &face,
                    safe_fs,
                    word_str,
                    word_start,
                    rb_buf.take().unwrap_or_default(),
                    &mut visit,
                );
                rb_buf = Some(next_b);
                if !cont {
                    return;
                }
            } else {
                let keep_going = self.for_each_word_subrun(font_id, word_str, |seg| match seg {
                    WordSegment::Control(rel_b, ch) => {
                        let b0 = word_start + rel_b;
                        visit(b0, b0 + ch.len_utf8(), 0.0)
                    }
                    WordSegment::Subrun(fid, run_start, sub) => {
                        let (cont, next_b) = self.visit_fallback_or_opentype_subrun(
                            fid,
                            safe_fs,
                            sub,
                            word_start + run_start,
                            rb_buf.take().unwrap_or_default(),
                            &mut visit,
                        );
                        rb_buf = Some(next_b);
                        cont
                    }
                });
                if !keep_going {
                    return;
                }
            }
        }
    }

    fn visit_fallback_or_opentype_subrun(
        &self,
        fid: FontId,
        font_size: f32,
        sub: &str,
        base_byte: usize,
        rb_buf: rustybuzz::UnicodeBuffer,
        visit: &mut impl FnMut(usize, usize, f32) -> bool,
    ) -> (bool, rustybuzz::UnicodeBuffer) {
        if fid == FontId::FALLBACK || (fid.0 as usize) >= self.fonts.len() {
            let char_adv = (font_size * 0.56).max(1.0);
            let ok = visit_fallback_char_cells(sub, base_byte, char_adv, visit);
            return (ok, rb_buf);
        }
        let ent = &self.fonts[fid.0 as usize];
        let face = rustybuzz::Face::from_slice(&ent.data, ent.face_index).unwrap();
        visit_shaped_subrun(&face, font_size, sub, base_byte, rb_buf, visit)
    }

    /// Measure exact horizontal caret `x` offset (in logical pixels) at `byte_offset` in a single-line string.
    /// Preserves OpenType kerning, ligatures (proportional sub-character split), multi-font fallback, and UTF-8 clusters.
    pub fn byte_offset_to_x(
        &self,
        text: &str,
        byte_offset: usize,
        font_id: FontId,
        font_size: f32,
    ) -> f32 {
        let clamped = text.floor_char_boundary(byte_offset.min(text.len()));
        if clamped == 0 {
            return 0.0;
        }
        // Support measuring within the active `\n` line if multiline text is passed:
        let line_start = text[..clamped].rfind('\n').map_or(0, |i| i + 1);
        let line_end = text[clamped..]
            .find('\n')
            .map_or(text.len(), |rel| clamped + rel);
        let line_str = text[line_start..line_end]
            .strip_suffix('\r')
            .unwrap_or(&text[line_start..line_end]);
        let target_rel = (clamped - line_start).min(line_str.len());
        if target_rel == 0 {
            return 0.0;
        }

        let mut pen_x = 0.0_f32;
        self.walk_line_char_advances(line_str, font_id, font_size, |b0, _b1, adv| {
            if b0 >= target_rel {
                return false;
            }
            pen_x += adv;
            true
        });
        pen_x
    }

    /// Hit-test a local logical `x` coordinate inside single-line `text` to the nearest UTF-8 char boundary,
    /// splitting each shaped glyph / ligature character cell at its exact `0.5 * advance` midpoint!
    pub fn x_to_byte_offset(&self, text: &str, local_x: f32, font_id: FontId, font_size: f32) -> usize {
        if !local_x.is_finite() || local_x <= 0.0 || text.is_empty() {
            return 0;
        }
        let line_end = text.find('\n').unwrap_or(text.len());
        let line_str = text[..line_end].strip_suffix('\r').unwrap_or(&text[..line_end]);
        if line_str.is_empty() {
            return 0;
        }

        let mut pen = 0.0_f32;
        let mut hit: Option<usize> = None;
        self.walk_line_char_advances(line_str, font_id, font_size, |b0, _b1, adv| {
            if adv > 0.0 && local_x < pen + adv * 0.5 {
                hit = Some(b0);
                return false;
            }
            pen += adv;
            true
        });
        hit.unwrap_or(line_str.len())
    }

    /// Rasterize (or fetch from `GlyphAtlas` cache) a shaped glyph with `F26Dot6` (`1/64` px) size quantization
    /// and `1/4`-pixel horizontal subpixel positioning!
    pub fn rasterize_glyph(
        &mut self,
        font_id: FontId,
        glyph_id: u16,
        physical_font_size: f32,
        physical_subpx_x: f32,
        aa_mode: AntialiasMode,
    ) -> AtlasRegion {
        let key = GlyphKey::new(
            font_id,
            glyph_id,
            physical_font_size,
            physical_subpx_x,
            aa_mode,
        );
        if key.size_f26dot6 == 0 {
            return AtlasRegion::default();
        }

        if let Some(&region) = self.atlas.cache.get(&key) {
            return region;
        }

        // Feed the exact IEEE-754 `2^-6`-quantized `F26Dot6` size into `swash` so `skrifa::Size::fixed_linear_scale`
        // (`(ppem * 64.) as i32`) and `swash`'s `HintingCache` never suffer `1 ULP` truncation or order-dependent drift!
        let quantized_px = GlyphKey::f26dot6_to_px(key.size_f26dot6);
        let (region, cacheable) =
            if font_id == FontId::FALLBACK || (font_id.0 as usize) >= self.fonts.len() {
                self.rasterize_fallback_ascii(glyph_id, quantized_px)
            } else {
                self.rasterize_swash_glyph(font_id, glyph_id, quantized_px, key.subpx_bin, aa_mode)
            };

        if cacheable {
            self.atlas.cache.insert(key, region);
        }
        region
    }

    fn rasterize_swash_glyph(
        &mut self,
        font_id: FontId,
        glyph_id: u16,
        font_size: f32,
        subpx_bin: u8,
        aa_mode: AntialiasMode,
    ) -> (AtlasRegion, bool) {
        let entry = &self.fonts[font_id.0 as usize];
        let Some(font_ref) = FontRef::from_index(&entry.data, entry.face_index as usize) else {
            return (AtlasRegion::default(), false);
        };
        let mut scaler = self
            .scale_cx
            .builder(font_ref)
            .size(font_size)
            .hint(true)
            .build();

        if aa_mode == AntialiasMode::SubpixelLcd {
            // 1. Check color outlines / color bitmaps (e.g., NotoColorEmoji) at 1x native scale first.
            //    Note: never probe monochrome `Source::Bitmap` before `Source::Outline`, which would hijack
            //    TrueType faces containing legacy `EBDT` strikes and resample bitmaps at fractional DPI!
            self.raw_glyph_img.clear();
            let color_sources: &[Source] = if scaler.has_outlines() {
                &[
                    Source::ColorOutline(0),
                    Source::ColorBitmap(StrikeWith::BestFit),
                ]
            } else {
                &[
                    Source::ColorOutline(0),
                    Source::ColorBitmap(StrikeWith::BestFit),
                    Source::Bitmap(StrikeWith::BestFit),
                ]
            };
            if Render::new(color_sources)
                .format(Format::Alpha)
                .render_into(&mut scaler, glyph_id, &mut self.raw_glyph_img)
            {
                return self.upload_raw_swash_image();
            }

            // 2. Hint outline at 1x `font_size`, rasterize 3x horizontal subpixel coverage in 1 pass,
            //    and convolve with normalized 5-tap ClearType FIR filter `[1, 2, 3, 2, 1] / 9`:
            self.raw_glyph_img.clear();
            let subpx_offset = Vector::new((subpx_bin as f32) * 0.75, 0.0);
            if !Render::new(&[Source::Outline])
                .format(Format::Alpha)
                .transform(Some(Transform::scale(3.0, 1.0)))
                .offset(subpx_offset)
                .render_into(&mut scaler, glyph_id, &mut self.raw_glyph_img)
            {
                return (AtlasRegion::default(), true);
            }

            let sub_w = self.raw_glyph_img.placement.width;
            let h_u32 = self.raw_glyph_img.placement.height;
            if sub_w == 0 || h_u32 == 0 {
                return (AtlasRegion::default(), true);
            }
            let Ok(h) = u16::try_from(h_u32) else {
                return (AtlasRegion::default(), false);
            };
            let sub_left = self.raw_glyph_img.placement.left;
            let p_min = (sub_left - 2).div_euclid(3);
            let p_max = (sub_left + (sub_w as i32) + 1).div_euclid(3);
            let out_w_i32 = p_max - p_min + 1;
            if out_w_i32 <= 0 {
                return (AtlasRegion::default(), true);
            }
            let Ok(out_w) = u16::try_from(out_w_i32 as u32) else {
                return (AtlasRegion::default(), false);
            };
            if !self.atlas.can_fit_after_growth(true, out_w, h) {
                return (AtlasRegion::default(), false);
            }

            let bx = p_min.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            let by = self
                .raw_glyph_img
                .placement
                .top
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;

            let out_w_usize = out_w as usize;
            let sub_w_usize = sub_w as usize;
            let total_rgba = out_w_usize * (h as usize) * 4;
            self.lcd_filter_buf.clear();
            self.lcd_filter_buf.resize(total_rgba, 0);

            for row in 0..(h as usize) {
                let src_row = &self.raw_glyph_img.data[row * sub_w_usize..(row + 1) * sub_w_usize];
                let dst_row =
                    &mut self.lcd_filter_buf[row * out_w_usize * 4..(row + 1) * out_w_usize * 4];
                filter_cleartype_5tap_row(src_row, sub_left, p_min, out_w_usize, dst_row);
            }

            return match self
                .atlas
                .upload_rgba8(out_w, h, bx, by, &self.lcd_filter_buf)
            {
                Some(mut reg) => {
                    reg.is_color = false;
                    (reg, true)
                }
                None => (AtlasRegion::default(), false),
            };
        }

        // Grayscale path (`Format::Alpha`): prioritize scalable `Source::Outline` over monochrome `Bitmap` strikes!
        self.raw_glyph_img.clear();
        let offset = Vector::new((subpx_bin as f32) * 0.25, 0.0);
        if !Render::new(&[
            Source::ColorOutline(0),
            Source::ColorBitmap(StrikeWith::BestFit),
            Source::Outline,
            Source::Bitmap(StrikeWith::BestFit),
        ])
        .format(Format::Alpha)
        .offset(offset)
        .render_into(&mut scaler, glyph_id, &mut self.raw_glyph_img)
        {
            return (AtlasRegion::default(), true);
        }
        self.upload_raw_swash_image()
    }

    fn upload_raw_swash_image(&mut self) -> (AtlasRegion, bool) {
        if self.raw_glyph_img.placement.width == 0 || self.raw_glyph_img.placement.height == 0 {
            return (AtlasRegion::default(), true);
        }
        let Ok(w) = u16::try_from(self.raw_glyph_img.placement.width) else {
            return (AtlasRegion::default(), false);
        };
        let Ok(h) = u16::try_from(self.raw_glyph_img.placement.height) else {
            return (AtlasRegion::default(), false);
        };
        let bx = self
            .raw_glyph_img
            .placement
            .left
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let by = self
            .raw_glyph_img
            .placement
            .top
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;

        match self.raw_glyph_img.content {
            Content::Mask => match self.atlas.upload_r8(w, h, bx, by, &self.raw_glyph_img.data) {
                Some(reg) => (reg, true),
                None => (AtlasRegion::default(), false),
            },
            Content::SubpixelMask => match self
                .atlas
                .upload_rgba8(w, h, bx, by, &self.raw_glyph_img.data)
            {
                Some(mut reg) => {
                    reg.is_color = false;
                    (reg, true)
                }
                None => (AtlasRegion::default(), false),
            },
            Content::Color => match self
                .atlas
                .upload_rgba8(w, h, bx, by, &self.raw_glyph_img.data)
            {
                Some(reg) => (reg, true),
                None => (AtlasRegion::default(), false),
            },
        }
    }

    fn rasterize_fallback_ascii(&mut self, ascii_code: u16, font_size: f32) -> (AtlasRegion, bool) {
        let ch = (ascii_code.min(127) as u8) as char;
        if ch.is_whitespace() || ch.is_control() {
            return (AtlasRegion::default(), true);
        }
        let scale = (font_size / 10.0).ceil().max(1.0) as usize;
        let Ok(gw) = u16::try_from(5usize.saturating_mul(scale)) else {
            return (AtlasRegion::default(), false);
        };
        let Ok(gh) = u16::try_from(7usize.saturating_mul(scale)) else {
            return (AtlasRegion::default(), false);
        };
        if !self.atlas.can_fit_after_growth(false, gw, gh) {
            return (AtlasRegion::default(), false);
        }
        let pattern = mini_glyph_5x7(ch);
        let total = (gw as usize) * (gh as usize);
        self.lcd_filter_buf.clear();
        self.lcd_filter_buf.resize(total, 0);
        for (row, &bits) in pattern.iter().enumerate() {
            for col in 0..5usize {
                if (bits >> (4 - col)) & 1 != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let py = row * scale + sy;
                            let px = col * scale + sx;
                            self.lcd_filter_buf[py * (gw as usize) + px] = 255;
                        }
                    }
                }
            }
        }
        match self.atlas.upload_r8(
            gw,
            gh,
            0,
            (gh as i32).min(i16::MAX as i32) as i16,
            &self.lcd_filter_buf,
        ) {
            Some(reg) => (reg, true),
            None => (AtlasRegion::default(), false),
        }
    }
}

#[inline]
fn is_ignorable_control_char(ch: char) -> bool {
    (ch <= '\x1F' && ch != '\t' && ch != '\n') || ('\u{007F}'..='\u{009F}').contains(&ch)
}

#[inline]
fn is_zero_width_format_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FE0E}' | '\u{FE0F}' | '\u{FEFF}'
    ) || ('\u{E0100}'..='\u{E01EF}').contains(&ch)
        || ('\u{E0020}'..='\u{E007F}').contains(&ch)
}

#[inline]
fn is_combining_or_emoji_modifier(ch: char) -> bool {
    matches!(
        ch,
        '\u{0300}'..='\u{036F}'
            | '\u{1AB0}'..='\u{1AFF}'
            | '\u{1DC0}'..='\u{1DFF}'
            | '\u{20D0}'..='\u{20FF}'
            | '\u{FE20}'..='\u{FE2F}'
            | '\u{1F3FB}'..='\u{1F3FF}'
    )
}

#[inline]
fn is_cluster_continuation(ch: char) -> bool {
    is_zero_width_format_char(ch) || is_combining_or_emoji_modifier(ch)
}

#[inline]
fn visit_cluster_char_cells(
    cluster_slice: &str,
    c_start: usize,
    cluster_adv: f32,
    visit: &mut impl FnMut(usize, usize, f32) -> bool,
) -> bool {
    if cluster_slice.len() <= 1 {
        return visit(c_start, c_start + cluster_slice.len(), cluster_adv);
    }
    let mut chars = cluster_slice.chars();
    if chars.next().is_none() {
        return visit(c_start, c_start, cluster_adv);
    }
    if chars.next().is_none() {
        // Single multi-byte UTF-8 codepoint (e.g. `é`, `🦀`, `⚡`, `✅`):
        return visit(c_start, c_start + cluster_slice.len(), cluster_adv);
    }

    // Atomic multi-codepoint emoji sequence (`👨‍👩‍👧‍👦` ZWJ, `⚡\u{FE0F}` VS16, skin tone, keycap `\u{20E3}`, flags):
    // Keep full `cluster_adv` on the leading codepoint and `0.0` on joined codepoints so hit-testing snaps
    // cleanly to the outer cluster UTF-8 boundaries (`c_start` or `c_end`) without splitting ZWJ sequences!
    let is_atomic_emoji = cluster_slice.chars().any(|c| {
        matches!(
            c,
            '\u{200D}' | '\u{FE0E}' | '\u{FE0F}' | '\u{20E3}' | '\u{1F3FB}'..='\u{1F3FF}'
        ) || ('\u{E0020}'..='\u{E007F}').contains(&c)
    });
    if is_atomic_emoji {
        let mut first = true;
        for (rel_b, ch) in cluster_slice.char_indices() {
            let adv = if first {
                first = false;
                cluster_adv
            } else {
                0.0
            };
            let b0 = c_start + rel_b;
            if !visit(b0, b0 + ch.len_utf8(), adv) {
                return false;
            }
        }
        return true;
    }

    // Latin ligature (`fi`, `ffi`, `fl`) or base + combining mark (`e\u{0301}`):
    // Subdivide `cluster_adv` across spacing base codepoints; non-spacing combining marks get `0.0`.
    let base_count = cluster_slice
        .chars()
        .filter(|&c| !is_cluster_continuation(c) && !is_ignorable_control_char(c))
        .count();
    if base_count == 0 {
        let char_count = cluster_slice.chars().count().max(1);
        let sub_adv = cluster_adv / (char_count as f32);
        for (rel_b, ch) in cluster_slice.char_indices() {
            let b0 = c_start + rel_b;
            if !visit(b0, b0 + ch.len_utf8(), sub_adv) {
                return false;
            }
        }
        return true;
    }
    let sub_adv = cluster_adv / (base_count as f32);
    for (rel_b, ch) in cluster_slice.char_indices() {
        let adv = if !is_cluster_continuation(ch) && !is_ignorable_control_char(ch) {
            sub_adv
        } else {
            0.0
        };
        let b0 = c_start + rel_b;
        if !visit(b0, b0 + ch.len_utf8(), adv) {
            return false;
        }
    }
    true
}

fn shape_fallback_slice(
    slice: &str,
    base_byte: usize,
    char_adv: f32,
    out_glyphs: &mut Vec<ShapedGlyph>,
    out_clusters: &mut Vec<ShapedCluster>,
) -> f32 {
    let mut w = 0.0_f32;
    let mut after_zwj = false;
    for (rel_b, ch) in slice.char_indices() {
        let b_start = (base_byte + rel_b) as u32;
        if is_ignorable_control_char(ch) {
            after_zwj = false;
            out_clusters.push(ShapedCluster {
                byte_start: b_start,
                advance: 0.0,
            });
            continue;
        }
        if ch == '\u{200D}' {
            after_zwj = true;
            out_clusters.push(ShapedCluster {
                byte_start: b_start,
                advance: 0.0,
            });
            continue;
        }
        if is_cluster_continuation(ch) {
            out_clusters.push(ShapedCluster {
                byte_start: b_start,
                advance: 0.0,
            });
            continue;
        }
        if after_zwj {
            after_zwj = false;
            out_clusters.push(ShapedCluster {
                byte_start: b_start,
                advance: 0.0,
            });
            continue;
        }
        out_glyphs.push(ShapedGlyph {
            glyph_id: (ch as u32).min(127) as u16,
            font_id: FontId::FALLBACK,
            x_advance: char_adv,
            x_offset: 0.0,
            y_offset: 0.0,
        });
        out_clusters.push(ShapedCluster {
            byte_start: b_start,
            advance: char_adv,
        });
        w += char_adv;
    }
    w
}

fn visit_fallback_char_cells(
    slice: &str,
    base_byte: usize,
    char_adv: f32,
    visit: &mut impl FnMut(usize, usize, f32) -> bool,
) -> bool {
    let mut after_zwj = false;
    for (rel_b, ch) in slice.char_indices() {
        let adv = if ch == ' ' {
            after_zwj = false;
            char_adv
        } else if ch == '\t' {
            after_zwj = false;
            char_adv * 4.0
        } else if ch == '\n' || is_ignorable_control_char(ch) {
            after_zwj = false;
            0.0
        } else if ch == '\u{200D}' {
            after_zwj = true;
            0.0
        } else if is_cluster_continuation(ch) {
            0.0
        } else if after_zwj {
            after_zwj = false;
            0.0
        } else {
            char_adv
        };
        let b0 = base_byte + rel_b;
        if !visit(b0, b0 + ch.len_utf8(), adv) {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn shape_run_into(
    face: &rustybuzz::Face<'_>,
    font_id: FontId,
    font_size: f32,
    run_str: &str,
    base_byte: usize,
    buf_opt: Option<rustybuzz::UnicodeBuffer>,
    glyphs: &mut Vec<ShapedGlyph>,
    clusters: &mut Vec<ShapedCluster>,
) -> (f32, rustybuzz::UnicodeBuffer) {
    let upem = (face.units_per_em() as f32).max(1.0);
    let scale = font_size / upem;
    let mut buf = buf_opt.unwrap_or_default();
    buf.push_str(run_str);
    let shaped = rustybuzz::shape(face, &[], buf);
    let infos = shaped.glyph_infos();
    let positions = shaped.glyph_positions();

    if infos.is_empty() {
        if !run_str.is_empty() {
            clusters.push(ShapedCluster {
                byte_start: base_byte as u32,
                advance: 0.0,
            });
        }
        return (0.0, shaped.clear());
    }

    let mut is_ltr = true;
    for w in infos.windows(2) {
        if w[1].cluster < w[0].cluster {
            is_ltr = false;
            break;
        }
    }

    // Group adjacent glyphs sharing a cluster so non-spacing combining marks (`x_advance == 0`) keep
    // exact 0 advance while standalone bitmap emoji clusters (`CBDT` reporting 0 advance) clamp to `font_size`:
    let sub_cl_start = clusters.len();
    let mut run_w = 0.0_f32;
    let mut i = 0usize;
    while i < infos.len() {
        let cl = infos[i].cluster;
        let mut j = i + 1;
        while j < infos.len() && infos[j].cluster == cl {
            j += 1;
        }
        let mut cluster_adv = 0.0_f32;
        for pos in &positions[i..j] {
            cluster_adv += (pos.x_advance as f32) * scale;
        }
        let cl_byte = (cl as usize).min(run_str.len());
        let first_ch = run_str[cl_byte..].chars().next();
        let needs_emoji_fallback_adv = cluster_adv <= 0.01
            && first_ch.is_some_and(|c| {
                !c.is_whitespace()
                    && !is_cluster_continuation(c)
                    && !is_ignorable_control_char(c)
            });

        let mut final_cluster_adv = 0.0_f32;
        for (k, (info, pos)) in infos[i..j].iter().zip(positions[i..j].iter()).enumerate() {
            let mut adv = (pos.x_advance as f32) * scale;
            if k == 0 && needs_emoji_fallback_adv {
                adv = font_size;
            }
            glyphs.push(ShapedGlyph {
                glyph_id: info.glyph_id as u16,
                font_id,
                x_advance: adv,
                x_offset: (pos.x_offset as f32) * scale,
                y_offset: (pos.y_offset as f32) * scale,
            });
            run_w += adv;
            final_cluster_adv += adv;
        }
        clusters.push(ShapedCluster {
            byte_start: (base_byte + cl_byte) as u32,
            advance: final_cluster_adv,
        });
        i = j;
    }

    if !is_ltr && clusters.len() > sub_cl_start + 1 {
        clusters[sub_cl_start..].sort_unstable_by_key(|c| c.byte_start);
        let mut write = sub_cl_start;
        for read in (sub_cl_start + 1)..clusters.len() {
            if clusters[read].byte_start == clusters[write].byte_start {
                clusters[write].advance += clusters[read].advance;
            } else {
                write += 1;
                clusters[write] = clusters[read];
            }
        }
        clusters.truncate(write + 1);
    }

    if clusters[sub_cl_start].byte_start > base_byte as u32 {
        clusters.insert(
            sub_cl_start,
            ShapedCluster {
                byte_start: base_byte as u32,
                advance: 0.0,
            },
        );
    }

    (run_w, shaped.clear())
}

fn visit_shaped_subrun(
    face: &rustybuzz::Face<'_>,
    font_size: f32,
    run_str: &str,
    base_byte: usize,
    mut buf: rustybuzz::UnicodeBuffer,
    visit: &mut impl FnMut(usize, usize, f32) -> bool,
) -> (bool, rustybuzz::UnicodeBuffer) {
    let upem = (face.units_per_em() as f32).max(1.0);
    let scale = font_size / upem;
    buf.push_str(run_str);
    let shaped = rustybuzz::shape(face, &[], buf);
    let infos = shaped.glyph_infos();
    let positions = shaped.glyph_positions();

    if infos.is_empty() {
        if !run_str.is_empty() && !visit_cluster_char_cells(run_str, base_byte, 0.0, visit) {
            return (false, shaped.clear());
        }
        return (true, shaped.clear());
    }

    let mut is_ltr = true;
    for w in infos.windows(2) {
        if w[1].cluster < w[0].cluster {
            is_ltr = false;
            break;
        }
    }

    if is_ltr {
        let first_cl = (infos[0].cluster as usize).min(run_str.len());
        if first_cl > 0 && !visit_cluster_char_cells(&run_str[..first_cl], base_byte, 0.0, visit) {
            return (false, shaped.clear());
        }
    }

    let mut i = 0usize;
    while i < infos.len() {
        let cl = infos[i].cluster;
        let mut j = i + 1;
        while j < infos.len() && infos[j].cluster == cl {
            j += 1;
        }
        let mut cluster_adv = 0.0_f32;
        for pos in &positions[i..j] {
            cluster_adv += (pos.x_advance as f32) * scale;
        }
        let c_start = (cl as usize).min(run_str.len());
        let c_end = if is_ltr {
            if j < infos.len() {
                (infos[j].cluster as usize).min(run_str.len())
            } else {
                run_str.len()
            }
        } else {
            let mut next_higher = run_str.len();
            for inf in infos {
                let oc = inf.cluster as usize;
                if oc > c_start && oc < next_higher {
                    next_higher = oc;
                }
            }
            next_higher
        };

        let first_ch = run_str[c_start..c_end].chars().next();
        if cluster_adv <= 0.01
            && first_ch.is_some_and(|c| {
                !c.is_whitespace()
                    && !is_cluster_continuation(c)
                    && !is_ignorable_control_char(c)
            })
        {
            cluster_adv = font_size;
        }

        if !visit_cluster_char_cells(&run_str[c_start..c_end], base_byte + c_start, cluster_adv, visit) {
            return (false, shaped.clear());
        }
        i = j;
    }

    (true, shaped.clear())
}

/// Tokenize a single `\n`-delimited hard line into `WordRun`s and contiguous `ShapedCluster`s:
/// - Single spaces `' '` between words use the zero-glyph inter-`WordRun` `space_w` separator.
/// - Leading spaces, trailing spaces, extra consecutive spaces (`"  "`), and tabs `'\t'` emit zero-primitive
///   whitespace `ShapedGlyph` advances (`FontId::FALLBACK`, `b' '`), preserving exact monospace column alignment
///   and `text_input` caret alignment while keeping normal single-spaced prose 100% zero-overhead!
#[allow(clippy::too_many_arguments)]
fn shape_hard_line_words(
    line_str: &str,
    line_base_byte: usize,
    space_w: f32,
    glyphs: &mut Vec<ShapedGlyph>,
    words: &mut Vec<WordRun>,
    clusters: &mut Vec<ShapedCluster>,
    min_word_w: &mut f32,
    mut shape_word_slice: impl FnMut(&str, usize, &mut Vec<ShapedGlyph>, &mut Vec<ShapedCluster>) -> f32,
) -> f32 {
    if line_str.is_empty() {
        clusters.push(ShapedCluster {
            byte_start: line_base_byte as u32,
            advance: 0.0,
        });
        return 0.0;
    }

    let push_space_glyphs = |out: &mut Vec<ShapedGlyph>, extra_w: f32| {
        if extra_w > 0.0 {
            out.push(ShapedGlyph {
                glyph_id: b' ' as u16,
                font_id: FontId::FALLBACK,
                x_advance: extra_w,
                x_offset: 0.0,
                y_offset: 0.0,
            });
        }
    };

    let bytes = line_str.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut leading_ws_w = 0.0_f32;
    while pos < len && matches!(bytes[pos], b' ' | b'\t') {
        let w = if bytes[pos] == b'\t' { space_w * 4.0 } else { space_w };
        clusters.push(ShapedCluster {
            byte_start: (line_base_byte + pos) as u32,
            advance: w,
        });
        leading_ws_w += w;
        pos += 1;
    }

    // Line consists solely of spaces/tabs (e.g., `"   "` or `"\t"` in an input field):
    if pos == len {
        let start_g = glyphs.len() as u32;
        push_space_glyphs(glyphs, leading_ws_w);
        let end_g = glyphs.len() as u32;
        *min_word_w = min_word_w.max(leading_ws_w);
        words.push(WordRun {
            width: leading_ws_w,
            glyphs: CopyRange {
                start: start_g,
                end: end_g,
            },
        });
        clusters.push(ShapedCluster {
            byte_start: (line_base_byte + len) as u32,
            advance: 0.0,
        });
        return leading_ws_w;
    }

    let first_word_idx = words.len();
    let mut unwrapped_w = 0.0_f32;
    let mut pending_lead_w = leading_ws_w;

    while pos < len {
        let w_start = pos;
        while pos < len && !matches!(bytes[pos], b' ' | b'\t') {
            pos += 1;
        }
        let word_slice = &line_str[w_start..pos];

        let start_g = glyphs.len() as u32;
        if pending_lead_w > 0.0 {
            push_space_glyphs(glyphs, pending_lead_w);
        }
        let core_w = shape_word_slice(word_slice, line_base_byte + w_start, glyphs, clusters);

        let mut ws_after_w = 0.0_f32;
        while pos < len && matches!(bytes[pos], b' ' | b'\t') {
            let w = if bytes[pos] == b'\t' { space_w * 4.0 } else { space_w };
            clusters.push(ShapedCluster {
                byte_start: (line_base_byte + pos) as u32,
                advance: w,
            });
            ws_after_w += w;
            pos += 1;
        }

        // If another word follows on this hard line, 1 `space_w` is supplied by the inter-word separator;
        // any surplus (`ws_after_w - space_w`) or trailing whitespace at EOL stays attached to this word.
        let extra_trail_w = if pos < len {
            (ws_after_w - space_w).max(0.0)
        } else {
            ws_after_w
        };
        if extra_trail_w > 0.0 {
            push_space_glyphs(glyphs, extra_trail_w);
        }
        let end_g = glyphs.len() as u32;

        let word_w = pending_lead_w + core_w + extra_trail_w;
        pending_lead_w = 0.0;

        *min_word_w = min_word_w.max(word_w);
        if words.len() > first_word_idx {
            unwrapped_w += space_w;
        }
        unwrapped_w += word_w;
        words.push(WordRun {
            width: word_w,
            glyphs: CopyRange {
                start: start_g,
                end: end_g,
            },
        });
    }

    clusters.push(ShapedCluster {
        byte_start: (line_base_byte + len) as u32,
        advance: 0.0,
    });
    unwrapped_w
}

/// Compact 16-byte shaped glyph (`font_id` occupies the 2-byte alignment slot next to `glyph_id` — 0 extra bytes!).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ShapedGlyph {
    pub glyph_id: u16,
    pub font_id: FontId,
    pub x_advance: f32,
    pub x_offset: f32,
    pub y_offset: f32,
}

const _: () = assert!(core::mem::size_of::<ShapedGlyph>() == 16);

#[derive(Clone, Copy, Debug)]
pub struct WordRun {
    pub width: f32,
    pub glyphs: CopyRange<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct WrappedLine {
    pub words: CopyRange<u32>,
    pub width: f32,
    pub baseline_y: f32,
}

/// Compact 8-byte shaped cluster record enabling zero-allocation `$O(\text{clusters})$` caret & hit-test queries
/// directly on [`ShapedParagraph`] without re-invoking `rustybuzz::shape`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct ShapedCluster {
    byte_start: u32,
    advance: f32,
}

const _: () = assert!(core::mem::size_of::<ShapedCluster>() == 8);

/// Shaped paragraph with Explicit `\n` Hard-Line Support, Zero-Allocation Caret Hit-Testing, and
/// Exact `[W_tight, W_next_unwrap)` Width-Interval Memoization!
#[derive(Clone, Debug)]
pub struct ShapedParagraph {
    pub content_hash: u64,
    pub font_id: FontId,
    pub font_size: f32,
    pub line_height: f32,
    pub ascent: f32,
    pub space_w: f32,
    pub max_unwrapped_w: f32,
    pub min_word_w: f32,
    pub glyphs: Vec<ShapedGlyph>,
    pub words: Vec<WordRun>,
    clusters: Vec<ShapedCluster>,
    hard_lines: Vec<CopyRange<u32>>,
    hard_line_clusters: Vec<CopyRange<u32>>,
    pub valid_lo_w: f32,
    pub valid_hi_w: f32,
    pub cached_wrapped_h: f32,
    pub cached_tight_w: f32,
    pub cached_lines: Vec<WrappedLine>,
}

impl ShapedParagraph {
    pub fn shape_fallback(
        text: &str,
        font_size: f32,
        line_height: f32,
        content_hash: u64,
    ) -> Self {
        let mut para = Self {
            content_hash: !content_hash,
            font_id: FontId::FALLBACK,
            font_size,
            line_height,
            ascent: line_height * 0.78,
            space_w: (font_size * 0.56).max(1.0),
            max_unwrapped_w: 0.0,
            min_word_w: 0.0,
            glyphs: Vec::with_capacity(text.len()),
            words: Vec::new(),
            clusters: Vec::with_capacity(text.len() + 1),
            hard_lines: Vec::new(),
            hard_line_clusters: Vec::new(),
            valid_lo_w: f32::INFINITY,
            valid_hi_w: -1.0,
            cached_wrapped_h: line_height,
            cached_tight_w: 0.0,
            cached_lines: Vec::new(),
        };
        para.populate_fallback(text, font_size, line_height, content_hash);
        para
    }

    /// Update in place (zero heap allocations when `glyphs`/`words`/`clusters` capacity suffices) for `FontId::FALLBACK`!
    pub fn sync_fallback_in_place(&mut self, text: &str, font_size: f32, line_height: f32) -> bool {
        let h = hash_text_style(text, FontId::FALLBACK, font_size, line_height);
        if self.content_hash == h && self.font_id == FontId::FALLBACK {
            return false;
        }
        self.populate_fallback(text, font_size, line_height, h);
        true
    }

    fn populate_fallback(
        &mut self,
        text: &str,
        font_size: f32,
        line_height: f32,
        content_hash: u64,
    ) {
        let safe_fs = if font_size.is_finite() { font_size.max(1.0) } else { 14.0 };
        let safe_lh = if line_height.is_finite() { line_height.max(0.0) } else { safe_fs };
        let char_adv = (safe_fs * 0.56).max(1.0);

        self.content_hash = content_hash;
        self.font_id = FontId::FALLBACK;
        self.font_size = safe_fs;
        self.line_height = safe_lh;
        self.ascent = safe_lh * 0.78;
        self.space_w = char_adv;
        self.glyphs.clear();
        self.words.clear();
        self.clusters.clear();
        self.hard_lines.clear();
        self.hard_line_clusters.clear();

        let mut max_unwrapped_w = 0.0_f32;
        let mut min_word_w = 0.0_f32;
        let has_newlines = text.contains('\n');

        if !text.is_empty() {
            let mut line_base_byte = 0usize;
            for raw_line in text.split('\n') {
                let line_str = raw_line.strip_suffix('\r').unwrap_or(raw_line);
                let hl_start = self.words.len() as u32;
                let cl_start = self.clusters.len() as u32;
                let line_w = shape_hard_line_words(
                    line_str,
                    line_base_byte,
                    char_adv,
                    &mut self.glyphs,
                    &mut self.words,
                    &mut self.clusters,
                    &mut min_word_w,
                    |word_slice, word_base_byte, out_glyphs, out_clusters| {
                        shape_fallback_slice(
                            word_slice,
                            word_base_byte,
                            char_adv,
                            out_glyphs,
                            out_clusters,
                        )
                    },
                );
                let cl_end = (self.clusters.len() - 1) as u32;
                max_unwrapped_w = max_unwrapped_w.max(line_w);
                if has_newlines {
                    self.hard_lines.push(CopyRange {
                        start: hl_start,
                        end: self.words.len() as u32,
                    });
                    self.hard_line_clusters.push(CopyRange {
                        start: cl_start,
                        end: cl_end,
                    });
                }
                line_base_byte += raw_line.len() + 1;
            }
        }

        self.max_unwrapped_w = max_unwrapped_w;
        self.min_word_w = min_word_w;
        self.valid_lo_w = f32::INFINITY;
        self.valid_hi_w = -1.0;
        self.cached_tight_w = max_unwrapped_w;
    }

    #[inline]
    fn line_cluster_range(&self, clamped_byte: usize) -> CopyRange<usize> {
        if self.hard_line_clusters.is_empty() {
            CopyRange {
                start: 0,
                end: self.clusters.len().saturating_sub(1),
            }
        } else {
            let idx = self
                .hard_line_clusters
                .partition_point(|hl| {
                    (self.clusters[hl.start as usize].byte_start as usize) <= clamped_byte
                })
                .saturating_sub(1);
            let hl = self.hard_line_clusters[idx];
            CopyRange {
                start: hl.start as usize,
                end: hl.end as usize,
            }
        }
    }

    /// Zero-allocation `$O(\text{clusters})$` horizontal caret `x` query directly on the already-shaped paragraph!
    /// Never re-invokes `rustybuzz::shape` and preserves exact OpenType kerning, ligatures, `\t` tabs, and ZWJ emojis.
    pub fn byte_offset_to_x(&self, text: &str, byte_offset: usize) -> f32 {
        let clamped = text.floor_char_boundary(byte_offset.min(text.len()));
        if clamped == 0 || self.clusters.is_empty() {
            return 0.0;
        }
        let cl_range = self.line_cluster_range(clamped);
        if cl_range.start >= cl_range.end {
            return 0.0;
        }
        let line_start = self.clusters[cl_range.start].byte_start as usize;
        let line_end = self.clusters[cl_range.end].byte_start as usize;
        let target = clamped.min(line_end);
        if target <= line_start {
            return 0.0;
        }

        let mut pen_x = 0.0_f32;
        for k in cl_range.start..cl_range.end {
            let c_start = self.clusters[k].byte_start as usize;
            if c_start >= target {
                break;
            }
            let c_end = self.clusters[k + 1].byte_start as usize;
            let adv = self.clusters[k].advance;
            if c_end <= target {
                pen_x += adv;
            } else {
                let s = text.floor_char_boundary(c_start.min(text.len()));
                let e = text.floor_char_boundary(c_end.min(text.len())).max(s);
                visit_cluster_char_cells(&text[s..e], s, adv, &mut |b0, _b1, sub_adv| {
                    if b0 >= target {
                        return false;
                    }
                    pen_x += sub_adv;
                    true
                });
                break;
            }
        }
        pen_x
    }

    /// Zero-allocation `$O(\text{clusters})$` local `x` hit-test to the nearest UTF-8 char boundary on the shaped line!
    /// Never re-invokes `rustybuzz::shape`, splits multi-char ligatures proportionally, and keeps ZWJ emojis intact.
    pub fn x_to_byte_offset(&self, text: &str, target_x: f32) -> usize {
        if !target_x.is_finite() || target_x <= 0.0 || text.is_empty() || self.clusters.is_empty() {
            return 0;
        }
        let cl_range = if let Some(hl0) = self.hard_line_clusters.first() {
            CopyRange {
                start: hl0.start as usize,
                end: hl0.end as usize,
            }
        } else {
            CopyRange {
                start: 0,
                end: self.clusters.len().saturating_sub(1),
            }
        };
        if cl_range.start >= cl_range.end {
            return 0;
        }
        let line_end = text.floor_char_boundary(
            (self.clusters[cl_range.end].byte_start as usize).min(text.len()),
        );
        if line_end == 0 {
            return 0;
        }

        let mut pen = 0.0_f32;
        let mut hit: Option<usize> = None;
        for k in cl_range.start..cl_range.end {
            let adv = self.clusters[k].advance;
            if adv <= 0.0 {
                continue;
            }
            if target_x >= pen + adv {
                pen += adv;
                continue;
            }
            let c_start = self.clusters[k].byte_start as usize;
            let c_end = self.clusters[k + 1].byte_start as usize;
            if c_end.saturating_sub(c_start) <= 1 {
                if target_x < pen + adv * 0.5 {
                    hit = Some(text.floor_char_boundary(c_start.min(text.len())));
                    break;
                }
                pen += adv;
            } else {
                let s = text.floor_char_boundary(c_start.min(text.len()));
                let e = text.floor_char_boundary(c_end.min(text.len())).max(s);
                let mut sub_pen = pen;
                visit_cluster_char_cells(&text[s..e], s, adv, &mut |b0, _b1, sub_adv| {
                    if sub_adv > 0.0 && target_x < sub_pen + sub_adv * 0.5 {
                        hit = Some(b0);
                        return false;
                    }
                    sub_pen += sub_adv;
                    true
                });
                if hit.is_some() {
                    break;
                }
                pen += adv;
            }
        }
        hit.unwrap_or(line_end)
    }

    /// Exact greedy word-wrap across all hard `\n` lines + maximal mathematical invariance interval `[w_tight, w_next_unwrap)`.
    #[inline]
    pub fn wrap_for_width(&mut self, avail_w: f32) -> (f32, f32, bool) {
        let w = if avail_w.is_nan() {
            0.0
        } else {
            avail_w.max(0.0)
        };
        if w >= self.valid_lo_w
            && (w < self.valid_hi_w || (w == f32::INFINITY && self.valid_hi_w == f32::INFINITY))
        {
            return (self.cached_wrapped_h, self.cached_tight_w, true);
        }

        self.cached_lines.clear();
        if self.words.is_empty() && self.hard_lines.is_empty() {
            self.valid_lo_w = 0.0;
            self.valid_hi_w = f32::INFINITY;
            self.cached_wrapped_h = 0.0;
            self.cached_tight_w = 0.0;
            return (0.0, 0.0, false);
        }

        let mut w_lo = 0.0_f32;
        let mut w_hi = f32::INFINITY;
        let mut actual_tight = 0.0_f32;

        let single_span = [CopyRange {
            start: 0,
            end: self.words.len() as u32,
        }];
        let spans: &[CopyRange<u32>] = if self.hard_lines.is_empty() {
            &single_span
        } else {
            &self.hard_lines
        };

        for &hl in spans {
            if hl.start == hl.end {
                // Explicit empty hard line (`\n\n`, leading `\n`, or trailing `\n`):
                let line_idx = self.cached_lines.len() as f32;
                self.cached_lines.push(WrappedLine {
                    words: hl,
                    width: 0.0,
                    baseline_y: line_idx * self.line_height + self.ascent,
                });
                continue;
            }

            let first_w = self.words[hl.start as usize].width;
            let mut line_start = hl.start;
            let mut cur_line_w = first_w;
            actual_tight = actual_tight.max(first_w);

            for i in (hl.start + 1)..hl.end {
                let ww = self.words[i as usize].width;
                actual_tight = actual_tight.max(ww);
                let candidate = cur_line_w + self.space_w + ww;
                if candidate <= w {
                    cur_line_w = candidate;
                    actual_tight = actual_tight.max(cur_line_w);
                    w_lo = w_lo.max(cur_line_w);
                } else {
                    w_hi = w_hi.min(candidate);
                    let line_idx = self.cached_lines.len() as f32;
                    self.cached_lines.push(WrappedLine {
                        words: CopyRange {
                            start: line_start,
                            end: i,
                        },
                        width: cur_line_w,
                        baseline_y: line_idx * self.line_height + self.ascent,
                    });
                    line_start = i;
                    cur_line_w = ww;
                }
            }

            let last_idx = self.cached_lines.len() as f32;
            self.cached_lines.push(WrappedLine {
                words: CopyRange {
                    start: line_start,
                    end: hl.end,
                },
                width: cur_line_w,
                baseline_y: last_idx * self.line_height + self.ascent,
            });
        }

        let total_h = (self.cached_lines.len() as f32) * self.line_height;
        self.valid_lo_w = w_lo;
        self.valid_hi_w = w_hi;
        self.cached_wrapped_h = total_h;
        self.cached_tight_w = actual_tight;
        (total_h, actual_tight, false)
    }
}

#[inline]
pub fn hash_text_style(text: &str, font_id: FontId, font_size: f32, line_height: f32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in text.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h ^= (font_id.0 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= (font_size.to_bits() as u64).rotate_left(17);
    h ^= (line_height.to_bits() as u64).rotate_left(41);
    h
}

fn collect_system_font_roots() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut push_unique = |p: PathBuf| {
        if p.is_dir() && !dirs.contains(&p) {
            dirs.push(p);
        }
    };

    // Parse `<dir>...</dir>` from `/etc/fonts/fonts.conf` and `/etc/fonts/conf.d/*.conf` (NixOS & Linux!)
    parse_fontconfig_xml(Path::new("/etc/fonts/fonts.conf"), &mut push_unique);
    if let Ok(conf_d) = std::fs::read_dir("/etc/fonts/conf.d") {
        for ent in conf_d.flatten() {
            let p = ent.path();
            if p.extension().is_some_and(|e| e == "conf") {
                parse_fontconfig_xml(&p, &mut push_unique);
            }
        }
    }

    // Standard Linux, macOS, Windows, and XDG user font directories:
    for std_dir in [
        "/run/current-system/sw/share/fonts",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/System/Library/Fonts",
        "/Library/Fonts",
        "C:\\Windows\\Fonts",
    ] {
        push_unique(PathBuf::from(std_dir));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let hp = PathBuf::from(home);
        push_unique(hp.join(".local/share/fonts"));
        push_unique(hp.join("Library/Fonts"));
    }
    dirs
}

fn parse_fontconfig_xml(xml_path: &Path, push: &mut impl FnMut(PathBuf)) {
    let Ok(txt) = std::fs::read_to_string(xml_path) else {
        return;
    };
    let mut rest = txt.as_str();
    while let Some(open) = rest.find("<dir") {
        let after_tag = &rest[open + 4..];
        let Some(gt) = after_tag.find('>') else {
            break;
        };
        let body_start = &after_tag[gt + 1..];
        let Some(close) = body_start.find("</dir>") else {
            break;
        };
        let raw_dir = body_start[..close].trim();
        if raw_dir.starts_with('/') || raw_dir.contains(":\\") {
            push(PathBuf::from(raw_dir));
        }
        rest = &body_start[close + 6..];
    }
}

fn scan_font_dir_shallow(dir: &Path, depth: u8, cb: &mut impl FnMut(&Path)) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let p = ent.path();
        if p.is_dir() {
            scan_font_dir_shallow(&p, depth + 1, cb);
        } else {
            cb(&p);
        }
    }
}

fn mini_glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' | '0' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' | '5' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b10010, 0b10010, 0b10010, 0b11111, 0b00010, 0b00010, 0b00010],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        ';' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b01000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '\\' => [0b10000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00010, 0b00001],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '{' => [0b00110, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00110],
        '}' => [0b01100, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b01100],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '>' => [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '?' => [0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        '&' => [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
        '*' => [0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000],
        '\'' | '`' => [0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '"' => [0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '@' => [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110],
        '$' => [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100],
        '^' => [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
        '~' => [0b00000, 0b00000, 0b01000, 0b10101, 0b00010, 0b00000, 0b00000],
        _ => [0b11111, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b11111],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_atlas_oversized_and_in_place_bounds_safety() {
        // 0x0 and 1x1 degenerate atlases must never panic:
        let mut zero_atlas = GlyphAtlas::new(0, 0);
        assert!(zero_atlas.upload_r8(4, 4, 0, 0, &[255; 16]).is_none());
        zero_atlas.clear_on_saturation();

        let mut one_atlas = GlyphAtlas::new(1, 1);
        assert!(one_atlas.upload_rgba8(2, 2, 0, 0, &[255; 16]).is_none());

        // Oversized uploads (> width or > height) and short pixel slices must return `None`
        // WITHOUT wiping cached glyphs or bumping `generation`!
        let mut atlas = GlyphAtlas::new(32, 32);
        let valid = atlas.upload_r8(8, 8, 0, 0, &[200u8; 64]);
        assert!(valid.is_some());
        let gen_after_valid = atlas.generation();

        assert!(atlas.upload_r8(33, 8, 0, 0, &[255u8; 33 * 8]).is_none());
        assert!(atlas.upload_r8(8, 33, 0, 0, &[255u8; 8 * 33]).is_none());
        assert!(atlas.upload_r8(64, 64, 0, 0, &[255u8; 64 * 64]).is_none());
        assert!(atlas.upload_r8(8, 8, 0, 0, &[255u8; 63]).is_none()); // 1 byte short
        assert!(atlas.upload_rgba8(40, 10, 0, 0, &[255u8; 40 * 10 * 4]).is_none());
        assert!(atlas.upload_rgba8(8, 8, 0, 0, &[255u8; 8 * 8 * 4 - 1]).is_none());
        assert_eq!(
            atlas.generation(),
            gen_after_valid,
            "Failed oversized upload must not wipe atlas generation!"
        );

        // Out-of-bounds `update_r8_in_place` and `update_rgba8_in_place` must not panic or push dirty rects:
        atlas.dirty_rects.clear();
        let bad_regions = [
            AtlasRegion {
                plane: Some(AtlasPlane::R8),
                x: 28,
                y: 0,
                w: 8,
                h: 8,
                ..Default::default()
            },
            AtlasRegion {
                plane: Some(AtlasPlane::R8),
                x: 0,
                y: 28,
                w: 8,
                h: 8,
                ..Default::default()
            },
            AtlasRegion {
                plane: Some(AtlasPlane::R8),
                x: u16::MAX,
                y: u16::MAX,
                w: 16,
                h: 16,
                ..Default::default()
            },
            AtlasRegion {
                plane: Some(AtlasPlane::Rgba8), // wrong plane for `update_r8_in_place`
                x: 4,
                y: 4,
                w: 4,
                h: 4,
                ..Default::default()
            },
        ];
        for br in bad_regions {
            atlas.update_r8_in_place(br, &[255u8; 1024]);
            let mut br_rgba = br;
            br_rgba.plane = Some(AtlasPlane::Rgba8);
            if br.x > 8 || br.y > 8 {
                atlas.update_rgba8_in_place(br_rgba, &[255u8; 4096]);
            }
        }
        // Also short slice on valid region:
        atlas.update_r8_in_place(valid.unwrap(), &[1u8; 10]);
        assert!(atlas.dirty_rects.is_empty());

        // Valid in-place update succeeds cleanly:
        atlas.update_r8_in_place(valid.unwrap(), &[123u8; 64]);
        assert_eq!(atlas.dirty_rects.len(), 1);
    }

    #[test]
    fn cleartype_5tap_fir_energy_balance_and_zero_color_fringing() {
        // 1. Verify exact impulse response `[1, 2, 3, 2, 1] / 9` and equal R/G/B total energy
        //    across all 3 subpixel impulse phases `k0 in {0, 1, 2}`:
        for k0 in 0..3i32 {
            let sub = [255u8];
            let p_min = (k0 - 2).div_euclid(3);
            let p_max = (k0 + 2).div_euclid(3);
            let out_w = (p_max - p_min + 1) as usize;
            let mut rgba = vec![0u8; out_w * 4];
            filter_cleartype_5tap_row(&sub, k0, p_min, out_w, &mut rgba);

            let sum_r: u32 = rgba.as_chunks::<4>().0.iter().map(|px| px[0] as u32).sum();
            let sum_g: u32 = rgba.as_chunks::<4>().0.iter().map(|px| px[1] as u32).sum();
            let sum_b: u32 = rgba.as_chunks::<4>().0.iter().map(|px| px[2] as u32).sum();
            // `[1, 2, 3, 2, 1]/9` on 255 gives taps `[28, 57, 85, 57, 28]` -> R=85, G=57+28=85, B=57+28=85!
            assert_eq!(sum_r, 85);
            assert_eq!(sum_g, 85);
            assert_eq!(sum_b, 85);
        }

        // 2. Verify solid coverage (`255`) yields exact `[255, 255, 255, 255]` in interior:
        let solid_sub = [255u8; 15];
        let mut solid_rgba = [0u8; 7 * 4];
        filter_cleartype_5tap_row(&solid_sub, 0, -1, 7, &mut solid_rgba);
        // Interior pixel `px = 2` (`p = 1`, subpixels `3, 4, 5` surrounded by `1..=7`):
        assert_eq!(&solid_rgba[2 * 4..3 * 4], &[255, 255, 255, 255]);

        // 3. Verify arbitrary random-like subpixel stem pattern has balanced RGB sums:
        let stem = [40u8, 190, 255, 255, 120, 15, 210, 255, 80];
        let p_min = (-2i32).div_euclid(3);
        let p_max = (stem.len() as i32 + 1).div_euclid(3);
        let out_w = (p_max - p_min + 1) as usize;
        let mut out_rgba = vec![0u8; out_w * 4];
        filter_cleartype_5tap_row(&stem, 0, p_min, out_w, &mut out_rgba);
        let sr: i32 = out_rgba.as_chunks::<4>().0.iter().map(|c| c[0] as i32).sum();
        let sg: i32 = out_rgba.as_chunks::<4>().0.iter().map(|c| c[1] as i32).sum();
        let sb: i32 = out_rgba.as_chunks::<4>().0.iter().map(|c| c[2] as i32).sum();
        assert!((sr - sg).abs() <= out_w as i32 && (sg - sb).abs() <= out_w as i32);
    }

    #[test]
    fn explicit_newlines_empty_lines_wide_words_and_exact_width_interval_cache() {
        let fonts = FontSystem::with_system_fonts();
        let cases = [
            "",
            "\n",
            "SingleWideSupercalifragilisticexpialidociousWord",
            "Line one wraps cleanly\nLine two has explicit break\n\nLine four after blank line\n",
            "  Indented   monospace   columns  \n#000001  [layout::iale   ]  ok\n#000002  [text::cleartype]  ok",
        ];

        for &fid in &[FontId::FALLBACK, fonts.ui_font, fonts.mono_font] {
            for text in cases {
                let mut cached = fonts.shape_paragraph(text, fid, 14.0, 20.0);
                for w in (0..=600).step_by(3) {
                    let wf = w as f32;
                    let (h_c, tw_c, _hit) = cached.wrap_for_width(wf);
                    assert!(
                        cached.valid_lo_w <= wf && wf < cached.valid_hi_w,
                        "Active width {wf} outside [{}, {}) for {text:?}",
                        cached.valid_lo_w,
                        cached.valid_hi_w
                    );

                    let mut fresh = fonts.shape_paragraph(text, fid, 14.0, 20.0);
                    let (h_f, tw_f, _) = fresh.wrap_for_width(wf);
                    assert_eq!(h_c, h_f, "Wrapped height mismatch at w={wf} for {text:?}");
                    assert_eq!(tw_c, tw_f, "Tight width mismatch at w={wf} for {text:?}");
                    assert_eq!(cached.cached_lines.len(), fresh.cached_lines.len());
                    for (la, lb) in cached.cached_lines.iter().zip(fresh.cached_lines.iter()) {
                        assert_eq!(la.words, lb.words);
                        assert!((la.width - lb.width).abs() < 1e-5);
                        assert!((la.baseline_y - lb.baseline_y).abs() < 1e-5);
                    }
                }
                // Check `INFINITY` cache hit as well:
                let _ = cached.wrap_for_width(f32::INFINITY);
                let (_, _, inf_hit) = cached.wrap_for_width(f32::INFINITY);
                assert!(inf_hit);
            }
        }

        // Verify monospace column preservation (`[layout::iale   ]` vs `[text::cleartype]`):
        let row_a = fonts.shape_paragraph("#000001  [layout::iale   ]  ok", FontId::FALLBACK, 13.0, 18.0);
        let row_b = fonts.shape_paragraph("#000002  [text::cleartype]  ok", FontId::FALLBACK, 13.0, 18.0);
        assert!(
            (row_a.max_unwrapped_w - row_b.max_unwrapped_w).abs() < 1e-4,
            "Monospace padded rows must have identical unwrapped width: {} vs {}",
            row_a.max_unwrapped_w,
            row_b.max_unwrapped_w
        );
    }

    #[test]
    fn byte_offset_to_x_and_x_to_byte_offset_utf8_ligature_and_fallback_exactness() {
        let fonts = FontSystem::with_system_fonts();
        let samples = [
            "AV Wa To fi ffi fl ->",
            "Skima 1.0 🦀 ✓ café naïve",
            "  leading and  multiple   spaces ",
        ];

        for &fid in &[FontId::FALLBACK, fonts.ui_font, fonts.mono_font] {
            for s in samples {
                let para = fonts.shape_paragraph(s, fid, 14.0, 20.0);
                let end_x = fonts.byte_offset_to_x(s, s.len(), fid, 14.0);
                assert!(
                    (end_x - para.max_unwrapped_w).abs() < 1e-3,
                    "Total caret width {end_x} != paragraph width {} for {s:?} (fid={fid:?})",
                    para.max_unwrapped_w
                );

                // Leading and past-trailing hit-test bounds:
                assert_eq!(fonts.x_to_byte_offset(s, -50.0, fid, 14.0), 0);
                assert_eq!(fonts.x_to_byte_offset(s, 0.0, fid, 14.0), 0);
                assert_eq!(fonts.x_to_byte_offset(s, end_x + 100.0, fid, 14.0), s.len());

                // Collect all valid UTF-8 char boundaries in `s`:
                let mut boundaries: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
                boundaries.push(s.len());

                let mut prev_x = 0.0_f32;
                for w in boundaries.windows(2) {
                    let b0 = w[0];
                    let b1 = w[1];
                    let x0 = fonts.byte_offset_to_x(s, b0, fid, 14.0);
                    let x1 = fonts.byte_offset_to_x(s, b1, fid, 14.0);
                    assert!(x0 >= prev_x - 1e-5 && x1 >= x0 - 1e-5);
                    prev_x = x1;

                    // Mid-UTF-8 byte indices clamp cleanly to `b0`:
                    for mid_byte in (b0 + 1)..b1 {
                        let xm = fonts.byte_offset_to_x(s, mid_byte, fid, 14.0);
                        assert!((xm - x0).abs() < 1e-5);
                    }

                    let adv = x1 - x0;
                    if adv > 1e-3 {
                        // Clicking at `25%` (left half) snaps to `b0`; clicking at `75%` (right half) snaps to `b1`:
                        let hit_left = fonts.x_to_byte_offset(s, x0 + adv * 0.25, fid, 14.0);
                        let hit_right = fonts.x_to_byte_offset(s, x0 + adv * 0.75, fid, 14.0);
                        assert_eq!(hit_left, b0, "Left-half click failed on {:?} in {s:?}", &s[b0..b1]);
                        assert_eq!(hit_right, b1, "Right-half click failed on {:?} in {s:?}", &s[b0..b1]);
                        assert!(s.is_char_boundary(hit_left) && s.is_char_boundary(hit_right));
                    }
                }
            }
        }
    }

    #[test]
    fn subpixel_lcd_glyph_rasterization_phases_and_oversized_font_resilience() {
        let mut fonts = FontSystem::with_system_fonts();
        let para = fonts.shape_paragraph("HMWg!", fonts.ui_font, 18.0, 24.0);
        assert!(!para.glyphs.is_empty());

        for sg in &para.glyphs {
            for phase in [0.0_f32, 0.25, 0.50, 0.75, -1.35] {
                let reg = fonts.rasterize_glyph(
                    sg.font_id,
                    sg.glyph_id,
                    18.0,
                    phase,
                    AntialiasMode::SubpixelLcd,
                );
                if sg.font_id != FontId::FALLBACK && reg.plane == Some(AtlasPlane::Rgba8) {
                    assert!(!reg.is_color, "Monochrome SubpixelLcd mask must have is_color == false");
                    assert!(reg.w > 0 && reg.h > 0);
                    // Verify every row in the uploaded atlas glyph has balanced R, G, B channel energy:
                    let stride = (fonts.atlas.width as usize) * 4;
                    for r in 0..(reg.h as usize) {
                        let start = (reg.y as usize + r) * stride + (reg.x as usize) * 4;
                        let row_px = &fonts.atlas.rgba8_data[start..start + (reg.w as usize) * 4];
                        let sr: i32 = row_px.as_chunks::<4>().0.iter().map(|p| p[0] as i32).sum();
                        let sg_c: i32 = row_px.as_chunks::<4>().0.iter().map(|p| p[1] as i32).sum();
                        let sb: i32 = row_px.as_chunks::<4>().0.iter().map(|p| p[2] as i32).sum();
                        assert!(
                            (sr - sg_c).abs() <= (reg.w as i32) && (sg_c - sb).abs() <= (reg.w as i32),
                            "Subpixel LCD row energy imbalance: R={sr}, G={sg_c}, B={sb}"
                        );
                    }
                }
            }
        }

        // Non-finite and oversized font sizes on a tiny atlas must never panic:
        fonts.atlas = GlyphAtlas::new(24, 24);
        let _ = fonts.rasterize_glyph(fonts.ui_font, para.glyphs[0].glyph_id, 900.0, 0.0, AntialiasMode::SubpixelLcd);
        let _ = fonts.rasterize_glyph(FontId::FALLBACK, b'A' as u16, 900.0, 0.0, AntialiasMode::Grayscale);
        let nan_reg = fonts.rasterize_glyph(fonts.ui_font, 1, f32::NAN, f32::INFINITY, AntialiasMode::Grayscale);
        assert_eq!(nan_reg.w, 0);

        // Verify interval maximality (`valid_lo_w - eps` breaks, `valid_hi_w` unwraps) and `sync_fallback_in_place`:
        let prose = "Exact width interval memoization avoids redundant paragraph line breaking";
        let mut p = ShapedParagraph::shape_fallback(prose, 14.0, 20.0, hash_text_style(prose, FontId::FALLBACK, 14.0, 20.0));
        let (h_mid, _, _) = p.wrap_for_width(220.0);
        let lo = p.valid_lo_w;
        let hi = p.valid_hi_w;
        assert!(lo > 0.0 && hi.is_finite() && lo <= 220.0 && 220.0 < hi);
        assert!(p.wrap_for_width(lo).2, "valid_lo_w must be a cache hit");
        assert!(p.wrap_for_width(hi - 0.01).2, "valid_hi_w - eps must be a cache hit");
        let (h_narrower, _, hit_lo_out) = p.wrap_for_width(lo - 0.05);
        assert!(!hit_lo_out && h_narrower >= h_mid);
        let _ = p.wrap_for_width(220.0);
        let (h_wider, _, hit_hi_out) = p.wrap_for_width(hi + 0.05);
        assert!(!hit_hi_out && h_wider <= h_mid);

        // Verify `sync_fallback_in_place` with CRLF `\r\n` and `usize::MAX` + NFD combining marks:
        assert!(p.sync_fallback_in_place("First\r\n\r\nThird", 14.0, 20.0));
        assert!(!p.sync_fallback_in_place("First\r\n\r\nThird", 14.0, 20.0)); // idempotent hash hit
        let (h_crlf, _, _) = p.wrap_for_width(500.0);
        assert_eq!(h_crlf, 60.0);

        let nfd = "e\u{0301} a\u{0308} 🦀";
        let w_nfd = fonts.byte_offset_to_x(nfd, usize::MAX, fonts.ui_font, 14.0);
        assert!(w_nfd > 0.0);
        for x in [0.0, w_nfd * 0.2, w_nfd * 0.5, w_nfd * 0.8, w_nfd + 10.0] {
            let off = fonts.x_to_byte_offset(nfd, x, fonts.ui_font, 14.0);
            assert!(nfd.is_char_boundary(off));
        }
    }

    #[test]
    fn shaped_paragraph_caret_hit_test_parity_tabs_control_chars_and_zwj_emojis() {
        let mut fonts = FontSystem::with_system_fonts();
        let samples = [
            "",
            "AV Wa To fi ffi fl ->",
            "\tcol1\tcol2\t\tcol3 ",
            "ctrl\x00\x07\x1b\rmid\x7fend",
            "ZWJ 👨‍👩‍👧‍👦 🦀 ⚡ ⚡\u{FE0F} ✅ ❤️ café e\u{0301}",
            "first line\r\n\n\tindented third line\n",
        ];

        for &fid in &[FontId::FALLBACK, fonts.ui_font, fonts.mono_font] {
            // 1. Verify `\t` == `4.0 * space_w` and control chars `\0..\x1F`, `\r`, `\x7F` emit 0 `.notdef` boxes:
            let tab_para = fonts.shape_paragraph("\t", fid, 14.0, 20.0);
            assert!(
                (tab_para.max_unwrapped_w - 4.0 * tab_para.space_w).abs() < 1e-5,
                "Tab must advance by exact 4 * space_w"
            );
            for sg in &tab_para.glyphs {
                let reg = fonts.rasterize_glyph(sg.font_id, sg.glyph_id, 14.0, 0.0, AntialiasMode::SubpixelLcd);
                assert_eq!((reg.w, reg.h), (0, 0), "Tab must never rasterize a visible .notdef glyph");
            }

            let ctrl_only = "\x00\x01\x07\x08\x0b\x0c\r\x1b\x1f\x7f";
            let ctrl_para = fonts.shape_paragraph(ctrl_only, fid, 14.0, 20.0);
            assert_eq!(ctrl_para.max_unwrapped_w, 0.0);
            assert!(
                ctrl_para.glyphs.is_empty(),
                "Ignorable control chars must emit 0 ShapedGlyphs (no .notdef tofu boxes!)"
            );

            // 2. Verify `ShapedParagraph::byte_offset_to_x` & `x_to_byte_offset` match `FontSystem` across every sample:
            for s in samples {
                let para = fonts.shape_paragraph(s, fid, 14.0, 20.0);
                for byte_i in 0..=s.len() {
                    let px = para.byte_offset_to_x(s, byte_i);
                    let fx = fonts.byte_offset_to_x(s, byte_i, fid, 14.0);
                    assert!(
                        (px - fx).abs() < 1e-3,
                        "byte_offset_to_x mismatch at byte {byte_i} in {s:?} ({fid:?}): para={px} vs fs={fx}"
                    );
                }

                let max_x = para.max_unwrapped_w + 24.0;
                let mut x = -8.0_f32;
                while x <= max_x {
                    let pb = para.x_to_byte_offset(s, x);
                    let fb = fonts.x_to_byte_offset(s, x, fid, 14.0);
                    assert!(s.is_char_boundary(pb), "Non-UTF8 boundary {pb} at x={x} in {s:?}");
                    assert_eq!(
                        pb, fb,
                        "x_to_byte_offset mismatch at x={x} in {s:?} ({fid:?}): para={pb} vs fs={fb}"
                    );
                    x += 0.75;
                }
            }

            // 3. Verify `👨‍👩‍👧‍👦` ZWJ cluster hit-tests cleanly to outer start (byte 0) or outer end (byte 25), never mid-ZWJ:
            let family = "👨‍👩‍👧‍👦";
            assert_eq!(family.len(), 25);
            let fam_para = fonts.shape_paragraph(family, fid, 16.0, 22.0);
            assert!(fam_para.max_unwrapped_w > 1.0);
            assert_eq!(fam_para.x_to_byte_offset(family, fam_para.max_unwrapped_w * 0.25), 0);
            assert_eq!(fam_para.x_to_byte_offset(family, fam_para.max_unwrapped_w * 0.75), 25);
        }
    }

    #[test]
    fn dynamic_glyph_atlas_growth_1024_to_2048_to_4096_preserves_texels_and_rescales_uvs() {
        let mut fonts = FontSystem::with_system_fonts();
        assert_eq!((fonts.atlas.width, fonts.atlas.height), (1024, 1024));

        // Cache a glyph in the initial `1024x1024` atlas before saturation:
        let para = fonts.shape_paragraph("A", fonts.ui_font, 20.0, 26.0);
        let sg = para.glyphs[0];
        let reg_1024 = fonts.rasterize_glyph(sg.font_id, sg.glyph_id, 20.0, 0.0, AntialiasMode::Grayscale);
        assert!(reg_1024.w > 0 && reg_1024.h > 0);
        assert_eq!(fonts.atlas.cached_glyph_count(), 1);

        // Fill `1024x1024` with 5 `500x500` tiles (4 fit 2x2; the 5th saturates `1024x1024` -> grows `1024 -> 2048`)!
        let tile = vec![177u8; 500 * 500];
        for _ in 0..5 {
            assert!(fonts.atlas.upload_r8(500, 500, 0, 0, &tile).is_some());
        }
        assert_eq!((fonts.atlas.width, fonts.atlas.height), (2048, 2048));
        assert_eq!(
            fonts.atlas.cached_glyph_count(),
            1,
            "Growing 1024->2048 must preserve cached glyphs instead of wiping cache!"
        );

        let reg_2048 = fonts.rasterize_glyph(sg.font_id, sg.glyph_id, 20.0, 0.0, AntialiasMode::Grayscale);
        assert_eq!((reg_2048.x, reg_2048.y, reg_2048.w, reg_2048.h), (reg_1024.x, reg_1024.y, reg_1024.w, reg_1024.h));
        assert!((reg_2048.u1 - reg_1024.u1 * 0.5).abs() < 1e-6, "UVs must rescale by 0.5 on 1024->2048 growth");

        // Push past `2048x2048` -> dynamically grows to `4096x4096`!
        let big_tile = vec![99u8; 1000 * 1000];
        for _ in 0..10 {
            assert!(fonts.atlas.upload_r8(1000, 1000, 0, 0, &big_tile).is_some());
        }
        assert_eq!((fonts.atlas.width, fonts.atlas.height), (4096, 4096));
        assert_eq!(fonts.atlas.cached_glyph_count(), 1);
        let reg_4096 = fonts.rasterize_glyph(sg.font_id, sg.glyph_id, 20.0, 0.0, AntialiasMode::Grayscale);
        assert!((reg_4096.u1 - reg_1024.u1 * 0.25).abs() < 1e-6, "UVs must rescale by 0.25 on 1024->4096 growth");

        // Regression check: failed oversized rasterization on a small initial atlas must NEVER poison `atlas.cache`,
        // so once `try_grow()` enlarges the atlas, the same glyph rasterizes and caches cleanly!
        fonts.atlas = GlyphAtlas::new(24, 24);
        let fail_reg = fonts.rasterize_glyph(FontId::FALLBACK, b'M' as u16, 40.0, 0.0, AntialiasMode::Grayscale);
        assert_eq!((fail_reg.w, fail_reg.h), (0, 0));
        assert_eq!(fonts.atlas.cached_glyph_count(), 0, "Failed oversized upload must not poison glyph cache!");
        assert!(fonts.atlas.try_grow()); // 24x24 -> 48x48
        let ok_reg = fonts.rasterize_glyph(FontId::FALLBACK, b'M' as u16, 40.0, 0.0, AntialiasMode::Grayscale);
        assert_eq!((ok_reg.w, ok_reg.h), (20, 28), "Glyph must succeed after try_grow without cache poisoning!");
        assert_eq!(fonts.atlas.cached_glyph_count(), 1);

        // Regression check (#62): QHD / 2x Retina wide single-item uploads (`1024 < w <= 4096`, e.g. `1480x62`
        // and `2960x32`) on a fresh `1024x1024` default atlas auto-grow `1024 -> 2048 -> 4096` cleanly,
        // while `w > 4096` (`5000x16`) rejects immediately without mutating dimensions or generation!
        let mut qhd_atlas = GlyphAtlas::default();
        let gen_init = qhd_atlas.generation();
        assert!(qhd_atlas.upload_r8(5000, 16, 0, 0, &[255u8; 5000 * 16]).is_none());
        assert_eq!((qhd_atlas.width, qhd_atlas.height), (1024, 1024));
        assert_eq!(qhd_atlas.generation(), gen_init);

        let qhd_wave = vec![200u8; 1480 * 62];
        let reg_qhd = qhd_atlas.upload_r8(1480, 62, 0, 0, &qhd_wave).expect("1480x62 QHD upload must auto-grow 1024->2048");
        assert_eq!((qhd_atlas.width, qhd_atlas.height), (2048, 2048));
        assert_eq!((reg_qhd.w, reg_qhd.h), (1480, 62));

        let retina_rgba = vec![255u8; 2960 * 32 * 4];
        let reg_retina = qhd_atlas.upload_rgba8(2960, 32, 0, 0, &retina_rgba).expect("2960x32 2x-QHD upload must auto-grow 2048->4096");
        assert_eq!((qhd_atlas.width, qhd_atlas.height), (4096, 4096));
        assert_eq!((reg_retina.w, reg_retina.h), (2960, 32));
    }

    #[test]
    fn f26dot6_fractional_font_size_quantization_ulp_immunity_and_swash_hinting_clarity() {
        let mut fonts = FontSystem::with_system_fonts();
        let fid = fonts.ui_font;
        let para = fonts.shape_paragraph("HIlm01!", fid, 14.0, 20.0);
        assert!(!para.glyphs.is_empty());
        let gid = para.glyphs[0].glyph_id;

        // 1. Exact `F26Dot6` (`1/64` px) ULP-immune quantization across Wayland `1/120` fractional ratios:
        let s_exact = 17.5_f32;
        let s_ratio = 14.0_f32 * (150.0_f32 / 120.0_f32);
        let s_minus_ulp = f32::from_bits(s_exact.to_bits() - 1);
        let s_plus_ulp = f32::from_bits(s_exact.to_bits() + 1);
        assert_eq!(GlyphKey::quantize_f26dot6(s_exact), 1120);
        assert_eq!(GlyphKey::quantize_f26dot6(s_ratio), 1120);
        assert_eq!(GlyphKey::quantize_f26dot6(s_minus_ulp), 1120);
        assert_eq!(GlyphKey::quantize_f26dot6(s_plus_ulp), 1120);
        assert_eq!(
            GlyphKey::new(fid, gid, s_ratio, 0.0, AntialiasMode::SubpixelLcd),
            GlyphKey::new(fid, gid, s_exact, 0.0, AntialiasMode::SubpixelLcd)
        );

        // Non-dyadic `4/3` (`160/120`) and `5/3` (`200/120`) float reassociation immunity:
        let s_4_3_a = 14.0_f32 * (160.0_f32 / 120.0_f32);
        let s_4_3_b = (14.0_f32 * 160.0_f32) / 120.0_f32;
        assert_eq!(
            GlyphKey::quantize_f26dot6(s_4_3_a),
            GlyphKey::quantize_f26dot6(s_4_3_b)
        );
        let s_5_3_a = 13.0_f32 * (200.0_f32 / 120.0_f32);
        let s_5_3_b = (13.0_f32 * 200.0_f32) / 120.0_f32;
        assert_eq!(
            GlyphKey::quantize_f26dot6(s_5_3_a),
            GlyphKey::quantize_f26dot6(s_5_3_b)
        );

        // 2. Order-independent bitmap & `GlyphAtlas` cache-hit determinism across `-1 ULP`, `+0 ULP`, `+1 ULP`:
        let reg_first = fonts.rasterize_glyph(fid, gid, s_minus_ulp, 0.0, AntialiasMode::SubpixelLcd);
        let count_after_first = fonts.atlas.cached_glyph_count();
        assert_eq!(count_after_first, 1);
        assert!(fonts.atlas.contains_glyph_key(&GlyphKey::new(
            fid,
            gid,
            s_exact,
            0.0,
            AntialiasMode::SubpixelLcd
        )));
        let reg_second = fonts.rasterize_glyph(fid, gid, s_exact, 0.0, AntialiasMode::SubpixelLcd);
        let reg_third = fonts.rasterize_glyph(fid, gid, s_plus_ulp, 0.0, AntialiasMode::SubpixelLcd);
        let reg_ratio = fonts.rasterize_glyph(fid, gid, s_ratio, 0.0, AntialiasMode::SubpixelLcd);
        assert_eq!(fonts.atlas.cached_glyph_count(), 1, "1 ULP variants must hit existing GlyphAtlas entry!");
        assert_eq!(reg_first, reg_second);
        assert_eq!(reg_first, reg_third);
        assert_eq!(reg_first, reg_ratio);

        // Verify reverse insertion order (`s_exact` first on a fresh FontSystem) produces bit-identical texels!
        let mut fonts_rev = FontSystem::with_system_fonts();
        let reg_rev = fonts_rev.rasterize_glyph(
            fonts_rev.ui_font,
            gid,
            s_exact,
            0.0,
            AntialiasMode::SubpixelLcd,
        );
        assert_eq!((reg_first.w, reg_first.h, reg_first.bearing_x, reg_first.bearing_y), (reg_rev.w, reg_rev.h, reg_rev.bearing_x, reg_rev.bearing_y));
        if reg_first.plane == Some(AtlasPlane::Rgba8) && reg_first.w > 0 && reg_first.h > 0 {
            let stride_a = (fonts.atlas.width as usize) * 4;
            let stride_b = (fonts_rev.atlas.width as usize) * 4;
            let row_bytes = (reg_first.w as usize) * 4;
            for r in 0..(reg_first.h as usize) {
                let off_a = (reg_first.y as usize + r) * stride_a + (reg_first.x as usize) * 4;
                let off_b = (reg_rev.y as usize + r) * stride_b + (reg_rev.x as usize) * 4;
                assert_eq!(
                    &fonts.atlas.rgba8_data[off_a..off_a + row_bytes],
                    &fonts_rev.atlas.rgba8_data[off_b..off_b + row_bytes],
                    "F26Dot6 quantized swash rasterization must be 100% bit-identical regardless of ULP call order!"
                );
            }
        }

        // 3. Sub-`1/64` animated font-size sweep (`16.0 ..= 17.0` in 1,000 micro-steps) bounds cache to `<= 65` entries:
        let mut anim_fonts = FontSystem::with_system_fonts();
        for step in 0..=1000 {
            let sz = 16.0 + (step as f32) * 0.001;
            let _ = anim_fonts.rasterize_glyph(
                anim_fonts.ui_font,
                gid,
                sz,
                0.0,
                AntialiasMode::Grayscale,
            );
        }
        assert!(
            anim_fonts.atlas.cached_glyph_count() <= 65,
            "1px animated sweep must quantize to <= 65 F26Dot6 entries, got {}",
            anim_fonts.atlas.cached_glyph_count()
        );

        // 4. Fractional `phys_font_size` hinting & stem contrast audit (`13.5 * 1.25 = 16.875 px`, `18.6667 px`, `24.1875 px`):
        for &phys_sz in &[
            13.5 * 1.125,
            13.5 * 1.25,
            14.0 * (160.0 / 120.0),
            13.5 * 1.5,
            13.5 * 1.625,
            13.5 * 1.75,
            13.5 * 2.25,
        ] {
            for &sg in &para.glyphs {
                let mut phase_energies = [0u32; 4];
                for (p_idx, sub_x) in [0.0_f32, 0.25, 0.50, 0.75].into_iter().enumerate() {
                    let reg = fonts.rasterize_glyph(
                        sg.font_id,
                        sg.glyph_id,
                        phys_sz,
                        sub_x,
                        AntialiasMode::SubpixelLcd,
                    );
                    if sg.font_id != FontId::FALLBACK && reg.w > 0 && reg.h > 0 && reg.plane == Some(AtlasPlane::Rgba8) {
                        let stride = (fonts.atlas.width as usize) * 4;
                        let mut peak_alpha = 0u8;
                        let mut total_alpha = 0u32;
                        for r in 0..(reg.h as usize) {
                            let start = (reg.y as usize + r) * stride + (reg.x as usize) * 4;
                            let row_px = &fonts.atlas.rgba8_data[start..start + (reg.w as usize) * 4];
                            let mut sr = 0i32;
                            let mut sg_c = 0i32;
                            let mut sb = 0i32;
                            for px in row_px.as_chunks::<4>().0 {
                                sr += px[0] as i32;
                                sg_c += px[1] as i32;
                                sb += px[2] as i32;
                                peak_alpha = peak_alpha.max(px[3]);
                                total_alpha += px[3] as u32;
                            }
                            assert!(
                                (sr - sg_c).abs() <= (reg.w as i32) && (sg_c - sb).abs() <= (reg.w as i32),
                                "Chromatic imbalance at fractional phys_sz={phys_sz}: R={sr}, G={sg_c}, B={sb}"
                            );
                        }
                        // A ~1.2px stem straddling a half-pixel boundary (`sub_x = 0.50`) splits ~0.68/0.52 (`~173`),
                        // while preserving crisp stem contrast (`>= 155` at any phase, `>= 190` above `18px`) and
                        // invariant integrated optical weight across all 4 subpixel phases!
                        let min_peak = if phys_sz >= 18.0 && p_idx == 0 { 190 } else { 155 };
                        assert!(
                            peak_alpha >= min_peak,
                            "Weak stem contrast ({peak_alpha} < {min_peak}) at fractional phys_sz={phys_sz}, phase={sub_x}"
                        );
                        phase_energies[p_idx] = total_alpha;
                    }
                }
                if phase_energies[0] > 0 {
                    let e0 = phase_energies[0] as f32;
                    for &ep in &phase_energies[1..] {
                        assert!(
                            ((ep as f32) - e0).abs() <= e0 * 0.04 + 12.0,
                            "Subpixel phase optical weight blotching at phys_sz={phys_sz}: {phase_energies:?}"
                        );
                    }
                }
            }
        }
    }
}
