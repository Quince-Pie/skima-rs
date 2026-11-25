//! Perceptual OKLab / OKLCH, Linear sRGB color algebra, CSS Color 4 premultiplied interpolation,
//! perceptual chroma-preserving sRGB gamut mapping, and WCAG 2.1 / APCA-W3 contrast helpers.
//! Avoids muddy gamma-corrupted sRGB midpoints, transparent-endpoint hue/lightness pollution,
//! and out-of-gamut hue shifts in gradients, themes, hover states, and antialiasing.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Color(pub u32); // Packed 0xRRGGBBAA in sRGB

#[inline]
const fn clamp01(x: f32) -> f32 {
    if x.is_nan() || x <= 0.0 {
        0.0 // Canonical `+0.0` (eliminates `-0.0` sign bit too)
    } else if x >= 1.0 {
        1.0
    } else {
        x
    }
}

#[inline]
const fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        if x == 0.0 { 0.0 } else { x }
    } else {
        0.0
    }
}

#[inline]
const fn wrap_hue_360(hue_deg: f32) -> f32 {
    if !hue_deg.is_finite() {
        return 0.0;
    }
    let r = hue_deg % 360.0;
    let h = if r < 0.0 { r + 360.0 } else { r };
    // Guard against small negative `r` (e.g. `-1e-9`) rounding `r + 360.0` up to `360.0` in IEEE-754 `f32`:
    if h >= 360.0 || h == 0.0 {
        0.0
    } else {
        h
    }
}

impl Color {
    pub const TRANSPARENT: Self = Self(0x0000_0000);
    pub const BLACK: Self = Self(0x0000_00FF);
    pub const WHITE: Self = Self(0xFFFF_FFFF);

    #[inline]
    pub const fn rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32))
    }

    #[inline]
    pub const fn rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::rgba8(r, g, b, 255)
    }

    #[inline]
    pub const fn hex(rgba: u32) -> Self {
        Self(rgba)
    }

    #[inline]
    pub const fn r(self) -> u8 {
        (self.0 >> 24) as u8
    }

    #[inline]
    pub const fn g(self) -> u8 {
        (self.0 >> 16) as u8
    }

    #[inline]
    pub const fn b(self) -> u8 {
        (self.0 >> 8) as u8
    }

    #[inline]
    pub const fn a(self) -> u8 {
        self.0 as u8
    }

    #[inline]
    pub const fn with_alpha_u8(self, a: u8) -> Self {
        Self((self.0 & 0xFFFF_FF00) | (a as u32))
    }

    #[inline]
    pub const fn channels(self) -> [u8; 4] {
        [
            (self.0 >> 24) as u8,
            (self.0 >> 16) as u8,
            (self.0 >> 8) as u8,
            self.0 as u8,
        ]
    }

    #[inline]
    pub fn with_alpha(self, alpha: f32) -> Self {
        let a = (clamp01(alpha) * 255.0 + 0.5) as u8;
        self.with_alpha_u8(a)
    }

    #[inline]
    pub const fn to_srgb_f32(self) -> [f32; 4] {
        let [r, g, b, a] = self.channels();
        [
            (r as f32) * (1.0 / 255.0),
            (g as f32) * (1.0 / 255.0),
            (b as f32) * (1.0 / 255.0),
            (a as f32) * (1.0 / 255.0),
        ]
    }

    #[inline]
    pub fn to_linear_rgba(self) -> [f32; 4] {
        let [r, g, b, a] = self.to_srgb_f32();
        [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a]
    }

    #[inline]
    pub fn from_linear_rgba(lin: [f32; 4]) -> Self {
        let r = (linear_to_srgb(clamp01(lin[0])) * 255.0).round() as u8;
        let g = (linear_to_srgb(clamp01(lin[1])) * 255.0).round() as u8;
        let b = (linear_to_srgb(clamp01(lin[2])) * 255.0).round() as u8;
        let a = (clamp01(lin[3]) * 255.0).round() as u8;
        Self::rgba8(r, g, b, a)
    }

    /// Construct directly from perceptual OKLCH (`l` in `0..=1`, `c` in `0..=0.4`, `hue_deg` in degrees)
    /// using perceptual chroma-reduction gamut mapping so out-of-gamut colors preserve exact hue and lightness.
    #[inline]
    pub fn oklch(l: f32, c: f32, hue_deg: f32, alpha: f32) -> Self {
        Oklch::new(l, c, hue_deg, alpha).to_color()
    }

    /// Convert to cylindrical perceptual `Oklch` (`l` in `0..=1`, `c >= 0`, `h` in `0..360` degrees).
    #[inline]
    pub fn to_oklch(self) -> Oklch {
        Oklch::from_color(self)
    }

    /// Perceptually uniform CSS Color 4 premultiplied-alpha interpolation in Björn Ottosson's OKLab space.
    /// Transparent or low-alpha endpoints never pollute the hue, chroma, or lightness of opaque colors.
    #[inline]
    pub fn mix_oklab(self, other: Self, t: f32) -> Self {
        let tc = clamp01(t);
        if tc <= 0.0 {
            return if self.a() == 0 { Self::TRANSPARENT } else { self };
        }
        if tc >= 1.0 {
            return if other.a() == 0 { Self::TRANSPARENT } else { other };
        }
        if self == other {
            return if self.a() == 0 { Self::TRANSPARENT } else { self };
        }
        let a = Oklab::from_color(self);
        let b = Oklab::from_color(other);
        a.mix_premultiplied(b, tc).gamut_clip_srgb().to_color()
    }

    /// Perceptually shift lightness by `delta_l` (e.g., `+0.06` for hover highlight, `-0.04` for active press),
    /// preserving hue via perceptual sRGB gamut mapping.
    #[inline]
    pub fn shift_lightness(self, delta_l: f32) -> Self {
        let delta = finite_or_zero(delta_l);
        if delta == 0.0 {
            return self;
        }
        let mut lab = Oklab::from_color(self);
        lab.l = clamp01(lab.l + delta);
        lab.gamut_clip_srgb().to_color()
    }

    /// Exact Porter-Duff `source-over` compositing of `self` over `backdrop` in linear sRGB space.
    #[inline]
    pub fn composite_over(self, backdrop: Self) -> Self {
        let fg_a8 = self.a();
        if fg_a8 == 255 {
            return self;
        }
        if fg_a8 == 0 {
            return if backdrop.a() == 0 {
                Self::TRANSPARENT
            } else {
                backdrop
            };
        }
        let fg = self.to_linear_rgba();
        let bg = backdrop.to_linear_rgba();
        let inv_fg_a = 1.0 - fg[3];
        let out_a = fg[3] + bg[3] * inv_fg_a;
        if out_a <= 1e-7 {
            return Self::TRANSPARENT;
        }
        let inv_out_a = 1.0 / out_a;
        Self::from_linear_rgba([
            (fg[0] * fg[3] + bg[0] * bg[3] * inv_fg_a) * inv_out_a,
            (fg[1] * fg[3] + bg[1] * bg[3] * inv_fg_a) * inv_out_a,
            (fg[2] * fg[3] + bg[2] * bg[3] * inv_fg_a) * inv_out_a,
            out_a,
        ])
    }

    /// WCAG 2.1 / ITU-R BT.709 relative luminance `Y` in `0.0..=1.0` (`BLACK = 0.0`, `WHITE = 1.0`).
    #[inline]
    pub fn relative_luminance(self) -> f32 {
        let [r, g, b, _] = self.to_linear_rgba();
        (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0)
    }

    /// WCAG 2.1 contrast ratio `(L_max + 0.05) / (L_min + 0.05)` in `1.0..=21.0`.
    /// Symmetric: `a.wcag_contrast_ratio(b) == b.wcag_contrast_ratio(a)`.
    #[inline]
    pub fn wcag_contrast_ratio(self, other: Self) -> f32 {
        let y0 = self.relative_luminance();
        let y1 = other.relative_luminance();
        let (hi, lo) = if y0 >= y1 { (y0, y1) } else { (y1, y0) };
        (hi + 0.05) / (lo + 0.05)
    }

    /// APCA-W3 (`0.0.98G-4g`) perceptual lightness contrast `Lc` of foreground `self` over `background`.
    /// Returns signed `Lc` in `[-108.0, +106.1]`:
    /// - Positive `Lc > 0`: dark foreground on light background (e.g., `BLACK` on `WHITE` ≈ `+106.0`).
    /// - Negative `Lc < 0`: light foreground on dark background (e.g., `WHITE` on `BLACK` ≈ `-107.9`).
    /// - Automatically composites translucent foreground (`alpha < 255`) over `background`.
    #[inline]
    pub fn apca_contrast(self, background: Self) -> f32 {
        let fg = if self.channels()[3] < 255 {
            self.composite_over(background)
        } else {
            self
        };
        let y_txt = apca_soft_clamp(apca_screen_luminance(fg));
        let y_bg = apca_soft_clamp(apca_screen_luminance(background));
        if (y_bg - y_txt).abs() < 0.0005 {
            return 0.0;
        }
        if y_bg > y_txt {
            let sapc = (y_bg.powf(0.56) - y_txt.powf(0.57)) * 1.14;
            if sapc < 0.1 {
                0.0
            } else {
                (sapc - 0.027) * 100.0
            }
        } else {
            let sapc = (y_bg.powf(0.65) - y_txt.powf(0.62)) * 1.14;
            if sapc > -0.1 {
                0.0
            } else {
                (sapc + 0.027) * 100.0
            }
        }
    }

    /// Choose crisp dark (`rgb8(18, 20, 28)`) or pure `WHITE` foreground text maximizing `|APCA Lc|` over `self`.
    #[inline]
    pub fn readable_text_color(self) -> Self {
        let dark = Self::rgb8(18, 20, 28);
        if dark.apca_contrast(self).abs() >= Self::WHITE.apca_contrast(self).abs() {
            dark
        } else {
            Self::WHITE
        }
    }
}

#[inline]
fn apca_screen_luminance(c: Color) -> f32 {
    let [r, g, b, _] = c.to_srgb_f32();
    0.212_672_9 * r.powf(2.4) + 0.715_152_2 * g.powf(2.4) + 0.072_175_0 * b.powf(2.4)
}

#[inline]
fn apca_soft_clamp(y: f32) -> f32 {
    let yc = y.max(0.0);
    if yc < 0.022 {
        yc + (0.022 - yc).powf(1.414)
    } else {
        yc
    }
}

/// Cylindrical perceptual OKLCH color (`l` lightness `0..=1`, `c` chroma `>= 0`, `h` hue `0..360` deg).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oklch {
    pub l: f32,
    pub c: f32,
    pub h: f32,
    pub alpha: f32,
}

impl Oklch {
    pub const TRANSPARENT: Self = Self {
        l: 0.0,
        c: 0.0,
        h: 0.0,
        alpha: 0.0,
    };
    pub const BLACK: Self = Self {
        l: 0.0,
        c: 0.0,
        h: 0.0,
        alpha: 1.0,
    };
    pub const WHITE: Self = Self {
        l: 1.0,
        c: 0.0,
        h: 0.0,
        alpha: 1.0,
    };

    #[inline]
    pub const fn new(l: f32, c: f32, hue_deg: f32, alpha: f32) -> Self {
        let clean_c = if c.is_finite() && c > 0.0 { c } else { 0.0 };
        Self {
            l: clamp01(l),
            c: clean_c,
            h: wrap_hue_360(hue_deg),
            alpha: clamp01(alpha),
        }
    }

    #[inline]
    pub fn from_oklab(lab: Oklab) -> Self {
        let a = finite_or_zero(lab.a);
        let b = finite_or_zero(lab.b);
        let raw_c = a.hypot(b);
        let c = if raw_c.is_finite() && raw_c > 0.0 {
            raw_c
        } else {
            0.0
        };
        let h = if c <= 1e-6 {
            0.0
        } else {
            wrap_hue_360(b.atan2(a).to_degrees())
        };
        Self {
            l: clamp01(lab.l),
            c,
            h,
            alpha: clamp01(lab.alpha),
        }
    }

    #[inline]
    pub fn to_oklab(self) -> Oklab {
        let c = if self.c.is_finite() && self.c > 0.0 {
            self.c
        } else {
            0.0
        };
        let rad = wrap_hue_360(self.h).to_radians();
        let (sin_h, cos_h) = rad.sin_cos();
        Oklab {
            l: clamp01(self.l),
            a: finite_or_zero(c * cos_h),
            b: finite_or_zero(c * sin_h),
            alpha: clamp01(self.alpha),
        }
    }

    #[inline]
    pub fn from_color(c: Color) -> Self {
        Self::from_oklab(Oklab::from_color(c))
    }

    /// Convert to `Color` using perceptual chroma-preserving sRGB gamut mapping.
    #[inline]
    pub fn to_color(self) -> Color {
        self.to_oklab().gamut_clip_srgb().to_color()
    }
}

/// Cartesian perceptual Björn Ottosson OKLab color (`l` in `0..=1`, `a`/`b` opponent axes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
    pub alpha: f32,
}

impl Oklab {
    pub const TRANSPARENT: Self = Self {
        l: 0.0,
        a: 0.0,
        b: 0.0,
        alpha: 0.0,
    };
    pub const BLACK: Self = Self {
        l: 0.0,
        a: 0.0,
        b: 0.0,
        alpha: 1.0,
    };
    pub const WHITE: Self = Self {
        l: 1.0,
        a: 0.0,
        b: 0.0,
        alpha: 1.0,
    };

    #[inline]
    pub const fn new(l: f32, a: f32, b: f32, alpha: f32) -> Self {
        Self {
            l: clamp01(l),
            a: finite_or_zero(a),
            b: finite_or_zero(b),
            alpha: clamp01(alpha),
        }
    }

    #[inline]
    pub fn from_linear_rgb(lin: [f32; 3], alpha: f32) -> Self {
        let r = if lin[0].is_finite() && lin[0] > 0.0 {
            lin[0]
        } else {
            0.0
        };
        let g = if lin[1].is_finite() && lin[1] > 0.0 {
            lin[1]
        } else {
            0.0
        };
        let b = if lin[2].is_finite() && lin[2] > 0.0 {
            lin[2]
        } else {
            0.0
        };
        let l_c = 0.412_221_47 * r + 0.536_332_54 * g + 0.051_445_993 * b;
        let m_c = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
        let s_c = 0.088_302_46 * r + 0.281_718_84 * g + 0.629_978_7 * b;
        let l = if l_c.is_finite() && l_c > 0.0 {
            l_c.cbrt()
        } else {
            0.0
        };
        let m = if m_c.is_finite() && m_c > 0.0 {
            m_c.cbrt()
        } else {
            0.0
        };
        let s = if s_c.is_finite() && s_c > 0.0 {
            s_c.cbrt()
        } else {
            0.0
        };
        Self {
            l: finite_or_zero(0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s),
            a: finite_or_zero(1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s),
            b: finite_or_zero(0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s),
            alpha: clamp01(alpha),
        }
    }

    #[inline]
    pub fn from_color(c: Color) -> Self {
        let lin = c.to_linear_rgba();
        Self::from_linear_rgb([lin[0], lin[1], lin[2]], lin[3])
    }

    #[inline]
    pub fn from_oklch(lch: Oklch) -> Self {
        lch.to_oklab()
    }

    #[inline]
    pub fn to_oklch(self) -> Oklch {
        Oklch::from_oklab(self)
    }

    /// Raw unclamped linear sRGB conversion (values outside `0.0..=1.0` indicate out-of-sRGB-gamut).
    #[inline]
    pub const fn to_linear_rgb_unclamped(self) -> [f32; 3] {
        let l = finite_or_zero(self.l);
        let a = finite_or_zero(self.a);
        let b = finite_or_zero(self.b);
        let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
        let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
        let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
        let lc = l_ * l_ * l_;
        let mc = m_ * m_ * m_;
        let sc = s_ * s_ * s_;
        [
            4.076_741_7 * lc - 3.307_711_6 * mc + 0.230_969_94 * sc,
            -1.268_438 * lc + 2.609_757_4 * mc - 0.341_319_38 * sc,
            -0.004_196_086_3 * lc - 0.703_418_6 * mc + 1.707_614_7 * sc,
        ]
    }

    /// Check whether this OKLab color lies within the displayable linear sRGB `[0, 1]^3` cube.
    #[inline]
    pub const fn is_in_srgb_gamut(self) -> bool {
        if !self.l.is_finite()
            || !self.a.is_finite()
            || !self.b.is_finite()
            || !self.alpha.is_finite()
        {
            return false;
        }
        let [r, g, b] = self.to_linear_rgb_unclamped();
        const EPS: f32 = 1e-5;
        r >= -EPS && r <= 1.0 + EPS && g >= -EPS && g <= 1.0 + EPS && b >= -EPS && b <= 1.0 + EPS
    }

    /// Perceptual chroma-preserving sRGB gamut clip: keeps `L` and hue angle `atan2(b, a)` invariant
    /// while scaling chroma `(a, b)` onto the exact sRGB gamut boundary when out of gamut.
    /// Pre-normalizes extreme `(a, b)` onto the `C = 0.40` cylinder (bounding the entire `[0, 1]^3` sRGB cube)
    /// so arbitrary finite chroma (`100.0`, `1e30`) never overflows `l^3` or collapses to gray, followed by
    /// 20-step trig-free bisection (`< 4e-7` OKLab chroma error).
    #[inline]
    pub fn gamut_clip_srgb(self) -> Self {
        let l = clamp01(self.l);
        let alpha = clamp01(self.alpha);
        if l <= 0.0 {
            return Self {
                l: 0.0,
                a: 0.0,
                b: 0.0,
                alpha,
            };
        }
        if l >= 1.0 {
            return Self {
                l: 1.0,
                a: 0.0,
                b: 0.0,
                alpha,
            };
        }
        let mut a = finite_or_zero(self.a);
        let mut b = finite_or_zero(self.b);
        let candidate = Self { l, a, b, alpha };
        if candidate.is_in_srgb_gamut() {
            return candidate;
        }

        // Bound `(a, b)` onto the `C = 0.40` outer cylinder (`max_sRGB C ≈ 0.322 < 0.40`) without changing hue:
        let c = a.hypot(b);
        if !c.is_finite() || c <= 0.0 {
            return Self {
                l,
                a: 0.0,
                b: 0.0,
                alpha,
            };
        }
        if c > 0.40 {
            let scale = 0.40 / c;
            a *= scale;
            b *= scale;
        }

        let mut lo = 0.0_f32;
        let mut hi = 1.0_f32;
        for _ in 0..20 {
            let mid = (lo + hi) * 0.5;
            let probe = Self {
                l,
                a: a * mid,
                b: b * mid,
                alpha,
            };
            let [r, g, bl] = probe.to_linear_rgb_unclamped();
            let unit = 0.0..=1.0;
            if unit.contains(&r) && unit.contains(&g) && unit.contains(&bl) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Self {
            l,
            a: a * lo,
            b: b * lo,
            alpha,
        }
    }

    /// CSS Color 4 premultiplied-alpha linear interpolation in OKLab space.
    /// Sanitizes `NaN`/`Inf` components so `alpha == 0.0` endpoints never trigger `0.0 * Inf = NaN`.
    #[inline]
    pub const fn mix_premultiplied(self, other: Self, t: f32) -> Self {
        let t = clamp01(t);
        let a0 = clamp01(self.alpha);
        let a1 = clamp01(other.alpha);
        let out_a = a0 + (a1 - a0) * t;
        if out_a <= 1e-7 {
            return Self::TRANSPARENT;
        }
        let w0 = a0 * (1.0 - t);
        let w1 = a1 * t;
        let inv_a = 1.0 / out_a;
        let l0 = finite_or_zero(self.l);
        let oa0 = finite_or_zero(self.a);
        let ob0 = finite_or_zero(self.b);
        let l1 = finite_or_zero(other.l);
        let oa1 = finite_or_zero(other.a);
        let ob1 = finite_or_zero(other.b);
        Self {
            l: clamp01((l0 * w0 + l1 * w1) * inv_a),
            a: finite_or_zero((oa0 * w0 + oa1 * w1) * inv_a),
            b: finite_or_zero((ob0 * w0 + ob1 * w1) * inv_a),
            alpha: clamp01(out_a),
        }
    }

    #[inline]
    pub const fn to_linear_rgb(self) -> [f32; 3] {
        let [r, g, b] = self.to_linear_rgb_unclamped();
        [clamp01(r), clamp01(g), clamp01(b)]
    }

    #[inline]
    pub fn to_color(self) -> Color {
        let rgb = self.to_linear_rgb();
        Color::from_linear_rgba([rgb[0], rgb[1], rgb[2], self.alpha])
    }
}

#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    let x = clamp01(c);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    let x = clamp01(c);
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_oklab_oklch_roundtrip_exactness() {
        let samples = [
            Color::BLACK,
            Color::WHITE,
            Color::rgb8(255, 0, 0),
            Color::rgb8(0, 255, 0),
            Color::rgb8(0, 0, 255),
            Color::rgb8(45, 95, 225),
            Color::rgb8(180, 55, 210),
            Color::rgb8(128, 128, 128),
            Color::rgba8(64, 196, 140, 173),
        ];
        for c in samples {
            let lin = c.to_linear_rgba();
            let rt_lin = Color::from_linear_rgba(lin);
            assert_eq!(c, rt_lin, "sRGB <-> linear roundtrip failed for {c:?}");

            let lab = Oklab::from_color(c);
            assert!(lab.is_in_srgb_gamut(), "sRGB color must be in gamut: {c:?}");
            let rt_lab = lab.to_color();
            for (ch0, ch1) in c.channels().into_iter().zip(rt_lab.channels()) {
                assert!(
                    ch0.abs_diff(ch1) <= 1,
                    "Oklab roundtrip drifted > 1 LSB: {c:?} -> {rt_lab:?}"
                );
            }

            let lch = c.to_oklch();
            let rt_lch = lch.to_color();
            for (ch0, ch1) in c.channels().into_iter().zip(rt_lch.channels()) {
                assert!(
                    ch0.abs_diff(ch1) <= 1,
                    "Oklch roundtrip drifted > 1 LSB: {c:?} -> {rt_lch:?}"
                );
            }
        }

        // Verify neutral gray axis has ~0 chroma and canonical 0 hue
        let gray_lch = Color::rgb8(120, 120, 120).to_oklch();
        assert!(gray_lch.c < 1e-5);
        assert_eq!(gray_lch.h, 0.0);
    }

    #[test]
    fn premultiplied_mix_oklab_never_pollutes_hue_or_lightness_with_transparent_endpoints() {
        //Even a "poisoned" transparent red `rgba(255, 0, 0, 0)` or `TRANSPARENT` black `(0,0,0,0)`
        // must never darken or tint `WHITE` or chromatic colors when mixed!
        for transparent in [Color::TRANSPARENT, Color::rgba8(255, 0, 0, 0), Color::rgba8(0, 255, 0, 0)] {
            let mid_white = Color::WHITE.mix_oklab(transparent, 0.5);
            assert_eq!(mid_white.channels(), [255, 255, 255, 128]);

            let rev_white = transparent.mix_oklab(Color::WHITE, 0.25);
            assert_eq!(rev_white.channels(), [255, 255, 255, 64]);

            let sky = Color::rgb8(60, 165, 250);
            for t in [0.1, 0.25, 0.5, 0.75, 0.9] {
                let mixed = sky.mix_oklab(transparent, t);
                let [r, g, b, a] = mixed.channels();
                let [sr, sg, sb, _] = sky.channels();
                assert!(r.abs_diff(sr) <= 1 && g.abs_diff(sg) <= 1 && b.abs_diff(sb) <= 1);
                let expected_a = ((1.0 - t) * 255.0).round() as u8;
                assert!(a.abs_diff(expected_a) <= 1);
            }
        }

        assert_eq!(
            Color::TRANSPARENT.mix_oklab(Color::rgba8(200, 100, 50, 0), 0.5),
            Color::TRANSPARENT
        );
    }

    #[test]
    fn oklch_gamut_mapping_preserves_perceptual_hue_across_full_wheel() {
        // High-chroma blue at L=0.75 previously shifted >20 degrees toward cyan under naive RGB clamp.
        for hue_deg in (0..360).step_by(15) {
            let h = hue_deg as f32;
            let raw = Oklch::new(0.72, 0.35, h, 1.0).to_oklab();
            let clipped = raw.gamut_clip_srgb();
            assert!(clipped.is_in_srgb_gamut());
            assert!((clipped.l - 0.72).abs() < 1e-5);
            if clipped.a.hypot(clipped.b) > 1e-3 {
                let clipped_h = clipped.to_oklch().h;
                let diff = (clipped_h - h + 180.0).rem_euclid(360.0) - 180.0;
                assert!(
                    diff.abs() < 0.05,
                    "Gamut clip shifted hue at {h} deg -> {clipped_h} deg (diff {diff})"
                );
            }
            // Verify final quantized 8-bit sRGB color stays close in hue
            let col = Color::oklch(0.72, 0.35, h, 1.0);
            let col_h = col.to_oklch().h;
            let q_diff = (col_h - h + 180.0).rem_euclid(360.0) - 180.0;
            assert!(
                q_diff.abs() < 4.0,
                "8-bit OKLCH color hue drifted too far at {h} deg -> {col_h} deg"
            );
        }
    }

    #[test]
    fn wcag_and_apca_luminance_and_contrast_invariants() {
        assert_eq!(Color::BLACK.relative_luminance(), 0.0);
        assert!((Color::WHITE.relative_luminance() - 1.0).abs() < 1e-6);

        let cr_bw = Color::WHITE.wcag_contrast_ratio(Color::BLACK);
        let cr_wb = Color::BLACK.wcag_contrast_ratio(Color::WHITE);
        assert!((cr_bw - 21.0).abs() < 1e-4);
        assert_eq!(cr_bw, cr_wb);
        assert!((Color::WHITE.wcag_contrast_ratio(Color::WHITE) - 1.0).abs() < 1e-6);

        // APCA-W3 canonical polarity & extrema checks:
        let lc_bow = Color::BLACK.apca_contrast(Color::WHITE);
        let lc_wob = Color::WHITE.apca_contrast(Color::BLACK);
        assert!(
            (lc_bow - 106.04).abs() < 0.5,
            "Expected BoW APCA ~ +106.04, got {lc_bow}"
        );
        assert!(
            (lc_wob - (-107.9)).abs() < 0.5,
            "Expected WoB APCA ~ -107.9, got {lc_wob}"
        );
        assert_eq!(Color::rgb8(90, 100, 110).apca_contrast(Color::rgb8(90, 100, 110)), 0.0);

        // Readable text color selection on light vs dark surfaces:
        assert_eq!(Color::WHITE.readable_text_color(), Color::rgb8(18, 20, 28));
        assert_eq!(Color::BLACK.readable_text_color(), Color::WHITE);
        assert_eq!(Color::rgb8(250, 225, 60).readable_text_color(), Color::rgb8(18, 20, 28));

        // NaN resilience & negative/wrapped hue + composite_over + shift_lightness bounds:
        let nan_col = Color::oklch(f32::NAN, f32::INFINITY, f32::NAN, f32::NAN);
        assert_eq!(nan_col, Color::TRANSPARENT);
        assert_eq!(Color::oklch(0.65, 0.16, -90.0, 1.0), Color::oklch(0.65, 0.16, 270.0, 1.0));
        assert_eq!(Color::rgb8(80, 120, 200).shift_lightness(5.0), Color::WHITE);
        assert_eq!(Color::rgb8(80, 120, 200).shift_lightness(-5.0), Color::BLACK);

        let semi = Color::rgba8(255, 255, 255, 128);
        let comp = semi.composite_over(Color::BLACK);
        assert!((semi.apca_contrast(Color::BLACK) - comp.apca_contrast(Color::BLACK)).abs() < 0.2);
    }

    #[test]
    fn const_constructors_extreme_chroma_gamut_clip_and_nan_inf_resilience() {
        // 1. Compile-time `const` evaluation across `Color`, `Oklab`, and `Oklch`:
        const C_ACCENT: Color = Color::rgb8(64, 140, 255).with_alpha_u8(200);
        const C_SRGB: [f32; 4] = C_ACCENT.to_srgb_f32();
        const LAB_A: Oklab = Oklab::new(0.65, -0.05, 0.12, 1.0);
        const LAB_MIX: Oklab = LAB_A.mix_premultiplied(Oklab::WHITE, 0.5);
        const LCH_WRAP: Oklch = Oklch::new(0.7, 0.15, -1e-9, 1.0);
        assert_eq!(
            [C_ACCENT.r(), C_ACCENT.g(), C_ACCENT.b(), C_ACCENT.a()],
            [64, 140, 255, 200]
        );
        assert!((C_SRGB[0] - 64.0 / 255.0).abs() < 1e-6);
        assert!(LAB_MIX.is_in_srgb_gamut());
        // Sub-ULP negative hue `-1e-9` must wrap strictly into `[0.0, 360.0)` (never `360.0` or `-0.0`!):
        assert!(LCH_WRAP.h >= 0.0 && LCH_WRAP.h < 360.0 && !LCH_WRAP.h.is_sign_negative());

        // 2. Exhaustive 256-level `u8` sRGB <-> linear roundtrip:
        for v in 0..=255u8 {
            let c = Color::rgba8(v, 255 - v, v / 2, v);
            assert_eq!(Color::from_linear_rgba(c.to_linear_rgba()), c);
        }

        // 3. Extreme finite chroma (`c = 100.0` and `c = 1e30`) must clip onto the colorful sRGB boundary
        //    at exact hue rather than overflowing `l^3` or collapsing to gray `(a=0, b=0)`!
        for huge_c in [10.0_f32, 100.0, 1e20, 1e30] {
            let clipped = Oklch::new(0.65, huge_c, 32.0, 1.0).to_oklab().gamut_clip_srgb();
            assert!(clipped.is_in_srgb_gamut());
            let out_lch = clipped.to_oklch();
            assert!(
                out_lch.c > 0.14,
                "Huge chroma {huge_c} collapsed to gray: {out_lch:?}"
            );
            assert!((out_lch.h - 32.0).abs() < 0.05);
        }

        // 4. `NaN`/`Inf` in transparent `Oklab` endpoints must never trigger `0.0 * Inf = NaN`:
        let poisoned_transparent = Oklab {
            l: f32::NAN,
            a: f32::INFINITY,
            b: f32::NEG_INFINITY,
            alpha: 0.0,
        };
        let mixed = poisoned_transparent.mix_premultiplied(Oklab::WHITE, 0.5);
        assert_eq!(mixed.l, 1.0);
        assert_eq!(mixed.a, 0.0);
        assert_eq!(mixed.b, 0.0);
        assert!((mixed.alpha - 0.5).abs() < 1e-6);

        let nan_lch = Oklch::from_oklab(poisoned_transparent);
        assert!(nan_lch.l.is_finite() && nan_lch.c.is_finite() && nan_lch.h.is_finite());
        assert!(!poisoned_transparent.is_in_srgb_gamut());
    }
}
