//! Display Physics, Ergonomic Visual-Acuity Modeling, and Automatic Fractional Scaling.
//!
//! Resolves optimal HiDPI fractional scale factors across handhelds, laptops, standard and ultrawide
//! desktop monitors, portrait workstations, and large-format displays while preserving subpixel sharpness,
//! IEEE-754 binary floating-point exactness, and Wayland `wp_fractional_scale_v1` (`1/120`) protocol parity.
//!
//! # Physical & Algorithmic Foundations
//!
//! 1. **Angular Visual Acuity (`PPD` vs W3C CSS Reference Pixel):**
//!    Human foveal legibility depends on subtended visual angle rather than raw `PPI / 96.0` alone.
//!    A W3C CSS reference pixel (`1px` at `96 PPI` viewed at `28 in = 71.12 cm`) subtends
//!    `theta_0 = 2 * atan(1 / 5376) ≈ 0.0213149 deg`, corresponding to `PPD_0 ≈ 46.9156 px/deg`.
//!    Given physical pixel pitch `p_mm` and ergonomic viewing distance `d_cm`, the display's native
//!    angular density is `PPD = 20 * d_cm * tan(0.5 deg) / p_mm`, yielding raw acuity scale
//!    `s_raw = PPD / PPD_0`.
//!
//! 2. **Orientation-Invariant, Aspect-Aware Viewing Distance `d(W_mm, H_mm)`:**
//!    Using `short_mm = min(W_mm, H_mm)` and `16:9` height-equivalent diagonal `D_h` keeps `d` invariant
//!    under 90-degree portrait rotation and prevents `21:9` / `32:9` super-ultrawide desk monitors
//!    (e.g. `49" 5120x1440`, `336 mm` tall — two `27" 1440p` panels side by side) from misclassifying
//!    as living-room televisions (`55"+ 16:9`, `685+ mm` tall).
//!
//! 3. **Wayland `1/120` & IEEE-754 `15k/120` Dyadic Token Exclusivity:**
//!    Wayland `wp_fractional_scale_v1` encodes scale as `N_120 / 120`. Because `120 = 8 * 15`, every
//!    `1/8` dyadic multiple `k / 8` equals `(15 * k) / 120` — simultaneously:
//!    - an **exact integer numerator** in `wp_fractional_scale_v1`,
//!    - an **exact terminating IEEE-754 `f32` binary float** (`0` mantissa rounding error),
//!    - an **exact integer `F26Dot6` (`1/64 px`) font rasterization key** for `0.5px`/`1.0px` font sizes,
//!    - and aligned with power-of-two UI design tokens (`{1, 2, 4, 8}px`). By contrast, any reduced
//!      fraction `p / q` with odd prime factors (`3 | q` or `5 | q`, such as `4/3 = 160/120`) has an
//!      infinite repeating binary mantissa in IEEE-754 `f32` and maps zero `2^m` tokens to integer pixels.
//!
//! 4. **Multi-Objective Scale Optimization & Hysteresis:**
//!    Minimizes symmetric log-acuity error `(ln(s / s_raw))^2` jointly with binary design-grid harmony,
//!    smooth `1.0x` native pixel-identity deadzone protection (keeping `24" 1080p`, `27" 1440p 109 PPI`,
//!    `34"/38"/49"` ultrawides, and `42" 4K` desk OLEDs at razor-sharp `1:1`), fractional integer
//!    minimum logical workspace guards (`>= 1024x720`), and optional `previous_120ths` deadband
//!    hysteresis.

/// Fixed denominator used by the Wayland `wp_fractional_scale_v1` protocol (`1 / 120` units).
pub const WAYLAND_SCALE_DENOM: u32 = 120;

/// Minimum supported explicit UI scale factor (`60 / 120 = 0.50x`).
pub const MIN_SCALE: f32 = 0.5;

/// Maximum supported explicit UI scale factor (`480 / 120 = 4.00x`).
pub const MAX_SCALE: f32 = 4.0;

/// Exact W3C CSS reference pixel visual angle in degrees: `2 * atan(1 / (2 * 96 * 28))`.
pub const CSS_REFERENCE_PIXEL_DEG: f32 = 0.021_314_884;

/// Exact W3C CSS reference angular density in pixels per degree (`1.0 / CSS_REFERENCE_PIXEL_DEG`).
pub const CSS_REFERENCE_PPD: f32 = 46.915_573;

/// Common Wayland-exact scale constants (`N / 120`).
pub const SCALE_0_75: f32 = 0.75;
pub const SCALE_0_875: f32 = 0.875;
pub const SCALE_1_00: f32 = 1.0;
pub const SCALE_1_125: f32 = 1.125;
pub const SCALE_1_25: f32 = 1.25;
pub const SCALE_1_33: f32 = 160.0 / 120.0;
pub const SCALE_1_375: f32 = 1.375;
pub const SCALE_1_50: f32 = 1.5;
pub const SCALE_1_625: f32 = 1.625;
pub const SCALE_1_67: f32 = 200.0 / 120.0;
pub const SCALE_1_75: f32 = 1.75;
pub const SCALE_1_875: f32 = 1.875;
pub const SCALE_2_00: f32 = 2.0;
pub const SCALE_2_25: f32 = 2.25;
pub const SCALE_2_50: f32 = 2.5;
pub const SCALE_3_00: f32 = 3.0;

/// Ordered Wayland `1/120` numerator lattice used for automatic scale optimization and interactive zoom
/// (`Ctrl+=` / `Ctrl+-` / `Ctrl+0`).
///
/// Includes all exact `15k/120 = k/8` IEEE-754 dyadic steps in `[0.75, 4.0]` plus the exact
/// Wayland third-steps `160/120 = 4/3` (`1.333x`) and `200/120 = 5/3` (`1.667x`).
pub const SCALE_LATTICE_120: [u32; 19] = [
    90,  // 0.750 (3/4)
    105, // 0.875 (7/8)
    120, // 1.000 (1/1)
    135, // 1.125 (9/8)
    150, // 1.250 (5/4)
    160, // 1.333 (4/3)
    165, // 1.375 (11/8)
    180, // 1.500 (3/2)
    195, // 1.625 (13/8)
    200, // 1.667 (5/3)
    210, // 1.750 (7/4)
    225, // 1.875 (15/8)
    240, // 2.000 (2/1)
    270, // 2.250 (9/4)
    300, // 2.500 (5/2)
    330, // 2.750 (11/4)
    360, // 3.000 (3/1)
    420, // 3.500 (7/2)
    480, // 4.000 (4/1)
];

#[derive(Clone, Copy)]
struct LatticeCandidate {
    n120: u32,
    ln_s: f32,
    grid_penalty: f32,
}

const W_GRID: f32 = 0.020;
const W_SHORT_WS: f32 = 0.55;
const W_LONG_WS: f32 = 0.45;
const MIN_LOGICAL_SHORT_PX: f32 = 720.0;
const MIN_LOGICAL_LONG_PX: f32 = 1024.0;
const HYSTERESIS_HOLD_BONUS: f32 = 0.0035;

/// Precomputed compile-time lattice table eliminating transcendental `ln()` and `gcd()` calls
/// from the inner optimization loop (`0 B` heap, 228-byte cache-resident scan).
const LATTICE_TABLE: [LatticeCandidate; 19] = [
    LatticeCandidate {
        n120: 90,
        ln_s: -0.287_682_1,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(90)),
    },
    LatticeCandidate {
        n120: 105,
        ln_s: -0.133_531_39,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(105)),
    },
    LatticeCandidate {
        n120: 120,
        ln_s: 0.0,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(120)),
    },
    LatticeCandidate {
        n120: 135,
        ln_s: 0.117_783_03,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(135)),
    },
    LatticeCandidate {
        n120: 150,
        ln_s: 0.223_143_55,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(150)),
    },
    LatticeCandidate {
        n120: 160,
        ln_s: 0.287_682_1,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(160)),
    },
    LatticeCandidate {
        n120: 165,
        ln_s: 0.318_453_73,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(165)),
    },
    LatticeCandidate {
        n120: 180,
        ln_s: 0.405_465_1,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(180)),
    },
    LatticeCandidate {
        n120: 195,
        ln_s: 0.485_507_82,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(195)),
    },
    LatticeCandidate {
        n120: 200,
        ln_s: 0.510_825_6,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(200)),
    },
    LatticeCandidate {
        n120: 210,
        ln_s: 0.559_615_8,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(210)),
    },
    LatticeCandidate {
        n120: 225,
        ln_s: 0.628_608_64,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(225)),
    },
    LatticeCandidate {
        n120: 240,
        ln_s: core::f32::consts::LN_2,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(240)),
    },
    LatticeCandidate {
        n120: 270,
        ln_s: 0.810_930_2,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(270)),
    },
    LatticeCandidate {
        n120: 300,
        ln_s: 0.916_290_76,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(300)),
    },
    LatticeCandidate {
        n120: 330,
        ln_s: 1.011_601,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(330)),
    },
    LatticeCandidate {
        n120: 360,
        ln_s: 1.098_612_3,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(360)),
    },
    LatticeCandidate {
        n120: 420,
        ln_s: 1.252_762_9,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(420)),
    },
    LatticeCandidate {
        n120: 480,
        ln_s: 2.0 * core::f32::consts::LN_2,
        grid_penalty: W_GRID * (1.0 - grid_harmony_120ths(480)),
    },
];

/// Physical and compositor display observations consumed by [`ScalePolicy::resolve`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayMetrics {
    /// Active monitor mode width in physical pixels (or window width when monitor mode is unknown).
    pub width_px: u32,
    /// Active monitor mode height in physical pixels (or window height when monitor mode is unknown).
    pub height_px: u32,
    /// Physical active image width in millimeters (from EDID / `wl_output` / XRandR), if known.
    pub width_mm: Option<u32>,
    /// Physical active image height in millimeters (from EDID / `wl_output` / XRandR), if known.
    pub height_mm: Option<u32>,
    /// Scale factor reported by the OS / Wayland `wp_fractional_scale_v1` / X11 `Xft.dpi`, if any.
    pub os_scale_factor: Option<f64>,
    /// Optional current window surface dimensions in physical pixels used to compute
    /// [`ResolvedScale::logical_width`] and [`ResolvedScale::logical_height`].
    /// Defaults to `(width_px, height_px)` when `None`.
    pub window_size_px: Option<(u32, u32)>,
    /// Optional previously resolved scale in Wayland `1/120` units for deadband hysteresis stability.
    pub previous_120ths: Option<u32>,
}

impl Default for DisplayMetrics {
    #[inline]
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

impl DisplayMetrics {
    /// Construct display metrics for a monitor mode `(width_px, height_px)` with no physical `mm`
    /// or OS scale attached yet.
    #[inline]
    pub const fn new(width_px: u32, height_px: u32) -> Self {
        Self {
            width_px,
            height_px,
            width_mm: None,
            height_mm: None,
            os_scale_factor: None,
            window_size_px: None,
            previous_120ths: None,
        }
    }

    /// Construct display metrics directly from physical pixel resolution and physical millimeter dimensions.
    #[inline]
    pub const fn from_physical_mm(
        width_px: u32,
        height_px: u32,
        width_mm: u32,
        height_mm: u32,
    ) -> Self {
        Self {
            width_px,
            height_px,
            width_mm: Some(width_mm),
            height_mm: Some(height_mm),
            os_scale_factor: None,
            window_size_px: None,
            previous_120ths: None,
        }
    }

    /// Construct display metrics when only physical resolution and OS-reported scale are known.
    #[inline]
    pub const fn from_os_only(width_px: u32, height_px: u32, os_scale_factor: f64) -> Self {
        Self {
            width_px,
            height_px,
            width_mm: None,
            height_mm: None,
            os_scale_factor: Some(os_scale_factor),
            window_size_px: None,
            previous_120ths: None,
        }
    }

    /// Attach physical active image dimensions in millimeters.
    #[inline]
    pub const fn with_physical_mm(mut self, width_mm: u32, height_mm: u32) -> Self {
        self.width_mm = Some(width_mm);
        self.height_mm = Some(height_mm);
        self
    }

    /// Attach optional `(width_mm, height_mm)` tuple.
    #[inline]
    pub const fn with_physical_mm_opt(mut self, mm: Option<(u32, u32)>) -> Self {
        match mm {
            Some((w, h)) => {
                self.width_mm = Some(w);
                self.height_mm = Some(h);
            }
            None => {
                self.width_mm = None;
                self.height_mm = None;
            }
        }
        self
    }

    /// Attach OS/compositor-reported scale factor (`Window::scale_factor()`).
    #[inline]
    pub const fn with_os_scale(mut self, os_scale_factor: f64) -> Self {
        self.os_scale_factor = Some(os_scale_factor);
        self
    }

    /// Attach window swapchain physical dimensions `(win_width_px, win_height_px)` when the window
    /// is smaller than the full monitor mode.
    #[inline]
    pub const fn with_window_px(mut self, win_width_px: u32, win_height_px: u32) -> Self {
        self.window_size_px = Some((win_width_px, win_height_px));
        self
    }

    /// Attach previously active Wayland `1/120` scale for deadband hysteresis stability.
    #[inline]
    pub const fn with_previous_120ths(mut self, prev_120ths: u32) -> Self {
        self.previous_120ths = Some(prev_120ths);
        self
    }

    /// Attach optional previously active Wayland `1/120` scale.
    #[inline]
    pub const fn with_previous_120ths_opt(mut self, prev_120ths: Option<u32>) -> Self {
        self.previous_120ths = prev_120ths;
        self
    }
}

/// Policy controlling how `skima` resolves the active UI scale factor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ScalePolicy {
    /// Smart automatic scaling policy (default):
    /// - Respects an explicit non-unity scale configured in the OS / Wayland compositor
    ///   (`|os_scale - 1.0| >= 1/240`, quantized to `1/120` Wayland units).
    /// - When `os_scale` is `1.0` or absent (such as X11 with default `Xft.dpi = 96` or unscaled
    ///   tiling Wayland compositors) and plausible physical monitor dimensions `(width_mm, height_mm)`
    ///   are available, solves the multi-objective physical scale optimizer.
    /// - Falls back cleanly to `1.0x` (`120/120`) when physical geometry is unavailable or malformed.
    #[default]
    Auto,
    /// Always solves the multi-objective physical scale optimizer from physical monitor geometry
    /// `(width_px, height_px, width_mm, height_mm)`, ignoring `os_scale_factor` (falling back to
    /// valid `os_scale_factor` or `1.0x` only if physical geometry is missing/malformed).
    PurePhysicalAuto,
    /// Uses the raw OS-reported scale factor quantized to Wayland `1/120` units (`1.0x` if absent/invalid).
    System,
    /// Uses an explicit caller-provided scale factor clamped to `[MIN_SCALE, MAX_SCALE]` and
    /// quantized to Wayland `1/120` units (`1.0x` if `NaN` or non-positive).
    Fixed(f32),
}

/// Provenance explaining how [`ResolvedScale`] selected its active scale factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleReason {
    /// Selected by the physical multi-objective automatic scale optimizer.
    PhysicalAuto,
    /// Held at `previous_120ths` within the deadband hysteresis margin.
    PhysicalHysteresisHold,
    /// Adopted explicit non-unity fractional/integer scale reported by the OS/compositor.
    ExplicitOsScale,
    /// Applied OS scale directly under [`ScalePolicy::System`].
    SystemPolicy,
    /// Applied caller-specified scale under [`ScalePolicy::Fixed`].
    FixedPolicy,
    /// Fallback `1.0x` (`120/120`) when physical geometry or explicit scale was absent/malformed.
    FallbackUnity,
}

impl ScaleReason {
    /// Compact human-readable badge label for telemetry HUDs and inspectors.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PhysicalAuto => "Auto",
            Self::PhysicalHysteresisHold => "Auto (Hold)",
            Self::ExplicitOsScale => "OS Scale",
            Self::SystemPolicy => "System",
            Self::FixedPolicy => "Fixed",
            Self::FallbackUnity => "Fallback 1.0x",
        }
    }

    /// Returns `true` if the scale was produced by the physical automatic scale solver (including hysteresis hold).
    #[inline]
    pub const fn is_auto(self) -> bool {
        matches!(self, Self::PhysicalAuto | Self::PhysicalHysteresisHold)
    }
}

/// Complete resolved scale factor and display-physics telemetry exported to layout, rendering, and HUDs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedScale {
    /// Active UI scale factor (`wayland_120ths as f32 / 120.0`), guaranteed finite in `[MIN_SCALE, MAX_SCALE]`.
    pub scale_factor: f32,
    /// Exact numerator in Wayland `wp_fractional_scale_v1` `1/120` units (`120 == 1.0x`, `150 == 1.25x`).
    pub wayland_120ths: u32,
    /// Unquantized continuous visual-acuity scale (`pixels_per_degree / CSS_REFERENCE_PPD`) when
    /// physical geometry is valid, or `scale_factor` otherwise.
    pub raw_acuity_scale: f32,
    /// Physical pixel density in pixels per inch (`PPI`), if valid physical dimensions were provided.
    pub ppi: Option<f32>,
    /// Ergonomic viewing distance in centimeters (`d(W_mm, H_mm)`), if valid physical dimensions were provided.
    pub viewing_distance_cm: Option<f32>,
    /// Native physical angular resolution in pixels per degree (`PPD`) at `viewing_distance_cm`.
    pub pixels_per_degree: Option<f32>,
    /// Resolved logical viewport width (`target_width_px / scale_factor`).
    pub logical_width: f32,
    /// Resolved logical viewport height (`target_height_px / scale_factor`).
    pub logical_height: f32,
    /// Binary design-grid & IEEE-754 subpixel harmony score in `[0.0, 1.0]` (`1.0` = integer,
    /// `0.92` = `1/2`, `0.85` = `1/4`, `0.74` = `1/8` exact dyadic, `0.55` = `1/3` Wayland).
    pub grid_harmony: f32,
    /// Provenance of the resolved scale decision.
    pub reason: ScaleReason,
}

impl Default for ResolvedScale {
    #[inline]
    fn default() -> Self {
        ScalePolicy::Auto.resolve(&DisplayMetrics::default())
    }
}

impl ResolvedScale {
    /// Returns `true` iff `scale_factor` is an exact `k / 8` (`15k / 120`) dyadic float with zero
    /// IEEE-754 binary mantissa error and exact `F26Dot6` (`1/64 px`) glyph cache scaling.
    #[inline]
    pub const fn is_dyadic_exact(&self) -> bool {
        self.wayland_120ths.is_multiple_of(15)
    }

    /// Reduced rational representation `(numerator, denominator)` of `wayland_120ths / 120`
    /// (e.g. `(5, 4)` for `1.25x`, `(4, 3)` for `1.333x`, `(3, 2)` for `1.50x`, `(2, 1)` for `2.0x`).
    #[inline]
    pub const fn reduced_ratio(&self) -> (u32, u32) {
        if self.wayland_120ths == 0 {
            return (1, 1);
        }
        let g = gcd_u32(self.wayland_120ths, WAYLAND_SCALE_DENOM);
        (self.wayland_120ths / g, WAYLAND_SCALE_DENOM / g)
    }

    /// Effective angular resolution per logical UI pixel (`pixels_per_degree / scale_factor`).
    #[inline]
    pub fn effective_logical_ppd(&self) -> Option<f32> {
        self.pixels_per_degree.map(|ppd| ppd / self.scale_factor)
    }

    /// Returns `true` when both `logical_width` and `logical_height` land on exact integer logical pixels.
    #[inline]
    pub fn has_integer_logical_viewport(&self) -> bool {
        (self.logical_width - self.logical_width.round()).abs() < 1e-4
            && (self.logical_height - self.logical_height.round()).abs() < 1e-4
    }
}

#[derive(Clone, Copy)]
struct ValidatedPhysics {
    ppi: f32,
    viewing_distance_cm: f32,
    pixels_per_degree: f32,
    raw_acuity_scale: f32,
}

#[inline]
const fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[inline]
fn smoothstep_f64(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Convert a Wayland `1/120` numerator (`60..=480`) to `f32` scale factor.
///
/// For every `1/8` dyadic step `n120 = 15 * k`, `(15 * k) as f32 / 120.0` is IEEE-754 bit-identical
/// to `(k as f32) * 0.125`.
#[inline]
pub const fn scale_from_120ths(wayland_120ths: u32) -> f32 {
    let clamped = if wayland_120ths < 60 {
        60
    } else if wayland_120ths > 480 {
        480
    } else {
        wayland_120ths
    };
    (clamped as f32) / (WAYLAND_SCALE_DENOM as f32)
}

/// Quantize an arbitrary scale factor into clamped Wayland `1/120` integer units (`60..=480`),
/// returning `None` if `scale` is `NaN`, infinite, or `<= 0.0`.
#[inline]
pub fn quantize_wayland_120ths(scale: f64) -> Option<u32> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let clamped = scale.clamp(MIN_SCALE as f64, MAX_SCALE as f64);
    Some((clamped * (WAYLAND_SCALE_DENOM as f64)).round() as u32)
}

/// Subpixel, `F26Dot6` typography, and binary power-of-two UI design-grid harmony score in `[0.0, 1.0]`
/// for any Wayland `1/120` scale numerator.
///
/// Derived from the reduced denominator `q = 120 / gcd(n120, 120)` across all 16 divisors of `120`:
/// - `q in {1, 2, 4, 8}` (`15k/120` dyadic): exact IEEE-754 `f32`, exact `F26Dot6` (`1/64 px`) glyph
///   sizing, and `{1, 2, 4, 8}px` power-of-two design tokens land on exact integer physical pixels.
/// - `q in {3, 6, 12, 24}` (`4/3`, `5/3`, etc.): Wayland `1/120` exact with compact 3-pixel phase
///   period, but repeating IEEE-754 binary mantissa and zero exact `2^m` integer tokens.
/// - `q in {5, 10, 15, 20, 30, 40, 60, 120}`: longer phase-walk beat periods.
#[inline]
pub const fn grid_harmony_120ths(wayland_120ths: u32) -> f32 {
    if wayland_120ths == 0 {
        return 0.0;
    }
    let denom = WAYLAND_SCALE_DENOM / gcd_u32(wayland_120ths, WAYLAND_SCALE_DENOM);
    match denom {
        1 => 1.00,
        2 => 0.92,
        4 => 0.85,
        8 => 0.74,
        3 => 0.55,
        6 => 0.45,
        5 => 0.42,
        12 => 0.36,
        10 => 0.33,
        24 => 0.28,
        15 => 0.26,
        20 => 0.24,
        30 => 0.19,
        40 => 0.18,
        60 => 0.13,
        _ => 0.08,
    }
}

/// Orientation-invariant, aspect-aware ergonomic viewing distance `d(W_mm, H_mm)` in centimeters.
///
/// Models continuous ergonomic viewing distance across:
/// - Handheld consoles & tablets (`5"–11"` -> `~31–42 cm`),
/// - Clamshell laptops (`13"–17"` -> `~48–55 cm`),
/// - Standard & super-ultrawide desktop monitors (`21"–42"` 16:9 and `34"–49"` `21:9`/`32:9` -> `~63–69 cm`),
/// - Large-format / 10-foot living-room displays (`55"–85"+` -> `~179–275 cm`).
#[inline]
pub fn viewing_distance_cm(width_mm: f32, height_mm: f32) -> Option<f32> {
    if !width_mm.is_finite() || !height_mm.is_finite() || width_mm <= 0.0 || height_mm <= 0.0 {
        return None;
    }
    Some(viewing_distance_cm_f64(width_mm as f64, height_mm as f64) as f32)
}

const TAN_HALF_DEG_X20: f64 = 0.174_537_355_815_175_83;
const CSS_REFERENCE_PPD_F64: f64 = 46.915_573_427_076_85;

#[inline]
fn viewing_distance_cm_with_diag(w_mm: f64, h_mm: f64, diag_mm: f64) -> f64 {
    let short_mm = w_mm.min(h_mm);
    let diag_in = diag_mm / 25.4;
    // 16:9 height-equivalent diagonal: sqrt(16^2 + 9^2) / 9 = 2.03968979...
    let h_eq_in = (short_mm / 25.4) * 2.039_689_791_228_177;
    // Ultrawide desk monitors (21:9, 24:10, 32:9) sit at vertical-height desk distance plus a mild
    // horizontal field-of-view pushback, rather than misclassifying as 50"+ 16:9 living-room TVs.
    let d_eff = diag_in.min(h_eq_in) + 0.16 * (diag_in - h_eq_in).max(0.0);

    let d_hand_laptop = 17.8 + 2.62 * d_eff - 0.0205 * d_eff * d_eff;
    let d_desk = 57.5 + 0.30 * (d_eff - 4.0).max(0.0);
    let w_desk = smoothstep_f64(16.5, 22.5, d_eff);
    let d_near_desk = d_hand_laptop * (1.0 - w_desk) + d_desk * w_desk;

    let w_tv = smoothstep_f64(44.0, 54.0, d_eff);
    let d_tv = 3.25 * d_eff;
    (d_near_desk * (1.0 - w_tv) + d_tv * w_tv).clamp(22.0, 500.0)
}

#[inline]
fn viewing_distance_cm_f64(w_mm: f64, h_mm: f64) -> f64 {
    viewing_distance_cm_with_diag(w_mm, h_mm, w_mm.hypot(h_mm))
}

/// Validate physical pixel & millimeter dimensions against real-world EDID / firmware pathologies:
/// - Rejects `0` px/mm, `< 50x40 mm` aspect-ratio-in-mm firmware bugs (e.g. `16x9 mm`), and `> 5000 mm` walls.
/// - Automatically aligns unrotated EDID `(w_mm, h_mm)` when a monitor is rotated 90 degrees into portrait.
/// - Rejects severely non-square pixel aspect ratios (`> 33%` anisotropy) and implausible PPI (`< 20` or `> 1200`).
#[inline]
fn validate_display_physics(
    width_px: u32,
    height_px: u32,
    width_mm: Option<u32>,
    height_mm: Option<u32>,
) -> Option<ValidatedPhysics> {
    if width_px < 64 || height_px < 64 {
        return None;
    }
    let (raw_w_mm, raw_h_mm) = (width_mm?, height_mm?);
    let min_mm = raw_w_mm.min(raw_h_mm);
    let max_mm = raw_w_mm.max(raw_h_mm);
    if min_mm < 40 || !(50..=5_000).contains(&max_mm) {
        return None;
    }

    // Align physical (w_mm, h_mm) orientation with (width_px, height_px) so unrotated raw EDID
    // millimeters remain valid when the display is rotated 90/270 degrees into portrait mode.
    let px_landscape = width_px >= height_px;
    let mm_landscape = raw_w_mm >= raw_h_mm;
    let (w_mm, h_mm) = if px_landscape == mm_landscape {
        (raw_w_mm as f64, raw_h_mm as f64)
    } else {
        (raw_h_mm as f64, raw_w_mm as f64)
    };

    let w_px = width_px as f64;
    let h_px = height_px as f64;
    let pitch_x = w_mm / w_px;
    let pitch_y = h_mm / h_px;
    if pitch_x <= 0.0 || pitch_y <= 0.0 {
        return None;
    }
    let anisotropy = if pitch_x >= pitch_y {
        pitch_x / pitch_y
    } else {
        pitch_y / pitch_x
    };
    if anisotropy > 1.35 {
        return None;
    }

    // In f64, u32::MAX^2 == 1.84e19 << f64::MAX (1.8e308), so x*x + y*y never overflows and
    // lowers to hardware sqrtsd without external libm hypot() branches:
    let diag_mm = (w_mm * w_mm + h_mm * h_mm).sqrt();
    let diag_in = diag_mm / 25.4;
    if diag_in <= 0.0 {
        return None;
    }
    let diag_px = (w_px * w_px + h_px * h_px).sqrt();
    let ppi = diag_px / diag_in;
    if !(20.0..=1_200.0).contains(&ppi) {
        return None;
    }

    let d_cm = viewing_distance_cm_with_diag(w_mm, h_mm, diag_mm);
    let pitch_mm = (pitch_x * pitch_y).sqrt();
    let ppd = (d_cm * TAN_HALF_DEG_X20) / pitch_mm;
    let raw_acuity = ppd / CSS_REFERENCE_PPD_F64;
    if !raw_acuity.is_finite() || raw_acuity <= 0.0 {
        return None;
    }

    Some(ValidatedPhysics {
        ppi: ppi as f32,
        viewing_distance_cm: d_cm as f32,
        pixels_per_degree: ppd as f32,
        raw_acuity_scale: raw_acuity as f32,
    })
}

/// Solve the co-designed multi-objective scale optimization problem over `LATTICE_TABLE` (`>= 1.0x`),
/// returning `(selected_120ths, reason)`.
#[inline]
fn solve_physical_auto(
    target_w_px: u32,
    target_h_px: u32,
    phys: ValidatedPhysics,
    previous_120ths: Option<u32>,
) -> (u32, ScaleReason) {
    let ln_raw = phys.raw_acuity_scale.clamp(0.25, 8.0).ln();
    let short_px = (target_w_px.min(target_h_px).max(1)) as f32;
    let long_px = (target_w_px.max(target_h_px).max(1)) as f32;
    let ln_short_lim = (MIN_LOGICAL_SHORT_PX / short_px).ln();
    let ln_long_lim = (MIN_LOGICAL_LONG_PX / long_px).ln();

    // Smooth C1 native-1:1 pixel-identity deadzone: protects <= 112 PPI desktop & ultrawide displays
    // from blurring into 1.08x..1.13x while tapering smoothly to zero by 122 PPI.
    let unity_deadzone_bonus =
        0.0025 + 0.0095 * (1.0 - smoothstep_f32(110.0, 122.0, phys.ppi));

    let mut best_unbiased_120 = 120u32;
    let mut best_unbiased_cost = f32::INFINITY;
    let mut best_held_120 = 120u32;
    let mut best_held_cost = f32::INFINITY;

    let valid_prev = previous_120ths.filter(|&p| p >= 120);

    for cand in &LATTICE_TABLE {
        if cand.n120 < 120 {
            continue;
        }
        let log_err = cand.ln_s - ln_raw;
        let mut cost = log_err * log_err + cand.grid_penalty;

        if cand.n120 == 120 {
            cost -= unity_deadzone_bonus;
        } else {
            // Exact integer logical viewport bonus for fractional scales:
            let rem_w = ((target_w_px as u64) * (WAYLAND_SCALE_DENOM as u64)) % (cand.n120 as u64);
            let rem_h = ((target_h_px as u64) * (WAYLAND_SCALE_DENOM as u64)) % (cand.n120 as u64);
            if rem_w == 0 && rem_h == 0 {
                let w_log_int =
                    ((target_w_px as u64) * (WAYLAND_SCALE_DENOM as u64)) / (cand.n120 as u64);
                let h_log_int =
                    ((target_h_px as u64) * (WAYLAND_SCALE_DENOM as u64)) / (cand.n120 as u64);
                if (w_log_int & 1) == 0 && (h_log_int & 1) == 0 {
                    cost -= 0.0030;
                } else {
                    cost -= 0.0018;
                }
            }
        }

        // Globally Topkis-submodular log-space workspace guard:
        let short_def = (cand.ln_s + ln_short_lim).max(0.0);
        let long_def = (cand.ln_s + ln_long_lim).max(0.0);
        cost += W_SHORT_WS * short_def * short_def + W_LONG_WS * long_def * long_def;

        if cost < best_unbiased_cost {
            best_unbiased_cost = cost;
            best_unbiased_120 = cand.n120;
        }

        let held_cost = if valid_prev == Some(cand.n120) {
            cost - HYSTERESIS_HOLD_BONUS
        } else {
            cost
        };
        if held_cost < best_held_cost {
            best_held_cost = held_cost;
            best_held_120 = cand.n120;
        }
    }

    if best_held_120 != best_unbiased_120 {
        (best_held_120, ScaleReason::PhysicalHysteresisHold)
    } else {
        (best_unbiased_120, ScaleReason::PhysicalAuto)
    }
}

impl ScalePolicy {
    /// Resolve the active fractional scale and full display-physics telemetry (`0 B` heap).
    #[inline]
    pub fn resolve(self, metrics: &DisplayMetrics) -> ResolvedScale {
        let phys = validate_display_physics(
            metrics.width_px,
            metrics.height_px,
            metrics.width_mm,
            metrics.height_mm,
        );
        let os_120 = metrics.os_scale_factor.and_then(quantize_wayland_120ths);

        let (wayland_120ths, reason) = match self {
            Self::Auto => {
                if let Some(explicit_120) = os_120
                    && explicit_120 != WAYLAND_SCALE_DENOM
                {
                    (explicit_120, ScaleReason::ExplicitOsScale)
                } else if let Some(p) = phys {
                    solve_physical_auto(
                        metrics.width_px,
                        metrics.height_px,
                        p,
                        metrics.previous_120ths,
                    )
                } else {
                    (WAYLAND_SCALE_DENOM, ScaleReason::FallbackUnity)
                }
            }
            Self::PurePhysicalAuto => {
                if let Some(p) = phys {
                    solve_physical_auto(
                        metrics.width_px,
                        metrics.height_px,
                        p,
                        metrics.previous_120ths,
                    )
                } else if let Some(explicit_120) = os_120
                    && explicit_120 != WAYLAND_SCALE_DENOM
                {
                    (explicit_120, ScaleReason::ExplicitOsScale)
                } else {
                    (WAYLAND_SCALE_DENOM, ScaleReason::FallbackUnity)
                }
            }
            Self::System => match os_120 {
                Some(n120) => (n120, ScaleReason::SystemPolicy),
                None => (WAYLAND_SCALE_DENOM, ScaleReason::FallbackUnity),
            },
            Self::Fixed(s) => match quantize_wayland_120ths(s as f64) {
                Some(n120) => (n120, ScaleReason::FixedPolicy),
                None => (WAYLAND_SCALE_DENOM, ScaleReason::FallbackUnity),
            },
        };

        let scale_factor = scale_from_120ths(wayland_120ths);
        let (win_w, win_h) = metrics
            .window_size_px
            .unwrap_or((metrics.width_px, metrics.height_px));
        let logical_width = if win_w == 0 {
            0.0
        } else {
            (win_w as f32) / scale_factor
        };
        let logical_height = if win_h == 0 {
            0.0
        } else {
            (win_h as f32) / scale_factor
        };

        ResolvedScale {
            scale_factor,
            wayland_120ths,
            raw_acuity_scale: phys.map_or(scale_factor, |p| p.raw_acuity_scale),
            ppi: phys.map(|p| p.ppi),
            viewing_distance_cm: phys.map(|p| p.viewing_distance_cm),
            pixels_per_degree: phys.map(|p| p.pixels_per_degree),
            logical_width,
            logical_height,
            grid_harmony: grid_harmony_120ths(wayland_120ths),
            reason,
        }
    }

    /// Step cleanly along [`SCALE_LATTICE_120`] by `delta_steps` (`Ctrl+=` -> `+1`, `Ctrl+-` -> `-1`,
    /// `delta_steps == 0` -> snap to nearest lattice step).
    #[inline]
    pub fn zoom_step(cur_scale: f32, delta_steps: i32) -> f32 {
        zoom_step(cur_scale, delta_steps)
    }
}

/// Step cleanly along [`SCALE_LATTICE_120`] by `delta_steps` (`+1` zoom in, `-1` zoom out).
///
/// Guarantees exact `f32::to_bits()` round-trips (`zoom_step(zoom_step(s, k), -k) == s`) for all
/// interior lattice points and saturates cleanly at `[0.75x, 4.00x]`.
#[inline]
pub fn zoom_step(cur_scale: f32, delta_steps: i32) -> f32 {
    let cur = if cur_scale.is_finite() && cur_scale > 0.0 {
        cur_scale
    } else {
        1.0
    };
    let last_idx = (SCALE_LATTICE_120.len() - 1) as i32;
    let cur_120 = cur * (WAYLAND_SCALE_DENOM as f32);

    // Check if cur_scale already sits on (or within 0.25/120 of) a lattice step:
    let mut exact_idx: Option<i32> = None;
    for (i, &n120) in SCALE_LATTICE_120.iter().enumerate() {
        if ((n120 as f32) - cur_120).abs() <= 0.25 {
            exact_idx = Some(i as i32);
            break;
        }
    }

    let target_idx = if let Some(idx) = exact_idx {
        (idx + delta_steps).clamp(0, last_idx)
    } else if delta_steps == 0 {
        let mut best_i = 0i32;
        let mut best_diff = f32::INFINITY;
        for (i, &n120) in SCALE_LATTICE_120.iter().enumerate() {
            let d = ((n120 as f32) - cur_120).abs();
            if d < best_diff {
                best_diff = d;
                best_i = i as i32;
            }
        }
        best_i
    } else if delta_steps > 0 {
        // First lattice entry strictly above cur_120 counts as step +1:
        let mut first_above = last_idx;
        for (i, &n120) in SCALE_LATTICE_120.iter().enumerate() {
            if (n120 as f32) > cur_120 {
                first_above = i as i32;
                break;
            }
        }
        first_above.saturating_add(delta_steps - 1).clamp(0, last_idx)
    } else {
        // Last lattice entry strictly below cur_120 counts as step -1:
        let mut last_below = 0i32;
        for (i, &n120) in SCALE_LATTICE_120.iter().enumerate() {
            if (n120 as f32) < cur_120 {
                last_below = i as i32;
            } else {
                break;
            }
        }
        last_below.saturating_add(delta_steps + 1).clamp(0, last_idx)
    };

    scale_from_120ths(SCALE_LATTICE_120[target_idx as usize])
}

// ============================================================================
// Zero-Dependency VESA E-EDID 1.3/1.4 Parser & Read-Only Linux DRM Sysfs Probe
// ============================================================================

/// Physical active display geometry extracted from a 128+ byte VESA E-EDID base block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdidPhysicalSize {
    /// Preferred Detailed Timing Descriptor horizontal active pixels (`0` if derived from coarse cm fallback).
    pub preferred_width_px: u32,
    /// Preferred Detailed Timing Descriptor vertical active pixels (`0` if derived from coarse cm fallback).
    pub preferred_height_px: u32,
    /// Active image width in millimeters (`1 mm` precision from DTD bytes `12..=14`, or `10 mm` from byte `21`).
    pub width_mm: u32,
    /// Active image height in millimeters (`1 mm` precision from DTD bytes `13..=14`, or `10 mm` from byte `22`).
    pub height_mm: u32,
}

/// Parse physical active image dimensions `(width_mm, height_mm)` from a raw VESA E-EDID binary blob.
///
/// Validates the 8-byte EDID header (`00 FF FF FF FF FF FF 00`) and 128-byte modulo-256 checksum,
/// preferring 1-millimeter Detailed Timing Descriptor (`DTD`) active image dimensions (`bytes 54..126`)
/// over coarse centimeter fallback (`bytes 21..=22`), and rejecting projector/aspect-ratio-only encodings.
#[inline]
pub fn parse_edid_physical_mm(edid: &[u8]) -> Option<EdidPhysicalSize> {
    if edid.len() < 128 {
        return None;
    }
    if edid[0..8] != [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00] {
        return None;
    }
    let checksum = edid[0..128]
        .iter()
        .fold(0u8, |acc, &b| acc.wrapping_add(b));
    if checksum != 0 {
        return None;
    }

    // Inspect the four 18-byte Detailed Timing Descriptors (bytes 54..126):
    for i in 0..4 {
        let base = 54 + i * 18;
        let dtd = &edid[base..base + 18];
        // Non-zero pixel clock indicates a timing descriptor rather than a monitor descriptor:
        if dtd[0] != 0 || dtd[1] != 0 {
            let h_px = (dtd[2] as u32) | (((dtd[4] >> 4) as u32) << 8);
            let v_px = (dtd[5] as u32) | (((dtd[7] >> 4) as u32) << 8);
            let h_mm = (dtd[12] as u32) | (((dtd[14] >> 4) as u32) << 8);
            let v_mm = (dtd[13] as u32) | (((dtd[14] & 0x0F) as u32) << 8);
            if h_mm >= 40 && v_mm >= 40 && h_mm <= 5_000 && v_mm <= 5_000 {
                return Some(EdidPhysicalSize {
                    preferred_width_px: h_px,
                    preferred_height_px: v_px,
                    width_mm: h_mm,
                    height_mm: v_mm,
                });
            }
        }
    }

    // Coarse basic display parameter fallback (bytes 21 & 22 in cm; if either is 0, EDID 1.4
    // encodes aspect ratio or projector unknown size, NEVER physical centimeters!):
    let cm_w = edid[21] as u32;
    let cm_h = edid[22] as u32;
    if cm_w >= 5 && cm_h >= 4 && cm_w <= 500 && cm_h <= 500 {
        return Some(EdidPhysicalSize {
            preferred_width_px: 0,
            preferred_height_px: 0,
            width_mm: cm_w * 10,
            height_mm: cm_h * 10,
        });
    }
    None
}

#[cfg(target_os = "linux")]
fn connector_names_match(drm_connector: &str, query: &str) -> bool {
    if drm_connector.eq_ignore_ascii_case(query) {
        return true;
    }
    // XRandR drivers occasionally omit "-A" on HDMI connectors ("HDMI-1" vs DRM "HDMI-A-1"):
    if let (Some(a), Some(b)) = (
        drm_connector.strip_prefix("HDMI-A-"),
        query.strip_prefix("HDMI-"),
    ) && a == b
    {
        return true;
    }
    if let (Some(a), Some(b)) = (
        drm_connector.strip_prefix("HDMI-"),
        query.strip_prefix("HDMI-A-"),
    ) && a == b
    {
        return true;
    }
    false
}

/// Probe `/sys/class/drm/card*-*/edid` on Linux (read-only, zero external dependencies) to recover
/// exact millimeter monitor dimensions when `winit::monitor::MonitorHandle` omits physical size.
///
/// Matches by connector name first (e.g. `"DP-1"`, `"HDMI-A-1"`, `"eDP-1"`), falling back to
/// connected connectors whose EDID preferred resolution matches `(mode_width_px, mode_height_px)`
/// (including 90-degree portrait rotation) and passes [`DisplayMetrics`] square-pixel validation.
#[cfg(target_os = "linux")]
pub fn probe_linux_drm_physical_mm(
    connector_name: Option<&str>,
    mode_width_px: u32,
    mode_height_px: u32,
) -> Option<(u32, u32)> {
    use std::io::Read;

    let entries = std::fs::read_dir("/sys/class/drm").ok()?;
    let mut mode_fallback: Option<(u32, u32)> = None;
    let mut ambiguous_fallback = false;

    for ent in entries.flatten() {
        let file_name = ent.file_name();
        let name_str = file_name.to_string_lossy();
        if !name_str.starts_with("card") {
            continue;
        }
        let Some((_, drm_conn)) = name_str.split_once('-') else {
            continue;
        };
        if drm_conn.starts_with("Writeback") {
            continue;
        }

        let dir_path = ent.path();
        let mut status_buf = [0u8; 32];
        let Ok(mut status_file) = std::fs::File::open(dir_path.join("status")) else {
            continue;
        };
        let Ok(n_stat) = status_file.read(&mut status_buf) else {
            continue;
        };
        let status_trim = core::str::from_utf8(&status_buf[..n_stat])
            .unwrap_or("")
            .trim();
        if status_trim != "connected" {
            continue;
        }

        let mut edid_buf = [0u8; 256];
        let Ok(mut edid_file) = std::fs::File::open(dir_path.join("edid")) else {
            continue;
        };
        let Ok(n_edid) = edid_file.read(&mut edid_buf) else {
            continue;
        };
        let Some(parsed) = parse_edid_physical_mm(&edid_buf[..n_edid]) else {
            continue;
        };

        // Verify physical mm is square-pixel compatible with the active mode (if provided):
        if mode_width_px >= 64
            && mode_height_px >= 64
            && validate_display_physics(
                mode_width_px,
                mode_height_px,
                Some(parsed.width_mm),
                Some(parsed.height_mm),
            )
            .is_none()
        {
            continue;
        }

        let oriented = if (mode_width_px >= mode_height_px) == (parsed.width_mm >= parsed.height_mm)
            || mode_width_px < 64
            || mode_height_px < 64
        {
            (parsed.width_mm, parsed.height_mm)
        } else {
            (parsed.height_mm, parsed.width_mm)
        };

        if let Some(wanted) = connector_name
            && connector_names_match(drm_conn, wanted)
        {
            return Some(oriented);
        }

        let mode_matches_dtd = (parsed.preferred_width_px == mode_width_px
            && parsed.preferred_height_px == mode_height_px)
            || (parsed.preferred_width_px == mode_height_px
                && parsed.preferred_height_px == mode_width_px);

        if mode_matches_dtd {
            match mode_fallback {
                None => mode_fallback = Some(oriented),
                Some(prev) => {
                    if prev.0.abs_diff(oriented.0) > 10 || prev.1.abs_diff(oriented.1) > 10 {
                        ambiguous_fallback = true;
                    }
                }
            }
        }
    }

    if ambiguous_fallback {
        None
    } else {
        mode_fallback
    }
}

/// Non-Linux stub returning `None`.
#[cfg(not(target_os = "linux"))]
#[inline]
pub fn probe_linux_drm_physical_mm(
    _connector_name: Option<&str>,
    _mode_width_px: u32,
    _mode_height_px: u32,
) -> Option<(u32, u32)> {
    None
}

// ============================================================================
// Contender Algorithm Families & 30-Display Real-World Benchmark Matrix
// ============================================================================

/// Contender A (Conventional Baseline): Naive linear `PPI / 96.0` clamped to `>= 1.0`.
///
/// Fails across viewing-distance regimes (scales `65" 4K TV` at `1.0x` and `7" 1280x800` handheld
/// at `2.25x`) and produces arbitrary irrational scales (`1.133x`, `1.700x`) incompatible with
/// Wayland `1/120` and UI pixel grids.
#[inline]
pub fn contender_a_linear_ppi(metrics: &DisplayMetrics) -> f32 {
    validate_display_physics(
        metrics.width_px,
        metrics.height_px,
        metrics.width_mm,
        metrics.height_mm,
    )
    .map_or(1.0, |p| (p.ppi / 96.0).clamp(1.0, MAX_SCALE))
}

/// Contender B (Desktop Step-Table Control): Fixed `25%` PPI step buckets ignoring viewing distance
/// and logical viewport divisibility.
#[inline]
pub fn contender_b_ppi_step_table(metrics: &DisplayMetrics) -> f32 {
    let Some(p) = validate_display_physics(
        metrics.width_px,
        metrics.height_px,
        metrics.width_mm,
        metrics.height_mm,
    ) else {
        return 1.0;
    };
    if p.ppi < 120.0 {
        1.0
    } else if p.ppi < 144.0 {
        1.25
    } else if p.ppi < 180.0 {
        1.5
    } else if p.ppi < 204.0 {
        1.75
    } else if p.ppi < 252.0 {
        2.0
    } else if p.ppi < 300.0 {
        2.25
    } else {
        2.5
    }
}

/// Contender C (Pure Angular-Acuity `1/120` Quantizer): Viewing-distance visual-acuity scale
/// (`PPD / CSS_REFERENCE_PPD`) quantized directly to the nearest `1/120` Wayland step without
/// dyadic grid harmony, `1.0x` deadzone protection, or workspace bounds.
#[inline]
pub fn contender_c_acuity_120ths(metrics: &DisplayMetrics) -> (u32, f32) {
    let Some(p) = validate_display_physics(
        metrics.width_px,
        metrics.height_px,
        metrics.width_mm,
        metrics.height_mm,
    ) else {
        return (WAYLAND_SCALE_DENOM, 1.0);
    };
    let n120 = quantize_wayland_120ths((p.raw_acuity_scale as f64).max(1.0)).unwrap_or(120);
    (n120, scale_from_120ths(n120))
}

/// Side-by-side outputs of all four contender families on a single [`DisplayMetrics`] sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContenderComparison {
    pub contender_a_scale: f32,
    pub contender_b_scale: f32,
    pub contender_c_120ths: u32,
    pub contender_c_scale: f32,
    pub contender_c_harmony: f32,
    pub contender_d: ResolvedScale,
}

/// Evaluate Contenders A, B, C, and D (`PurePhysicalAuto`) on the same [`DisplayMetrics`].
#[inline]
pub fn evaluate_contenders(metrics: &DisplayMetrics) -> ContenderComparison {
    let contender_a_scale = contender_a_linear_ppi(metrics);
    let contender_b_scale = contender_b_ppi_step_table(metrics);
    let (contender_c_120ths, contender_c_scale) = contender_c_acuity_120ths(metrics);
    let contender_d = ScalePolicy::PurePhysicalAuto.resolve(metrics);
    ContenderComparison {
        contender_a_scale,
        contender_b_scale,
        contender_c_120ths,
        contender_c_scale,
        contender_c_harmony: grid_harmony_120ths(contender_c_120ths),
        contender_d,
    }
}

/// Real-world reference display specification for benchmarking, documentation, and QA verification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayProfile {
    pub name: &'static str,
    pub width_px: u32,
    pub height_px: u32,
    pub width_mm: u32,
    pub height_mm: u32,
    pub diagonal_in: f32,
    /// Ergonomic target scale band `(min_scale, max_scale)` grounded in vendor & desktop standards.
    pub expected_scale_range: (f32, f32),
}

impl DisplayProfile {
    /// Convert this profile into [`DisplayMetrics`] with `os_scale_factor = None`.
    #[inline]
    pub const fn metrics(&self) -> DisplayMetrics {
        DisplayMetrics::from_physical_mm(
            self.width_px,
            self.height_px,
            self.width_mm,
            self.height_mm,
        )
    }
}

/// Comprehensive 30-display benchmark matrix spanning handhelds, 13"–16" laptops, 21.5"–42" 16:9
/// desktop monitors, portrait & `16:18` DualUp workstations, `21:9` / `24:10` / `32:9` ultrawides,
/// and `55"` / `65"` living-room 4K TVs.
pub const REFERENCE_DISPLAY_PROFILES: [DisplayProfile; 30] = [
    DisplayProfile {
        name: "7.0\" Handheld 1280x800",
        width_px: 1280,
        height_px: 800,
        width_mm: 151,
        height_mm: 94,
        diagonal_in: 7.0,
        expected_scale_range: (1.0, 1.25),
    },
    DisplayProfile {
        name: "8.8\" Handheld 2560x1600",
        width_px: 2560,
        height_px: 1600,
        width_mm: 190,
        height_mm: 119,
        diagonal_in: 8.8,
        expected_scale_range: (1.75, 2.0),
    },
    DisplayProfile {
        name: "13.3\" Laptop 1920x1080",
        width_px: 1920,
        height_px: 1080,
        width_mm: 294,
        height_mm: 165,
        diagonal_in: 13.3,
        expected_scale_range: (1.25, 1.25),
    },
    DisplayProfile {
        name: "13.6\" Laptop 2560x1664",
        width_px: 2560,
        height_px: 1664,
        width_mm: 290,
        height_mm: 188,
        diagonal_in: 13.6,
        expected_scale_range: (1.625, 1.75),
    },
    DisplayProfile {
        name: "14.0\" Laptop 1920x1200",
        width_px: 1920,
        height_px: 1200,
        width_mm: 302,
        height_mm: 189,
        diagonal_in: 14.0,
        expected_scale_range: (1.125, 1.25),
    },
    DisplayProfile {
        name: "14.0\" Laptop 2880x1800",
        width_px: 2880,
        height_px: 1800,
        width_mm: 302,
        height_mm: 189,
        diagonal_in: 14.0,
        expected_scale_range: (1.75, 2.0),
    },
    DisplayProfile {
        name: "14.2\" Laptop 3024x1964",
        width_px: 3024,
        height_px: 1964,
        width_mm: 302,
        height_mm: 196,
        diagonal_in: 14.2,
        expected_scale_range: (1.75, 2.0),
    },
    DisplayProfile {
        name: "15.0\" Laptop 3:2 2496x1664",
        width_px: 2496,
        height_px: 1664,
        width_mm: 317,
        height_mm: 211,
        diagonal_in: 15.0,
        expected_scale_range: (1.5, 1.5),
    },
    DisplayProfile {
        name: "15.6\" Laptop 1920x1080",
        width_px: 1920,
        height_px: 1080,
        width_mm: 345,
        height_mm: 194,
        diagonal_in: 15.6,
        expected_scale_range: (1.125, 1.25),
    },
    DisplayProfile {
        name: "16.0\" Laptop 2560x1600",
        width_px: 2560,
        height_px: 1600,
        width_mm: 345,
        height_mm: 215,
        diagonal_in: 16.0,
        expected_scale_range: (1.5, 1.5),
    },
    DisplayProfile {
        name: "16.0\" Laptop 3840x2400",
        width_px: 3840,
        height_px: 2400,
        width_mm: 345,
        height_mm: 215,
        diagonal_in: 16.0,
        expected_scale_range: (2.0, 2.25),
    },
    DisplayProfile {
        name: "21.5\" Desk 4096x2304",
        width_px: 4096,
        height_px: 2304,
        width_mm: 476,
        height_mm: 268,
        diagonal_in: 21.5,
        expected_scale_range: (2.0, 2.0),
    },
    DisplayProfile {
        name: "24.0\" Desk 1920x1080",
        width_px: 1920,
        height_px: 1080,
        width_mm: 531,
        height_mm: 299,
        diagonal_in: 24.0,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "24.0\" Desk 2560x1440",
        width_px: 2560,
        height_px: 1440,
        width_mm: 531,
        height_mm: 299,
        diagonal_in: 24.0,
        expected_scale_range: (1.125, 1.25),
    },
    DisplayProfile {
        name: "24.0\" Desk 3840x2160",
        width_px: 3840,
        height_px: 2160,
        width_mm: 531,
        height_mm: 299,
        diagonal_in: 24.0,
        expected_scale_range: (1.625, 1.75),
    },
    DisplayProfile {
        name: "27.0\" Desk 1920x1080",
        width_px: 1920,
        height_px: 1080,
        width_mm: 597,
        height_mm: 336,
        diagonal_in: 27.0,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "27.0\" Desk 2560x1440",
        width_px: 2560,
        height_px: 1440,
        width_mm: 597,
        height_mm: 336,
        diagonal_in: 27.0,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "27.0\" Desk 3840x2160",
        width_px: 3840,
        height_px: 2160,
        width_mm: 597,
        height_mm: 336,
        diagonal_in: 27.0,
        expected_scale_range: (1.5, 1.5),
    },
    DisplayProfile {
        name: "27.0\" 4K Portrait 2160x3840",
        width_px: 2160,
        height_px: 3840,
        width_mm: 336,
        height_mm: 597,
        diagonal_in: 27.0,
        expected_scale_range: (1.5, 1.5),
    },
    DisplayProfile {
        name: "27.0\" Studio 5K 5120x2880",
        width_px: 5120,
        height_px: 2880,
        width_mm: 597,
        height_mm: 336,
        diagonal_in: 27.0,
        expected_scale_range: (2.0, 2.0),
    },
    DisplayProfile {
        name: "27.6\" DualUp 16:18 2560x2880",
        width_px: 2560,
        height_px: 2880,
        width_mm: 465,
        height_mm: 523,
        diagonal_in: 27.6,
        expected_scale_range: (1.25, 1.25),
    },
    DisplayProfile {
        name: "32.0\" Desk 2560x1440",
        width_px: 2560,
        height_px: 1440,
        width_mm: 708,
        height_mm: 398,
        diagonal_in: 32.0,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "32.0\" Desk 3840x2160",
        width_px: 3840,
        height_px: 2160,
        width_mm: 708,
        height_mm: 398,
        diagonal_in: 32.0,
        expected_scale_range: (1.25, 1.25),
    },
    DisplayProfile {
        name: "32.0\" Pro 6K 6016x3384",
        width_px: 6016,
        height_px: 3384,
        width_mm: 699,
        height_mm: 393,
        diagonal_in: 31.6,
        expected_scale_range: (2.0, 2.0),
    },
    DisplayProfile {
        name: "34.0\" UWQHD 21:9 3440x1440",
        width_px: 3440,
        height_px: 1440,
        width_mm: 797,
        height_mm: 334,
        diagonal_in: 34.0,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "38.0\" UWQHD+ 24:10 3840x1600",
        width_px: 3840,
        height_px: 1600,
        width_mm: 880,
        height_mm: 367,
        diagonal_in: 37.5,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "42.0\" Desk OLED 3840x2160",
        width_px: 3840,
        height_px: 2160,
        width_mm: 930,
        height_mm: 523,
        diagonal_in: 42.0,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "49.0\" DQHD 32:9 5120x1440",
        width_px: 5120,
        height_px: 1440,
        width_mm: 1193,
        height_mm: 336,
        diagonal_in: 49.0,
        expected_scale_range: (1.0, 1.0),
    },
    DisplayProfile {
        name: "55.0\" TV 4K 3840x2160",
        width_px: 3840,
        height_px: 2160,
        width_mm: 1218,
        height_mm: 685,
        diagonal_in: 55.0,
        expected_scale_range: (2.0, 2.0),
    },
    DisplayProfile {
        name: "65.0\" TV 4K 3840x2160",
        width_px: 3840,
        height_px: 2160,
        width_mm: 1439,
        height_mm: 809,
        diagonal_in: 65.0,
        expected_scale_range: (2.0, 2.25),
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ieee754_exactness_of_dyadic_15k_over_120_and_f26dot6_identity() {
        // Every 1/8 dyadic multiple k/8 == (15*k)/120 in [0.5, 4.0] must have ZERO IEEE-754
        // binary representation error and exact F26Dot6 (1/64 px) font size identity!
        for k in 4u32..=32 {
            let n120 = 15 * k;
            let from_wayland = scale_from_120ths(n120);
            let exact_dyadic = (k as f32) * 0.125_f32;
            assert_eq!(
                from_wayland.to_bits(),
                exact_dyadic.to_bits(),
                "15*{k}/120 ({n120}/120) failed exact IEEE-754 dyadic identity"
            );
            assert!(
                grid_harmony_120ths(n120) >= 0.74,
                "Expected dyadic harmony >= 0.74 for {n120}/120"
            );
        }

        // Verify F26Dot6 font-cache identity at 1.25x (150/120):
        let sf_125 = scale_from_120ths(150);
        assert_eq!((14.0_f32 * sf_125).to_bits(), 17.5_f32.to_bits());
        assert_eq!(
            ((13.5_f32 * sf_125) * 64.0).round() as u32,
            ((16.875_f32) * 64.0).round() as u32
        );

        // Contrast non-dyadic 4/3 = 160/120: Wayland-exact (reduced ratio 4/3), lower binary harmony:
        let res_4_3 = ScalePolicy::Fixed(SCALE_1_33).resolve(&DisplayMetrics::new(1920, 1080));
        assert_eq!(res_4_3.wayland_120ths, 160);
        assert_eq!(res_4_3.reduced_ratio(), (4, 3));
        assert!(!res_4_3.is_dyadic_exact());
        assert!((res_4_3.grid_harmony - 0.55).abs() < 1e-6);
        assert!(res_4_3.has_integer_logical_viewport()); // 1920x1080 / (4/3) = 1440x810!
    }

    #[test]
    fn precomputed_lattice_constants_match_runtime_math() {
        assert_eq!(LATTICE_TABLE.len(), SCALE_LATTICE_120.len());
        for (i, cand) in LATTICE_TABLE.iter().enumerate() {
            assert_eq!(cand.n120, SCALE_LATTICE_120[i]);
            let expected_scale = scale_from_120ths(cand.n120);
            let expected_ln = expected_scale.ln();
            assert!(
                (cand.ln_s - expected_ln).abs() <= 1e-7,
                "ln_s drift at n120={}: got {}, expected {}",
                cand.n120,
                cand.ln_s,
                expected_ln
            );
            let expected_pen = W_GRID * (1.0 - grid_harmony_120ths(cand.n120));
            assert!((cand.grid_penalty - expected_pen).abs() <= 1e-7);
        }
    }

    #[test]
    fn all_30_reference_displays_hit_ergonomic_scale_bands_and_dominate_contenders() {
        let mut pass_a = 0usize;
        let mut pass_b = 0usize;
        let mut pass_c = 0usize;
        let mut pass_d = 0usize;

        for prof in &REFERENCE_DISPLAY_PROFILES {
            let cmp = evaluate_contenders(&prof.metrics());
            let (lo, hi) = prof.expected_scale_range;
            let d_scale = cmp.contender_d.scale_factor;

            assert!(
                d_scale >= lo - 1e-4 && d_scale <= hi + 1e-4,
                "Contender D missed {} ({}x{} on {}x{}mm): got {}x ({}/120, raw={:.3}, ppi={:?}, d_cm={:?}), expected [{}, {}]",
                prof.name,
                prof.width_px,
                prof.height_px,
                prof.width_mm,
                prof.height_mm,
                d_scale,
                cmp.contender_d.wayland_120ths,
                cmp.contender_d.raw_acuity_scale,
                cmp.contender_d.ppi,
                cmp.contender_d.viewing_distance_cm,
                lo,
                hi
            );
            assert!(cmp.contender_d.is_dyadic_exact());
            assert_eq!(cmp.contender_d.reason, ScaleReason::PhysicalAuto);
            pass_d += 1;

            let a_wayland_exact = ((cmp.contender_a_scale * 8.0).fract()).abs() < 1e-3;
            if cmp.contender_a_scale >= lo - 0.01
                && cmp.contender_a_scale <= hi + 0.01
                && a_wayland_exact
            {
                pass_a += 1;
            }
            if cmp.contender_b_scale >= lo - 0.01 && cmp.contender_b_scale <= hi + 0.01 {
                pass_b += 1;
            }
            if cmp.contender_c_scale >= lo - 0.01
                && cmp.contender_c_scale <= hi + 0.01
                && cmp.contender_c_harmony >= 0.74
            {
                pass_c += 1;
            }
        }

        assert_eq!(pass_d, REFERENCE_DISPLAY_PROFILES.len());
        assert!(pass_d > pass_b && pass_b > pass_c && pass_d > pass_a);

        // Pin decisive canonical displays explicitly:
        let find = |substr: &str| -> ResolvedScale {
            let p = REFERENCE_DISPLAY_PROFILES
                .iter()
                .find(|p| p.name.contains(substr))
                .unwrap();
            ScalePolicy::Auto.resolve(&p.metrics())
        };
        assert_eq!(find("24.0\" Desk 1920x1080").wayland_120ths, 120); // 1.00x
        assert_eq!(find("27.0\" Desk 2560x1440").wayland_120ths, 120); // 1.00x (never 1.08x..1.13x!)
        assert_eq!(find("27.0\" Desk 3840x2160").wayland_120ths, 180); // 1.50x (exact 2560x1440!)
        assert_eq!(find("27.0\" Studio 5K").wayland_120ths, 240); // 2.00x (exact 2560x1440!)
        assert_eq!(find("32.0\" Desk 3840x2160").wayland_120ths, 150); // 1.25x (exact 3072x1728!)
        assert_eq!(find("42.0\" Desk OLED").wayland_120ths, 120); // 1.00x
        assert_eq!(find("49.0\" DQHD 32:9").wayland_120ths, 120); // 1.00x (ultrawide != TV!)
        assert_eq!(find("55.0\" TV 4K").wayland_120ths, 240); // 2.00x (10ft couch UI!)
    }

    #[test]
    fn portrait_rotation_and_unrotated_edid_orientation_invariance() {
        let landscape = DisplayMetrics::from_physical_mm(3840, 2160, 597, 336);
        let portrait = DisplayMetrics::from_physical_mm(2160, 3840, 336, 597);
        // Also simulate raw unrotated EDID (597, 336) reported on a rotated portrait mode (2160, 3840):
        let portrait_raw_edid = DisplayMetrics::from_physical_mm(2160, 3840, 597, 336);

        let r_land = ScalePolicy::Auto.resolve(&landscape);
        let r_port = ScalePolicy::Auto.resolve(&portrait);
        let r_port_raw = ScalePolicy::Auto.resolve(&portrait_raw_edid);

        assert_eq!(r_land.wayland_120ths, 180);
        assert_eq!(r_port.wayland_120ths, 180);
        assert_eq!(r_port_raw.wayland_120ths, 180);
        assert!((r_land.ppi.unwrap() - r_port.ppi.unwrap()).abs() < 1e-4);
        assert!(
            (r_land.viewing_distance_cm.unwrap() - r_port.viewing_distance_cm.unwrap()).abs()
                < 1e-4
        );
        assert!((r_port.logical_width - 1440.0).abs() < 1e-4);
        assert!((r_port.logical_height - 2560.0).abs() < 1e-4);
    }

    #[test]
    fn monotonicity_across_resolutions_and_physical_diagonals() {
        // 1. Fixed physical size across increasing standard display resolutions:
        for diag_in in [13.3_f32, 14.0, 16.0, 24.0, 27.0, 32.0] {
            let diag_mm = diag_in * 25.4;
            let w_mm = (diag_mm * 16.0 / 16.0_f32.hypot(9.0)).round() as u32;
            let h_mm = (diag_mm * 9.0 / 16.0_f32.hypot(9.0)).round() as u32;
            let modes = [
                (1280, 720),
                (1600, 900),
                (1920, 1080),
                (2560, 1440),
                (2880, 1620),
                (3200, 1800),
                (3840, 2160),
                (5120, 2880),
                (6016, 3384),
                (7680, 4320),
            ];
            let mut prev_120 = 0u32;
            for (w_px, h_px) in modes {
                let res = ScalePolicy::Auto
                    .resolve(&DisplayMetrics::from_physical_mm(w_px, h_px, w_mm, h_mm));
                assert!(
                    res.wayland_120ths >= prev_120,
                    "Non-monotone mode step on {diag_in}\" at {w_px}x{h_px}: {} < {prev_120}",
                    res.wayland_120ths
                );
                prev_120 = res.wayland_120ths;
            }
        }

        // 2. Fixed pixel resolution (3840x2160 and 2560x1440) as physical diagonal shrinks from 42" -> 12":
        for (w_px, h_px) in [(3840u32, 2160u32), (2560u32, 1440u32)] {
            let mut prev_120 = 0u32;
            for tenth_in in (120u32..=420).rev() {
                let diag_mm = (tenth_in as f32) * 0.1 * 25.4;
                let w_mm = (diag_mm * 16.0 / 16.0_f32.hypot(9.0)).round() as u32;
                let h_mm = (diag_mm * 9.0 / 16.0_f32.hypot(9.0)).round() as u32;
                let res = ScalePolicy::Auto
                    .resolve(&DisplayMetrics::from_physical_mm(w_px, h_px, w_mm, h_mm));
                assert!(
                    res.wayland_120ths >= prev_120,
                    "Non-monotone shrinking diagonal at {}.{} in ({w_px}x{h_px}): {} < {prev_120}",
                    tenth_in / 10,
                    tenth_in % 10,
                    res.wayland_120ths
                );
                prev_120 = res.wayland_120ths;
            }
        }
    }

    #[test]
    fn deadband_hysteresis_prevents_boundary_chatter_and_breaks_out_cleanly() {
        // Construct a synthetic 3840x2160 monitor sweeping across the 1.25x (150) <-> 1.50x (180)
        // decision boundary to verify hysteresis hold vs breakout:
        let mut boundary_mm: Option<(u32, u32)> = None;
        for w_mm in (580u32..=720).rev() {
            let h_mm = ((w_mm as f32) * 9.0 / 16.0).round() as u32;
            let r = ScalePolicy::Auto
                .resolve(&DisplayMetrics::from_physical_mm(3840, 2160, w_mm, h_mm));
            if r.wayland_120ths >= 165 {
                boundary_mm = Some((w_mm, h_mm));
                break;
            }
        }
        let (bw, bh) = boundary_mm.expect("Expected crossover between 720mm and 580mm");

        // Just past the unbiased upward crossover, holding previous_120ths = 150 (1.25x) keeps 150:
        let held = ScalePolicy::Auto.resolve(
            &DisplayMetrics::from_physical_mm(3840, 2160, bw, bh).with_previous_120ths(150),
        );
        assert_eq!(held.wayland_120ths, 150);
        assert_eq!(held.reason, ScaleReason::PhysicalHysteresisHold);
        assert!(held.reason.is_auto());
        assert_eq!(held.reason.label(), "Auto (Hold)");

        // Well inside the 27" 4K regime (597x336mm), hysteresis cleanly breaks out to 1.50x (180):
        let breakout = ScalePolicy::Auto.resolve(
            &DisplayMetrics::from_physical_mm(3840, 2160, 597, 336).with_previous_120ths(150),
        );
        assert_eq!(breakout.wayland_120ths, 180);
        assert_eq!(breakout.reason, ScaleReason::PhysicalAuto);
        assert_eq!(breakout.reason.label(), "Auto");
    }

    #[test]
    fn scale_policies_and_window_subregion_semantics() {
        // 27" 4K monitor on X11 reporting bogus os_scale = 1.0 with a 1200x900 window:
        let m_x11 = DisplayMetrics::from_physical_mm(3840, 2160, 597, 336)
            .with_os_scale(1.0)
            .with_window_px(1200, 900);

        let auto = ScalePolicy::Auto.resolve(&m_x11);
        assert_eq!(auto.wayland_120ths, 180); // 1.50x physical auto overrides X11 1.0x!
        assert_eq!(auto.reason, ScaleReason::PhysicalAuto);
        assert!((auto.logical_width - 800.0).abs() < 1e-4);
        assert!((auto.logical_height - 600.0).abs() < 1e-4);

        // Explicit Wayland fractional scale (e.g. user chose 1.25x = 150/120 in compositor settings):
        let m_wayland = m_x11.with_os_scale(1.25);
        let auto_wl = ScalePolicy::Auto.resolve(&m_wayland);
        assert_eq!(auto_wl.wayland_120ths, 150);
        assert_eq!(auto_wl.reason, ScaleReason::ExplicitOsScale);
        assert!(auto_wl.ppi.is_some()); // Physical telemetry still preserved!

        // PurePhysicalAuto ignores explicit OS scale when physical geometry is valid:
        let pure_phys = ScalePolicy::PurePhysicalAuto.resolve(&m_wayland);
        assert_eq!(pure_phys.wayland_120ths, 180);
        assert_eq!(pure_phys.reason, ScaleReason::PhysicalAuto);

        // System policy uses OS scale directly even when 1.0x:
        let sys_pol = ScalePolicy::System.resolve(&m_x11);
        assert_eq!(sys_pol.wayland_120ths, 120);
        assert_eq!(sys_pol.reason, ScaleReason::SystemPolicy);

        // Fixed policy quantizes 1.33 -> 160/120 (4/3):
        let fixed_pol = ScalePolicy::Fixed(1.33).resolve(&m_x11);
        assert_eq!(fixed_pol.wayland_120ths, 160);
        assert_eq!(fixed_pol.reason, ScaleReason::FixedPolicy);
    }

    #[test]
    fn zoom_step_round_trips_and_off_lattice_snapping() {
        for (i, &n120) in SCALE_LATTICE_120.iter().enumerate() {
            let s = scale_from_120ths(n120);
            assert_eq!(zoom_step(s, 0).to_bits(), s.to_bits());
            if i + 1 < SCALE_LATTICE_120.len() {
                let up = zoom_step(s, 1);
                assert!(up > s);
                assert_eq!(zoom_step(up, -1).to_bits(), s.to_bits());
            }
            if i + 3 < SCALE_LATTICE_120.len() {
                let up3 = zoom_step(s, 3);
                assert_eq!(zoom_step(up3, -3).to_bits(), s.to_bits());
            }
        }

        // Off-lattice 1.20x: +1 -> 1.25x, -1 -> 1.125x:
        assert_eq!(zoom_step(1.20, 1).to_bits(), SCALE_1_25.to_bits());
        assert_eq!(zoom_step(1.20, -1).to_bits(), SCALE_1_125.to_bits());

        // Saturation & NaN safety:
        assert_eq!(zoom_step(1.0, 1000).to_bits(), MAX_SCALE.to_bits());
        assert_eq!(zoom_step(1.0, -1000).to_bits(), SCALE_0_75.to_bits());
        assert_eq!(zoom_step(f32::NAN, 0).to_bits(), 1.0_f32.to_bits());
        assert_eq!(zoom_step(f32::INFINITY, 1).to_bits(), SCALE_1_125.to_bits());
        assert_eq!(zoom_step(-5.0, 0).to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn malformed_edid_and_extreme_adversarial_inputs_never_nan_or_panic() {
        let bad_cases = [
            DisplayMetrics::new(0, 0),
            DisplayMetrics::from_physical_mm(0, 0, 0, 0),
            DisplayMetrics::from_physical_mm(1920, 1080, 0, 0),
            DisplayMetrics::from_physical_mm(1920, 1080, 16, 9), // Aspect-ratio-in-mm TV firmware bug!
            DisplayMetrics::from_physical_mm(5120, 1440, 597, 336), // Non-square 2x aspect mismatch!
            DisplayMetrics::from_physical_mm(u32::MAX, u32::MAX, u32::MAX, u32::MAX),
            DisplayMetrics::from_physical_mm(u32::MAX, u32::MAX, 500, 300),
            DisplayMetrics::from_physical_mm(1920, 1080, u32::MAX, u32::MAX),
            DisplayMetrics::from_os_only(1920, 1080, f64::NAN),
            DisplayMetrics::from_os_only(1920, 1080, f64::INFINITY),
            DisplayMetrics::from_os_only(1920, 1080, f64::NEG_INFINITY),
            DisplayMetrics::from_os_only(1920, 1080, -2.0),
            DisplayMetrics::from_os_only(1920, 1080, 0.0),
        ];

        let policies = [
            ScalePolicy::Auto,
            ScalePolicy::PurePhysicalAuto,
            ScalePolicy::System,
            ScalePolicy::Fixed(f32::NAN),
            ScalePolicy::Fixed(f32::INFINITY),
            ScalePolicy::Fixed(-1.0),
            ScalePolicy::Fixed(0.0),
            ScalePolicy::Fixed(100.0),
        ];

        for m in &bad_cases {
            for &pol in &policies {
                let res = pol.resolve(m);
                assert!(
                    res.scale_factor.is_finite()
                        && res.scale_factor >= MIN_SCALE
                        && res.scale_factor <= MAX_SCALE
                );
                assert!(res.wayland_120ths >= 60 && res.wayland_120ths <= 480);
                assert!(res.raw_acuity_scale.is_finite() && res.raw_acuity_scale > 0.0);
                assert!(res.logical_width.is_finite() && res.logical_width >= 0.0);
                assert!(res.logical_height.is_finite() && res.logical_height >= 0.0);
                assert!(res.grid_harmony.is_finite() && res.grid_harmony > 0.0);
            }
        }

        // Verify VESA EDID parser on valid synthetic 27" 4K DTD + corrupted/truncated inputs:
        let mut edid = [0u8; 128];
        edid[0..8].copy_from_slice(&[0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]);
        edid[21] = 60;
        edid[22] = 34;
        // DTD #0 at byte 54: pixel clock non-zero, 3840x2160, 597x336 mm:
        edid[54] = 0x02;
        edid[55] = 0x3A;
        edid[56] = (3840 & 0xFF) as u8;
        edid[58] = (((3840 >> 8) & 0x0F) << 4) as u8;
        edid[59] = (2160 & 0xFF) as u8;
        edid[61] = (((2160 >> 8) & 0x0F) << 4) as u8;
        edid[66] = (597 & 0xFF) as u8;
        edid[67] = (336 & 0xFF) as u8;
        edid[68] = ((((597 >> 8) & 0x0F) << 4) | ((336 >> 8) & 0x0F)) as u8;
        let sum: u8 = edid[0..127]
            .iter()
            .fold(0u8, |acc, &b| acc.wrapping_add(b));
        edid[127] = sum.wrapping_neg();

        let parsed = parse_edid_physical_mm(&edid).expect("Valid synthetic 4K EDID should parse");
        assert_eq!(parsed.preferred_width_px, 3840);
        assert_eq!(parsed.preferred_height_px, 2160);
        assert_eq!(parsed.width_mm, 597);
        assert_eq!(parsed.height_mm, 336);

        // Corrupt checksum by +1 -> rejected:
        let mut bad_csum = edid;
        bad_csum[127] = bad_csum[127].wrapping_add(1);
        assert!(parse_edid_physical_mm(&bad_csum).is_none());

        // Truncated slice & all-0xFF slice -> rejected:
        assert!(parse_edid_physical_mm(&edid[..127]).is_none());
        assert!(parse_edid_physical_mm(&[0xFF; 256]).is_none());
    }
}
