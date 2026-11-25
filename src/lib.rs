//! `skima` 1.0: High-Efficiency Rust GUI & Incremental Axis-Factored Layout Engine (`IALE`).
//!
//! Co-designed around Rust 1.100.0 (Edition 2024) ownership, lifetimes, and RAII:
//! - [`app`]: Turnkey `winit 0.30` + Vulkan 1.3 desktop window runner (`run_window`) and headless GPU exporter
//! - [`arena`]: 64-byte aligned expandable `FrameArena` implementing `std::alloc::Allocator`
//! - [`color`]: Perceptual Björn Ottosson `Oklab` / `oklch` and linear sRGB color algebra
//! - [`pool`]: Typed contiguous generational `SlotPool<T>`, multi-type `Realm` (`get_disjoint_mut`), and analytical `Spring`
//! - [`layout`]: 128 B/node SoA `IALE` solver (1D Flex, `WrapRow`, 2D `Grid` with spans, `FirstBaseline`, Relayout & Inline Boundaries, `VirtualListIndex`, and Anchored Popovers)
//! - [`text`]: OpenType shaping (`rustybuzz` + `swash`), zero-C OS font discovery, per-glyph multi-font fallback, ClearType `GlyphAtlas`, and Exact `[W_tight, W_next_unwrap)` Width-Interval Line-Break Cache
//! - [`vector`]: Non-recursive Wang's-formula analytical signed-area 2D vector path rasterizer (`Path2D`)
//! - [`ui`]: Macro-free `#[track_caller]` retained `SlotTape` reconciler, two-lifetime `Ui<'sys, 's, S>`, OKLCH `Theme`, `use_spring`, `DropGuard` unwind rollback, and composable 1.0 controls
//! - [`scale`]: Display physics, ergonomic viewing-distance visual acuity (`PPD`), Wayland `1/120` + IEEE-754 `15k/120` dyadic automatic fractional scaling (`ScalePolicy::Auto`), and read-only Linux DRM EDID probing
//! - [`a11y`]: Incremental `accesskit 0.24` `TreeUpdate` export and action routing
//! - [`gpu`]: Precompiled SPIR-V 1.3 analytical SDF + OKLab + Gaussian shadow + Dual-Source ClearType uber-shader, swapchain `DamageTracker`, native Vulkan 1.3 (`ash`) ReBAR backend, and CPU reference backend

pub mod a11y;
pub mod app;
pub mod arena;
pub mod color;
pub mod gpu;
pub mod layout;
pub mod pool;
pub mod scale;
pub mod text;
pub mod ui;
pub mod vector;

pub use app::{WindowConfig, render_headless_rgba8, save_rgba8_ppm};
#[cfg(feature = "window")]
pub use app::{WindowError, run_window};
pub use color::{Color, Oklab};
pub use gpu::snap_physical_border_width;
pub use layout::{Align, AnchorPlacement, Axis, ComputedBox, LayoutSpec, MainAlign, NodeId, TrackSize, VirtualListIndex};
pub use pool::{Key, Realm, ScopeId, SlotPool, Spring};
pub use scale::{
    DisplayMetrics, EdidPhysicalSize, ResolvedScale, SCALE_LATTICE_120, ScalePolicy, ScaleReason,
    parse_edid_physical_mm, zoom_step,
};
pub use ui::{
    CursorIcon, GradientSpec, GuiSystem, KeyInput, PaintStyle, PointerEvent, ShadowSpec,
    StatelessView, TextEditState, Theme, Ui,
};
