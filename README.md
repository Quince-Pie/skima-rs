# Skima 1.0 (`skima`)

**High-efficiency Rust GUI & Incremental Axis-Factored Layout Engine (`IALE`)** co-designed around **Rust 1.100.0 (Edition 2024)** ownership, lifetimes, and RAII — with **Automatic Fractional HiDPI Scaling (`ScalePolicy::Auto`)**, native **Vulkan 1.3 ReBAR + Hardware Dual-Source ClearType** rendering, and **0 B steady-state heap allocation** across scale resolution, layout, vector rasterization, and GPU/CPU frame updates.

---

## Why `skima`?

Conventional Rust GUI toolkits often trade fighting the borrow checker (`Rc<RefCell<...>>`, runtime entity lease panics, message-passing boilerplate, or full-tree VDOM diffing) against GPU staging bottlenecks and quadratic flex-wrap text measurement.

`skima` co-designs state ownership, retained reconciliation, layout, text/vector rasterization, and GPU submission around explicit phase invariants:

1. **Two-Lifetime Retained Cursor `Ui<'sys, 's, S>` (`0` `RefCell`s, `0` Macros):**
   - Decouples mutable engine state (`'sys`) from shared application state (`&'s S`). Read `&ui.state().items` directly while mutating the retained `SlotTape` with zero borrow-checker friction.
   - Collision-free per-parent callsite ordinal + `ui.keyed(...)` reconciliation preserves local hooks (`use_state`, `use_state_updated`) across unkeyed `for` loops, conditionals, and dynamic reorders, alongside zero-allocation state projections (`ui.lens`), `StatelessView` (`+ use<...>`), 3-regime closed-form analytical springs (`use_spring`, `use_spring_damped`), and `core::mem::DropGuard` unwind rollback.
   - Full-featured 1.0 widget kit including **UTF-8 `ui.text_input`** (`TextEditState` with non-destructive `Ctrl+A`, `Shift+Arrows`, `Ctrl+Shift+Arrows`, mouse click + drag + double-click word selection, translucent selection highlight quad, and `Ctrl+C` / `Ctrl+X` / `Ctrl+V` clipboard integration) and **Pass-4 viewport-flipping hover tooltips** (`ui.tooltip`, `ui.with_tooltip`).
2. **Typed Generational `SlotPool<T>` & Multi-Pool `Realm`:**
   - Compact 8-byte `Copy` `Key<T>` handles indexing dense contiguous storage with wraparound-immune slot retirement, `iter` / `iter_mut` / `retain` / generation-safe `clear`, simultaneous disjoint mutable borrows of same-type entities (`SlotPool::get_disjoint_mut`) and distinct-type pools (`Realm::pools_disjoint_mut`), reactive `ScopeId` invalidation, and `DropGuard` transactional updates (`try_update_transactional`).
3. **128 B/node SoA Incremental Axis-Factored Layout (`IALE`) & Fenwick Virtualizer:**
   - **Record Payload:** `TopoNode` (16 B) + `LayoutSpec` (64 B, 1 cache line) + `ComputedBox` (32 B) + `ClipRect` (16 B) = **128 B/node**.
   - **Non-Recursive 4-Pass Solver:** Iterative DFS across 1D Flex (`Row`, `Col` with unbounded min/max water-filling), `WrapRow` (per-line `cross_align` and `main_align`), 2D `Grid` (`Fixed`, `Fit`, `Auto`, `Percent`, `MinMax`, `Fr` + CSS §11.5/§11.6 multi-track `1..=255` `col_span`/`row_span`), deep `FirstBaseline`, and **Pass 4** 12-variant viewport-flipping `AnchorPlacement` overlays (`BottomEnd`, `TopCenter`, etc.).
   - **Sublinear Incremental & Fractional-Pixel Solving:** **Scroll-Only Fast Path (`FLAG_DIRTY_SCROLL` / `set_scroll_y`)** translating descendants, nested `clip_rects`, and Pass-4 anchors bit-identically without re-solving widths or wrapping; 53-bit `f64` reciprocal **Physical Device-Pixel Edge Snapping** (`LayoutTree::set_device_scale`, `0.0` physical-pixel seams across contiguous siblings and `snap_ceil_px` `Fit` label protection across fractional scales); 2D Relayout Boundaries + 1D Inline-Only Relayout Skip; **Exact `[W_tight, W_next_unwrap)` Width-Interval Text-Wrap Cache**; and 106-bit `two_sum` zero-drift **$O(\log N)$ Fenwick `VirtualListIndex`** (`visible_range`, `visible_range_overscan`, `set_height`, `push`, `pop`, `resize`).
4. **OpenType Shaping (`F26Dot6`), Elastic `GlyphAtlas`, Zero-Alloc `Path2D` + `RasterScratch`, & Perceptual `Oklab`:**
   - OpenType shaping (`rustybuzz` + `swash`), zero-C OS font discovery, per-glyph fallback + inline ZWJ/VS16 color emoji & `\t` tab stops, deterministic **`F26Dot6` (`1/64` physical px)** `GlyphKey` quantization + integer-snapped horizontal line baselines, zero-allocation `$O(\text{clusters})$` `ShapedParagraph::{byte_offset_to_x, x_to_byte_offset}` caret hit-testing, texel-preserving dynamic `1024 -> 2048 -> 4096` `GlyphAtlas` growth, and normalized `[1, 2, 3, 2, 1] / 9` FIR ClearType subpixel filtering.
   - Zero-allocation unified-contour round-cap/round-join `Path2D::stroke_polyline` (`Path2D::clear` + `RasterScratch` + `rasterize_into` — **0 B steady-state heap**) & Wang's-formula Bézier flattening into a tight-span active-edge analytical signed-area rasterizer (wired directly into `ui.vector_canvas`).
   - CSS Color 4 premultiplied Björn Ottosson `Oklab` / `Oklch` interpolation, sRGB gamut clipping, and WCAG 2.1 / APCA-W3 contrast utilities.
5. **Native Vulkan 1.3 (`ash`) ReBAR + Span-Culled `CpuRenderer` Reference Backend:**
   - Precompiled SPIR-V 1.3 (`build.rs` via `naga`) — **0.0 ms** startup shader compilation, vertex-stage clip/scissor culling, Sterbenz-exact `rounded_box_cov_edges` + `snap_physical_border_width` (`0` row-divider Moiré striping across fractional scales), exact `1:1` integer `textureLoad` glyph sampling, and fragment inner-safe-rect clip trivial accept.
   - **Persistently Mapped ReBAR (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) SSBOs** patching coalesced `dirty_prim_ranges` in place, with merged `nonCoherentAtomSize` flushes for non-coherent, integrated, and software (`lavapipe`) Vulkan 1.3 devices.
   - **Single-Draw-Call Hardware Dual-Source ClearType (`VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR`)** interleaved with 4-corner analytical SDF rounded boxes & borders, premultiplied OKLab gradients, separable `erf` Gaussian drop shadows, arbitrary-depth rounded clip chains, and swapchain `buffer_age` `DamageTracker` scissoring — matched within `<= 3 LSB` by a 64-prim hierarchical-bounds + active-scanline `CpuRenderer`.
6. **Automatic Fractional Scaling Engine (`src/scale.rs`):**
   - Multi-objective `ScalePolicy::Auto` optimizer (`90 ns`, `0 B` heap) combining orientation-invariant ergonomic viewing-distance visual acuity (`PPD` vs W3C CSS `46.92 PPD`), Wayland `wp_fractional_scale_v1` (`1/120`) + IEEE-754 exact `15k/120 == k/8` dyadic grid harmony, `1.0×` native pixel-identity deadzone protection, `u64` integer logical viewport divisibility, cached read-only Linux `/sys/class/drm` VESA E-EDID millimeter probing, and live `Ctrl+=` / `Ctrl+-` / `Ctrl+0` scale-lattice zoom.
7. **Incremental `AccessKit 0.24` Accessibility:**
   - Zero-allocation in-place semantics sync (`sync_semantics_in_place`), root HiDPI `A11yAffine::scale` coordinate transform, dirty-slot `TreeUpdate` export, and `Focus`, `Click`, `Increment`, `Decrement`, and `SetValue` action routing.

---

## Feature Flags

| Feature | Default | Description |
| :--- | :---: | :--- |
| `window` | **Yes** | Enables `skima::run_window` and `WindowError` via `winit 0.30` (Wayland `dlopen` + X11) with `ScalePolicy::Auto`, `Ctrl+=`/`Ctrl+-`/`Ctrl+0` lattice zoom, demand-driven `ControlFlow::Wait` (0% idle CPU/GPU), keyboard selection/clipboard shortcuts, and automatic VSync spring animation scheduling. |
| *(none)* | `--no-default-features` | Headless `IALE` layout, retained `GuiSystem`, offscreen hardware Vulkan 1.3 (`VulkanRenderer::new_headless`, `render_headless_rgba8`), and `CpuRenderer` with zero window-system dependencies. |

---

## Quickstart (`examples/hello_counter.rs`)

```rust,no_run
use skima::{Align, Color, LayoutSpec, MainAlign, PaintStyle, Theme, Ui, WindowConfig};

struct CounterApp {
    title: String,
    count: i32,
    dark_mode: bool,
}

fn view(ui: &mut Ui<'_, '_, CounterApp>) {
    ui.sys.theme = if ui.state().dark_mode { Theme::dark() } else { Theme::light() };
    let th = ui.theme();
    ui.sys.clear_color = th.bg_canvas;

    ui.element(
        LayoutSpec::col().grow(1.0).pad(32.0).align(Align::Center).justify(MainAlign::Center),
        PaintStyle::solid(th.bg_canvas),
        |ui| {
            ui.element(
                LayoutSpec::col().width_fixed(436.0).pad(24.0).gap(16.0),
                PaintStyle::solid(th.bg_surface)
                    .rounded(th.radius_lg)
                    .border(1.0, th.border_subtle)
                    .drop_shadow(14.0, [0.0, 8.0], Color::rgba8(0, 0, 0, 115)),
                |ui| {
                    let banner = format!("{} — Count: {}", ui.state().title, ui.state().count);
                    ui.heading(&banner);
                    // Supports Ctrl+A, Shift+Arrows, drag-select, double-click word select, & Ctrl+C/X/V:
                    ui.text_input("Project title...", |s| &s.title, |s| &mut s.title);
                    ui.progress_bar((ui.state().count.clamp(0, 20) as f32) / 20.0, 388.0, th.accent);

                    ui.element(
                        LayoutSpec::row().width_grow(1.0).justify(MainAlign::SpaceBetween).align(Align::Center),
                        PaintStyle::default(),
                        |ui| {
                            ui.element(LayoutSpec::row().gap(10.0), PaintStyle::default(), |ui| {
                                let inc = ui.primary_button("+ Increment", |s, _| s.count += 1);
                                ui.tooltip(inc, "Increase counter (animates spring progress)");
                                let dec = ui.button("- Decrement", th.bg_control, |s, _| s.count -= 1);
                                ui.tooltip(dec, "Decrease counter");
                            });
                            let dark = ui.state().dark_mode;
                            let sw = ui.toggle_switch("Dark", dark, |s, _| s.dark_mode = !s.dark_mode);
                            ui.tooltip(sw, "Toggle calibrated Dark / Light OKLCH theme");
                        },
                    );
                },
            );
        },
    );
}

fn main() {
    let app = CounterApp { title: "Hello Skima 1.0 🦀".into(), count: 7, dark_mode: true };
    skima::run_window(WindowConfig::default(), app, view).unwrap();
}
```

---

## Core 1.0 Patterns

### Disjoint Pool Borrows, Live $O(\log N)$ `VirtualListIndex::push`, Tooltips, & Zero-Alloc Vector Canvas

```rust,no_run
use skima::vector::{FillRule, Path2D, RasterScratch};
use skima::{AnchorPlacement, Key, LayoutSpec, PaintStyle, SlotPool, Ui, VirtualListIndex};

struct Task { title: String, prio: f32 }
struct AppState {
    pool: SlotPool<Task>,
    keys: Vec<Key<Task>>,
    vlist: VirtualListIndex,
    filter: String,
    scroll_y: f64,
    show_menu: bool,
}

impl AppState {
    fn swap_priorities(&mut self) {
        if let [k0, k1, ..] = self.keys.as_slice()
            && let Ok([a, b]) = self.pool.get_disjoint_mut([*k0, *k1])
        {
            core::mem::swap(&mut a.prio, &mut b.prio);
        }
    }

    fn stream_batch(&mut self, count: usize) {
        for _ in 0..count {
            self.vlist.push(34.0); // O(log N) Fenwick append without rebuilding
        }
    }
}

fn dashboard(ui: &mut Ui<'_, '_, AppState>) {
    let th = ui.theme();
    let menu_btn = ui.primary_button("Inspector", |s, _| s.show_menu = !s.show_menu);
    ui.tooltip(menu_btn, "Open Pass-4 viewport-clamped inspector");

    // Full UTF-8 selection (`Ctrl+A`, `Shift+Arrows`, drag-select) & clipboard (`Ctrl+C/X/V`):
    ui.text_input("Filter logs...", |s| &s.filter, |s| &mut s.filter);

    // Zero-allocation dynamic vector canvas (`Path2D::clear` + `RasterScratch` + `rasterize_into` under the hood):
    ui.vector_canvas(320, 48, 1, th.accent, FillRule::NonZero, |p| {
        p.stroke_polyline(&[[4.0, 36.0], [84.0, 12.0], [164.0, 34.0], [316.0, 10.0]], 2.4);
    });

    // Standalone zero-allocation vector rasterization into caller-owned buffers (`0 B` steady-state heap):
    let mut path = Path2D::new();
    let mut scratch = RasterScratch::new();
    let mut mask = Vec::new();
    path.clear();
    path.stroke_polyline(&[[0.0, 8.0], [32.0, 24.0]], 2.0);
    path.rasterize_into(32, 32, FillRule::NonZero, &mut scratch, &mut mask);

    // Turnkey non-virtualized scroll container (auto-pairs 14px scrollbar on overflow):
    ui.scroll_area(
        "Tasks",
        |s| s.scroll_y,
        |s, y| s.scroll_y = y,
        LayoutSpec::col().grow(1.0).pad(12.0).gap(8.0),
        PaintStyle::solid(th.bg_surface).rounded(th.radius_md),
        |ui| {
            for &k in &ui.state().keys {
                if let Some(t) = ui.state().pool.get(k) {
                    ui.ui_text(&t.title, 13.5, th.text_primary);
                }
            }
        },
    );

    // Pass 4 viewport-flipping popover anchored directly to `menu_btn`:
    if ui.state().show_menu {
        ui.anchored_popover(
            menu_btn,
            AnchorPlacement::BottomEnd,
            [0.0, 6.0],
            LayoutSpec::col().width_fixed(260.0).pad(12.0),
            PaintStyle::solid(th.bg_elevated).rounded(th.radius_md).border(1.0, th.accent),
            |ui| { ui.ui_text("Viewport-clamped overlay", 13.0, th.text_primary); },
        );
    }
}
```

---

## Automatic Fractional Scaling (`src/scale.rs`)

Desktop and embedded displays span a 5× range in pixel density (`68–342 PPI`) and a 6× range in viewing distance (`35 cm` handhelds to `211 cm` living-room 4K TVs), while X11 (`Xft.dpi = 96`) and unconfigured Wayland tiling compositors frequently report a naive `1.0×` scale on high-PPI panels. `skima` solves automatic fractional scaling end-to-end across display physics, floating-point representation, layout snapping, and GPU shading:

1. **Orientation-Invariant Visual Acuity (`PPD` vs W3C CSS Reference Pixel):**
   - Models continuous ergonomic viewing distance $d(W_{\text{mm}}, H_{\text{mm}})$ from `short_mm = min(W_mm, H_mm)` and `16:9` height-equivalent diagonal $D_h$, keeping $d$ invariant under 90° portrait rotation and keeping `21:9` / `32:9` desk ultrawides (`34"–49"`) in the desk regime (`~65 cm`) rather than misclassifying them as 50"+ TVs.
   - Computes native angular resolution $\text{PPD} = 20 \, d_{\text{cm}} \tan(0.5^\circ) / p_{\text{mm}}$ against the W3C CSS reference pixel ($\theta_0 = 2\arctan(1/5376) \approx 0.021315^\circ \implies \text{PPD}_0 \approx 46.9156\text{ px/deg}$), yielding continuous acuity target $s_{\text{raw}} = \text{PPD} / \text{PPD}_0$.
   - Recovers physical active millimeters on Linux via a zero-dependency VESA E-EDID 1.3/1.4 parser (`parse_edid_physical_mm`, validating checksum + 1 mm Detailed Timing Descriptors while rejecting `16×9` aspect-ratio firmware bugs) and read-only `/sys/class/drm/card*-*/edid` probing (`probe_linux_drm_physical_mm`), cached with **0 syscalls and 0 B heap** on steady redraw frames.
2. **Wayland `1/120` & IEEE-754 `15k/120 == k/8` Dyadic Token Harmony:**
   - Wayland `wp_fractional_scale_v1` encodes scale as `N / 120`. Because $120 = 8 \times 15$, every `1/8` dyadic step $k/8 = (15k)/120$ (`1.00×`, `1.125×`, `1.25×`, `1.375×`, `1.50×`, `1.625×`, `1.75×`, `1.875×`, `2.00×`, `2.25×`, `2.50×`) is simultaneously an **exact integer `N/120` Wayland numerator**, an **exact terminating IEEE-754 `f32` binary float** (`0` mantissa rounding), and an **exact `F26Dot6` (`1/64 px`) glyph rasterization key**.
   - Over a 228-byte `const` precomputed `LATTICE_TABLE` (`90 ns` p50, `0 B` heap), the optimizer jointly minimizes symmetric log-acuity error $(\ln(s / s_{\text{raw}}))^2$, binary design-grid penalty, a smooth $C^1$ `1.0×` native pixel-identity deadzone (`110..122 PPI`), `u64` integer logical viewport divisibility, Topkis log-workspace bounds (`>= 1024×720`), and `previous_120ths` deadband hysteresis.
3. **Zero-Seam, Zero-Moiré Fractional Rendering:**
   - `IALE` snaps `ComputedBox` edges to the physical pixel grid via 53-bit `f64` reciprocals + error diffusion (`0.0` physical-pixel seams across 100k `VirtualList` rows even at non-dyadic `4/3 = 160/120`), while `snap_physical_border_width` + Sterbenz-exact `rounded_box_cov_edges` eliminate row-divider Moiré striping (`<= 1 LSB` variance) and integer-snapped baselines + `1:1` `textureLoad` keep ClearType glyphs razor-sharp.

### Contender Family Comparison (30 Real-World Display Profiles)

| Contender Family | Ergonomic Scale Hits | RMS Log-Acuity Error | Mean Grid Harmony | Key Failure Mode Avoided by Contender D |
| :--- | :---: | :---: | :---: | :--- |
| **A. Linear `PPI / 96.0`** | `3 / 30` (10.0%) | `31.81%` | `0.208` | Ignores viewing distance (`65" 4K TV` at `1.0×`, `7"` handheld at `2.25×`); irrational beat scales (`1.133×`, `1.700×`). |
| **B. Fixed `25%` PPI Step Table** | `20 / 30` (66.7%) | `23.62%` | `0.942` | Collapses `7" 1280×800` handheld workspace to `640×400` (`2.0×`) and leaves `55"/65" 4K` TVs illegible at `1.0×`. |
| **C. Viewing-Distance CSS Acuity (`1/120`)** | `3 / 30` (10.0%) | `3.88%` | `0.276` | Lands on non-dyadic `1/120` fractions with repeating IEEE-754 mantissas; blurs `38"/42"/49"` desk displays to `1.04×–1.07×`. |
| **D. `ScalePolicy::Auto` (`skima`)** | **`30 / 30` (100%)** | **`3.23%`** | **`0.920`** | **100% exact `15k/120 == k/8` IEEE-754 dyadic + exact `F26Dot6` font keys across all 30 profiles (`90 ns`, `0 B` heap).** |

### `ScalePolicy` & `DisplayMetrics` API (`Ctrl+=` / `Ctrl+-` / `Ctrl+0` Built In)

```rust,no_run
use skima::{DisplayMetrics, GuiSystem, ScalePolicy, WindowConfig, zoom_step};

// 1. Windowed apps automatically use `ScalePolicy::Auto` + `Ctrl+=` / `Ctrl+-` / `Ctrl+0` lattice zoom:
let config = WindowConfig {
    scale_policy: ScalePolicy::Auto, // or PurePhysicalAuto, System, Fixed(1.25)
    ..WindowConfig::default()
};

// 2. Direct `0 B`-heap display-physics resolution (`~90 ns`):
let metrics = DisplayMetrics::from_physical_mm(3840, 2160, 597, 336); // 27" 4K (163.4 PPI)
let resolved = ScalePolicy::Auto.resolve(&metrics);
assert_eq!(resolved.scale_factor, 1.5);          // 180/120 = 3/2 exact dyadic scale
assert_eq!(resolved.wayland_120ths, 180);
assert_eq!((resolved.logical_width, resolved.logical_height), (2560.0, 1440.0));
assert!(resolved.is_dyadic_exact());

// 3. Programmatic lattice zoom on `GuiSystem` (or standalone `zoom_step`):
let mut sys = GuiSystem::new();
sys.resolve_display_scale(metrics);
assert_eq!(sys.zoom_scale(1).scale_factor, 1.625); // 195/120 = 13/8
assert_eq!(zoom_step(1.25, 1), 160.0 / 120.0);     // 160/120 = 4/3 Wayland step
sys.reset_scale_auto();                            // Restores 1.50x ScalePolicy::Auto
```

### Reference Display Scale Matrix (`skima_bench --print-scale-table`)

| Display Profile | Physical Px | Size (mm) | PPI | View Dist | Native PPD | Auto Scale | Wayland 120ths | Logical Viewport | Harmony |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7.0" Handheld 1280x800 | `1280×800` | `151×94` | 215.6 | 35.1 cm | 52.1 | **1.125×** | `135/120` (`9/8`) | `1138×711` | 0.74 |
| 8.8" Handheld 2560x1600 | `2560×1600` | `190×119` | 342.0 | 39.3 cm | 92.4 | **2.00×** | `240/120` (`2/1`) | `1280×800` | 1.00 |
| 13.3" Laptop 1920x1080 | `1920×1080` | `294×165` | 166.0 | 48.9 cm | 55.8 | **1.25×** | `150/120` (`5/4`) | `1536×864` | 0.85 |
| 13.6" Laptop 2560x1664 | `2560×1664` | `290×188` | 224.4 | 49.7 cm | 76.6 | **1.625×** | `195/120` (`13/8`) | `1575×1024` | 0.74 |
| 14.0" Laptop 1920x1200 | `1920×1200` | `302×189` | 161.4 | 50.5 cm | 56.0 | **1.25×** | `150/120` (`5/4`) | `1536×960` | 0.85 |
| 14.0" Laptop 2880x1800 | `2880×1800` | `302×189` | 242.1 | 50.5 cm | 84.0 | **1.75×** | `210/120` (`7/4`) | `1646×1029` | 0.85 |
| 14.2" Laptop 3024x1964 | `3024×1964` | `302×196` | 254.4 | 50.8 cm | 88.8 | **2.00×** | `240/120` (`2/1`) | `1512×982` | 1.00 |
| 15.0" Laptop 3:2 2496x1664 | `2496×1664` | `317×211` | 200.1 | 52.5 cm | 72.2 | **1.50×** | `180/120` (`3/2`) | `1664×1109` | 0.92 |
| 15.6" Laptop 1920x1080 | `1920×1080` | `345×194` | 141.4 | 53.6 cm | 52.1 | **1.125×** | `135/120` (`9/8`) | `1707×960` | 0.74 |
| 16.0" Laptop 2560x1600 | `2560×1600` | `345×215` | 188.6 | 54.5 cm | 70.7 | **1.50×** | `180/120` (`3/2`) | `1707×1067` | 0.92 |
| 16.0" Laptop 3840x2400 | `3840×2400` | `345×215` | 282.9 | 54.5 cm | 106.0 | **2.25×** | `270/120` (`9/4`) | `1707×1067` | 0.85 |
| 21.5" Desk 4096x2304 | `4096×2304` | `476×268` | 218.5 | 62.9 cm | 94.4 | **2.00×** | `240/120` (`2/1`) | `2048×1152` | 1.00 |
| 24.0" Desk 1920x1080 | `1920×1080` | `531×299` | 91.8 | 63.5 cm | 40.1 | **1.00×** | `120/120` (`1/1`) | `1920×1080` | 1.00 |
| 24.0" Desk 2560x1440 | `2560×1440` | `531×299` | 122.4 | 63.5 cm | 53.4 | **1.125×** | `135/120` (`9/8`) | `2276×1280` | 0.74 |
| 24.0" Desk 3840x2160 | `3840×2160` | `531×299` | 183.6 | 63.5 cm | 80.1 | **1.75×** | `210/120` (`7/4`) | `2194×1234` | 0.85 |
| 27.0" Desk 1920x1080 | `1920×1080` | `597×336` | 81.7 | 64.4 cm | 36.1 | **1.00×** | `120/120` (`1/1`) | `1920×1080` | 1.00 |
| 27.0" Desk 2560x1440 | `2560×1440` | `597×336` | 108.9 | 64.4 cm | 48.2 | **1.00×** | `120/120` (`1/1`) | `2560×1440` | 1.00 |
| 27.0" Desk 3840x2160 | `3840×2160` | `597×336` | 163.4 | 64.4 cm | 72.3 | **1.50×** | `180/120` (`3/2`) | `2560×1440` | 0.92 |
| 27.0" 4K Portrait 2160x3840 | `2160×3840` | `336×597` | 163.4 | 64.4 cm | 72.3 | **1.50×** | `180/120` (`3/2`) | `1440×2560` | 0.92 |
| 27.0" Studio 5K 5120x2880 | `5120×2880` | `597×336` | 217.8 | 64.4 cm | 96.4 | **2.00×** | `240/120` (`2/1`) | `2560×1440` | 1.00 |
| 27.6" DualUp 16:18 2560x2880 | `2560×2880` | `465×523` | 139.9 | 64.6 cm | 62.0 | **1.25×** | `150/120` (`5/4`) | `2048×2304` | 0.85 |
| 32.0" Desk 2560x1440 | `2560×1440` | `708×398` | 91.9 | 65.9 cm | 41.6 | **1.00×** | `120/120` (`1/1`) | `2560×1440` | 1.00 |
| 32.0" Desk 3840x2160 | `3840×2160` | `708×398` | 137.8 | 65.9 cm | 62.4 | **1.25×** | `150/120` (`5/4`) | `3072×1728` | 0.85 |
| 32.0" Pro 6K 6016x3384 | `6016×3384` | `699×393` | 218.6 | 65.8 cm | 98.8 | **2.00×** | `240/120` (`2/1`) | `3008×1692` | 1.00 |
| 34.0" UWQHD 21:9 3440x1440 | `3440×1440` | `797×334` | 109.6 | 64.7 cm | 48.7 | **1.00×** | `120/120` (`1/1`) | `3440×1440` | 1.00 |
| 38.0" UWQHD+ 24:10 3840x1600 | `3840×1600` | `880×367` | 110.8 | 65.5 cm | 49.9 | **1.00×** | `120/120` (`1/1`) | `3840×1600` | 1.00 |
| 42.0" Desk OLED 3840x2160 | `3840×2160` | `930×523` | 104.9 | 68.9 cm | 49.7 | **1.00×** | `120/120` (`1/1`) | `3840×2160` | 1.00 |
| 49.0" DQHD 32:9 5120x1440 | `5120×1440` | `1193×336` | 109.0 | 65.4 cm | 49.0 | **1.00×** | `120/120` (`1/1`) | `5120×1440` | 1.00 |
| 55.0" TV 4K 3840x2160 | `3840×2160` | `1218×685` | 80.1 | 178.8 cm | 98.4 | **2.00×** | `240/120` (`2/1`) | `1920×1080` | 1.00 |
| 65.0" TV 4K 3840x2160 | `3840×2160` | `1439×809` | 67.8 | 211.2 cm | 98.4 | **2.00×** | `240/120` (`2/1`) | `1920×1080` | 1.00 |

---

## Running Examples, Screenshots, & Benchmarks

```bash
# 1. Launch Flagship Interactive Workbench (Vulkan 1.3 Wayland/X11 Window + Scale Inspector):
cargo run --release --example skima_studio

# 2. Launch Minimal Counter App:
cargo run --release --example hello_counter

# 3. Run Headless Vulkan 1.3 Verification ([1.0x..2.0x] Fractional Scales x Dark & Light + 0 Oscillation + 0 Moiré + 101k Stream):
cargo run --release --example skima_studio -- --smoke-test

# 4. Export Lossless Vulkan 1.3 PNG Screenshot Suite (or Fractional Scale Proof / Custom Scale PNG):
cargo run --release --example skima_studio -- --export-screenshots ./screenshots
cargo run --release --example skima_studio -- --export-scale-proof ./screenshots
cargo run --release --example skima_studio -- --render-png studio_125x.png --size 1280x800 --scale 1.25

# 5. Run Complete Isolated Engine, ScalePolicy::Auto, & Hardware GPU Benchmark Suite:
cargo run --release --example skima_bench

# 6. Run Contract, Unwind-Safety, Property Fuzzing, Fractional Scaling, & Vulkan/CPU Parity Tests:
CARGO_BUILD_WARNINGS=deny cargo test
```

---

## Measured Performance (`AMD Ryzen 9 9950X3D` + `AMD Radeon RX 9070 XT`, `rustc 1.100.0`, `--release`)

Reproduce in an idle session via `cargo build --release --example skima_bench && ./target/release/examples/skima_bench` (`rustc 1.100.0 (0ef1bc386 2026-11-09)`, Linux, Vulkan 1.3 RADV `GFX1201` with `ReBAR=true`, `DualSrcBlend=true`):

| Operation | p50 | mean | p99 | ±SIQR | N | Heap Allocs | Verified Details |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **1. IDE 6,785n: 1-Leaf Text + Layout** | **5.70 µs** | 5.84 µs | 10.26 µs | ±0.02 µs | 500 | **0 (0 B)** | 63/64 scopes skipped, 2n solved |
| **2. IDE 6,785n: Paint-Only Hover Patch** | **5.60 µs** | 5.66 µs | 8.95 µs | ±0.03 µs | 500 | **0 (0 B)** | 0 layout visited, 1 quad + damage |
| **3. IALE Scroll Fast-Path (1,026n)** | **9.33 µs** | 9.64 µs | 17.37 µs | ±0.06 µs | 500 | **0 (0 B)** | `FLAG_DIRTY_SCROLL`, 0 wrap/shaper calls |
| **4. Smooth Resize (512 Wrapped Texts)** | **12.87 µs** | 13.69 µs | 31.48 µs | ±0.06 µs | 300 | **0 (0 B)** | 96.7% `[W_tight, W_next_unwrap)` hit (`620..320px`) |
| **5a. 100k Var-Height VirtualList Scroll** | **4.10 µs** | 4.05 µs | 5.64 µs | ±0.12 µs | 300 | **0 (0 B)** | Fenwick $O(\log N)$ + slot reuse |
| **5b. 100k Fenwick `set_height` + Query** | **0.22 µs** | 0.26 µs | 0.66 µs | ±0.05 µs | 1000 | **0 (0 B)** | 106-bit `two_sum`, 17 bit-hops, 0 B heap |
| **5c. 100k Fenwick Live `push` + `pop`** | **0.02 µs** | 0.02 µs | 0.03 µs | ±0.00 µs | 1000 | **0 (0 B)** | $O(\log N)$ stream append, 0 B heap |
| **6. `SlotPool<T>` 1,024n `get_disjoint_mut`** | **0.02 µs** | 0.02 µs | 0.03 µs | ±0.00 µs | 1000 | **0 (0 B)** | Zero-RefCell generation check |
| **7. `Path2D` `stroke_polyline` + Scratch** | **33.50 µs** | 33.66 µs | 36.91 µs | ±0.24 µs | 200 | **0 (0 B)** | `RasterScratch` + `rasterize_into` (`520×62` R8) |
| **8a. 1080p Hover E2E + CPU Scissored** | **32.78 µs** | 32.88 µs | 36.12 µs | ±0.05 µs | 150 | **0 (0 B)** | 64-prim hierarchical bounds + scanline SDF |
| **8b. 1080p Hover E2E + Vulkan 1.3 GPU** | **53.65 µs** | 55.78 µs | 113.38 µs | ±2.78 µs | 100 | **0 (0 B)** | `ReBAR=true`, `DualSrc=true` + Fence wait |
| **9a. `ScalePolicy::Auto` Resolve** | **0.09 µs** | 0.09 µs | 0.11 µs | ±0.01 µs | 2000 | **0 (0 B)** | **90 ns p50**, `30/30` reference display hits |
| **9b. Fractional `1.333×` Scroll (1,026n)** | **9.53 µs** | 10.05 µs | 16.99 µs | ±0.37 µs | 500 | **0 (0 B)** | `160/120` (`4/3`) device-pixel snap, 0 seams |
| **9c. Fractional `1.25×` Hover + CPU** | **44.77 µs** | 45.34 µs | 48.88 µs | ±0.86 µs | 150 | **0 (0 B)** | `150/120` -> `2400×1350` physical scissored |
| **9d. Fractional `1.333×` Hover + Vulkan** | **50.13 µs** | 49.82 µs | 68.62 µs | ±1.67 µs | 100 | **0 (0 B)** | `160/120` -> `2560×1440` QHD physical + Fence |

### Memory & Binary Footprint Audit

- **Layout SoA Record Payload:** `128 B/node` (`TopoNode = 16 B`, `LayoutSpec = 64 B`, `ComputedBox = 32 B`, `ClipRect = 16 B` — `848.1 KiB` payload across `6,785` IDE nodes).
- **GPU Scene Records:** `GpuPrimitive = 64 B` (1 cache line), `GpuClip = 48 B`.
- **Precompiled SPIR-V 1.3 Uber-Shaders:** `UBER_VERT_SPV = 22,472 B`, `UBER_FRAG_DUAL_SRC_SPV = 28,768 B` embedded at compile time via `naga` (`0 B` runtime shader compiler overhead; elastic `1024×1024` -> `4096×4096` dual-plane `GlyphAtlas`).

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
