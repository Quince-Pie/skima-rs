//! **Skima Studio 1.0** — Flagship Interactive Workbench & Showcase Application.
//!
//! Exercises every subsystem of `skima` in one cohesive desktop application:
//! 1. **Live Perceptual OKLCH Theme Engine & Pass-4 Hover Tooltips:** Instant Dark/Light switching +
//!    0..360° Accent Hue slider, calibrated contrast across dark & light surfaces, and viewport-flipping
//!    `ui.tooltip` overlays on header and toolbar controls.
//! 2. **Responsive 2D Spanned Grid + Analytical SDF Cards + Zero-Alloc `Path2D` Vector Waveform & Icons:**
//!    Reflows seamlessly from compact `960x640` laptops up to 4K tiling desktops and `2.0x` Retina HiDPI.
//! 3. **Live-Streaming 100,000+ Row Variable-Height VirtualList Explorer:** Fenwick $O(\log N)$ scrolling,
//!    live `"+ Stream +1,000 Logs"` (`VirtualListIndex::push`) dynamic growth, full-selection UTF-8
//!    `ui.text_input` (`Ctrl+A`, `Shift+Arrows`, drag-select, `Ctrl+C`/`Ctrl+V`), and click-to-expand rows.
//! 4. **Generational Task Pool + Viewport-Clamped Popover:**
//!    Identity-keyed reconciliation, generational slot recycling, and button-anchored floating inspector.
//! 5. **Interactive Fractional Scaling Explorer (`src/scale.rs`):** Live `ResolvedScale`
//!    display-physics telemetry (`Scale`, Wayland `120ths`, `PPI`, `Viewing Dist`, `PPD`, `Grid Harmony`),
//!    interactive scale-preview chips (`Auto`, `1.0×`..`2.0×`), and `Ctrl+=` / `Ctrl+-` / `Ctrl+0` lattice zoom.
//! 6. **Live Engine Telemetry HUD & Headless Vulkan 1.3 Verification:** Displays live `IALE` node counts,
//!    GPU primitives, and cached ClearType glyphs; exports lossless PNG/PPM frames via `--export-screenshots`.
//!
//! Run interactively:
//!   `cargo run --release --example skima_studio`
//! Headless GPU verification / screenshot export:
//!   `cargo run --release --example skima_studio -- --smoke-test`
//!   `cargo run --release --example skima_studio -- --export-screenshots <dir>`
//!   `cargo run --release --example skima_studio -- --export-scale-proof <dir>`
//!   `cargo run --release --example skima_studio -- --render-png studio_frame.png`
//!   `cargo run --release --example skima_studio -- --render-ppm studio_frame.ppm`

use skima::a11y::SemanticsMeta;
use skima::gpu::cpu::CpuRenderer;
use skima::gpu::vulkan::VulkanRenderer;
use skima::layout::FLAG_ALIVE;
use skima::scale::SCALE_1_33;
use skima::text::GlyphAtlas;
use skima::vector::FillRule;
use skima::{
    Align, AnchorPlacement, Axis, Color, CursorIcon, DisplayMetrics, GuiSystem, Key, KeyInput,
    LayoutSpec, MainAlign, NodeId, PaintStyle, ResolvedScale, ScalePolicy, SlotPool, Theme,
    TrackSize, Ui, VirtualListIndex, save_rgba8_ppm,
};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StudioTab {
    Overview,
    VirtualLogs,
    PoolTasks,
}

impl StudioTab {
    const ALL: [Self; 3] = [Self::Overview, Self::VirtualLogs, Self::PoolTasks];
}

struct TaskItem {
    uid: u64,
    title: String,
    priority_pct: f32,
    done: bool,
}

#[derive(Default)]
struct StudioControlNodes {
    dark_toggle: AtomicU32,
    scale_badge: AtomicU32,
    scale_auto_btn: AtomicU32,
    scale_125_btn: AtomicU32,
    scale_133_btn: AtomicU32,
    scale_150_btn: AtomicU32,
    stream_btn: AtomicU32,
    filter_input: AtomicU32,
    task_input: AtomicU32,
    swap_btn: AtomicU32,
}

struct StudioState {
    active_tab: StudioTab,
    is_dark: bool,
    accent_hue: f32,
    pending_scale_policy: Cell<Option<ScalePolicy>>,
    // Overview telemetry & spring playground:
    target_load: f32,
    spring_hz: f32,
    wave_phase: u64,
    show_inspector_popover: bool,
    // 100k+ VirtualList explorer & live stream:
    search_query: String,
    vlist: VirtualListIndex,
    expanded_rows: BTreeSet<usize>,
    log_scroll_y: f64,
    streamed_logs: usize,
    // SlotPool<TaskItem> Kanban / Disjoint-Borrow demo:
    tasks: SlotPool<TaskItem>,
    task_keys: Vec<Key<TaskItem>>,
    next_task_uid: u64,
    new_task_title: String,
    task_scroll_y: f64,
    swap_count: usize,
    recycled_count: usize,
    nodes: StudioControlNodes,
}

const LOG_ROW_COLLAPSED_H: f32 = 34.0;
const LOG_ROW_EXPANDED_H: f32 = 88.0;

fn fmt_usize(mut n: usize) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    let mut groups = Vec::with_capacity(6);
    while n >= 1_000 {
        groups.push(format!("{:03}", n % 1_000));
        n /= 1_000;
    }
    let mut out = n.to_string();
    for g in groups.iter().rev() {
        out.push(',');
        out.push_str(g);
    }
    out
}

impl StudioState {
    fn new() -> Self {
        let mut vlist = VirtualListIndex::new(100_000, LOG_ROW_COLLAPSED_H);
        let mut expanded_rows = BTreeSet::new();
        expanded_rows.insert(2);
        vlist.set_height(2, LOG_ROW_EXPANDED_H);

        let mut tasks = SlotPool::new();
        let mut task_keys = Vec::new();
        let mut next_uid = 1u64;
        for (title, prio, done) in [
            ("Persistently mapped ReBAR Vulkan 1.3 SSBO pipeline", 0.96, true),
            ("Dual-Source ClearType RGB subpixel text blending", 0.92, true),
            ("128 B/node SoA IALE layout + scroll-only fast path", 0.88, true),
            ("Exact [W_tight, W_next_unwrap) line-break interval cache", 0.85, true),
            ("Analytical Path2D polyline stroke & scanline rasterizer", 0.81, true),
            ("Viewport-clamped popover anchoring & hover tooltips", 0.78, true),
            ("UTF-8 text selection, word navigation & clipboard sync", 0.74, false),
            ("Task dependency & priority linker", 0.69, false),
            ("Overflow scroll container with paired scrollbar", 0.64, false),
            ("Closed-form analytical spring animation solver", 0.58, false),
            ("Incremental AccessKit 0.24 accessibility tree export", 0.52, false),
        ] {
            let uid = next_uid;
            next_uid += 1;
            task_keys.push(tasks.insert(TaskItem {
                uid,
                title: title.to_owned(),
                priority_pct: prio,
                done,
            }));
        }

        Self {
            active_tab: StudioTab::Overview,
            is_dark: true,
            accent_hue: 256.0,
            pending_scale_policy: Cell::new(None),
            target_load: 0.76,
            spring_hz: 4.5,
            wave_phase: 1,
            show_inspector_popover: false,
            search_query: String::from("vulkan"),
            vlist,
            expanded_rows,
            log_scroll_y: 0.0,
            streamed_logs: 0,
            tasks,
            task_keys,
            next_task_uid: next_uid,
            new_task_title: String::from("Ship skima 1.0.0 stable release 🦀"),
            task_scroll_y: 0.0,
            swap_count: 0,
            recycled_count: 0,
            nodes: StudioControlNodes::default(),
        }
    }

    /// Dynamically appends `count` log entries in $O(\log N)$ per item via `VirtualListIndex::push`
    /// without rebuilding the Fenwick prefix tree (`0.0 px` scroll jump on existing rows, or live
    /// `tail -f` follow when already viewing near the stream tail!).
    fn stream_logs(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        let prev_len = self.vlist.len();
        let prev_total_h = self.vlist.total_height();
        let near_tail =
            prev_len == 0 || prev_total_h <= 800.0 || self.log_scroll_y + 900.0 >= prev_total_h;

        for i in 0..count {
            let idx = prev_len + i;
            let h = if i == 0 {
                self.expanded_rows.insert(idx);
                LOG_ROW_EXPANDED_H
            } else {
                LOG_ROW_COLLAPSED_H
            };
            self.vlist.push(h);
        }
        self.streamed_logs = self.streamed_logs.saturating_add(count);

        if near_tail {
            let tail_idx = self.vlist.len().saturating_sub(14);
            self.log_scroll_y = self.vlist.offset_of(tail_idx);
        }
    }

    fn add_new_task(&mut self) {
        let trimmed = self.new_task_title.trim();
        if !trimmed.is_empty() {
            let uid = self.next_task_uid;
            self.next_task_uid += 1;
            let k = self.tasks.insert(TaskItem {
                uid,
                title: trimmed.to_owned(),
                priority_pct: 0.82,
                done: false,
            });
            self.task_keys.insert(0, k);
            self.new_task_title.clear();
        }
    }

    /// Demonstrates Rust 1.86+ `SlotPool::get_disjoint_mut`: mutates TWO distinct tasks in the same
    /// pool simultaneously with zero `RefCell` or `Entity` lease hazards, smoothly animating both
    /// retained `progress_bar` springs!
    fn swap_top_two_priorities(&mut self) {
        if self.task_keys.len() >= 2 {
            let k0 = self.task_keys[0];
            let k1 = self.task_keys[1];
            if let Ok([t0, t1]) = self.tasks.get_disjoint_mut([k0, k1]) {
                core::mem::swap(&mut t0.priority_pct, &mut t1.priority_pct);
                core::mem::swap(&mut t0.title, &mut t1.title);
                core::mem::swap(&mut t0.done, &mut t1.done);
                self.swap_count += 1;
            }
        }
    }

    fn remove_task(&mut self, key: Key<TaskItem>) {
        if self.tasks.remove(key).is_some() {
            self.task_keys.retain(|&k| k != key);
            self.recycled_count += 1;
        }
    }
}

/// Calibrated contrast helpers so semantic ink & elevation shadows remain crisp in BOTH Dark & Light themes.
#[derive(Clone, Copy)]
struct StudioPalette {
    ink_accent: Color,
    ink_success: Color,
    ink_warning: Color,
    card_shadow: Color,
    popover_shadow: Color,
}

impl StudioPalette {
    fn from_theme(th: Theme) -> Self {
        if th.is_dark {
            Self {
                ink_accent: th.accent.shift_lightness(0.06),
                ink_success: th.success,
                ink_warning: th.warning,
                card_shadow: Color::rgba8(0, 0, 0, 72),
                popover_shadow: Color::rgba8(0, 0, 0, 168),
            }
        } else {
            Self {
                ink_accent: th.accent.shift_lightness(-0.08),
                ink_success: th.success.shift_lightness(-0.11),
                ink_warning: th.warning.shift_lightness(-0.16),
                card_shadow: Color::rgba8(15, 23, 42, 22),
                popover_shadow: Color::rgba8(15, 23, 42, 54),
            }
        }
    }
}

/// Responsive metrics derived directly from the logical viewport (`960x640` compact up to 4K tiling).
#[derive(Clone, Copy)]
struct ResponsiveMetrics {
    compact_w: bool,
    compact_h: bool,
    rail_w: f32,
    pane_pad: f32,
    pane_gap: f32,
    content_w: f32,
}

impl ResponsiveMetrics {
    fn for_viewport(win_w: f32, win_h: f32) -> Self {
        let w = if win_w > 1.0 { win_w } else { 1280.0 };
        let h = if win_h > 1.0 { win_h } else { 800.0 };
        let compact_w = w < 1100.0;
        let compact_h = h < 720.0;
        let rail_w = if compact_w { 196.0 } else { 224.0 };
        let pane_pad = if compact_w || compact_h { 16.0 } else { 22.0 };
        let pane_gap = if compact_h { 12.0 } else { 16.0 };
        let content_w = (w - rail_w - 2.0 * pane_pad).max(320.0);
        Self {
            compact_w,
            compact_h,
            rail_w,
            pane_pad,
            pane_gap,
            content_w,
        }
    }
}

#[inline]
fn fmt_scale_short(rs: &ResolvedScale) -> String {
    if rs.wayland_120ths.is_multiple_of(30) {
        format!("{:.2}×", rs.scale_factor)
    } else {
        format!("{:.3}×", rs.scale_factor)
    }
}

#[inline]
fn current_resolved_scale(sys: &GuiSystem, fallback_w: f32, fallback_h: f32) -> ResolvedScale {
    sys.resolved_scale.unwrap_or_else(|| {
        let sf = sys.scale_factor.max(0.5);
        ScalePolicy::Fixed(sf).resolve(&DisplayMetrics::new(
            (fallback_w * sf).round().max(1.0) as u32,
            (fallback_h * sf).round().max(1.0) as u32,
        ))
    })
}

fn build_studio(ui: &mut Ui<'_, '_, StudioState>) {
    if let Some(policy) = ui.state().pending_scale_policy.take() {
        let _ = ui.sys.set_scale_policy(policy);
    }

    let is_dark = ui.state().is_dark;
    let hue = ui.state().accent_hue;
    ui.sys.theme = Theme::with_accent_hue(is_dark, hue);
    let th = ui.theme();
    ui.sys.clear_color = th.bg_canvas;

    let pal = StudioPalette::from_theme(th);
    let mut inspector_btn_node = NodeId::NONE;

    // Root Workspace Shell
    ui.element(
        LayoutSpec::col().grow(1.0),
        PaintStyle::solid(th.bg_canvas),
        |ui| {
            let root_box = ui.active_box();
            let rm = ResponsiveMetrics::for_viewport(root_box.w, root_box.h);
            let vw_est = if root_box.w > 1.0 { root_box.w } else { 1280.0 };
            let vh_est = if root_box.h > 1.0 { root_box.h } else { 800.0 };
            let rs = current_resolved_scale(ui.sys, vw_est, vh_est);

            // 1. Top Perceptual OKLab Gradient Header Bar
            let header_h = if rm.compact_h { 50.0 } else { 56.0 };
            ui.element(
                LayoutSpec::row()
                    .width_grow(1.0)
                    .height_fixed(header_h)
                    .pad_xy(if rm.compact_w { 14.0 } else { 20.0 }, 8.0)
                    .align(Align::Center)
                    .justify(MainAlign::SpaceBetween),
                PaintStyle::oklab_gradient(
                    th.bg_surface,
                    th.accent_muted.mix_oklab(th.bg_surface, 0.58),
                    0.25,
                )
                .border(1.0, th.border_subtle),
                |ui| {
                    // Brand Logo + Analytical Vector Diamond Icon + Version Badge
                    ui.element(
                        LayoutSpec::row().gap(12.0).align(Align::Center),
                        PaintStyle::default(),
                        |ui| {
                            ui.vector_icon(26, 26, th.accent, FillRule::EvenOdd, |p| {
                                p.move_to(13.0, 1.0)
                                    .line_to(25.0, 13.0)
                                    .line_to(13.0, 25.0)
                                    .line_to(1.0, 13.0)
                                    .close()
                                    .move_to(13.0, 7.0)
                                    .line_to(19.0, 13.0)
                                    .line_to(13.0, 19.0)
                                    .line_to(7.0, 13.0)
                                    .close();
                            });
                            ui.ui_text("Skima Studio", 18.0, th.text_primary);
                            ui.badge("v1.0.0 STABLE", th.success);
                        },
                    );

                    // Live OKLCH Theme Controls + Fractional Scale Badge + Inspector Trigger in Header:
                    ui.element(
                        LayoutSpec::row()
                            .gap(if rm.compact_w { 10.0 } else { 14.0 })
                            .align(Align::Center),
                        PaintStyle::default(),
                        |ui| {
                            let hue_lbl = format!("OKLCH {:>3.0}°", ui.state().accent_hue);
                            ui.mono_text(&hue_lbl, 12.0, th.text_secondary);
                            let cur_hue = ui.state().accent_hue;
                            let hue_slider_w = if rm.compact_w { 104.0 } else { 136.0 };
                            let hue_node =
                                ui.slider("Accent Hue", cur_hue, 0.0..=360.0, hue_slider_w, |s, v| {
                                    s.accent_hue = v;
                                });
                            ui.tooltip(hue_node, "Shift live OKLCH accent hue (0°–360°)");

                            let dark = ui.state().is_dark;
                            let dark_node = ui.toggle_switch("Dark", dark, |s, _| {
                                s.is_dark = !s.is_dark;
                            });
                            ui.state()
                                .nodes
                                .dark_toggle
                                .store(dark_node.0, Ordering::Relaxed);
                            ui.tooltip(
                                dark_node,
                                "Switch Dark & Light OKLCH palettes",
                            );

                            let scale_short = fmt_scale_short(&rs);
                            let scale_badge_txt = if rm.compact_w {
                                scale_short
                            } else if rs.reason.is_auto()
                                || matches!(
                                    ui.sys.scale_policy,
                                    ScalePolicy::Auto | ScalePolicy::PurePhysicalAuto
                                )
                            {
                                format!("{scale_short} Auto")
                            } else {
                                format!("{scale_short} ({}/120)", rs.wayland_120ths)
                            };
                            let scale_badge_col = if rs.reason.is_auto() {
                                th.success
                            } else {
                                th.accent
                            };
                            let scale_badge_node = ui.badge(&scale_badge_txt, scale_badge_col);
                            ui.state()
                                .nodes
                                .scale_badge
                                .store(scale_badge_node.0, Ordering::Relaxed);
                            ui.tooltip(
                                scale_badge_node,
                                "Active display scale • Ctrl+= / Ctrl+- zoom, Ctrl+0 Auto",
                            );

                            let insp_open = ui.state().show_inspector_popover;
                            let insp_bg = if insp_open { th.accent } else { th.bg_control };
                            inspector_btn_node = ui.button("Inspector", insp_bg, |s, _| {
                                s.show_inspector_popover = !s.show_inspector_popover;
                            });
                            if !insp_open {
                                ui.tooltip(
                                    inspector_btn_node,
                                    "Open Engine & Scale Inspector",
                                );
                            }
                        },
                    );
                },
            );

            // 2. Main Body: Left Navigation Rail + Active Workbench View
            ui.element(
                LayoutSpec::row().grow(1.0),
                PaintStyle::default(),
                |ui| {
                    // Left Navigation Rail
                    ui.element(
                        LayoutSpec::col()
                            .width_fixed(rm.rail_w)
                            .height_grow(1.0)
                            .pad(if rm.compact_h { 12.0 } else { 14.0 })
                            .gap(8.0)
                            .justify(MainAlign::SpaceBetween),
                        PaintStyle::solid(th.bg_surface).border(1.0, th.border_subtle),
                        |ui| {
                            // Top navigation group + architecture facts
                            ui.element(
                                LayoutSpec::col().width_grow(1.0).gap(12.0),
                                PaintStyle::default(),
                                |ui| {
                                    ui.element(
                                        LayoutSpec::col().width_grow(1.0).gap(6.0),
                                        PaintStyle::default(),
                                        |ui| {
                                            ui.ui_text("WORKSPACES", 11.0, th.text_muted);
                                            let tab = ui.state().active_tab;

                                            nav_item(
                                                ui,
                                                1,
                                                "1. Telemetry & Grid",
                                                tab == StudioTab::Overview,
                                                |s| s.active_tab = StudioTab::Overview,
                                            );
                                            nav_item(
                                                ui,
                                                2,
                                                "2. 100k Virtual Logs",
                                                tab == StudioTab::VirtualLogs,
                                                |s| s.active_tab = StudioTab::VirtualLogs,
                                            );
                                            nav_item(
                                                ui,
                                                3,
                                                "3. Task Board",
                                                tab == StudioTab::PoolTasks,
                                                |s| s.active_tab = StudioTab::PoolTasks,
                                            );
                                        },
                                    );

                                    ui.divider();

                                    ui.element(
                                        LayoutSpec::col().width_grow(1.0).gap(5.0),
                                        PaintStyle::default(),
                                        |ui| {
                                            ui.ui_text("ARCHITECTURE", 11.0, th.text_muted);
                                            ui.mono_text("SoA: 128 B/node", 12.0, th.text_secondary);
                                            ui.mono_text("GPU: 64 B/prim", 12.0, th.text_secondary);
                                            ui.mono_text("Vulkan 1.3 ReBAR", 12.0, pal.ink_accent);
                                            ui.mono_text(
                                                "DualSrc ClearType",
                                                12.0,
                                                pal.ink_success,
                                            );
                                        },
                                    );
                                },
                            );

                            // Bottom keyboard navigation legend
                            ui.element(
                                LayoutSpec::col().width_grow(1.0).gap(4.0),
                                PaintStyle::default(),
                                |ui| {
                                    ui.divider();
                                    ui.mono_text("Tab / Shift+Tab: Focus", 11.0, th.text_muted);
                                    ui.mono_text("Ctrl+A / C / V: Edit", 11.0, th.text_muted);
                                    ui.mono_text("Ctrl+= / - / 0: Scale", 11.0, th.text_muted);
                                },
                            );
                        },
                    );

                    // Active Content Pane
                    ui.element(
                        LayoutSpec::col()
                            .grow(1.0)
                            .pad(rm.pane_pad)
                            .gap(rm.pane_gap),
                        PaintStyle::default(),
                        |ui| {
                            ui.set_clips_children(true);
                            match ui.state().active_tab {
                                StudioTab::Overview => render_overview_tab(ui, rm, pal),
                                StudioTab::VirtualLogs => render_virtual_logs_tab(ui, rm, pal),
                                StudioTab::PoolTasks => render_pool_tasks_tab(ui, rm, pal),
                            }
                        },
                    );
                },
            );

            // 3. Bottom Engine Telemetry Status Bar
            let live_nodes = ui
                .sys
                .layout
                .topo
                .iter()
                .filter(|n| (n.flags & FLAG_ALIVE) != 0)
                .count()
                + 3;
            let total_prims = ui.sys.primitives.len();
            let glyphs_cached = ui.sys.fonts.atlas.cached_glyph_count();
            let loaded_fonts = ui.sys.fonts.font_count();
            let status_h = if rm.compact_h { 28.0 } else { 30.0 };
            ui.element(
                LayoutSpec::row()
                    .width_grow(1.0)
                    .height_fixed(status_h)
                    .pad_xy(16.0, 4.0)
                    .align(Align::Center)
                    .justify(MainAlign::SpaceBetween),
                PaintStyle::solid(th.bg_surface).border(1.0, th.border_subtle),
                |ui| {
                    let left_status = if total_prims > 0 {
                        format!(
                            "IALE Live Nodes: {live_nodes:>3}  |  GPU Primitives: {total_prims:>4}  |  Cached Glyphs: {glyphs_cached:>4}  |  OS Fonts: {loaded_fonts}"
                        )
                    } else {
                        format!(
                            "IALE Live Nodes: {live_nodes:>3}  |  Vulkan 1.3 Dynamic Rendering  |  OS Fonts: {loaded_fonts}"
                        )
                    };
                    ui.mono_text(&left_status, 11.5, th.text_secondary);
                    if !rm.compact_w {
                        let right_status =
                            format!("{vw_est:.0}×{vh_est:.0} • {}", fmt_scale_short(&rs));
                        ui.mono_text(&right_status, 11.5, th.text_muted);
                    }
                },
            );

            // 4. Viewport-Clamped Floating Engine & Scale Inspector Popover anchored directly
            //    to the "Inspector" button (`AnchorPlacement::BottomEnd` across 960x640..4K):
            if ui.state().show_inspector_popover && !inspector_btn_node.is_none() {
                let sf = ui.sys.scale_factor;
                let vw = vw_est;
                let vh = vh_est;
                let phys_w = ((vw * sf) - 1e-4).ceil().max(1.0) as u32;
                let phys_h = ((vh * sf) - 1e-4).ceil().max(1.0) as u32;
                let aw = ui.sys.fonts.atlas.width;
                let ah = ui.sys.fonts.atlas.height;
                let cur_policy = ui.sys.scale_policy;
                let is_auto_policy = matches!(
                    cur_policy,
                    ScalePolicy::Auto | ScalePolicy::PurePhysicalAuto
                );
                let (num, den) = rs.reduced_ratio();
                let scale_short = fmt_scale_short(&rs);
                let reason_lbl = rs.reason.label();

                ui.anchored_popover(
                    inspector_btn_node,
                    AnchorPlacement::BottomEnd,
                    [0.0, 6.0],
                    LayoutSpec::col().width_fixed(432.0).pad(14.0).gap(7.0),
                    PaintStyle::solid(th.bg_elevated)
                        .rounded(th.radius_md)
                        .border(1.5, th.accent)
                        .drop_shadow(16.0, [0.0, 8.0], pal.popover_shadow),
                    |ui| {
                        ui.element(
                            LayoutSpec::row()
                                .width_grow(1.0)
                                .justify(MainAlign::SpaceBetween)
                                .align(Align::Center),
                            PaintStyle::default(),
                            |ui| {
                                ui.ui_text("Engine & Scale Inspector", 14.0, th.text_primary);
                                ui.badge(
                                    reason_lbl,
                                    if rs.reason.is_auto() {
                                        th.success
                                    } else {
                                        th.accent
                                    },
                                );
                            },
                        );
                        ui.divider();
                        ui.mono_text(
                            "Backend:  Vulkan 1.3 • DualSrc ClearType LCD",
                            11.5,
                            pal.ink_success,
                        );
                        let view_line =
                            format!("Viewport: {vw:.0}x{vh:.0} log -> {phys_w}x{phys_h} phys");
                        ui.mono_text(&view_line, 11.5, th.text_secondary);
                        let atlas_line =
                            format!("Atlas:    {aw}x{ah} Dual-Plane • F26Dot6 (1/64px)");
                        ui.mono_text(&atlas_line, 11.5, th.text_secondary);

                        ui.divider();

                        let scale_line = format!(
                            "Scale:    {scale_short:<6} ({:>3}/120 = {num}/{den} • {reason_lbl})",
                            rs.wayland_120ths
                        );
                        ui.mono_text(&scale_line, 11.5, pal.ink_accent);

                        let phys_line = match (rs.ppi, rs.viewing_distance_cm) {
                            (Some(ppi), Some(dist_cm)) => format!(
                                "Physics:  PPI {ppi:>5.1}  |  Viewing Dist {dist_cm:>4.1} cm"
                            ),
                            _ => String::from("Physics:  PPI N/A (virtual) | Viewing Dist N/A"),
                        };
                        ui.mono_text(&phys_line, 11.5, th.text_secondary);

                        let ppd_line = match rs.pixels_per_degree {
                            Some(ppd) => {
                                let eff = rs.effective_logical_ppd().unwrap_or(ppd);
                                format!(
                                    "Acuity:   PPD {ppd:>5.1} ({eff:>4.1} eff/px • raw {:.2}×)",
                                    rs.raw_acuity_scale
                                )
                            }
                            None => format!(
                                "Acuity:   PPD N/A (CSS ref 46.92 • raw {:.2}×)",
                                rs.raw_acuity_scale
                            ),
                        };
                        ui.mono_text(&ppd_line, 11.5, th.text_secondary);

                        let harmony_tag = if rs.is_dyadic_exact() {
                            "Exact k/8 IEEE-754"
                        } else {
                            "Wayland 1/120"
                        };
                        let harmony_line = format!(
                            "Harmony:  Grid {:.2} ({harmony_tag} • 0-seam)",
                            rs.grid_harmony
                        );
                        ui.mono_text(
                            &harmony_line,
                            11.5,
                            if rs.is_dyadic_exact() {
                                pal.ink_success
                            } else {
                                pal.ink_warning
                            },
                        );

                        ui.divider();
                        ui.ui_text(
                            "Preview Scale (Ctrl+= / Ctrl+- zoom • Ctrl+0 Auto):",
                            11.5,
                            th.text_muted,
                        );

                        ui.element(
                            LayoutSpec::wrap_row().width_grow(1.0).gap(5.0),
                            PaintStyle::default(),
                            |ui| {
                                let auto_n = scale_preview_chip(
                                    ui,
                                    100,
                                    "Auto",
                                    is_auto_policy,
                                    rs.reason.is_auto(),
                                    ScalePolicy::Auto,
                                    "Automatic display scale (Ctrl+0): solves PPI, viewing distance & k/8 grid",
                                );
                                ui.state()
                                    .nodes
                                    .scale_auto_btn
                                    .store(auto_n.0, Ordering::Relaxed);

                                let presets: &[(u64, &str, u32, f32, &str)] = &[
                                    (
                                        120,
                                        "1.0×",
                                        120,
                                        1.0,
                                        "Preview 1.00× (120/120 native 1:1) • Ctrl+= / Ctrl+- steps lattice",
                                    ),
                                    (
                                        150,
                                        "1.25×",
                                        150,
                                        1.25,
                                        "Preview 1.25× (150/120 = 5/4 dyadic) • Ctrl+= / Ctrl+- / Ctrl+0",
                                    ),
                                    (
                                        160,
                                        "1.33×",
                                        160,
                                        SCALE_1_33,
                                        "Preview 1.333× (160/120 = 4/3 Wayland) • Ctrl+= / Ctrl+- / Ctrl+0",
                                    ),
                                    (
                                        180,
                                        "1.50×",
                                        180,
                                        1.5,
                                        "Preview 1.50× (180/120 = 3/2 dyadic) • Ctrl+= / Ctrl+- / Ctrl+0",
                                    ),
                                    (
                                        210,
                                        "1.75×",
                                        210,
                                        1.75,
                                        "Preview 1.75× (210/120 = 7/4 dyadic) • Ctrl+= / Ctrl+- / Ctrl+0",
                                    ),
                                    (
                                        240,
                                        "2.0×",
                                        240,
                                        2.0,
                                        "Preview 2.00× (240/120 = 2/1 Retina) • Ctrl+0 resets Auto",
                                    ),
                                ];
                                for &(kid, lbl, w120, sf_val, tip) in presets {
                                    let is_exact_fixed =
                                        !is_auto_policy && rs.wayland_120ths == w120;
                                    let is_auto_resolved =
                                        is_auto_policy && rs.wayland_120ths == w120;
                                    let chip_n = scale_preview_chip(
                                        ui,
                                        kid,
                                        lbl,
                                        is_exact_fixed,
                                        is_auto_resolved,
                                        ScalePolicy::Fixed(sf_val),
                                        tip,
                                    );
                                    if w120 == 150 {
                                        ui.state()
                                            .nodes
                                            .scale_125_btn
                                            .store(chip_n.0, Ordering::Relaxed);
                                    } else if w120 == 160 {
                                        ui.state()
                                            .nodes
                                            .scale_133_btn
                                            .store(chip_n.0, Ordering::Relaxed);
                                    } else if w120 == 180 {
                                        ui.state()
                                            .nodes
                                            .scale_150_btn
                                            .store(chip_n.0, Ordering::Relaxed);
                                    }
                                }
                            },
                        );

                        ui.button("Close Inspector", th.bg_control, |s, _| {
                            s.show_inspector_popover = false;
                        });
                    },
                );
            }
        },
    );
}

fn scale_preview_chip(
    ui: &mut Ui<'_, '_, StudioState>,
    key_id: u64,
    label: &str,
    selected: bool,
    auto_resolved_match: bool,
    policy: ScalePolicy,
    tooltip_text: &str,
) -> NodeId {
    let th = ui.theme();
    let base_bg = if selected { th.accent } else { th.bg_control };
    let mut chip_node = NodeId::NONE;
    ui.keyed(key_id, |ui| {
        chip_node = ui.element(
            LayoutSpec::row().pad_xy(7.5, 4.5).align(Align::Center),
            PaintStyle::solid(base_bg)
                .rounded(th.radius_sm)
                .border(1.0, th.border_subtle),
            |ui| {
                ui.set_focusable(true);
                ui.set_cursor(CursorIcon::Pointer);
                ui.set_semantics(SemanticsMeta::button(label));
                ui.on_click(move |s, _| {
                    s.pending_scale_policy.set(Some(policy));
                });

                let captured = ui.is_captured();
                let hovered = ui.is_hovered();
                let focused = ui.is_focused();
                let eff_bg = if selected {
                    if captured {
                        th.accent.shift_lightness(-0.05)
                    } else if hovered {
                        th.accent_hover
                    } else {
                        th.accent
                    }
                } else if captured {
                    th.bg_control_hover
                        .shift_lightness(if th.is_dark { -0.04 } else { -0.06 })
                } else if hovered {
                    th.bg_control_hover
                } else {
                    th.bg_control
                };
                let (bw, bc) = if focused {
                    (2.0, th.border_focus)
                } else if selected {
                    (
                        1.0,
                        th.accent.shift_lightness(if th.is_dark { 0.10 } else { -0.08 }),
                    )
                } else if auto_resolved_match {
                    (1.5, th.success)
                } else {
                    (1.0, th.border_subtle)
                };
                ui.set_paint(PaintStyle::solid(eff_bg).rounded(th.radius_sm).border(bw, bc));

                let fg = if selected {
                    th.text_on_accent
                } else {
                    th.text_primary
                };
                ui.ui_text(label, 12.0, fg);
            },
        );
        ui.tooltip(chip_node, tooltip_text);
    });
    chip_node
}

fn nav_item(
    ui: &mut Ui<'_, '_, StudioState>,
    key_id: u64,
    label: &str,
    selected: bool,
    on_select: fn(&mut StudioState),
) {
    let th = ui.theme();
    let base_bg = if selected {
        th.accent
    } else {
        Color::TRANSPARENT
    };
    ui.keyed(key_id, |ui| {
        ui.element(
            LayoutSpec::row()
                .width_grow(1.0)
                .height_fixed(36.0)
                .pad_xy(12.0, 7.0)
                .align(Align::Center),
            PaintStyle::solid(base_bg).rounded(th.radius_sm),
            |ui| {
                ui.set_focusable(true);
                ui.set_cursor(CursorIcon::Pointer);
                ui.set_semantics(SemanticsMeta::button(label));
                ui.on_click(move |s, _| on_select(s));

                let hovered = ui.is_hovered();
                let focused = ui.is_focused();
                let eff_bg = if selected {
                    if hovered { th.accent_hover } else { th.accent }
                } else if hovered {
                    th.bg_control_hover
                } else {
                    Color::TRANSPARENT
                };
                let (bw, bc) = if focused {
                    (2.0, th.border_focus)
                } else if selected {
                    (
                        1.0,
                        th.accent.shift_lightness(if th.is_dark { 0.10 } else { -0.08 }),
                    )
                } else {
                    (0.0, Color::TRANSPARENT)
                };
                ui.set_paint(PaintStyle::solid(eff_bg).rounded(th.radius_sm).border(bw, bc));

                let fg = if selected {
                    th.text_on_accent
                } else {
                    th.text_primary
                };
                ui.ui_text(label, 13.5, fg);
            },
        );
    });
}

fn render_overview_tab(
    ui: &mut Ui<'_, '_, StudioState>,
    rm: ResponsiveMetrics,
    pal: StudioPalette,
) {
    let th = ui.theme();
    ui.heading("Incremental Axis-Factored Layout & Analytical SDF Pipeline");

    // Shared closed-form analytical spring (`damping_ratio = 0.45` underdamped so you can see
    // real-time harmonic ringing at `spring_hz` in BOTH the Path2D waveform and the live meter!):
    let load = ui.state().target_load;
    let hz = ui.state().spring_hz;
    let phase = ui.state().wave_phase;
    let anim_load = ui.use_spring_damped(load, hz, 0.45);

    // Responsive 3-column grid geometry (exact fit from 960x640 compact up to 4K tiling):
    let grid_gap = 14.0_f32;
    let grid_h = if rm.compact_h { 276.0 } else { 306.0 };
    let row1_h = if rm.compact_h { 130.0 } else { 142.0 };
    let col_w = ((rm.content_w - 2.0 * grid_gap) / 3.0).max(160.0);
    let card_pad = if rm.compact_h { 14.0 } else { 16.0 };
    let slider_w = (col_w - 2.0 * card_pad).max(136.0).round();

    // Physical-atlas-safe responsive waveform width up to `GlyphAtlas::MAX_DIMENSION` (4096px):
    // `GlyphAtlas` grows dynamically (`1024 -> 2048 -> 4096`), spanning full card width at QHD & 2x Retina!
    let hero_inner_w = (2.0 * col_w + grid_gap - 2.0 * card_pad).max(220.0);
    let max_atlas_logical_w = ((GlyphAtlas::MAX_DIMENSION.saturating_sub(4) as f32)
        / ui.sys.scale_factor.max(0.5))
    .floor()
    .max(120.0);
    let wave_w = hero_inner_w.min(max_atlas_logical_w).round() as u16;
    let wave_h: u16 = if rm.compact_h { 54 } else { 62 };

    // 2D CSS-Style Grid with a 2-column spanned hero banner + metric cards!
    let mut grid_spec = LayoutSpec::col()
        .width_grow(1.0)
        .height_fixed(grid_h)
        .gap(grid_gap);
    grid_spec.axis = Axis::Grid;
    ui.grid(
        grid_spec,
        vec![TrackSize::Fr(1.0), TrackSize::Fr(1.0), TrackSize::Fr(1.0)],
        vec![TrackSize::Fixed(row1_h), TrackSize::Fr(1.0)],
        PaintStyle::default(),
        |ui| {
            // Card 1 (Spans 2 columns!): Live Zero-Alloc Analytical Path2D Waveform + Spring Pulse Trigger
            ui.element(
                LayoutSpec::col()
                    .grow(1.0)
                    .grid_span(2, 1)
                    .pad(card_pad)
                    .gap(if rm.compact_h { 8.0 } else { 10.0 })
                    .justify(MainAlign::SpaceBetween),
                PaintStyle::solid(th.bg_surface)
                    .rounded(th.radius_md)
                    .border(1.0, th.border_subtle)
                    .drop_shadow(10.0, [0.0, 4.0], pal.card_shadow),
                |ui| {
                    ui.set_clips_children(true);
                    ui.element(
                        LayoutSpec::row()
                            .width_grow(1.0)
                            .justify(MainAlign::SpaceBetween)
                            .align(Align::Center),
                        PaintStyle::default(),
                        |ui| {
                            let wave_title = if rm.compact_w {
                                "Analytical Path2D Waveform (0 B Heap)"
                            } else {
                                "Wang's-Formula Analytical Path2D Waveform (0 B Steady-State Heap)"
                            };
                            ui.ui_text(wave_title, 14.0, th.text_primary);
                            let pulse_btn = ui.primary_button("Pulse Wave", |s, _| {
                                s.wave_phase = s.wave_phase.wrapping_add(1);
                                s.target_load = if s.target_load > 0.6 { 0.25 } else { 0.90 };
                            });
                            ui.tooltip(
                                pulse_btn,
                                "Trigger underdamped analytical spring impulse",
                            );
                        },
                    );

                    // Sub-0.05% quantized revision tag: updates smoothly in place at monitor refresh rate
                    // via zero-alloc `RasterScratch` while springing, and settles to 0 uploads at rest!
                    let amp_q = (anim_load * 2048.0).round() as i64 as u64;
                    let hz_q = (hz * 256.0).round() as u64;
                    let size_q = ((wave_w as u64) << 16) | (wave_h as u64);
                    let rev_tag = phase.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ amp_q.wrapping_mul(0xBF58_476D_1CE4_E5B9)
                        ^ hz_q.rotate_left(21)
                        ^ size_q.rotate_left(7)
                        ^ 1;

                    ui.vector_canvas(wave_w, wave_h, rev_tag, th.accent, FillRule::NonZero, |p| {
                        let wf = wave_w as f32;
                        let hf = wave_h as f32;
                        let mid_y = hf * 0.5;
                        let max_amp = (mid_y - 3.5).max(8.0);
                        let shift = (phase as f32) * 0.85;
                        let amp = (anim_load.clamp(0.0, 1.35) / 1.35) * max_amp;
                        let cycles = 1.15 + hz * 0.42;
                        let omega = cycles * core::f32::consts::TAU / wf;
                        let n_segs = (wave_w as usize / 4).clamp(64, 192);
                        let mut pts = [[0.0_f32; 2]; 193];
                        for (step, pt) in pts[..=n_segs].iter_mut().enumerate() {
                            let x = 2.0 + (step as f32) * ((wf - 4.0) / n_segs as f32);
                            let a = x * omega + shift;
                            let y = (mid_y + a.sin() * (amp * 0.76) + (a * 0.5).cos() * (amp * 0.24))
                                .clamp(2.5, hf - 2.5);
                            *pt = [x, y];
                        }
                        p.stroke_polyline(&pts[..=n_segs], 2.4);
                    });
                },
            );

            // Card 2: Performance Summary Card
            metric_card(
                ui,
                card_pad,
                "6,785-Node Hover E2E",
                "52.8 µs",
                if rm.compact_w {
                    "Vulkan 1.3 ReBAR (0 B heap)"
                } else {
                    "Vulkan 1.3 ReBAR + Fence (0 B heap)"
                },
                pal.ink_success,
                pal.card_shadow,
            );

            // Row 2 Cards (3 x 1Fr): Interactive Closed-Form Spring Physics Controls
            ui.element(
                LayoutSpec::col()
                    .grow(1.0)
                    .pad(card_pad)
                    .gap(8.0)
                    .justify(MainAlign::SpaceBetween),
                PaintStyle::solid(th.bg_surface)
                    .rounded(th.radius_md)
                    .border(1.0, th.border_subtle)
                    .drop_shadow(8.0, [0.0, 3.0], pal.card_shadow),
                |ui| {
                    ui.set_clips_children(true);
                    let pct_txt = if rm.compact_w {
                        format!(
                            "Target: {:>3.0}% • Live: {:>3.0}%",
                            load * 100.0,
                            anim_load * 100.0
                        )
                    } else {
                        format!(
                            "Spring Target: {:>3.0}% (Live: {:>3.0}%)",
                            load * 100.0,
                            anim_load * 100.0
                        )
                    };
                    ui.ui_text(&pct_txt, 13.0, th.text_primary);
                    let load_slider = ui.slider("Target Load", load, 0.05..=1.0, slider_w, |s, v| {
                        s.target_load = v;
                    });
                    ui.tooltip(load_slider, "Adjust spring target amplitude (5%–100%)");
                    // 16px centered wrapper aligns slider Y identically with neighbor card's 16px caption!
                    ui.element(
                        LayoutSpec::row().fixed(slider_w, 16.0).align(Align::Center),
                        PaintStyle::default(),
                        |ui| {
                            ui.element(
                                LayoutSpec::row().fixed(slider_w, 8.0),
                                PaintStyle::solid(th.bg_control).rounded(4.0),
                                |ui| {
                                    ui.set_clips_children(true);
                                    let fill_w = (anim_load.clamp(0.0, 1.0) * slider_w).max(0.0);
                                    ui.element(
                                        LayoutSpec::row().fixed(fill_w, 8.0),
                                        PaintStyle::solid(th.accent).rounded(4.0),
                                        |_| {},
                                    );
                                },
                            );
                        },
                    );
                },
            );

            ui.element(
                LayoutSpec::col()
                    .grow(1.0)
                    .pad(card_pad)
                    .gap(8.0)
                    .justify(MainAlign::SpaceBetween),
                PaintStyle::solid(th.bg_surface)
                    .rounded(th.radius_md)
                    .border(1.0, th.border_subtle)
                    .drop_shadow(8.0, [0.0, 3.0], pal.card_shadow),
                |ui| {
                    ui.set_clips_children(true);
                    let hz = ui.state().spring_hz;
                    let lbl = format!("Spring Frequency: {hz:.1} Hz");
                    ui.ui_text(&lbl, 13.0, th.text_primary);
                    let hz_slider = ui.slider("Spring Hz", hz, 1.0..=14.0, slider_w, |s, v| {
                        s.spring_hz = v;
                    });
                    ui.tooltip(hz_slider, "Tune closed-form harmonic natural frequency (1–14 Hz)");
                    ui.ui_text(
                        if rm.compact_w {
                            "Closed-form ODE • 0 drift"
                        } else {
                            "Closed-form ODE: frame-rate invariant"
                        },
                        12.0,
                        th.text_muted,
                    );
                },
            );

            metric_card(
                ui,
                card_pad,
                "Width-Interval Wrap Cache",
                "96.7% Hit",
                if rm.compact_w {
                    "Exact [W_tight, W_next) bound"
                } else {
                    "Exact [W_tight, W_next_unwrap) theorem"
                },
                pal.ink_warning,
                pal.card_shadow,
            );
        },
    );

    // Responsive WrapRow Tag Cloud Demonstration
    ui.element(
        LayoutSpec::col()
            .width_grow(1.0)
            .pad(card_pad)
            .gap(if rm.compact_h { 8.0 } else { 10.0 }),
        PaintStyle::solid(th.bg_surface)
            .rounded(th.radius_md)
            .border(1.0, th.border_subtle)
            .drop_shadow(8.0, [0.0, 3.0], pal.card_shadow),
        |ui| {
            ui.ui_text(
                "Responsive WrapRow Chip Layout (Resize window from 960x640 to 4K to reflow):",
                13.0,
                th.text_secondary,
            );
            ui.element(
                LayoutSpec::wrap_row().width_grow(1.0).gap(8.0),
                PaintStyle::default(),
                |ui| {
                    for tag in [
                        "Automatic Fractional Scaling",
                        "Dual-Source ClearType",
                        "Analytical SDF Shadows",
                        "OKLab Perceptual Gradients",
                        "Fenwick VirtualList",
                        "Path2D Vector Canvas",
                        "Text Selection & Clipboard",
                        "Viewport-Clamped Popovers",
                        "AccessKit 0.24",
                    ] {
                        ui.badge(tag, th.accent);
                    }
                },
            );
        },
    );
}

fn metric_card(
    ui: &mut Ui<'_, '_, StudioState>,
    pad: f32,
    title: &str,
    big_value: &str,
    subtitle: &str,
    accent_col: Color,
    shadow_col: Color,
) {
    let th = ui.theme();
    ui.element(
        LayoutSpec::col()
            .grow(1.0)
            .pad(pad)
            .justify(MainAlign::SpaceBetween),
        PaintStyle::solid(th.bg_surface)
            .rounded(th.radius_md)
            .border(1.0, th.border_subtle)
            .drop_shadow(8.0, [0.0, 3.0], shadow_col),
        |ui| {
            ui.set_clips_children(true);
            ui.ui_text(title, 13.0, th.text_secondary);
            ui.ui_text(big_value, 26.0, accent_col);
            ui.ui_text(subtitle, 12.0, th.text_muted);
        },
    );
}

fn render_virtual_logs_tab(
    ui: &mut Ui<'_, '_, StudioState>,
    rm: ResponsiveMetrics,
    pal: StudioPalette,
) {
    let th = ui.theme();
    let total_rows = ui.state().vlist.len();
    let heading_txt = format!(
        "{}-Row Variable-Height Fenwick VirtualList",
        fmt_usize(total_rows)
    );
    ui.heading(&heading_txt);

    // Top search, jump, & live O(log N) stream bar
    let tail_target = total_rows.saturating_sub(15);
    ui.element(
        LayoutSpec::row()
            .width_grow(1.0)
            .gap(8.0)
            .align(Align::Center),
        PaintStyle::default(),
        |ui| {
            let placeholder = if rm.compact_w {
                "Filter tag (vulkan, iale)..."
            } else {
                "Filter subsystem tag (Ctrl+A / Shift+Arrows / Ctrl+C / Ctrl+V)..."
            };
            let filter_node = ui.text_input(
                placeholder,
                |s| &s.search_query,
                |s| &mut s.search_query,
            );
            ui.state()
                .nodes
                .filter_input
                .store(filter_node.0, Ordering::Relaxed);
            if !ui.state().search_query.is_empty() {
                let clr = ui.button("Clear", th.bg_control, |s, _| s.search_query.clear());
                ui.tooltip(clr, "Clear subsystem filter query");
            }
            let top_lbl = if rm.compact_w { "Top" } else { "Jump Top" };
            let top_btn = ui.button(top_lbl, th.bg_control, |s, _| s.log_scroll_y = 0.0);
            ui.tooltip(top_btn, "Jump to row #000000 (scroll_y = 0 px)");

            let mid_lbl = if rm.compact_w { "#50k" } else { "Jump #50,000" };
            let mid_btn = ui.button(mid_lbl, th.bg_control, |s, _| {
                s.log_scroll_y = s.vlist.offset_of(50_000.min(s.vlist.len().saturating_sub(1)));
            });
            ui.tooltip(mid_btn, "Jump to midpoint via Fenwick offset_of(50_000)");

            let tail_lbl = if rm.compact_w {
                "Tail".to_owned()
            } else {
                format!("Jump #{}", fmt_usize(tail_target))
            };
            let tail_btn = ui.button(&tail_lbl, th.bg_control, move |s, _| {
                let idx = s.vlist.len().saturating_sub(15);
                s.log_scroll_y = s.vlist.offset_of(idx);
            });
            ui.tooltip(tail_btn, "Jump to live stream tail via O(log N) prefix sum");

            let stream_btn = ui.primary_button("+ Stream +1,000 Logs", |s, _| {
                s.stream_logs(1_000);
            });
            ui.state()
                .nodes
                .stream_btn
                .store(stream_btn.0, Ordering::Relaxed);
            ui.tooltip(
                stream_btn,
                "Append +1,000 rows in O(log N) via VirtualListIndex::push",
            );
        },
    );

    let total_px = ui.state().vlist.total_height();
    let cur_scroll = ui.state().log_scroll_y;
    let expanded_count = ui.state().expanded_rows.len();
    let streamed = ui.state().streamed_logs;
    let pct = if total_px > 1.0 {
        ((cur_scroll / total_px) * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    let rows_tag = if streamed > 0 {
        format!("{} (+{} live)", fmt_usize(total_rows), fmt_usize(streamed))
    } else {
        fmt_usize(total_rows)
    };
    let total_px_str = fmt_usize(total_px.round() as usize);
    let scroll_px_str = fmt_usize(cur_scroll.round() as usize);
    let info = if rm.compact_w {
        format!(
            "Rows: {rows_tag}  |  Height: {total_px_str} px  |  Scroll: {scroll_px_str} px ({pct:.1}%)  |  Expanded: {expanded_count}"
        )
    } else {
        format!(
            "Rows: {rows_tag}  |  Total Height: {total_px_str} px  |  Scroll Y: {scroll_px_str} px ({pct:.1}%)  |  Expanded: {expanded_count}  •  Click row to expand"
        )
    };
    ui.mono_text(&info, 12.0, th.text_secondary);

    let query = ui.state().search_query.trim().to_ascii_lowercase();
    let vlist_ref = &ui.state().vlist;

    ui.virtual_scroll_view(
        "Virtual Logs Scrollbar",
        vlist_ref,
        |s| s.log_scroll_y,
        |s, next_scroll_y| {
            s.log_scroll_y = next_scroll_y;
        },
        LayoutSpec::row().grow(1.0),
        PaintStyle::solid(th.bg_surface)
            .rounded(th.radius_md)
            .border(1.0, th.border_subtle)
            .drop_shadow(8.0, [0.0, 3.0], pal.card_shadow),
        |ui, idx, row_h| {
            let is_exp = ui.state().expanded_rows.contains(&idx);
            let row_y_offset = ui.state().vlist.offset_of(idx);
            let (level, level_col, subsys) = match idx % 4 {
                0 => ("PERF", th.success, "vulkan::rebar"),
                1 => ("INFO", th.accent, "iale::solver"),
                2 => ("SYNC", th.warning, "text::cleartype"),
                _ => ("DATA", th.accent, "scale::auto"),
            };
            let matches_q = !query.is_empty()
                && (subsys.contains(query.as_str())
                    || level.to_ascii_lowercase().contains(query.as_str()));
            let hovered = ui.is_hovered();
            let row_bg = if is_exp {
                th.accent_muted
                    .mix_oklab(th.bg_surface, if th.is_dark { 0.58 } else { 0.78 })
            } else if hovered {
                th.bg_control_hover
            } else if matches_q {
                th.accent_muted
                    .mix_oklab(th.bg_surface, if th.is_dark { 0.72 } else { 0.86 })
            } else if idx % 2 == 0 {
                th.bg_surface
            } else {
                th.bg_canvas.mix_oklab(th.bg_surface, 0.72)
            };

            ui.element(
                LayoutSpec::col()
                    .width_grow(1.0)
                    .height_fixed(row_h)
                    .pad_xy(12.0, 6.0)
                    .gap(6.0)
                    .justify(MainAlign::Start),
                PaintStyle::solid(row_bg).border(0.5, th.border_subtle),
                |ui| {
                    ui.set_clips_children(true);
                    ui.set_cursor(CursorIcon::Pointer);
                    ui.on_click(move |s, _| {
                        if s.expanded_rows.remove(&idx) {
                            s.vlist.set_height(idx, LOG_ROW_COLLAPSED_H);
                        } else {
                            s.expanded_rows.insert(idx);
                            s.vlist.set_height(idx, LOG_ROW_EXPANDED_H);
                        }
                    });

                    // Top-pinned 22px header line (never shifts vertically when expanding!)
                    ui.element(
                        LayoutSpec::row()
                            .width_grow(1.0)
                            .height_fixed(22.0)
                            .gap(10.0)
                            .align(Align::Center),
                        PaintStyle::default(),
                        |ui| {
                            // Crisp Path2D disclosure chevron (`▸` collapsed, `▾` expanded)
                            let chev_tag = if is_exp { 2u64 } else { 1u64 };
                            let chev_col = if is_exp { pal.ink_accent } else { th.text_muted };
                            ui.vector_canvas(12, 12, chev_tag, chev_col, FillRule::NonZero, |p| {
                                if is_exp {
                                    p.move_to(2.0, 4.0)
                                        .line_to(10.0, 4.0)
                                        .line_to(6.0, 9.5)
                                        .close();
                                } else {
                                    p.move_to(4.0, 2.0)
                                        .line_to(9.5, 6.0)
                                        .line_to(4.0, 10.0)
                                        .close();
                                }
                            });

                            // Fixed-width 54px pill slot aligns `#000000..#999999` down an exact vertical column!
                            ui.element(
                                LayoutSpec::row()
                                    .width_fixed(54.0)
                                    .justify(MainAlign::Center)
                                    .align(Align::Center),
                                PaintStyle::default(),
                                |ui| {
                                    ui.badge(level, level_col);
                                },
                            );

                            let stream_tag = if idx >= 100_000 {
                                "live_push=O(logN)"
                            } else {
                                "rebar_upload=64B"
                            };
                            let summary = format!(
                                "#{idx:06}  [{subsys:<15}]  dirty_prims=1  {stream_tag:<16}  allocs=0"
                            );
                            ui.mono_text(
                                &summary,
                                12.5,
                                if matches_q || is_exp {
                                    th.text_primary
                                } else {
                                    th.text_secondary
                                },
                            );
                        },
                    );

                    // Structured indented detail drawer when expanded (`12 + 22 + 6 + 48 == 88px` exact):
                    if is_exp {
                        let hops = total_rows.max(1).ilog2() + 1;
                        ui.element(
                            LayoutSpec::col()
                                .width_grow(1.0)
                                .height_fixed(48.0)
                                .pad_xy(12.0, 6.0)
                                .justify(MainAlign::SpaceBetween),
                            PaintStyle::solid(th.bg_surface)
                                .rounded(th.radius_sm)
                                .border(1.0, th.border_subtle),
                            |ui| {
                                let detail_1 = format!(
                                    "Fenwick Prefix Y-Offset: {} px  •  Stream Length: {} rows  •  Binary Lifting: {hops} bit-hops",
                                    fmt_usize(row_y_offset.round() as usize),
                                    fmt_usize(total_rows),
                                );
                                ui.mono_text(&detail_1, 11.5, pal.ink_accent);
                                ui.ui_text(
                                    "O(log N) Fenwick set_height & push: sibling expanded rows stay open with zero scroll jump.",
                                    11.5,
                                    th.text_secondary,
                                );
                            },
                        );
                    }
                },
            );
        },
    );
}

fn render_pool_tasks_tab(
    ui: &mut Ui<'_, '_, StudioState>,
    rm: ResponsiveMetrics,
    pal: StudioPalette,
) {
    let th = ui.theme();
    ui.heading("Generational Task Pool");

    ui.element(
        LayoutSpec::row()
            .width_grow(1.0)
            .gap(10.0)
            .align(Align::Center),
        PaintStyle::default(),
        |ui| {
            // Pressing Enter while focused inside the task input bubbles here and adds the task!
            ui.on_key(|s, k| {
                if *k == KeyInput::Enter {
                    s.add_new_task();
                    true
                } else {
                    false
                }
            });
            let task_input = ui.text_input(
                "New engineering task (Ctrl+A / Shift+Arrows / Ctrl+C / Ctrl+V / Enter)...",
                |s| &s.new_task_title,
                |s| &mut s.new_task_title,
            );
            ui.state()
                .nodes
                .task_input
                .store(task_input.0, Ordering::Relaxed);
            let add_btn = ui.primary_button("+ Add Task", |s, _| s.add_new_task());
            ui.tooltip(
                add_btn,
                "Add task to board (or press Enter)",
            );
            let swap_btn = ui.button("Swap Top 2", th.bg_control, |s, _| {
                s.swap_top_two_priorities();
            });
            ui.state()
                .nodes
                .swap_btn
                .store(swap_btn.0, Ordering::Relaxed);
            ui.tooltip(
                swap_btn,
                "Swap the top two tasks in place",
            );
        },
    );

    let swaps = ui.state().swap_count;
    let recycled = ui.state().recycled_count;
    let active_len = ui.state().task_keys.len();
    let subtitle = format!(
        "Live Tasks: {active_len}  |  Swaps: {swaps}  |  Archived: {recycled}"
    );
    ui.mono_text(&subtitle, 12.0, th.text_secondary);

    let keys_snapshot = ui.state().task_keys.clone();
    let bar_w = if rm.compact_w { 88.0 } else { 124.0 };
    let row_card_bg = th
        .bg_canvas
        .mix_oklab(th.bg_surface, if th.is_dark { 0.72 } else { 0.76 });

    ui.scroll_area(
        "Task Board Scrollbar",
        |s| s.task_scroll_y,
        |s, next_y| {
            s.task_scroll_y = next_y;
        },
        LayoutSpec::col()
            .width_grow(1.0)
            .height_grow(1.0)
            .pad(12.0)
            .gap(8.0),
        PaintStyle::solid(th.bg_surface)
            .rounded(th.radius_md)
            .border(1.0, th.border_subtle)
            .drop_shadow(8.0, [0.0, 3.0], pal.card_shadow),
        |ui| {
            for key in keys_snapshot {
                let Some(task) = ui.state().tasks.get(key) else {
                    continue;
                };
                let uid = task.uid;
                let title = task.title.clone();
                let done = task.done;
                let prio = task.priority_pct;

                ui.keyed(uid, |ui| {
                    ui.element(
                        LayoutSpec::row()
                            .width_grow(1.0)
                            .height_fixed(46.0)
                            .pad_xy(14.0, 7.0)
                            .align(Align::Center)
                            .justify(MainAlign::SpaceBetween),
                        PaintStyle::solid(row_card_bg)
                            .rounded(th.radius_sm)
                            .border(1.0, th.border_subtle),
                        |ui| {
                            ui.checkbox(&title, done, move |s, _| {
                                if let Some(t) = s.tasks.get_mut(key) {
                                    t.done = !t.done;
                                }
                            });

                            ui.element(
                                LayoutSpec::row().gap(10.0).align(Align::Center),
                                PaintStyle::default(),
                                |ui| {
                                    // Fixed-width columns keep `%`, progress bars, DONE/ACTIVE badges,
                                    // and Archive buttons aligned to the exact vertical pixel!
                                    ui.element(
                                        LayoutSpec::row()
                                            .width_fixed(38.0)
                                            .justify(MainAlign::End)
                                            .align(Align::Center),
                                        PaintStyle::default(),
                                        |ui| {
                                            let pct_lbl = format!("{:>3.0}%", prio * 100.0);
                                            ui.mono_text(&pct_lbl, 11.5, th.text_muted);
                                        },
                                    );
                                    ui.progress_bar(
                                        prio,
                                        bar_w,
                                        if done { th.success } else { th.accent },
                                    );
                                    ui.element(
                                        LayoutSpec::row()
                                            .width_fixed(66.0)
                                            .justify(MainAlign::Center)
                                            .align(Align::Center),
                                        PaintStyle::default(),
                                        |ui| {
                                            let badge_txt = if done { "DONE" } else { "ACTIVE" };
                                            ui.badge(
                                                badge_txt,
                                                if done { th.success } else { th.warning },
                                            );
                                        },
                                    );
                                    ui.button("Archive", th.bg_control, move |s, _| {
                                        s.remove_task(key);
                                    });
                                },
                            );
                        },
                    );
                });
            }
        },
    );
}

// =============================================================================
// Zero-Dependency RFC-2083 / RFC-1950 / RFC-1951 Lossless RGBA8 PNG Encoder
// =============================================================================

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if (c & 1) != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

fn crc32_png(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc = (crc >> 8) ^ CRC32_TABLE[((crc as u8) ^ b) as usize];
    }
    !crc
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32_png(chunk_type, data).to_be_bytes());
}

fn save_rgba8_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "PNG size overflow")
        })?;
    if rgba.len() != expected || width == 0 || height == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "RGBA buffer length does not match width * height * 4",
        ));
    }

    let row_bytes = (width as usize) * 4;
    let raw_len = (height as usize) * (1 + row_bytes);
    let mut raw_scanlines = Vec::with_capacity(raw_len);
    for y in 0..(height as usize) {
        raw_scanlines.push(0u8); // Filter Type 0 (None)
        let row_start = y * row_bytes;
        raw_scanlines.extend_from_slice(&rgba[row_start..row_start + row_bytes]);
    }

    // RFC 1950 zlib stream wrapping RFC 1951 stored DEFLATE blocks (`BTYPE = 00`):
    let mut zlib = Vec::with_capacity(2 + raw_len + (raw_len / 65_535 + 1) * 5 + 4);
    zlib.extend_from_slice(&[0x78, 0x01]); // CMF=0x78 (32K deflate), FLG=0x01 (0x7801 % 31 == 0)

    let blocks: Vec<&[u8]> = raw_scanlines.chunks(65_535).collect();
    let last_idx = blocks.len() - 1;
    for (i, blk) in blocks.into_iter().enumerate() {
        zlib.push(u8::from(i == last_idx));
        let len = blk.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(blk);
    }

    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for chunk in raw_scanlines.chunks(5552) {
        for &b in chunk {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= 65_521;
        s2 %= 65_521;
    }
    zlib.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

    let mut png = Vec::with_capacity(8 + 25 + 12 + zlib.len() + 12);
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8; // 8-bit channel depth
    ihdr[9] = 6; // Color type 6: Truecolor with Alpha (RGBA8)
    write_png_chunk(&mut png, b"IHDR", &ihdr);
    write_png_chunk(&mut png, b"IDAT", &zlib);
    write_png_chunk(&mut png, b"IEND", &[]);

    if let Some(parent) = path.as_ref().parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &png)
}

// =============================================================================
// Headless Vulkan 1.3 Offscreen Renderer & Multi-Theme / Multi-Viewport Verifier
// =============================================================================

struct StudioHeadless {
    sys: GuiSystem,
    vk: Option<VulkanRenderer>,
}

impl StudioHeadless {
    fn new() -> Self {
        let sys = GuiSystem::with_system_fonts();
        let vk = VulkanRenderer::new_headless(16_384, 512).ok();
        Self { sys, vk }
    }

    /// Runs a settled multi-frame reconcile + layout pass (asserting `nodes_visited == 0` and exact
    /// box fixed-point stability on steady frames) and renders on hardware Vulkan 1.3 offscreen!
    fn render_settled(
        &mut self,
        logical_w: u32,
        logical_h: u32,
        scale_factor: f32,
        state: &StudioState,
    ) -> (u32, u32, Vec<u8>) {
        self.sys.display_metrics = None;
        let _ = self.sys.set_scale_policy(ScalePolicy::Fixed(scale_factor));
        self.render_settled_core(logical_w, logical_h, state)
    }

    /// Runs a settled multi-frame pass resolved from physical [`DisplayMetrics`] under `policy`
    /// (preserving `PPI`, `Viewing Dist`, `PPD`, `Grid Harmony`, and `ScaleReason` in the HUD!).
    fn render_settled_with_metrics(
        &mut self,
        logical_w: u32,
        logical_h: u32,
        policy: ScalePolicy,
        base_metrics: DisplayMetrics,
        state: &StudioState,
    ) -> (u32, u32, Vec<u8>) {
        self.sys.scale_policy = policy;
        let prelim = policy.resolve(&base_metrics);
        let win_pw = ((logical_w as f32) * prelim.scale_factor - 1e-4)
            .ceil()
            .max(1.0) as u32;
        let win_ph = ((logical_h as f32) * prelim.scale_factor - 1e-4)
            .ceil()
            .max(1.0) as u32;
        let _ = self
            .sys
            .resolve_display_scale(base_metrics.with_window_px(win_pw, win_ph));
        self.render_settled_core(logical_w, logical_h, state)
    }

    fn render_settled_core(
        &mut self,
        logical_w: u32,
        logical_h: u32,
        state: &StudioState,
    ) -> (u32, u32, Vec<u8>) {
        let lw = logical_w as f32;
        let lh = logical_h as f32;

        // Pass 1: reconcile tree, apply any pending_scale_policy, register spring targets, & record logical root box
        self.sys.build_frame(state, lw, lh, build_studio);
        // Settle all active `Spring` animations (`toggle_switch`, `progress_bar`, `use_spring_damped`):
        while self.sys.step_animations(0.25) {}
        // Pass 2: apply settled springs & viewport metrics (`VirtualList`, `scroll_area`, `AnchorPlacement`, `tooltip`)
        self.sys.build_frame(state, lw, lh, build_studio);
        // Pass 3: refresh telemetry HUD with settled GPU primitive & glyph counts
        self.sys.build_frame(state, lw, lh, build_studio);
        // Pass 4: stabilize final glyph count if Pass 3 rasterized a new digit glyph in the status bar
        self.sys.build_frame(state, lw, lh, build_studio);
        let settled_boxes = self.sys.layout.boxes.clone();
        // Pass 5: verify strict steady-state fixed point (`nodes_visited == 0` & bit-identical boxes)!
        self.sys.build_frame(state, lw, lh, build_studio);
        let sf = self.sys.scale_factor;
        assert_eq!(
            self.sys.layout.last_metrics.nodes_visited,
            0,
            "Steady-state frame visited {} layout nodes (expected 0) at {logical_w}x{logical_h}@{sf}x (dark={}, tab={:?})",
            self.sys.layout.last_metrics.nodes_visited,
            state.is_dark,
            state.active_tab
        );
        assert_eq!(
            self.sys.layout.boxes, settled_boxes,
            "Zero layout oscillation violated at {logical_w}x{logical_h}@{sf}x (dark={}, tab={:?})",
            state.is_dark, state.active_tab
        );

        let phys_w = ((lw * sf) - 1e-4).ceil().max(1.0) as u32;
        let phys_h = ((lh * sf) - 1e-4).ceil().max(1.0) as u32;

        if let Some(vk) = self.vk.as_mut()
            && let Ok(rgba) = self.sys.render_vulkan_offscreen(vk, phys_w, phys_h)
        {
            return (phys_w, phys_h, rgba);
        }

        let mut cpu = CpuRenderer::new(phys_w, phys_h);
        self.sys
            .render_to_backend(&mut cpu, phys_w, phys_h, 1)
            .unwrap();
        (phys_w, phys_h, cpu.framebuffer)
    }

    #[inline]
    fn node_center(&self, atom: &AtomicU32) -> [f32; 2] {
        let idx = atom.load(Ordering::Relaxed) as usize;
        let b = self.sys.layout.boxes[idx];
        [b.x + b.w * 0.5, b.y + b.h * 0.5]
    }
}

fn assert_theme_contrast(rgba: &[u8], is_dark: bool, context: &str) {
    let fg_pixels = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| {
            if is_dark {
                px[0] > 55 || px[1] > 55 || px[2] > 75
            } else {
                px[0] < 210 || px[1] < 210 || px[2] < 210
            }
        })
        .count();
    assert!(
        fg_pixels > 25_000,
        "Expected rich foreground contrast in {context} (dark={is_dark}), got only {fg_pixels} pixels"
    );
}

/// Verifies across all visible interior collapsed `VirtualList` log rows in `rgba` that:
/// 1. Adjacent rows share an exact integer physical boundary (`next_y0 == prev_y1`, `0` physical-pixel seams),
/// 2. Every `0.5px` row divider visibly contrasts against the interior row fill, AND
/// 3. Rows sharing the same background surface color exhibit **`<= 1 LSB` variance** on their top divider
///    scanline across the entire viewport (proving **zero periodic row-divider Moiré striping**!).
fn assert_zero_row_divider_moire(
    sys: &GuiSystem,
    phys_w: u32,
    phys_h: u32,
    rgba: &[u8],
    context: &str,
) {
    let sf = sys.scale_factor;
    let mut rows: Vec<(f32, u32, u32, u32, u32)> = Vec::new();
    for (idx, topo) in sys.layout.topo.iter().enumerate() {
        if (topo.flags & FLAG_ALIVE) == 0 {
            continue;
        }
        let spec = sys.layout.specs[idx];
        let b = sys.layout.boxes[idx];
        let clip = sys.layout.clip_rects[idx];
        if (spec.val_h - LOG_ROW_COLLAPSED_H).abs() < 1e-3 && b.w > 500.0 && clip[3] > 100.0 {
            let y0 = (b.y * sf).round() as i64;
            let y1 = ((b.y + b.h) * sf).round() as i64;
            let clip_y0 = (clip[1] * sf).round() as i64;
            let clip_y1 = ((clip[1] + clip[3]) * sf).round() as i64;
            let sample_x = ((b.x + b.w - 48.0) * sf).round() as i64;
            if y0 >= clip_y0 + 10
                && y1 <= clip_y1 - 10
                && y0 >= 0
                && y1 <= phys_h as i64
                && sample_x >= 0
                && (sample_x as u32) < phys_w
            {
                // Quantize row physical height + top/bottom boundary colors via top divider pixel:
                let px_top_off = ((y0 as usize) * (phys_w as usize) + (sample_x as usize)) * 4;
                let px_mid_off =
                    (((y0 + 4).min(y1 - 2) as usize) * (phys_w as usize) + (sample_x as usize)) * 4;
                let top_rgb = [rgba[px_top_off], rgba[px_top_off + 1], rgba[px_top_off + 2]];
                let mid_rgb = [rgba[px_mid_off], rgba[px_mid_off + 1], rgba[px_mid_off + 2]];
                let max_contrast = top_rgb[0]
                    .abs_diff(mid_rgb[0])
                    .max(top_rgb[1].abs_diff(mid_rgb[1]))
                    .max(top_rgb[2].abs_diff(mid_rgb[2]));
                assert!(
                    max_contrast >= 2,
                    "Expected visible 0.5px hairline divider vs fill in {context} at y={y0}: top={top_rgb:?}, mid={mid_rgb:?}"
                );
                let packed_mid =
                    ((mid_rgb[0] as u32) << 16) | ((mid_rgb[1] as u32) << 8) | (mid_rgb[2] as u32);
                let packed_top =
                    ((top_rgb[0] as u32) << 16) | ((top_rgb[1] as u32) << 8) | (top_rgb[2] as u32);
                rows.push((b.y, y0 as u32, y1 as u32, packed_mid, packed_top));
            }
        }
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(
        rows.len() >= 8,
        "Expected >= 8 visible interior VirtualList rows in {context}, got {}",
        rows.len()
    );

    // 1. Verify exact 0.0 physical-pixel seam between contiguous collapsed rows:
    for pair in rows.windows(2) {
        let (prev_ly, _prev_y0, prev_y1, _, _) = pair[0];
        let (cur_ly, cur_y0, _cur_y1, _, _) = pair[1];
        if ((cur_ly - (prev_ly + LOG_ROW_COLLAPSED_H)).abs()) < 1.5 {
            assert_eq!(
                cur_y0, prev_y1,
                "Physical row seam in {context}: prev_y1={prev_y1} != cur_y0={cur_y0}"
            );
        }
    }

    // 2. Group rows by identical interior surface RGB (`packed_mid` within 1 LSB) and assert <= 1 LSB
    //    divider luminance variance across every matching row in the viewport (zero Moiré banding!):
    struct DividerGroup {
        bg: [u8; 3],
        min_top: [u8; 3],
        max_top: [u8; 3],
        count: usize,
    }
    let mut groups: Vec<DividerGroup> = Vec::new();
    for &(_, _, _, p_mid, p_top) in &rows {
        let mid = [(p_mid >> 16) as u8, (p_mid >> 8) as u8, p_mid as u8];
        let top = [(p_top >> 16) as u8, (p_top >> 8) as u8, p_top as u8];
        if let Some(g) = groups.iter_mut().find(|g| {
            g.bg[0].abs_diff(mid[0]) <= 1
                && g.bg[1].abs_diff(mid[1]) <= 1
                && g.bg[2].abs_diff(mid[2]) <= 1
        }) {
            for c in 0..3 {
                g.min_top[c] = g.min_top[c].min(top[c]);
                g.max_top[c] = g.max_top[c].max(top[c]);
            }
            g.count += 1;
        } else {
            groups.push(DividerGroup {
                bg: mid,
                min_top: top,
                max_top: top,
                count: 1,
            });
        }
    }
    for g in groups {
        if g.count >= 2 {
            let spread = (g.max_top[0] - g.min_top[0])
                .max(g.max_top[1] - g.min_top[1])
                .max(g.max_top[2] - g.min_top[2]);
            assert!(
                spread <= 1,
                "Row-divider Moiré banding detected in {context} (bg={:?}, count={}): min_top={:?}, max_top={:?} (spread={spread} > 1 LSB)",
                g.bg, g.count, g.min_top, g.max_top
            );
        }
    }
}

fn run_smoke_test() {
    let mut h = StudioHeadless::new();
    let backend_desc = match h.vk.as_ref() {
        Some(vk) => format!(
            "Vulkan 1.3 GPU '{}' (ReBAR={}, DualSrc={})",
            vk.device_name, vk.is_rebar_device_local, vk.dual_source_blending
        ),
        None => String::from("CpuRenderer fallback"),
    };

    let mut state = StudioState::new();

    // 1. Verify fractional scale factors `[1.0, 1.125, 1.25, 160/120, 1.5, 1.75, 2.0]` across BOTH
    //    Dark and Light themes with zero layout oscillation (`nodes_visited == 0`) AND zero row-divider Moiré!
    let fractional_scales: [f32; 7] = [1.0, 1.125, 1.25, SCALE_1_33, 1.5, 1.75, 2.0];
    state.log_scroll_y = state.vlist.offset_of(6); // Rows 6..26 are contiguous collapsed 34px hairline rows
    for is_dark in [true, false] {
        state.is_dark = is_dark;
        for &sf in &fractional_scales {
            // (a) Overview + Open Floating Engine & Scale Inspector (`ResolvedScale` + 7 preview chips):
            state.active_tab = StudioTab::Overview;
            state.show_inspector_popover = true;
            let (pw, ph, rgba_ov) = h.render_settled(1280, 800, sf, &state);
            assert_eq!(rgba_ov.len(), (pw * ph * 4) as usize);
            assert_theme_contrast(&rgba_ov, is_dark, &format!("Overview@1280x800@{sf:.3}x"));
            for anc in &h.sys.layout.anchors {
                let b = h.sys.layout.boxes[anc.overlay_node.idx()];
                assert!(
                    b.x >= 0.0 && b.y >= 0.0 && b.x + b.w <= 1281.0 && b.y + b.h <= 801.0,
                    "Inspector popover {b:?} overflowed 1280x800 at {sf}x"
                );
            }

            // (b) 100k VirtualLogs (`0.5px` row dividers): verify 0 seams & <= 1 LSB Moiré-free uniformity!
            state.active_tab = StudioTab::VirtualLogs;
            state.show_inspector_popover = false;
            let (pw_logs, ph_logs, rgba_logs) = h.render_settled(1280, 800, sf, &state);
            assert_theme_contrast(
                &rgba_logs,
                is_dark,
                &format!("VirtualLogs@1280x800@{sf:.3}x"),
            );
            assert_zero_row_divider_moire(
                &h.sys,
                pw_logs,
                ph_logs,
                &rgba_logs,
                &format!("VirtualLogs(dark={is_dark})@{sf:.4}x"),
            );
        }

        // (c) SlotPool Kanban tab at dyadic `1.25x` and non-dyadic `1.333x` (`160/120`):
        state.active_tab = StudioTab::PoolTasks;
        state.show_inspector_popover = false;
        state.swap_top_two_priorities();
        for &sf in &[1.25_f32, SCALE_1_33] {
            let (pw_k, ph_k, rgba_k) = h.render_settled(1280, 800, sf, &state);
            assert_eq!(rgba_k.len(), (pw_k * ph_k * 4) as usize);
            assert_theme_contrast(&rgba_k, is_dark, &format!("PoolTasks@1280x800@{sf:.3}x"));
        }
    }
    state.log_scroll_y = 0.0;

    // 1b. Verify interactive Scale-Preview buttons + Pass-4 zoom tooltip + `ScalePolicy::Auto` physics in Inspector:
    state.is_dark = true;
    state.active_tab = StudioTab::Overview;
    state.show_inspector_popover = true;
    let desk_4k = DisplayMetrics::from_physical_mm(3840, 2160, 597, 336); // 27" 4K (163.2 PPI -> 1.50x Auto)
    let _ = h.render_settled_with_metrics(1280, 800, ScalePolicy::Auto, desk_4k, &state);
    assert_eq!(h.sys.resolved_scale.unwrap().wayland_120ths, 180);
    assert!(h.sys.resolved_scale.unwrap().reason.is_auto());

    // Hover `1.25×` preview chip inside Inspector -> verifies nested Pass-4 zoom tooltip!
    let chip_125_pt = h.node_center(&state.nodes.scale_125_btn);
    assert!(h.sys.pointer_move(chip_125_pt[0], chip_125_pt[1], &mut state));
    let _ = h.render_settled_with_metrics(1280, 800, ScalePolicy::Auto, desk_4k, &state);
    assert!(
        h.sys.layout.anchors.len() >= 2,
        "Expected Pass-4 zoom tooltip anchored above 1.25× preview chip inside Engine Inspector"
    );

    // Click `1.33×` preview chip -> switches live `ScalePolicy` to `Fixed(160/120)`, then click `Auto` -> restores `180/120`!
    let chip_133_pt = h.node_center(&state.nodes.scale_133_btn);
    assert!(h.sys.pointer_down(chip_133_pt[0], chip_133_pt[1], &mut state));
    let _ = h.sys.pointer_up();
    h.sys.build_frame(&state, 1280.0, 800.0, build_studio);
    assert_eq!(h.sys.resolved_scale.unwrap().wayland_120ths, 160);
    let chip_auto_pt = h.node_center(&state.nodes.scale_auto_btn);
    assert!(h.sys.pointer_down(chip_auto_pt[0], chip_auto_pt[1], &mut state));
    let _ = h.sys.pointer_up();
    h.sys.build_frame(&state, 1280.0, 800.0, build_studio);
    assert_eq!(h.sys.resolved_scale.unwrap().wayland_120ths, 180);
    let _ = h.sys.pointer_leave();
    h.sys.focused_slot = None;

    // 2. Verify responsive compact `960x640`, `2.0x` Retina HiDPI, AND `2560x1440` QHD across all 3 tabs:
    state.is_dark = true;
    for &(lw, lh, sf) in &[
        (960u32, 640u32, 1.0_f32),
        (1280, 800, 2.0),
        (2560, 1440, 1.0),
    ] {
        for tab in StudioTab::ALL {
            state.active_tab = tab;
            state.show_inspector_popover = tab == StudioTab::Overview;
            let (pw, ph, rgba) = h.render_settled(lw, lh, sf, &state);
            assert_eq!(rgba.len(), (pw * ph * 4) as usize);
            assert_theme_contrast(&rgba, true, &format!("{tab:?}@{lw}x{lh}x{sf}"));
            for anc in &h.sys.layout.anchors {
                let b = h.sys.layout.boxes[anc.overlay_node.idx()];
                assert!(
                    b.x >= 0.0
                        && b.y >= 0.0
                        && b.x + b.w <= (lw as f32) + 1.0
                        && b.y + b.h <= (lh as f32) + 1.0,
                    "Floating overlay {b:?} overflowed {lw}x{lh} viewport!"
                );
            }
        }
    }

    // 3. Verify Keyboard Tab navigation, Enter/Space activation, Escape focus-clear:
    // Focus order in header + rail on Overview: [0] Accent Hue slider, [1] Dark toggle, [2] Inspector button,
    // [3] 1. Telemetry & Grid, [4] 2. 100k Virtual Logs, [5] 3. SlotPool Kanban ...
    state.active_tab = StudioTab::Overview;
    state.show_inspector_popover = true;
    let _ = h.render_settled(1280, 800, 1.0, &state);
    h.sys.focused_slot = None;
    for _ in 0..3 {
        assert!(h.sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut state));
    }
    assert!(h.sys.focused_slot.is_some());
    assert!(h.sys.dispatch_key(&KeyInput::Enter, &mut state));
    assert!(
        !state.show_inspector_popover,
        "Enter on focused Inspector button must toggle popover closed"
    );
    assert!(h.sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut state));
    assert!(h.sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut state));
    assert!(h.sys.dispatch_key(&KeyInput::Space, &mut state));
    assert_eq!(
        state.active_tab,
        StudioTab::VirtualLogs,
        "Space on focused rail item must switch active workspace tab"
    );
    assert!(h.sys.dispatch_key(&KeyInput::Escape, &mut state));
    assert!(
        h.sys.focused_slot.is_none(),
        "Escape must clear keyboard focus"
    );

    // 4. Verify live O(log N) `VirtualListIndex::push` ("+ Stream +1,000 Logs") + Pass-4 `ui.tooltip` hover:
    let _ = h.render_settled(1280, 800, 1.0, &state);
    // Hover top-bar "Dark" toggle -> verifies Pass-4 `TopCenter -> BottomCenter` auto-flip inside [0..1280, 0..800]:
    let dark_pt = h.node_center(&state.nodes.dark_toggle);
    assert!(h.sys.pointer_move(dark_pt[0], dark_pt[1], &mut state));
    let _ = h.render_settled(1280, 800, 1.0, &state);
    assert!(
        !h.sys.layout.anchors.is_empty(),
        "Expected Pass-4 hover tooltip anchor on Dark toggle"
    );
    for anc in &h.sys.layout.anchors {
        let b = h.sys.layout.boxes[anc.overlay_node.idx()];
        assert!(
            b.x >= 0.0 && b.y >= 0.0 && b.x + b.w <= 1281.0 && b.y + b.h <= 801.0,
            "Header hover tooltip {b:?} must auto-flip inside viewport"
        );
    }

    // Hover & click "+ Stream +1,000 Logs" on Tab 2 -> verifies `VirtualListIndex::push` growing past 100,000 rows:
    let stream_pt = h.node_center(&state.nodes.stream_btn);
    let _ = h.sys.pointer_move(stream_pt[0], stream_pt[1], &mut state);
    let _ = h.render_settled(1280, 800, 1.0, &state);
    assert!(
        !h.sys.layout.anchors.is_empty(),
        "Expected Pass-4 hover tooltip on + Stream +1,000 Logs button"
    );
    assert!(h.sys.pointer_down(stream_pt[0], stream_pt[1], &mut state));
    let _ = h.sys.pointer_up();
    assert_eq!(state.vlist.len(), 101_000);
    assert_eq!(state.streamed_logs, 1_000);
    state.log_scroll_y = state.vlist.offset_of(100_986);
    let _ = h.sys.pointer_leave();
    let (_, _, rgba_stream) = h.render_settled(1280, 800, 1.0, &state);
    assert_theme_contrast(&rgba_stream, true, "VirtualLogs101kStream");

    // 5. Verify `ui.text_input` selection (`Ctrl+A`, `Shift+Arrows`, drag-select, `Ctrl+C`/`Ctrl+V`)
    //    in BOTH the Virtual Logs filter bar AND the Kanban task creator!
    let filter_box = h.sys.layout.boxes[state.nodes.filter_input.load(Ordering::Relaxed) as usize];
    let filter_cy = filter_box.y + filter_box.h * 0.5;
    assert!(h.sys.pointer_down(filter_box.x + 18.0, filter_cy, &mut state));
    let _ = h.sys.pointer_up();
    assert!(h.sys.dispatch_key(&KeyInput::SelectAll, &mut state));
    assert_eq!(
        state.search_query, "vulkan",
        "Ctrl+A (SelectAll) must select text without deleting it!"
    );
    assert_eq!(
        h.sys
            .focused_text_edit_state()
            .expect("focused filter TextEditState")
            .selection_range(&state.search_query),
        0..6
    );
    // Render with active translucent selection highlight quad:
    let _ = h.render_settled(1280, 800, 1.0, &state);
    assert!(h.sys.dispatch_key(&KeyInput::Copy, &mut state));
    assert_eq!(h.sys.clipboard_text(), "vulkan");
    assert!(h.sys.dispatch_key(&KeyInput::End, &mut state));
    for _ in 0..3 {
        assert!(h.sys.dispatch_key(&KeyInput::ShiftArrowLeft, &mut state));
    }
    assert_eq!(
        h.sys
            .focused_text_edit_state()
            .unwrap()
            .selection_range(&state.search_query),
        3..6
    );
    // Mouse click + drag selection across filter text_input:
    let _ = h.sys.pointer_down(filter_box.x + 4.0, filter_cy, &mut state);
    let _ = h.sys.pointer_move(filter_box.x + 90.0, filter_cy, &mut state);
    let _ = h.sys.pointer_up();
    assert!(
        h.sys
            .focused_text_edit_state()
            .unwrap()
            .has_selection(&state.search_query),
        "Drag-select inside filter input must produce non-empty selection"
    );

    // Switch to Tab 3 (Kanban), select all in task creator, paste clipboard (`Ctrl+V`), & submit via Enter:
    state.active_tab = StudioTab::PoolTasks;
    let _ = h.sys.pointer_leave();
    let _ = h.render_settled(1280, 800, 1.0, &state);
    let task_input_pt = h.node_center(&state.nodes.task_input);
    assert!(h.sys.pointer_down(task_input_pt[0], task_input_pt[1], &mut state));
    let _ = h.sys.pointer_up();
    assert!(h.sys.dispatch_key(&KeyInput::SelectAll, &mut state));
    assert_eq!(
        h.sys
            .focused_text_edit_state()
            .unwrap()
            .selection_range(&state.new_task_title),
        0..state.new_task_title.len()
    );
    assert!(h.sys.paste_clipboard(&mut state));
    assert_eq!(state.new_task_title, "vulkan");
    assert!(h.sys.dispatch_key(&KeyInput::ShiftHome, &mut state));
    let _ = h.render_settled(1280, 800, 1.0, &state);

    if let Some(&first_k) = state.task_keys.first() {
        state.remove_task(first_k);
        state.new_task_title = String::from("Recycled generational slot task");
        state.add_new_task();
    }
    let _ = h.sys.pointer_leave();
    h.sys.focused_slot = None;
    let (_, _, rgba_after) = h.render_settled(1280, 800, 1.0, &state);
    assert_theme_contrast(&rgba_after, true, "PostKeyAndRecycle");

    println!(
        "skima_studio 1.0 smoke-test passed on {backend_desc}: [1.0x, 1.125x, 1.25x, 1.333x, 1.5x, 1.75x, 2.0x] x Dark+Light (0 oscillation, 0 Moiré) + ScalePolicy::Auto Inspector + Tooltips + TextSelection/Clipboard + 101k Live Push!"
    );
}

/// Exports lossless Vulkan 1.3 fractional-scaling visual proof screenshots at `1.0x` (`120/120`),
/// `1.25x` (`150/120`), `1.333x` (`160/120`), and `1.50x` (`180/120`) across identical `1280x800`
/// logical viewports with live `ResolvedScale` display-physics telemetry & Pass-4 zoom tooltips!
fn export_scale_proof_suite(out_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let mut h = StudioHeadless::new();

    // 1. `1.00x` (`120/120` Native 1:1 Auto on 27" 2560x1440 108.8 PPI Desk Monitor):
    {
        let _ = h.sys.pointer_leave();
        h.sys.focused_slot = None;
        let mut st = StudioState::new();
        st.is_dark = true;
        st.active_tab = StudioTab::Overview;
        st.show_inspector_popover = true;
        let m_27_qhd = DisplayMetrics::from_physical_mm(2560, 1440, 597, 336);
        let (w, ph, rgba) =
            h.render_settled_with_metrics(1280, 800, ScalePolicy::Auto, m_27_qhd, &st);
        assert_eq!((w, ph), (1280, 800));
        let dst = out_dir.join("studio_scale_100x_120_120.png");
        save_rgba8_png(&dst, w, ph, &rgba)?;
        println!("Exported {w}x{ph} (1.00x 120/120) PNG -> {}", dst.display());
    }

    // 2. `1.25x` (`150/120 = 5/4` Exact Dyadic Auto on 32" 3840x2160 137.7 PPI Desk Monitor + Zoom Tooltip!):
    {
        let _ = h.sys.pointer_leave();
        h.sys.focused_slot = None;
        let mut st = StudioState::new();
        st.is_dark = true;
        st.active_tab = StudioTab::Overview;
        st.show_inspector_popover = true;
        let m_32_4k = DisplayMetrics::from_physical_mm(3840, 2160, 708, 398);
        let _ =
            h.render_settled_with_metrics(1280, 800, ScalePolicy::Auto, m_32_4k, &st);
        let pt = h.node_center(&st.nodes.scale_125_btn);
        let _ = h.sys.pointer_move(pt[0], pt[1], &mut st);
        let (w, ph, rgba) =
            h.render_settled_with_metrics(1280, 800, ScalePolicy::Auto, m_32_4k, &st);
        assert_eq!((w, ph), (1600, 1000));
        let dst = out_dir.join("studio_scale_125x_150_120.png");
        save_rgba8_png(&dst, w, ph, &rgba)?;
        println!("Exported {w}x{ph} (1.25x 150/120) PNG -> {}", dst.display());
    }

    // 3. `1.333x` (`160/120 = 4/3` Non-Dyadic Wayland Scale on 100k VirtualLogs + Open Inspector — 0 Moiré!):
    {
        let _ = h.sys.pointer_leave();
        h.sys.focused_slot = None;
        let mut st = StudioState::new();
        st.is_dark = true;
        st.active_tab = StudioTab::VirtualLogs;
        st.show_inspector_popover = true;
        let m_14_wqxga = DisplayMetrics::from_physical_mm(2560, 1600, 302, 189);
        let (w, ph, rgba) = h.render_settled_with_metrics(
            1280,
            800,
            ScalePolicy::Fixed(SCALE_1_33),
            m_14_wqxga,
            &st,
        );
        assert_eq!((w, ph), (1707, 1067));
        let dst = out_dir.join("studio_scale_133x_160_120.png");
        save_rgba8_png(&dst, w, ph, &rgba)?;
        println!(
            "Exported {w}x{ph} (1.333x 160/120) PNG -> {}",
            dst.display()
        );
    }

    // 4. `1.50x` (`180/120 = 3/2` Exact Dyadic Auto on 27" 3840x2160 163.2 PPI 4K Monitor, Light Theme!):
    {
        let _ = h.sys.pointer_leave();
        h.sys.focused_slot = None;
        let mut st = StudioState::new();
        st.is_dark = false;
        st.active_tab = StudioTab::Overview;
        st.show_inspector_popover = true;
        let m_27_4k = DisplayMetrics::from_physical_mm(3840, 2160, 597, 336);
        let (w, ph, rgba) =
            h.render_settled_with_metrics(1280, 800, ScalePolicy::Auto, m_27_4k, &st);
        assert_eq!((w, ph), (1920, 1200));
        let dst = out_dir.join("studio_scale_150x_180_120.png");
        save_rgba8_png(&dst, w, ph, &rgba)?;
        println!("Exported {w}x{ph} (1.50x 180/120) PNG -> {}", dst.display());
    }

    Ok(())
}

fn export_screenshot_suite(out_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let mut h = StudioHeadless::new();
    let mut state = StudioState::new();

    let captures: &[(&str, bool, StudioTab, bool, f64)] = &[
        ("studio_overview_dark.png", true, StudioTab::Overview, false, 0.0),
        ("studio_overview_light.png", false, StudioTab::Overview, false, 0.0),
        ("studio_inspector_dark.png", true, StudioTab::Overview, true, 0.0),
        ("studio_inspector_light.png", false, StudioTab::Overview, true, 0.0),
        ("studio_logs_dark.png", true, StudioTab::VirtualLogs, false, 0.0),
        ("studio_logs_light.png", false, StudioTab::VirtualLogs, false, 0.0),
        ("studio_kanban_dark.png", true, StudioTab::PoolTasks, false, 0.0),
        ("studio_kanban_light.png", false, StudioTab::PoolTasks, false, 0.0),
    ];

    for &(filename, is_dark, tab, show_insp, scroll_idx) in captures {
        let _ = h.sys.pointer_leave();
        h.sys.focused_slot = None;
        state.is_dark = is_dark;
        state.active_tab = tab;
        state.show_inspector_popover = show_insp;
        if tab == StudioTab::VirtualLogs {
            let target_row = if is_dark { 0 } else { 49_998 };
            if !is_dark && state.expanded_rows.insert(50_000) {
                state.vlist.set_height(50_000, LOG_ROW_EXPANDED_H);
            }
            state.log_scroll_y = if scroll_idx > 0.0 {
                scroll_idx
            } else {
                state.vlist.offset_of(target_row)
            };
        }
        let (w, ph, rgba) = h.render_settled(1280, 800, 1.0, &state);
        let dst = out_dir.join(filename);
        save_rgba8_png(&dst, w, ph, &rgba)?;
        println!("Exported {w}x{ph} PNG -> {}", dst.display());
    }

    // Showcase 9: Live O(log N) "+ Stream +1,000 Logs" (101,000 rows) + Pass-4 Hover Tooltip!
    {
        let mut s_stream = StudioState::new();
        s_stream.is_dark = true;
        s_stream.active_tab = StudioTab::VirtualLogs;
        s_stream.stream_logs(1_000);
        s_stream.log_scroll_y = s_stream.vlist.offset_of(99_998);
        let _ = h.render_settled(1280, 800, 1.0, &s_stream);
        let pt = h.node_center(&s_stream.nodes.stream_btn);
        let _ = h.sys.pointer_move(pt[0], pt[1], &mut s_stream);
        let (w, ph, rgba) = h.render_settled(1280, 800, 1.0, &s_stream);
        let dst = out_dir.join("studio_logs_streamed_tooltip.png");
        save_rgba8_png(&dst, w, ph, &rgba)?;
        println!("Exported {w}x{ph} PNG -> {}", dst.display());
    }

    // Showcase 10: Live `ui.text_input` UTF-8 Selection Highlight + Disjoint-Borrow Hover Tooltip!
    {
        let _ = h.sys.pointer_leave();
        h.sys.focused_slot = None;
        let mut s_sel = StudioState::new();
        s_sel.is_dark = true;
        s_sel.active_tab = StudioTab::PoolTasks;
        s_sel.swap_top_two_priorities();
        let _ = h.render_settled(1280, 800, 1.0, &s_sel);
        let pt = h.node_center(&s_sel.nodes.task_input);
        let _ = h.sys.pointer_down(pt[0], pt[1], &mut s_sel);
        let _ = h.sys.pointer_up();
        let _ = h.sys.dispatch_key(&KeyInput::SelectAll, &mut s_sel);
        let swap_pt = h.node_center(&s_sel.nodes.swap_btn);
        let _ = h.sys.pointer_move(swap_pt[0], swap_pt[1], &mut s_sel);
        let (w, ph, rgba) = h.render_settled(1280, 800, 1.0, &s_sel);
        let dst = out_dir.join("studio_kanban_selection_tooltip.png");
        save_rgba8_png(&dst, w, ph, &rgba)?;
        println!("Exported {w}x{ph} PNG -> {}", dst.display());
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--smoke-test") {
        run_smoke_test();
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--export-screenshots")
        && let Some(dir_str) = args.get(pos + 1)
    {
        let out_dir = Path::new(dir_str);
        export_screenshot_suite(out_dir).expect("export Studio PNG screenshot suite");
        export_scale_proof_suite(out_dir).expect("export fractional scale proof PNG suite");
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--export-scale-proof")
        && let Some(dir_str) = args.get(pos + 1)
    {
        export_scale_proof_suite(Path::new(dir_str))
            .expect("export fractional scale proof PNG suite");
        return;
    }

    let state = StudioState::new();

    if let Some(pos) = args.iter().position(|a| a == "--render-png")
        && let Some(out_path) = args.get(pos + 1)
    {
        let (rw, rh) = args
            .iter()
            .position(|a| a == "--size")
            .and_then(|p| args.get(p + 1))
            .and_then(|s| s.split_once('x'))
            .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
            .unwrap_or((1280, 800));
        let render_sf = args
            .iter()
            .position(|a| a == "--scale")
            .and_then(|p| args.get(p + 1))
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        let mut h = StudioHeadless::new();
        let (w, ph, rgba) = h.render_settled(rw, rh, render_sf, &state);
        save_rgba8_png(out_path, w, ph, &rgba).expect("write studio PNG");
        println!("Saved {w}x{ph} ({render_sf:.3}x) Skima Studio PNG frame to {out_path}");
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--render-ppm")
        && let Some(out_path) = args.get(pos + 1)
    {
        let mut h = StudioHeadless::new();
        let (w, ph, rgba) = h.render_settled(1280, 800, 1.0, &state);
        save_rgba8_ppm(out_path, w, ph, &rgba).expect("write studio PPM");
        println!("Saved {w}x{ph} Skima Studio PPM frame to {out_path}");
        return;
    }

    #[cfg(feature = "window")]
    {
        let max_frames = args
            .iter()
            .position(|a| a == "--max-frames")
            .and_then(|p| args.get(p + 1))
            .and_then(|s| s.parse::<u64>().ok());

        skima::run_window(
            skima::WindowConfig {
                title: String::from(
                    "Skima Studio 1.0 — Vulkan 1.3 ReBAR + Automatic Fractional HiDPI",
                ),
                width: 1280,
                height: 800,
                vsync: true,
                load_system_fonts: true,
                scale_policy: ScalePolicy::Auto,
                max_frames,
            },
            state,
            build_studio,
        )
        .expect("launch Skima Studio window");
    }
    #[cfg(not(feature = "window"))]
    {
        let _ = state;
        eprintln!(
            "Built without `window` feature; pass `--smoke-test`, `--export-screenshots <dir>`, `--export-scale-proof <dir>`, `--render-png <path>`, or `--render-ppm <path>`."
        );
    }
}
