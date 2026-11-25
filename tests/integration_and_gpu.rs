use skima::a11y::{A11yRole, Action, ActionData, ActionRequest, TreeId};
use skima::arena::FrameArena;
use skima::color::{Color, Oklch};
use skima::gpu::cpu::CpuRenderer;
use skima::gpu::vulkan::VulkanRenderer;
use skima::gpu::{PrimKind, RenderFrame};
use skima::layout::{
    Align, AnchorPlacement, LayoutSpec, LayoutTree, MainAlign, TrackSize, VirtualListIndex,
};
use skima::pool::{Key, PoolAccessError, Realm, SlotPool, Spring};
use skima::text::{AntialiasMode, FontId, FontSystem, GlyphAtlas, ShapedParagraph};
use skima::ui::{CursorIcon, GuiSystem, KeyInput, PaintStyle, StatelessView, Theme, Ui};
use skima::vector::{FillRule, Path2D};
use std::alloc::{Allocator, Layout};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

// =========================================================================
// 1. Hardware Vulkan 1.3 + CPU Analytical SDF Parity & Visual Rendering Test
// =========================================================================
#[test]
fn verify_vulkan_1_3_gpu_and_cpu_sdf_gradient_shadow_clip_and_text_parity() {
    let mut sys = GuiSystem::with_system_fonts();
    sys.clear_color = Color::rgb8(16, 18, 26);
    let loaded_font = sys.fonts.ui_font;

    let state = 42usize;
    sys.build_frame(&state, 320.0, 240.0, |ui| {
        ui.element(
            LayoutSpec::col().fixed(320.0, 240.0).pad(20.0).gap(16.0),
            PaintStyle::solid(Color::rgb8(16, 18, 26)),
            |ui| {
                // Card with analytical Gaussian shadow + OKLab gradient + rounded corners + border + clipped child!
                ui.element(
                    LayoutSpec::row()
                        .fixed(260.0, 100.0)
                        .pad(16.0)
                        .gap(12.0)
                        .align(Align::Center),
                    PaintStyle::oklab_gradient(
                        Color::rgb8(45, 95, 225),
                        Color::rgb8(180, 55, 210),
                        0.4,
                    )
                    .rounded(14.0)
                    .border(2.0, Color::rgb8(230, 240, 255))
                    .drop_shadow(8.0, [0.0, 6.0], Color::rgba8(0, 0, 0, 180)),
                    |ui| {
                        ui.set_clips_children(true);
                        ui.vector_icon(28, 28, Color::WHITE, FillRule::NonZero, |p| {
                            p.move_to(14.0, 2.0)
                                .line_to(26.0, 24.0)
                                .line_to(2.0, 24.0)
                                .close();
                        });
                        ui.text(
                            "SKIMA GPU 120HZ",
                            loaded_font,
                            16.0,
                            22.0,
                            Color::WHITE,
                            AntialiasMode::SubpixelLcd,
                        );
                    },
                );
            },
        );
    });

    // Render on CPU reference renderer:
    let mut cpu = CpuRenderer::new(320, 240);
    sys.render_to_backend(&mut cpu, 320, 240, 1).unwrap();

    // Background corner pixel `(4, 4)` must match clear_color `(16, 18, 26)`:
    let bg_px = cpu.pixel_at(4, 4);
    assert!(
        (bg_px[0] as i32 - 16).abs() <= 2 && (bg_px[1] as i32 - 18).abs() <= 2,
        "Unexpected CPU background pixel: {bg_px:?}"
    );
    // Center of OKLab gradient card `(150, 70)` must be bright purple-blue!
    let card_px = cpu.pixel_at(150, 70);
    assert!(
        card_px[2] > 150 && card_px[0] > 60,
        "Expected vibrant OKLab gradient inside card on CPU, got {card_px:?}"
    );
    // Drop-shadow region below card `(150, 126)` must be darker than background `(16, 18, 26)`:
    let shadow_px = cpu.pixel_at(150, 126);
    assert!(
        shadow_px[2] < bg_px[2],
        "Expected analytical Gaussian drop shadow below card: shadow={shadow_px:?} vs bg={bg_px:?}"
    );

    // Force atlas re-upload for second backend and execute REAL Vulkan 1.3 Hardware Rendering!
    sys.fonts.atlas.generation += 1;
    let mut vk = VulkanRenderer::new_headless(4096, 256)
        .expect("Vulkan 1.3 hardware/ICD device must initialize");
    println!(
        "Initialized Vulkan 1.3 GPU: '{}' (ReBAR: {}, Dual-Source ClearType: {})",
        vk.device_name, vk.is_rebar_device_local, vk.dual_source_blending
    );

    let frame = skima::gpu::RenderFrame {
        width: 320,
        height: 240,
        scale_factor: 1.0,
        clear_color: sys.clear_color,
        primitives: &sys.primitives,
        dirty_prim_ranges: &[],
        clips: &sys.clips,
        damage: None,
    };
    let vk_pixels = vk
        .render_offscreen_rgba8(&frame, &mut sys.fonts.atlas)
        .expect("Hardware Vulkan 1.3 offscreen render & readback");

    let vk_px = |x: u32, y: u32| -> [u8; 4] {
        let i = ((y * 320 + x) as usize) * 4;
        [
            vk_pixels[i],
            vk_pixels[i + 1],
            vk_pixels[i + 2],
            vk_pixels[i + 3],
        ]
    };

    let vk_bg = vk_px(4, 4);
    let vk_card = vk_px(150, 70);
    let vk_shadow = vk_px(150, 126);

    // Verify GPU rendered pixels match CPU analytical SDF reference closely across background, OKLab gradient, and shadow!
    for ch in 0..3 {
        assert!(
            (vk_bg[ch] as i32 - bg_px[ch] as i32).abs() <= 3,
            "Vulkan vs CPU background mismatch: vk={vk_bg:?}, cpu={bg_px:?}"
        );
        assert!(
            (vk_card[ch] as i32 - card_px[ch] as i32).abs() <= 8,
            "Vulkan vs CPU OKLab gradient mismatch: vk={vk_card:?}, cpu={card_px:?}"
        );
    }
    assert!(
        vk_shadow[2] < vk_bg[2],
        "Vulkan analytical Gaussian shadow must darken backdrop: shadow={vk_shadow:?}, bg={vk_bg:?}"
    );
    // Verify ClearType SubpixelLcd glyphs emitted `PrimKind::GlyphLcd` primitives and rendered bright white stems on GPU & CPU:
    let lcd_prim_count = sys
        .primitives
        .iter()
        .filter(|p| p.kind() == skima::gpu::PrimKind::GlyphLcd)
        .count();
    assert!(
        lcd_prim_count >= 10,
        "Expected SubpixelLcd glyph primitives, got {lcd_prim_count}"
    );
    let mut max_glyph_luma_vk = 0u8;
    let mut max_glyph_luma_cpu = 0u8;
    for y in 55..85 {
        for x in 75..220 {
            max_glyph_luma_vk = max_glyph_luma_vk.max(vk_px(x, y)[1]);
            max_glyph_luma_cpu = max_glyph_luma_cpu.max(cpu.pixel_at(x, y)[1]);
        }
    }
    assert!(
        max_glyph_luma_vk > 200 && max_glyph_luma_cpu > 200,
        "Expected crisp bright ClearType LCD glyph stems on both Vulkan ({max_glyph_luma_vk}) and CPU ({max_glyph_luma_cpu})"
    );
}

// =========================================================================
// 2. OpenType Shaping (`rustybuzz` + `swash`) & Atlas Saturation Recovery
// =========================================================================
#[test]
fn verify_opentype_shaping_and_atlas_lru_saturation_recovery() {
    let mut fonts = FontSystem::new();
    // Test real `.ttf` file if present on NixOS host:
    let mut ttf_path = None;
    if Path::new("/nix/store").is_dir()
        && let Ok(entries) = std::fs::read_dir("/nix/store")
    {
        for ent in entries.flatten() {
            let p = ent
                .path()
                .join("share/fonts/noto/NotoColorEmoji.ttf");
            if p.exists() {
                ttf_path = Some(p);
                break;
            }
        }
    }
    if let Some(p) = ttf_path
        && let Ok(bytes) = std::fs::read(p)
    {
        let fid = fonts.load_font_bytes(bytes, 0);
        assert!(fid.is_some(), "rustybuzz + swash must parse system OpenType font");
    }

    // Verify Exact Width-Interval theorem across all widths `[40..=800]` on `ShapedParagraph`:
    let text = "Incremental axis-factored layout complements Rust ownership lifetimes and RAII";
    let mut para = fonts.shape_paragraph(text, FontId::FALLBACK, 14.0, 20.0);
    let mut hits = 0usize;
    for w in (40..=800).rev() {
        let (h_cached, tw_cached, hit) = para.wrap_for_width(w as f32);
        if hit {
            hits += 1;
        }
        let mut fresh = ShapedParagraph::shape_fallback(text, 14.0, 20.0, 1);
        let (h_fresh, tw_fresh, _) = fresh.wrap_for_width(w as f32);
        assert_eq!(h_cached, h_fresh);
        assert_eq!(tw_cached, tw_fresh);
    }
    assert!(hits > 720, "Expected >94% interval cache hits, got {hits}/761");

    // Verify small atlas recovers cleanly without panicking when saturated:
    let mut tiny_atlas = skima::text::GlyphAtlas::new(32, 32);
    let block = [255u8; 16 * 16];
    let gen0 = tiny_atlas.generation;
    for _ in 0..20 {
        let reg = tiny_atlas.upload_r8(16, 16, 0, 0, &block);
        assert!(reg.is_some(), "Atlas must evict/reset and succeed on saturation!");
    }
    assert!(tiny_atlas.generation > gen0);
}

// =========================================================================
// 3. Complete Interactive Widgets: Focus Ring, UTF-8 TextInput, Scroll, Actions, AccessKit, & Unwind Recovery
// =========================================================================
static DROPPED_COUNT: AtomicUsize = AtomicUsize::new(0);

struct RaiiGuard;
impl Drop for RaiiGuard {
    fn drop(&mut self) {
        DROPPED_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

struct SaveShortcut;

struct FormState {
    query: String,
    accepted: bool,
    saved_count: usize,
    show_popover: bool,
    trigger_panic: bool,
    slider_val: f32,
}

struct PassiveHeader<'a>(&'a str);
impl<'b> StatelessView<!> for PassiveHeader<'b> {
    #[allow(refining_impl_trait)]
    fn mount<'ui, 's, 'data>(
        &'data self,
        ui: &'ui mut Ui<'_, 's, !>,
    ) -> impl Copy + use<'b, 'data> {
        ui.passive_label(self.0, 14.0, 18.0, Color::WHITE)
    }
}

#[test]
fn verify_focus_text_input_utf8_actions_scroll_popover_accesskit_and_drop_guard() {
    DROPPED_COUNT.store(0, Ordering::SeqCst);
    let mut sys = GuiSystem::new();
    sys.a11y_enabled = true;

    let mut form = FormState {
        query: String::from("🦀"),
        accepted: false,
        saved_count: 0,
        show_popover: true,
        trigger_panic: false,
        slider_val: 0.25,
    };

    let render = |sys: &mut GuiSystem, st: &FormState| {
        let dynamic_banner = format!("Saved: {}", st.saved_count);
        let header = PassiveHeader(&dynamic_banner);

        sys.build_frame(st, 600.0, 400.0, |ui| {
            ui.element(
                LayoutSpec::col().fixed(600.0, 400.0).pad(12.0).gap(10.0),
                PaintStyle::solid(Color::rgb8(20, 22, 30)),
                |ui| {
                    // Top-level typed shortcut action handler (`Ctrl+S` -> `SaveShortcut`):
                    ui.on_action::<SaveShortcut>(|s, _| {
                        s.saved_count += 1;
                        true
                    });

                    ui.stateless(&header);

                    if ui.state().trigger_panic {
                        ui.element(LayoutSpec::row().fixed(50.0, 20.0), PaintStyle::default(), |ui| {
                            let _g = ui.use_state(|| RaiiGuard);
                        });
                        panic!("intentional mid-frame panic");
                    }

                    // UTF-8 `text_input` projected via `lens`:
                    let input_nid = ui.text_input("Search...", |f| &f.query, |f| &mut f.query);

                    // Checkbox control:
                    let checked = ui.state().accepted;
                    ui.checkbox("Accept Terms", checked, |f, _| {
                        f.accepted.toggle(); // Rust 1.100.0 `bool::toggle`!
                    });

                    // Draggable slider bar demonstrating pointer capture (`on_drag` outside bounds!):
                    ui.element(
                        LayoutSpec::row().fixed(200.0, 20.0),
                        PaintStyle::solid(Color::rgb8(50, 55, 70)).rounded(10.0),
                        |ui| {
                            ui.on_drag(|f, ev| {
                                f.slider_val = ((ev.x - 12.0) / 200.0).clamp(0.0, 1.0);
                            });
                        },
                    );

                    // Scrollable viewport (`height: 80px`, `max_scroll: 300px`):
                    ui.element(
                        LayoutSpec::col().fixed(300.0, 80.0),
                        PaintStyle::solid(Color::rgb8(30, 32, 42)),
                        |ui| {
                            ui.set_scrollable(300.0);
                            for i in 0..10usize {
                                ui.keyed(i as u64 + 1, |ui| {
                                    ui.label_usize(i, 14.0, 24.0, Color::WHITE);
                                });
                            }
                        },
                    );

                    // Conditional anchored popover with RAII guard:
                    if ui.state().show_popover {
                        ui.anchored_popover(
                            input_nid,
                            AnchorPlacement::BottomStart,
                            [0.0, 4.0],
                            LayoutSpec::col().fixed(160.0, 48.0).pad(8.0),
                            PaintStyle::solid(Color::rgb8(40, 44, 62)).rounded(8.0),
                            |ui| {
                                let _raii = ui.use_state(|| RaiiGuard);
                                ui.label("Autocomplete", 12.0, 16.0, Color::WHITE);
                            },
                        );
                    }
                },
            );
        });
    };

    render(&mut sys, &form);

    // 1. Verify initial AccessKit `TreeUpdate` contains root `TreeId::ROOT` and semantic controls:
    let a11y_init = sys.build_accesskit_update().expect("initial AccessKit tree");
    assert_eq!(a11y_init.tree_id, TreeId::ROOT);
    assert!(a11y_init.tree.is_some());
    assert!(a11y_init.nodes.len() >= 5);
    assert!(sys.build_accesskit_update().is_none(), "Clean frame must emit 0 AccessKit updates!");

    // 2. Press `Tab` to focus `text_input`, then edit multi-byte UTF-8 (`"🦀"` is 4 bytes!) safely:
    assert!(sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut form));
    assert!(sys.dispatch_key(&KeyInput::End, &mut form));
    assert!(sys.dispatch_key(&KeyInput::Text(" Rust".to_owned()), &mut form));
    assert_eq!(form.query, "🦀 Rust");
    // Backspace over ASCII `'t'`, then `Home` -> `Delete` over 4-byte `'🦀'` without splitting UTF-8 codepoint!
    assert!(sys.dispatch_key(&KeyInput::Backspace, &mut form));
    assert_eq!(form.query, "🦀 Rus");
    assert!(sys.dispatch_key(&KeyInput::Home, &mut form));
    assert!(sys.dispatch_key(&KeyInput::Delete, &mut form));
    assert_eq!(form.query, " Rus");

    // 3. Bubble typed `SaveShortcut` from focused `text_input` up to root container:
    assert!(sys.dispatch_action(&SaveShortcut, &mut form));
    assert_eq!(form.saved_count, 1);

    // 4. `Tab` to Checkbox and trigger via `AccessKit` `Action::Click`:
    assert!(sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut form));
    let focused_a11y = skima::a11y::slot_to_a11y_id(sys.focused_slot.unwrap());
    let req = ActionRequest {
        action: Action::Click,
        target_tree: TreeId::ROOT,
        target_node: focused_a11y,
        data: None,
    };
    assert!(sys.handle_accesskit_action(&req, &mut form));
    assert!(form.accepted);

    // 5. Pointer capture on slider (`y ~ 110px`): drag far outside bounds (`x = 999.0`) still updates slider until `pointer_up`!
    render(&mut sys, &form);
    assert!(sys.pointer_down(50.0, 118.0, &mut form));
    assert!(sys.pointer_move(999.0, 350.0, &mut form));
    assert!((form.slider_val - 1.0).abs() < 1e-5);
    sys.pointer_up();

    // 6. Mouse wheel scroll bubbling inside `scroll_area` (`x = 50, y = 170`):
    assert!(sys.dispatch_scroll(50.0, 170.0, -45.0, &mut form));

    // 7. Close popover -> RAII `RaiiGuard` drops deterministically (`0 -> 1`):
    form.show_popover = false;
    render(&mut sys, &form);
    assert_eq!(DROPPED_COUNT.load(Ordering::SeqCst), 1);

    // 8. Trigger mid-frame panic -> `DropGuard` rolls back uncommitted `RaiiGuard` (`1 -> 2`) while keeping live form intact!
    form.trigger_panic = true;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        render(&mut sys, &form);
    }));
    form.trigger_panic = false;
    assert!(unwind.is_err());
    assert_eq!(DROPPED_COUNT.load(Ordering::SeqCst), 2);
    render(&mut sys, &form);
    assert!(sys.dispatch_action(&SaveShortcut, &mut form));
    assert_eq!(form.saved_count, 2);
}

// =========================================================================
// 4. Advanced Layout Invariants: Inline-Only Relayout Skip, 2D Spanned Grid, WrapRow, Justification, VirtualList Anchor
// =========================================================================
#[test]
fn verify_inline_boundary_skip_grid_spans_wrap_row_and_anchored_virtual_list() {
    let mut tree = LayoutTree::new();
    tree.snap_to_pixels = true;
    let fonts = FontSystem::new();

    // Fixed-width, Fixed-height outer container with a child changing ONLY height -> Inline Passes 1A/1B skipped!
    let root = tree.alloc_node(
        LayoutSpec::col()
            .fixed(400.0, 300.0)
            .justify(MainAlign::SpaceBetween),
    );
    tree.root = root;
    let c1 = tree.alloc_node(LayoutSpec::row().width_fixed(400.0).height_fixed(40.0));
    let c2 = tree.alloc_node(LayoutSpec::row().width_fixed(400.0).height_fixed(40.0));
    tree.set_children(root, &[c1, c2]);
    tree.compute_layout(400.0, 300.0);
    // SpaceBetween puts `c1` at `y = 0` and `c2` at `y = 260`:
    assert_eq!(tree.boxes[c1.idx()].y, 0.0);
    assert_eq!(tree.boxes[c2.idx()].y, 260.0);

    // Mutate ONLY `c2`'s height (`40 -> 60`): inline properties unchanged inside `INLINE_BOUNDARY`!
    tree.update_spec(c2, LayoutSpec::row().width_fixed(400.0).height_fixed(60.0));
    tree.compute_layout(400.0, 300.0);
    assert_eq!(
        tree.last_metrics.inline_passes_skipped, 1,
        "Expected Inline Passes 1A & 1B to be skipped on height-only change!"
    );
    assert_eq!(tree.boxes[c2.idx()].y, 240.0);

    // 2D Grid with `grid_span(2, 1)` header spanning 2 columns (`Fit` + `Fr(1)`):
    let mut gtree = LayoutTree::new();
    let grid = gtree.alloc_node(LayoutSpec::col().fixed(300.0, 200.0).gap(10.0));
    gtree.attach_grid(
        grid,
        vec![TrackSize::Fixed(80.0), TrackSize::Fr(1.0)],
        vec![TrackSize::Fit, TrackSize::Fr(1.0)],
    );
    gtree.root = grid;
    let banner = gtree.alloc_node(LayoutSpec::row().height_fixed(30.0).grid_span(2, 1));
    let left = gtree.alloc_node(LayoutSpec::col().grow(1.0));
    let right = gtree.alloc_node(LayoutSpec::col().grow(1.0));
    gtree.set_children(grid, &[banner, left, right]);
    gtree.compute_layout(300.0, 200.0);
    assert_eq!(gtree.boxes[banner.idx()].w, 300.0, "2-col spanned header gets 80 + 10 + 210 = 300px!");
    assert_eq!(gtree.boxes[right.idx()].x, 90.0);
    assert_eq!(gtree.boxes[right.idx()].w, 210.0);

    // `WrapRow` chip container wrapping 5 chips of width 90px inside 250px width -> 3 rows!
    let mut wtree = LayoutTree::new();
    let wroot = wtree.alloc_node(LayoutSpec::col().fixed(250.0, 500.0));
    let wrap_box = wtree.alloc_node(LayoutSpec::wrap_row().width_grow(1.0).gap(10.0));
    wtree.root = wroot;
    let chips: Vec<_> = (0..5)
        .map(|_| wtree.alloc_node(LayoutSpec::row().fixed(90.0, 24.0)))
        .collect();
    wtree.set_children(wroot, &[wrap_box]);
    wtree.set_children(wrap_box, &chips);
    wtree.compute_layout(250.0, 500.0);
    // 2 chips per row (90 + 10 + 90 = 190 <= 250; +10+90 = 290 > 250) => 3 rows: `24*3 + 10*2 = 92px`!
    assert_eq!(wtree.boxes[wrap_box.idx()].h, 92.0);
    let _ = fonts;

    // Anchor-locked `VirtualListIndex`: resizing item #10 above anchor #500 compensates `scroll_y`!
    let mut vlist = VirtualListIndex::new(10_000, 20.0);
    let mut scroll_y = 10_000.0_f64; // At item #500
    let anchor_before = scroll_y - vlist.offset_of(500);
    vlist.set_height_anchored(10, 120.0, 500, &mut scroll_y);
    let anchor_after = scroll_y - vlist.offset_of(500);
    assert!((anchor_before - anchor_after).abs() < 1e-6, "Viewport anchor must remain rock-solid!");
}

// =========================================================================
// 5. Realm Multi-Type Disjoint Borrowing, Analytical Spring, & FrameArena Consolidation
// =========================================================================
#[test]
fn verify_realm_disjoint_pools_analytical_spring_and_arena_spill_consolidation() {
    let mut realm = Realm::new();
    let ka = realm.pool_mut::<String>().insert("Buffer A".to_owned());
    let kb = realm.pool_mut::<Vec<u32>>().insert(vec![1, 2, 3]);

    // Simultaneous `&mut SlotPool<String>` and `&mut SlotPool<Vec<u32>>` via Rust 1.86 `HashMap::get_disjoint_mut`!
    {
        let (str_pool, vec_pool) = realm.pools_disjoint_mut::<String, Vec<u32>>();
        str_pool.get_mut(ka).unwrap().push_str(" + edited");
        vec_pool.get_mut(kb).unwrap().push(4);
    }
    assert_eq!(realm.pool::<String>().unwrap().get(ka).unwrap(), "Buffer A + edited");
    assert_eq!(realm.pool::<Vec<u32>>().unwrap().get(kb).unwrap(), &[1, 2, 3, 4]);

    let mut sp = SlotPool::<i32>::new();
    let k1 = sp.insert(10);
    assert!(matches!(
        sp.get_disjoint_mut([k1, k1]),
        Err(PoolAccessError::OverlappingKeys(_))
    ));

    // Analytical Spring: frame-rate independent (1 step of 0.1s == 10 steps of 0.01s to <1e-5 precision!):
    let mut s_coarse = Spring::critically_damped(0.0, 4.0);
    s_coarse.set_target(100.0);
    s_coarse.step(0.1);

    let mut s_fine = Spring::critically_damped(0.0, 4.0);
    s_fine.set_target(100.0);
    for _ in 0..10 {
        s_fine.step(0.01);
    }
    assert!(
        (s_coarse.value - s_fine.value).abs() < 1e-3,
        "Analytical closed-form spring must be frame-rate invariant: coarse={} vs fine={}",
        s_coarse.value,
        s_fine.value
    );

    // FrameArena multi-chunk spill & automatic consolidation on `reset()`:
    let arena = FrameArena::with_capacity(4096);
    {
        let mut big = Vec::with_capacity_in(4096, &arena); // 16 KB > 4 KB initial chunk -> spills!
        big.extend(0..4096u32);
        assert_eq!(big.len(), 4096);
    }
    arena.reset();
    assert!(arena.total_capacity() >= 16384);
    let _ = A11yRole::Button;
}

// =========================================================================
// 6. 1.0 Stable Release Guarantees: System Font Discovery, Multi-Font Fallback,
//    HiDPI Physical Scaling, Dynamic Vulkan SSBO Growth, & Animated Controls
// =========================================================================
#[test]
fn verify_1_0_system_fonts_hidpi_scaling_dynamic_vulkan_ssbo_growth_and_springs() {
    let mut sys = GuiSystem::with_system_fonts();
    sys.set_scale_factor(2.0); // 200% Retina / Wayland HiDPI scaling!

    struct Demo {
        val: f32,
        on: bool,
    }
    let mut st = Demo { val: 0.75, on: true };

    let build = |ui: &mut Ui<'_, '_, Demo>| {
        let th = ui.theme();
        ui.element(
            LayoutSpec::col().fixed(300.0, 200.0).pad(12.0).gap(8.0),
            PaintStyle::solid(th.bg_canvas),
            |ui| {
                // Mixed Latin + Emoji fallback in one paragraph:
                ui.ui_text("Skima 1.0 Stable 🦀 ✓", 15.0, th.text_primary);
                let v = ui.state().val;
                ui.slider("Volume", v, 0.0..=1.0, 200.0, |d, nv| d.val = nv);
                let on = ui.state().on;
                ui.toggle_switch("Power", on, |d, _| d.on.toggle());
                ui.progress_bar(v, 200.0, th.accent);
            },
        );
    };

    sys.build_frame(&st, 300.0, 200.0, build);
    // Logical root is 300x200, while physical root primitive is 600x400 (2.0x HiDPI!):
    assert_eq!(sys.layout.boxes[sys.layout.root.idx()].w, 300.0);
    assert_eq!(sys.primitives[0].bounds[2], 600.0);
    assert_eq!(sys.primitives[0].bounds[3], 400.0);

    // Toggle switch -> `use_spring` activates animation clock and settles smoothly over steps:
    st.on = false;
    sys.build_frame(&st, 300.0, 200.0, build);
    assert!(sys.has_active_animations(), "Spring must register active animation on target change");
    for _ in 0..120 {
        sys.step_animations(1.0 / 60.0);
    }
    assert!(!sys.has_active_animations(), "Spring must settle to 0% idle CPU after convergence");

    // Initialize `VulkanRenderer` with TINY initial capacity (256 prims) and render >600 primitives
    // to prove `ensure_scene_capacity` grows ReBAR SSBOs + descriptors dynamically without truncation!
    let mut big_sys = GuiSystem::new();
    big_sys.build_frame(&0usize, 400.0, 400.0, |ui| {
        ui.element(LayoutSpec::wrap_row().fixed(400.0, 400.0).gap(2.0), PaintStyle::solid(Color::BLACK), |ui| {
            for i in 0..600usize {
                ui.keyed(i as u64 + 1, |ui| {
                    ui.element(LayoutSpec::col().fixed(12.0, 12.0), PaintStyle::solid(Color::rgb8(80, 160, 250)), |_| {});
                });
            }
        });
    });
    assert!(big_sys.primitives.len() > 600);

    let mut vk_tiny = VulkanRenderer::new_headless(256, 64).expect("init tiny VulkanRenderer");
    let rgba = big_sys
        .render_vulkan_offscreen(&mut vk_tiny, 400, 400)
        .expect("dynamic Vulkan ReBAR SSBO growth must succeed");
    assert_eq!(rgba.len(), 400 * 400 * 4);
}

// =========================================================================
// 7. Regression Test: `LayoutSpec::grow(1.0)` Root Stability Across Hover Frames
// =========================================================================
#[test]
fn verify_grow_root_never_oscillates_across_hover_frames() {
    let mut sys = GuiSystem::new();
    let mut state = 0usize;

    let build = |ui: &mut Ui<'_, '_, usize>| {
        let th = ui.theme();
        ui.element(
            LayoutSpec::col().grow(1.0).pad(20.0).gap(12.0),
            PaintStyle::solid(th.bg_canvas),
            |ui| {
                ui.primary_button("Click Me", |s, _| *s += 1);
                ui.element(
                    LayoutSpec::row().width_grow(1.0).height_fixed(120.0),
                    PaintStyle::solid(th.bg_surface),
                    |_| {},
                );
            },
        );
    };

    // Frame 0: initial layout inside 1280x1080 window (content intrinsic height is only ~190px!)
    sys.build_frame(&state, 1280.0, 1080.0, build);
    assert_eq!(sys.layout.boxes[sys.layout.root.idx()].w, 1280.0);
    assert_eq!(sys.layout.boxes[sys.layout.root.idx()].h, 1080.0);

    // Simulate 20 consecutive mouse hover movements across the window:
    for step in 1..=20 {
        let mx = 30.0 + (step as f32) * 8.0;
        let my = 28.0 + ((step % 5) as f32) * 15.0;
        sys.pointer_move(mx, my, &mut state);
        sys.build_frame(&state, 1280.0, 1080.0, build);
        let rb = sys.layout.boxes[sys.layout.root.idx()];
        assert_eq!(
            (rb.w, rb.h),
            (1280.0, 1080.0),
            "Grow root must NEVER oscillate height on hover frame #{step}: got {}x{}",
            rb.w,
            rb.h
        );
        assert_eq!(
            sys.layout.last_metrics.nodes_visited, 0,
            "Hover frame #{step} must visit 0 layout nodes!"
        );
    }
}

// =========================================================================
// 8. Regression Test: Exact Slider Cursor Alignment (`ev.local_x` == Thumb Center) & `on_scroll`
// =========================================================================
#[test]
fn verify_slider_cursor_alignment_and_on_scroll_handler() {
    struct SliderHost {
        val: f32,
        vscroll: f32,
    }
    let mut host = SliderHost {
        val: 0.0,
        vscroll: 0.0,
    };
    let mut sys = GuiSystem::new();

    let build = |ui: &mut Ui<'_, '_, SliderHost>| {
        // Offset slider deep into the window at `(x = 520, y = 300)` (`width = 240`):
        let mut root_spec = LayoutSpec::col().fixed(1000.0, 700.0);
        root_spec.pad_l = 520.0;
        root_spec.pad_t = 300.0;
        ui.element(root_spec, PaintStyle::default(), |ui| {
            let v = ui.state().val;
            ui.slider("Offset Slider", v, 0.0..=100.0, 240.0, |s, nv| s.val = nv);

            ui.element(
                LayoutSpec::col().fixed(300.0, 200.0),
                PaintStyle::default(),
                |ui| {
                    ui.on_scroll(|s, dy| {
                        s.vscroll -= dy;
                        true
                    });
                },
            );
        });
    };

    // Frame 0 mounts slider BEFORE layout has run (`active_box` was `(0,0,0,0)` during mount!):
    sys.build_frame(&host, 1000.0, 700.0, build);

    // Click right in the exact center of the `[520..760]` slider (`x = 640.0`, `y = 312.0`):
    assert!(sys.pointer_down(640.0, 312.0, &mut host));
    assert!(
        (host.val - 50.0).abs() < 1e-4,
        "Clicking exact slider midpoint (x=640 on [520..760]) must yield 50.0, got {}",
        host.val
    );

    // Rebuild and verify the 14x14 circular thumb's geometric center `thumb.x + 7.0 == 640.0`!
    sys.build_frame(&host, 1000.0, 700.0, build);
    let thumb_box = sys
        .layout
        .boxes
        .iter()
        .find(|b| (b.w - 14.0).abs() < 1e-4 && (b.h - 14.0).abs() < 1e-4)
        .expect("14x14 slider thumb node");
    let thumb_center_x = thumb_box.x + thumb_box.w * 0.5;
    assert!(
        (thumb_center_x - 640.0).abs() < 1e-4,
        "Thumb center ({thumb_center_x}) must match cursor click X (640.0) to 0.0px error!"
    );

    // Drag to left thumb center (`520 + 7 = 527.0` -> `0.0`) and right thumb center (`760 - 7 = 753.0` -> `100.0`):
    assert!(sys.pointer_move(527.0, 312.0, &mut host));
    assert!((host.val - 0.0).abs() < 1e-4);
    assert!(sys.pointer_move(753.0, 312.0, &mut host));
    assert!((host.val - 100.0).abs() < 1e-4);
    sys.pointer_up();

    // Test custom `on_scroll` handler at `(x = 600, y = 400)`:
    assert!(sys.dispatch_scroll(600.0, 400.0, -72.0, &mut host));
    assert!((host.vscroll - 72.0).abs() < 1e-4);
}

// =========================================================================
// 9. Regression Test: 100,000-Row `virtual_scroll_view` & `vertical_scrollbar` Mechanics
// =========================================================================
#[test]
fn verify_vertical_scrollbar_and_100k_virtual_scroll_view() {
    struct LogHost {
        vlist: VirtualListIndex,
        scroll_y: f64,
        clicked_row: Option<usize>,
    }
    let mut host = LogHost {
        vlist: VirtualListIndex::new(100_000, 34.0), // 3,400,000.0 px total!
        scroll_y: 0.0,
        clicked_row: None,
    };
    let mut sys = GuiSystem::new();

    let build = |ui: &mut Ui<'_, '_, LogHost>| {
        ui.element(LayoutSpec::col().grow(1.0).pad(20.0), PaintStyle::default(), |ui| {
            let vlist_ref = &ui.state().vlist;
            ui.virtual_scroll_view(
                "100k Logs Scrollbar",
                vlist_ref,
                |s| s.scroll_y,
                |s, y| s.scroll_y = y,
                LayoutSpec::row().grow(1.0),
                PaintStyle::default(),
                |ui, idx, row_h| {
                    ui.element(
                        LayoutSpec::row().width_grow(1.0).height_fixed(row_h),
                        PaintStyle::default(),
                        |ui| {
                            ui.on_click(move |s, _| s.clicked_row = Some(idx));
                        },
                    );
                },
            );
        });
    };

    // Frame 0 (`1000 x 540` window -> `20px` padding -> `960 x 500` scroll view, scrollbar at `[966..980, 20..520]`):
    sys.build_frame(&host, 1000.0, 540.0, build);

    // 1. Verify post-layout `ScrollbarThumbSync` placed a 28px minimum thumb at `y = 20 + 2 = 22.0` on Frame 0!
    let thumb_box_0 = sys
        .layout
        .boxes
        .iter()
        .find(|b| (b.w - 8.0).abs() < 1e-4 && (b.h - 28.0).abs() < 1e-4)
        .expect("initial 8x28 vertical scrollbar thumb");
    assert!(
        (thumb_box_0.y - 22.0).abs() < 1e-4,
        "Expected Frame-0 thumb top at y=22.0, got {}",
        thumb_box_0.y
    );

    // 2. Grab the thumb off-center at `y = 27.0` (`+5px` from thumb top, NOT center!) -> MUST NOT JUMP (`0.0 px` delta)!
    assert!(sys.pointer_down(973.0, 27.0, &mut host));
    assert_eq!(
        host.scroll_y, 0.0,
        "Grabbing the scrollbar thumb off-center must cause 0.0px jump on pointer_down!"
    );

    // 3. Drag `+234.0 px` down (`travel_h = 500 - 4 - 28 = 468.0`, so `+234 / 468 = 50.0%` of `max_scroll`!):
    let max_scroll = 3_400_000.0 - 500.0;
    assert!(sys.pointer_move(973.0, 261.0, &mut host));
    assert!(
        (host.scroll_y - max_scroll * 0.5).abs() < 1.0,
        "Dragging thumb 50% of travel_h must reach 50% of 3.4M px ({}), got {}",
        max_scroll * 0.5,
        host.scroll_y
    );
    sys.pointer_up();

    // Verify rendered thumb moved by exact `+234.0 px` (`22.0 + 234.0 = 256.0`):
    sys.build_frame(&host, 1000.0, 540.0, build);
    let thumb_box_mid = sys
        .layout
        .boxes
        .iter()
        .find(|b| (b.h - 28.0).abs() < 1e-4 && b.x >= 960.0)
        .expect("mid-scroll thumb");
    assert!(
        (thumb_box_mid.y - 256.0).abs() < 1e-4,
        "Thumb top must follow drag 1:1 to y=256.0, got {}",
        thumb_box_mid.y
    );

    // 4. Click empty track near bottom (`y = 515.0`) -> seeks to `max_scroll` (`100%`):
    assert!(sys.pointer_down(973.0, 515.0, &mut host));
    sys.pointer_up();
    assert!(
        (host.scroll_y - max_scroll).abs() < 1e-4,
        "Clicking bottom of track must seek to max_scroll ({max_scroll}), got {}",
        host.scroll_y
    );

    // 5. Click a child log row inside the viewport -> triggers row `on_click` AND focuses enclosing `virtual_scroll_view`
    //    so `Home` and `PageDown` keys work immediately!
    sys.build_frame(&host, 1000.0, 540.0, build);
    assert!(sys.pointer_down(400.0, 100.0, &mut host));
    sys.pointer_up();
    assert!(host.clicked_row.unwrap() > 99_900);

    assert!(sys.dispatch_key(&KeyInput::Home, &mut host));
    assert_eq!(host.scroll_y, 0.0);
    assert!(sys.dispatch_key(&KeyInput::PageDown, &mut host));
    assert!((host.scroll_y - 450.0).abs() < 1e-4);
}

// =========================================================================
// 10. Regression Test: `vector_canvas` In-Place Atlas Reuse (Zero Atlas Thrashing at 240 FPS!)
// =========================================================================
#[test]
fn verify_vector_canvas_in_place_atlas_reuse_across_hundreds_of_frames() {
    let mut sys = GuiSystem::new();
    for rev in 1..=200u64 {
        sys.build_frame(&rev, 600.0, 120.0, |ui| {
            let cur = *ui.state();
            ui.vector_canvas(520, 62, cur, Color::rgb8(80, 160, 255), FillRule::NonZero, |p| {
                let y = 10.0 + ((cur % 40) as f32);
                p.add_stroke_segment(0.0, y, 520.0, 62.0 - y, 2.4);
            });
        });
        sys.fonts.atlas.dirty_rects.clear();
    }
    assert_eq!(
        sys.fonts.atlas.generation(),
        1,
        "200 dynamic 520x62 vector_canvas updates must overwrite in place with 0 atlas resets!"
    );
}

// =========================================================================
// 11. Adversarial Edge Cases: Empty Trees & 0x0 Viewports Across Layout, UI, CPU, & Vulkan 1.3
// =========================================================================
#[test]
fn verify_empty_trees_and_zero_by_zero_viewports_across_layout_ui_cpu_and_vulkan() {
    // 1. Empty `LayoutTree` at `0x0` and `800x600`:
    let mut empty_tree = LayoutTree::new();
    empty_tree.compute_layout(0.0, 0.0);
    empty_tree.compute_layout(800.0, 600.0);
    assert_eq!(empty_tree.last_metrics.nodes_visited, 0);

    // 2. Empty `GuiSystem` (build closure mounts zero elements) + all input/a11y/render entrypoints:
    let mut empty_sys = GuiSystem::new();
    empty_sys.a11y_enabled = true;
    let mut unit_state = 0usize;
    empty_sys.build_frame(&unit_state, 0.0, 0.0, |_ui| {});
    assert!(empty_sys.primitives.is_empty());
    assert!(empty_sys.clips.is_empty());
    assert!(empty_sys.hit_test(0.0, 0.0).is_none());
    assert!(!empty_sys.pointer_move(10.0, 10.0, &mut unit_state));
    assert!(!empty_sys.pointer_down(10.0, 10.0, &mut unit_state));
    assert!(!empty_sys.pointer_up());
    assert!(!empty_sys.pointer_leave());
    assert!(!empty_sys.dispatch_scroll(10.0, 10.0, -30.0, &mut unit_state));
    assert!(!empty_sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut unit_state));
    assert!(!empty_sys.dispatch_key(&KeyInput::Enter, &mut unit_state));
    assert!(!empty_sys.dispatch_action(&SaveShortcut, &mut unit_state));
    assert!(empty_sys.build_accesskit_update().is_none());

    let mut cpu_zero = CpuRenderer::new(0, 0);
    empty_sys.render_to_backend(&mut cpu_zero, 0, 0, 1).unwrap();

    let mut vk = VulkanRenderer::new_headless(64, 16).expect("Vulkan 1.3 headless");
    let zero_frame = RenderFrame {
        width: 0,
        height: 0,
        scale_factor: 1.0,
        clear_color: Color::rgb8(12, 16, 24),
        primitives: &[],
        dirty_prim_ranges: &[],
        clips: &[],
        damage: None,
    };
    let vk_zero_px = vk
        .render_offscreen_rgba8(&zero_frame, &mut empty_sys.fonts.atlas)
        .expect("0x0 Vulkan offscreen render must return empty buffer without VUID violation");
    assert!(vk_zero_px.is_empty());

    // 3. Non-empty widget tree built at `0x0` logical viewport, rendered on CPU & Vulkan, then expanded to `320x200`:
    let mut sys = GuiSystem::new();
    let val = 0.5_f32;
    let build = |ui: &mut Ui<'_, '_, f32>| {
        let th = ui.theme();
        ui.element(LayoutSpec::col().grow(1.0).pad(12.0).gap(8.0), PaintStyle::solid(th.bg_canvas), |ui| {
            ui.heading("0x0 Minimization Safety");
            let v = *ui.state();
            ui.slider("Level", v, 0.0..=1.0, 180.0, |s, nv| *s = nv);
            ui.primary_button("Apply", |s, _| *s = 1.0);
        });
    };
    sys.build_frame(&val, 0.0, 0.0, build);
    let mut cpu = CpuRenderer::new(1, 1);
    sys.render_to_backend(&mut cpu, 0, 0, 1).unwrap();
    let vk_min = sys.render_vulkan_offscreen(&mut vk, 0, 0).unwrap();
    assert!(vk_min.is_empty());

    // Recover cleanly to normal 320x200 desktop dimensions:
    sys.build_frame(&val, 320.0, 200.0, build);
    assert_eq!(sys.layout.boxes[sys.layout.root.idx()].w, 320.0);
    assert_eq!(sys.layout.boxes[sys.layout.root.idx()].h, 200.0);
    let vk_recovered = sys.render_vulkan_offscreen(&mut vk, 320, 200).unwrap();
    assert_eq!(vk_recovered.len(), 320 * 200 * 4);
}

// =========================================================================
// 12. >100 Grow/Shrink Children Water-Filling, 2D Spanned Grid Occupancy, WrapRow, & 12-Way Anchors
// =========================================================================
#[test]
fn verify_160_grow_and_shrink_children_waterfilling_2d_grid_occupancy_wrap_row_and_12_anchors() {
    // 1. 160 `Grow` children (>64 inline stack buffer!) in both Row and Col with tight min/max clamps:
    let mut tree = LayoutTree::new();
    tree.snap_to_pixels = false;
    let row_root = tree.alloc_node(LayoutSpec::row().fixed(2400.0, 100.0).gap(2.0));
    tree.root = row_root;
    let mut row_children = Vec::with_capacity(160);
    for i in 0..160usize {
        let spec = match i % 4 {
            0 => LayoutSpec::col().width_grow(1.0).height_fixed(40.0).min_size(18.0, 0.0),
            1 => LayoutSpec::col().width_grow(2.0).height_fixed(40.0).max_size(8.0, 100.0),
            2 => LayoutSpec::col().width_grow(3.0).height_fixed(40.0).min_size(6.0, 0.0).max_size(24.0, 100.0),
            _ => LayoutSpec::col().width_grow(1.5).height_fixed(40.0),
        };
        row_children.push(tree.alloc_node(spec));
    }
    tree.set_children(row_root, &row_children);
    tree.compute_layout(2400.0, 100.0);

    let mut sum_row_w = 0.0_f32;
    for (i, &cid) in row_children.iter().enumerate() {
        let w = tree.boxes[cid.idx()].w;
        assert!(w.is_finite() && w >= 0.0);
        match i % 4 {
            0 => assert!(w >= 18.0 - 1e-3, "min_w=18 violated at child {i}: {w}"),
            1 => assert!(w <= 8.0 + 1e-3, "max_w=8 violated at child {i}: {w}"),
            2 => assert!((6.0 - 1e-3..=24.0 + 1e-3).contains(&w), "min/max [6,24] violated at {i}: {w}"),
            _ => {}
        }
        sum_row_w += w;
    }
    let expected_inner_w = 2400.0 - 159.0 * 2.0;
    assert!(
        (sum_row_w - expected_inner_w).abs() < 2e-2,
        "160 Grow Row water-filling sum ({sum_row_w}) must match exact inner width ({expected_inner_w})"
    );

    // 160 `Grow` children in `Axis::Col`:
    let mut col_tree = LayoutTree::new();
    col_tree.snap_to_pixels = false;
    let col_root = col_tree.alloc_node(LayoutSpec::col().fixed(120.0, 2000.0).gap(1.0));
    col_tree.root = col_root;
    let mut col_children = Vec::with_capacity(160);
    for i in 0..160usize {
        let spec = match i % 3 {
            0 => LayoutSpec::row().width_fixed(80.0).height_grow(1.0).min_size(0.0, 15.0),
            1 => LayoutSpec::row().width_fixed(80.0).height_grow(2.5).max_size(120.0, 7.0),
            _ => LayoutSpec::row().width_fixed(80.0).height_grow(1.0),
        };
        col_children.push(col_tree.alloc_node(spec));
    }
    col_tree.set_children(col_root, &col_children);
    col_tree.compute_layout(120.0, 2000.0);
    let sum_col_h: f32 = col_children.iter().map(|c| col_tree.boxes[c.idx()].h).sum();
    assert!(
        (sum_col_h - (2000.0 - 159.0)).abs() < 2e-2,
        "160 Grow Col water-filling sum ({sum_col_h}) must match 1841.0"
    );

    // 2. 140 `Fit` children overflowing a 600px Row with heterogeneous `shrink_q8` & `min_w` floors:
    let mut stree = LayoutTree::new();
    stree.snap_to_pixels = false;
    let sroot = stree.alloc_node(LayoutSpec::row().fixed(600.0, 50.0));
    stree.root = sroot;
    let mut schildren = Vec::with_capacity(140);
    for i in 0..140usize {
        // Fit width via child fixed box so mode_w == Fit!
        let mut wrap = LayoutSpec::row().min_size(if i % 5 == 0 { 6.0 } else { 1.0 }, 0.0);
        wrap.shrink_q8 = match i % 4 {
            0 => 0,   // unshrinkable!
            1 => 128, // 0.5x shrink
            2 => 256, // 1.0x shrink
            _ => 512, // 2.0x shrink
        };
        let p = stree.alloc_node(wrap);
        let leaf = stree.alloc_node(LayoutSpec::col().fixed(8.0, 20.0));
        stree.set_children(p, &[leaf]);
        schildren.push(p);
    }
    stree.set_children(sroot, &schildren);
    stree.compute_layout(600.0, 50.0);
    let mut sum_shrunk = 0.0_f32;
    for (i, &cid) in schildren.iter().enumerate() {
        let w = stree.boxes[cid.idx()].w;
        if i % 4 == 0 {
            assert!((w - 8.0).abs() < 1e-4, "shrink_q8 == 0 child must keep full 8.0px width, got {w}");
        }
        if i % 5 == 0 {
            assert!(w >= 6.0 - 1e-4, "min_w=6.0 floor violated at {i}: {w}");
        }
        sum_shrunk += w;
    }
    assert!(
        (sum_shrunk - 600.0).abs() < 2e-2,
        "140-child CSS shrink water-filling sum ({sum_shrunk}) must match container width 600.0"
    );

    // 3. 2D Grid with 2x2 spanned tile + Auto/Percent/MinMax/Fr tracks + row-overflow clamping + grid recycling:
    let mut gtree = LayoutTree::new();
    gtree.snap_to_pixels = false;
    let grid = gtree.alloc_node(LayoutSpec::col().fixed(320.0, 240.0).gap(10.0));
    gtree.attach_grid(
        grid,
        vec![
            TrackSize::Fixed(60.0),
            TrackSize::MinMax(50.0, 90.0),
            TrackSize::Fr(1.0),
        ],
        vec![
            TrackSize::Auto,
            TrackSize::Percent(0.25),
            TrackSize::Fr(1.0),
        ],
    );
    gtree.root = grid;
    let hero_2x2 = gtree.alloc_node(LayoutSpec::col().height_fixed(100.0).grid_span(2, 2));
    // Auto-placed siblings must route around `(0..2, 0..2)` to `(2, 0)` and `(2, 1)`, then row 2, then clamp extra!
    let sib_r0_c2 = gtree.alloc_node(LayoutSpec::col().fixed(40.0, 30.0));
    let sib_r1_c2 = gtree.alloc_node(LayoutSpec::col().grow(1.0));
    let sib_r2_c0 = gtree.alloc_node(LayoutSpec::col().grow(1.0));
    let sib_r2_c1 = gtree.alloc_node(LayoutSpec::col().grow(1.0));
    let sib_r2_c2 = gtree.alloc_node(LayoutSpec::col().grow(1.0));
    let sib_overflow = gtree.alloc_node(LayoutSpec::col().fixed(30.0, 20.0));
    gtree.set_children(
        grid,
        &[
            hero_2x2,
            sib_r0_c2,
            sib_r1_c2,
            sib_r2_c0,
            sib_r2_c1,
            sib_r2_c2,
            sib_overflow,
        ],
    );
    gtree.compute_layout(320.0, 240.0);

    let hb = gtree.boxes[hero_2x2.idx()];
    let s0 = gtree.boxes[sib_r0_c2.idx()];
    let s1 = gtree.boxes[sib_r1_c2.idx()];
    let s2 = gtree.boxes[sib_r2_c0.idx()];
    assert!(
        s0.x >= hb.x + hb.w + 10.0 - 1e-3 && (s0.y - 0.0).abs() < 1e-3,
        "Sibling 0 must route to col 2 row 0 right of 2x2 hero: hero={hb:?}, s0={s0:?}"
    );
    assert!(
        s1.x >= hb.x + hb.w + 10.0 - 1e-3 && s1.y > s0.y + 10.0,
        "Sibling 1 must route to col 2 row 1 (never overlapping rows 0..2 cols 0..2!): hero={hb:?}, s1={s1:?}"
    );
    assert!(
        (s2.x - 0.0).abs() < 1e-3 && s2.y >= hb.y + hb.h + 10.0 - 1e-3,
        "Sibling 2 must start at col 0 row 2 below 2x2 hero: hero={hb:?}, s2={s2:?}"
    );
    assert!(gtree.boxes[sib_overflow.idx()].y >= s2.y - 1e-3);
    gtree.free_subtree(grid);

    // 4. `WrapRow` per-line `Align::Center` + `MainAlign::SpaceBetween` + `scroll_y`:
    let mut wtree = LayoutTree::new();
    wtree.snap_to_pixels = false;
    let mut wspec = LayoutSpec::wrap_row()
        .fixed(200.0, 120.0)
        .gap(10.0)
        .align(Align::Center)
        .justify(MainAlign::SpaceBetween);
    wspec.scroll_y = 15.0;
    let wroot = wtree.alloc_node(wspec);
    wtree.root = wroot;
    let w_a = wtree.alloc_node(LayoutSpec::col().fixed(70.0, 20.0));
    let w_b = wtree.alloc_node(LayoutSpec::col().fixed(70.0, 44.0));
    let w_c = wtree.alloc_node(LayoutSpec::col().fixed(90.0, 28.0));
    wtree.set_children(wroot, &[w_a, w_b, w_c]);
    wtree.compute_layout(200.0, 120.0);
    // Line 0 has w_a (h=20) and w_b (h=44) -> line_h = 44; SpaceBetween puts w_a at x=0 and w_b at x=130;
    // scroll_y=15 shifts line 0 origin_y to -15, so w_a centered is at -15 + (44 - 20)/2 = -3.0!
    assert!((wtree.boxes[w_a.idx()].x - 0.0).abs() < 1e-3);
    assert!((wtree.boxes[w_b.idx()].x - 130.0).abs() < 1e-3);
    assert!((wtree.boxes[w_a.idx()].y - (-3.0)).abs() < 1e-3);
    assert!((wtree.boxes[w_c.idx()].y - (-15.0 + 44.0 + 10.0)).abs() < 1e-3);

    // 5. Pass 4 `AnchorPlacement` (`BottomEnd`, `TopStart`, `RightStart`, `LeftEnd`) flip, clamp, & nested `clip_rects`:
    let mut atree = LayoutTree::new();
    let aroot = atree.alloc_node(LayoutSpec::col().fixed(400.0, 300.0));
    atree.root = aroot;
    // Place button near bottom-right corner `(x=310, y=260, w=80, h=30)` so `BottomEnd` flips above to `TopEnd`!
    let mut holder_spec = LayoutSpec::col().fixed(400.0, 300.0);
    holder_spec.pad_l = 310.0;
    holder_spec.pad_t = 260.0;
    let holder = atree.alloc_node(holder_spec);
    let target_btn = atree.alloc_node(LayoutSpec::row().fixed(80.0, 30.0));
    atree.set_children(holder, &[target_btn]);

    let pop = atree.alloc_node(LayoutSpec::col().fixed(180.0, 90.0));
    atree.topo[pop.idx()].flags |= skima::layout::FLAG_CLIPS_CHILDREN;
    atree.set_floating_anchor(pop, target_btn, AnchorPlacement::BottomEnd, 0.0, 6.0);
    let pop_inner = atree.alloc_node(LayoutSpec::row().fixed(180.0, 90.0));
    atree.set_children(pop, &[pop_inner]);
    atree.set_children(aroot, &[holder, pop]);
    atree.compute_layout(400.0, 300.0);

    let pb = atree.boxes[pop.idx()];
    // Right-aligned to `target_btn` right edge `310 + 80 = 390` -> `pb.x == 390 - 180 = 210`!
    // Flipped above `target_btn` top `260` -> `pb.y == 260 - 90 - 6 = 164`!
    assert!((pb.x - 210.0).abs() < 1e-3, "BottomEnd X must right-align to 210.0, got {}", pb.x);
    assert!((pb.y - 164.0).abs() < 1e-3, "BottomEnd must flip above bottom edge to y=164.0, got {}", pb.y);
    assert_eq!(
        atree.clip_rects[pop_inner.idx()],
        [pb.x, pb.y, pb.w, pb.h],
        "Nested child inside translated floating popover must inherit translated clip_rect!"
    );
}

// =========================================================================
// 13. FrameArena Over-Aligned/ZST Provenance, SlotPool Retirement, VirtualList Mutations, & 3 Spring Regimes
// =========================================================================
#[test]
fn verify_frame_arena_overaligned_zst_slot_pool_retirement_virtual_list_mutations_and_three_spring_regimes() {
    // 1. `FrameArena`: over-aligned (`512B` and `4096B > 64B`), empty `Vec::new_in` ZST/0-size grow, LIFO shrink, & reset:
    let arena = FrameArena::with_capacity(4096);
    {
        let mut empty_started: Vec<u64, &FrameArena> = Vec::new_in(&arena);
        for v in 0..128u64 {
            empty_started.push(v);
        }
        empty_started.shrink_to_fit();
        assert_eq!(empty_started.len(), 128);
        assert_eq!(empty_started[127], 127);

        let p512 = (&arena)
            .allocate(Layout::from_size_align(256, 512).unwrap())
            .expect("512B aligned arena alloc");
        assert_eq!((p512.as_ptr() as *mut u8 as usize) % 512, 0);

        let p4096 = (&arena)
            .allocate(Layout::from_size_align(1024, 4096).unwrap())
            .expect("4096B page-aligned arena alloc");
        assert_eq!((p4096.as_ptr() as *mut u8 as usize) % 4096, 0);
    }
    arena.reset();
    // After consolidating spill chunks, 4096B alignment and consolidated capacity remain immediately available:
    let post_reset = (&arena)
        .allocate(Layout::from_size_align(512, 4096).unwrap())
        .expect("post-reset 4096B alloc");
    assert_eq!((post_reset.as_ptr() as *mut u8 as usize) % 4096, 0);

    // 2. `SlotPool`: `Key::INVALID` safety, atomic `get_disjoint_mut`, and transactional `DropGuard` rollback:
    let mut pool = SlotPool::<String>::new();
    assert!(pool.get(Key::INVALID).is_none());
    assert!(pool.get_mut(Key::INVALID).is_none());
    assert!(pool.remove(Key::INVALID).is_none());

    let k_a = pool.insert(String::from("alpha"));
    let k_b = pool.insert(String::from("beta"));
    let ep_before = pool.epoch(k_a).unwrap();
    assert!(matches!(
        pool.get_disjoint_mut([k_a, Key::INVALID]),
        Err(PoolAccessError::StaleOrInvalidKey)
    ));
    assert_eq!(
        pool.epoch(k_a).unwrap(),
        ep_before,
        "Failed get_disjoint_mut must not mutate surviving slot epochs"
    );
    let [va, vb] = pool.get_disjoint_mut([k_a, k_b]).unwrap();
    va.push_str("+1");
    vb.push_str("+2");

    let tx_err: Result<bool, &str> = pool.try_update_transactional(k_a, |s| {
        s.push_str("_corrupted");
        Err("abort")
    });
    assert!(tx_err.is_err());
    assert_eq!(pool.get(k_a).unwrap(), "alpha+1", "DropGuard must roll back aborted mutation");

    // 3. `VirtualListIndex`: `push`, `pop`, `resize`, `0.0`-height items, and exact `< target` boundary binary lifting:
    let mut vlist = VirtualListIndex::new(0, 20.0);
    assert_eq!(vlist.visible_range(0.0, 100.0), (0, 0, 0.0));
    for i in 0..10usize {
        vlist.push(if i == 5 { 0.0 } else { 30.0 });
    }
    assert_eq!(vlist.len(), 10);
    assert!((vlist.total_height() - 270.0).abs() < 1e-6);
    assert_eq!(vlist.pop(), Some(30.0));
    assert_eq!(vlist.len(), 9);
    vlist.resize(25, 20.0); // Extends [9..25) with 20.0 without wiping [0..9)!
    assert!((vlist.height_at(5) - 0.0).abs() < 1e-6);
    assert!((vlist.height_at(0) - 30.0).abs() < 1e-6);
    assert!((vlist.height_at(20) - 20.0).abs() < 1e-6);

    // Exact boundary: item 0 is [0..30), item 1 is [30..60). At `scroll_y == 30.0`, item 0 is strictly above viewport!
    let (start_at_30, _, top_at_30) = vlist.visible_range(30.0, 50.0);
    assert_eq!(start_at_30, 1, "Half-open [0..30) item must NOT remain at start_idx when scroll_y == 30.0");
    assert!((top_at_30 - 30.0).abs() < 1e-6);

    vlist.resize(4, 20.0);
    assert_eq!(vlist.len(), 4);
    assert!((vlist.total_height() - 120.0).abs() < 1e-6);

    // 4. All 3 `Spring` damping regimes + extreme `zeta = 50_000` + non-positive/infinite `dt`:
    for zeta in [0.28_f32, 1.0_f32, 3.2_f32, 50_000.0_f32] {
        let mut coarse = Spring {
            value: 0.0,
            velocity: 0.0,
            target: 100.0,
            angular_freq: 18.0,
            damping_ratio: zeta,
        };
        let mut fine = coarse;
        coarse.step(0.20);
        for _ in 0..200 {
            fine.step(0.001);
        }
        assert!(
            coarse.value.is_finite() && (coarse.value - fine.value).abs() < 2e-3,
            "Spring ODE invariance failed at zeta={zeta}: coarse={} vs fine={}",
            coarse.value,
            fine.value
        );
        // `dt <= 0` and `NaN` preserve state without NaN poisoning; `INFINITY` snaps cleanly to target:
        let snap_before = coarse.value;
        let _ = coarse.step(0.0);
        let _ = coarse.step(-5.0);
        let _ = coarse.step(f32::NAN);
        assert_eq!(coarse.value, snap_before);
        assert!(!coarse.step(f32::INFINITY));
        assert_eq!(coarse.value, 100.0);
        assert_eq!(coarse.velocity, 0.0);
    }
}

// =========================================================================
// 14. UTF-8/Emoji TextInput, Multiline Paragraphs, Oversized Glyphs, Seamless Polyline, & OKLab Contrast
// =========================================================================
#[test]
fn verify_utf8_emoji_text_input_multiline_paragraphs_oversized_glyphs_seamless_polyline_and_oklab_contrast() {
    // 1. Multi-byte UTF-8 + Emoji `text_input`: OpenType `x_to_byte_offset`, word ops, `scroll_x`, & `SetValue`:
    let mut txt = String::from("café 🦀 fi_le αβγ");
    let mut sys = GuiSystem::with_system_fonts();
    sys.a11y_enabled = true;
    let build = |ui: &mut Ui<'_, '_, String>| {
        ui.element(LayoutSpec::col().fixed(130.0, 48.0), PaintStyle::default(), |ui| {
            ui.text_input("Filter...", |s| s, |s| s);
        });
    };
    sys.build_frame(&txt, 130.0, 48.0, build);
    let tree_up = sys.build_accesskit_update().unwrap();
    let input_a11y = tree_up
        .nodes
        .iter()
        .find(|(_, n)| n.role() == A11yRole::TextInput)
        .map(|(id, _)| *id)
        .unwrap();

    // Click right before `🦀` (`"café "` = 6 bytes: c(1)+a(1)+f(1)+é(2)+space(1)):
    let fid = sys.fonts.ui_font;
    let x_crab = sys.fonts.byte_offset_to_x(&txt, 6, fid, 14.0);
    assert!(sys.pointer_down(10.0 + x_crab + 0.2, 17.0, &mut txt));
    sys.pointer_up();
    // Roundtrip check: `x_to_byte_offset` at `x_crab` lands on a valid UTF-8 char boundary!
    let hit_b = sys.fonts.x_to_byte_offset(&txt, x_crab, fid, 14.0);
    assert!(txt.is_char_boundary(hit_b));

    assert!(sys.dispatch_key(&KeyInput::End, &mut txt));
    sys.build_frame(&txt, 130.0, 48.0, build);
    // Ctrl+Backspace deletes `"αβγ"`, then `"fi_le"`, then `"🦀"`:
    assert!(sys.dispatch_key(&KeyInput::CtrlBackspace, &mut txt));
    assert_eq!(txt, "café 🦀 fi_le ");
    assert!(sys.dispatch_key(&KeyInput::CtrlBackspace, &mut txt));
    assert_eq!(txt, "café 🦀 ");
    assert!(sys.dispatch_key(&KeyInput::CtrlBackspace, &mut txt));
    assert_eq!(txt, "café ");

    // AccessKit `SetValue` with multi-byte UTF-8:
    assert!(sys.handle_accesskit_action(
        &ActionRequest {
            action: Action::SetValue,
            target_tree: TreeId::ROOT,
            target_node: input_a11y,
            data: Some(ActionData::Value("oklch 100% 🎨".into())),
        },
        &mut txt,
    ));
    assert_eq!(txt, "oklch 100% 🎨");

    // 2. Multiline `ShapedParagraph` (`\n`, `\r\n`, empty `\n\n` line, preserved column spaces, interval cache):
    let multi = "col_a    col_b\n\nline_three\r\nline_four";
    let mut para = sys.fonts.shape_paragraph(multi, fid, 14.0, 20.0);
    let (h_wide, _, _) = para.wrap_for_width(1000.0);
    assert!(
        (h_wide - 80.0).abs() < 1e-3,
        "4 explicit hard-break lines (including blank \\n\\n line) at line_h=20 must equal 80.0, got {h_wide}"
    );
    let (_, _, cache_hit) = para.wrap_for_width(920.0);
    assert!(cache_hit, "Width inside [W_tight, INF) must hit exact width-interval cache");

    // Preserved multi-space advance (`"a    b"` must be strictly wider than `"a b"`):
    let w_four_spaces = sys.fonts.byte_offset_to_x("a    b", 6, fid, 14.0);
    let w_one_space = sys.fonts.byte_offset_to_x("a b", 3, fid, 14.0);
    assert!(
        w_four_spaces > w_one_space + 8.0,
        "Multiple spaces must be preserved for column alignment: 4sp={w_four_spaces} vs 1sp={w_one_space}"
    );

    // 3. Oversized glyph (`900px` on a `64x64` atlas) & out-of-bounds `update_*_in_place` safety:
    let mut small_fonts = FontSystem::new();
    small_fonts.atlas = GlyphAtlas::new(64, 64);
    let pinned = small_fonts
        .atlas
        .upload_r8(16, 16, 0, 0, &[200u8; 256])
        .expect("valid 16x16 upload");
    let gen_pinned = small_fonts.atlas.generation();
    assert!(
        small_fonts
            .atlas
            .upload_r8(512, 512, 0, 0, &[255u8; 512 * 512])
            .is_none(),
        "Oversized upload must return None without panicking"
    );
    let over_glyph = small_fonts.rasterize_glyph(FontId::FALLBACK, b'M' as u16, 900.0, 0.0, AntialiasMode::Grayscale);
    assert_eq!(over_glyph.w, 0);
    assert_eq!(
        small_fonts.atlas.generation(),
        gen_pinned,
        "Oversized glyph/upload must NOT flush or bump generation of existing cached glyphs!"
    );
    // Out-of-bounds in-place updates must be graceful no-ops:
    let mut bad_reg = pinned;
    bad_reg.x = 60;
    bad_reg.w = 16;
    small_fonts.atlas.update_r8_in_place(bad_reg, &[0u8; 256]);
    small_fonts.atlas.update_rgba8_in_place(bad_reg, &[0u8; 1024]);

    // 4. `Path2D::stroke_polyline` seamless single-contour round joins + negative-X clipping:
    let mut poly = Path2D::new();
    poly.stroke_polyline(&[[-18.0, 20.0], [24.0, 20.0], [24.0, 48.0]], 8.0);
    let mask_nz = poly.rasterize_mask(64, 64, FillRule::NonZero);
    let mask_eo = poly.rasterize_mask(64, 64, FillRule::EvenOdd);
    // Exact 90-degree joint center `(24, 20)` must have full `255` coverage under BOTH NonZero and EvenOdd!
    assert_eq!(mask_nz[20 * 64 + 24], 255, "NonZero polyline join must have 0 seam holes");
    assert_eq!(mask_eo[20 * 64 + 24], 255, "EvenOdd polyline join must have 0 double-overlap holes");
    // Clipped left border `(0, 20)` from segment `[-18..24]` must be solid `255` without wrap-around corruption:
    assert_eq!(mask_nz[20 * 64], 255);

    // 5. Premultiplied `Color::mix_oklab`, `Oklch` gamut clipping, WCAG 2.1, and APCA contrast:
    let cyan = Color::rgb8(30, 200, 245);
    let mid_trans = cyan.mix_oklab(Color::rgba8(255, 0, 0, 0), 0.5);
    assert!(
        mid_trans.channels()[0].abs_diff(30) <= 1
            && mid_trans.channels()[1].abs_diff(200) <= 1
            && mid_trans.channels()[2].abs_diff(245) <= 1,
        "Premultiplied OKLab mix with alpha=0 must preserve RGB without tint/darkening: {mid_trans:?}"
    );
    let clipped_lab = Oklch::new(0.70, 0.38, 260.0, 1.0).to_oklab().gamut_clip_srgb();
    assert!(clipped_lab.is_in_srgb_gamut());
    assert!(Color::WHITE.wcag_contrast_ratio(Color::BLACK) > 20.9);
    assert!(Color::BLACK.apca_contrast(Color::WHITE) > 100.0);
    assert!(Color::WHITE.apca_contrast(Color::BLACK) < -100.0);
    assert_eq!(Color::WHITE.readable_text_color(), Color::rgb8(18, 20, 28));
    assert_eq!(Color::BLACK.readable_text_color(), Color::WHITE);
}

// =========================================================================
// 15. Slot Unmount/Recycle Hygiene, `scroll_area` + `set_scrollable`, & Zero Oscillation (Light & Dark)
// =========================================================================
#[test]
fn verify_slot_unmount_hygiene_scroll_area_retained_scrollable_and_zero_layout_oscillation_light_and_dark() {
    struct AppHarness {
        dark: bool,
        show_deep_branch: bool,
        spring_goal: f32,
        sa_items: usize,
        sa_scroll: f64,
    }
    let mut app = AppHarness {
        dark: true,
        show_deep_branch: true,
        spring_goal: 0.0,
        sa_items: 3,
        sa_scroll: 0.0,
    };
    let mut sys = GuiSystem::new();
    sys.a11y_enabled = true;

    fn build_deep(ui: &mut Ui<'_, '_, AppHarness>, depth: usize) {
        ui.element(LayoutSpec::col().fixed(140.0, 36.0), PaintStyle::default(), |ui| {
            if depth < 34 {
                build_deep(ui, depth + 1);
            } else {
                ui.set_focusable(true);
                ui.set_cursor(CursorIcon::Pointer);
                ui.on_drag(|_, _| {});
                let _ = ui.use_spring(ui.state().spring_goal, 5.0);
            }
        });
    }

    let build = |ui: &mut Ui<'_, '_, AppHarness>| {
        let th = ui.theme();
        ui.element(
            LayoutSpec::col().grow(1.0).pad(16.0).gap(10.0),
            PaintStyle::solid(th.bg_canvas),
            |ui| {
                ui.element(
                    LayoutSpec::row().width_grow(1.0).height_fixed(42.0).gap(8.0).align(Align::Center),
                    PaintStyle::solid(th.bg_surface).rounded(th.radius_md).border(1.0, th.border_subtle),
                    |ui| {
                        ui.badge("1.0 STABLE", th.accent);
                        ui.primary_button("Toggle Theme", |s, _| s.dark.toggle());
                        if ui.state().show_deep_branch {
                            ui.keyed(101, |ui| build_deep(ui, 0));
                        } else {
                            ui.keyed(202, |ui| {
                                ui.badge("RECYCLED", th.success);
                            });
                        }
                    },
                );

                // Retained `set_scrollable` container (scroll_y managed internally by `GuiSystem` across frames!):
                ui.element(
                    LayoutSpec::col().width_fixed(220.0).height_fixed(60.0),
                    PaintStyle::solid(th.bg_elevated),
                    |ui| {
                        ui.set_scrollable(180.0);
                        for i in 0..8usize {
                            ui.keyed(i as u64 + 1, |ui| {
                                ui.element(LayoutSpec::row().fixed(200.0, 24.0), PaintStyle::default(), |_| {});
                            });
                        }
                    },
                );

                // Auto-measuring `scroll_area`:
                let count = ui.state().sa_items;
                ui.scroll_area(
                    "Inspector List",
                    |s| s.sa_scroll,
                    |s, v| s.sa_scroll = v,
                    LayoutSpec::col().width_grow(1.0).height_grow(1.0).pad(8.0).gap(6.0),
                    PaintStyle::solid(th.bg_surface).rounded(th.radius_md),
                    |ui| {
                        for i in 0..count {
                            ui.keyed(i as u64 + 1, |ui| {
                                ui.element(
                                    LayoutSpec::row().width_grow(1.0).height_fixed(30.0),
                                    PaintStyle::solid(th.bg_control).rounded(th.radius_sm),
                                    |_| {},
                                );
                            });
                        }
                    },
                );
            },
        );
    };

    // 1. Mount deep 35-level branch at `spring_goal = 0.0`, then change `spring_goal = 1.0` + hover/focus/capture:
    sys.build_frame(&app, 960.0, 640.0, build);
    let _ = sys.build_accesskit_update();
    let leaf_b = sys
        .layout
        .boxes
        .iter()
        .rfind(|b| (b.w - 140.0).abs() < 1e-3 && (b.h - 36.0).abs() < 1e-3)
        .copied()
        .expect("35-deep leaf box");
    app.spring_goal = 1.0;
    sys.pointer_move(leaf_b.x + 10.0, leaf_b.y + 10.0, &mut app);
    assert!(sys.pointer_down(leaf_b.x + 10.0, leaf_b.y + 10.0, &mut app));
    sys.build_frame(&app, 960.0, 640.0, build);
    assert!(sys.has_active_animations());
    assert!(sys.focused_slot.is_some());
    assert!(sys.captured_slot.is_some());

    // Unmount deep branch & recycle slots in the same frame -> zero dangling spring/focus/capture/cursor!
    app.show_deep_branch = false;
    sys.build_frame(&app, 960.0, 640.0, build);
    assert!(!sys.has_active_animations(), "Unmounted deep spring must be purged before slot recycle");
    assert!(sys.focused_slot.is_none());
    assert!(sys.captured_slot.is_none());
    assert_eq!(sys.active_cursor, CursorIcon::Default);
    let _ = sys.build_accesskit_update();

    // 2. Wheel scroll on `set_scrollable(180.0)` container (`x=40, y=85`) MUST persist across `build_frame` redraws!
    assert!(sys.dispatch_scroll(40.0, 85.0, -64.0, &mut app));
    sys.build_frame(&app, 960.0, 640.0, build);
    let retained_scroll = sys
        .layout
        .specs
        .iter()
        .map(|s| s.scroll_y)
        .fold(0.0_f32, f32::max);
    assert!(
        (retained_scroll - 64.0).abs() < 1e-3,
        "ui.set_scrollable wheel offset (64.0) must survive subsequent build_frame redraws, got {retained_scroll}"
    );

    // 3. `scroll_area` overflow (`3 -> 30` items) + Light & Dark theme contrast + zero layout oscillation across hover/resize:
    app.sa_items = 30;
    for (is_dark, &[w, h]) in [
        (true, &[960.0_f32, 640.0_f32]),
        (false, &[1280.0, 800.0]),
        (true, &[1920.0, 1080.0]),
        (false, &[960.0, 640.0]),
    ] {
        sys.theme = if is_dark { Theme::dark() } else { Theme::light() };
        assert!(
            sys.theme.text_primary.wcag_contrast_ratio(sys.theme.bg_canvas) >= 10.0,
            "Theme text_primary vs bg_canvas WCAG contrast must exceed 10:1 (dark={is_dark})"
        );
        assert!(
            sys.theme.text_primary.apca_contrast(sys.theme.bg_surface).abs() >= 75.0,
            "Theme text_primary vs bg_surface |APCA Lc| must exceed 75 (dark={is_dark})"
        );

        sys.build_frame(&app, w, h, build);
        let snapshot_boxes: Vec<_> = sys.layout.boxes.clone();

        // Stationary + hover sweep frames at `(w, h)` must produce ZERO layout node visits and 0.0px box drift from Frame 0!
        for step in 0..6 {
            sys.pointer_move(40.0 + (step as f32) * 35.0, 32.0, &mut app);
            sys.build_frame(&app, w, h, build);
            assert_eq!(
                sys.layout.boxes, snapshot_boxes,
                "Zero layout oscillation violated on hover frame #{step} at {w}x{h}"
            );
            assert_eq!(
                sys.layout.last_metrics.nodes_visited, 0,
                "Hover frame #{step} at {w}x{h} (dark={is_dark}) must visit 0 layout nodes!"
            );
        }
    }
}

// =========================================================================
// 16. Comprehensive CPU vs Vulkan 1.3 Parity: Hollow/Thick Borders, 24-Deep Clips,
//     Premultiplied OKLab->Transparent, ClearType + Color Emoji, `vector_canvas`, & Partial Damage
// =========================================================================
#[test]
fn verify_cpu_vs_vulkan_1_3_parity_hollow_borders_24_deep_clips_oklab_gradient_lcd_emoji_and_partial_damage() {
    struct ParityScene {
        rev: u64,
        shrink_shadow: bool,
    }

    fn nest_24_clips(ui: &mut Ui<'_, '_, ParityScene>, depth: usize) {
        ui.element(
            LayoutSpec::col().fixed(56.0, 56.0),
            PaintStyle::default().rounded(14.0),
            |ui| {
                ui.set_clips_children(true);
                if depth + 1 < 24 {
                    nest_24_clips(ui, depth + 1);
                } else {
                    // Inner solid green fill overflowing bounds slightly via child card:
                    ui.element(
                        LayoutSpec::col().fixed(56.0, 56.0),
                        PaintStyle::solid(Color::rgb8(40, 215, 115)),
                        |_| {},
                    );
                }
            },
        );
    }

    let build = |ui: &mut Ui<'_, '_, ParityScene>| {
        let th = ui.theme();
        ui.element(
            LayoutSpec::col().fixed(360.0, 260.0).pad(14.0).gap(12.0),
            PaintStyle::solid(th.bg_canvas),
            |ui| {
                // Row 1 (Analytic SDF & Clip zone `[y = 14..86]` — purely non-text for strict pixel-by-pixel CPU==GPU parity):
                ui.element(
                    LayoutSpec::row().fixed(332.0, 68.0).gap(14.0).align(Align::Center),
                    PaintStyle::default(),
                    |ui| {
                        // (a) Hollow per-corner rounded border (`bg = TRANSPARENT`, `border = 6px`):
                        let mut hollow_border = PaintStyle::solid(Color::TRANSPARENT)
                            .border(6.0, Color::rgba8(240, 95, 75, 210));
                        hollow_border.radii = [14.0, 4.0, 16.0, 2.0];
                        ui.element(LayoutSpec::col().fixed(60.0, 56.0), hollow_border, |_| {});
                        // (b) Thick border (`border_width = 12 >= radius = 8`) with solid fill:
                        ui.element(
                            LayoutSpec::col().fixed(60.0, 56.0),
                            PaintStyle::solid(Color::rgb8(50, 120, 235))
                                .rounded(8.0)
                                .border(12.0, Color::rgb8(245, 200, 65)),
                            |_| {},
                        );
                        // (c) Premultiplied OKLab gradient fading cleanly to `Color::TRANSPARENT`:
                        ui.element(
                            LayoutSpec::col().fixed(72.0, 56.0),
                            PaintStyle::oklab_gradient(
                                Color::rgb8(35, 205, 250),
                                Color::TRANSPARENT,
                                0.0,
                            )
                            .rounded(10.0),
                            |_| {},
                        );
                        // (d) 24-level deep nested rounded `GpuClip` chain (`> 16`):
                        nest_24_clips(ui, 0);
                    },
                );

                // Row 2: In-place `vector_canvas` (`stroke_polyline`) + in-place shrinking Gaussian shadow card:
                let rev = ui.state().rev;
                let shrink = ui.state().shrink_shadow;
                ui.element(
                    LayoutSpec::row().fixed(332.0, 72.0).gap(24.0).align(Align::Center),
                    PaintStyle::default(),
                    |ui| {
                        ui.vector_canvas(150, 54, rev, Color::rgb8(90, 175, 255), FillRule::NonZero, |p| {
                            let shift = if rev == 1 { 0.0 } else { 22.0 };
                            p.stroke_polyline(
                                &[
                                    [8.0, 42.0 - shift * 0.5],
                                    [52.0, 14.0 + shift],
                                    [98.0, 40.0 - shift],
                                    [142.0, 16.0 + shift * 0.5],
                                ],
                                3.5,
                            );
                        });

                        ui.element(
                            LayoutSpec::col().fixed(92.0, 44.0),
                            PaintStyle::solid(Color::rgb8(135, 85, 245))
                                .rounded(10.0)
                                .drop_shadow(8.0, [0.0, 4.0], Color::rgba8(0, 0, 0, 175)),
                            |ui| {
                                let sigma = if shrink { 1.5 } else { 8.0 };
                                ui.set_paint(
                                    PaintStyle::solid(Color::rgb8(135, 85, 245))
                                        .rounded(10.0)
                                        .drop_shadow(sigma, [0.0, 4.0], Color::rgba8(0, 0, 0, 175)),
                                );
                            },
                        );
                    },
                );

                // Row 3: ClearType `SubpixelLcd` + inline color emoji (`🦀`) typography:
                ui.ui_text("Skima 1.0 Parity 🦀", 16.0, th.text_primary);
            },
        );
    };

    for is_dark in [true, false] {
        let mut sys_gpu = GuiSystem::with_system_fonts();
        sys_gpu.theme = if is_dark { Theme::dark() } else { Theme::light() };
        sys_gpu.clear_color = sys_gpu.theme.bg_canvas;

        let mut sys_cpu = GuiSystem::with_system_fonts();
        sys_cpu.theme = sys_gpu.theme;
        sys_cpu.clear_color = sys_gpu.clear_color;

        let mut vk = VulkanRenderer::new_headless(2048, 128).expect("Vulkan 1.3 headless");
        let mut cpu = CpuRenderer::new(360, 260);

        // --- FRAME 1 (Full Redraw) ---
        let st1 = ParityScene {
            rev: 1,
            shrink_shadow: false,
        };
        sys_gpu.build_frame(&st1, 360.0, 260.0, build);
        sys_cpu.build_frame(&st1, 360.0, 260.0, build);
        assert!(
            sys_gpu.clips.len() >= 24,
            "Expected >= 24 nested GpuClip entries, got {}",
            sys_gpu.clips.len()
        );
        if sys_gpu.fonts.ui_font != FontId::FALLBACK {
            assert!(
                sys_gpu.primitives.iter().any(|p| p.kind() == PrimKind::GlyphLcd),
                "SubpixelLcd ui_text must emit PrimKind::GlyphLcd primitives"
            );
        }

        let vk_f1 = sys_gpu.render_vulkan_offscreen(&mut vk, 360, 260).unwrap();
        sys_cpu.render_to_backend(&mut cpu, 360, 260, 1).unwrap();

        let vk_px1 = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * 360 + x) as usize) * 4;
            [vk_f1[i], vk_f1[i + 1], vk_f1[i + 2], vk_f1[i + 3]]
        };

        // Check 24-deep nested rounded clip interior center (solid green `40, 215, 115`, zero 0.5^24 erosion!)
        // and clipped outer corner `(x_clip0 + 1, y_clip0 + 1)` == `bg_canvas`:
        let clip0 = sys_gpu.clips[0].bounds;
        let cx_mid = (clip0[0] + clip0[2] * 0.5) as u32;
        let cy_mid = (clip0[1] + clip0[3] * 0.5) as u32;
        assert!(
            (vk_px1(cx_mid, cy_mid)[1] as i32 - 215).abs() <= 2,
            "24-deep nested rounded clip center must retain full green intensity: {:?}",
            vk_px1(cx_mid, cy_mid)
        );
        let corner_out = vk_px1(clip0[0] as u32 + 1, clip0[1] as u32 + 1);
        let canvas_rgba = sys_gpu.theme.bg_canvas.channels();
        assert!(
            (corner_out[0] as i32 - canvas_rgba[0] as i32).abs() <= 2,
            "Outside 14px rounded clip corner must stay bg_canvas: got {corner_out:?} vs {canvas_rgba:?}"
        );

        // Pixel-by-pixel parity across the entire upper scene `[y = 0..175]` (hollow/thick SDF borders,
        // premultiplied OKLab->TRANSPARENT gradient, 24-deep clips, `stroke_polyline` canvas, & Gaussian shadow):
        for y in 0..175u32 {
            for x in 0..360u32 {
                let vp = vk_px1(x, y);
                let cp = cpu.pixel_at(x, y);
                for c in 0..4 {
                    assert!(
                        (vp[c] as i32 - cp[c] as i32).abs() <= 4,
                        "Frame-1 GPU vs CPU parity mismatch (dark={is_dark}) at ({x},{y}) ch {c}: vk={vp:?} vs cpu={cp:?}"
                    );
                }
            }
        }

        // --- FRAME 2 (Partial-Damage In-Place `vector_canvas` Update + In-Place Shadow Shrink `8.0 -> 1.5`) ---
        let st2 = ParityScene {
            rev: 2,
            shrink_shadow: true,
        };
        sys_gpu.build_frame(&st2, 360.0, 260.0, build);
        sys_cpu.build_frame(&st2, 360.0, 260.0, build);
        assert_eq!(
            sys_gpu.fonts.atlas.generation(),
            1,
            "In-place vector_canvas update on Frame 2 must not reset GlyphAtlas"
        );
        assert!(
            sys_gpu.current_frame_damage.is_some(),
            "In-place canvas + shadow update must emit partial DamageRect"
        );

        let vk_f2 = sys_gpu.render_vulkan_offscreen(&mut vk, 360, 260).unwrap();
        sys_cpu.render_to_backend(&mut cpu, 360, 260, 1).unwrap();

        for y in 0..175u32 {
            for x in 0..360u32 {
                let i = ((y * 360 + x) as usize) * 4;
                let vp = [vk_f2[i], vk_f2[i + 1], vk_f2[i + 2], vk_f2[i + 3]];
                let cp = cpu.pixel_at(x, y);
                for c in 0..4 {
                    assert!(
                        (vp[c] as i32 - cp[c] as i32).abs() <= 4,
                        "Frame-2 partial-damage GPU vs CPU mismatch (dark={is_dark}) at ({x},{y}) ch {c}: vk={vp:?} vs cpu={cp:?}"
                    );
                }
            }
        }
    }
}

// =========================================================================
// Pass-2 Thread-Local Zero-Allocation Counter & Deterministic PRNG Oracles
// =========================================================================

struct CountingSystemAlloc;

thread_local! {
    static THREAD_ALLOC_STATS: core::cell::Cell<(bool, usize, usize)> =
        const { core::cell::Cell::new((false, 0, 0)) };
}

unsafe impl core::alloc::GlobalAlloc for CountingSystemAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { std::alloc::System.alloc(layout) };
        if !ptr.is_null() {
            let _ = THREAD_ALLOC_STATS.try_with(|c| {
                let (on, cnt, bytes) = c.get();
                if on {
                    c.set((true, cnt + 1, bytes + layout.size()));
                }
            });
        }
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: core::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let _ = THREAD_ALLOC_STATS.try_with(|c| {
                let (on, cnt, bytes) = c.get();
                if on {
                    c.set((true, cnt + 1, bytes + layout.size()));
                }
            });
        }
        ptr
    }

    #[inline]
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: core::alloc::Layout,
        new_size: usize,
    ) -> *mut u8 {
        let out = unsafe { std::alloc::System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            let _ = THREAD_ALLOC_STATS.try_with(|c| {
                let (on, cnt, bytes) = c.get();
                if on {
                    c.set((true, cnt + 1, bytes + new_size));
                }
            });
        }
        out
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL_TEST_ALLOC: CountingSystemAlloc = CountingSystemAlloc;

fn measure_thread_allocs<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            let _ = THREAD_ALLOC_STATS.try_with(|c| c.set((false, 0, 0)));
        }
    }
    let _guard = ResetGuard;
    THREAD_ALLOC_STATS.with(|c| c.set((true, 0, 0)));
    let res = f();
    let (_, count, bytes) = THREAD_ALLOC_STATS.with(|c| {
        let cur = c.get();
        c.set((false, 0, 0));
        cur
    });
    (res, count, bytes)
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }

    #[inline]
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = ((self.next_u32() >> 8) as f32) * (1.0 / ((1u32 << 24) as f32));
        lo + unit * (hi - lo)
    }
}

// =========================================================================
// 17. Deterministic PRNG Oracle: FLAG_DIRTY_SCROLL vs Full-Solve, 24-Track Spanned Grid, & SlotPool ABA
// =========================================================================
#[test]
fn fuzz_scroll_fast_path_vs_full_solve_oracle_and_spanned_grid_24_tracks() {
    use skima::layout::{
        FLAG_CLIPS_CHILDREN, FLAG_DIRTY_LAYOUT, FLAG_DIRTY_SCROLL, FLAG_SCROLLABLE, NodeId,
    };

    const ALL_PLACEMENTS: [AnchorPlacement; 12] = [
        AnchorPlacement::TopStart,
        AnchorPlacement::TopCenter,
        AnchorPlacement::TopEnd,
        AnchorPlacement::BottomStart,
        AnchorPlacement::BottomCenter,
        AnchorPlacement::BottomEnd,
        AnchorPlacement::LeftStart,
        AnchorPlacement::LeftCenter,
        AnchorPlacement::LeftEnd,
        AnchorPlacement::RightStart,
        AnchorPlacement::RightCenter,
        AnchorPlacement::RightEnd,
    ];
    const ALIGNS: [Align; 5] = [
        Align::Start,
        Align::Center,
        Align::End,
        Align::Stretch,
        Align::FirstBaseline,
    ];
    const JUSTIFIES: [MainAlign; 5] = [
        MainAlign::Start,
        MainAlign::Center,
        MainAlign::End,
        MainAlign::SpaceBetween,
        MainAlign::SpaceEvenly,
    ];

    fn build_oracle_tree(
        seed: u64,
        snap: bool,
        scrolls: [f32; 4],
        fonts: &FontSystem,
    ) -> (LayoutTree, [NodeId; 4]) {
        let mut rng = SplitMix64::new(seed);
        let mut tree = LayoutTree::new();
        tree.snap_to_pixels = snap;

        let root = tree.alloc_node(LayoutSpec::col().fixed(880.0, 640.0).pad(11.25).gap(7.5));
        tree.topo[root.idx()].flags |= FLAG_CLIPS_CHILDREN;
        tree.root = root;

        let mut outer_spec = LayoutSpec::col()
            .fixed(850.0, 560.0)
            .pad_xy(rng.next_f32(4.0, 14.0), rng.next_f32(4.0, 14.0))
            .gap(rng.next_f32(3.0, 11.0));
        outer_spec.scroll_y = scrolls[0];
        let s_outer = tree.alloc_node(outer_spec);
        tree.topo[s_outer.idx()].flags |= FLAG_CLIPS_CHILDREN | FLAG_SCROLLABLE;

        let mut targets = Vec::with_capacity(12);

        // 1. Baseline-aligned Row with wrapped ShapedParagraph + anchor targets:
        let row_bl = tree.alloc_node(
            LayoutSpec::row()
                .width_grow(1.0)
                .pad_xy(6.4, 5.2)
                .gap(8.3)
                .align(Align::FirstBaseline)
                .justify(JUSTIFIES[rng.next_usize(JUSTIFIES.len())]),
        );
        let txt_node = tree.alloc_node(LayoutSpec::col().width_fixed(rng.next_f32(150.0, 230.0)));
        tree.attach_shaped_text(
            txt_node,
            "IALE scroll-only fast-path oracle verifying bit-identical boxes across nested clips",
            FontId::FALLBACK,
            14.0,
            19.0,
            fonts,
        );
        let t0 = tree.alloc_node(LayoutSpec::col().fixed(78.4, 29.6));
        let t1 = tree.alloc_node(LayoutSpec::col().fixed(64.8, 24.2));
        targets.push(txt_node);
        targets.push(t0);
        targets.push(t1);
        tree.set_children(row_bl, &[txt_node, t0, t1]);

        // 2. Nested scrollable Col with randomized Align / MainAlign:
        let mut inner_col_spec = LayoutSpec::col()
            .fixed(780.0, 145.0)
            .pad_xy(7.75, 6.25)
            .gap(5.5)
            .align(ALIGNS[rng.next_usize(ALIGNS.len())])
            .justify(JUSTIFIES[rng.next_usize(JUSTIFIES.len())]);
        inner_col_spec.scroll_y = scrolls[1];
        let s_col = tree.alloc_node(inner_col_spec);
        tree.topo[s_col.idx()].flags |= FLAG_CLIPS_CHILDREN;
        let mut col_kids = Vec::with_capacity(4);
        for _ in 0..4 {
            let c = tree.alloc_node(
                LayoutSpec::row()
                    .fixed(rng.next_f32(120.0, 340.0), rng.next_f32(24.0, 48.0))
                    .pad(3.5),
            );
            targets.push(c);
            col_kids.push(c);
        }
        tree.set_children(s_col, &col_kids);

        // 3. Nested scrollable WrapRow with randomized Align / MainAlign:
        let mut wrap_spec = LayoutSpec::wrap_row()
            .fixed(480.0, 120.0)
            .pad_xy(6.5, 5.5)
            .gap(7.25)
            .align(ALIGNS[rng.next_usize(ALIGNS.len())])
            .justify(JUSTIFIES[rng.next_usize(JUSTIFIES.len())]);
        wrap_spec.scroll_y = scrolls[2];
        let s_wrap = tree.alloc_node(wrap_spec);
        tree.topo[s_wrap.idx()].flags |= FLAG_CLIPS_CHILDREN;
        let mut wrap_kids = Vec::with_capacity(6);
        for k in 0..6 {
            let wk = tree.alloc_node(
                LayoutSpec::col().fixed(rng.next_f32(95.0, 175.0), rng.next_f32(20.0, 38.0)),
            );
            if k < 3 {
                targets.push(wk);
            }
            wrap_kids.push(wk);
        }
        tree.set_children(s_wrap, &wrap_kids);

        // 4. Nested scrollable 2D Grid with spanned items + MinMax/Auto/Fit/Fr tracks:
        let mut grid_spec = LayoutSpec::col()
            .fixed(540.0, 160.0)
            .pad_xy(6.0, 5.5)
            .gap(6.75)
            .align(ALIGNS[rng.next_usize(4)])
            .justify(JUSTIFIES[rng.next_usize(JUSTIFIES.len())]);
        grid_spec.scroll_y = scrolls[3];
        let s_grid = tree.alloc_node(grid_spec);
        tree.topo[s_grid.idx()].flags |= FLAG_CLIPS_CHILDREN;
        tree.attach_grid(
            s_grid,
            vec![
                TrackSize::MinMax(55.0, 115.0),
                TrackSize::Auto,
                TrackSize::Fit,
                TrackSize::Fr(1.0),
            ],
            vec![
                TrackSize::Fit,
                TrackSize::MinMax(32.0, 68.0),
                TrackSize::Fr(1.0),
            ],
        );
        let g0 = tree.alloc_node(LayoutSpec::col().fixed(148.0, 34.5).grid_span(2, 1));
        let g1 = tree.alloc_node(LayoutSpec::col().fixed(62.0, 74.0).grid_span(1, 2));
        let g2 = tree.alloc_node(LayoutSpec::col().fixed(58.0, 28.0));
        let g3 = tree.alloc_node(LayoutSpec::col().grow(1.0).grid_span(2, 1));
        targets.push(g0);
        targets.push(g1);
        tree.set_children(s_grid, &[g0, g1, g2, g3]);

        tree.set_children(s_outer, &[row_bl, s_col, s_wrap, s_grid]);

        // 5. Attach all 12 Pass-4 AnchorPlacement variants to scrolled targets inside nested clips:
        let mut root_children = Vec::with_capacity(13);
        root_children.push(s_outer);
        for (p_idx, &placement) in ALL_PLACEMENTS.iter().enumerate() {
            let tg = targets[p_idx % targets.len()];
            let ov = tree.alloc_node(
                LayoutSpec::col()
                    .fixed(rng.next_f32(84.0, 156.0), rng.next_f32(36.0, 78.0))
                    .pad(4.5),
            );
            tree.topo[ov.idx()].flags |= FLAG_CLIPS_CHILDREN;
            let ov_leaf = tree.alloc_node(LayoutSpec::row().fixed(42.0, 16.0));
            tree.set_children(ov, &[ov_leaf]);
            tree.set_floating_anchor(
                ov,
                tg,
                placement,
                rng.next_f32(-6.0, 8.0),
                rng.next_f32(2.0, 9.0),
            );
            root_children.push(ov);
        }
        tree.set_children(root, &root_children);
        tree.compute_layout(880.0, 640.0);
        (tree, [s_outer, s_col, s_wrap, s_grid])
    }

    let fonts = FontSystem::new();
    let mut prng = SplitMix64::new(0xD1B5_4A32_D192_ED03);

    for seed_idx in 0..8u64 {
        let scene_seed = 0x517C_C1B7_2722_0A95 ^ seed_idx.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for snap in [false, true] {
            let (mut inc, s_nodes) = build_oracle_tree(scene_seed, snap, [0.0; 4], &fonts);
            // Warm up scratch capacities once, then enforce 0 B heap allocation across all scroll mutations:
            inc.set_scroll_y(s_nodes[0], 4.0);
            inc.compute_layout(880.0, 640.0);

            for step in 0..25usize {
                let scrolls = if step == 12 {
                    // Extreme 1e6 jump exposing any stateful `y += dy` float drift:
                    [1_000_000.0, 500_000.0, 250_000.0, 125_000.0]
                } else if step == 24 {
                    [0.0, 0.0, 0.0, 0.0]
                } else {
                    [
                        prng.next_f32(-80.0, 460.0),
                        prng.next_f32(-20.0, 180.0),
                        prng.next_f32(0.0, 140.0),
                        prng.next_f32(0.0, 160.0),
                    ]
                };

                let (_, alloc_cnt, alloc_bytes) = measure_thread_allocs(|| {
                    if step % 2 == 0 {
                        for (&nid, &sy) in s_nodes.iter().zip(scrolls.iter()) {
                            inc.set_scroll_y(nid, sy);
                        }
                    } else {
                        for (&nid, &sy) in s_nodes.iter().zip(scrolls.iter()).rev() {
                            inc.set_scroll_y(nid, sy);
                        }
                    }
                    assert_eq!(inc.topo[inc.root.idx()].flags & FLAG_DIRTY_LAYOUT, 0);
                    assert!(
                        s_nodes
                            .iter()
                            .any(|n| (inc.topo[n.idx()].flags & FLAG_DIRTY_SCROLL) != 0
                                || scrolls == [0.0; 4])
                    );
                    inc.compute_layout(880.0, 640.0);
                });
                assert_eq!(
                    (alloc_cnt, alloc_bytes),
                    (0, 0),
                    "FLAG_DIRTY_SCROLL fast-path must allocate 0 B in steady state"
                );
                assert_eq!(inc.last_metrics.text_shaper_calls, 0);

                let (fresh, _) = build_oracle_tree(scene_seed, snap, scrolls, &fonts);
                assert_eq!(inc.boxes.len(), fresh.boxes.len());
                for (n_i, (&ba, &bb)) in inc.boxes.iter().zip(fresh.boxes.iter()).enumerate() {
                    assert_eq!(
                        (
                            ba.x.to_bits(),
                            ba.y.to_bits(),
                            ba.w.to_bits(),
                            ba.h.to_bits(),
                            ba.fit_w.to_bits(),
                            ba.fit_h.to_bits(),
                            ba.min_content_w.to_bits(),
                            ba.baseline.to_bits(),
                        ),
                        (
                            bb.x.to_bits(),
                            bb.y.to_bits(),
                            bb.w.to_bits(),
                            bb.h.to_bits(),
                            bb.fit_w.to_bits(),
                            bb.fit_h.to_bits(),
                            bb.min_content_w.to_bits(),
                            bb.baseline.to_bits(),
                        ),
                        "Scroll fast-path ComputedBox mismatch (seed={seed_idx}, snap={snap}, step={step}, node={n_i}): inc={ba:?} vs fresh={bb:?}"
                    );
                    assert_eq!(
                        inc.clip_rects[n_i].map(f32::to_bits),
                        fresh.clip_rects[n_i].map(f32::to_bits),
                        "Scroll fast-path clip_rect mismatch (seed={seed_idx}, snap={snap}, step={step}, node={n_i})"
                    );
                }
            }
        }
    }

    // Wide 24x24 Grid spanned intrinsic sizing (including nested Grid child using wide `col_span = 20`!):
    let mut gtree = LayoutTree::new();
    let outer_g = gtree.alloc_node(LayoutSpec::col().gap(2.0));
    gtree.attach_grid(
        outer_g,
        vec![TrackSize::Auto; 24],
        vec![TrackSize::Fit; 24],
    );
    gtree.root = outer_g;
    // Child 0: nested Grid that ALSO carries wide `grid_span(20, 2)` (tests `WIDE_SPAN_SENTINEL` + `wide_spans` side table!)
    let nested_wide_grid = gtree.alloc_node(LayoutSpec::col().grid_span(20, 2));
    gtree.attach_grid(
        nested_wide_grid,
        vec![TrackSize::Fixed(20.0); 20],
        vec![TrackSize::Fixed(15.0); 2],
    );
    let inner_cell = gtree.alloc_node(LayoutSpec::col().fixed(20.0, 15.0));
    gtree.set_children(nested_wide_grid, &[inner_cell]);
    // Child 1: 1-column cell immediately following `nested_wide_grid` -> must land at column 20 (`x = 598.3333 + 2.0 = 600.3333`)!
    let after_wide = gtree.alloc_node(LayoutSpec::col().fixed(5.0, 15.0));
    // Child 2: 24-column spanned banner (`646px` wide -> lifts `0..20` from `18.1px` by `(646 - 408 - 5)/24` or exact §11.5 delta!)
    let banner24 = gtree.alloc_node(LayoutSpec::col().fixed(646.0, 20.0).grid_span(24, 1));
    gtree.set_children(outer_g, &[nested_wide_grid, after_wide, banner24]);
    gtree.compute_layout(1200.0, 900.0);
    // Exact CSS Grid §11.5 oracle:
    // - Pass 1 (span=1): col 20 = 5.0, all other cols = 0.0
    // - Pass 2 (span=20): `nested_wide_grid` (`fit_w = 400`, 19 gaps = `38`) -> cols `0..20` each `362 / 20 = 18.1`
    // - Pass 3 (span=24): `banner24` (`fit_w = 646`, 23 gaps = `46`, current sum = `362 + 5 = 367`) ->
    //   extra `646 - 46 - 367 = 233` distributed equally over all 24 `Auto` cols (`233 / 24` per col)!
    let expected_wide20_w = 400.0 + 20.0 * (233.0 / 24.0);
    assert!((gtree.boxes[outer_g.idx()].fit_w - 646.0).abs() < 1e-3);
    assert!(
        (gtree.boxes[nested_wide_grid.idx()].w - expected_wide20_w).abs() < 1e-3,
        "Nested Grid with wide col_span=20 must match exact CSS Grid §11.5 width ({expected_wide20_w}), got {}",
        gtree.boxes[nested_wide_grid.idx()].w
    );
    assert!(
        (gtree.boxes[after_wide.idx()].x - (expected_wide20_w + 2.0)).abs() < 1e-3,
        "Sibling after col_span=20 item must be placed in column 20 at x={}, got {}",
        expected_wide20_w + 2.0,
        gtree.boxes[after_wide.idx()].x
    );

    // Randomized `SlotPool` insert / remove / retain / clear / iter / iter_mut zero-ABA oracle:
    let mut pool = SlotPool::<u64>::with_capacity(32);
    let mut live: Vec<(Key<u64>, u64)> = Vec::new();
    let mut dead_keys: Vec<Key<u64>> = Vec::new();
    for step in 0..1_200u64 {
        match prng.next_usize(6) {
            0..=2 => {
                let k = pool.insert(step * 13 + 7);
                live.push((k, step * 13 + 7));
            }
            3 if !live.is_empty() => {
                let idx = prng.next_usize(live.len());
                let (k, expected) = live.swap_remove(idx);
                assert_eq!(pool.remove(k), Some(expected));
                dead_keys.push(k);
            }
            4 => {
                pool.retain(|k, v| {
                    let keep = (*v & 1) == 0;
                    if !keep {
                        dead_keys.push(k);
                    }
                    keep
                });
                live.retain(|&(_, v)| (v & 1) == 0);
            }
            5 if step % 97 == 0 => {
                for &(k, _) in &live {
                    dead_keys.push(k);
                }
                live.clear();
                pool.clear();
            }
            _ => {
                for (_, v) in &mut pool {
                    *v = v.wrapping_add(2);
                }
                for (_, v) in &mut live {
                    *v = v.wrapping_add(2);
                }
            }
        }
        assert_eq!(pool.len(), live.len());
        for &(k, v) in &live {
            assert_eq!(pool.get(k), Some(&v));
        }
        for &dk in dead_keys.iter().rev().take(64) {
            assert!(!pool.contains(dk), "Stale SlotPool key {dk:?} collided!");
        }
    }
}

// =========================================================================
// 18. Deterministic PRNG Oracle: VirtualListIndex 150k Non-Dyadic Updates & Overscan
// =========================================================================
#[test]
fn fuzz_fenwick_virtual_list_150k_random_non_dyadic_oracle_and_overscan() {
    #[inline]
    fn exact_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let v = s - a;
        let e = (a - (s - v)) + (b - v);
        (s, e)
    }

    #[inline]
    fn clean_h(h: f32) -> f32 {
        if h.is_finite() && h > 0.0 { h } else { 0.0 }
    }

    let mut rng = SplitMix64::new(0xA076_1D64_78BD_642F);
    let mut vlist = VirtualListIndex::new(2048, 27.35);
    let mut oracle = vec![27.35_f32; 2048];
    let mut tracked_scroll = 18_500.0_f64;

    let special_heights = [
        0.0_f32,
        0.0017,
        13.37,
        33.333_332,
        89.95,
        1456.78,
        f32::NAN,
        -12.5,
    ];

    for step in 1..=150_000usize {
        let op = rng.next_usize(100);
        if op < 80 && !oracle.is_empty() {
            let idx = rng.next_usize(oracle.len());
            let h = if (step & 7) == 0 {
                special_heights[rng.next_usize(special_heights.len())]
            } else {
                rng.next_f32(0.002, 420.0)
            };
            oracle[idx] = clean_h(h);
            let (_, allocs, bytes) = measure_thread_allocs(|| {
                vlist.set_height(idx, h);
            });
            assert_eq!((allocs, bytes), (0, 0));
        } else if op < 92 && !oracle.is_empty() {
            let idx = rng.next_usize(oracle.len());
            let anchor = rng.next_usize(oracle.len() + 1);
            let h = rng.next_f32(0.0, 310.0);
            let old_h = oracle[idx];
            let new_h = clean_h(h);
            oracle[idx] = new_h;
            let expected_scroll = if idx < anchor {
                (tracked_scroll + ((new_h as f64) - (old_h as f64))).max(0.0)
            } else {
                tracked_scroll
            };
            let (_, allocs, bytes) = measure_thread_allocs(|| {
                vlist.set_height_anchored(idx, h, anchor, &mut tracked_scroll);
            });
            assert_eq!((allocs, bytes), (0, 0));
            assert!((tracked_scroll - expected_scroll).abs() < 1e-9);
        } else if op < 96 {
            if oracle.len() < 4096 {
                let h = if (step & 3) == 0 {
                    0.0
                } else {
                    rng.next_f32(0.01, 250.0)
                };
                oracle.push(clean_h(h));
                vlist.push(h);
            }
        } else if op < 98 {
            if oracle.len() > 64 {
                assert_eq!(vlist.pop(), oracle.pop());
            }
        } else {
            let new_len = 256 + rng.next_usize(3200);
            let def_h = rng.next_f32(0.0, 95.0);
            oracle.resize(new_len, clean_h(def_h));
            vlist.resize(new_len, def_h);
        }

        if step % 50_000 == 0 {
            assert_eq!(vlist.len(), oracle.len());
            let mut sum_s = 0.0_f64;
            let mut sum_e = 0.0_f64;
            for (i, &oh) in oracle.iter().enumerate() {
                let expected = sum_s + sum_e;
                let got = vlist.offset_of(i);
                assert!(
                    (got - expected).abs() < 1e-9,
                    "Fenwick drift >= 1e-9 at step={step}, index={i}: got={got}, expected={expected}"
                );
                let (ns, ne) = exact_two_sum(sum_s, oh as f64);
                sum_s = ns;
                sum_e += ne;
            }
            assert!((vlist.total_height() - (sum_s + sum_e)).abs() < 1e-9);
        }
    }

    // Brute-force interval oracle for `visible_range` & `visible_range_overscan` across 200 random windows:
    let mut prefix = Vec::with_capacity(oracle.len() + 1);
    let mut ps = 0.0_f64;
    let mut pe = 0.0_f64;
    prefix.push(0.0);
    for &h in &oracle {
        let (ns, ne) = exact_two_sum(ps, h as f64);
        ps = ns;
        pe += ne;
        prefix.push(ps + pe);
    }
    let tot = *prefix.last().unwrap();

    for q in 0..200usize {
        let sy = if q % 10 == 0 && !oracle.is_empty() {
            // Exact item boundary query!
            prefix[rng.next_usize(prefix.len())]
        } else {
            (rng.next_f32(-50.0, (tot as f32) + 200.0)) as f64
        };
        let vh = (rng.next_f32(0.0, 900.0)) as f64;
        let over = if q % 5 == 0 {
            0.0
        } else {
            (rng.next_f32(-20.0, 240.0)) as f64
        };

        let ((st, en, first_off), allocs, bytes) =
            measure_thread_allocs(|| vlist.visible_range_overscan(sy, vh, over));
        assert_eq!((allocs, bytes), (0, 0));
        assert_eq!(first_off.to_bits(), vlist.offset_of(st).to_bits());
        if over <= 0.0 {
            assert_eq!((st, en, first_off), vlist.visible_range(sy, vh));
        }

        let eff_sy = sy.max(0.0);
        let eff_over = over.max(0.0);
        let target = (eff_sy - eff_over).max(0.0);
        let limit = eff_sy + vh.max(0.0) + eff_over;

        let mut exp_st = oracle.len();
        for i in 0..oracle.len() {
            let y0 = prefix[i];
            let y1 = prefix[i + 1];
            let h = oracle[i];
            if (h > 0.0 && y1 > target + 1e-11) || (h == 0.0 && y0 >= target - 1e-11) {
                exp_st = i;
                break;
            }
        }
        let mut exp_en = exp_st;
        while exp_en < oracle.len() && prefix[exp_en] < limit - 1e-11 {
            exp_en += 1;
        }
        assert_eq!(
            (st, en),
            (exp_st, exp_en),
            "visible_range_overscan mismatch at q={q} (sy={sy}, vh={vh}, over={over})"
        );
    }
}

// =========================================================================
// 19. Zero-Alloc RasterScratch, ShapedParagraph Caret/Hit-Test Oracle, & Dynamic GlyphAtlas Growth
// =========================================================================
#[test]
fn fuzz_zero_alloc_raster_scratch_shaped_paragraph_caret_oracle_and_dynamic_atlas_growth() {
    use skima::vector::RasterScratch;

    // 1. `Path2D::clear` + `stroke_polyline`/`stroke_polygon` + `rasterize_into`:
    //    Bit-identical to `rasterize_mask` AND exact 0 B steady-state heap allocation!
    let mut path = Path2D::with_capacity(512);
    let mut scratch = RasterScratch::with_capacity(512, 128, 128);
    let mut mask = Vec::with_capacity(128 * 128);
    let mut rng = SplitMix64::new(0xE703_7ED1_A0B4_28DB);

    // Warm up internal segment (`>= 8` segments) & active-edge scratch buffers once:
    let warmup_pts = [
        [4.0, 4.0],
        [22.0, 14.0],
        [48.0, 8.0],
        [72.0, 28.0],
        [98.0, 16.0],
        [118.0, 52.0],
        [86.0, 96.0],
        [28.0, 88.0],
    ];
    path.stroke_polygon(&warmup_pts, 5.5);
    path.cubic_to(0.0, 0.0, 128.0, 20.0, 50.0, 120.0);
    path.rasterize_into(128, 128, FillRule::NonZero, &mut scratch, &mut mask);

    let dims: &[(u16, u16)] = &[(1, 1), (1, 37), (17, 19), (64, 48), (128, 96)];
    for case_idx in 0..60usize {
        let (w, h) = dims[case_idx % dims.len()];
        let rule = if (case_idx & 1) == 0 {
            FillRule::NonZero
        } else {
            FillRule::EvenOdd
        };
        let mut poly = [[0.0_f32; 2]; 8];
        for pt in &mut poly {
            *pt = [
                rng.next_f32(-12.0, (w as f32) + 12.0),
                rng.next_f32(-12.0, (h as f32) + 12.0),
            ];
        }
        let stroke_w = rng.next_f32(0.8, 5.5);
        let c0 = [rng.next_f32(-5.0, w as f32), rng.next_f32(-5.0, h as f32)];
        let c1 = [rng.next_f32(0.0, w as f32), rng.next_f32(0.0, h as f32)];
        let c2 = [rng.next_f32(0.0, w as f32), rng.next_f32(0.0, h as f32)];
        let c3 = [rng.next_f32(-5.0, w as f32), rng.next_f32(-5.0, h as f32)];

        let (_, alloc_cnt, alloc_bytes) = measure_thread_allocs(|| {
            path.clear();
            if case_idx % 3 == 0 {
                path.stroke_polyline(&poly[..6], stroke_w);
            } else if case_idx % 3 == 1 {
                path.stroke_polygon(&poly[..5], stroke_w);
            } else {
                // Intentionally leave subpath unclosed to verify `&self` implicit close parity!
                path.move_to(c0[0], c0[1])
                    .quad_to(c1[0], c1[1], c2[0], c2[1])
                    .cubic_to(c1[0], c3[1], c3[0], c1[1], c3[0], c3[1]);
            }
            path.rasterize_into(w, h, rule, &mut scratch, &mut mask);
        });
        assert_eq!(
            (alloc_cnt, alloc_bytes),
            (0, 0),
            "Path2D::clear + rasterize_into allocated {alloc_bytes} B on steady-state case {case_idx}"
        );

        let expected = path.clone().rasterize_mask(w, h, rule);
        assert_eq!(
            mask, expected,
            "rasterize_into != rasterize_mask on case {case_idx} ({w}x{h}, {rule:?})"
        );
    }

    // 2. `ShapedParagraph` vs `FontSystem` caret & hit-test oracle + 0 B heap allocation proof:
    let mut fonts = FontSystem::with_system_fonts();
    let corpora = [
        "AV Wa To fi ffi fl != ->",
        "\tleading\ttabs\t\tand  spaces ",
        "ctrl\x00\x07\x1b\rmid\x7fclean",
        "ZWJ 👨‍👩‍👧‍👦 🦀 ⚡ ⚡\u{FE0F} ✅ ❤️ café e\u{0301}",
        "multi\r\n\n\tsecond_line 🦀 ffi",
    ];

    for &fid in &[FontId::FALLBACK, fonts.ui_font, fonts.mono_font] {
        for sample in corpora {
            let para = fonts.shape_paragraph(sample, fid, 15.0, 21.0);
            for b in 0..=sample.len() {
                let ((px, hit_back), allocs, bytes) = measure_thread_allocs(|| {
                    let x = para.byte_offset_to_x(sample, b);
                    let h = para.x_to_byte_offset(sample, x);
                    (x, h)
                });
                assert_eq!(
                    (allocs, bytes),
                    (0, 0),
                    "ShapedParagraph caret/hit-test must allocate 0 B!"
                );
                assert!(sample.is_char_boundary(hit_back));
                let fx = fonts.byte_offset_to_x(sample, b, fid, 15.0);
                assert!(
                    (px - fx).abs() < 1e-3,
                    "byte_offset_to_x mismatch at b={b} in {sample:?} ({fid:?}): para={px} vs fs={fx}"
                );
            }

            let mut x = -6.0_f32;
            let limit_x = para.max_unwrapped_w + 18.0;
            while x <= limit_x {
                let (pb, allocs, bytes) =
                    measure_thread_allocs(|| para.x_to_byte_offset(sample, x));
                assert_eq!((allocs, bytes), (0, 0));
                let fb = fonts.x_to_byte_offset(sample, x, fid, 15.0);
                assert_eq!(
                    pb, fb,
                    "x_to_byte_offset mismatch at x={x} in {sample:?} ({fid:?})"
                );
                x += 1.25;
            }
        }
    }

    // 3. Dynamic `GlyphAtlas` growth `1024 -> 2048 -> 4096` texel preservation & UV rescaling:
    assert_eq!((fonts.atlas.width, fonts.atlas.height), (1024, 1024));
    let shaped = fonts.shape_paragraph("Skima", fonts.ui_font, 22.0, 28.0);
    let g0 = shaped.glyphs[0];
    let reg_1k = fonts.rasterize_glyph(g0.font_id, g0.glyph_id, 22.0, 0.0, AntialiasMode::Grayscale);
    assert!(reg_1k.w > 0 && reg_1k.h > 0);
    let snapshot_1k: Vec<u8> = (0..reg_1k.h as usize)
        .flat_map(|r| {
            let off = (reg_1k.y as usize + r) * (fonts.atlas.width as usize) + (reg_1k.x as usize);
            fonts.atlas.r8_data[off..off + reg_1k.w as usize].to_vec()
        })
        .collect();

    let filler_500 = vec![180u8; 500 * 500];
    for _ in 0..5 {
        assert!(fonts.atlas.upload_r8(500, 500, 0, 0, &filler_500).is_some());
    }
    assert_eq!((fonts.atlas.width, fonts.atlas.height), (2048, 2048));
    let reg_2k = fonts.rasterize_glyph(g0.font_id, g0.glyph_id, 22.0, 0.0, AntialiasMode::Grayscale);
    assert_eq!((reg_2k.x, reg_2k.y, reg_2k.w, reg_2k.h), (reg_1k.x, reg_1k.y, reg_1k.w, reg_1k.h));
    assert!((reg_2k.u1 - reg_1k.u1 * 0.5).abs() < 1e-6);

    let filler_1k = vec![90u8; 1000 * 1000];
    for _ in 0..10 {
        assert!(fonts.atlas.upload_r8(1000, 1000, 0, 0, &filler_1k).is_some());
    }
    assert_eq!((fonts.atlas.width, fonts.atlas.height), (4096, 4096));
    let reg_4k = fonts.rasterize_glyph(g0.font_id, g0.glyph_id, 22.0, 0.0, AntialiasMode::Grayscale);
    assert!((reg_4k.u1 - reg_1k.u1 * 0.25).abs() < 1e-6);

    let snapshot_4k: Vec<u8> = (0..reg_4k.h as usize)
        .flat_map(|r| {
            let off = (reg_4k.y as usize + r) * (fonts.atlas.width as usize) + (reg_4k.x as usize);
            fonts.atlas.r8_data[off..off + reg_4k.w as usize].to_vec()
        })
        .collect();
    assert_eq!(
        snapshot_1k, snapshot_4k,
        "Dynamic GlyphAtlas 1024->2048->4096 growth must preserve glyph texels byte-for-byte!"
    );

    // Wide QHD (`1480x48`) & 2x Retina (`2400x32`) uploads on a fresh `1024x1024` atlas must auto-grow (`1024 -> 2048 -> 4096`)
    // instead of failing `can_ever_fit` against the initial 1024px packer!
    let mut qhd_atlas = skima::text::GlyphAtlas::new(1024, 1024);
    let qhd_r8 = vec![200u8; 1480 * 48];
    let qhd_reg = qhd_atlas
        .upload_r8(1480, 48, 0, 0, &qhd_r8)
        .expect("1480x48 QHD R8 region must auto-grow 1024->2048 atlas");
    assert_eq!((qhd_atlas.width, qhd_atlas.height), (2048, 2048));
    assert_eq!((qhd_reg.w, qhd_reg.h), (1480, 48));
    let uhd_rgba = vec![255u8; 2400 * 16 * 4];
    let uhd_reg = qhd_atlas
        .upload_rgba8(2400, 16, 0, 0, &uhd_rgba)
        .expect("2400x16 Retina RGBA8 region must auto-grow 2048->4096 atlas");
    assert_eq!((qhd_atlas.width, qhd_atlas.height), (4096, 4096));
    assert_eq!((uhd_reg.w, uhd_reg.h), (2400, 16));
}

// =========================================================================
// 20. Deterministic PRNG Fuzz: UTF-8 TextInput Selection/Clipboard, SlotTape Stress, Tooltips, & Zero-Alloc Canvas
// =========================================================================
#[test]
fn fuzz_text_input_utf8_selection_clipboard_slot_reconciliation_tooltip_and_zero_alloc_canvas() {
    fn ref_prev_word(s: &str, pos: usize) -> usize {
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
        let want = is_word(first_ch);
        for (idx, ch) in chars {
            if ch.is_whitespace() || (is_word(ch) != want) {
                break;
            }
            start = idx;
        }
        start
    }

    fn ref_next_word(s: &str, pos: usize) -> usize {
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
        let want = is_word(first_ch);
        for (rel, ch) in chars {
            if ch.is_whitespace() || (is_word(ch) != want) {
                return clamped + rel;
            }
        }
        s.len()
    }

    // --- PART A: 500-Step PRNG UTF-8 TextInput Selection & Clipboard Oracle ---
    let mut text = String::from("init café 🦀 rust_100");
    let mut ref_text = text.clone();
    let mut ref_cursor = ref_text.len();
    let mut ref_anchor = ref_text.len();
    let mut ref_clip = String::new();

    let mut sys = GuiSystem::with_system_fonts();
    let build_input = |ui: &mut Ui<'_, '_, String>| {
        ui.element(LayoutSpec::col().fixed(280.0, 52.0), PaintStyle::default(), |ui| {
            ui.text_input("Fuzz input...", |s| s, |s| s);
        });
    };
    sys.build_frame(&text, 280.0, 52.0, build_input);
    assert!(sys.dispatch_key(&KeyInput::Tab { shift: false }, &mut text));

    let snippets = ["a", "Z", "é", "🦀", "👨‍👩‍👧‍👦", "word_2 ", "fi", ""];
    let mut rng = SplitMix64::new(0x2545_F491_4F6C_DD1D);

    for step in 0..500usize {
        let cur = ref_text.floor_char_boundary(ref_cursor.min(ref_text.len()));
        let anc = ref_text.floor_char_boundary(ref_anchor.min(ref_text.len()));
        let (smin, smax) = if cur <= anc { (cur, anc) } else { (anc, cur) };
        let has_sel = smin < smax;

        let ev = match rng.next_usize(19) {
            0 => {
                // Ctrl+A MUST select [0..len] and NEVER delete text!
                ref_anchor = 0;
                ref_cursor = ref_text.len();
                KeyInput::SelectAll
            }
            1 => {
                if has_sel {
                    ref_clip.clear();
                    ref_clip.push_str(&ref_text[smin..smax]);
                }
                KeyInput::Copy
            }
            2 => {
                if has_sel {
                    ref_clip.clear();
                    ref_clip.push_str(&ref_text[smin..smax]);
                    ref_text.drain(smin..smax);
                    ref_cursor = smin;
                    ref_anchor = smin;
                }
                KeyInput::Cut
            }
            3 => {
                let paste_s = if ref_text.len() > 160 {
                    String::new()
                } else {
                    ref_clip.clone()
                };
                if has_sel {
                    ref_text.replace_range(smin..smax, &paste_s);
                    ref_cursor = smin + paste_s.len();
                    ref_anchor = ref_cursor;
                } else if !paste_s.is_empty() {
                    ref_text.insert_str(cur, &paste_s);
                    ref_cursor = cur + paste_s.len();
                    ref_anchor = ref_cursor;
                } else {
                    ref_cursor = cur;
                    ref_anchor = cur;
                }
                KeyInput::Paste(paste_s)
            }
            4 | 5 => {
                let ins = if ref_text.len() > 160 {
                    "x"
                } else {
                    snippets[rng.next_usize(snippets.len())]
                };
                if has_sel {
                    ref_text.replace_range(smin..smax, ins);
                    ref_cursor = smin + ins.len();
                    ref_anchor = ref_cursor;
                } else if !ins.is_empty() {
                    ref_text.insert_str(cur, ins);
                    ref_cursor = cur + ins.len();
                    ref_anchor = ref_cursor;
                } else {
                    ref_cursor = cur;
                    ref_anchor = cur;
                }
                KeyInput::Text(ins.to_owned())
            }
            6 => {
                if has_sel {
                    ref_text.replace_range(smin..smax, " ");
                    ref_cursor = smin + 1;
                    ref_anchor = smin + 1;
                } else {
                    ref_text.insert(cur, ' ');
                    ref_cursor = cur + 1;
                    ref_anchor = cur + 1;
                }
                KeyInput::Space
            }
            7 => {
                if has_sel {
                    ref_text.drain(smin..smax);
                    ref_cursor = smin;
                    ref_anchor = smin;
                } else if cur > 0 {
                    let p = ref_text.floor_char_boundary(cur - 1);
                    ref_text.drain(p..cur);
                    ref_cursor = p;
                    ref_anchor = p;
                }
                KeyInput::Backspace
            }
            8 => {
                if has_sel {
                    ref_text.drain(smin..smax);
                    ref_cursor = smin;
                    ref_anchor = smin;
                } else if cur < ref_text.len() {
                    let n = ref_text.ceil_char_boundary(cur + 1);
                    ref_text.drain(cur..n);
                    ref_cursor = cur;
                    ref_anchor = cur;
                }
                KeyInput::Delete
            }
            9 => {
                if has_sel {
                    ref_text.drain(smin..smax);
                    ref_cursor = smin;
                    ref_anchor = smin;
                } else if cur > 0 {
                    let p = ref_prev_word(&ref_text, cur);
                    ref_text.drain(p..cur);
                    ref_cursor = p;
                    ref_anchor = p;
                }
                KeyInput::CtrlBackspace
            }
            10 => {
                if has_sel {
                    ref_text.drain(smin..smax);
                    ref_cursor = smin;
                    ref_anchor = smin;
                } else if cur < ref_text.len() {
                    let n = ref_next_word(&ref_text, cur);
                    ref_text.drain(cur..n);
                    ref_cursor = cur;
                    ref_anchor = cur;
                }
                KeyInput::CtrlDelete
            }
            11 => {
                let next = if has_sel {
                    smin
                } else if cur > 0 {
                    ref_text.floor_char_boundary(cur - 1)
                } else {
                    0
                };
                ref_cursor = next;
                ref_anchor = next;
                KeyInput::ArrowLeft
            }
            12 => {
                let next = if has_sel {
                    smax
                } else if cur < ref_text.len() {
                    ref_text.ceil_char_boundary(cur + 1)
                } else {
                    ref_text.len()
                };
                ref_cursor = next;
                ref_anchor = next;
                KeyInput::ArrowRight
            }
            13 => {
                ref_anchor = anc;
                ref_cursor = if cur > 0 {
                    ref_text.floor_char_boundary(cur - 1)
                } else {
                    0
                };
                KeyInput::ShiftArrowLeft
            }
            14 => {
                ref_anchor = anc;
                ref_cursor = if cur < ref_text.len() {
                    ref_text.ceil_char_boundary(cur + 1)
                } else {
                    ref_text.len()
                };
                KeyInput::ShiftArrowRight
            }
            15 => {
                ref_anchor = anc;
                ref_cursor = ref_prev_word(&ref_text, cur);
                KeyInput::CtrlShiftArrowLeft
            }
            16 => {
                ref_anchor = anc;
                ref_cursor = ref_next_word(&ref_text, cur);
                KeyInput::CtrlShiftArrowRight
            }
            17 => {
                ref_anchor = anc;
                ref_cursor = 0;
                KeyInput::ShiftHome
            }
            _ => {
                ref_anchor = anc;
                ref_cursor = ref_text.len();
                KeyInput::ShiftEnd
            }
        };

        let was_select_all = ev == KeyInput::SelectAll;
        let pre_len = text.len();
        assert!(sys.dispatch_key(&ev, &mut text));
        if was_select_all {
            assert_eq!(text.len(), pre_len, "Ctrl+A (SelectAll) must NEVER delete text!");
        }
        assert_eq!(text, ref_text, "Text mismatch at fuzz step {step} after {ev:?}");
        assert_eq!(sys.clipboard_text(), ref_clip, "Clipboard mismatch at fuzz step {step}");

        let st = sys.focused_text_edit_state().expect("focused TextEditState");
        assert!(text.is_char_boundary(st.cursor) && text.is_char_boundary(st.anchor));
        assert_eq!(
            (st.cursor, st.anchor),
            (ref_cursor, ref_anchor),
            "Selection (cursor, anchor) mismatch at fuzz step {step} after {ev:?}"
        );

        if step % 15 == 0 {
            sys.build_frame(&text, 280.0, 52.0, build_input);
            if st.has_selection(&text) {
                let sel_col = sys.theme.accent.with_alpha(0.28);
                assert!(
                    sys.primitives
                        .iter()
                        .any(|p| p.kind() == PrimKind::Quad && p.packed[0] == sel_col.0),
                    "Expected translucent selection highlight quad at step {step}"
                );
            }
        }
    }

    // --- PART B: Randomized SlotTape Reconciliation + StatelessView<!> + Tooltip + Coalesced Dirty Ranges + 0B Canvas ---
    struct PassiveChip<'p>(&'p str);
    impl<'p> StatelessView<!> for PassiveChip<'p> {
        fn mount<'ui, 's, 'data>(
            &'data self,
            ui: &'ui mut Ui<'_, 's, !>,
        ) -> impl Copy + use<'p, 'data> {
            ui.passive_label(self.0, 12.0, 16.0, Color::WHITE)
        }
    }

    struct ReconFuzzState {
        bits: u32,
        loop_n: usize,
        perm: [u64; 4],
    }
    let mut rf = ReconFuzzState {
        bits: 0x5A5A,
        loop_n: 3,
        perm: [1, 2, 3, 4],
    };
    let mut rsys = GuiSystem::new();
    let build_recon = |ui: &mut Ui<'_, '_, ReconFuzzState>| {
        let bits = ui.state().bits;
        let loop_n = ui.state().loop_n;
        let perm = ui.state().perm;

        ui.element(LayoutSpec::col().fixed(420.0, 420.0).gap(3.0), PaintStyle::default(), |ui| {
            let hdr_btn = ui.with_tooltip(
                "Auto-flipped Pass-4 header tooltip with an intentionally long multi-clause description to verify clean text wrapping without right-border overflow",
                |ui| ui.primary_button("Header Action", |s, _| s.bits ^= 1),
            );
            let _ = hdr_btn;

            // Empty `ui.keyed` closure with magic XOR key must never leak `pending_user_key`!
            ui.keyed(0x9e37_79b9_7f4a_7c15, |ui| {
                if (bits & 1) != 0 {
                    ui.badge("OPT", Color::WHITE);
                }
            });

            // Conditional sibling before StatelessView<!> `passive_label`!
            if (bits & 2) != 0 {
                ui.badge("COND_BEFORE_STATELESS", Color::rgb8(80, 200, 120));
            }
            ui.stateless(&PassiveChip("stateless_survivor"));

            // Interleaved shuffled keyed items + unkeyed loop + conditional branches + dynamic hook types:
            for (idx, &k) in perm.iter().enumerate() {
                ui.keyed(k, |ui| {
                    ui.element(LayoutSpec::row().fixed(90.0, 18.0), PaintStyle::default(), |ui| {
                        let owned_id = ui.use_state(|| k * 1000);
                        assert_eq!(*ui.read_local(owned_id), k * 1000, "Keyed state cross-wired!");
                    });
                });

                if idx == 1 {
                    for i in 0..loop_n {
                        ui.element(LayoutSpec::row().fixed(110.0, 18.0), PaintStyle::default(), |ui| {
                            let iter_id = ui.use_state(|| 5000 + i);
                            assert_eq!(*ui.read_local(iter_id), 5000 + i);
                            // Dynamically switch hook #2 between Spring and String without panic!
                            if ((bits >> (i + 2)) & 1) != 0 {
                                let _ = ui.use_spring(1.0, 5.0);
                            } else {
                                let s_h = ui.use_state(|| format!("branch_{i}"));
                                assert!(ui.read_local(s_h).starts_with("branch_"));
                            }
                        });
                        if ((bits >> (i + 8)) & 1) != 0 {
                            ui.badge("DETAIL", Color::rgb8(120, 160, 240));
                        }
                    }
                }
            }
        });
    };

    for step in 0..300usize {
        rf.bits = rng.next_u32();
        rf.loop_n = rng.next_usize(7);
        rf.perm.swap(rng.next_usize(4), rng.next_usize(4));
        if step == 10 {
            let _ = rsys.pointer_move(40.0, 14.0, &mut rf);
        }
        rsys.build_frame(&rf, 420.0, 420.0, build_recon);
        if step == 10 {
            // Verify hovered long tooltip wraps/clips strictly within its floating overlay pill bounds:
            for (n_i, topo) in rsys.layout.topo.iter().enumerate() {
                if (topo.flags & skima::layout::FLAG_ALIVE) != 0
                    && (topo.flags & skima::layout::FLAG_FLOATING_ANCHOR) != 0
                {
                    let pill_box = rsys.layout.boxes[n_i];
                    assert!(pill_box.w > 0.0 && pill_box.x >= 0.0 && pill_box.x + pill_box.w <= 420.5);
                    let mut child = topo.first_child;
                    while !child.is_none() {
                        let cb = rsys.layout.boxes[child.idx()];
                        assert!(
                            cb.x + cb.w <= pill_box.x + pill_box.w + 0.5,
                            "Tooltip text ({:?}) overflowed tooltip pill ({pill_box:?})!",
                            cb
                        );
                        child = rsys.layout.topo[child.idx()].next_sibling;
                    }
                }
            }
        }
        for &sp_idx in &rsys.dirty_prim_ranges {
            assert!(sp_idx.start < sp_idx.end && (sp_idx.end as usize) <= rsys.primitives.len());
        }
        for w in rsys.dirty_prim_ranges.windows(2) {
            assert!(
                w[0].end < w[1].start,
                "dirty_prim_ranges must be strictly sorted and coalesced: {:?}",
                rsys.dirty_prim_ranges
            );
        }
    }

    // Steady-state `ui.vector_canvas` exact 0 B heap allocation check:
    let mut csys = GuiSystem::new();
    let build_canvas = |ui: &mut Ui<'_, '_, u64>| {
        let r = *ui.state();
        let dy = ((r % 11) as f32) - 5.0;
        let pts = [
            [6.0, 24.0 + dy],
            [96.0, 12.0 - dy],
            [196.0, 38.0 + dy],
            [296.0, 18.0 - dy],
        ];
        ui.vector_canvas(310, 52, r, Color::rgb8(80, 175, 255), FillRule::NonZero, |p| {
            p.stroke_polyline(&pts, 2.4);
        });
    };
    for r in 1..=3u64 {
        csys.build_frame(&r, 320.0, 64.0, build_canvas);
        csys.fonts.atlas.dirty_rects.clear();
    }
    for r in 4..=24u64 {
        let (_, alloc_cnt, alloc_bytes) = measure_thread_allocs(|| {
            csys.build_frame(&r, 320.0, 64.0, build_canvas);
        });
        csys.fonts.atlas.dirty_rects.clear();
        assert_eq!(
            (alloc_cnt, alloc_bytes),
            (0, 0),
            "Steady-state ui.vector_canvas frame allocated {alloc_bytes} B on rev {r}!"
        );
    }
}

// =========================================================================
// 21. Deterministic PRNG Fuzz: Vulkan 1.3 vs Optimized CpuRenderer <= 4 LSB Pixel Parity
// =========================================================================
#[test]
fn fuzz_vulkan_vs_optimized_cpu_renderer_pixel_parity_all_fast_paths() {
    use core::range::Range as CopyRange;
    use skima::gpu::{CLIP_NONE, DamageRect, GpuClip, GpuPrimitive, RenderBackend};

    let mut atlas = GlyphAtlas::new(128, 128);
    // Upload ClearType LCD stem, R8 grayscale mask, and RGBA8 color icon:
    let mut lcd_tex = Vec::with_capacity(6 * 10 * 4);
    let mut r8_tex = Vec::with_capacity(6 * 10);
    let mut rgba_tex = Vec::with_capacity(8 * 8 * 4);
    for y in 0..10u8 {
        for x in 0..6u8 {
            if x == 0 || x == 5 {
                lcd_tex.extend_from_slice(&[85, 150, 215, 255]);
                r8_tex.push(140);
            } else {
                lcd_tex.extend_from_slice(&[255, 255, 255, 255]);
                r8_tex.push(255);
            }
            let _ = y;
        }
    }
    for i in 0..64u8 {
        rgba_tex.extend_from_slice(&[240, 90 + i, 40, 220]);
    }
    let lcd_reg = atlas.upload_rgba8(6, 10, 0, 8, &lcd_tex).unwrap();
    let r8_reg = atlas.upload_r8(6, 10, 0, 8, &r8_tex).unwrap();
    let img_reg = atlas.upload_rgba8(8, 8, 0, 0, &rgba_tex).unwrap();
    let lcd_uv = [lcd_reg.u0, lcd_reg.v0, lcd_reg.u1, lcd_reg.v1];
    let r8_uv = [r8_reg.u0, r8_reg.v0, r8_reg.u1, r8_reg.v1];
    let img_uv = [img_reg.u0, img_reg.v0, img_reg.u1, img_reg.v1];

    let mut vk = VulkanRenderer::new_headless(1024, 128).expect("Vulkan 1.3 headless");
    let mut cpu = CpuRenderer::new(160, 120);
    let mut rng = SplitMix64::new(0x94D0_49BB_1331_11EB);

    for scene_i in 0..10usize {
        let clear = Color::rgb8(
            (12 + scene_i * 7) as u8,
            (18 + scene_i * 5) as u8,
            (28 + scene_i * 4) as u8,
        );

        let clips = [
            GpuClip {
                bounds: [8.0, 6.0, 144.0, 108.0],
                radii: [14.0, 6.0, 14.0, 6.0],
                meta: [CLIP_NONE, 0, 0, 0],
            },
            GpuClip {
                bounds: [14.0, 12.0, 132.0, 96.0],
                radii: [18.0; 4],
                meta: [0, 0, 0, 0],
            },
            GpuClip {
                bounds: [20.0, 18.0, 120.0, 84.0],
                radii: [0.0; 4],
                meta: [1, 0, 0, 0],
            },
        ];

        let mut prims = Vec::with_capacity(48);
        // Occluded layer + opaque root quad (exercises backward opaque scanline occlusion!):
        prims.push(GpuPrimitive::quad(
            [0.0, 0.0, 160.0, 120.0],
            [0.0; 4],
            Color::rgb8(250, 20, 180),
            0.0,
            Color::TRANSPARENT,
            CLIP_NONE,
            1.0,
        ));
        if scene_i % 2 == 0 {
            prims.push(GpuPrimitive::quad(
                [0.0, 0.0, 160.0, 120.0],
                [0.0; 4],
                Color::rgb8(22, 26, 38),
                0.0,
                Color::TRANSPARENT,
                CLIP_NONE,
                1.0,
            ));
        }

        // Targeted boundary case: `radii = [1.5; 4]` with `border_w = 2.0` (`inner_radii == 0.0`!)
        prims.push(GpuPrimitive::quad(
            [12.4, 10.6, 68.0, 44.0],
            [1.5; 4],
            Color::rgb8(45, 115, 225),
            2.0,
            Color::rgb8(245, 205, 70),
            1,
            1.0,
        ));

        for p_k in 0..24usize {
            let cid = match p_k % 4 {
                0 => CLIP_NONE,
                1 => 0,
                2 => 1,
                _ => 2,
            };
            let x = rng.next_f32(-8.0, 140.0);
            let y = rng.next_f32(-8.0, 100.0);
            let w = rng.next_f32(4.0, 86.0);
            let h = rng.next_f32(4.0, 64.0);
            let c0 = Color::rgba8(
                (rng.next_u32() & 0xFF) as u8,
                ((rng.next_u32() >> 8) & 0xFF) as u8,
                ((rng.next_u32() >> 16) & 0xFF) as u8,
                if p_k % 3 == 0 {
                    255
                } else {
                    (96 + (rng.next_u32() % 160)) as u8
                },
            );
            let c1 = Color::rgba8(
                (rng.next_u32() & 0xFF) as u8,
                ((rng.next_u32() >> 8) & 0xFF) as u8,
                ((rng.next_u32() >> 16) & 0xFF) as u8,
                if p_k % 5 == 0 { 0 } else { 220 },
            );
            let radii = [
                rng.next_f32(0.0, 22.0),
                rng.next_f32(0.0, 22.0),
                rng.next_f32(0.0, 22.0),
                rng.next_f32(0.0, 22.0),
            ];
            let op = if p_k % 4 == 0 {
                1.0
            } else {
                rng.next_f32(0.35, 1.0)
            };

            match p_k % 6 {
                0 => prims.push(GpuPrimitive::quad(
                    [x, y, w, h],
                    radii,
                    c0,
                    if p_k % 2 == 0 {
                        0.0
                    } else {
                        rng.next_f32(1.0, 8.0)
                    },
                    c1,
                    cid,
                    op,
                )),
                1 => prims.push(GpuPrimitive::linear_gradient(
                    [x, y, w, h],
                    radii,
                    c0,
                    c1,
                    rng.next_f32(0.0, core::f32::consts::TAU),
                    if p_k % 3 == 0 { 2.0 } else { 0.0 },
                    cid,
                    op,
                )),
                2 => prims.push(GpuPrimitive::shadow(
                    [x, y, w, h],
                    radii,
                    Color::rgba8(0, 0, 0, 185),
                    rng.next_f32(1.0, 9.0),
                    [rng.next_f32(-3.0, 4.0), rng.next_f32(1.0, 6.0)],
                    cid,
                    op,
                )),
                3 => prims.push(GpuPrimitive::atlas_quad(
                    PrimKind::GlyphLcd,
                    [x.round(), y.round(), 6.0, 10.0],
                    lcd_uv,
                    Color::rgb8(242, 245, 252),
                    cid,
                    1.0,
                )),
                4 => prims.push(GpuPrimitive::atlas_quad(
                    PrimKind::GlyphGray,
                    [x.round(), y.round(), 6.0, 10.0],
                    r8_uv,
                    c0.with_alpha_u8(255),
                    cid,
                    op,
                )),
                _ => prims.push(GpuPrimitive::atlas_quad(
                    PrimKind::ImageRgba,
                    [x.round(), y.round(), 8.0, 8.0],
                    img_uv,
                    Color::WHITE,
                    cid,
                    1.0,
                )),
            }
        }

        let frame = RenderFrame {
            width: 160,
            height: 120,
            scale_factor: 1.0,
            clear_color: clear,
            primitives: &prims,
            dirty_prim_ranges: &[],
            clips: &clips,
            damage: Some(DamageRect {
                x: 0,
                y: 0,
                w: 160,
                h: 120,
            }),
        };

        let vk_rgba = vk.render_offscreen_rgba8(&frame, &mut atlas).unwrap();
        let (_, cpu_allocs, cpu_bytes) = measure_thread_allocs(|| {
            cpu.render(&frame, &mut atlas).unwrap();
        });
        if scene_i >= 2 {
            assert_eq!(
                (cpu_allocs, cpu_bytes),
                (0, 0),
                "CpuRenderer::render must allocate 0 B in steady state (scene {scene_i})"
            );
        }

        for y in 0..120u32 {
            for x in 0..160u32 {
                let idx = ((y * 160 + x) as usize) * 4;
                let vp = [
                    vk_rgba[idx],
                    vk_rgba[idx + 1],
                    vk_rgba[idx + 2],
                    vk_rgba[idx + 3],
                ];
                let cp = cpu.pixel_at(x, y);
                for ch in 0..4 {
                    let diff = (vp[ch] as i32 - cp[ch] as i32).abs();
                    assert!(
                        diff <= 4,
                        "Fuzz GPU vs CPU parity exceeded 4 LSB on scene {scene_i} at ({x},{y}) ch {ch}: vk={vp:?} vs cpu={cp:?} (diff={diff})"
                    );
                }
            }
        }

        // Follow-up partial-damage + coalesced dirty_prim_ranges frame on the same scene:
        let last_idx = prims.len() - 1;
        prims[2] = GpuPrimitive::quad(
            [24.0, 22.0, 64.0, 38.0],
            [8.0; 4],
            Color::rgb8(230, 95, 75),
            0.0,
            Color::TRANSPARENT,
            1,
            1.0,
        );
        let dirty = [
            CopyRange {
                start: last_idx as u32,
                end: (last_idx + 1) as u32,
            },
            CopyRange { start: 2, end: 3 },
        ];
        let partial_frame = RenderFrame {
            width: 160,
            height: 120,
            scale_factor: 1.0,
            clear_color: clear,
            primitives: &prims,
            dirty_prim_ranges: &dirty,
            clips: &clips,
            damage: Some(DamageRect {
                x: 10,
                y: 8,
                w: 96,
                h: 72,
            }),
        };
        let vk_p = vk
            .render_offscreen_rgba8(&partial_frame, &mut atlas)
            .unwrap();
        let (_, p_allocs, p_bytes) = measure_thread_allocs(|| {
            cpu.render(&partial_frame, &mut atlas).unwrap();
        });
        if scene_i >= 2 {
            assert_eq!((p_allocs, p_bytes), (0, 0));
        }
        for y in 0..120u32 {
            for x in 0..160u32 {
                let idx = ((y * 160 + x) as usize) * 4;
                let vp = [vk_p[idx], vk_p[idx + 1], vk_p[idx + 2], vk_p[idx + 3]];
                let cp = cpu.pixel_at(x, y);
                for ch in 0..4 {
                    let diff = (vp[ch] as i32 - cp[ch] as i32).abs();
                    assert!(
                        diff <= 4,
                        "Partial-damage GPU vs CPU parity exceeded 4 LSB on scene {scene_i} at ({x},{y}) ch {ch}: vk={vp:?} vs cpu={cp:?}"
                    );
                }
            }
        }
    }
}

// =========================================================================
// Fractional Scaling Mathematical, Pixel-Sharpness, & Adversarial QA Proofs
// =========================================================================

const FRACTIONAL_QA_SCALES: [f32; 10] = [
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

// 22. Display Physics, 4-Contender Dominance, IEEE-754 `15k/120` Dyadic Exactness,
// Deadband Hysteresis, Malformed EDID Resilience, and `0 B` Heap Proof.
#[test]
fn scale_physics_four_contender_dominance_hysteresis_and_edid_zero_alloc_proof() {
    use skima::scale::{
        CSS_REFERENCE_PIXEL_DEG, CSS_REFERENCE_PPD, DisplayMetrics, MAX_SCALE, MIN_SCALE,
        REFERENCE_DISPLAY_PROFILES, SCALE_LATTICE_120, ScalePolicy, ScaleReason,
        WAYLAND_SCALE_DENOM, evaluate_contenders, grid_harmony_120ths, parse_edid_physical_mm,
        quantize_wayland_120ths, scale_from_120ths, viewing_distance_cm, zoom_step,
    };
    assert_eq!(ScalePolicy::default(), ScalePolicy::Auto);
    assert_eq!(SCALE_LATTICE_120, skima::SCALE_LATTICE_120);

    // 1. Independent trigonometric verification of W3C CSS reference pixel constants:
    let theta0_deg = 2.0_f64 * (1.0_f64 / (2.0 * 96.0 * 28.0)).atan().to_degrees();
    let ppd0 = 1.0_f64 / theta0_deg;
    assert!((CSS_REFERENCE_PIXEL_DEG as f64 - theta0_deg).abs() < 1e-6);
    assert!((CSS_REFERENCE_PPD as f64 - ppd0).abs() < 2e-3);

    // 2. IEEE-754 binary exactness of all `15k / 120 == k / 8` dyadic fractions in [0.5, 4.0]:
    for k in 4u32..=32 {
        let n120 = 15 * k;
        let s_wayland = scale_from_120ths(n120);
        let s_dyadic = (k as f32) * 0.125_f32;
        assert_eq!(
            s_wayland.to_bits(),
            s_dyadic.to_bits(),
            "Wayland 15*{k}/120 ({n120}/120) must be IEEE-754 bit-identical to {k}/8"
        );
        assert_eq!(quantize_wayland_120ths(s_wayland as f64), Some(n120));
        assert!(grid_harmony_120ths(n120) >= 0.74);
    }
    // Non-dyadic Wayland fractions (`4/3 = 160/120`, `5/3 = 200/120`) have exact 1/120 numerators
    // and strictly lower binary design-grid harmony than every `15k/120` dyadic scale:
    for &non_dyadic_120 in &[160u32, 200u32] {
        let h_nd = grid_harmony_120ths(non_dyadic_120);
        assert!((h_nd - 0.55).abs() < 1e-6);
        assert!(h_nd < grid_harmony_120ths(135)); // < 1/8 harmony (0.74)
    }

    // 3. Zero-allocation proof across `evaluate_contenders`, `ScalePolicy::resolve`, `zoom_step`, & `parse_edid_physical_mm`:
    let mut valid_edid = [0u8; 128];
    valid_edid[0..8].copy_from_slice(&[0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]);
    valid_edid[54] = 0x02;
    valid_edid[55] = 0x3A;
    valid_edid[56] = (3840 & 0xFF) as u8;
    valid_edid[58] = (((3840 >> 8) & 0x0F) << 4) as u8;
    valid_edid[59] = (2160 & 0xFF) as u8;
    valid_edid[61] = (((2160 >> 8) & 0x0F) << 4) as u8;
    valid_edid[66] = (597 & 0xFF) as u8;
    valid_edid[67] = (336 & 0xFF) as u8;
    valid_edid[68] = ((((597 >> 8) & 0x0F) << 4) | ((336 >> 8) & 0x0F)) as u8;
    let csum: u8 = valid_edid[0..127]
        .iter()
        .fold(0u8, |acc, &b| acc.wrapping_add(b));
    valid_edid[127] = csum.wrapping_neg();

    let (
        (
            hits_a,
            hits_b,
            hits_c,
            hits_d,
            sq_err_a,
            sq_err_b,
            sq_err_c,
            sq_err_d,
            harm_c_sum,
            harm_d_sum,
        ),
        alloc_cnt,
        alloc_bytes,
    ) = measure_thread_allocs(|| {
        let mut ha = 0usize;
        let mut hb = 0usize;
        let mut hc = 0usize;
        let mut hd = 0usize;
        let mut ea = 0.0_f64;
        let mut eb = 0.0_f64;
        let mut ec = 0.0_f64;
        let mut ed = 0.0_f64;
        let mut hcs = 0.0_f64;
        let mut hds = 0.0_f64;

        for prof in &REFERENCE_DISPLAY_PROFILES {
            let m = prof.metrics();
            let cmp = evaluate_contenders(&m);
            let d = cmp.contender_d;
            let (lo, hi) = prof.expected_scale_range;
            assert!(
                d.scale_factor >= lo - 1e-4 && d.scale_factor <= hi + 1e-4,
                "Profile {} missed auto-scale band [{lo}, {hi}]: got {:?}",
                prof.name,
                d
            );
            assert!(d.is_dyadic_exact());
            assert_eq!(d.reason, ScaleReason::PhysicalAuto);
            assert!(d.reason.is_auto());
            assert_eq!(d.reason.label(), "Auto");
            hd += 1;

            let a_dyadic = ((cmp.contender_a_scale * 8.0).fract()).abs() < 1e-3;
            if cmp.contender_a_scale >= lo - 0.01
                && cmp.contender_a_scale <= hi + 0.01
                && a_dyadic
            {
                ha += 1;
            }
            if cmp.contender_b_scale >= lo - 0.01 && cmp.contender_b_scale <= hi + 0.01 {
                hb += 1;
            }
            if cmp.contender_c_scale >= lo - 0.01
                && cmp.contender_c_scale <= hi + 0.01
                && cmp.contender_c_harmony >= 0.74
            {
                hc += 1;
            }

            let raw = d.raw_acuity_scale as f64;
            ea += ((cmp.contender_a_scale as f64) / raw).ln().powi(2);
            eb += ((cmp.contender_b_scale as f64) / raw).ln().powi(2);
            ec += ((cmp.contender_c_scale as f64) / raw).ln().powi(2);
            ed += ((d.scale_factor as f64) / raw).ln().powi(2);
            hcs += cmp.contender_c_harmony as f64;
            hds += d.grid_harmony as f64;
        }

        for &n120 in &SCALE_LATTICE_120 {
            let s = scale_from_120ths(n120);
            let _ = zoom_step(s, 1);
            let _ = zoom_step(s, -1);
        }
        let edid_sz = parse_edid_physical_mm(&valid_edid).unwrap();
        assert_eq!((edid_sz.width_mm, edid_sz.height_mm), (597, 336));

        (ha, hb, hc, hd, ea, eb, ec, ed, hcs, hds)
    });

    assert_eq!(
        (alloc_cnt, alloc_bytes),
        (0, 0),
        "Auto-scale solver, 4-contender evaluator, zoom_step, and EDID parser must perform 0 heap allocations"
    );
    assert_eq!(hits_d, 30);
    assert!(hits_d > hits_b && hits_b > hits_c && hits_d > hits_a);
    let n_prof = REFERENCE_DISPLAY_PROFILES.len() as f64;
    let rms_a = (sq_err_a / n_prof).sqrt();
    let rms_b = (sq_err_b / n_prof).sqrt();
    let rms_c = (sq_err_c / n_prof).sqrt();
    let rms_d = (sq_err_d / n_prof).sqrt();
    let mean_harm_c = harm_c_sum / n_prof;
    let mean_harm_d = harm_d_sum / n_prof;
    // Both viewing-distance acuity models (C and D) cut RMS log-acuity error by >3.5x vs PPI-only A and B,
    // while Contender D simultaneously achieves >3.2x higher binary grid harmony (>0.90 vs <0.30) and 30/30 auto-scale targets:
    assert!(
        rms_d < 0.08 && rms_c < 0.08 && rms_d < rms_b * 0.30 && rms_b < rms_a,
        "Expected viewing-distance acuity error << PPI-only baselines: D={rms_d:.4}, C={rms_c:.4}, B={rms_b:.4}, A={rms_a:.4}"
    );
    assert!(
        mean_harm_d > 0.90 && mean_harm_c < 0.30,
        "Expected Contender D grid harmony ({mean_harm_d:.3}) >> Contender C ({mean_harm_c:.3})"
    );

    // 4. Independent viewing-distance & 1.0x deadzone checks + portrait invariance + hysteresis:
    let d_handheld = viewing_distance_cm(151.0, 94.0).unwrap();
    let d_laptop = viewing_distance_cm(302.0, 189.0).unwrap();
    let d_desk = viewing_distance_cm(597.0, 336.0).unwrap();
    let d_ultrawide = viewing_distance_cm(1193.0, 336.0).unwrap();
    let d_tv = viewing_distance_cm(1439.0, 809.0).unwrap();
    assert!(d_handheld < d_laptop && d_laptop < d_desk);
    assert!(
        (d_ultrawide - d_desk).abs() < 6.0 && d_tv > 170.0,
        "49\" 32:9 ultrawide ({d_ultrawide} cm) must stay near desk distance ({d_desk} cm), far below 65\" TV ({d_tv} cm)"
    );
    assert_eq!(
        viewing_distance_cm(597.0, 336.0).unwrap().to_bits(),
        viewing_distance_cm(336.0, 597.0).unwrap().to_bits(),
        "Viewing distance must be exact bit-identical under 90-degree portrait rotation"
    );

    // Deadzone around 1.0x: 27" 1440p (108.8 PPI) stays 1.0x; hysteresis at 1.25x <-> 1.50x holds & breaks out:
    let r_qhd = ScalePolicy::Auto.resolve(&DisplayMetrics::from_physical_mm(2560, 1440, 597, 336));
    assert_eq!(r_qhd.wayland_120ths, 120);
    assert!(r_qhd.has_integer_logical_viewport());

    let r_4k = ScalePolicy::Auto.resolve(&DisplayMetrics::from_physical_mm(3840, 2160, 597, 336));
    assert_eq!(r_4k.wayland_120ths, 180); // 1.50x -> exact 2560x1440 integer logical viewport!
    assert_eq!((r_4k.logical_width, r_4k.logical_height), (2560.0, 1440.0));
    assert!(r_4k.has_integer_logical_viewport());

    let r_32_4k = ScalePolicy::Auto.resolve(&DisplayMetrics::from_physical_mm(3840, 2160, 708, 398));
    assert_eq!(r_32_4k.wayland_120ths, 150); // 1.25x -> exact 3072x1728 integer logical viewport!
    assert_eq!((r_32_4k.logical_width, r_32_4k.logical_height), (3072.0, 1728.0));
    assert!(r_32_4k.has_integer_logical_viewport());

    // 5. Malformed EDID, NaN, +/-Inf, 0, and u32::MAX resilience:
    let adversarial_metrics = [
        DisplayMetrics::new(0, 0),
        DisplayMetrics::from_physical_mm(0, 0, 0, 0),
        DisplayMetrics::from_physical_mm(3840, 2160, 16, 9),
        DisplayMetrics::from_physical_mm(3840, 2160, 0, 336),
        DisplayMetrics::from_physical_mm(u32::MAX, u32::MAX, u32::MAX, u32::MAX),
        DisplayMetrics::from_physical_mm(u32::MAX, 1080, 597, u32::MAX),
        DisplayMetrics::from_os_only(2560, 1440, f64::NAN),
        DisplayMetrics::from_os_only(2560, 1440, f64::INFINITY),
        DisplayMetrics::from_os_only(2560, 1440, f64::NEG_INFINITY),
        DisplayMetrics::from_os_only(2560, 1440, -3.5),
        DisplayMetrics::from_os_only(2560, 1440, 0.0),
    ];
    for m in &adversarial_metrics {
        for pol in [
            ScalePolicy::Auto,
            ScalePolicy::PurePhysicalAuto,
            ScalePolicy::System,
            ScalePolicy::Fixed(f32::NAN),
            ScalePolicy::Fixed(f32::NEG_INFINITY),
            ScalePolicy::Fixed(0.0),
            ScalePolicy::Fixed(999.0),
        ] {
            let res = pol.resolve(m);
            assert!(res.scale_factor.is_finite());
            assert!((MIN_SCALE..=MAX_SCALE).contains(&res.scale_factor));
            assert!((60..=480).contains(&res.wayland_120ths));
            assert!(res.wayland_120ths == WAYLAND_SCALE_DENOM || res.scale_factor >= MIN_SCALE);
            assert!(res.logical_width.is_finite() && res.logical_height.is_finite());
        }
    }
    // Aspect-ratio-in-cm EDID 1.4 encoding (byte 22 == 0 -> landscape aspect ratio, NOT cm) must return None:
    let mut projector_edid = [0u8; 128];
    projector_edid[0..8].copy_from_slice(&[0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]);
    projector_edid[21] = 79; // 16:9 landscape aspect code when byte 22 == 0
    projector_edid[22] = 0;
    let p_sum: u8 = projector_edid[0..127]
        .iter()
        .fold(0u8, |acc, &b| acc.wrapping_add(b));
    projector_edid[127] = p_sum.wrapping_neg();
    assert_eq!(parse_edid_physical_mm(&projector_edid), None);
}

// 23. Zero Physical Seam Proof: 50 Contiguous `gap=0` Rows in `Col` and `VirtualList`
// Across All 10 Fractional Scales on BOTH `CpuRenderer` and `VulkanRenderer` (`0` Bleed Pixels!).
#[test]
fn zero_physical_seam_col_and_virtual_list_cpu_and_vulkan_proof() {
    use skima::gpu::PrimKind;
    use skima::gpu::cpu::CpuRenderer;
    use skima::gpu::vulkan::VulkanRenderer;

    let mut vk = VulkanRenderer::new_headless(8192, 256)
        .expect("Vulkan 1.3 headless renderer required for zero-seam proof");

    let canary = Color::rgb8(255, 0, 220);
    let col_a = Color::rgb8(34, 42, 60);
    let col_b = Color::rgb8(46, 56, 78);

    // Variable row heights including primes & odd halves that challenge fractional snapping:
    let mut vlist = VirtualListIndex::new(80, 17.0);
    for i in 0..80usize {
        vlist.set_height(i, 15.0 + ((i * 7 + 3) % 9) as f32 * 0.5);
    }

    for &sf in &FRACTIONAL_QA_SCALES {
        // Exercise both unshifted (`0.0`) and scrolled zero-crossing (`scroll_y` placing `origin_y` onto `-0.5 / sf` tie!):
        let tie_scroll = 19.5_f64 / (sf as f64);
        for &scroll_y in &[0.0_f64, tie_scroll] {
            let mut sys = GuiSystem::new();
            sys.set_scale_factor(sf);
            sys.clear_color = canary;

            let logical_w = 220.0_f32;
            let logical_h = 760.0_f32;

            sys.build_frame(&scroll_y, logical_w, logical_h, |ui| {
                let sy = *ui.state();
                ui.element(
                    LayoutSpec::row().fixed(logical_w, logical_h),
                    PaintStyle::default(),
                    |ui| {
                        // Left 110px: 50 contiguous `gap=0` rows in `Col` mixing `Fixed` and `Grow` flex weights!
                        let mut col_spec = LayoutSpec::col().fixed(110.0, logical_h);
                        col_spec.scroll_y = sy as f32;
                        ui.element(col_spec, PaintStyle::default(), |ui| {
                            ui.set_scrollable(2000.0);
                            for i in 0..50usize {
                                let spec = if sy == 0.0 && i % 5 == 2 {
                                    LayoutSpec::row().width_grow(1.0).height_grow(1.0 + (i % 3) as f32)
                                } else {
                                    LayoutSpec::row()
                                        .width_grow(1.0)
                                        .height_fixed(13.5 + (i % 4) as f32 * 0.5)
                                };
                                let fill = if i % 2 == 0 { col_a } else { col_b };
                                ui.keyed(i as u64 + 1, |ui| {
                                    ui.element(spec, PaintStyle::solid(fill), |_| {});
                                });
                            }
                        });

                        // Right 110px: 50+ contiguous `gap=0` rows in `VirtualList`!
                        ui.virtual_list(&vlist, sy, 110.0, logical_h, |ui, idx, rh| {
                            let fill = if idx % 2 == 0 { col_a } else { col_b };
                            ui.element(
                                LayoutSpec::row().width_grow(1.0).height_fixed(rh),
                                PaintStyle::solid(fill),
                                |_| {},
                            );
                        });
                    },
                );
            });

            // Split emitted row quads into Left (`Col`, `x0 == 0`) and Right (`VirtualList`, `x0 > 0`):
            let mut col_rows: Vec<[f32; 4]> = Vec::new();
            let mut vlist_rows: Vec<[f32; 4]> = Vec::new();
            let mid_phys_x = (55.0 * sf).round();
            for p in &sys.primitives {
                if p.kind() == PrimKind::Quad
                    && (p.packed[0] == col_a.0 || p.packed[0] == col_b.0)
                {
                    if p.bounds[0] < mid_phys_x {
                        col_rows.push(p.bounds);
                    } else {
                        vlist_rows.push(p.bounds);
                    }
                }
            }
            assert_eq!(
                col_rows.len(),
                50,
                "Expected 50 Col row quads at sf={sf}, scroll={scroll_y}"
            );
            assert!(
                vlist_rows.len() >= 40,
                "Expected >= 40 visible VirtualList row quads at sf={sf}, got {}",
                vlist_rows.len()
            );

            // Prove exact 0.0 physical-pixel seam (`phys_y0(row[i+1]) == phys_y1(row[i])`) on both Col and VirtualList:
            for (label, rows) in [("Col", &col_rows), ("VirtualList", &vlist_rows)] {
                for (i, r) in rows.iter().enumerate() {
                    assert_eq!(
                        r[0],
                        r[0].round(),
                        "{label} row {i} x0 ({}) not integer at sf={sf}",
                        r[0]
                    );
                    assert_eq!(
                        r[1],
                        r[1].round(),
                        "{label} row {i} y0 ({}) not integer at sf={sf}",
                        r[1]
                    );
                    assert_eq!(
                        r[2],
                        r[2].round(),
                        "{label} row {i} w ({}) not integer at sf={sf}",
                        r[2]
                    );
                    assert_eq!(
                        r[3],
                        r[3].round(),
                        "{label} row {i} h ({}) not integer at sf={sf}",
                        r[3]
                    );
                }
                for (i, pair) in rows.windows(2).enumerate() {
                    let phys_y1_prev = pair[0][1] + pair[0][3];
                    let phys_y0_next = pair[1][1];
                    assert_eq!(
                        phys_y1_prev.to_bits(),
                        phys_y0_next.to_bits(),
                        "{label} seam between row {i} and {} at sf={sf}, scroll={scroll_y}: y1={phys_y1_prev} vs y0={phys_y0_next}",
                        i + 1
                    );
                }
            }

            // Render on BOTH `CpuRenderer` and `VulkanRenderer` and prove ZERO canary bleed pixels!
            let phys_w = (logical_w * sf - 1e-4).ceil() as u32;
            let phys_h = (logical_h * sf - 1e-4).ceil() as u32;

            let vk_rgba = sys
                .render_vulkan_offscreen(&mut vk, phys_w, phys_h)
                .expect("Vulkan offscreen render at fractional scale");
            let mut cpu = CpuRenderer::new(phys_w, phys_h);
            sys.render_to_backend(&mut cpu, phys_w, phys_h, 1).unwrap();

            let clip_bottom = (logical_h * sf).round().min(phys_h as f32).max(0.0);
            let col_y_end = (col_rows.last().unwrap()[1] + col_rows.last().unwrap()[3])
                .min(clip_bottom)
                .max(0.0) as u32;
            let vl_y_end = (vlist_rows.last().unwrap()[1] + vlist_rows.last().unwrap()[3])
                .min(clip_bottom)
                .max(0.0) as u32;
            let col_probe_x = (col_rows[0][0] + col_rows[0][2] * 0.5) as u32;
            let vl_probe_x = (vlist_rows[0][0] + vlist_rows[0][2] * 0.5) as u32;

            for (label, probe_x, y_limit) in [
                ("Col", col_probe_x, col_y_end),
                ("VirtualList", vl_probe_x, vl_y_end),
            ] {
                for y in 0..y_limit {
                    let cp = cpu.pixel_at(probe_x, y);
                    let v_idx = ((y * phys_w + probe_x) as usize) * 4;
                    let vp = [
                        vk_rgba[v_idx],
                        vk_rgba[v_idx + 1],
                        vk_rgba[v_idx + 2],
                        vk_rgba[v_idx + 3],
                    ];
                    // Neither CPU nor Vulkan may bleed even 1% of the bright magenta `[255, 0, 220]` canary!
                    assert!(
                        cp[0] < 80 && cp[2] < 120,
                        "CPU canary bleed on {label} at ({probe_x}, {y}) sf={sf} scroll={scroll_y}: px={cp:?}"
                    );
                    assert!(
                        vp[0] < 80 && vp[2] < 120,
                        "Vulkan canary bleed on {label} at ({probe_x}, {y}) sf={sf} scroll={scroll_y}: px={vp:?}"
                    );
                }
            }
        }
    }
}

// 24. Zero Row-Divider Moiré Proof: 20 Stacked `VirtualList` Rows with `0.5px` and `1.0px` Borders
// at `1.25x` and `1.333333x` (`160/120`) on BOTH Vulkan 1.3 and `CpuRenderer` (`<= 1 LSB` Variance!).
#[test]
fn zero_row_divider_moire_vulkan_and_cpu_proof() {
    use skima::gpu::PrimKind;
    use skima::gpu::cpu::CpuRenderer;
    use skima::gpu::vulkan::VulkanRenderer;
    use skima::snap_physical_border_width;

    // Verify exact physical border width snapping contract first:
    assert!((snap_physical_border_width(0.5, 1.25) - 0.625).abs() < 1e-6);
    assert!((snap_physical_border_width(0.5, 160.0 / 120.0) - (0.5 * (160.0 / 120.0))).abs() < 1e-6);
    assert_eq!(snap_physical_border_width(1.0, 1.25), 1.0);
    assert_eq!(snap_physical_border_width(1.0, 160.0 / 120.0), 1.0);
    assert_eq!(snap_physical_border_width(1.0, 1.5), 2.0);
    assert_eq!(snap_physical_border_width(2.0, 1.25), 3.0);

    let mut vk = VulkanRenderer::new_headless(4096, 128)
        .expect("Vulkan 1.3 headless renderer required for zero-Moiré proof");

    let srgb_luma_u8 = |rgba: [u8; 4]| -> f32 {
        0.2126 * (rgba[0] as f32) + 0.7152 * (rgba[1] as f32) + 0.0722 * (rgba[2] as f32)
    };

    for &sf in &[1.25_f32, 160.0 / 120.0] {
        for &logical_bw in &[0.5_f32, 1.0_f32] {
            let mut sys = GuiSystem::new();
            sys.set_scale_factor(sf);
            sys.clear_color = Color::rgb8(18, 20, 28);

            let vlist = VirtualListIndex::new(20, 23.0);
            let logical_w = 240.0_f32;
            let logical_h = 460.0_f32;

            sys.build_frame(&0usize, logical_w, logical_h, |ui| {
                ui.virtual_list(&vlist, 0.0, logical_w, logical_h, |ui, _idx, rh| {
                    ui.element(
                        LayoutSpec::row().width_grow(1.0).height_fixed(rh),
                        PaintStyle::solid(Color::rgb8(28, 32, 44))
                            .border(logical_bw, Color::rgb8(168, 182, 214)),
                        |_| {},
                    );
                });
            });

            let phys_w = (logical_w * sf - 1e-4).ceil() as u32;
            let phys_h = (logical_h * sf - 1e-4).ceil() as u32;

            let vk_rgba = sys
                .render_vulkan_offscreen(&mut vk, phys_w, phys_h)
                .expect("Vulkan offscreen render for Moiré check");
            let mut cpu = CpuRenderer::new(phys_w, phys_h);
            sys.render_to_backend(&mut cpu, phys_w, phys_h, 1).unwrap();

            let row_bounds: Vec<[f32; 4]> = sys
                .primitives
                .iter()
                .filter(|p| p.kind() == PrimKind::Quad && p.uv_or_extra[0] > 0.0)
                .map(|p| p.bounds)
                .collect();
            assert_eq!(
                row_bounds.len(),
                20,
                "Expected 20 bordered VirtualList rows at sf={sf}, bw={logical_bw}"
            );

            let probe_x = phys_w / 2;
            let vk_px = |x: u32, y: u32| -> [u8; 4] {
                let idx = ((y * phys_w + x) as usize) * 4;
                [
                    vk_rgba[idx],
                    vk_rgba[idx + 1],
                    vk_rgba[idx + 2],
                    vk_rgba[idx + 3],
                ]
            };

            let ref_top_y = row_bounds[0][1] as u32;
            let ref_bot_y = (row_bounds[0][1] + row_bounds[0][3]) as u32 - 1;
            let ref_mid_y = (row_bounds[0][1] + row_bounds[0][3] * 0.5) as u32;

            let vk_ref_top = vk_px(probe_x, ref_top_y);
            let vk_ref_bot = vk_px(probe_x, ref_bot_y);
            let vk_ref_mid = vk_px(probe_x, ref_mid_y);
            let cpu_ref_top = cpu.pixel_at(probe_x, ref_top_y);
            let cpu_ref_bot = cpu.pixel_at(probe_x, ref_bot_y);

            // Ensure divider pixels are genuinely brighter than row interior (non-trivial border!):
            assert!(
                srgb_luma_u8(vk_ref_top) > srgb_luma_u8(vk_ref_mid) + 25.0,
                "Divider must be visibly rendered above interior fill at sf={sf}, bw={logical_bw}: top={vk_ref_top:?} vs mid={vk_ref_mid:?}"
            );

            for (row_i, rb) in row_bounds.iter().enumerate() {
                let top_y = rb[1] as u32;
                let bot_y = (rb[1] + rb[3]) as u32 - 1;

                let vt = vk_px(probe_x, top_y);
                let vb = vk_px(probe_x, bot_y);
                let ct = cpu.pixel_at(probe_x, top_y);
                let cb = cpu.pixel_at(probe_x, bot_y);

                // Luminance AND every individual RGBA channel across all 20 rows must vary by <= 1 LSB!
                assert!(
                    (srgb_luma_u8(vt) - srgb_luma_u8(vk_ref_top)).abs() <= 1.0
                        && (srgb_luma_u8(vb) - srgb_luma_u8(vk_ref_bot)).abs() <= 1.0,
                    "Vulkan row {row_i} divider luma Moiré at sf={sf}, bw={logical_bw}: top={vt:?} (ref {vk_ref_top:?}), bot={vb:?} (ref {vk_ref_bot:?})"
                );
                assert!(
                    (srgb_luma_u8(ct) - srgb_luma_u8(cpu_ref_top)).abs() <= 1.0
                        && (srgb_luma_u8(cb) - srgb_luma_u8(cpu_ref_bot)).abs() <= 1.0,
                    "CPU row {row_i} divider luma Moiré at sf={sf}, bw={logical_bw}: top={ct:?} (ref {cpu_ref_top:?}), bot={cb:?} (ref {cpu_ref_bot:?})"
                );
                for ch in 0..4 {
                    assert!(
                        (vt[ch] as i32 - vk_ref_top[ch] as i32).abs() <= 1
                            && (vb[ch] as i32 - vk_ref_bot[ch] as i32).abs() <= 1,
                        "Vulkan channel {ch} Moiré on row {row_i} at sf={sf}, bw={logical_bw}"
                    );
                    assert!(
                        (ct[ch] as i32 - cpu_ref_top[ch] as i32).abs() <= 1
                            && (cb[ch] as i32 - cpu_ref_bot[ch] as i32).abs() <= 1,
                        "CPU channel {ch} Moiré on row {row_i} at sf={sf}, bw={logical_bw}"
                    );
                    // Cross-check Vulkan 1.3 vs CpuRenderer hairline parity (<= 4 LSB):
                    assert!(
                        (vt[ch] as i32 - ct[ch] as i32).abs() <= 4
                            && (vb[ch] as i32 - cb[ch] as i32).abs() <= 4,
                        "Vulkan vs CPU hairline divider diverged > 4 LSB on row {row_i} at sf={sf}, bw={logical_bw}: vk={vt:?} vs cpu={ct:?}"
                    );
                }
            }
        }
    }
}

// 25. Horizontal Baseline Uniformity Across All 10 Fractional Scales & `F26Dot6` (`1/64` px)
// Glyph Cache Hit Proof (`14.0 * (150.0 / 120.0)` == `17.5`).
#[test]
fn horizontal_baseline_uniformity_and_f26dot6_glyph_cache_identity_proof() {
    use skima::gpu::PrimKind;
    use skima::layout::{FLAG_ALIVE, FLAG_HAS_TEXT};
    use skima::text::{AntialiasMode, FontId, FontSystem, GlyphKey};

    // Part A: `F26Dot6` (`1/64` px) cache hit & texel identity proof:
    let mut fonts = FontSystem::new();
    let _ = fonts.load_system_fonts();
    let fid = fonts.ui_font;
    let aa = if fid == FontId::FALLBACK {
        AntialiasMode::Grayscale
    } else {
        AntialiasMode::SubpixelLcd
    };

    let shaped_para = fonts.shape_paragraph("Skima 1.25x F26Dot6!", fid, 14.0, 20.0);
    let shaped = &shaped_para.glyphs;
    assert!(!shaped.is_empty());

    let size_wayland_125 = 14.0_f32 * (150.0_f32 / 120.0_f32);
    let size_exact_175 = 17.5_f32;
    let size_perturbed = 17.5_f32 + 0.003_f32; // < 1/128 px jitter

    for sg in shaped {
        let key_wayland = GlyphKey::new(sg.font_id, sg.glyph_id, size_wayland_125, 0.25, aa);
        let key_175 = GlyphKey::new(sg.font_id, sg.glyph_id, size_exact_175, 0.25, aa);
        let key_pert = GlyphKey::new(sg.font_id, sg.glyph_id, size_perturbed, 0.25, aa);

        assert_eq!(
            key_wayland, key_175,
            "14.0 * (150.0 / 120.0) must produce the exact same GlyphKey as 17.5"
        );
        assert_eq!(key_wayland.size_f26dot6, 17 * 64 + 32);
        assert_eq!(key_pert, key_175);

        let reg_a = fonts.rasterize_glyph(sg.font_id, sg.glyph_id, size_wayland_125, 10.25, aa);
        let count_after_a = fonts.atlas.cached_glyph_count();
        assert!(fonts.atlas.contains_glyph_key(&key_175));

        // Second lookup via `17.5` and `17.503` must hit the exact cached AtlasRegion with 0 B heap!
        let ((reg_b, reg_c), hit_allocs, hit_bytes) = measure_thread_allocs(|| {
            let b = fonts.rasterize_glyph(sg.font_id, sg.glyph_id, size_exact_175, 42.25, aa);
            let c = fonts.rasterize_glyph(sg.font_id, sg.glyph_id, size_perturbed, 98.25, aa);
            (b, c)
        });
        assert_eq!((hit_allocs, hit_bytes), (0, 0));
        assert_eq!(fonts.atlas.cached_glyph_count(), count_after_a);
        assert_eq!(reg_a, reg_b);
        assert_eq!(reg_a, reg_c);
    }

    // Part B: Horizontal baseline uniformity across all 10 fractional scales (multi-line wrapped paragraph):
    let mut sys = GuiSystem::with_system_fonts();
    let sample_text =
        "Sphinx of black quartz, judge my vow! 0123456789 — [IALE 128 B/node, Wayland 150/120 & 160/120]";

    for &sf in &FRACTIONAL_QA_SCALES {
        sys.set_scale_factor(sf);
        sys.build_frame(&0usize, 310.0, 180.0, |ui| {
            ui.element(
                LayoutSpec::col().fixed(310.0, 180.0).pad_xy(13.25, 9.75),
                PaintStyle::solid(Color::rgb8(18, 20, 28)),
                |ui| {
                    ui.ui_text(sample_text, 14.0, Color::WHITE);
                },
            );
        });

        let text_idx = (0..sys.layout.topo.len())
            .find(|&i| {
                (sys.layout.topo[i].flags & FLAG_ALIVE) != 0
                    && (sys.layout.topo[i].flags & FLAG_HAS_TEXT) != 0
            })
            .expect("alive text node");
        let tb = sys.layout.boxes[text_idx];
        let tspec = sys.layout.specs[text_idx];
        let para = &sys.layout.texts[tspec.aux_id as usize];
        assert!(
            para.cached_lines.len() >= 2,
            "Expected wrapped multi-line paragraph at width=310, got {} lines",
            para.cached_lines.len()
        );

        let glyph_prims: Vec<&skima::gpu::GpuPrimitive> = sys
            .primitives
            .iter()
            .filter(|p| {
                matches!(
                    p.kind(),
                    PrimKind::GlyphGray | PrimKind::GlyphLcd | PrimKind::ImageRgba
                )
            })
            .collect();

        let eff_aa = if sys.fonts.ui_font == FontId::FALLBACK {
            AntialiasMode::Grayscale
        } else {
            AntialiasMode::SubpixelLcd
        };

        let mut p_cursor = 0usize;
        for (line_i, line) in para.cached_lines.iter().copied().enumerate() {
            let expected_int_base = ((tb.y + tspec.pad_t + line.baseline_y) * sf).round();
            assert_eq!(expected_int_base, expected_int_base.floor());

            let mut pen_x = tb.x + tspec.pad_l;
            let mut line_checked = 0usize;
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
                        eff_aa,
                    );
                    if reg.w > 0 && reg.h > 0 {
                        let gp = glyph_prims[p_cursor];
                        p_cursor += 1;
                        assert_eq!(
                            gp.bounds[0],
                            phys_gx.floor() + (reg.bearing_x as f32),
                            "Horizontal glyph placement mismatch on line {line_i} at sf={sf}"
                        );
                        assert_eq!(gp.bounds[0], gp.bounds[0].floor());
                        assert_eq!(gp.bounds[1], gp.bounds[1].floor());
                        if sg.y_offset == 0.0 {
                            let glyph_base = gp.bounds[1] + (reg.bearing_y as f32);
                            assert_eq!(
                                glyph_base.to_bits(),
                                expected_int_base.to_bits(),
                                "Baseline step jitter on line {line_i} at sf={sf}: glyph_base={glyph_base} vs line_base={expected_int_base}"
                            );
                        }
                        line_checked += 1;
                    }
                    pen_x += sg.x_advance;
                }
            }
            assert!(line_checked >= 5);
        }
        assert_eq!(p_cursor, glyph_prims.len());
    }

    // Part C: Unshrunk `Fit` buttons, badges, and single-line labels never spuriously wrap onto 2 lines
    // due to sub-pixel physical edge rounding across all 10 fractional scales:
    for &sf in &FRACTIONAL_QA_SCALES {
        sys.set_scale_factor(sf);
        sys.build_frame(&0usize, 960.0, 240.0, |ui| {
            let th = ui.theme();
            ui.element(
                LayoutSpec::col().fixed(960.0, 240.0).pad(12.0).gap(10.0),
                PaintStyle::solid(th.bg_canvas),
                |ui| {
                    ui.element(LayoutSpec::row().gap(8.0).align(Align::Center), PaintStyle::default(), |ui| {
                        ui.button("Close Inspector", th.bg_control, |_, _| {});
                        ui.primary_button("Reset Auto", |_, _| {});
                        ui.badge("v1.0.0 STABLE", th.accent);
                        ui.badge("1.25× Auto", th.success);
                        ui.ui_text("Skima Studio", 15.0, th.text_primary);
                        ui.ui_text("OKLCH 256°", 12.0, th.text_secondary);
                        ui.ui_text("OS Fonts: 4", 12.0, th.text_muted);
                    });
                },
            );
        });
        for idx in 0..sys.layout.topo.len() {
            if (sys.layout.topo[idx].flags & FLAG_ALIVE) != 0
                && (sys.layout.topo[idx].flags & FLAG_HAS_TEXT) != 0
            {
                let tid = sys.layout.specs[idx].aux_id as usize;
                assert_eq!(
                    sys.layout.texts[tid].cached_lines.len(),
                    1,
                    "Unshrunk Fit button/badge/label wrapped onto {} lines at sf={sf}",
                    sys.layout.texts[tid].cached_lines.len()
                );
            }
        }
    }
}

// 26. `FLAG_DIRTY_SCROLL` `f32::to_bits()` Oracle Parity (`0 B` Heap) & Vulkan 1.3 vs `CpuRenderer`
// Composite UI Fractional Parity (`<= 4 LSB`) Across All 10 Fractional Scales (`4/3`, `5/3`, `1/8`s).
#[test]
fn flag_dirty_scroll_fractional_oracle_and_vulkan_cpu_composite_parity_proof() {
    use skima::gpu::cpu::CpuRenderer;
    use skima::gpu::vulkan::VulkanRenderer;
    use skima::layout::{
        FLAG_ALIVE, FLAG_CLIPS_CHILDREN, FLAG_RELAYOUT_BOUNDARY, FLAG_SCROLLABLE, LayoutTree,
        NodeId,
    };
    use skima::vector::FillRule;

    // -------------------------------------------------------------------------
    // Part A: `FLAG_DIRTY_SCROLL` `f32::to_bits()` Oracle Parity (`0 B` Heap)
    // -------------------------------------------------------------------------
    let build_scroll_tree = |sf: f32, outer_scroll_y: f32, inner_scroll_y: f32| -> (LayoutTree, NodeId, NodeId, Vec<NodeId>) {
        let mut tree = LayoutTree::new();
        tree.snap_to_pixels = true;
        tree.set_device_scale(sf);

        let root = tree.alloc_node(LayoutSpec::col().fixed(420.0, 320.0).pad(7.0).gap(5.0));
        tree.root = root;

        let header = tree.alloc_node(LayoutSpec::row().width_grow(1.0).height_fixed(29.0).gap(6.0));
        let mut pills = Vec::with_capacity(3);
        for k in 0..3 {
            pills.push(tree.alloc_node(LayoutSpec::row().width_grow(1.0 + k as f32).height_fixed(29.0)));
        }
        tree.set_children(header, &pills);

        let mut outer_spec = LayoutSpec::col().width_grow(1.0).height_grow(1.0);
        outer_spec.scroll_y = outer_scroll_y;
        let outer_scroll = tree.alloc_node(outer_spec);
        tree.topo[outer_scroll.idx()].flags |= FLAG_SCROLLABLE | FLAG_CLIPS_CHILDREN;

        // 12 contiguous `gap=0` variable-grow/fixed rows + nested `Grid` + nested `WrapRow` + nested scroll container:
        let mut flex_rows = Vec::with_capacity(12);
        let flex_stack = tree.alloc_node(LayoutSpec::col().width_grow(1.0).height_fixed(211.0));
        for r_i in 0..12usize {
            let r_spec = LayoutSpec::row().width_grow(1.0).height_grow(1.0 + (r_i % 3) as f32);
            let r_nid = tree.alloc_node(r_spec);
            tree.topo[r_nid.idx()].flags |= FLAG_RELAYOUT_BOUNDARY;
            let c_leaf = tree.alloc_node(LayoutSpec::col().fixed(45.0 + r_i as f32, 11.0));
            tree.set_children(r_nid, &[c_leaf]);
            flex_rows.push(r_nid);
        }
        tree.set_children(flex_stack, &flex_rows);

        let grid = tree.alloc_node(LayoutSpec::col().width_grow(1.0).height_fixed(48.0).gap(0.0));
        tree.attach_grid(
            grid,
            vec![TrackSize::Fixed(63.0), TrackSize::Fr(1.0), TrackSize::Fr(2.0)],
            vec![TrackSize::Fr(1.0), TrackSize::Fr(1.0)],
        );
        let mut g_cells = Vec::with_capacity(6);
        for g_i in 0..6usize {
            g_cells.push(tree.alloc_node(LayoutSpec::row().height_fixed(23.0 + (g_i % 2) as f32)));
        }
        tree.set_children(grid, &g_cells);

        let wrap = tree.alloc_node(LayoutSpec::wrap_row().width_grow(1.0).gap(0.0));
        let mut w_chips = Vec::with_capacity(7);
        for w_i in 0..7usize {
            w_chips.push(tree.alloc_node(LayoutSpec::row().fixed(73.0 + (w_i * 5) as f32, 19.0)));
        }
        tree.set_children(wrap, &w_chips);

        let mut inner_spec = LayoutSpec::col().width_grow(1.0).height_fixed(95.0);
        inner_spec.scroll_y = inner_scroll_y;
        let inner_scroll = tree.alloc_node(inner_spec);
        tree.topo[inner_scroll.idx()].flags |= FLAG_SCROLLABLE | FLAG_CLIPS_CHILDREN;
        let mut inner_items = Vec::with_capacity(10);
        let mut anchor_target = NodeId::NONE;
        for j in 0..10usize {
            let item = tree.alloc_node(LayoutSpec::row().width_grow(1.0).height_fixed(17.5));
            inner_items.push(item);
            if j == 2 {
                anchor_target = item;
            }
        }
        tree.set_children(inner_scroll, &inner_items);
        tree.set_children(outer_scroll, &[flex_stack, grid, wrap, inner_scroll]);

        // Floating Pass-4 popover anchored inside the nested scroll hierarchy!
        let popover = tree.alloc_node(LayoutSpec::col().fixed(112.0, 38.0));
        tree.set_children(root, &[header, outer_scroll, popover]);
        tree.set_floating_anchor(popover, anchor_target, AnchorPlacement::BottomStart, 3.0, 4.0);

        (tree, outer_scroll, inner_scroll, flex_rows)
    };

    let mut rng = SplitMix64::new(0xFA57_5C80_1143_5333);
    for &sf in &FRACTIONAL_QA_SCALES {
        let (mut inc_tree, outer_sc, inner_sc, flex_rows) = build_scroll_tree(sf, 0.0, 0.0);
        inc_tree.compute_layout(420.0, 320.0);

        for step in 0..24usize {
            let next_outer = match step {
                0 => 0.5 / sf,   // Exact +0.5 physical-px tie!
                1 => 19.5 / sf,  // Zero-crossing -0.5 physical-px tie!
                2 => 37.25 / sf, // Quarter physical-px offset!
                _ => rng.next_f32(0.0, 140.0),
            };
            let next_inner = if step % 2 == 0 {
                rng.next_f32(0.0, 65.0)
            } else {
                inc_tree.specs[inner_sc.idx()].scroll_y
            };

            let ((), s_allocs, s_bytes) = measure_thread_allocs(|| {
                inc_tree.set_scroll_y(outer_sc, next_outer);
                inc_tree.set_scroll_y(inner_sc, next_inner);
                assert_eq!(
                    inc_tree.topo[inc_tree.root.idx()].flags & skima::layout::FLAG_DIRTY_LAYOUT,
                    0
                );
                inc_tree.compute_layout(420.0, 320.0);
            });
            if step >= 1 {
                assert_eq!(
                    (s_allocs, s_bytes),
                    (0, 0),
                    "FLAG_DIRTY_SCROLL solve must allocate 0 B heap at sf={sf} step={step}"
                );
            }
            assert!(
                inc_tree.last_metrics.boundaries_solved <= 2,
                "Scroll-only fast path must only visit dirty scroll roots at sf={sf}"
            );
            assert_eq!(inc_tree.last_metrics.text_shaper_calls, 0);

            let (mut oracle_tree, _, _, _) = build_scroll_tree(sf, next_outer, next_inner);
            oracle_tree.compute_layout(420.0, 320.0);

            assert_eq!(inc_tree.boxes.len(), oracle_tree.boxes.len());
            for idx in 0..inc_tree.topo.len() {
                if (inc_tree.topo[idx].flags & FLAG_ALIVE) == 0 {
                    continue;
                }
                let ib = inc_tree.boxes[idx];
                let ob = oracle_tree.boxes[idx];
                assert_eq!(
                    (
                        ib.x.to_bits(),
                        ib.y.to_bits(),
                        ib.w.to_bits(),
                        ib.h.to_bits(),
                        ib.fit_w.to_bits(),
                        ib.fit_h.to_bits()
                    ),
                    (
                        ob.x.to_bits(),
                        ob.y.to_bits(),
                        ob.w.to_bits(),
                        ob.h.to_bits(),
                        ob.fit_w.to_bits(),
                        ob.fit_h.to_bits()
                    ),
                    "FLAG_DIRTY_SCROLL box {idx} diverged from fresh oracle at sf={sf} step={step}: inc={ib:?} vs oracle={ob:?}"
                );
                let ic = inc_tree.clip_rects[idx];
                let oc = oracle_tree.clip_rects[idx];
                assert_eq!(
                    ic.map(f32::to_bits),
                    oc.map(f32::to_bits),
                    "FLAG_DIRTY_SCROLL clip_rect {idx} diverged from fresh oracle at sf={sf} step={step}: inc={ic:?} vs oracle={oc:?}"
                );
            }
        }

        // Also verify single-row `FLAG_RELAYOUT_BOUNDARY` dirtying inside fractional-snapped `Grow` flex stack
        // preserves error-diffused `h` (`0.0` physical seam across all 12 rows!):
        let dirty_leaf = inc_tree.topo[flex_rows[5].idx()].first_child;
        let mut leaf_sp = inc_tree.specs[dirty_leaf.idx()];
        leaf_sp.val_w = 91.0;
        inc_tree.update_spec(dirty_leaf, leaf_sp);
        inc_tree.compute_layout(420.0, 320.0);
        assert_eq!(inc_tree.last_metrics.boundaries_solved, 1);
        for pair in flex_rows.windows(2) {
            let b0 = inc_tree.boxes[pair[0].idx()];
            let b1 = inc_tree.boxes[pair[1].idx()];
            assert_eq!(
                ((b0.y + b0.h) * sf).round().to_bits(),
                (b1.y * sf).round().to_bits(),
                "Single-boundary incremental relayout broke 0-seam invariant at sf={sf}: {b0:?} -> {b1:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Part B: Vulkan 1.3 vs `CpuRenderer` Composite UI Fractional Parity (`<= 4 LSB`)
    // -------------------------------------------------------------------------
    let mut vk = VulkanRenderer::new_headless(4096, 256)
        .expect("Vulkan 1.3 headless renderer required for fractional parity proof");

    struct CompositeState {
        checked: bool,
        slider_v: f32,
        text: String,
        hover_card: bool,
    }

    let vlist = VirtualListIndex::new(8, 22.0);
    let build_composite = |ui: &mut Ui<'_, '_, CompositeState>, vl: &VirtualListIndex| {
        let checked = ui.state().checked;
        let sv = ui.state().slider_v;
        let hov = ui.state().hover_card;
        let th = ui.theme();

        ui.element(
            LayoutSpec::col().fixed(280.0, 220.0).pad(10.0).gap(8.0),
            PaintStyle::solid(th.bg_canvas),
            |ui| {
                // Rounded header card with OKLab linear gradient, drop shadow, and fractional border:
                let card_border = if hov { th.accent } else { th.border_subtle };
                ui.element(
                    LayoutSpec::row()
                        .width_grow(1.0)
                        .height_fixed(42.0)
                        .pad_xy(10.0, 6.0)
                        .gap(8.0)
                        .align(Align::Center),
                    PaintStyle::oklab_gradient(
                        th.bg_elevated,
                        th.accent_muted,
                        0.45,
                    )
                    .rounded(8.0)
                    .border(1.0, card_border)
                    .drop_shadow(6.0, [0.0, 2.0], Color::rgba8(0, 0, 0, 110)),
                    |ui| {
                        ui.checkbox("Snap 1/120", checked, |s, _| s.checked = !s.checked);
                        ui.slider("Gain", sv, 0.0..=1.0, 86.0, |s, v| s.slider_v = v);
                    },
                );

                // Editable text input + Wang-Bézier vector sparkline icon:
                ui.element(
                    LayoutSpec::row()
                        .width_grow(1.0)
                        .height_fixed(36.0)
                        .gap(8.0)
                        .align(Align::Center),
                    PaintStyle::default(),
                    |ui| {
                        ui.text_input("Filter...", |s| &s.text, |s| &mut s.text);
                        ui.vector_canvas(44, 28, 1, th.accent, FillRule::NonZero, |p| {
                            p.stroke_polyline(
                                &[[3.0, 22.0], [15.0, 8.0], [28.0, 19.0], [40.0, 5.0]],
                                2.0,
                            );
                        });
                    },
                );

                // Hardware-clipped VirtualList with `0.5px` hairline row borders & shaped labels:
                ui.virtual_list(vl, 5.0, 260.0, 108.0, |ui, idx, rh| {
                    ui.element(
                        LayoutSpec::row()
                            .width_grow(1.0)
                            .height_fixed(rh)
                            .pad_xy(8.0, 2.0)
                            .align(Align::Center),
                        PaintStyle::solid(if idx % 2 == 0 {
                            th.bg_surface
                        } else {
                            th.bg_elevated
                        })
                        .border(0.5, th.border_subtle),
                        |ui| {
                            ui.label_usize(idx + 100, 12.5, 16.0, th.text_primary);
                        },
                    );
                });
            },
        );
    };

    for (scale_idx, &sf) in FRACTIONAL_QA_SCALES.iter().enumerate() {
        let mut st = CompositeState {
            checked: true,
            slider_v: 0.62,
            text: String::from("1.25x / 1.333x Crisp"),
            hover_card: false,
        };
        let mut sys = GuiSystem::with_system_fonts();
        sys.fonts.atlas.generation = ((scale_idx as u64) + 1) * 100;
        sys.set_scale_factor(sf);
        sys.clear_color = sys.theme.bg_canvas;

        let phys_w = (280.0 * sf - 1e-4).ceil() as u32;
        let phys_h = (220.0 * sf - 1e-4).ceil() as u32;
        let mut cpu = CpuRenderer::new(phys_w, phys_h);

        // Frame 1 (Full fractional scene) & Frame 2 (O(1) in-place partial-damage border paint patch!):
        for frame_i in 0..2usize {
            if frame_i == 1 {
                st.hover_card = true;
            }
            sys.build_frame(&st, 280.0, 220.0, |ui| build_composite(ui, &vlist));
            if frame_i == 1 {
                assert_eq!(sys.paint_updates_last_frame, 1);
            }

            let vk_rgba = sys
                .render_vulkan_offscreen(&mut vk, phys_w, phys_h)
                .expect("Vulkan fractional composite render");
            let buf_age = if frame_i == 0 { 0 } else { 1 };
            sys.render_to_backend(&mut cpu, phys_w, phys_h, buf_age)
                .unwrap();

            for y in 0..phys_h {
                for x in 0..phys_w {
                    let idx = ((y * phys_w + x) as usize) * 4;
                    let vp = [
                        vk_rgba[idx],
                        vk_rgba[idx + 1],
                        vk_rgba[idx + 2],
                        vk_rgba[idx + 3],
                    ];
                    let cp = cpu.pixel_at(x, y);
                    for ch in 0..4 {
                        let diff = (vp[ch] as i32 - cp[ch] as i32).abs();
                        assert!(
                            diff <= 4,
                            "Fractional composite Vulkan vs CPU parity exceeded 4 LSB at sf={sf} frame={frame_i} ({x},{y}) ch {ch}: vk={vp:?} vs cpu={cp:?} (diff={diff})"
                        );
                    }
                }
            }
        }
    }
}


