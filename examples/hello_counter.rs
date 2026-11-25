//! Minimal `skima 1.0` Quickstart Example:
//! - Launches a native Vulkan 1.3 Wayland/X11 window with `ScalePolicy::Auto`
//!   (`Ctrl+=` / `Ctrl+-` / `Ctrl+0` zoom; `cargo run --release --example hello_counter`)
//! - Supports `--smoke-test` (headless GPU render verification across Dark & Light themes and fractional HiDPI)
//!   or `--render-ppm <path>`.

use skima::{
    Align, Color, LayoutSpec, MainAlign, PaintStyle, Theme, Ui, render_headless_rgba8,
    save_rgba8_ppm,
};

struct CounterApp {
    title: String,
    count: i32,
    dark_mode: bool,
}

fn view(ui: &mut Ui<'_, '_, CounterApp>) {
    ui.sys.theme = if ui.state().dark_mode {
        Theme::dark()
    } else {
        Theme::light()
    };
    let th = ui.theme();
    ui.sys.clear_color = th.bg_canvas;

    let shadow_col = if th.is_dark {
        Color::rgba8(0, 0, 0, 115)
    } else {
        Color::rgba8(15, 23, 42, 28)
    };

    ui.element(
        LayoutSpec::col()
            .grow(1.0)
            .pad(32.0)
            .gap(16.0)
            .align(Align::Center)
            .justify(MainAlign::Center),
        PaintStyle::solid(th.bg_canvas),
        |ui| {
            ui.element(
                LayoutSpec::col().width_fixed(436.0).pad(24.0).gap(16.0),
                PaintStyle::solid(th.bg_surface)
                    .rounded(th.radius_lg)
                    .border(1.0, th.border_subtle)
                    .drop_shadow(14.0, [0.0, 8.0], shadow_col),
                |ui| {
                    let banner = format!("{} — Count: {}", ui.state().title, ui.state().count);
                    ui.heading(&banner);

                    ui.text_input("Project title...", |s| &s.title, |s| &mut s.title);

                    let progress = ((ui.state().count.clamp(0, 20)) as f32) / 20.0;
                    ui.progress_bar(progress, 388.0, th.accent);

                    ui.element(
                        LayoutSpec::row()
                            .width_grow(1.0)
                            .justify(MainAlign::SpaceBetween)
                            .align(Align::Center),
                        PaintStyle::default(),
                        |ui| {
                            ui.element(
                                LayoutSpec::row().gap(10.0).align(Align::Center),
                                PaintStyle::default(),
                                |ui| {
                                    let inc = ui.primary_button("+ Increment", |s, _| s.count += 1);
                                    ui.tooltip(inc, "Increase counter (animates spring progress)");
                                    let dec =
                                        ui.button("- Decrement", th.bg_control, |s, _| s.count -= 1);
                                    ui.tooltip(dec, "Decrease counter");
                                },
                            );
                            let dark = ui.state().dark_mode;
                            let sw =
                                ui.toggle_switch("Dark", dark, |s, _| s.dark_mode = !s.dark_mode);
                            ui.tooltip(sw, "Toggle calibrated Dark / Light OKLCH theme");
                        },
                    );
                },
            );
        },
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut app = CounterApp {
        title: String::from("Hello Skima 1.0 🦀"),
        count: 7,
        dark_mode: true,
    };

    if args.iter().any(|a| a == "--smoke-test") {
        for (dark, sf) in [(true, 1.0_f32), (false, 1.0), (true, 1.25), (false, 1.5)] {
            app.dark_mode = dark;
            let (w, h, rgba) = render_headless_rgba8(640, 420, sf, true, &app, view);
            assert_eq!(rgba.len(), (w * h * 4) as usize);
            let distinct = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|px| {
                    if dark {
                        px[0] > 45 || px[1] > 45 || px[2] > 65
                    } else {
                        px[0] < 215 || px[1] < 215 || px[2] < 215
                    }
                })
                .count();
            assert!(
                distinct > 8_000,
                "Expected crisp counter card in dark={dark} at {sf}x"
            );
        }
        println!(
            "hello_counter smoke-test passed: rendered Dark & Light frames across 1.0x, 1.25x, and 1.5x HiDPI."
        );
        return;
    }
    if let Some(pos) = args.iter().position(|a| a == "--render-ppm")
        && let Some(out_path) = args.get(pos + 1)
    {
        let (w, h, rgba) = render_headless_rgba8(640, 420, 1.0, true, &app, view);
        save_rgba8_ppm(out_path, w, h, &rgba).expect("write PPM");
        println!("Saved {w}x{h} rendered frame to {out_path}");
        return;
    }

    #[cfg(feature = "window")]
    {
        skima::run_window(
            skima::WindowConfig {
                title: String::from("Skima 1.0 — Hello Counter"),
                width: 640,
                height: 420,
                ..skima::WindowConfig::default()
            },
            app,
            view,
        )
        .expect("run window");
    }
    #[cfg(not(feature = "window"))]
    {
        let _ = app;
        eprintln!("Built without `window` feature; pass `--smoke-test` or `--render-ppm <path>`.");
    }
}
