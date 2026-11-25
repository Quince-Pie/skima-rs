//! Turnkey Desktop Windowing (`winit 0.30` + Native Vulkan 1.3 Swapchain), Automatic Fractional
//! HiDPI Scaling, and Headless Frame Export.
//!
//! - Demand-driven `ControlFlow::Wait` event loop: **0.0% CPU & GPU** when idle, automatically switching
//!   to smooth native VSync refresh (`120Hz`/`240Hz`) whenever `Spring` animations or interactions occur!
//! - Full HiDPI fractional physical-pixel scaling (`ScalePolicy::Auto` + Wayland `1/120` +
//!   cached Linux `/sys/class/drm` VESA EDID millimeter probing), live `Ctrl+=` / `Ctrl+-` / `Ctrl+0`
//!   scale-lattice zoom, OS `CursorIcon` updates, keyboard focus/text routing, and swapchain
//!   `buffer_age` partial-damage rendering.

use crate::gpu::vulkan::VulkanRenderer;
use crate::scale::ScalePolicy;
use crate::ui::{GuiSystem, Ui};
use std::io::Write;
use std::path::Path;

/// Configuration for launching a native desktop `skima` window or headless GPU render session.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
    pub load_system_fonts: bool,
    /// Fractional HiDPI scaling policy (defaults to [`ScalePolicy::Auto`]).
    pub scale_policy: ScalePolicy,
    /// Optional frame limit for automated smoke-testing (`None` = run until user closes window).
    pub max_frames: Option<u64>,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: String::from("Skima Application"),
            width: 1280,
            height: 820,
            vsync: true,
            load_system_fonts: true,
            scale_policy: ScalePolicy::Auto,
            max_frames: None,
        }
    }
}

/// Render a single frame on hardware Vulkan 1.3 (or fallback CPU renderer if no Vulkan ICD exists)
/// and return the physical `(width, height, rgba8_bytes)` buffer!
pub fn render_headless_rgba8<S: 'static>(
    logical_width: u32,
    logical_height: u32,
    scale_factor: f32,
    load_system_fonts: bool,
    state: &S,
    build_fn: impl FnOnce(&mut Ui<'_, '_, S>),
) -> (u32, u32, Vec<u8>) {
    let mut sys = if load_system_fonts {
        GuiSystem::with_system_fonts()
    } else {
        GuiSystem::new()
    };
    sys.set_scale_factor(scale_factor);
    sys.clear_color = sys.theme.bg_canvas;
    sys.build_frame(state, logical_width as f32, logical_height as f32, build_fn);

    let phys_w = ((logical_width as f32) * sys.scale_factor - 1e-4)
        .ceil()
        .max(1.0) as u32;
    let phys_h = ((logical_height as f32) * sys.scale_factor - 1e-4)
        .ceil()
        .max(1.0) as u32;

    if let Ok(mut vk) = VulkanRenderer::new_headless(
        sys.primitives.len().max(4096),
        sys.clips.len().max(256),
    ) && let Ok(rgba) = sys.render_vulkan_offscreen(&mut vk, phys_w, phys_h)
    {
        return (phys_w, phys_h, rgba);
    }

    let mut cpu = crate::gpu::cpu::CpuRenderer::new(phys_w, phys_h);
    sys.render_to_backend(&mut cpu, phys_w, phys_h, 1).unwrap();
    (phys_w, phys_h, cpu.framebuffer)
}

/// Write an RGBA8 pixel buffer to a binary Netpbm `.ppm` (`P6`) image file (zero external image crate bloat).
pub fn save_rgba8_ppm(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{width} {height}\n255\n")?;
    let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for &[r, g, b, _] in rgba.as_chunks::<4>().0 {
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    f.write_all(&rgb)
}

#[cfg(feature = "window")]
pub use windowed::{WindowError, run_window};

#[cfg(feature = "window")]
mod windowed {
    use super::WindowConfig;
    use crate::gpu::vulkan::{VulkanError, VulkanRenderer, VulkanSwapchain};
    use crate::scale::{DisplayMetrics, parse_edid_physical_mm, probe_linux_drm_physical_mm};
    use crate::ui::{CursorIcon, GuiSystem, KeyInput, Ui};
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
    use std::time::Instant;
    use winit::application::ApplicationHandler;
    use winit::dpi::{LogicalSize, PhysicalSize};
    use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
    use winit::window::{Window, WindowAttributes, WindowId};

    #[derive(Debug)]
    pub enum WindowError {
        EventLoop(winit::error::EventLoopError),
        Os(winit::error::OsError),
        Vulkan(VulkanError),
        Handle(raw_window_handle::HandleError),
    }

    impl From<winit::error::EventLoopError> for WindowError {
        fn from(e: winit::error::EventLoopError) -> Self {
            Self::EventLoop(e)
        }
    }
    impl From<winit::error::OsError> for WindowError {
        fn from(e: winit::error::OsError) -> Self {
            Self::Os(e)
        }
    }
    impl From<VulkanError> for WindowError {
        fn from(e: VulkanError) -> Self {
            Self::Vulkan(e)
        }
    }
    impl From<raw_window_handle::HandleError> for WindowError {
        fn from(e: raw_window_handle::HandleError) -> Self {
            Self::Handle(e)
        }
    }

    /// Preload Wayland/XKB/X11 shared objects into the dynamic linker map on NixOS systems
    /// where `LD_LIBRARY_PATH` may omit `/nix/store/*-wayland-*` or `/nix/store/*-libxkbcommon-*`.
    #[cfg(all(unix, target_os = "linux"))]
    fn preload_linux_display_libs() {
        unsafe extern "C" {
            fn dlopen(
                filename: *const core::ffi::c_char,
                flags: core::ffi::c_int,
            ) -> *mut core::ffi::c_void;
        }
        const RTLD_NOW: core::ffi::c_int = 2;
        const RTLD_GLOBAL: core::ffi::c_int = 0x100;

        let wanted = [
            ("wayland-", "lib/libwayland-client.so.0"),
            ("wayland-", "lib/libwayland-cursor.so.0"),
            ("wayland-", "lib/libwayland-egl.so.1"),
            ("libxkbcommon-", "lib/libxkbcommon.so.0"),
            ("libX11-", "lib/libX11.so.6"),
        ];
        let Ok(store) = std::fs::read_dir("/nix/store") else {
            return;
        };
        let mut loaded = [false; 5];
        for ent in store.flatten() {
            if loaded.iter().all(|&b| b) {
                break;
            }
            let fname = ent.file_name();
            let s = fname.to_string_lossy();
            for (i, &(pkg_pat, rel_so)) in wanted.iter().enumerate() {
                if !loaded[i] && s.contains(pkg_pat) {
                    let full = ent.path().join(rel_so);
                    if full.exists()
                        && let Ok(cstr) = std::ffi::CString::new(full.to_string_lossy().as_bytes())
                    {
                        let handle = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
                        if !handle.is_null() {
                            loaded[i] = true;
                        }
                    }
                }
            }
        }
    }

    /// Secondary Linux `/sys/class/drm` fallback for DisplayID / tiled 5K+ monitors whose base EDID
    /// DTD lists a fallback timing (`2560x1440`) while `/sys/class/drm/card*-*/modes` and EDID `0xFC`
    /// monitor name expose the true `5120x2880` panel mode.
    #[cfg(target_os = "linux")]
    fn probe_linux_drm_modes_or_model_mm(
        monitor_name: Option<&str>,
        mode_w: u32,
        mode_h: u32,
    ) -> Option<(u32, u32)> {
        use std::io::Read;

        if let Some(direct) = probe_linux_drm_physical_mm(monitor_name, mode_w, mode_h) {
            return Some(direct);
        }
        if mode_w < 64 || mode_h < 64 {
            return None;
        }
        let entries = std::fs::read_dir("/sys/class/drm").ok()?;
        let mut matched: Option<(u32, u32)> = None;
        let mut ambiguous = false;

        for ent in entries.flatten() {
            let fname = ent.file_name();
            let s = fname.to_string_lossy();
            if !s.starts_with("card") || !s.contains('-') || s.contains("Writeback") {
                continue;
            }
            let dir = ent.path();
            let mut status_buf = [0u8; 32];
            let Ok(mut sf) = std::fs::File::open(dir.join("status")) else {
                continue;
            };
            let Ok(ns) = sf.read(&mut status_buf) else {
                continue;
            };
            if core::str::from_utf8(&status_buf[..ns]).unwrap_or("").trim() != "connected" {
                continue;
            }

            let mut edid_buf = [0u8; 256];
            let Ok(mut ef) = std::fs::File::open(dir.join("edid")) else {
                continue;
            };
            let Ok(ne) = ef.read(&mut edid_buf) else {
                continue;
            };
            let Some(parsed) = parse_edid_physical_mm(&edid_buf[..ne]) else {
                continue;
            };

            let oriented = if (mode_w >= mode_h) == (parsed.width_mm >= parsed.height_mm) {
                (parsed.width_mm, parsed.height_mm)
            } else {
                (parsed.height_mm, parsed.width_mm)
            };

            // Check EDID 0xFC ASCII Monitor Name descriptor match if compositor exposed model string:
            if let Some(q_name) = monitor_name
                && !q_name.is_empty()
            {
                for slot in 0..4 {
                    let base = 54 + slot * 18;
                    let desc = &edid_buf[base..base + 18];
                    if desc[0] == 0
                        && desc[1] == 0
                        && desc[2] == 0
                        && desc[3] == 0xFC
                        && let Ok(ascii) = core::str::from_utf8(&desc[5..18])
                    {
                        let model = ascii.split('\n').next().unwrap_or("").trim();
                        if !model.is_empty()
                            && (q_name.contains(model) || model.eq_ignore_ascii_case(q_name))
                        {
                            return Some(oriented);
                        }
                    }
                }
            }

            // Check `/sys/class/drm/card*-*/modes` for active `(mode_w, mode_h)` (including 90° rotation):
            let mut modes_buf = [0u8; 512];
            if let Ok(mut mf) = std::fs::File::open(dir.join("modes"))
                && let Ok(nm) = mf.read(&mut modes_buf)
                && let Ok(modes_str) = core::str::from_utf8(&modes_buf[..nm])
            {
                let has_mode = modes_str.lines().any(|line| {
                    if let Some((ws, hs)) = line.trim().split_once('x')
                        && let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>())
                    {
                        (w == mode_w && h == mode_h) || (w == mode_h && h == mode_w)
                    } else {
                        false
                    }
                });
                if has_mode {
                    match matched {
                        None => matched = Some(oriented),
                        Some(prev) => {
                            if prev.0.abs_diff(oriented.0) > 10 || prev.1.abs_diff(oriented.1) > 10
                            {
                                ambiguous = true;
                            }
                        }
                    }
                }
            }
        }
        if ambiguous { None } else { matched }
    }

    #[cfg(not(target_os = "linux"))]
    #[inline]
    fn probe_linux_drm_modes_or_model_mm(
        monitor_name: Option<&str>,
        mode_w: u32,
        mode_h: u32,
    ) -> Option<(u32, u32)> {
        probe_linux_drm_physical_mm(monitor_name, mode_w, mode_h)
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct MonitorCacheKey {
        name: Option<String>,
        width_px: u32,
        height_px: u32,
    }

    /// Launch a native hardware-accelerated Vulkan 1.3 desktop window driving `state: S` with `build_fn`!
    pub fn run_window<S: 'static, F>(
        config: WindowConfig,
        initial_state: S,
        build_fn: F,
    ) -> Result<(), WindowError>
    where
        F: for<'sys, 's> FnMut(&mut Ui<'sys, 's, S>) + 'static,
    {
        #[cfg(all(unix, target_os = "linux"))]
        preload_linux_display_libs();

        let event_loop = EventLoop::new()?;
        let mut runner = AppRunner {
            config,
            state: initial_state,
            build_fn,
            sys: None,
            window: None,
            vk: None,
            swapchain: None,
            modifiers: ModifiersState::default(),
            cursor_physical: None,
            cursor_logical: [0.0, 0.0],
            applied_cursor: CursorIcon::Default,
            cached_monitor_key: None,
            cached_monitor_mm: None,
            last_tick: Instant::now(),
            frames_rendered: 0,
            init_err: None,
        };
        event_loop.run_app(&mut runner)?;
        if let Some(err) = runner.init_err.take() {
            return Err(err);
        }
        Ok(())
    }

    struct AppRunner<S: 'static, F> {
        config: WindowConfig,
        state: S,
        build_fn: F,
        sys: Option<GuiSystem>,
        window: Option<Window>,
        vk: Option<VulkanRenderer>,
        swapchain: Option<VulkanSwapchain>,
        modifiers: ModifiersState,
        cursor_physical: Option<[f64; 2]>,
        cursor_logical: [f32; 2],
        applied_cursor: CursorIcon,
        cached_monitor_key: Option<MonitorCacheKey>,
        cached_monitor_mm: Option<(u32, u32)>,
        last_tick: Instant,
        frames_rendered: u64,
        init_err: Option<WindowError>,
    }

    impl<S: 'static, F> Drop for AppRunner<S, F> {
        fn drop(&mut self) {
            if let (Some(vk), Some(sc)) = (self.vk.as_ref(), self.swapchain.take()) {
                vk.destroy_swapchain(sc);
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ZoomCommand {
        Step(i32),
        ResetAuto,
    }

    fn detect_zoom_shortcut(
        kev: &winit::event::KeyEvent,
        ctrl: bool,
        alt: bool,
        shift: bool,
    ) -> Option<ZoomCommand> {
        if !ctrl || alt {
            return None;
        }
        if let Key::Character(c) = &kev.logical_key {
            match c.as_str() {
                "+" | "=" => return Some(ZoomCommand::Step(1)),
                "-" | "_" => return Some(ZoomCommand::Step(-1)),
                "0" => return Some(ZoomCommand::ResetAuto),
                _ => {}
            }
        }
        match kev.physical_key {
            PhysicalKey::Code(KeyCode::Equal | KeyCode::NumpadAdd) => Some(ZoomCommand::Step(1)),
            PhysicalKey::Code(KeyCode::Minus | KeyCode::NumpadSubtract) => {
                Some(ZoomCommand::Step(-1))
            }
            PhysicalKey::Code(KeyCode::Digit0 | KeyCode::Numpad0) if !shift => {
                Some(ZoomCommand::ResetAuto)
            }
            _ => None,
        }
    }

    fn query_window_display_metrics(
        cached_key: &mut Option<MonitorCacheKey>,
        cached_mm: &mut Option<(u32, u32)>,
        win: &Window,
        event_loop: &ActiveEventLoop,
        os_scale_override: Option<f64>,
        prev_120ths: Option<u32>,
    ) -> DisplayMetrics {
        let phys = win.inner_size();
        // On Wayland, `event_loop.primary_monitor()` is always `None` and `win.current_monitor()`
        // may be `None` before the initial `wl_surface.enter` event; fall back cleanly to the first
        // enumerated `WlOutput` / RandR CRTC!
        let mon = win
            .current_monitor()
            .or_else(|| event_loop.primary_monitor())
            .or_else(|| win.available_monitors().next())
            .or_else(|| event_loop.available_monitors().next());

        let (mon_w, mon_h, mon_name, mon_scale) = if let Some(m) = mon {
            let sz = m.size();
            let mw = if sz.width > 0 {
                sz.width
            } else {
                phys.width.max(1)
            };
            let mh = if sz.height > 0 {
                sz.height
            } else {
                phys.height.max(1)
            };
            (mw, mh, m.name(), Some(m.scale_factor()))
        } else {
            (phys.width.max(1), phys.height.max(1), None, None)
        };

        let key = MonitorCacheKey {
            name: mon_name,
            width_px: mon_w,
            height_px: mon_h,
        };
        let mm = if cached_key.as_ref() == Some(&key) {
            *cached_mm
        } else {
            let probed = probe_linux_drm_modes_or_model_mm(key.name.as_deref(), mon_w, mon_h);
            *cached_key = Some(key);
            *cached_mm = probed;
            probed
        };

        let win_os_scale = os_scale_override.unwrap_or_else(|| win.scale_factor());
        let eff_os_scale = if (win_os_scale - 1.0).abs() >= (1.0 / 240.0) {
            win_os_scale
        } else {
            mon_scale
                .filter(|&ms| (ms - 1.0).abs() >= (1.0 / 240.0))
                .unwrap_or(win_os_scale)
        };

        DisplayMetrics::new(mon_w, mon_h)
            .with_physical_mm_opt(mm)
            .with_os_scale(eff_os_scale)
            .with_window_px(phys.width, phys.height)
            .with_previous_120ths_opt(prev_120ths)
    }

    impl<S: 'static, F> ApplicationHandler for AppRunner<S, F>
    where
        F: for<'sys, 's> FnMut(&mut Ui<'sys, 's, S>) + 'static,
    {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = WindowAttributes::default()
                .with_title(self.config.title.clone())
                .with_inner_size(LogicalSize::new(
                    self.config.width as f64,
                    self.config.height as f64,
                ));
            let win = match event_loop.create_window(attrs) {
                Ok(w) => w,
                Err(e) => {
                    self.init_err = Some(WindowError::Os(e));
                    event_loop.exit();
                    return;
                }
            };

            let mut sys = if self.config.load_system_fonts {
                GuiSystem::with_system_fonts()
            } else {
                GuiSystem::new()
            };
            sys.scale_policy = self.config.scale_policy;
            let init_metrics = query_window_display_metrics(
                &mut self.cached_monitor_key,
                &mut self.cached_monitor_mm,
                &win,
                event_loop,
                None,
                None,
            );
            let resolved = sys.resolve_display_scale(init_metrics);

            // When X11 or an unscaled compositor reports `1.0` on a HiDPI monitor where `ScalePolicy::Auto`
            // resolves e.g. `1.5x` or `2.0x`, scale the initial physical window size so `config.width x height`
            // preserves its intended ergonomic logical workspace (clamped to 92% of monitor bounds):
            let win_os_sf = win.scale_factor();
            if (resolved.scale_factor as f64 - win_os_sf).abs() > 0.05
                && init_metrics.width_px >= 640
                && init_metrics.height_px >= 480
            {
                let want_pw = ((self.config.width as f32 * resolved.scale_factor).round() as u32)
                    .min((init_metrics.width_px * 92) / 100)
                    .max(1);
                let want_ph = ((self.config.height as f32 * resolved.scale_factor).round() as u32)
                    .min((init_metrics.height_px * 92) / 100)
                    .max(1);
                let _ = win.request_inner_size(PhysicalSize::new(want_pw, want_ph));
            }

            let phys = win.inner_size();
            let refreshed = init_metrics
                .with_window_px(phys.width, phys.height)
                .with_previous_120ths(resolved.wayland_120ths);
            let _ = sys.resolve_display_scale(refreshed);

            let init_res = (|| -> Result<(VulkanRenderer, VulkanSwapchain), WindowError> {
                let raw_disp = win.display_handle()?.as_raw();
                let raw_win = win.window_handle()?.as_raw();
                let vk = VulkanRenderer::new_for_window(raw_disp, 8192, 512)?;
                let surface = unsafe { vk.create_window_surface(raw_disp, raw_win)? };
                let sc = unsafe {
                    vk.create_swapchain(
                        surface,
                        phys.width.max(1),
                        phys.height.max(1),
                        self.config.vsync,
                    )?
                };
                Ok((vk, sc))
            })();

            match init_res {
                Ok((vk, sc)) => {
                    sys.clear_color = sys.theme.bg_canvas;
                    win.request_redraw();
                    self.sys = Some(sys);
                    self.vk = Some(vk);
                    self.swapchain = Some(sc);
                    self.window = Some(win);
                    self.last_tick = Instant::now();
                }
                Err(e) => {
                    self.init_err = Some(e);
                    event_loop.exit();
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            let (Some(win), Some(sys), Some(vk), Some(sc)) = (
                self.window.as_ref(),
                self.sys.as_mut(),
                self.vk.as_mut(),
                self.swapchain.as_mut(),
            ) else {
                return;
            };

            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(new_phys) => {
                    if new_phys.width > 0 && new_phys.height > 0 {
                        let prev_sf = sys.scale_factor;
                        let prev_120 = sys.resolved_scale.map(|r| r.wayland_120ths);
                        let metrics = query_window_display_metrics(
                            &mut self.cached_monitor_key,
                            &mut self.cached_monitor_mm,
                            win,
                            event_loop,
                            None,
                            prev_120,
                        )
                        .with_window_px(new_phys.width, new_phys.height);
                        let _ = sys.resolve_display_scale(metrics);
                        if (sys.scale_factor - prev_sf).abs() > 1e-5
                            && let Some([px, py]) = self.cursor_physical
                        {
                            let lx = (px as f32) / sys.scale_factor;
                            let ly = (py as f32) / sys.scale_factor;
                            self.cursor_logical = [lx, ly];
                            let _ = sys.pointer_move(lx, ly, &mut self.state);
                        }
                        let _ = unsafe {
                            vk.recreate_swapchain_images(
                                sc,
                                new_phys.width,
                                new_phys.height,
                                self.config.vsync,
                            )
                        };
                        win.request_redraw();
                    }
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    let prev_120 = sys.resolved_scale.map(|r| r.wayland_120ths);
                    let metrics = query_window_display_metrics(
                        &mut self.cached_monitor_key,
                        &mut self.cached_monitor_mm,
                        win,
                        event_loop,
                        Some(scale_factor),
                        prev_120,
                    );
                    let _ = sys.resolve_display_scale(metrics);
                    if let Some([px, py]) = self.cursor_physical {
                        let lx = (px as f32) / sys.scale_factor;
                        let ly = (py as f32) / sys.scale_factor;
                        self.cursor_logical = [lx, ly];
                        let _ = sys.pointer_move(lx, ly, &mut self.state);
                    }
                    let phys = win.inner_size();
                    if phys.width > 0 && phys.height > 0 {
                        let _ = unsafe {
                            vk.recreate_swapchain_images(
                                sc,
                                phys.width,
                                phys.height,
                                self.config.vsync,
                            )
                        };
                    }
                    win.request_redraw();
                }
                WindowEvent::ModifiersChanged(m) => {
                    self.modifiers = m.state();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.cursor_physical = Some([position.x, position.y]);
                    let sf = sys.scale_factor;
                    let lx = (position.x as f32) / sf;
                    let ly = (position.y as f32) / sf;
                    self.cursor_logical = [lx, ly];
                    if sys.pointer_move(lx, ly, &mut self.state) {
                        win.request_redraw();
                    }
                    if sys.active_cursor != self.applied_cursor {
                        self.applied_cursor = sys.active_cursor;
                        win.set_cursor(map_cursor_icon(self.applied_cursor));
                    }
                }
                WindowEvent::CursorLeft { .. } => {
                    self.cursor_physical = None;
                    if sys.pointer_leave() {
                        win.request_redraw();
                    }
                    if sys.active_cursor != self.applied_cursor {
                        self.applied_cursor = sys.active_cursor;
                        win.set_cursor(map_cursor_icon(self.applied_cursor));
                    }
                }
                WindowEvent::MouseInput {
                    state: btn_state,
                    button: MouseButton::Left,
                    ..
                } => match btn_state {
                    ElementState::Pressed => {
                        let [lx, ly] = self.cursor_logical;
                        if sys.pointer_down(lx, ly, &mut self.state) {
                            win.request_redraw();
                        }
                        if sys.active_cursor != self.applied_cursor {
                            self.applied_cursor = sys.active_cursor;
                            win.set_cursor(map_cursor_icon(self.applied_cursor));
                        }
                    }
                    ElementState::Released => {
                        if sys.pointer_up() {
                            win.request_redraw();
                        }
                        if sys.active_cursor != self.applied_cursor {
                            self.applied_cursor = sys.active_cursor;
                            win.set_cursor(map_cursor_icon(self.applied_cursor));
                        }
                    }
                },
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, lines_y) => lines_y * 52.0,
                        MouseScrollDelta::PixelDelta(px) => (px.y as f32) / sys.scale_factor,
                    };
                    let [lx, ly] = self.cursor_logical;
                    if sys.dispatch_scroll(lx, ly, dy, &mut self.state) {
                        win.request_redraw();
                    }
                    if sys.active_cursor != self.applied_cursor {
                        self.applied_cursor = sys.active_cursor;
                        win.set_cursor(map_cursor_icon(self.applied_cursor));
                    }
                }
                WindowEvent::KeyboardInput { event: kev, .. } => {
                    if kev.state == ElementState::Pressed {
                        let ctrl = self.modifiers.control_key() || self.modifiers.super_key();
                        let shift = self.modifiers.shift_key();
                        let alt = self.modifiers.alt_key();

                        // Live scale-lattice zoom (`Ctrl+=`/`Ctrl++`, `Ctrl+-`, `Ctrl+0`):
                        if let Some(zoom_cmd) = detect_zoom_shortcut(&kev, ctrl, alt, shift) {
                            let prev_sf = sys.scale_factor;
                            match zoom_cmd {
                                ZoomCommand::Step(delta) => {
                                    let _ = sys.zoom_scale(delta);
                                }
                                ZoomCommand::ResetAuto => {
                                    let metrics = query_window_display_metrics(
                                        &mut self.cached_monitor_key,
                                        &mut self.cached_monitor_mm,
                                        win,
                                        event_loop,
                                        None,
                                        None,
                                    );
                                    sys.display_metrics = Some(metrics);
                                    let _ = sys.reset_scale_auto();
                                }
                            }
                            if (sys.scale_factor - prev_sf).abs() > 1e-6
                                && let Some([px, py]) = self.cursor_physical
                            {
                                let lx = (px as f32) / sys.scale_factor;
                                let ly = (py as f32) / sys.scale_factor;
                                self.cursor_logical = [lx, ly];
                                let _ = sys.pointer_move(lx, ly, &mut self.state);
                            }
                            win.request_redraw();
                            return;
                        }

                        let mapped = match &kev.logical_key {
                            Key::Named(NamedKey::Tab) => Some(KeyInput::Tab { shift }),
                            Key::Named(NamedKey::Enter) => Some(KeyInput::Enter),
                            Key::Named(NamedKey::Space) if !ctrl => Some(KeyInput::Space),
                            Key::Named(NamedKey::Escape) => Some(KeyInput::Escape),
                            Key::Named(NamedKey::Backspace) if ctrl => {
                                Some(KeyInput::CtrlBackspace)
                            }
                            Key::Named(NamedKey::Backspace) => Some(KeyInput::Backspace),
                            Key::Named(NamedKey::Delete) if ctrl => Some(KeyInput::CtrlDelete),
                            Key::Named(NamedKey::Delete) => Some(KeyInput::Delete),
                            Key::Named(NamedKey::ArrowLeft) if ctrl && shift => {
                                Some(KeyInput::CtrlShiftArrowLeft)
                            }
                            Key::Named(NamedKey::ArrowLeft) if ctrl => {
                                Some(KeyInput::CtrlArrowLeft)
                            }
                            Key::Named(NamedKey::ArrowLeft) if shift => {
                                Some(KeyInput::ShiftArrowLeft)
                            }
                            Key::Named(NamedKey::ArrowLeft) => Some(KeyInput::ArrowLeft),
                            Key::Named(NamedKey::ArrowRight) if ctrl && shift => {
                                Some(KeyInput::CtrlShiftArrowRight)
                            }
                            Key::Named(NamedKey::ArrowRight) if ctrl => {
                                Some(KeyInput::CtrlArrowRight)
                            }
                            Key::Named(NamedKey::ArrowRight) if shift => {
                                Some(KeyInput::ShiftArrowRight)
                            }
                            Key::Named(NamedKey::ArrowRight) => Some(KeyInput::ArrowRight),
                            Key::Named(NamedKey::ArrowUp) => Some(KeyInput::ArrowUp),
                            Key::Named(NamedKey::ArrowDown) => Some(KeyInput::ArrowDown),
                            Key::Named(NamedKey::PageUp) => Some(KeyInput::PageUp),
                            Key::Named(NamedKey::PageDown) => Some(KeyInput::PageDown),
                            Key::Named(NamedKey::Home) if shift => Some(KeyInput::ShiftHome),
                            Key::Named(NamedKey::Home) => Some(KeyInput::Home),
                            Key::Named(NamedKey::End) if shift => Some(KeyInput::ShiftEnd),
                            Key::Named(NamedKey::End) => Some(KeyInput::End),
                            Key::Character(c) if ctrl && c.eq_ignore_ascii_case("a") => {
                                Some(KeyInput::SelectAll)
                            }
                            Key::Character(c) if ctrl && c.eq_ignore_ascii_case("c") => {
                                Some(KeyInput::Copy)
                            }
                            Key::Character(c) if ctrl && c.eq_ignore_ascii_case("x") => {
                                Some(KeyInput::Cut)
                            }
                            Key::Character(c) if ctrl && c.eq_ignore_ascii_case("v") => {
                                Some(KeyInput::Paste(sys.clipboard_text().to_owned()))
                            }
                            _ => kev
                                .text
                                .as_ref()
                                .filter(|t| !ctrl && !t.chars().any(|ch| ch.is_control()))
                                .map(|t| KeyInput::Text(t.to_string())),
                        };
                        if let Some(ki) = mapped
                            && sys.dispatch_key(&ki, &mut self.state)
                        {
                            win.request_redraw();
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
                    self.last_tick = now;
                    let _ = sys.step_animations(dt);

                    let phys = win.inner_size();
                    if phys.width == 0 || phys.height == 0 {
                        return;
                    }
                    if sc.extent.width != phys.width || sc.extent.height != phys.height {
                        let _ = unsafe {
                            vk.recreate_swapchain_images(
                                sc,
                                phys.width,
                                phys.height,
                                self.config.vsync,
                            )
                        };
                    }

                    // Resolve monitor scale using cached EDID geometry (`0` syscalls, `0 B` heap):
                    let prev_120 = sys.resolved_scale.map(|r| r.wayland_120ths);
                    let metrics = query_window_display_metrics(
                        &mut self.cached_monitor_key,
                        &mut self.cached_monitor_mm,
                        win,
                        event_loop,
                        None,
                        prev_120,
                    );
                    let _ = sys.resolve_display_scale(metrics);

                    let sf_before = sys.scale_factor;
                    let log_w = (phys.width as f32) / sf_before;
                    let log_h = (phys.height as f32) / sf_before;

                    let prev_hovered = sys.hovered_slot;
                    sys.clear_color = sys.theme.bg_canvas;
                    let build = &mut self.build_fn;
                    sys.build_frame(&self.state, log_w, log_h, |ui| build(ui));

                    // If an interactive in-UI scale control updated `ui.sys.scale_policy` / `scale_factor`
                    // during `build_frame`, immediately re-solve at the exact new `(log_w, log_h)` so the
                    // very first presented frame has zero viewport lag:
                    if (sys.scale_factor - sf_before).abs() > 1e-5 {
                        if let Some([px, py]) = self.cursor_physical {
                            let lx = (px as f32) / sys.scale_factor;
                            let ly = (py as f32) / sys.scale_factor;
                            self.cursor_logical = [lx, ly];
                            let _ = sys.pointer_move(lx, ly, &mut self.state);
                        }
                        let new_lw = (phys.width as f32) / sys.scale_factor;
                        let new_lh = (phys.height as f32) / sys.scale_factor;
                        sys.build_frame(&self.state, new_lw, new_lh, |ui| build(ui));
                    }

                    if sys.active_cursor != self.applied_cursor {
                        self.applied_cursor = sys.active_cursor;
                        win.set_cursor(map_cursor_icon(self.applied_cursor));
                    }

                    // Register Wayland `wl_surface.frame` VBlank callback before presenting!
                    win.pre_present_notify();
                    if let Err(VulkanError::Vk(ash::vk::Result::ERROR_OUT_OF_DATE_KHR)) =
                        sys.present_vulkan_swapchain(vk, sc)
                    {
                        let _ = unsafe {
                            vk.recreate_swapchain_images(
                                sc,
                                phys.width,
                                phys.height,
                                self.config.vsync,
                            )
                        };
                    }

                    self.frames_rendered += 1;
                    if let Some(limit) = self.config.max_frames
                        && self.frames_rendered >= limit
                    {
                        event_loop.exit();
                        return;
                    }

                    if sys.has_active_animations() {
                        win.request_redraw();
                        event_loop.set_control_flow(ControlFlow::Poll);
                    } else if sys.hovered_slot != prev_hovered {
                        // Layout shifted a new element under stationary pointer; repaint hover visual once:
                        win.request_redraw();
                        event_loop.set_control_flow(ControlFlow::Wait);
                    } else {
                        event_loop.set_control_flow(ControlFlow::Wait);
                    }
                }
                _ => {}
            }
        }
    }

    fn map_cursor_icon(ic: CursorIcon) -> winit::window::CursorIcon {
        match ic {
            CursorIcon::Default => winit::window::CursorIcon::Default,
            CursorIcon::Pointer => winit::window::CursorIcon::Pointer,
            CursorIcon::Text => winit::window::CursorIcon::Text,
            CursorIcon::Grab => winit::window::CursorIcon::Grab,
            CursorIcon::Grabbing => winit::window::CursorIcon::Grabbing,
            CursorIcon::EwResize => winit::window::CursorIcon::EwResize,
            CursorIcon::NsResize => winit::window::CursorIcon::NsResize,
        }
    }
}
