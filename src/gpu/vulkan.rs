//! Zero-Overhead Native Vulkan 1.3 Backend (`ash`).
//!
//! Why Native Vulkan 1.3 beats `wgpu` on desktop UI workloads:
//! 1. **Persistently Mapped ReBAR (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) SSBOs** (with full
//!    automatic `nonCoherentAtomSize` flush/invalidate support on non-coherent discrete, mobile, and
//!    `lavapipe` devices): writes dirty 64-byte `GpuPrimitive` spans directly into mapped memory with
//!    **zero** `wgpu-core` staging belt copies or `SnatchLock` tracking.
//! 2. **Hardware Dual-Source Subpixel Blending (`VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR`):** Renders exact
//!    per-channel RGB ClearType subpixel glyphs interleaved in the **same single instanced draw call** as
//!    analytical SDF rounded boxes, premultiplied OKLab gradients, and Gaussian shadows (zero pipeline switches!).
//! 3. **Vulkan 1.3 Dynamic Rendering (`vkCmdBeginRendering`) + Vertex-Stage Scissor Culling:** Zero
//!    `VkRenderPass`/`VkFramebuffer` objects and zero fragment overdraw outside the dirty `DamageRect`.
//! 4. **Precompiled SPIR-V Bytecode (`build.rs`):** 0.0 ms shader compilation at startup; no runtime `naga`.

use crate::gpu::{
    DamageRect, DamageTracker, GpuClip, GpuPrimitive, RenderBackend, RenderFrame,
    UBER_FRAG_DUAL_SRC_SPV, UBER_FRAG_SPV, UBER_VERT_SPV, ViewportUniform,
};
use crate::text::{AtlasPlane, GlyphAtlas};
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::CStr;
use std::path::Path;

#[derive(Debug)]
pub enum VulkanError {
    LibraryNotFound,
    Vk(vk::Result),
    NoSuitableDevice,
    NoMemoryType,
}

impl From<vk::Result> for VulkanError {
    fn from(e: vk::Result) -> Self {
        Self::Vk(e)
    }
}

pub(crate) struct MappedBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) mapped: *mut u8,
    pub(crate) size: vk::DeviceSize,
    pub(crate) alloc_size: vk::DeviceSize,
    pub(crate) is_coherent: bool,
}

impl MappedBuffer {
    /// Compute a Vulkan-spec-compliant `VkMappedMemoryRange` aligned to `nonCoherentAtomSize`
    /// (`VUID-VkMappedMemoryRange-offset-00687`, `01389`, `01390`). Returns `None` when `is_coherent`.
    #[inline]
    pub(crate) fn aligned_range(
        &self,
        offset: u64,
        len: u64,
        atom_size: u64,
    ) -> Option<vk::MappedMemoryRange<'static>> {
        if self.is_coherent || len == 0 || offset >= self.alloc_size || self.memory == vk::DeviceMemory::null() {
            return None;
        }
        let atom = atom_size.max(1);
        let aligned_offset = (offset / atom) * atom;
        let end = offset.saturating_add(len).min(self.alloc_size);
        let aligned_end = end.div_ceil(atom) * atom;
        let range_size = if aligned_end >= self.alloc_size {
            vk::WHOLE_SIZE
        } else {
            aligned_end - aligned_offset
        };
        Some(
            vk::MappedMemoryRange::default()
                .memory(self.memory)
                .offset(aligned_offset)
                .size(range_size),
        )
    }
}

struct GpuTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    initialized: bool,
}

/// Complete Window Swapchain Presenter with per-image synchronization semaphores!
pub struct VulkanSwapchain {
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    acquire_sem: vk::Semaphore,
    render_done_sems: Vec<vk::Semaphore>,
    custom_pipeline: Option<vk::Pipeline>,
    vsync: bool,
}

/// Native Vulkan 1.3 UI Renderer supporting both Window Swapchains (`VulkanSwapchain` + `raw-window-handle`)
/// and Hardware Offscreen Render Targets (`render_offscreen_rgba8`).
pub struct VulkanRenderer {
    entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family: u32,
    pub device_name: String,
    pub is_rebar_device_local: bool,
    pub dual_source_blending: bool,
    pub(crate) non_coherent_atom_size: vk::DeviceSize,
    pub(crate) device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    desc_set: vk::DescriptorSet,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pub(crate) uniform_buf: MappedBuffer,
    pub(crate) prim_buf: MappedBuffer,
    pub(crate) clip_buf: MappedBuffer,
    pub(crate) staging_buf: MappedBuffer,
    pub(crate) prim_synced_count: usize,
    dirty_span_scratch: Vec<(usize, usize)>,
    flush_scratch: Vec<vk::MappedMemoryRange<'static>>,
    atlas_r8: GpuTexture,
    atlas_rgba8: GpuTexture,
    atlas_gen_uploaded: u64,
    r8_copy_scratch: Vec<vk::BufferImageCopy>,
    rgba_copy_scratch: Vec<vk::BufferImageCopy>,
    offscreen_target: Option<GpuTexture>,
    pub(crate) readback_buf: Option<MappedBuffer>,
}

impl VulkanRenderer {
    /// Initialize a hardware Vulkan 1.3 headless context (automatically discovering system/NixOS Vulkan loaders).
    pub fn new_headless(max_primitives: usize, max_clips: usize) -> Result<Self, VulkanError> {
        Self::init_internal(None, false, max_primitives.max(256), max_clips.max(64))
    }

    /// Initialize a hardware Vulkan 1.3 context with platform window surface & `VK_KHR_swapchain` extensions enabled!
    pub fn new_for_window(
        display: RawDisplayHandle,
        max_primitives: usize,
        max_clips: usize,
    ) -> Result<Self, VulkanError> {
        let exts = ash_window::enumerate_required_extensions(display)?;
        Self::init_internal(
            Some(exts),
            true,
            max_primitives.max(256),
            max_clips.max(64),
        )
    }

    /// Create a Vulkan `VkSurfaceKHR` for any `raw-window-handle` 0.6 window (Wayland, X11, Win32, Metal).
    ///
    /// # Safety
    /// `display` and `window` handles must refer to a live native window outliving the returned surface.
    pub unsafe fn create_window_surface(
        &self,
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<vk::SurfaceKHR, VulkanError> {
        let surface =
            unsafe { ash_window::create_surface(&self.entry, &self.instance, display, window, None)? };
        Ok(surface)
    }

    /// Create a low-latency sRGB `VulkanSwapchain` with per-image presentation synchronization!
    ///
    /// # Safety
    /// `surface` must have been created on `self.instance` from a valid live window.
    pub unsafe fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<VulkanSwapchain, VulkanError> {
        let surface_loader = ash::khr::surface::Instance::new(&self.entry, &self.instance);
        let swapchain_loader = ash::khr::swapchain::Device::new(&self.instance, &self.device);

        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(self.physical_device, surface)?
        };
        if formats.is_empty() {
            return Err(VulkanError::NoSuitableDevice);
        }
        let chosen_fmt = formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f.format,
                    vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                )
            })
            .unwrap_or(formats[0]);

        let acquire_sem =
            unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };

        let custom_pipeline = if chosen_fmt.format != vk::Format::R8G8B8A8_SRGB {
            match Self::build_graphics_pipeline(
                &self.device,
                self.pipeline_layout,
                chosen_fmt.format,
                self.dual_source_blending,
            ) {
                Ok(p) => Some(p),
                Err(e) => {
                    unsafe { self.device.destroy_semaphore(acquire_sem, None) };
                    return Err(e);
                }
            }
        } else {
            None
        };

        let mut sc = VulkanSwapchain {
            surface_loader,
            swapchain_loader,
            surface,
            swapchain: vk::SwapchainKHR::null(),
            format: chosen_fmt.format,
            color_space: chosen_fmt.color_space,
            extent: vk::Extent2D {
                width: width.max(1),
                height: height.max(1),
            },
            images: Vec::new(),
            views: Vec::new(),
            acquire_sem,
            render_done_sems: Vec::new(),
            custom_pipeline,
            vsync,
        };
        if let Err(e) = unsafe { self.recreate_swapchain_images(&mut sc, width.max(1), height.max(1), vsync) } {
            self.destroy_swapchain(sc);
            return Err(e);
        }
        Ok(sc)
    }

    /// Resize or recreate swapchain images cleanly after `SUBOPTIMAL_KHR`, `ERROR_OUT_OF_DATE_KHR`, or window resize.
    /// Safely ignores `0x0` window minimization without violating `VUID-VkSwapchainCreateInfoKHR-imageExtent-01689`.
    ///
    /// # Safety
    /// `sc` must belong to `self`.
    pub unsafe fn recreate_swapchain_images(
        &self,
        sc: &mut VulkanSwapchain,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<(), VulkanError> {
        if width == 0 || height == 0 {
            sc.extent = vk::Extent2D {
                width: 0,
                height: 0,
            };
            return Ok(());
        }

        let caps = unsafe {
            sc.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, sc.surface)?
        };
        let modes = unsafe {
            sc.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, sc.surface)?
        };

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        // Window minimized on Win32 / X11 (`current_extent == 0x0`): defer swapchain creation until restored.
        if extent.width == 0 || extent.height == 0 {
            sc.extent = vk::Extent2D {
                width: 0,
                height: 0,
            };
            if sc.swapchain == vk::SwapchainKHR::null() {
                return Err(VulkanError::Vk(vk::Result::ERROR_OUT_OF_DATE_KHR));
            }
            return Ok(());
        }

        unsafe { self.device.device_wait_idle()? };
        for v in sc.views.drain(..) {
            unsafe { self.device.destroy_image_view(v, None) };
        }
        for sem in sc.render_done_sems.drain(..) {
            unsafe { self.device.destroy_semaphore(sem, None) };
        }
        if sc.acquire_sem != vk::Semaphore::null() {
            unsafe { self.device.destroy_semaphore(sc.acquire_sem, None) };
            sc.acquire_sem = vk::Semaphore::null();
        }
        sc.acquire_sem =
            unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };

        let present_mode = if !vsync && modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if !vsync && modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        };

        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|&flag| caps.supported_composite_alpha.contains(flag))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

        let mut image_count = caps.min_image_count.max(2);
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let old_sc = sc.swapchain;
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(sc.surface)
            .min_image_count(image_count)
            .image_format(sc.format)
            .image_color_space(sc.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_sc);

        let new_sc = unsafe { sc.swapchain_loader.create_swapchain(&create_info, None)? };
        if old_sc != vk::SwapchainKHR::null() {
            unsafe { sc.swapchain_loader.destroy_swapchain(old_sc, None) };
        }

        let images = unsafe { sc.swapchain_loader.get_swapchain_images(new_sc)? };
        let mut views = Vec::with_capacity(images.len());
        for &img in &images {
            let v = unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(img)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(sc.format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )?
            };
            views.push(v);
        }

        let mut render_done_sems = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            let sem =
                unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
            render_done_sems.push(sem);
        }

        sc.swapchain = new_sc;
        sc.extent = extent;
        sc.images = images;
        sc.views = views;
        sc.render_done_sems = render_done_sems;
        sc.vsync = vsync;
        Ok(())
    }

    /// Upload only dirty `GpuPrimitive` spans via ReBAR and render a full coherent swapchain image!
    ///
    /// Note: Unlike single-image offscreen render targets (`VulkanRenderer::render`), `VkSwapchainKHR`
    /// rotates across 3–5 compositor-managed `wl_buffer`/WSI images without `EXT_buffer_age` guarantees;
    /// rendering the full swapchain `render_area` in a single ~25 µs instanced draw guarantees zero
    /// partial-backbuffer flicker across Wayland/X11/macOS/Windows while still writing only dirty
    /// 64-byte primitives to VRAM!
    pub fn present_frame(
        &mut self,
        sc: &mut VulkanSwapchain,
        _damage_tracker: &mut DamageTracker,
        frame: &RenderFrame<'_>,
        atlas: &mut GlyphAtlas,
    ) -> Result<bool, VulkanError> {
        if sc.swapchain == vk::SwapchainKHR::null()
            || sc.extent.width == 0
            || sc.extent.height == 0
            || frame.width == 0
            || frame.height == 0
        {
            return Ok(true);
        }

        unsafe {
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        // Grow scene SSBOs / atlas textures / staging buffer AND upload host/ReBAR scene buffers
        // BEFORE acquiring swapchain image or resetting fence, minimizing acquire-to-present VBlank latency
        // and ensuring early errors never strand `sc.acquire_sem` signaled or `self.fence` unsignaled!
        let grew = self.ensure_scene_capacity(frame.primitives.len(), frame.clips.len())?;
        self.ensure_atlas_and_staging(atlas)?;

        let w = sc.extent.width.max(1);
        let h = sc.extent.height.max(1);
        let full_rect = DamageRect {
            x: 0,
            y: 0,
            w,
            h,
        };
        self.upload_scene_buffers(frame, w, h, full_rect, grew)?;

        let (img_idx, suboptimal_acq) = unsafe {
            sc.swapchain_loader.acquire_next_image(
                sc.swapchain,
                u64::MAX,
                sc.acquire_sem,
                vk::Fence::null(),
            )?
        };
        let slot = img_idx as usize;

        unsafe {
            self.device
                .reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.cmd_buf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        self.record_atlas_uploads(atlas)?;

        let img = sc.images[slot];
        let view = sc.views[slot];
        let done_sem = sc.render_done_sems[slot];

        record_image_barrier(
            &self.device,
            self.cmd_buf,
            img,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::NONE,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        let active_pipeline = sc.custom_pipeline.unwrap_or(self.pipeline);
        let max_prims = (self.prim_buf.size as usize) / core::mem::size_of::<GpuPrimitive>();
        let prim_count = frame.primitives.len().min(max_prims);
        self.record_dynamic_render_pass(
            view,
            w,
            h,
            full_rect,
            frame.clear_color,
            active_pipeline,
            prim_count,
        );

        record_image_barrier(
            &self.device,
            self.cmd_buf,
            img,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        );

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let suboptimal = unsafe {
            self.device.end_command_buffer(self.cmd_buf)?;
            self.device.reset_fences(&[self.fence])?;
            let submit = [vk::SubmitInfo::default()
                .wait_semaphores(core::slice::from_ref(&sc.acquire_sem))
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(core::slice::from_ref(&self.cmd_buf))
                .signal_semaphores(core::slice::from_ref(&done_sem))];
            self.device.queue_submit(self.queue, &submit, self.fence)?;

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(core::slice::from_ref(&done_sem))
                .swapchains(core::slice::from_ref(&sc.swapchain))
                .image_indices(core::slice::from_ref(&img_idx));
            let suboptimal_pres = sc.swapchain_loader.queue_present(self.queue, &present_info)?;
            suboptimal_acq || suboptimal_pres
        };

        if suboptimal {
            let _ = unsafe {
                self.recreate_swapchain_images(sc, frame.width, frame.height, sc.vsync)
            };
        }
        Ok(suboptimal)
    }

    /// Cleanly destroy a `VulkanSwapchain` and its `VkSurfaceKHR` before dropping `VulkanRenderer`.
    pub fn destroy_swapchain(&self, mut sc: VulkanSwapchain) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(p) = sc.custom_pipeline.take() {
                self.device.destroy_pipeline(p, None);
            }
            for v in sc.views.drain(..) {
                self.device.destroy_image_view(v, None);
            }
            if sc.swapchain != vk::SwapchainKHR::null() {
                sc.swapchain_loader.destroy_swapchain(sc.swapchain, None);
                sc.swapchain = vk::SwapchainKHR::null();
            }
            if sc.acquire_sem != vk::Semaphore::null() {
                self.device.destroy_semaphore(sc.acquire_sem, None);
                sc.acquire_sem = vk::Semaphore::null();
            }
            for sem in sc.render_done_sems.drain(..) {
                self.device.destroy_semaphore(sem, None);
            }
            if sc.surface != vk::SurfaceKHR::null() {
                sc.surface_loader.destroy_surface(sc.surface, None);
                sc.surface = vk::SurfaceKHR::null();
            }
        }
    }

    fn load_vulkan_entry() -> Result<ash::Entry, VulkanError> {
        let nixos_icd_dir = Path::new("/run/opengl-driver/share/vulkan/icd.d");
        if std::env::var_os("VK_ICD_FILENAMES").is_none()
            && std::env::var_os("VK_DRIVER_FILES").is_none()
            && nixos_icd_dir.is_dir()
            && let Ok(entries) = std::fs::read_dir(nixos_icd_dir)
        {
            let mut paths = Vec::new();
            let mut lvp = Vec::new();
            for ent in entries.flatten() {
                let p = ent.path();
                if p.extension().is_some_and(|e| e == "json") {
                    if p.file_name()
                        .is_some_and(|n| n.to_string_lossy().contains("lvp"))
                    {
                        lvp.push(p);
                    } else {
                        paths.push(p);
                    }
                }
            }
            paths.extend(lvp);
            if !paths.is_empty()
                && let Ok(joined) = std::env::join_paths(&paths)
            {
                // SAFETY: Only sets fallback Vulkan loader discovery env before `vkCreateInstance` if unset.
                unsafe { std::env::set_var("VK_DRIVER_FILES", joined) };
            }
        }

        if let Ok(entry) = unsafe { ash::Entry::load() } {
            return Ok(entry);
        }

        let candidates = [
            "/run/opengl-driver/lib/libvulkan.so.1",
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/usr/lib64/libvulkan.so.1",
        ];
        for cand in candidates {
            if Path::new(cand).exists()
                && let Ok(entry) = unsafe { ash::Entry::load_from(cand) }
            {
                return Ok(entry);
            }
        }
        if let Ok(store) = std::fs::read_dir("/nix/store") {
            for ent in store.flatten() {
                let name = ent.file_name();
                if name.to_string_lossy().contains("vulkan-loader-") {
                    let lib = ent.path().join("lib/libvulkan.so.1");
                    if lib.exists()
                        && let Ok(entry) = unsafe { ash::Entry::load_from(lib) }
                    {
                        return Ok(entry);
                    }
                }
            }
        }
        Err(VulkanError::LibraryNotFound)
    }

    fn init_internal(
        surface_exts: Option<&[*const i8]>,
        enable_swapchain_ext: bool,
        max_primitives: usize,
        max_clips: usize,
    ) -> Result<Self, VulkanError> {
        let entry = Self::load_vulkan_entry()?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"skima")
            .engine_name(c"skima-iale")
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let exts = surface_exts.unwrap_or(&[]);
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(exts);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let pdevices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(p) => p,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(VulkanError::Vk(e));
            }
        };
        let mut chosen = None;
        let mut best_score = -1i32;

        for pd in pdevices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.api_version < vk::make_api_version(0, 1, 3, 0) {
                continue;
            }
            if enable_swapchain_ext {
                let Ok(dev_exts) =
                    (unsafe { instance.enumerate_device_extension_properties(pd) })
                else {
                    continue;
                };
                let has_swapchain = dev_exts.iter().any(|ext| {
                    let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                    name == ash::khr::swapchain::NAME
                });
                if !has_swapchain {
                    continue;
                }
            }
            let qfamilies =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(q_idx) = qfamilies
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            else {
                continue;
            };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 200,
                vk::PhysicalDeviceType::CPU => 50,
                _ => 10,
            };
            if score > best_score {
                best_score = score;
                chosen = Some((pd, q_idx as u32, props));
            }
        }

        let Some((physical_device, queue_family, props)) = chosen else {
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::NoSuitableDevice);
        };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let non_coherent_atom_size = props.limits.non_coherent_atom_size.max(1);

        let supported_features = unsafe { instance.get_physical_device_features(physical_device) };
        let dual_source_blending = supported_features.dual_src_blend == vk::TRUE;

        let priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];

        let enabled_v10 =
            vk::PhysicalDeviceFeatures::default().dual_src_blend(dual_source_blending);
        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let dev_exts: &[*const i8] = if enable_swapchain_ext {
            &[ash::khr::swapchain::NAME.as_ptr()]
        } else {
            &[]
        };

        let dev_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(dev_exts)
            .enabled_features(&enabled_v10)
            .push_next(&mut vk13_features);

        let device = match unsafe { instance.create_device(physical_device, &dev_info, None) } {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(VulkanError::Vk(e));
            }
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_alloc)? }[0];

        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        let mem_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let (uniform_buf, _) = Self::create_mapped_buffer(
            &device,
            &mem_props,
            core::mem::size_of::<ViewportUniform>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            true,
            false,
        )?;
        let (prim_buf, is_rebar_device_local) = Self::create_mapped_buffer(
            &device,
            &mem_props,
            (max_primitives * core::mem::size_of::<GpuPrimitive>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
            false,
        )?;
        let (clip_buf, _) = Self::create_mapped_buffer(
            &device,
            &mem_props,
            (max_clips * core::mem::size_of::<GpuClip>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
            false,
        )?;
        let (staging_buf, _) = Self::create_mapped_buffer(
            &device,
            &mem_props,
            8 * 1024 * 1024,
            vk::BufferUsageFlags::TRANSFER_SRC,
            false,
            false,
        )?;

        let atlas_r8 = Self::create_texture(
            &device,
            &mem_props,
            1024,
            1024,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let atlas_rgba8 = Self::create_texture(
            &device,
            &mem_props,
            1024,
            1024,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let desc_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
            },
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        let desc_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(core::slice::from_ref(&desc_layout)),
            )?
        }[0];

        let u_info = [vk::DescriptorBufferInfo {
            buffer: uniform_buf.buffer,
            offset: 0,
            range: uniform_buf.size,
        }];
        let p_info = [vk::DescriptorBufferInfo {
            buffer: prim_buf.buffer,
            offset: 0,
            range: prim_buf.size,
        }];
        let c_info = [vk::DescriptorBufferInfo {
            buffer: clip_buf.buffer,
            offset: 0,
            range: clip_buf.size,
        }];
        let r8_info = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: atlas_r8.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let rgba_info = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: atlas_rgba8.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let samp_info = [vk::DescriptorImageInfo {
            sampler,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        }];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&u_info),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&p_info),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&c_info),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&r8_info),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&rgba_info),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&samp_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(core::slice::from_ref(&desc_layout)),
                None,
            )?
        };

        let pipeline = Self::build_graphics_pipeline(
            &device,
            pipeline_layout,
            vk::Format::R8G8B8A8_SRGB,
            dual_source_blending,
        )?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            queue_family,
            device_name,
            is_rebar_device_local,
            dual_source_blending,
            non_coherent_atom_size,
            device,
            queue,
            command_pool,
            cmd_buf,
            fence,
            descriptor_pool,
            desc_layout,
            desc_set,
            pipeline_layout,
            pipeline,
            sampler,
            uniform_buf,
            prim_buf,
            clip_buf,
            staging_buf,
            prim_synced_count: 0,
            dirty_span_scratch: Vec::with_capacity(64),
            flush_scratch: Vec::with_capacity(16),
            atlas_r8,
            atlas_rgba8,
            atlas_gen_uploaded: u64::MAX,
            r8_copy_scratch: Vec::with_capacity(64),
            rgba_copy_scratch: Vec::with_capacity(64),
            offscreen_target: None,
            readback_buf: None,
        })
    }

    pub(crate) fn build_graphics_pipeline(
        device: &ash::Device,
        pipeline_layout: vk::PipelineLayout,
        target_format: vk::Format,
        use_dual_src: bool,
    ) -> Result<vk::Pipeline, VulkanError> {
        let vert_mod = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(UBER_VERT_SPV),
                None,
            )?
        };
        let (frag_spv, frag_entry): (&[u32], &CStr) = if use_dual_src {
            (UBER_FRAG_DUAL_SRC_SPV, c"fs_main_dual_src")
        } else {
            (UBER_FRAG_SPV, c"fs_main")
        };
        let frag_mod = match unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(frag_spv), None)
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_shader_module(vert_mod, None) };
                return Err(VulkanError::Vk(e));
            }
        };

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_mod)
                .name(c"vs_main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_mod)
                .name(frag_entry),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let blend_attach = [if use_dual_src {
            // Hardware Dual-Source Blending: exact per-channel RGB ClearType + premultiplied alpha in ONE pipeline!
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC1_COLOR)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC1_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        } else {
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        }];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attach);

        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

        let color_formats = [target_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats);

        let gp_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);

        let pipeline_res = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[gp_info], None)
                .map_err(|(_, e)| e)
        };

        unsafe {
            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);
        }
        Ok(pipeline_res?[0])
    }

    /// Execute hardware Vulkan 1.3 rendering into an sRGB GPU image and read back the RGBA8 pixels.
    pub fn render_offscreen_rgba8(
        &mut self,
        frame: &RenderFrame<'_>,
        atlas: &mut GlyphAtlas,
    ) -> Result<Vec<u8>, VulkanError> {
        if frame.width == 0 || frame.height == 0 {
            return Ok(Vec::new());
        }
        let full_frame = RenderFrame {
            width: frame.width,
            height: frame.height,
            scale_factor: frame.scale_factor,
            clear_color: frame.clear_color,
            primitives: frame.primitives,
            dirty_prim_ranges: frame.dirty_prim_ranges,
            clips: frame.clips,
            damage: frame.damage.or(Some(DamageRect {
                x: 0,
                y: 0,
                w: frame.width,
                h: frame.height,
            })),
        };
        self.render_internal(&full_frame, atlas, true)?;
        let byte_len = (frame.width as usize) * (frame.height as usize) * 4;
        let rb = self.readback_buf.as_ref().unwrap();
        if let Some(range) = rb.aligned_range(0, byte_len as u64, self.non_coherent_atom_size) {
            unsafe {
                self.device
                    .invalidate_mapped_memory_ranges(core::slice::from_ref(&range))?;
            }
        }
        let mut out = vec![0u8; byte_len];
        unsafe {
            core::ptr::copy_nonoverlapping(rb.mapped, out.as_mut_ptr(), byte_len);
        }
        Ok(out)
    }

    fn ensure_offscreen_target(&mut self, width: u32, height: u32) -> Result<(), VulkanError> {
        let needs_recreate = self
            .offscreen_target
            .as_ref()
            .is_none_or(|t| t.width != width || t.height != height);
        if !needs_recreate {
            return Ok(());
        }
        unsafe {
            self.device.device_wait_idle()?;
        }
        if let Some(old) = self.offscreen_target.take() {
            Self::destroy_texture(&self.device, old);
        }
        if let Some(old_rb) = self.readback_buf.take() {
            Self::destroy_mapped_buffer(&self.device, old_rb);
        }
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let tex = Self::create_texture(
            &self.device,
            &mem_props,
            width,
            height,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;
        let (rb, _) = Self::create_mapped_buffer(
            &self.device,
            &mem_props,
            (width as u64) * (height as u64) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            false,
            true,
        )?;
        self.offscreen_target = Some(tex);
        self.readback_buf = Some(rb);
        Ok(())
    }

    pub(crate) fn create_mapped_buffer(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
        usage: vk::BufferUsageFlags,
        prefer_rebar: bool,
        prefer_cached: bool,
    ) -> Result<(MappedBuffer, bool), VulkanError> {
        let size = size.max(64);
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buf_info, None)? };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };

        let rebar_coh = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let rebar_noncoh =
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        let cached_coh = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_CACHED
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached_noncoh =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED;
        let host_coh =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let host_vis = vk::MemoryPropertyFlags::HOST_VISIBLE;

        let mem_idx_opt = if prefer_cached {
            find_mem_type(mem_props, req.memory_type_bits, cached_coh)
                .or_else(|| find_mem_type(mem_props, req.memory_type_bits, cached_noncoh))
                .or_else(|| find_mem_type(mem_props, req.memory_type_bits, host_coh))
                .or_else(|| find_mem_type(mem_props, req.memory_type_bits, host_vis))
        } else if prefer_rebar {
            find_mem_type(mem_props, req.memory_type_bits, rebar_coh)
                .or_else(|| find_mem_type(mem_props, req.memory_type_bits, rebar_noncoh))
                .or_else(|| find_mem_type(mem_props, req.memory_type_bits, host_coh))
                .or_else(|| find_mem_type(mem_props, req.memory_type_bits, host_vis))
        } else {
            find_mem_type(mem_props, req.memory_type_bits, host_coh)
                .or_else(|| find_mem_type(mem_props, req.memory_type_bits, host_vis))
        };

        let Some(mem_idx) = mem_idx_opt else {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(VulkanError::NoMemoryType);
        };

        let chosen_flags = mem_props.memory_types[mem_idx as usize].property_flags;
        let is_rebar = chosen_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let is_coherent = chosen_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_idx);
        let memory = match unsafe { device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(VulkanError::Vk(e));
            }
        };
        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(VulkanError::Vk(e));
        }
        // Map `vk::WHOLE_SIZE` (`req.size`) so `nonCoherentAtomSize`-aligned flush/invalidate ranges
        // up to `alloc_size` always lie strictly inside the mapped range (`VUID-VkMappedMemoryRange-size-01389`).
        let mapped = match unsafe {
            device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => ptr as *mut u8,
            Err(e) => {
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                return Err(VulkanError::Vk(e));
            }
        };

        Ok((
            MappedBuffer {
                buffer,
                memory,
                mapped,
                size,
                alloc_size: req.size,
                is_coherent,
            },
            is_rebar,
        ))
    }

    fn create_texture(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<GpuTexture, VulkanError> {
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);
        let image = unsafe { device.create_image(&img_info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let Some(mem_idx) = find_mem_type(
            mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| find_mem_type(mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::empty()))
        else {
            unsafe { device.destroy_image(image, None) };
            return Err(VulkanError::NoMemoryType);
        };

        let memory = match unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_idx),
                None,
            )
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                return Err(VulkanError::Vk(e));
            }
        };
        if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(VulkanError::Vk(e));
        }

        let view = match unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        } {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    device.destroy_image(image, None);
                    device.free_memory(memory, None);
                }
                return Err(VulkanError::Vk(e));
            }
        };

        Ok(GpuTexture {
            image,
            memory,
            view,
            width,
            height,
            initialized: false,
        })
    }

    pub(crate) fn destroy_mapped_buffer(device: &ash::Device, buf: MappedBuffer) {
        unsafe {
            if !buf.mapped.is_null() && buf.memory != vk::DeviceMemory::null() {
                device.unmap_memory(buf.memory);
            }
            if buf.buffer != vk::Buffer::null() {
                device.destroy_buffer(buf.buffer, None);
            }
            if buf.memory != vk::DeviceMemory::null() {
                device.free_memory(buf.memory, None);
            }
        }
    }

    fn destroy_texture(device: &ash::Device, tex: GpuTexture) {
        unsafe {
            if tex.view != vk::ImageView::null() {
                device.destroy_image_view(tex.view, None);
            }
            if tex.image != vk::Image::null() {
                device.destroy_image(tex.image, None);
            }
            if tex.memory != vk::DeviceMemory::null() {
                device.free_memory(tex.memory, None);
            }
        }
    }
}

impl RenderBackend for VulkanRenderer {
    type Error = VulkanError;

    fn render(&mut self, frame: &RenderFrame<'_>, atlas: &mut GlyphAtlas) -> Result<(), Self::Error> {
        if frame.width == 0 || frame.height == 0 {
            return Ok(());
        }
        self.render_internal(frame, atlas, false)
    }
}

impl VulkanRenderer {
    /// Dynamically grow mapped ReBAR `prim_buf` or `clip_buf` if the scene exceeds current capacity
    /// (zero silent primitive or clip truncation at arbitrary UI scale!).
    fn ensure_scene_capacity(
        &mut self,
        needed_prims: usize,
        needed_clips: usize,
    ) -> Result<bool, VulkanError> {
        let cur_prims = (self.prim_buf.size as usize) / core::mem::size_of::<GpuPrimitive>();
        let cur_clips = (self.clip_buf.size as usize) / core::mem::size_of::<GpuClip>();
        if needed_prims <= cur_prims && needed_clips <= cur_clips {
            return Ok(false);
        }

        unsafe {
            self.device.device_wait_idle()?;
        }
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        let mut grew_prims = false;
        if needed_prims > cur_prims {
            let new_cap = needed_prims.next_power_of_two().max(512);
            let (new_buf, is_rebar) = Self::create_mapped_buffer(
                &self.device,
                &mem_props,
                (new_cap * core::mem::size_of::<GpuPrimitive>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
                false,
            )?;
            let old = core::mem::replace(&mut self.prim_buf, new_buf);
            Self::destroy_mapped_buffer(&self.device, old);
            self.is_rebar_device_local = is_rebar;
            self.prim_synced_count = 0;
            grew_prims = true;
        }

        if needed_clips > cur_clips {
            let new_cap = needed_clips.next_power_of_two().max(128);
            let (new_buf, _) = Self::create_mapped_buffer(
                &self.device,
                &mem_props,
                (new_cap * core::mem::size_of::<GpuClip>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
                false,
            )?;
            let old = core::mem::replace(&mut self.clip_buf, new_buf);
            Self::destroy_mapped_buffer(&self.device, old);
        }

        let p_info = [vk::DescriptorBufferInfo {
            buffer: self.prim_buf.buffer,
            offset: 0,
            range: self.prim_buf.size,
        }];
        let c_info = [vk::DescriptorBufferInfo {
            buffer: self.clip_buf.buffer,
            offset: 0,
            range: self.clip_buf.size,
        }];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&p_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&c_info),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(grew_prims)
    }

    /// Ensure GPU `atlas_r8` / `atlas_rgba8` match `atlas.width x atlas.height` and dynamically grow
    /// `self.staging_buf` whenever full or dirty-rect upload bytes exceed current `staging_buf.size`.
    fn ensure_atlas_and_staging(&mut self, atlas: &GlyphAtlas) -> Result<(), VulkanError> {
        let aw = (atlas.width as u32).max(1);
        let ah = (atlas.height as u32).max(1);
        if self.atlas_r8.width != aw || self.atlas_r8.height != ah {
            unsafe {
                self.device.device_wait_idle()?;
            }
            let mem_props = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical_device)
            };
            let new_r8 = Self::create_texture(
                &self.device,
                &mem_props,
                aw,
                ah,
                vk::Format::R8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            let new_rgba = match Self::create_texture(
                &self.device,
                &mem_props,
                aw,
                ah,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ) {
                Ok(t) => t,
                Err(e) => {
                    Self::destroy_texture(&self.device, new_r8);
                    return Err(e);
                }
            };
            let old_r8 = core::mem::replace(&mut self.atlas_r8, new_r8);
            let old_rgba = core::mem::replace(&mut self.atlas_rgba8, new_rgba);
            Self::destroy_texture(&self.device, old_r8);
            Self::destroy_texture(&self.device, old_rgba);

            let r8_info = [vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: self.atlas_r8.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let rgba_info = [vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: self.atlas_rgba8.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.desc_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&r8_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.desc_set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&rgba_info),
            ];
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
            self.atlas_gen_uploaded = u64::MAX;
        }

        let force_full = !self.atlas_r8.initialized
            || !self.atlas_rgba8.initialized
            || self.atlas_gen_uploaded != atlas.generation;
        let needed_bytes = Self::compute_atlas_staging_bytes(atlas, force_full);
        self.ensure_staging_capacity(needed_bytes)
    }

    fn compute_atlas_staging_bytes(atlas: &GlyphAtlas, force_full: bool) -> usize {
        if atlas.width == 0 || atlas.height == 0 {
            return 0;
        }
        if force_full {
            let r8_aligned = (atlas.r8_data.len() + 3) & !3;
            r8_aligned.saturating_add(atlas.rgba8_data.len())
        } else {
            let aw = atlas.width as usize;
            let ah = atlas.height as usize;
            let mut cursor = 0usize;
            for rect in &atlas.dirty_rects {
                let x = (rect.x as usize).min(aw);
                let y = (rect.y as usize).min(ah);
                let rw = (rect.w as usize).min(aw - x);
                let rh = (rect.h as usize).min(ah - y);
                if rw == 0 || rh == 0 {
                    continue;
                }
                let bpp = match rect.plane {
                    AtlasPlane::R8 => 1usize,
                    AtlasPlane::Rgba8 => 4usize,
                };
                let aligned = (cursor + 3) & !3;
                cursor = aligned.saturating_add(rw * rh * bpp);
            }
            cursor
        }
    }

    fn ensure_staging_capacity(&mut self, needed_bytes: usize) -> Result<(), VulkanError> {
        if needed_bytes <= self.staging_buf.size as usize {
            return Ok(());
        }
        let new_size = (needed_bytes.next_power_of_two().max(64 * 1024)) as u64;
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let (mut new_buf, _) = Self::create_mapped_buffer(
            &self.device,
            &mem_props,
            new_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            false,
            false,
        )?;
        // Preserve explicit non-coherent mode if forced under test
        if !self.staging_buf.is_coherent {
            new_buf.is_coherent = false;
        }
        let old = core::mem::replace(&mut self.staging_buf, new_buf);
        Self::destroy_mapped_buffer(&self.device, old);
        Ok(())
    }

    fn upload_scene_buffers(
        &mut self,
        frame: &RenderFrame<'_>,
        w: u32,
        h: u32,
        scissor_rect: DamageRect,
        force_full_prim_sync: bool,
    ) -> Result<(), VulkanError> {
        let atom = self.non_coherent_atom_size.max(1);
        self.flush_scratch.clear();

        // 1. Write 32-byte ViewportUniform (with physical scissor bounds clamped to viewport for vertex-stage culling!)
        let sx0 = scissor_rect.x.min(w);
        let sy0 = scissor_rect.y.min(h);
        let sx1 = scissor_rect.x.saturating_add(scissor_rect.w).min(w);
        let sy1 = scissor_rect.y.saturating_add(scissor_rect.h).min(h);
        let uniform = ViewportUniform {
            transform: [2.0 / (w as f32), 2.0 / (h as f32), frame.scale_factor, 0.0],
            scissor: [sx0 as f32, sy0 as f32, sx1 as f32, sy1 as f32],
        };
        unsafe {
            core::ptr::write(self.uniform_buf.mapped.cast::<ViewportUniform>(), uniform);
        }
        if let Some(r) = self.uniform_buf.aligned_range(
            0,
            core::mem::size_of::<ViewportUniform>() as u64,
            atom,
        ) {
            self.flush_scratch.push(r);
        }

        // 2. Write ONLY coalesced dirty `GpuPrimitive` spans directly into Persistently Mapped ReBAR/Host memory!
        let prim_stride = core::mem::size_of::<GpuPrimitive>();
        let max_prims = (self.prim_buf.size as usize) / prim_stride;
        let prim_count = frame.primitives.len().min(max_prims);
        let need_full_prims = force_full_prim_sync
            || frame.dirty_prim_ranges.is_empty()
            || prim_count != self.prim_synced_count;
        if need_full_prims {
            if prim_count > 0 {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        frame.primitives.as_ptr(),
                        self.prim_buf.mapped.cast::<GpuPrimitive>(),
                        prim_count,
                    );
                }
                if let Some(r) = self
                    .prim_buf
                    .aligned_range(0, (prim_count * prim_stride) as u64, atom)
                {
                    self.flush_scratch.push(r);
                }
            }
            self.prim_synced_count = prim_count;
        } else {
            let dst_base = self.prim_buf.mapped.cast::<GpuPrimitive>();
            let src_base = frame.primitives.as_ptr();
            if frame.dirty_prim_ranges.len() == 1 {
                let span = frame.dirty_prim_ranges[0];
                let s = (span.start as usize).min(prim_count);
                let e = (span.end as usize).min(prim_count);
                if e > s {
                    unsafe {
                        core::ptr::copy_nonoverlapping(src_base.add(s), dst_base.add(s), e - s);
                    }
                    if let Some(r) = self.prim_buf.aligned_range(
                        (s * prim_stride) as u64,
                        ((e - s) * prim_stride) as u64,
                        atom,
                    ) {
                        self.flush_scratch.push(r);
                    }
                }
            } else {
                // Coalesce unsorted, overlapping, or adjacent dirty spans into minimal disjoint ReBAR burst copies:
                self.dirty_span_scratch.clear();
                let mut sorted_disjoint = true;
                let mut prev_end = 0usize;
                for span in frame.dirty_prim_ranges {
                    let s = (span.start as usize).min(prim_count);
                    let e = (span.end as usize).min(prim_count);
                    if e > s {
                        if !self.dirty_span_scratch.is_empty() && s <= prev_end {
                            sorted_disjoint = false;
                        }
                        self.dirty_span_scratch.push((s, e));
                        prev_end = e;
                    }
                }
                if !sorted_disjoint {
                    self.dirty_span_scratch.sort_unstable_by_key(|&(s, _)| s);
                    let mut write_idx = 0usize;
                    for i in 1..self.dirty_span_scratch.len() {
                        let (s, e) = self.dirty_span_scratch[i];
                        if s <= self.dirty_span_scratch[write_idx].1 {
                            self.dirty_span_scratch[write_idx].1 =
                                self.dirty_span_scratch[write_idx].1.max(e);
                        } else {
                            write_idx += 1;
                            self.dirty_span_scratch[write_idx] = (s, e);
                        }
                    }
                    self.dirty_span_scratch.truncate(write_idx + 1);
                }

                // Copy each coalesced primitive span and merge overlapping/contiguous `nonCoherentAtomSize` flush atoms:
                let track_flush = !self.prim_buf.is_coherent
                    && self.prim_buf.memory != vk::DeviceMemory::null();
                let alloc_sz = self.prim_buf.alloc_size;
                let mut cur_flush: Option<(u64, u64)> = None;

                for &(s, e) in &self.dirty_span_scratch {
                    unsafe {
                        core::ptr::copy_nonoverlapping(src_base.add(s), dst_base.add(s), e - s);
                    }
                    if track_flush {
                        let byte_off = (s * prim_stride) as u64;
                        let byte_end = ((e * prim_stride) as u64).min(alloc_sz);
                        if byte_end > byte_off {
                            let aligned_off = (byte_off / atom) * atom;
                            let aligned_end = byte_end.div_ceil(atom) * atom;
                            if let Some((f_off, f_end)) = cur_flush {
                                if aligned_off <= f_end {
                                    cur_flush = Some((f_off, f_end.max(aligned_end)));
                                } else {
                                    let r_sz = if f_end >= alloc_sz {
                                        vk::WHOLE_SIZE
                                    } else {
                                        f_end - f_off
                                    };
                                    self.flush_scratch.push(
                                        vk::MappedMemoryRange::default()
                                            .memory(self.prim_buf.memory)
                                            .offset(f_off)
                                            .size(r_sz),
                                    );
                                    cur_flush = Some((aligned_off, aligned_end));
                                }
                            } else {
                                cur_flush = Some((aligned_off, aligned_end));
                            }
                        }
                    }
                }
                if let Some((f_off, f_end)) = cur_flush {
                    let r_sz = if f_end >= alloc_sz {
                        vk::WHOLE_SIZE
                    } else {
                        f_end - f_off
                    };
                    self.flush_scratch.push(
                        vk::MappedMemoryRange::default()
                            .memory(self.prim_buf.memory)
                            .offset(f_off)
                            .size(r_sz),
                    );
                }
            }
        }

        // 3. Write `GpuClip` array:
        let clip_stride = core::mem::size_of::<GpuClip>();
        let max_clips = (self.clip_buf.size as usize) / clip_stride;
        let clip_count = frame.clips.len().min(max_clips);
        if clip_count > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    frame.clips.as_ptr(),
                    self.clip_buf.mapped.cast::<GpuClip>(),
                    clip_count,
                );
            }
            if let Some(r) = self
                .clip_buf
                .aligned_range(0, (clip_count * clip_stride) as u64, atom)
            {
                self.flush_scratch.push(r);
            }
        }

        if !self.flush_scratch.is_empty() {
            unsafe {
                self.device.flush_mapped_memory_ranges(&self.flush_scratch)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_dynamic_render_pass(
        &self,
        view: vk::ImageView,
        w: u32,
        h: u32,
        scissor_rect: DamageRect,
        clear_color: crate::color::Color,
        active_pipeline: vk::Pipeline,
        prim_count: usize,
    ) {
        let clear_lin = clear_color.to_linear_rgba();
        // Per Vulkan 1.3 Dynamic Rendering (`VkRenderingInfo::renderArea`), `AttachmentLoadOp::CLEAR`
        // clears ONLY `render_area` (`scissor_rect`) and preserves all pixels outside `render_area`,
        // preventing ghost trails and translucent alpha double-accumulation on partial-damage redraws!
        let color_attach = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [
                        clear_lin[0] * clear_lin[3],
                        clear_lin[1] * clear_lin[3],
                        clear_lin[2] * clear_lin[3],
                        clear_lin[3],
                    ],
                },
            })];

        let sx = scissor_rect.x.min(w);
        let sy = scissor_rect.y.min(h);
        let render_area = vk::Rect2D {
            offset: vk::Offset2D {
                x: sx as i32,
                y: sy as i32,
            },
            extent: vk::Extent2D {
                width: scissor_rect.w.min(w - sx),
                height: scissor_rect.h.min(h - sy),
            },
        };

        if render_area.extent.width == 0 || render_area.extent.height == 0 {
            return;
        }

        let render_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attach);

        unsafe {
            self.device.cmd_begin_rendering(self.cmd_buf, &render_info);
            self.device.cmd_bind_pipeline(
                self.cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                active_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.desc_set],
                &[],
            );
            self.device.cmd_set_viewport(
                self.cmd_buf,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: w as f32,
                    height: h as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device
                .cmd_set_scissor(self.cmd_buf, 0, &[render_area]);

            if prim_count > 0 {
                // Single instanced draw call for ALL shadows, rounded boxes, OKLab gradients, borders, and ClearType LCD glyphs!
                self.device
                    .cmd_draw(self.cmd_buf, 6, prim_count as u32, 0, 0);
            }
            self.device.cmd_end_rendering(self.cmd_buf);
        }
    }

    pub(crate) fn render_internal_with_pipeline(
        &mut self,
        frame: &RenderFrame<'_>,
        atlas: &mut GlyphAtlas,
        readback_to_cpu: bool,
        pipeline_override: Option<vk::Pipeline>,
    ) -> Result<(), VulkanError> {
        let w = frame.width.max(1);
        let h = frame.height.max(1);
        self.ensure_offscreen_target(w, h)?;

        let target_already_init = self
            .offscreen_target
            .as_ref()
            .is_some_and(|t| t.initialized);
        if !readback_to_cpu && target_already_init && frame.damage.is_none() {
            if !frame.dirty_prim_ranges.is_empty() || frame.primitives.len() != self.prim_synced_count {
                self.prim_synced_count = 0;
            }
            return Ok(());
        }

        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        let scissor_rect = frame.damage.unwrap_or(DamageRect {
            x: 0,
            y: 0,
            w,
            h,
        });
        let effective_scissor = if !target_already_init {
            DamageRect {
                x: 0,
                y: 0,
                w,
                h,
            }
        } else {
            scissor_rect
        };
        let grew = self.ensure_scene_capacity(frame.primitives.len(), frame.clips.len())?;
        self.ensure_atlas_and_staging(atlas)?;
        self.upload_scene_buffers(frame, w, h, effective_scissor, grew)?;

        unsafe {
            self.device
                .reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.cmd_buf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        self.record_atlas_uploads(atlas)?;

        let (target_img, target_view, was_init) = {
            let target = self.offscreen_target.as_mut().unwrap();
            let init = target.initialized;
            target.initialized = true;
            (target.image, target.view, init)
        };

        record_image_barrier(
            &self.device,
            self.cmd_buf,
            target_img,
            if was_init {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            } else {
                vk::ImageLayout::UNDEFINED
            },
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            if was_init {
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags2::NONE
            },
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            if was_init {
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
            } else {
                vk::PipelineStageFlags2::TOP_OF_PIPE
            },
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        let max_prims = (self.prim_buf.size as usize) / core::mem::size_of::<GpuPrimitive>();
        let prim_count = frame.primitives.len().min(max_prims);
        let active_pipeline = pipeline_override.unwrap_or(self.pipeline);
        self.record_dynamic_render_pass(
            target_view,
            w,
            h,
            effective_scissor,
            frame.clear_color,
            active_pipeline,
            prim_count,
        );

        if readback_to_cpu {
            record_image_barrier(
                &self.device,
                self.cmd_buf,
                target_img,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::TRANSFER,
            );

            let rb = self.readback_buf.as_ref().unwrap();
            let copy_region = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })];

            unsafe {
                self.device.cmd_copy_image_to_buffer(
                    self.cmd_buf,
                    target_img,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    rb.buffer,
                    &copy_region,
                );
            }

            // Transition image back AND expose `TRANSFER_WRITE` on `readback_buf` to `HOST_READ`
            // (required by Vulkan 1.3 Synchronization2 before `invalidate_mapped_memory_ranges` + host read!).
            let mem_barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            let img_barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(target_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            let dep = vk::DependencyInfo::default()
                .memory_barriers(&mem_barrier)
                .image_memory_barriers(&img_barrier);
            unsafe { self.device.cmd_pipeline_barrier2(self.cmd_buf, &dep) };
        }

        unsafe {
            self.device.end_command_buffer(self.cmd_buf)?;
            self.device.reset_fences(&[self.fence])?;
            let submit = [vk::SubmitInfo::default()
                .command_buffers(core::slice::from_ref(&self.cmd_buf))];
            self.device.queue_submit(self.queue, &submit, self.fence)?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        Ok(())
    }

    fn render_internal(
        &mut self,
        frame: &RenderFrame<'_>,
        atlas: &mut GlyphAtlas,
        readback_to_cpu: bool,
    ) -> Result<(), VulkanError> {
        self.render_internal_with_pipeline(frame, atlas, readback_to_cpu, None)
    }

    pub(crate) fn record_atlas_uploads(&mut self, atlas: &mut GlyphAtlas) -> Result<(), VulkanError> {
        if atlas.width == 0 || atlas.height == 0 {
            for tex in [&mut self.atlas_r8, &mut self.atlas_rgba8] {
                if !tex.initialized {
                    record_image_barrier(
                        &self.device,
                        self.cmd_buf,
                        tex.image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags2::NONE,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    );
                    tex.initialized = true;
                }
            }
            return Ok(());
        }

        let force_full = !self.atlas_r8.initialized
            || !self.atlas_rgba8.initialized
            || self.atlas_gen_uploaded != atlas.generation;
        if !force_full && atlas.dirty_rects.is_empty() {
            return Ok(());
        }

        let needed_bytes = Self::compute_atlas_staging_bytes(atlas, force_full);
        self.ensure_staging_capacity(needed_bytes)?;
        self.atlas_gen_uploaded = atlas.generation;

        for tex in [&mut self.atlas_r8, &mut self.atlas_rgba8] {
            let was_init = tex.initialized;
            record_image_barrier(
                &self.device,
                self.cmd_buf,
                tex.image,
                if was_init {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                if was_init {
                    vk::AccessFlags2::SHADER_SAMPLED_READ
                } else {
                    vk::AccessFlags2::NONE
                },
                vk::AccessFlags2::TRANSFER_WRITE,
                if was_init {
                    vk::PipelineStageFlags2::FRAGMENT_SHADER
                } else {
                    vk::PipelineStageFlags2::TOP_OF_PIPE
                },
                vk::PipelineStageFlags2::TRANSFER,
            );
            tex.initialized = true;
        }

        let aw = (atlas.width as usize).min(self.atlas_r8.width as usize);
        let ah = (atlas.height as usize).min(self.atlas_r8.height as usize);

        if force_full {
            atlas.dirty_rects.clear();
            let r8_expected = aw * ah;
            let rgba_expected = aw * ah * 4;
            if atlas.r8_data.len() >= r8_expected && atlas.rgba8_data.len() >= rgba_expected {
                let rgba_off = (r8_expected + 3) & !3;
                let written_bytes = rgba_off + rgba_expected;
                self.ensure_staging_capacity(written_bytes)?;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        atlas.r8_data.as_ptr(),
                        self.staging_buf.mapped,
                        r8_expected,
                    );
                    core::ptr::copy_nonoverlapping(
                        atlas.rgba8_data.as_ptr(),
                        self.staging_buf.mapped.add(rgba_off),
                        rgba_expected,
                    );
                    if let Some(range) = self.staging_buf.aligned_range(
                        0,
                        written_bytes as u64,
                        self.non_coherent_atom_size,
                    ) {
                        self.device
                            .flush_mapped_memory_ranges(core::slice::from_ref(&range))?;
                    }
                    self.device.cmd_copy_buffer_to_image(
                        self.cmd_buf,
                        self.staging_buf.buffer,
                        self.atlas_r8.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::BufferImageCopy::default()
                            .buffer_offset(0)
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image_extent(vk::Extent3D {
                                width: aw as u32,
                                height: ah as u32,
                                depth: 1,
                            })],
                    );
                    self.device.cmd_copy_buffer_to_image(
                        self.cmd_buf,
                        self.staging_buf.buffer,
                        self.atlas_rgba8.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::BufferImageCopy::default()
                            .buffer_offset(rgba_off as u64)
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image_extent(vk::Extent3D {
                                width: aw as u32,
                                height: ah as u32,
                                depth: 1,
                            })],
                    );
                }
            }
        } else {
            self.r8_copy_scratch.clear();
            self.rgba_copy_scratch.clear();
            let mut staging_cursor = 0usize;

            for rect in atlas.dirty_rects.drain(..) {
                let x = (rect.x as usize).min(aw);
                let y = (rect.y as usize).min(ah);
                let rw = (rect.w as usize).min(aw - x);
                let rh = (rect.h as usize).min(ah - y);
                if rw == 0 || rh == 0 {
                    continue;
                }
                let bpp = match rect.plane {
                    AtlasPlane::R8 => 1usize,
                    AtlasPlane::Rgba8 => 4usize,
                };
                let aligned_off = (staging_cursor + 3) & !3;
                let byte_size = rw * rh * bpp;
                if aligned_off + byte_size > self.staging_buf.size as usize {
                    self.ensure_staging_capacity(aligned_off + byte_size)?;
                }
                let atlas_stride = (atlas.width as usize) * bpp;
                let row_bytes = rw * bpp;
                let src_buf = match rect.plane {
                    AtlasPlane::R8 => &atlas.r8_data,
                    AtlasPlane::Rgba8 => &atlas.rgba8_data,
                };
                let last_src_end = (y + rh - 1) * atlas_stride + x * bpp + row_bytes;
                if last_src_end > src_buf.len() {
                    continue;
                }
                for r in 0..rh {
                    let src_off = (y + r) * atlas_stride + x * bpp;
                    let dst_off = aligned_off + r * row_bytes;
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            src_buf.as_ptr().add(src_off),
                            self.staging_buf.mapped.add(dst_off),
                            row_bytes,
                        );
                    }
                }
                let copy_cmd = vk::BufferImageCopy::default()
                    .buffer_offset(aligned_off as u64)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D {
                        x: x as i32,
                        y: y as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: rw as u32,
                        height: rh as u32,
                        depth: 1,
                    });
                match rect.plane {
                    AtlasPlane::R8 => self.r8_copy_scratch.push(copy_cmd),
                    AtlasPlane::Rgba8 => self.rgba_copy_scratch.push(copy_cmd),
                }
                staging_cursor = aligned_off + byte_size;
            }
            let written_bytes = staging_cursor;

            if written_bytes > 0
                && let Some(range) = self.staging_buf.aligned_range(
                    0,
                    written_bytes as u64,
                    self.non_coherent_atom_size,
                )
            {
                unsafe {
                    self.device
                        .flush_mapped_memory_ranges(core::slice::from_ref(&range))?;
                }
            }
            if !self.r8_copy_scratch.is_empty() {
                unsafe {
                    self.device.cmd_copy_buffer_to_image(
                        self.cmd_buf,
                        self.staging_buf.buffer,
                        self.atlas_r8.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &self.r8_copy_scratch,
                    );
                }
            }
            if !self.rgba_copy_scratch.is_empty() {
                unsafe {
                    self.device.cmd_copy_buffer_to_image(
                        self.cmd_buf,
                        self.staging_buf.buffer,
                        self.atlas_rgba8.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &self.rgba_copy_scratch,
                    );
                }
            }
        }

        for tex in [&self.atlas_r8, &self.atlas_rgba8] {
            record_image_barrier(
                &self.device,
                self.cmd_buf,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            );
        }
        Ok(())
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(t) = self.offscreen_target.take() {
                Self::destroy_texture(&self.device, t);
            }
            if let Some(rb) = self.readback_buf.take() {
                Self::destroy_mapped_buffer(&self.device, rb);
            }
            Self::destroy_texture(&self.device, core::mem::replace(&mut self.atlas_r8, zero_tex()));
            Self::destroy_texture(
                &self.device,
                core::mem::replace(&mut self.atlas_rgba8, zero_tex()),
            );
            Self::destroy_mapped_buffer(
                &self.device,
                core::mem::replace(&mut self.uniform_buf, zero_buf()),
            );
            Self::destroy_mapped_buffer(
                &self.device,
                core::mem::replace(&mut self.prim_buf, zero_buf()),
            );
            Self::destroy_mapped_buffer(
                &self.device,
                core::mem::replace(&mut self.clip_buf, zero_buf()),
            );
            Self::destroy_mapped_buffer(
                &self.device,
                core::mem::replace(&mut self.staging_buf, zero_buf()),
            );
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

const fn zero_buf() -> MappedBuffer {
    MappedBuffer {
        buffer: vk::Buffer::null(),
        memory: vk::DeviceMemory::null(),
        mapped: core::ptr::null_mut(),
        size: 0,
        alloc_size: 0,
        is_coherent: true,
    }
}

const fn zero_tex() -> GpuTexture {
    GpuTexture {
        image: vk::Image::null(),
        memory: vk::DeviceMemory::null(),
        view: vk::ImageView::null(),
        width: 0,
        height: 0,
        initialized: false,
    }
}

fn find_mem_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}

#[allow(clippy::too_many_arguments)]
fn record_image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
    src_stage: vk::PipelineStageFlags2,
    dst_stage: vk::PipelineStageFlags2,
) {
    let barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&barrier);
    unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };
}
