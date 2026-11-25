// Skima Unified Analytical SDF + OKLab + Rounded-Clip + ClearType Dual-Source Atlas Uber-Shader
// Compatible with both native Vulkan 1.3 (compiled to SPIR-V 1.3 at build time, including hardware
// Dual-Source Blending `UBER_FRAG_DUAL_SRC_SPV`) and baseline WebGPU (`wgpu`).

struct ViewportUniform {
    // [2.0 / width, 2.0 / height, scale_factor, 0.0]
    transform: vec4<f32>,
    // Active damage scissor [x0, y0, x1, y1] in physical pixels for vertex-stage culling
    scissor: vec4<f32>,
}

// 64 bytes exact (1 cache line, std430 aligned)
struct GpuPrimitive {
    // [x, y, width, height] in physical pixels
    bounds: vec4<f32>,
    // [top_left, top_right, bottom_right, bottom_left] corner radii
    radii: vec4<f32>,
    // Mode-dependent payload:
    // - QUAD / GRADIENT / SHADOW: [border_width, shadow_blur, grad_cos_or_off_x, grad_sin_or_off_y]
    // - GLYPH / IMAGE: [u0, v0, u1, v1] normalized atlas coordinates
    uv_or_extra: vec4<f32>,
    // [color0_rgba8, color1_rgba8, (kind << 24) | clip_idx, opacity_and_flags]
    packed: vec4<u32>,
}

// 48 bytes exact (3 x 16B std430 aligned) supporting nested rounded-rect clip chains
struct GpuClip {
    bounds: vec4<f32>,
    radii: vec4<f32>,
    // [parent_clip_idx (0xFFFFFF = none), flags, 0, 0]
    parent_info: vec4<u32>,
}

@group(0) @binding(0) var<uniform> u_view: ViewportUniform;
@group(0) @binding(1) var<storage, read> s_prims: array<GpuPrimitive>;
@group(0) @binding(2) var<storage, read> s_clips: array<GpuClip>;
@group(0) @binding(3) var t_atlas_r8: texture_2d<f32>;
@group(0) @binding(4) var t_atlas_rgba8: texture_2d<f32>;
@group(0) @binding(5) var s_linear: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) screen_px: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) prim_idx: u32,
}

struct DualBlendPair {
    color: vec4<f32>,
    mask: vec4<f32>,
}

const KIND_QUAD: u32 = 0u;
const KIND_SHADOW: u32 = 1u;
const KIND_LINEAR_GRAD: u32 = 2u;
const KIND_GLYPH_GRAY: u32 = 3u;
const KIND_GLYPH_LCD: u32 = 4u;
const KIND_IMAGE_RGBA: u32 = 5u;

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    let prim = s_prims[iid];
    let kind = prim.packed.z >> 24u;
    let op_bits = prim.packed.w & 0xFFFFu;

    // Cull invisible layout-container quads & zero-opacity primitives in vertex stage (zero fragment overdraw!)
    let c0_a = prim.packed.x & 0xFFu;
    let c1_a = prim.packed.y & 0xFFu;
    let invisible_quad = kind == KIND_QUAD && c0_a == 0u && (prim.uv_or_extra.x <= 0.0 || c1_a == 0u);
    let invisible_grad = kind == KIND_LINEAR_GRAD && c0_a == 0u && c1_a == 0u;
    let invisible_tinted = (kind == KIND_SHADOW || kind == KIND_GLYPH_GRAY || kind == KIND_GLYPH_LCD || kind == KIND_IMAGE_RGBA)
        && c0_a == 0u;
    if op_bits == 0u || prim.bounds.z <= 0.0 || prim.bounds.w <= 0.0 || invisible_quad || invisible_grad || invisible_tinted {
        var culled: VertexOut;
        culled.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        culled.screen_px = vec2<f32>(0.0);
        culled.uv = vec2<f32>(0.0);
        culled.prim_idx = iid;
        return culled;
    }

    var margin = 1.0;
    var shift = vec2<f32>(0.0);
    if kind == KIND_SHADOW {
        let sigma = max(prim.uv_or_extra.y, 0.5);
        margin = sigma * 3.5 + 2.0;
        shift = prim.uv_or_extra.zw;
    } else if kind == KIND_GLYPH_GRAY || kind == KIND_GLYPH_LCD || kind == KIND_IMAGE_RGBA {
        margin = 0.0;
    }

    let x0 = prim.bounds.x + shift.x - margin;
    let y0 = prim.bounds.y + shift.y - margin;
    let x1 = prim.bounds.x + prim.bounds.z + shift.x + margin;
    let y1 = prim.bounds.y + prim.bounds.w + shift.y + margin;

    // Cull primitives outside the active damage scissor or physical viewport before rasterization
    let sc_x0 = max(u_view.scissor.x, 0.0);
    let sc_y0 = max(u_view.scissor.y, 0.0);
    if x1 <= sc_x0 || x0 >= u_view.scissor.z || y1 <= sc_y0 || y0 >= u_view.scissor.w
        || x1 <= 0.0 || y1 <= 0.0 || x0 * u_view.transform.x >= 2.0 || y0 * u_view.transform.y >= 2.0 {
        var culled: VertexOut;
        culled.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        culled.screen_px = vec2<f32>(0.0);
        culled.uv = vec2<f32>(0.0);
        culled.prim_idx = iid;
        return culled;
    }

    // Cull primitives whose footprint is outside their ancestor clip chain's outer bounds (and tighten procedural quads)
    let clip_id = prim.packed.z & 0xFFFFFFu;
    var rx0 = x0;
    var ry0 = y0;
    var rx1 = x1;
    var ry1 = y1;
    if clip_id != 0xFFFFFFu {
        let n_clips = arrayLength(&s_clips);
        var cid = clip_id;
        var steps = 0u;
        var clip_min = vec2<f32>(-1e30, -1e30);
        var clip_max = vec2<f32>(1e30, 1e30);
        var has_clip = false;
        loop {
            if cid == 0xFFFFFFu || cid >= n_clips || steps >= n_clips {
                break;
            }
            let clip = s_clips[cid];
            if clip.bounds.z <= 0.0 || clip.bounds.w <= 0.0 {
                var culled_zero: VertexOut;
                culled_zero.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
                culled_zero.screen_px = vec2<f32>(0.0);
                culled_zero.uv = vec2<f32>(0.0);
                culled_zero.prim_idx = iid;
                return culled_zero;
            }
            clip_min = max(clip_min, clip.bounds.xy);
            clip_max = min(clip_max, clip.bounds.xy + clip.bounds.zw);
            if clip_max.x <= clip_min.x || clip_max.y <= clip_min.y {
                var culled_disjoint: VertexOut;
                culled_disjoint.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
                culled_disjoint.screen_px = vec2<f32>(0.0);
                culled_disjoint.uv = vec2<f32>(0.0);
                culled_disjoint.prim_idx = iid;
                return culled_disjoint;
            }
            has_clip = true;
            cid = clip.parent_info.x;
            steps = steps + 1u;
        }
        if has_clip {
            if x1 <= clip_min.x || x0 >= clip_max.x || y1 <= clip_min.y || y0 >= clip_max.y {
                var culled_clip: VertexOut;
                culled_clip.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
                culled_clip.screen_px = vec2<f32>(0.0);
                culled_clip.uv = vec2<f32>(0.0);
                culled_clip.prim_idx = iid;
                return culled_clip;
            }
            if kind <= KIND_LINEAR_GRAD {
                rx0 = max(x0, clip_min.x - 1.0);
                ry0 = max(y0, clip_min.y - 1.0);
                rx1 = min(x1, clip_max.x + 1.0);
                ry1 = min(y1, clip_max.y + 1.0);
            }
        }
    }
    if kind <= KIND_LINEAR_GRAD {
        rx0 = max(rx0, sc_x0 - 2.0);
        ry0 = max(ry0, sc_y0 - 2.0);
        rx1 = min(rx1, u_view.scissor.z + 2.0);
        ry1 = min(ry1, u_view.scissor.w + 2.0);
    }

    // 6-vertex triangle-list quad (0,1,2, 2,1,3)
    var corner = vec2<f32>(0.0, 0.0);
    switch vid {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u, 4u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u, 3u: { corner = vec2<f32>(0.0, 1.0); }
        default: { corner = vec2<f32>(1.0, 1.0); }
    }

    let px = mix(vec2<f32>(rx0, ry0), vec2<f32>(rx1, ry1), corner);
    let ndc = vec2<f32>(
        px.x * u_view.transform.x - 1.0,
        px.y * u_view.transform.y - 1.0,
    );

    let uv = mix(prim.uv_or_extra.xy, prim.uv_or_extra.zw, corner);

    var out: VertexOut;
    out.pos = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    out.screen_px = px;
    out.uv = uv;
    out.prim_idx = iid;
    return out;
}

fn unpack_rgba(c: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((c >> 24u) & 0xFFu) / 255.0,
        f32((c >> 16u) & 0xFFu) / 255.0,
        f32((c >> 8u) & 0xFFu) / 255.0,
        f32(c & 0xFFu) / 255.0,
    );
}

fn srgb_to_linear_chan(c: f32) -> f32 {
    if c <= 0.04045 {
        return max(c, 0.0) / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_chan(rgb.r),
        srgb_to_linear_chan(rgb.g),
        srgb_to_linear_chan(rgb.b),
    );
}

fn cbrt_pos(x: f32) -> f32 {
    if x <= 1e-12 {
        return 0.0;
    }
    return pow(x, 1.0 / 3.0);
}

fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = cbrt_pos(0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b);
    let m = cbrt_pos(0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b);
    let s = cbrt_pos(0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b);
    return vec3<f32>(
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    );
}

fn oklab_to_linear(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return clamp(
        vec3<f32>(
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn select_corner_radius(p: vec2<f32>, radii: vec4<f32>) -> f32 {
    if p.x > 0.0 {
        if p.y > 0.0 { return radii.z; } else { return radii.y; }
    } else {
        if p.y > 0.0 { return radii.w; } else { return radii.x; }
    }
}

// Inigo Quilez exact 4-corner rounded rectangle SDF (`radii = [tl, tr, br, bl]`)
fn sd_rounded_box(p: vec2<f32>, half_ext: vec2<f32>, radii: vec4<f32>) -> f32 {
    let cr = clamp(select_corner_radius(p, radii), 0.0, max(min(half_ext.x, half_ext.y), 0.0));
    let q = abs(p) - half_ext + vec2<f32>(cr, cr);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - cr;
}

// Exact analytical pixel-square area coverage from Sterbenz-exact inward edge distances `dl = px - b_min`,
// `dr = b_max - px` (handles sub-pixel thin hairlines with exact energy conservation and zero inner leak,
// 0-size degenerate inner holes, sharp 90° inner corners, and curved SDF arcs without `+ cr - cr` drift).
fn rounded_box_cov_edges(dl: vec2<f32>, dr: vec2<f32>, size: vec2<f32>, raw_cr: f32) -> f32 {
    if size.x <= 0.0 || size.y <= 0.0 {
        return 0.0;
    }
    let edge_dist = min(dl, dr);
    let max_cov = clamp(size, vec2<f32>(0.0), vec2<f32>(1.0));
    let rect_cov = clamp(edge_dist + vec2<f32>(0.5), vec2<f32>(0.0), max_cov);
    let box_area = rect_cov.x * rect_cov.y;
    if box_area <= 0.0 {
        return 0.0;
    }
    let cr = clamp(raw_cr, 0.0, 0.5 * min(size.x, size.y));
    if cr <= 0.0 {
        return box_area;
    }
    let q = vec2<f32>(cr) - edge_dist;
    if q.x > 0.0 && q.y > 0.0 {
        let d = length(q) - cr;
        let sdf_cov = clamp(0.5 - d, 0.0, 1.0);
        return mix(box_area, min(box_area, sdf_cov), clamp(cr, 0.0, 1.0));
    }
    return box_area;
}

// Exact 1:1 texel fetch (`textureLoad`) when `[x, y, w, h]` and `[u0, v0, u1, v1] * atlas_dims` are
// integer-aligned (`1024`, `2048`, `4096`, or arbitrary atlas dimensions — zero bilinear neighbor bleed!),
// with pixel-center-exact bilinear fallback (`textureSampleLevel`) for scaled/subpixel quads.
fn sample_atlas_r8_prim(px: vec2<f32>, bounds: vec4<f32>, uv_rect: vec4<f32>) -> f32 {
    let dims_u = textureDimensions(t_atlas_r8);
    let dims = vec2<f32>(dims_u);
    let tex_min = uv_rect.xy * dims;
    let tex_size = (uv_rect.zw - uv_rect.xy) * dims;
    let b_xy_r = round(bounds.xy);
    let b_zw_r = round(bounds.zw);
    let t_min_r = round(tex_min);
    let snap_x = bounds.z >= 1.0 && t_min_r.x >= 0.0 && (t_min_r.x + b_zw_r.x) <= dims.x
        && abs(bounds.x - b_xy_r.x) < 1e-3 && abs(bounds.z - b_zw_r.x) < 1e-3
        && abs(tex_min.x - t_min_r.x) < 1e-3 && abs(tex_size.x - b_zw_r.x) < 1e-3;
    let snap_y = bounds.w >= 1.0 && t_min_r.y >= 0.0 && (t_min_r.y + b_zw_r.y) <= dims.y
        && abs(bounds.y - b_xy_r.y) < 1e-3 && abs(bounds.w - b_zw_r.y) < 1e-3
        && abs(tex_min.y - t_min_r.y) < 1e-3 && abs(tex_size.y - b_zw_r.y) < 1e-3;
    if snap_x && snap_y {
        let pix_i = vec2<i32>(floor(px));
        let rel_i = clamp(
            pix_i - vec2<i32>(b_xy_r),
            vec2<i32>(0),
            max(vec2<i32>(b_zw_r) - vec2<i32>(1), vec2<i32>(0)),
        );
        let max_tc = max(vec2<i32>(dims_u) - vec2<i32>(1), vec2<i32>(0));
        let tc = clamp(vec2<i32>(t_min_r) + rel_i, vec2<i32>(0), max_tc);
        return textureLoad(t_atlas_r8, tc, 0).r;
    }
    let local_t = clamp((px - bounds.xy) / max(bounds.zw, vec2<f32>(1e-5)), vec2<f32>(0.0), vec2<f32>(1.0));
    var uv = uv_rect.xy + (uv_rect.zw - uv_rect.xy) * local_t;
    if snap_x {
        let rel_x = clamp(floor(px.x) - b_xy_r.x, 0.0, max(b_zw_r.x - 1.0, 0.0));
        uv.x = (clamp(t_min_r.x + rel_x, 0.0, max(dims.x - 1.0, 0.0)) + 0.5) / dims.x;
    }
    if snap_y {
        let rel_y = clamp(floor(px.y) - b_xy_r.y, 0.0, max(b_zw_r.y - 1.0, 0.0));
        uv.y = (clamp(t_min_r.y + rel_y, 0.0, max(dims.y - 1.0, 0.0)) + 0.5) / dims.y;
    }
    return textureSampleLevel(t_atlas_r8, s_linear, uv, 0.0).r;
}

fn sample_atlas_rgba8_prim(px: vec2<f32>, bounds: vec4<f32>, uv_rect: vec4<f32>) -> vec4<f32> {
    let dims_u = textureDimensions(t_atlas_rgba8);
    let dims = vec2<f32>(dims_u);
    let tex_min = uv_rect.xy * dims;
    let tex_size = (uv_rect.zw - uv_rect.xy) * dims;
    let b_xy_r = round(bounds.xy);
    let b_zw_r = round(bounds.zw);
    let t_min_r = round(tex_min);
    let snap_x = bounds.z >= 1.0 && t_min_r.x >= 0.0 && (t_min_r.x + b_zw_r.x) <= dims.x
        && abs(bounds.x - b_xy_r.x) < 1e-3 && abs(bounds.z - b_zw_r.x) < 1e-3
        && abs(tex_min.x - t_min_r.x) < 1e-3 && abs(tex_size.x - b_zw_r.x) < 1e-3;
    let snap_y = bounds.w >= 1.0 && t_min_r.y >= 0.0 && (t_min_r.y + b_zw_r.y) <= dims.y
        && abs(bounds.y - b_xy_r.y) < 1e-3 && abs(bounds.w - b_zw_r.y) < 1e-3
        && abs(tex_min.y - t_min_r.y) < 1e-3 && abs(tex_size.y - b_zw_r.y) < 1e-3;
    if snap_x && snap_y {
        let pix_i = vec2<i32>(floor(px));
        let rel_i = clamp(
            pix_i - vec2<i32>(b_xy_r),
            vec2<i32>(0),
            max(vec2<i32>(b_zw_r) - vec2<i32>(1), vec2<i32>(0)),
        );
        let max_tc = max(vec2<i32>(dims_u) - vec2<i32>(1), vec2<i32>(0));
        let tc = clamp(vec2<i32>(t_min_r) + rel_i, vec2<i32>(0), max_tc);
        return textureLoad(t_atlas_rgba8, tc, 0);
    }
    let local_t = clamp((px - bounds.xy) / max(bounds.zw, vec2<f32>(1e-5)), vec2<f32>(0.0), vec2<f32>(1.0));
    var uv = uv_rect.xy + (uv_rect.zw - uv_rect.xy) * local_t;
    if snap_x {
        let rel_x = clamp(floor(px.x) - b_xy_r.x, 0.0, max(b_zw_r.x - 1.0, 0.0));
        uv.x = (clamp(t_min_r.x + rel_x, 0.0, max(dims.x - 1.0, 0.0)) + 0.5) / dims.x;
    }
    if snap_y {
        let rel_y = clamp(floor(px.y) - b_xy_r.y, 0.0, max(b_zw_r.y - 1.0, 0.0));
        uv.y = (clamp(t_min_r.y + rel_y, 0.0, max(dims.y - 1.0, 0.0)) + 0.5) / dims.y;
    }
    return textureSampleLevel(t_atlas_rgba8, s_linear, uv, 0.0);
}

// Abramowitz & Stegun 7.1.27 fast error function for closed-form Gaussian shadow integral
fn erf_scalar(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let p = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    let p4 = p * p * p * p;
    return s - s / p4;
}

fn erf_approx(x: vec2<f32>) -> vec2<f32> {
    let s = sign(x);
    let a = abs(x);
    let p = vec2<f32>(1.0) + (vec2<f32>(0.278393) + (vec2<f32>(0.230389) + vec2<f32>(0.078108) * (a * a)) * a) * a;
    let p4 = p * p * p * p;
    return s - s / p4;
}

// Arbitrary-depth nested rounded-rect clip intersection with smooth 1px SDF corner antialiasing,
// inner-safe-rect trivial accept (zero `length()` ALU inside scrollable glyph viewports!),
// and exact axis-aligned pixel-interval intersection (idempotent on coincident clips, zero depth cap).
fn eval_clip_chain(px: vec2<f32>, start_clip: u32) -> f32 {
    if start_clip == 0xFFFFFFu {
        return 1.0;
    }
    let n_clips = arrayLength(&s_clips);
    var ix0 = px.x - 0.5;
    var iy0 = px.y - 0.5;
    var ix1 = px.x + 0.5;
    var iy1 = px.y + 0.5;
    var min_arc_cov = 1.0;
    var cid = start_clip;
    var steps = 0u;
    loop {
        if cid == 0xFFFFFFu || cid >= n_clips || steps >= n_clips {
            break;
        }
        let clip = s_clips[cid];
        if clip.bounds.z <= 0.0 || clip.bounds.w <= 0.0 {
            return 0.0;
        }
        let c_min = clip.bounds.xy;
        let c_max = c_min + clip.bounds.zw;
        let max_r = max(max(clip.radii.x, clip.radii.y), max(clip.radii.z, clip.radii.w));
        if px.x >= c_min.x + 1.0 && px.x <= c_max.x - 1.0 && px.y >= c_min.y + 1.0 && px.y <= c_max.y - 1.0 {
            let safe_x = px.x >= c_min.x + max_r + 1.0 && px.x <= c_max.x - max_r - 1.0;
            let safe_y = px.y >= c_min.y + max_r + 1.0 && px.y <= c_max.y - max_r - 1.0;
            if max_r <= 0.0 || safe_x || safe_y {
                if clip.parent_info.x == 0xFFFFFFu && steps == 0u {
                    return 1.0;
                }
                cid = clip.parent_info.x;
                steps = steps + 1u;
                continue;
            }
        }
        ix0 = max(ix0, c_min.x);
        iy0 = max(iy0, c_min.y);
        ix1 = min(ix1, c_max.x);
        iy1 = min(iy1, c_max.y);
        if ix1 <= ix0 || iy1 <= iy0 {
            return 0.0;
        }
        let cur_box_cov = (ix1 - ix0) * (iy1 - iy0);
        let half_sz = clip.bounds.zw * 0.5;
        let rel = px - (clip.bounds.xy + half_sz);
        let cr = clamp(select_corner_radius(rel, clip.radii), 0.0, min(half_sz.x, half_sz.y));
        if cr > 0.0 {
            let edge_dist = min(px - c_min, c_max - px);
            let q = vec2<f32>(cr) - edge_dist;
            if q.x > 0.0 && q.y > 0.0 {
                let d = length(q) - cr;
                let sdf_cov = clamp(0.5 - d, 0.0, 1.0);
                let corner_cov = mix(cur_box_cov, min(cur_box_cov, sdf_cov), clamp(cr, 0.0, 1.0));
                min_arc_cov = min(min_arc_cov, corner_cov);
                if min_arc_cov <= 0.0005 {
                    return 0.0;
                }
            }
        }
        cid = clip.parent_info.x;
        steps = steps + 1u;
    }
    let box_cov = (ix1 - ix0) * (iy1 - iy0);
    return min(box_cov, min_arc_cov);
}

fn shade_primitive(in: VertexOut, dual_src: bool) -> DualBlendPair {
    let prim = s_prims[in.prim_idx];
    let kind = prim.packed.z >> 24u;
    let clip_id = prim.packed.z & 0xFFFFFFu;

    // Snap fragment center to exact `(floor(x) + 0.5, floor(y) + 0.5)` to eliminate NDC `2.0 / viewport`
    // and triangle-barycentric float drift across non-power-of-two fractional-DPI viewports.
    let screen_px = floor(in.screen_px) + vec2<f32>(0.5);

    let clip_alpha = eval_clip_chain(screen_px, clip_id);
    if clip_alpha <= 0.0005 {
        return DualBlendPair(vec4<f32>(0.0), vec4<f32>(0.0));
    }

    let c0_srgb = unpack_rgba(prim.packed.x);
    let c1_srgb = unpack_rgba(prim.packed.y);
    let opacity = f32(prim.packed.w & 0xFFFFu) / 65535.0;

    let half_sz = prim.bounds.zw * 0.5;
    let center = prim.bounds.xy + half_sz;
    let rel_p = screen_px - center;

    var out_lin = vec3<f32>(0.0);
    var out_a = 0.0;

    switch kind {
        case 0u, 2u: { // KIND_QUAD or KIND_LINEAR_GRAD
            let b_min = prim.bounds.xy;
            let size = prim.bounds.zw;
            let b_max = b_min + size;
            let dl = screen_px - b_min;
            let dr = b_max - screen_px;
            let raw_cr = select_corner_radius(rel_p, prim.radii);
            let outer_cov = rounded_box_cov_edges(dl, dr, size, raw_cr);
            if outer_cov <= 0.0005 {
                return DualBlendPair(vec4<f32>(0.0), vec4<f32>(0.0));
            }

            var fill_rgb = srgb_to_linear(c0_srgb.rgb);
            var fill_a = c0_srgb.a;

            if kind == KIND_LINEAR_GRAD {
                let dir = prim.uv_or_extra.zw;
                let span = max(abs(dir.x) * prim.bounds.z + abs(dir.y) * prim.bounds.w, 1e-4);
                let t = clamp(dot(rel_p, dir) / span + 0.5, 0.0, 1.0);
                let lab0 = linear_to_oklab(fill_rgb);
                let lab1 = linear_to_oklab(srgb_to_linear(c1_srgb.rgb));
                // CSS Color 4 Premultiplied OKLab interpolation (zero dark/muddy fringe toward transparent endpoints!)
                let p0 = vec4<f32>(lab0 * c0_srgb.a, c0_srgb.a);
                let p1 = vec4<f32>(lab1 * c1_srgb.a, c1_srgb.a);
                let pm = mix(p0, p1, t);
                fill_a = pm.w;
                if fill_a > 1e-6 {
                    fill_rgb = oklab_to_linear(pm.xyz / fill_a);
                } else {
                    fill_rgb = vec3<f32>(0.0);
                    fill_a = 0.0;
                }
            }

            let fill_premul = vec4<f32>(fill_rgb * fill_a, fill_a);
            let border_w = prim.uv_or_extra.x;
            var shape_premul: vec4<f32>;
            if border_w > 0.0 {
                let max_r = max(min(half_sz.x, half_sz.y), 0.0);
                let eff_outer_cr = clamp(raw_cr, 0.0, max_r);
                let inner_cr = max(eff_outer_cr - border_w, 0.0);
                let bw2 = vec2<f32>(border_w);
                let inner_cov = min(
                    rounded_box_cov_edges(dl - bw2, dr - bw2, size - bw2 * 2.0, inner_cr),
                    outer_cov,
                );
                let border_cov = max(outer_cov - inner_cov, 0.0);
                let border_rgb = srgb_to_linear(c1_srgb.rgb);
                let border_a = c1_srgb.a;
                let border_premul = vec4<f32>(border_rgb * border_a, border_a);
                let ring_premul = border_premul + fill_premul * (1.0 - border_a);
                shape_premul = fill_premul * inner_cov + ring_premul * border_cov;
            } else {
                shape_premul = fill_premul * outer_cov;
            }
            let out_premul = shape_premul * (opacity * clip_alpha);
            return DualBlendPair(
                out_premul,
                vec4<f32>(out_premul.w),
            );
        }
        case 1u: { // KIND_SHADOW (Analytical closed-form 2D Gaussian rounded-box shadow)
            let sigma = max(prim.uv_or_extra.y, 0.5);
            let offset = prim.uv_or_extra.zw;
            let sp = rel_p - offset;
            let inv_s = 1.0 / (sigma * 1.4142135624);
            let ix = erf_approx((half_sz - sp) * inv_s) + erf_approx((half_sz + sp) * inv_s);
            let rect_cov = clamp(0.25 * ix.x * ix.y, 0.0, 1.0);
            let cr = clamp(select_corner_radius(sp, prim.radii), 0.0, max(min(half_sz.x, half_sz.y), 0.0));
            var shadow_cov = rect_cov;
            if cr > 0.0 {
                let d = sd_rounded_box(sp, half_sz, prim.radii);
                let sdf_cov = clamp(0.5 * (1.0 - erf_scalar(d * inv_s)), 0.0, 1.0);
                shadow_cov = min(rect_cov, sdf_cov);
            }
            out_lin = srgb_to_linear(c0_srgb.rgb);
            out_a = c0_srgb.a * shadow_cov;
        }
        case 3u: { // KIND_GLYPH_GRAY (Photometrically balanced linear stem darkening on light & dark themes)
            let raw_cov = clamp(sample_atlas_r8_prim(screen_px, prim.bounds, prim.uv_or_extra), 0.0, 1.0);
            let fg_lin = srgb_to_linear(c0_srgb.rgb);
            let lum = dot(fg_lin, vec3<f32>(0.2126, 0.7152, 0.0722));
            let k = 0.46 - 0.58 * lum;
            let cov = clamp(raw_cov + k * raw_cov * (1.0 - raw_cov), 0.0, 1.0);
            out_lin = fg_lin;
            out_a = c0_srgb.a * cov;
        }
        case 4u: { // KIND_GLYPH_LCD (True Per-Channel RGB ClearType Subpixel Mask + clean single-target fallback)
            let raw_mask = clamp(sample_atlas_rgba8_prim(screen_px, prim.bounds, prim.uv_or_extra).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
            let fg_lin = srgb_to_linear(c0_srgb.rgb);
            let lum = dot(fg_lin, vec3<f32>(0.2126, 0.7152, 0.0722));
            let k = 0.44 - 0.56 * lum;
            let adj_mask = clamp(raw_mask + k * raw_mask * (vec3<f32>(1.0) - raw_mask), vec3<f32>(0.0), vec3<f32>(1.0));
            let scale = c0_srgb.a * opacity * clip_alpha;
            let eff_mask = adj_mask * scale;
            let avg_a = dot(eff_mask, vec3<f32>(0.299, 0.587, 0.114));
            if dual_src {
                return DualBlendPair(
                    vec4<f32>(fg_lin * eff_mask, avg_a),
                    vec4<f32>(eff_mask, avg_a),
                );
            }
            return DualBlendPair(
                vec4<f32>(fg_lin * avg_a, avg_a),
                vec4<f32>(avg_a),
            );
        }
        default: { // KIND_IMAGE_RGBA (Color emoji / raster image)
            let texel = sample_atlas_rgba8_prim(screen_px, prim.bounds, prim.uv_or_extra);
            out_lin = srgb_to_linear(texel.rgb) * srgb_to_linear(c0_srgb.rgb);
            out_a = texel.a * c0_srgb.a;
        }
    }

    let final_a = out_a * opacity * clip_alpha;
    return DualBlendPair(
        vec4<f32>(out_lin * final_a, final_a),
        vec4<f32>(final_a, final_a, final_a, final_a),
    );
}

// Baseline single-target premultiplied fragment entry (portable WebGPU / non-dualSrcBlend fallback devices)
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let p = shade_primitive(in, false);
    if p.mask.w <= 0.0005 {
        discard;
    }
    return p.color;
}
