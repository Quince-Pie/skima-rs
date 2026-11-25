use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use std::env;
use std::fs;
use std::path::PathBuf;

fn compile_entry(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    entry_name: &str,
) -> Vec<u32> {
    let mut flags = spv::WriterFlags::empty();
    flags.set(spv::WriterFlags::ADJUST_COORDINATE_SPACE, false);

    let options = spv::Options {
        lang_version: (1, 3),
        flags,
        ..spv::Options::default()
    };

    let pipeline_options = spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_name.to_owned(),
    };

    spv::write_vec(module, info, &options, Some(&pipeline_options))
        .unwrap_or_else(|e| panic!("SPIR-V codegen failed for {entry_name}: {e}"))
}

fn format_u32_slice(name: &str, words: &[u32]) -> String {
    let mut out = format!("pub const {name}: &[u32] = &[\n    ");
    for (i, &w) in words.iter().enumerate() {
        out.push_str(&format!("0x{w:08x}, "));
        if (i + 1) % 8 == 0 {
            out.push_str("\n    ");
        }
    }
    out.push_str("\n];\n");
    out
}

fn main() {
    println!("cargo::rerun-if-changed=shaders/uber.wgsl");

    let wgsl_src = fs::read_to_string("shaders/uber.wgsl").expect("read shaders/uber.wgsl");
    let module = wgsl::parse_str(&wgsl_src).expect("valid WGSL in shaders/uber.wgsl");

    // Validate baseline WGSL strictly with empty capabilities so `UBER_SHADER_WGSL` remains
    // 100% portable to baseline WebGPU (`wgpu`) without unguarded native-only extensions.
    let mut base_validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = base_validator
        .validate(&module)
        .expect("Baseline WGSL module passed strict Naga validation");

    let vert_spv = compile_entry(&module, &info, naga::ShaderStage::Vertex, "vs_main");
    let frag_spv = compile_entry(&module, &info, naga::ShaderStage::Fragment, "fs_main");

    // Also compile hardware Dual-Source Blending fragment shader (`VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR`)
    // for exact single-pass RGB ClearType subpixel text over arbitrary dynamic backgrounds!
    let dual_src_wgsl = format!(
        "enable dual_source_blending;\n{wgsl_src}\n\
struct DualSrcFragOut {{\n\
    @location(0) @blend_src(0) color: vec4<f32>,\n\
    @location(0) @blend_src(1) blend_mask: vec4<f32>,\n\
}}\n\
@fragment\n\
fn fs_main_dual_src(in: VertexOut) -> DualSrcFragOut {{\n\
    let p = shade_primitive(in, true);\n\
    if max(max(p.mask.x, p.mask.y), max(p.mask.z, p.mask.w)) <= 0.0005 {{\n\
        discard;\n\
    }}\n\
    return DualSrcFragOut(p.color, p.mask);\n\
}}\n"
    );
    let dual_mod = wgsl::parse_str(&dual_src_wgsl).expect("valid dual-source WGSL");
    let mut dual_validator = Validator::new(
        ValidationFlags::all(),
        Capabilities::DUAL_SOURCE_BLENDING,
    );
    let dual_info = dual_validator
        .validate(&dual_mod)
        .expect("dual-source WGSL passed Naga validation");
    let frag_dual_src_spv = compile_entry(
        &dual_mod,
        &dual_info,
        naga::ShaderStage::Fragment,
        "fs_main_dual_src",
    );

    let mut generated = String::new();
    generated.push_str("// Auto-generated precompiled SPIR-V 1.3 modules (zero runtime shader compiler overhead).\n");
    generated.push_str(&format_u32_slice("UBER_VERT_SPV", &vert_spv));
    generated.push_str(&format_u32_slice("UBER_FRAG_SPV", &frag_spv));
    generated.push_str(&format_u32_slice("UBER_FRAG_DUAL_SRC_SPV", &frag_dual_src_spv));

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(out_dir.join("uber_spv.rs"), generated).expect("write uber_spv.rs");
}
