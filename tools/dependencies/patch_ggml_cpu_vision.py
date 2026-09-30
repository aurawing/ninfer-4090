"""Checked, idempotent private adapter for llama.cpp b81c99b479d4c24e5eeca10de99032ebd343ef8f.

Run only on NInfer's private dependency copy. No new numerical kernels are introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path

PINNED_FILES = {
    "tools/mtmd/clip.cpp": "d20869d359fb135e7726e1d0ea1834d581d3222efba664a8fc5804cbdac29a48",
    "tools/mtmd/clip.h": "07b67b04559cd203e389efa9a0399e61abf2a134d8ba14b194b12adaf3aba21a",
    "tools/mtmd/clip-graph.h": "0c04795e22939263ff613226428096fbc0f0739cef34a00c3aa675856b4ba4aa",
    "tools/mtmd/models/qwen2vl.cpp": "06661c5dd008f1603678eb9def84cf8b566852728c2f8fe4fcd9b003dd1a23af",
    "tools/mtmd/models/qwen3vl.cpp": "5a248956cc965ec1229538585f66a0010fa03a5d5dd164bd495c522ecede0c7b",
    "ggml/include/ggml-cpu.h": "316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e",
    "ggml/src/ggml-cpu/ggml-cpu.cpp": "b620923a8966395ba9c5f2ee4acb35b8d42ea77a58943d7eb733a917c4c1cafb",
    "ggml/src/ggml-cpu/vec.h": "926330bae1c5d003bd654035426e31381fafcdca23ffcc23201d219dbb97cbeb",
}
LEGACY_CPU_PATCH_HASH = "25a39c5535bce11568e271b9b69baf772604cfcd237b2c663c86c83d9110877a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(path, old, new):
    text = path.read_text(encoding="utf-8")
    if new in text:
        if text.count(new) != 1:
            raise RuntimeError(f"ambiguous patched anchor in {path}")
        return
    if text.count(old) != 1:
        raise RuntimeError(f"pinned dependency anchor changed in {path}: {old[:100]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def patch(root):
    stamp = root / ".ninfer-cpu-vision-patch.json"
    script_hash = sha(Path(__file__))
    if stamp.exists():
        previous = json.loads(stamp.read_text(encoding="utf-8"))
        for name, expected in previous["files"].items():
            if sha(root / name) != expected:
                raise RuntimeError(f"patched dependency was modified: {name}")
        if previous["script"] == script_hash:
            return
        if previous["script"] != LEGACY_CPU_PATCH_HASH:
            raise RuntimeError("unknown NInfer dependency patch; recreate the private dependency copy")
    else:
        for name, expected in PINNED_FILES.items():
            if sha(root / name) != expected:
                raise RuntimeError(f"dependency is not the exact pinned original: {name}")
    clip = root / "tools/mtmd/clip.cpp"
    header = root / "tools/mtmd/clip.h"
    graph = root / "tools/mtmd/clip-graph.h"
    qwen2 = root / "tools/mtmd/models/qwen2vl.cpp"
    qwen3 = root / "tools/mtmd/models/qwen3vl.cpp"
    # Append private state so existing upstream positional aggregate initializers keep their ABI.
    replace(header, "    void * progress_callback_user_data;\n};",
        "    void * progress_callback_user_data;\n    bool ninfer_cpu_math = false; // private NInfer graph policy; upstream defaults unchanged\n};")
    replace(clip, '#include "gguf.h"', '#include "gguf.h"\n#include "ggml-cpu.h"')
    replace(clip, "    bool no_alloc = false;", "    bool no_alloc = false;\n    bool ninfer_cpu_math = false;")
    replace(clip, "        no_alloc = ctx_params.no_alloc;", "        no_alloc = ctx_params.no_alloc;\n        ninfer_cpu_math = ctx_params.ninfer_cpu_math;")
    replace(graph, "    const clip_flash_attn_type flash_attn_type;", "    const clip_flash_attn_type flash_attn_type;\n    const bool ninfer_cpu_math;")
    replace(clip, "        flash_attn_type(ctx->flash_attn_type) {", "        flash_attn_type(ctx->flash_attn_type),\n        ninfer_cpu_math(ctx->ninfer_cpu_math) {")
    replace(clip, "        flash_attn_type(parent.flash_attn_type) {", "        flash_attn_type(parent.flash_attn_type),\n        ninfer_cpu_math(parent.ninfer_cpu_math) {")
    replace(clip, "    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);", "    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);\n    if (ninfer_cpu_math) {\n        return clip_ninfer_cpu_position(ctx0, pos_embd, n_embd, width, height, n_per_side);\n    }")
    replace(clip, "        k = ggml_cast(ctx0, k, GGML_TYPE_F16);\n        v = ggml_cast(ctx0, v, GGML_TYPE_F16);", "        // BF16's exponent range must survive CPU attention.\n        k = ggml_cast(ctx0, k, ninfer_cpu_math ? GGML_TYPE_F32 : GGML_TYPE_F16);\n        v = ggml_cast(ctx0, v, ninfer_cpu_math ? GGML_TYPE_F32 : GGML_TYPE_F16);")
    replace(clip, "        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);", "        cur = ninfer_cpu_math\n            ? clip_ninfer_cpu_attention(ctx0, q, k, v, kq_scale)\n            : ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);")
    # Attach precision/sinks to the Flash Attention op before wrapping its output.
    replace(clip, "        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);",
        "        if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }\n        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);")
    # Reuse GGML's existing FP32 tanh formula instead of its default FP16 lookup.
    replace(root / "ggml/src/ggml-cpu/vec.h", "#define GGML_GELU_FP16\n",
        "// NInfer CPU vision uses the existing FP32 GELU path.\n")
    replace(clip, "    return cur;\n}\n\nggml_tensor * clip_graph::build_ffn(", "    return ninfer_cpu_math ? clip_ninfer_cpu_bf16(ctx0, cur) : cur;\n}\n\nggml_tensor * clip_graph::build_ffn(")
    replace(clip, "    ggml_tensor * tmp = up ? build_mm(up, cur) : cur;", "    ggml_tensor * tmp = up ? build_mm(up, cur) : cur;\n    if (ninfer_cpu_math && up) { tmp = clip_ninfer_cpu_bf16(ctx0, tmp); }")
    replace(clip, '        cb(tmp, "ffn_up_b", il);', '        if (ninfer_cpu_math) { tmp = clip_ninfer_cpu_bf16(ctx0, tmp); }\n        cb(tmp, "ffn_up_b", il);')
    replace(clip, "                cur = ggml_gelu_erf(ctx0, cur);", "                cur = ninfer_cpu_math ? clip_ninfer_cpu_merger_gelu(ctx0, cur) : ggml_gelu_erf(ctx0, cur);")
    replace(clip, "    if (down) {\n        cur = build_mm(down, cur);\n    }", "    if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }\n    if (down) {\n        cur = build_mm(down, cur);\n        if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }\n    }")
    replace(clip, "    return cur;\n}\n\n// MoE FFN", "    return ninfer_cpu_math ? clip_ninfer_cpu_bf16(ctx0, cur) : cur;\n}\n\n// MoE FFN")
    replace(clip, "        cur = build_mm(wo, cur);", "        cur = build_mm(wo, cur);\n        if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }")
    replace(clip, "    return cur;\n}\n\n// implementation of the 2D RoPE", "    return ninfer_cpu_math ? clip_ninfer_cpu_bf16(ctx0, cur) : cur;\n}\n\n// implementation of the 2D RoPE")
    replace(qwen2, "    ggml_tensor * inp_raw = build_inp_raw();", "    ggml_tensor * inp_raw = build_inp_raw();\n    if (ninfer_cpu_math) { inp_raw = clip_ninfer_cpu_bf16(ctx0, inp_raw); }")
    # Replace all four temporal conv calls together under a unique function-body anchor.
    for weight, inp in [("0", "inp_raw"), ("1", "inp_raw"), ("0", "inp_0"), ("1", "inp_1")]:
        old = f"ggml_conv_2d(ctx0, model.patch_embeddings_{weight}, {inp}, patch_size, patch_size, 0, 0, 1, 1)"
        new = f"(ninfer_cpu_math ? clip_ninfer_cpu_patch_projection(ctx0, model.patch_embeddings_{weight}, {inp}, patch_size) : {old})"
        replace(qwen2, old, new)
    replace(qwen3, "    ggml_tensor * inp = build_inp_with_temporal_merge();", "    ggml_tensor * inp = build_inp_with_temporal_merge();\n    if (ninfer_cpu_math) { inp = clip_ninfer_cpu_bf16(ctx0, inp); }")
    patch_merge = '''        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);'''
    replace(qwen3, patch_merge, '''        if (ninfer_cpu_math) {
            inp = clip_ninfer_cpu_merge_order(ctx0, inp, n_embd, n_patches_x, n_patches_y);
        } else {
''' + patch_merge + '\n        }')
    position_merge = '''    learned_pos_embd = ggml_cont_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    learned_pos_embd = ggml_reshape_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = ggml_cont_3d(
        ctx0, learned_pos_embd,
        n_embd, n_patches_x * n_patches_y, batch_size);'''
    replace(qwen3, position_merge, '''    if (ninfer_cpu_math) {
        learned_pos_embd = clip_ninfer_cpu_merge_order(ctx0, learned_pos_embd, n_embd, n_patches_x, n_patches_y);
    } else {
''' + position_merge + '\n    }')
    replace(qwen3, '        cb(inp, "patch_bias", -1);', '        if (ninfer_cpu_math) { inp = clip_ninfer_cpu_bf16(ctx0, inp); }\n        cb(inp, "patch_bias", -1);')
    replace(qwen3, '    cb(inp, "inp_pos_emb", -1);', '    if (ninfer_cpu_math) { inp = clip_ninfer_cpu_bf16(ctx0, inp); }\n    cb(inp, "inp_pos_emb", -1);')
    replace(qwen3, "            cur = build_mm(layer.qkv_w, cur);\n            cur = ggml_add(ctx0, cur, layer.qkv_b);", "            cur = build_mm(layer.qkv_w, cur);\n            if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }\n            cur = ggml_add(ctx0, cur, layer.qkv_b);\n            if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }")
    replace(qwen3, '            cb(Qcur, "Qcur_rope", il);', '            if (ninfer_cpu_math) {\n                Qcur = clip_ninfer_cpu_bf16(ctx0, Qcur);\n                Kcur = clip_ninfer_cpu_bf16(ctx0, Kcur);\n            }\n            cb(Qcur, "Qcur_rope", il);')
    replace(qwen3, "        cur = ggml_add(ctx0, cur, inpL);", "        cur = ggml_add(ctx0, cur, inpL);\n        if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }")
    replace(qwen3, '        cb(cur, "layer_out", il);', '        if (ninfer_cpu_math) { cur = clip_ninfer_cpu_bf16(ctx0, cur); }\n        cb(cur, "layer_out", il);')
    replace(qwen3, "        ffn_op_type::FFN_GELU, -1);", "        ninfer_cpu_math ? ffn_op_type::FFN_GELU_ERF : ffn_op_type::FFN_GELU, -1);")
    replace(clip, "    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);", "    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), gf)) {\n        return false;\n    }")
    replace(clip, "            img.set_size({sz, sz}, false, false);", "            img.set_size({sz, sz}, ctx_clip.ninfer_cpu_math, false);")
    replace(clip, "                    data_loaded += num_bytes;", "                    if (!fin) { throw std::runtime_error(\"short read while loading vision weights\"); }\n                    data_loaded += num_bytes;")
    declarations = '''
// NInfer private CPU-only hooks. Keep these out of NInfer's public interface.
struct clip_ninfer_cpu_memory {
    size_t graph_bytes;
    size_t scratch_bytes;
    size_t retained_graph_bytes;
    size_t retained_scratch_bytes;
    size_t metadata_bytes;
};
clip_ninfer_cpu_memory clip_ninfer_cpu_measure(clip_ctx * ctx, int width, int height, int frames, int threads);
void clip_ninfer_cpu_reserve(clip_ctx * ctx, int width, int height, int frames);
size_t clip_ninfer_cpu_weight_bytes(const clip_ctx * ctx);
void clip_ninfer_cpu_set_abort(clip_ctx * ctx, ggml_abort_callback callback, void * data);
ggml_tensor * clip_ninfer_cpu_bf16(ggml_context * ctx, ggml_tensor * x);
ggml_tensor * clip_ninfer_cpu_patch_projection(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, int patch);
ggml_tensor * clip_ninfer_cpu_merger_gelu(ggml_context * ctx, ggml_tensor * x);
ggml_tensor * clip_ninfer_cpu_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float scale);
ggml_tensor * clip_ninfer_cpu_position(ggml_context * ctx, ggml_tensor * table, int hidden, int width, int height, int side);
ggml_tensor * clip_ninfer_cpu_merge_order(ggml_context * ctx, ggml_tensor * x, int hidden, int width, int height);
'''
    replace(header, "struct clip_cap clip_get_cap(const char * fname);", "struct clip_cap clip_get_cap(const char * fname);\n" + declarations)
    definitions = '''
// NInfer private CPU-only hooks, adapter policy version 1.
ggml_tensor * clip_ninfer_cpu_bf16(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}

ggml_tensor * clip_ninfer_cpu_patch_projection(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, int patch) {
    // ggml_conv_2d's generic F32-weight path uses F16 im2col. Explicit F32 avoids
    // clipping BF16's exponent range. Preserve conv2d's layout and reduction axes.
    w = clip_ninfer_cpu_bf16(ctx, w);
    ggml_tensor * cols = ggml_im2col(ctx, w, x, patch, patch, 0, 0, 1, 1, true, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, cols, cols->ne[0], cols->ne[3]*cols->ne[2]*cols->ne[1]),
        ggml_reshape_2d(ctx, w, w->ne[0]*w->ne[1]*w->ne[2], w->ne[3]));
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    result = ggml_reshape_4d(ctx, result, cols->ne[1], cols->ne[2], cols->ne[3], w->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
}

ggml_tensor * clip_ninfer_cpu_merger_gelu(ggml_context * ctx, ggml_tensor * x) {
    return ggml_gelu_erf(ctx, x);
}

ggml_tensor * clip_ninfer_cpu_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float scale) {
    ggml_tensor * result = ggml_flash_attn_ext(ctx,
        ggml_cast(ctx, q, GGML_TYPE_F32), ggml_cast(ctx, k, GGML_TYPE_F32),
        ggml_cast(ctx, v, GGML_TYPE_F32), nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
    return result;
}

ggml_tensor * clip_ninfer_cpu_position(ggml_context * ctx, ggml_tensor * table, int hidden, int width, int height, int side) {
    if (width == side && height == side) { return table; }
    table = ggml_reshape_3d(ctx, table, hidden, side, side);
    table = ggml_permute(ctx, table, 2, 0, 1, 3);
    table = ggml_interpolate(ctx, table, width, height, hidden, 1,
        GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    table = ggml_permute(ctx, table, 1, 2, 0, 3);
    return ggml_cont_2d(ctx, table, hidden, width * height);
}

ggml_tensor * clip_ninfer_cpu_merge_order(ggml_context * ctx, ggml_tensor * x, int hidden, int width, int height) {
    x = ggml_cont_4d(ctx, x, hidden * 2, width / 2, height, 1);
    x = ggml_reshape_4d(ctx, x, hidden * 2, width / 2, 2, height / 2);
    x = ggml_permute(ctx, x, 0, 2, 1, 3);
    return ggml_cont_3d(ctx, x, hidden, width * height, 1);
}

size_t clip_ninfer_cpu_weight_bytes(const clip_ctx * ctx) {
    return ggml_backend_buffer_get_size(ctx->buf.get());
}

void clip_ninfer_cpu_set_abort(clip_ctx * ctx, ggml_abort_callback callback, void * data) {
    ggml_backend_cpu_set_abort_callback(ctx->backend_cpu, callback, data);
}

clip_ninfer_cpu_memory clip_ninfer_cpu_measure(clip_ctx * ctx, int width, int height, int frames, int threads) {
    if (!ctx->ninfer_cpu_math || ctx->backend != ctx->backend_cpu ||
        ctx->flash_attn_type != CLIP_FLASH_ATTN_TYPE_ENABLED) {
        throw std::runtime_error("NInfer CPU vision requires its private CPU/Flash Attention policy");
    }
    clip_ninfer_cpu_memory result{};
    result.retained_graph_bytes = ggml_backend_sched_get_buffer_size(ctx->sched.get(), ctx->backend_cpu);
    result.retained_scratch_bytes = ggml_backend_cpu_ninfer_work_size(ctx->backend_cpu);
    clip_image_f32_batch batch;
    for (int i = 0; i < frames; ++i) {
        clip_image_f32 image;
        image.set_size({width, height}, true, false);
        batch.entries.push_back(std::move(image));
    }
    auto builder = clip_get_graph_builder(ctx, batch);
    ggml_cgraph * graph = builder->build();
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(ctx->backend_cpu, ggml_graph_node(graph, i))) {
            throw std::runtime_error("CPU vision graph contains an unsupported CPU operator");
        }
    }
    result.scratch_bytes = ggml_graph_plan(graph, threads, nullptr).work_size;
    // One CPU backend only: reserve_size performs allocator measurement without allocating
    // compute storage. It may release an old compute buffer when its shape must grow.
    ggml_backend_sched_reserve_size(ctx->sched.get(), graph, &result.graph_bytes);
    ggml_backend_sched_reset(ctx->sched.get());
    result.metadata_bytes = ctx->buf_compute_meta.capacity();
    // Suppress clip's implicit warmup/reserve on encode: alloc_graph uses this measured shape.
    ctx->is_allocated = true;
    return result;
}

void clip_ninfer_cpu_reserve(clip_ctx * ctx, int width, int height, int frames) {
    // reserve_size records a no-allocation layout. alloc_graph can consider this layout
    // already valid despite null compute buffers, so explicitly reserve AFTER admission.
    ggml_backend_sched_reset(ctx->sched.get());
    clip_image_f32_batch batch;
    for (int i = 0; i < frames; ++i) {
        clip_image_f32 image;
        image.set_size({width, height}, true, false);
        batch.entries.push_back(std::move(image));
    }
    auto builder = clip_get_graph_builder(ctx, batch);
    if (!ggml_backend_sched_reserve(ctx->sched.get(), builder->build())) {
        throw std::runtime_error("CPU vision compute reservation failed");
    }
    ggml_backend_sched_reset(ctx->sched.get());
    ctx->is_allocated = true;
}
'''
    old_debug = "void clip_set_debug_output_embeddings(clip_ctx * ctx, bool enable) {\n    ctx->debug_output_embeddings = enable;\n}"
    replace(clip, old_debug, old_debug + "\n" + definitions)
    cpu_header = root / "ggml/include/ggml-cpu.h"
    cpu_source = root / "ggml/src/ggml-cpu/ggml-cpu.cpp"
    replace(cpu_header, "    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback", "    GGML_BACKEND_API size_t ggml_backend_cpu_ninfer_work_size(ggml_backend_t backend_cpu);\n    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback")
    replace(cpu_source, "static enum ggml_status ggml_backend_cpu_graph_compute(", "size_t ggml_backend_cpu_ninfer_work_size(ggml_backend_t backend) {\n    return ((ggml_backend_cpu_context *) backend->context)->work_size;\n}\n\nstatic enum ggml_status ggml_backend_cpu_graph_compute(")
    # Memory-only GGUF metadata plus one synchronous, checked fill per final CPU tensor.
    # The standard file loader retains its existing path and behaviour.
    memory_api = '''
using clip_ninfer_tensor_fill = bool (*)(void * user, const char * name, enum ggml_type type,
    const int64_t * dimensions, int rank, void * destination, size_t bytes);
struct clip_init_result clip_ninfer_init_from_metadata(const uint8_t * metadata, size_t metadata_bytes,
    clip_ninfer_tensor_fill fill, void * user, struct clip_context_params params);
'''
    replace(header, "struct clip_cap clip_get_cap(const char * fname);",
        "struct clip_cap clip_get_cap(const char * fname);\n" + memory_api)
    replace(clip, "    std::string fname;\n\n    size_t model_size",
        "    std::string fname;\n    clip_ninfer_tensor_fill ninfer_fill = nullptr;\n    void * ninfer_fill_user = nullptr;\n\n    size_t model_size")
    replace(clip, '''            void * progress_user_data = nullptr)
        : fname(fname),
          progress_callback(progress_cb),
          progress_callback_user_data(progress_user_data) {''', '''            void * progress_user_data = nullptr,
            const uint8_t * memory_metadata = nullptr,
            size_t memory_metadata_bytes = 0,
            clip_ninfer_tensor_fill memory_fill = nullptr,
            void * memory_fill_user = nullptr)
        : fname(fname),
          ninfer_fill(memory_fill),
          ninfer_fill_user(memory_fill_user),
          progress_callback(progress_cb),
          progress_callback_user_data(progress_user_data) {''')
    replace(clip, "        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));", '''        if (memory_metadata) {
            if (!memory_fill || memory_metadata_bytes < 24 || memory_metadata_bytes > 1024 * 1024) {
                throw std::runtime_error("invalid NInfer memory GGUF metadata or fill callback");
            }
            struct memory_source { const uint8_t * data; size_t size; } source{memory_metadata, memory_metadata_bytes};
            const auto read = +[](void * raw, void * output, uint64_t offset, size_t bytes) -> size_t {
                const auto & source = *static_cast<memory_source *>(raw);
                if (offset > source.size || bytes > source.size - offset) { return 0; }
                std::memcpy(output, source.data + offset, bytes);
                return bytes;
            };
            ctx_gguf = gguf_context_ptr(gguf_init_from_callback(
                read, &source, 64 * 1024, memory_metadata_bytes, params));
        } else {
            ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        }''')
    replace(clip, '''        auto fin = open_ifstream_binary(fname);
        if (!fin) {''', '''        std::ifstream fin;
        if (!ninfer_fill) { fin = open_ifstream_binary(fname); }
        if (!fin && !ninfer_fill) {''')
    replace(clip, '''            fin.seekg(it->second, std::ios::beg);
            fin.read(reinterpret_cast<char*>(result.data()), n_bytes);''', '''            if (ninfer_fill) {
                throw std::runtime_error("memory Vision loader does not support file-backed scalar vectors");
            }
            fin.seekg(it->second, std::ios::beg);
            fin.read(reinterpret_cast<char*>(result.data()), n_bytes);''')
    replace(clip, '''                    const size_t offset = it_off->second;
                    fin.seekg(offset, std::ios::beg);
                    if (!fin) {
                        throw std::runtime_error(string_format("%s: failed to seek for tensor %s\\n", __func__, t->name));
                    }
                    size_t num_bytes = ggml_nbytes(cur);
                    if (ggml_backend_buft_is_host(buft)) {''', '''                    const size_t offset = it_off->second;
                    size_t num_bytes = ggml_nbytes(cur);
                    if (ninfer_fill) {
                        if (!ggml_backend_buft_is_host(buft) || !cur->data ||
                            !ninfer_fill(ninfer_fill_user, t->name, cur->type, cur->ne,
                                         ggml_n_dims(cur), cur->data, num_bytes)) {
                            throw std::runtime_error(string_format("NInfer memory Vision tensor fill failed: %s", t->name));
                        }
                    } else {
                    fin.seekg(offset, std::ios::beg);
                    if (!fin) {
                        throw std::runtime_error(string_format("%s: failed to seek for tensor %s\\n", __func__, t->name));
                    }
                    if (ggml_backend_buft_is_host(buft)) {''')
    replace(clip, '''                    if (!fin) { throw std::runtime_error("short read while loading vision weights"); }
                    data_loaded += num_bytes;''', '''                    if (!fin) { throw std::runtime_error("short read while loading vision weights"); }
                    }
                    data_loaded += num_bytes;''')
    replace(clip, "struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params) {", '''template <typename Factory>
static struct clip_init_result clip_ninfer_init_impl(const char * label,
        struct clip_context_params ctx_params, Factory factory) {''')
    replace(clip, '''        clip_model_loader loader(fname,
            /* skip_tensors */ false,
            ctx_params.progress_callback,
            ctx_params.progress_callback_user_data);''', '''        clip_model_loader loader = factory();''')
    replace(clip, '''        LOG_ERR("%s: failed to load model '%s': %s\\n", __func__, fname, e.what());''', '''        LOG_ERR("%s: failed to load model '%s': %s\\n", __func__, label, e.what());''')
    wrappers = '''struct clip_init_result clip_init(const char * fname, struct clip_context_params params) {
    return clip_ninfer_init_impl(fname, params, [&] {
        return clip_model_loader(fname, false, params.progress_callback,
                                 params.progress_callback_user_data);
    });
}

struct clip_init_result clip_ninfer_init_from_metadata(const uint8_t * metadata,
        size_t metadata_bytes, clip_ninfer_tensor_fill fill, void * user,
        struct clip_context_params params) {
    return clip_ninfer_init_impl("<NInfer embedded Vision>", params, [&] {
        return clip_model_loader("<NInfer embedded Vision>", false, params.progress_callback,
            params.progress_callback_user_data, metadata, metadata_bytes, fill, user);
    });
}

'''
    replace(clip, "struct clip_cap clip_get_cap(const char * fname) {", wrappers + "struct clip_cap clip_get_cap(const char * fname) {")
    stamp.write_text(json.dumps({"script": script_hash,
        "files": {name: sha(root / name) for name in PINNED_FILES}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    patch(parser.parse_args().source)
