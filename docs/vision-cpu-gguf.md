# GGML CPU vision with an external mmproj GGUF

This optional backend moves the complete vision encoder and merger/projector to the CPU. The native NInfer text model, KV cache, MTP, and text consumption of image embeddings remain on the GPU. CUDA vision remains the default.

The CPU backend uses existing GGML operators and a pinned llama.cpp/mtmd dependency; it does not load a llama.cpp text model. OpenMP and llamafile matrix kernels are enabled alongside AVX2/FMA/F16C. The CPU path emits contiguous BF16 `[merged_tokens, 5120]` embeddings and uploads them with cacheable pinned memory on the text execution stream.

## Supported models

- Native Qwen3.8-27B and Qwen3.6-27B `groupwise-int` text artifacts.
- An external BF16/F32 `mmproj-BF16.gguf` matching the supported 27B vision topology: 334 tensors, 27 layers, hidden width 1152, 16 heads, head dimension 72, patch size 16, temporal merge 2, spatial merge 2, and output width 5120, without DeepStack.
- Images and videos, using NInfer's existing preprocessing and media token budget.

Initialization validates GGUF metadata, tensor types/shapes, file integrity, and weight-memory admission. Other vision layouts and the 35B target are rejected. Embedded `.ninfer` vision weights are not consumed by the CPU backend in this version. External BF16 weights can differ from the quantized embedded GPU weights; the two paths are not expected to produce bit-identical embeddings.

## Build

Use the Windows CUDA/MSVC/Ninja prerequisites from the repository [build instructions](../README.md#build-from-source), Python 3, and a configured vcpkg installation. Run in an MSVC x64 developer terminal; set `VCPKG_ROOT` to your vcpkg directory:

```powershell
cmake -S . -B build-vision-cpu -G Ninja `
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
    -DNINFER_BUILD_CPU_VISION=ON -DBUILD_TESTING=ON `
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
    -DVCPKG_TARGET_TRIPLET=x64-windows-static-md
cmake --build build-vision-cpu --parallel 4
```

The dependency is pinned to llama.cpp revision `b81c99b479d4c24e5eeca10de99032ebd343ef8f`, with a SHA256-verified archive. CMake privately applies [the guarded graph adaptation](../tools/dependencies/patch_ggml_cpu_vision.py) to a copy in the build directory. The script checks pinned file hashes, supports repeat application, and rejects changed inputs. Its source tree is not vendored into this repository.

For an offline build, add `-DNINFER_GGML_SOURCE_DIR=<full-unmodified-pinned-source-tree>`. Use the complete tree extracted from the pinned archive. Changing that source override requires a new build directory.

`NINFER_BUILD_CPU_VISION=OFF` keeps the original CUDA configuration and provides a clear error when CPU vision is requested. A CPU-only test project is also available:

```powershell
cmake -S tests/vision_cpu -B build-cpu-tests -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu-tests --parallel 4
ctest --test-dir build-cpu-tests --output-on-failure
```

The standalone project exercises the CPU bridge, cache, startup/CLI options, numerical graphs, real GGUF checks, and `bench_cpu_vision`; it does not detect CUDA. To test the backend-not-built contract, configure it with `-DNINFER_BUILD_CPU_VISION=OFF`.

When redistributing a Windows executable, provide its required x64 MSVC runtime, including `vcomp140.dll` for OpenMP, and preserve applicable NInfer, GGML/llama.cpp, and runtime licenses. Model weights and runtime binaries are not included in this source change.

## Run

```powershell
.\build-vision-cpu\apps\ninfer-serve.exe models\qwen3_8_27b.ninfer `
    --host 127.0.0.1 --port 8111 `
    --vision-device cpu --vision-mmproj models\mmproj-BF16.gguf `
    --vision-cpu-threads 6 --vision-cpu-memory-mib 4096 --vision-cpu-cache-mib 128 `
    --max-context 32768 --kv-capacity 32768 --kv-dtype int8 `
    --spec mtp --draft-tokens 3 --lm-head-draft `
    --request-log-jsonl requests.jsonl
```

Existing API authentication and serving controls apply unchanged. Both `ninfer` and `ninfer-serve` expose the CPU options; the public `EngineOptions` API exposes corresponding fields.

| Option | Default | Behavior |
| --- | --- | --- |
| `--vision-device` | `cuda` | Select `cpu` for this backend. |
| `--vision-mmproj` | unset | External compatible GGUF; also enables CPU vision when no device is explicitly selected. |
| `--vision-cpu-threads` | `6` | CPU workers, 1 through 512. |
| `--vision-cpu-memory-mib` | `4096` | Host-memory admission budget for CPU vision resources. |
| `--vision-cpu-cache-mib` | `128` | Image embedding cache cap within that budget; `0` disables caching. |

CPU settings explicitly combined with CUDA mode are rejected. `--vision-max-tokens` defaults to 8192 and limits each encoding item (native CUDA scratch is sized from it). Since the 2026-10-07 multi-image fix, `--vision-request-max-tokens` independently limits the total across request/history (default 32768). See [multi-image budgets](vision-multi-image-budget.zh-CN.md).

## Memory, caching, and cancellation

The total CPU budget includes vision weights, graph/scratch storage, worker-stack allowance, bridge buffers, and output/upload staging. It is not a limit on the entire server's working set: HTTP, tokenizer, other frontend payloads, and the text model have their own storage.

When caching is enabled, the configured cache cap plus 256 KiB of bounded metadata is reserved within that total. With defaults, the encoder receives 3967.75 MiB. Cache payloads allocate on demand; 128 MiB is not preallocated or added outside the 4096 MiB budget. Cache capacity must be smaller than the total, and remaining resources must still pass admission. Oversized images can be rejected before encoder allocation; the server remains usable after a rejected request.

The cache belongs to one loaded encoder instance and stores exact final BF16 still-image embeddings. It is process-local, bounded by payload bytes and 256 entries, and evicts the least recently used entry. Videos bypass the cache. An output larger than the cap is encoded normally without retention.

The key combines SHA256 of processed float patch bytes, grid dimensions, media type, and encoder identity (weight content and numerical policy). Text, URLs, and chat history do not affect the image key. The digest is calculated in the runtime using the existing frontend SHA256 implementation. Hits still validate inputs and host-memory admission, respond to cancellation, and perform GPU upload. Failed/cancelled encodes never insert partial outputs. Encoder/cache access to shared graph buffers is serialized; cancellation checks occur at graph-node boundaries rather than interrupting a running operator.

`request_done.cpu_vision_cache` records `hits`, `misses`, and `encode_calls` for CPU media items that successfully complete encoding/cache retrieval and GPU upload. Successful disabled-cache, video, missing-key, and oversized-entry bypasses count as misses/encode calls. Failed in-flight items are excluded. Earlier completed items remain counted when a later item is cancelled. Compatible text-prefix reuse can skip media processing entirely, leaving all three counters at zero.

Example of an independent cache hit after changing text before an image:

```json
{"cpu_vision_cache":{"hits":1,"misses":0,"encode_calls":0}}
```

This hit still incurs decoding, preprocessing, hashing, upload, and text inference. To compare encoder performance independently of caching, use `--vision-cpu-cache-mib 0` and verify that text-prefix cache tokens are zero.

## Numerical and functional validation

The private graph adapter retains F32 patch projection/im2col, F32 attention K/V, explicit BF16 rounding boundaries, block tanh-GELU, merger GELU-erf, positional interpolation, and spatial merge ordering. Independent scalar FP64 oracles cover the changed operators, including BF16 matrix products and irregular attention/matrix tails. Real GGUF tests cover rectangular inputs, video temporal ordering, cancellation/retry, and tight-memory repeated encoding. Fake-encoder and CUDA-upload tests cover exact cache reuse, key separation, byte/entry eviction, failures, cancellation, concurrency, and request statistics.

To run tests with real local weights:

```powershell
$env:NINFER_TEST_VISION_GGUF = (Resolve-Path models\mmproj-BF16.gguf).Path
$env:NINFER_QWEN3_8_27B_WEIGHTS = (Resolve-Path models\qwen3_8_27b.ninfer).Path
ctest --test-dir build-vision-cpu --output-on-failure --parallel 1
```

The tested Windows CPU-vision build completed 92 CTests: **88 passed, 0 failed, 4 skipped** because Qwen3.6/35B real-model fixtures were unavailable. Local API checks covered MTP, images, multiple images, video, compatible-prefix continuation, cancelled image encoding/recovery, budget rejection/recovery, and relocated runtime loading.

The [validation record](vision-cpu-validation.json) includes the skipped fixture names, timing samples, full-context settings, and memory measurements reported below.

## Measured behavior on one RTX 4090 system

2026-09-29, i5-10400 with six CPU workers, RTX 4090 24 GB, MSVC 19.35 and CUDA 13.3.73. The same BF16 GGUF and images were used; MTP, thinking, and text-prefix reuse were disabled for the image timing comparison. Each uncached result averages three encodes; cache-hit HTTP timing averages two requests with changed text before the image.

| Image | Original CPU encoder | OpenMP + llamafile, cache disabled | Encoding time reduction | Cache-hit whole HTTP request |
| --- | ---: | ---: | ---: | ---: |
| Colors, 72 vision tokens | 3.139 s | 2.035 s | 35.2% | 0.521 s |
| Chart, 384 vision tokens | 18.879 s | 13.415 s | 28.9% | 0.648 s |

Hits performed no vision encode; their vision paths, including fingerprint/admission/upload, averaged approximately 11/56 ms. These small local samples do not isolate the two GGML optimizations' contributions or predict all image/CPU performance. This is not a comprehensive visual quality evaluation; one MTP chart prompt answered only the circle count in both CPU versions, while the non-MTP timing prompts and multi-image prompt also read the title.

A separate CPU-vision + `rk4v4-e8` + MTP3 capacity test actually prefilled **262144 input tokens**, including an image, with zero text-prefix cache tokens. It sampled one output token successfully without OOM. The native engine can sample that first output from the last input position's logits without appending another KV position. The full-window request did not execute speculative rounds, although MTP weights and KV pools remained loaded.

| Full-window measurement | Result |
| --- | ---: |
| Input execution time | 215.7 s, approximately 1217.6 prefill tokens/s |
| Whole-card peak used memory | 23809 MiB |
| Minimum free GPU memory | 334 MiB |
| Windows process shared GPU memory | 76 MiB, unchanged through prefill |

Whole-card memory was sampled about every two seconds and includes desktop applications. This configuration fit under that desktop load with very little headroom; it is not a general safe-capacity guarantee. The workload used repetitive synthetic text and a small image, not a retrieval-quality test, large-image/video stress test, or long-output test. Reserve output positions within the configured window during ordinary use.
