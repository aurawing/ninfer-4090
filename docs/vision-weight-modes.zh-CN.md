# Qwen3.8-27B 五种视觉权重模式

本分支将视觉权重来源和执行设备分开选择。正文模型、KV 缓存与 MTP 仍使用 NInfer 的 CUDA 路径；选择 GGML 视觉时，仅最终 BF16 图片 embedding 进入正文模型。当前实现只支持 Qwen3.8/Qwen3.6-27B 的 27 层、1152 隐藏维、5120 输出维视觉拓扑。

| `.ninfer` 内嵌视觉权重 | 启动参数 | 视觉执行与权重位置 |
| --- | --- | --- |
| 有 | `--vision` | 原生 NInfer CUDA，内嵌量化权重在显存 |
| 无 | 不启用 `--vision` | 纯文本，不创建视觉编码器或显存工作区 |
| 有 | `--vision --vision-device cpu` | GGML CPU，启动时直接解码内嵌视觉权重 |
| 无 | `--vision --vision-device cpu --vision-mmproj mmproj-BF16.gguf` | GGML CPU，外挂 GGUF 权重在内存 |
| 无 | `--vision --vision-device cuda --vision-mmproj mmproj-BF16.gguf` | GGML CUDA，外挂 GGUF 权重在显存 |

内嵌视觉权重和外挂 GGUF 同时存在时，显式指定的外挂文件优先；内嵌张量仍会接受完整性校验，但不占视觉设备权重空间。只有 `--vision-mmproj` 而未指定设备时，继续隐式选择 CPU。视觉关闭时，有内嵌权重的完整模型也能运行纯文本。视觉张量只存在一部分的 `.ninfer` 会被拒绝；没有内嵌视觉权重却启用原生 CUDA、或 CPU 模式既无内嵌权重也无外挂文件时，会在加载阶段明确报错。

## 构建

在仓库的标准 Windows MSVC、CUDA 13.x、vcpkg 和 Ninja 环境中，增加：

```powershell
cmake -S . -B build-vision -G Ninja `
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
    -DNINFER_BUILD_CPU_VISION=ON -DNINFER_BUILD_GGML_CUDA_VISION=ON `
    -DBUILD_TESTING=ON `
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
    -DVCPKG_TARGET_TRIPLET=x64-windows-static-md
cmake --build build-vision --parallel 4
```

GGML CUDA 是独立的可选项，并要求 `NINFER_BUILD_CPU_VISION=ON`。不编入时，显式选择外挂 CUDA 会返回清楚的错误，不会自动改为 CPU。依赖固定在 llama.cpp `b81c99b479d4c24e5eeca10de99032ebd343ef8f`，通过仓库的受校验私有补丁接入；构建时不会改动原始依赖源码。Windows 分发 GPU 版时，需提供匹配 CUDA 13 的 `cublas64_13.dll` 和 `cublasLt64_13.dll`，以及 OpenMP/MSVC 运行时。

## 启动示例

以下示例均以 `ninfer-serve.exe` 为程序名；替换为构建目录内的实际绝对路径。根据自己的内存与显存预算调整上下文和 KV 容量。

```powershell
# 内嵌权重，完整视觉编码器在 CPU；不需要 --vision-mmproj
ninfer-serve.exe models\qwen3_8_27b.ninfer --host 127.0.0.1 --port 8111 `
    --vision --vision-device cpu --vision-max-tokens 1024 --vision-cpu-threads 6 `
    --vision-cpu-memory-mib 4096 --vision-cpu-cache-mib 128 `
    --max-context 32768 --kv-capacity 32768 --kv-dtype int8

# 外挂 BF16 GGUF，完整视觉编码器在 4090
ninfer-serve.exe models\qwen3_8_27b.ninfer --host 127.0.0.1 --port 8111 `
    --vision --vision-device cuda --vision-mmproj C:\models\mmproj-BF16.gguf `
    --max-context 32768 --kv-capacity 32768 --kv-dtype int8
```

内嵌 `.ninfer` 视觉 payload 约 282 MiB；其解码后的 GGML CPU 权重约 888 MiB。外挂 BF16 GGUF 的视觉权重在 GGML CUDA 上约占 888 MiB，另需计算图和图片缓冲。外部 GPU 模式首版沿用 host BF16 embedding 到正文模型的上传接口，因此存在一次设备到主机再上传的数据传输。本机 256K / `rk4v4-e8` 的完整实测见下文；外挂 GGUF 的 GPU 模式在当前桌面负载下启动时容量不足。

CPU 图继续使用 OpenMP、llamafile、显式预算与图片 embedding 缓存。GGML CUDA 路径复用同一图片 bridge 和缓存接口，显式检查权重实际位于 GPU；其取消回调在图片组之间检查，不支持像 CPU 路径那样在单个 GGML 图节点之间中断。纯文本实例收到图片请求应按现有前端的视觉能力检查返回错误。

CPU 图像编码目前在单个推理执行线程的 prefill 阶段同步运行。如果把 `--max-concurrency` 提高到大于 1，一张耗时图片编码期间其他正在运行的请求也可能暂停；本机启动脚本使用单并发。要真正消除阻塞，需要把图片 embedding 的计算提前到请求准备线程，并在提交/取消/前缀复用处管理其生命周期，不能只扩大 CPU 线程数。`--vision-max-tokens` 是图像 token 总上限，同时控制缩放与工作区预算；直接调用二进制程序默认仍为 8192。本机 `start-ninfer.ps1` 的 CPU 模式默认传入 1024，可用 `-VisionMaxTokens 2048` 等参数显式调高，以换取高分辨率/OCR 输入。

本机无 MTP 的 GGML CUDA 模式曾在启动时测得正文 CUDA Graph 准备占用约 75 MiB，高于原有 64 MiB 预算。现在该模式按每并发 128 MiB 预留并计入启动显存 admission；常规原生 CUDA 与 GGML CPU 仍按原预算。GPU 模式在默认 CUDA Graph 和启动脚本的 MTP 3 配置下均通过实际图片请求。

## 验证

`tests/artifact/test_vision_row_decode.cpp` 独立验证 Q4/Q5/Q6/W8 行编码、FP16 scale 和 BF16 最近偶数舍入。`ninfer_qwen3_6_27b_embedded_vision_test` 使用真实 `.ninfer` 完成内嵌权重 CPU 编码与无外挂启动；`ninfer_ggml_cuda_vision_real_test` 使用真实 GGUF 和 4090 验证 GPU 权重与图片编码。设置 `NINFER_QWEN3_8_27B_WEIGHTS`、`NINFER_TEST_VISION_GGUF` 后运行 `ctest --test-dir build-vision --output-on-failure`。没有真实模型文件时相关测试按 CTest 的 skip 规则跳过。

现有 [CPU 视觉文档](vision-cpu-gguf.md)保留第一阶段外挂 GGUF 的性能及 256K 上下文测量；那些数值不代表本阶段内嵌解码或 GGML CUDA 的性能。

2026-09-30 在本机的 `build-vision-integration` 运行了全部 96 项 CTest：92 通过、0 失败、4 项缺少 3.6/35B 测试模型而跳过。使用新的独立运行包 `D:\deeplearning\NInfer\runtime\ninfer-rtx4090-windows-x64-vision-modes-v3`，在 8192 token、`rk2v4-e8` 下分别验证纯文本、原生 CUDA 视觉、内嵌 `.ninfer` 权重的 GGML CPU 视觉、外挂 GGUF 的 GGML CPU 视觉、外挂 GGUF 的 GGML CUDA 视觉。四个视觉请求使用同一张红蓝测试图片，均得到红蓝识别结果；纯文本算术请求返回 `5`。CPU 内嵌与 CUDA 外挂两种模式还通过了带 API Key、MTP 3 的本机启动脚本图片请求。优化输入校验后再次运行全量 CTest，结果不变；缓存命中仍做完整校验，未命中避免在缓存层重复扫描，RGB 解包复用已验证的形状。测试进程已停止。日志位于 `D:\VSCodeProjects\ninfer-4090-research\embedded-vision`。

随后从原始 `qwen3_8_27b.ninfer` 生成本机专用的无视觉测试制品，保留了全部 791 个非视觉对象和逐字节相同的 payload。该文件不作为仓库模型发布；使用它完成了纯文本、外挂 GGUF CPU/GPU 三条服务路径的真实模型请求。

2026-09-30 在 RTX 4090 上用 v3 运行包、`--max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4-e8`、MTP-3、单并发实测五种模式。视觉请求使用一张小图，`--vision-max-tokens 1024`。四个成功模式均由 `/v1/messages/count_tokens` 确认输入恰好 262144 token，服务日志记录 `computed_prefill_tokens=262144`、`prefix_cache_hit_tokens=0`，并各输出 1 token。MTP 权重和 KV 池保持加载；满窗请求未执行推测轮次。

| 模式 | 满窗结果 | 采样最低空闲显存 |
| --- | --- | ---: |
| 内嵌权重、GGML CPU | 成功 | 307 MiB |
| 内嵌权重、原生 CUDA | 成功 | 338 MiB |
| 无内嵌权重、外挂 BF16 GGUF、GGML CPU | 成功 | 620 MiB |
| 无内嵌权重、外挂 BF16 GGUF、GGML CUDA | 启动容量检查失败，约差 298 MiB | — |
| 无视觉纯文本 | 成功 | 630 MiB |

整卡显存采样包含桌面程序，且各轮启动基线不同，不能把表中差额全部归因于视觉模式。成功模式余量仍小；这次重复文本加小图片的容量测试没有验证长输出、较大图片或长上下文检索质量。普通生成应在 262144 token 总窗口内为输出留位。本机原始结果位于 `D:\deeplearning\NInfer\logs\vision-modes-v3-rk4`；本地测试服务已退出。
