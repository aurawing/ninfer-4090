# KVMem 4090 实施进度

记录方式见 [README](README.zh-CN.md) 第 7 节。每条记录写明日期、提交号，并注明数字是实测还是估算。

## 待用户决定的问题

当前没有待决定的 D1–D15 冲突。dense 量化 small-T 注意力同步 bug 已按用户授权在独立分支修复并合并到 `feat/kvmem`；Q5 残差原子累加顺序波动按用户决定保留。后续传输修正及阶段 3 第 3 步继续执行；第 3 步完成后等审阅，不接入运行时。D3 的后续方向已由用户在 2026-09-30 确认。

## 阶段 0′：基线

- 状态：基线与阶段 0′ 补测已完成（2026-09-30，`78989fc0`；补测取得当前负载下的锁页分配成功/失败边界，见下文）。测量数据目录：`D:\deeplearning\NInfer\logs\kvmem-stage0-baseline`，仓库外。
- 模型：`D:\deeplearning\NInfer\models\qwen3_8_27b.ninfer`；视觉 GGUF：`D:\deeplearning\kvmem-v0.16.0-rc3-windows-x86_64-cuda13.2.86\models\Qwen3.8-27B\Qwen3.8-27B\mmproj-BF16.gguf`。两个文件均已确认存在。
- D3 静态核对：`src/ops/kernel/paged_kv_address.cuh` 的 `paged_kv_physical_page()` 用 `block_table[position >> 6]`；BF16 prefill 用 `block_table[kb]` 且 `kb*64` 表示 key 位置，量化 prefill 用 `block_table[tile_k0 >> 6]`；BF16/int8 decode 从 `last_pos+1` 推导连续窗口及 split 范围，并用 `block_table[key_position >> 6]` 和原始 key 位置做因果掩码。`PagedKVAllocation::materialize_tokens/trim_tokens` 将 frontier 转为连续 `ceil(tokens/64)` 页，`ProgramImplCore::materialize_sequence_kv/trim_sequence_kv` 沿用该连续范围。物理 page ID 可任意，但**逻辑页槽位必须对应 `[64i,64i+63]`**；D3 关于可直接紧凑重排视图的前提不满足。此结论来自源码检查，不是性能实测；不否定已 RoPE 的 K 逐字节搬运，只限制块表/掩码的实现方式。
- D3 具体位置（均为 `78989fc0` 源码行号）：`paged_kv_address.cuh:16` 统一 position→页号；`gqa_attention_prefill_bf16.cuh:225–257,407` 从零连续扫 key block；`gqa_attention_prefill_i8.cuh:241–247,457,514` 以 key 位置右移六位查表；`gqa_attention_decode_bf16.cuh:121–138,198,210,283–295` 与 `gqa_attention_decode_i8.cuh:185–202,403,547–559,632` 用连续 `[0,last_pos]` 窗口、原始 key 位置掩码和页号查表；`gqa_attention_decode.cuh:173–186` reducer 用 `last_pos+1` 计算有效 split；`paged_kv_cache.cu:24–26,485–486,529,543,552–568` 把 frontier 映射成连续页向量并从下标 0 发布块表；`program_impl.h:1222–1252` 沿用连续 materialize/trim，`525–668,2555–2580,2694–2720` 的磁盘 restore/snapshot 也按逻辑页序号索引（该磁盘状态缓存在 kvmem 模式拟关闭）。`text_context_impl.h:824–849` 在写 KV 之前已经对 Q/K 调用 `ops::rope`，RoPE 坐标与 cache ordinal 分开传入；因此保留原始页号时无需对换入 K 再做 RoPE。
- 用户确认后续阶段保留**原始逻辑页号**，不把紧凑视图槽位直接当原始位置；阶段 3 需要显式处理未驻留页。这是对 D3 前提的修订，不改变“262K 内不做 re-RoPE”的方向。本轮不实现。
- 首次构建因当前 shell 未加载 MSVC 开发者环境、找不到 C/C++ 标准头文件而失败；随后使用 `VsDevCmd.bat -arch=x64 -host_arch=x64` 调用 `cmake --build build-vision-integration --parallel 4`，完整构建成功（实测，退出码 0）。
- 全量 `ctest --test-dir build-vision-integration --output-on-failure -j 4`：96 项中 **92 通过、4 跳过、0 失败**（实测）。恢复临时诊断源码并重新构建后，以 `NINFER_QWEN3_8_27B_WEIGHTS` 和 `NINFER_TEST_VISION_GGUF` 指向上述实物文件再次取得相同结果；最终输出为数据目录 `ctest-final.log`。4 项跳过源于缺少 Qwen3.6-27B prefix / 35B-A3B 测试制品。
- 真实模型加载的 262144 token、`rk4v4-e8`、MTP-3、CPU 视觉、单并发服务就绪后，进程内 `cudaMemGetInfo` 报告空闲 **559.91 MiB**；同一时刻整卡 `nvidia-smi` 约 **425 MiB**（实测；启动日志 `prepare-server.log`、`prepare.json`）。两种口径不同，不能混用。
- 真实模型已加载时，在独立进程分别执行的微基准（64 MiB 块、预热 3 次、正式 10 次；实测）：锁页 H2D **9.83 GiB/s**，可分页 H2D **4.61 GiB/s**；锁页 D2H **9.68 GiB/s**，可分页 D2H **4.27 GiB/s**；可分页→锁页 `memcpy`，1 线程 **6.13 GiB/s**，2 线程 **5.87 GiB/s**。期间系统可用物理内存约 **19.6 GiB**。这些 CUDA 微基准运行在与模型服务不同的进程，其 WDDM 进程内 `cudaMemGetInfo` 读数不反映整卡可用显存。
- 锁页内存单块 `cudaHostAlloc` 的首次二分探针在 **8192 MiB** 安全上限成功；复测在模型服务已加载时，以“至少保留 4 GiB 可用物理内存”为约束，将上限提高到 **15360 MiB（15 GiB）**，再次二分并成功分配、释放。各次尝试均为独立微基准进程；复测前系统可用物理内存约 **19.0 GiB**。没有触及 CUDA/Windows 分配失败边界，因此严格结论是**可分配的单块锁页内存至少 15 GiB**，不是绝对最大值。原始记录分别为 `microbench-max-pin-8g.json` 与 `microbench-max-pin.json`，模型服务日志为 `pin-probe-server.log`（实测）。
- 已固定 9 个纯文本用例：32K、128K、262K 各含 1 个合成长提示词及 10%/90% 深度的两个 needle 提示词，真实输入分别为 **32704、131008、262080 token**，每档留 64 token 生成预算。输入 JSON、校准探针、SHA-256 见数据目录 `prepare.json`。基线命令均使用 `--kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 1024 --greedy --no-thinking --vision --vision-device cpu --max-new 64`，每例独立进程，无 KV 前缀复用；RTX 4090 同时驱动桌面，未主动施加额外 GPU 负载。
- 九个用例的未插桩 dense 基线均完成，真实生成均为 64 个贪心 token，`reused prompt tokens=0`；前 64 个 token ID 与原始 CLI 输出记录在各例 `*.result.json`/`*.stderr.txt`（实测）。就绪后空闲显存取引擎启动完成后的进程内 `cudaMemGetInfo`，每例独立加载模型，桌面 GPU 占用会漂移。

  | 输入类型 | 真实输入 token | 文本 prefill | 就绪后空闲显存 | needle 代码 |
  |---|---:|---:|---:|---|
  | 合成 32K | 32704 | 14.396 s | 4.36 GiB | — |
  | needle 32K，10% | 32704 | 14.422 s | 4.36 GiB | 答对 |
  | needle 32K，90% | 32704 | 14.452 s | 4.36 GiB | 答对 |
  | 合成 128K | 131008 | 77.277 s | 2.90 GiB | — |
  | needle 128K，10% | 131008 | 77.788 s | 2.90 GiB | 答对 |
  | needle 128K，90% | 131008 | 78.137 s | 2.90 GiB | 答对 |
  | 合成 262K | 262080 | 211.176 s | 657.91 MiB | — |
  | needle 262K，10% | 262080 | 213.126 s | 657.91 MiB | 答对 |
  | needle 262K，90% | 262080 | 213.580 s | 657.91 MiB | 答对 |

- 九例均另存了 **64 步 × 每步 top-8** 的目标模型 logits 摘要，见每例 `*.top8.json`；原始完整词表 BF16 输出见 `*.target-logits.bf16.bin`。JSON 中的 `logits_fp32` 是**模型 BF16 输出转换成 FP32 表示**，并非量化前的 FP32 中间值。临时诊断构建记录 MTP 接受数，并因数据导出而关闭 CUDA Graph；诊断时间不用于上表性能基线。每例 64 步 top-1 均与该次诊断生成的 token ID 相符；其中 **8/9 例**的全部 64 个 ID 与未插桩 dense 基线完全一致。
- 例外是 `needle-128k-90` 的第 9 个生成 token（零起始下标 8）：未插桩基线为 ID **271**，诊断为 ID **198**，其余 63 个 ID 相同。诊断导出的两个候选 BF16 logits **同为 24.125**，并列第一。额外关闭 CUDA Graph、但不启用诊断钩子的完整复跑仍选出 ID 271，见 `needle-128k-90.no-graph-check.json`。这例 top-8 是诊断路径的数值摘要，不能声称与原基线路径逐 token 完全相同。原始基线的 64 个 token ID 仍由其 `*.result.json` 记录。

### 阶段 0′ 补测（2026-09-30，`78989fc0`）

- 复用阶段 0′ 的 262144 token、`rk4v4-e8`、MTP-3、CPU 视觉、C=1 服务命令；`/health` 返回就绪**之后**才启动下列各独立测量进程。服务就绪时整卡 `nvidia-smi` 空闲 **377 MiB**；Windows 可用物理内存 **19.49 GiB**（实测，`followup-server.log`、`followup-measure.json`、`followup-available.json`）。WDDM 下测量进程自己的 `cudaMemGetInfo` 不表示整卡空闲量。
- 64 MiB 可分页→锁页 `memcpy`，预热 3 次、正式 10 次：1 线程 **5.98 GiB/s**，2 线程 **6.00 GiB/s**（实测；`followup-copy-1.json`、`followup-copy-2.json`）。此前相同口径的独立进程结果为 6.13/5.87 GiB/s；两组都保留，不能据这点差距断言 2 线程稳定更快。
- 4×64 MiB 锁页中转环 + 2×64 MiB 设备槽/两个 CUDA stream 的双缓冲流水线：每项独立进程，预热 4 个 64 MiB tile，正式传 32 个 tile（2 GiB），计时包含可分页→锁页 `memcpy`、异步 H2D 和最后的 stream 同步；复用设备槽前同步上次传输。1 拷贝线程 **4.19 GiB/s**，2 线程 **3.86 GiB/s**；设备槽末尾抽样字节校验通过（实测；`followup-pipeline-1.json`、`followup-pipeline-2.json`）。这是合成的整块传输上限，尚未包含真实 KV 多 plane 打包和注意力调度。
- 服务就绪后单块 `cudaHostAlloc` 以 64 MiB 为粒度二分：**15488 MiB 成功，15552 MiB 返回 CUDA allocation error 2**；探针开始时可用物理内存 **19.40 GiB**，并保持至少 2 GiB 的物理内存安全余量。每次尝试后释放分配。这个相邻边界是本次状态和粒度下的实测值，不保证其他桌面负载下恒定（`followup-max-pin.json`）。
- `needle-128k-90` 原基线第 **9** 个 token（零起始步骤 **8**）为 **271**，导出 logits 的诊断运行选 **198**；诊断原始 BF16 值在这一步对 198/271 均为 **24.125**。两次额外的**未插桩、原 MTP-3 命令**连跑分别选 **198** 和 **271**，前 64 个 token 仅此步不同；`--no-cuda-graph` 未插桩对照与一次关闭 MTP 的单 token decode 对照均选 **271**（实测；`needle-128k-90.repeat-check.json` 及 `needle-128k-90.no-graph-check.json`）。`src/ops/kernel/argmax.cuh::argmax_better` 和 `src/ops/kernel/sampling_device.cuh::sampling_better` 对**真正相等**的 BF16 值都选较小 ID；MTP 验证先对各列调用 `ops::argmax`，再由 `speculative_accept_greedy_drafts` 许可 token。相同 MTP 形状的两次运行已经出现分歧，因此“验证批与单 token decode 的形状不同”不是分歧的必要条件。
- 为区分 tie-break 和数值变化，临时只在 MTP 当轮**已完成 `device.synchronize()` 和 token 选择后**读回第 8 步对应验证列的两个 BF16 原值及目标 argmax；不在当轮计算期间同步。在 **6 次**捕获中都选 271，ID 271 的值每次均高于 ID 198，差 **2–4 个 BF16 ulp**，与目标 argmax=271 一致（实测；`needle-128k-90.post-sync-capture.json`）。例如一次为 **24.25 对 23.875**，另一次为 **24.125 对 23.625**。因此选 271 的运行并非在完全相等的值上改变 tie-break；选 198 的诊断运行显示完全相等时按较小 ID 选择。**结论：取值规则固定为相等时选较小 ID；相同 MTP 形状的运行存在上游 BF16 logits 波动，形状差异不是必要解释。** 尚未定位具体上游算子，且未抓到选 198 的未插桩运行对应两个原值，不能声称两类运行的完整 logits 已逐元素比较。全部临时读回源码在提交前撤回。
- 临时读回代码撤回后重新构建成功；全量 CTest **96 项：92 通过、4 因缺少其他模型制品而跳过、0 失败**（实测，`ctest-followup-final.log`）。

## 阶段 3：分层 + 精确

- 状态：视图设计说明已写入 `stage3-view-design.zh-CN.md`；门禁 A 已改为同一次运行的影子注意力比对。dense 定位与门禁文档已在 `445dcfd0` 独立提交并 push；第 1 步归档组件在 `7eb5655a` 独立提交并 push，第 2 步传输引擎和测量已完成，完整构建/CTest 结果见下文。停在第 2 步审阅点；内核、视图状态机、目标运行时接线和产品 CLI 尚未实现。
- 量纲核对：补测文件 `followup-max-pin.json` 记载单块 `cudaHostAlloc` **15488 MiB 成功、15552 MiB 失败**。用户表述的「15.5 GiB」为近似说法；按二进制换算实际成功点是 **15.125 GiB**，文档统一写原始 MiB 与约 15.1 GiB。

### 阶段 3 前的 BF16 波动核对（2026-09-30；临时代码已撤回）

- 在 `needle-128k-90`、131008 输入 token、rk4v4-e8、MTP-3、`--no-cuda-graph` 下，临时在 prefill 的 LM head 后导出最后位置的完整 BF16 词表 logits（有效 248077 项）；两次原始 split 配置运行各生成 1 个 token，top-1 均为 ID 22，但有效项有 **244071/248077** 项逐位不同，相对 L2 为 **0.0936233**。因此 README 阶段 3 门禁 A 的「dense 两次逐位相同」前提在本次诊断路径上**不成立**；以后正式对照须先复核并在仍不成立时停止报告。
- 临时将 `w8_small_t.cu` 的 `launch_splitk_exact` 默认 `KSplits` 从 2 改为 1、`q5_small_t_mma.cu` 的 `launch_residual_exact` `kSplits` 从 2 改为 1；未改 dense 注意力内核或分派。用同一 needle、MTP-3 连跑三次各 64 token：**三次 token ID 序列完全一致**，且三次都正确回答 `73184269`；但 prefill 的完整 logits 三次哈希各异，三组两两有效项不同数为 **244955、244408、244345**，相对 L2 分别为 **0.130036、0.111502、0.098805**。每次有 17 组 MTP target logits 记录；三次对应的 17 组没有一组逐位相同。因此关闭这两处拆分**不足以消除 logits 波动**，本试验不能坐实唯一来源，也不能单独归因于 W8 或 Q5 原子累加。
- 诊断为导出 logits 在对应 CUDA stream 上同步，且关闭 CUDA Graph；这会改变执行时序，故不能把「三次 token 一致」外推为原始 Graph 路径的稳定性。原始 `needle-128k-90` 未插桩两次在第 9 个 token 的 198/271 分歧仍以阶段 0′ 补测为准。原始二进制日志、逐轮哈希和 token 序列放在仓库外 `D:\deeplearning\NInfer\logs\kvmem-stage0-baseline\stage3-atomic-probe\`（`baseline-summary.json`、`nosplit-summary.json`）；上述三处临时源码修改均已撤回。

### Dense 波动限时定位与影子门禁（2026-09-30，基于 `3ad5c6d5`）

定位在一个工作日的上限内得到结论；按用户授权仅诊断，不修 dense 内核或分派。以下均为 RTX 4090 同时驱动桌面时的实测，未施加额外桌面负载，所有诊断源码、临时参数和最小测试修改均已撤回。门禁 A 按用户决定改为同次运行的逐层影子注意力比对，门禁 B 不变；之前记录的「独立 dense 两次不相同则停止」是旧门禁，已被替换。

**最小矩阵与实际分块。** 不带 `--vision`，`--no-cuda-graph --greedy --no-thinking --kv-dtype rk4v4-e8 --prefill-chunk 1024 --max-context 32768 --kv-capacity 32768 --max-new 1`；MTP 关闭、MTP-3（另加 `--spec mtp --draft-tokens 3 --lm-head-draft`）各运行两次。提示词 JSON 固定且实测 token 数精确为下表长度。运行时会在 turn checkpoint 再拆一个 4-token 尾块，所以为落实「1024 单块、2048 两块」另外用临时环境变量禁用 checkpoint 分块；没有改变算子或其分派。

| 输入 token | 禁用 checkpoint 时真实边界 | MTP 关两次完整 BF16 logits | MTP-3 两次完整 BF16 logits |
|---|---|---|---|
| 512 | `[0,512)` | 逐位一致 | 逐位一致 |
| 1024 | `[0,1024)` | 逐位一致 | 逐位一致 |
| 2048 | `[0,1024), [1024,2048)` | 逐位一致 | 逐位一致 |
| 8192 | 8 个连续 1024-token 块 | 逐位一致 | 逐位一致 |
| 32768 | 32 个连续 1024-token 块 | 逐位一致 | 逐位一致 |

矩阵共 20 次运行；比较 prefill LM head 后最后位置的完整 248320 项 BF16 原始向量（其中 248077 项有效），上述每对都逐字节一致，两种 MTP 配置在各长度的哈希也相同。数据和每次完整命令、实际边界在仓库外 `dense-minimal-matrix-no-checkpoint/summary.json`。这不能外推成所有短尾块稳定。

**保留 checkpoint 的最小失败用例。** 同一 512-token JSON 实際拆为 `[0,508), [508,512)`；两次有效 logits 有 241404/248077 项不同。1024-token 用例拆为 `[0,1020), [1020,1024)`，有 242711 项不同（`dense-minimal-matrix/`）。不是一般的多块边界或 MTP 专属问题：关闭 MTP 也复现，而上表完整 1024 块的多块运行一致。

**第一处：Q5 小批量残差输出投影的求和顺序。** 对 512-token 最小用例逐块逐层导出隐藏输出、新 KV 页四个 plane、GDN recurrent/conv 状态哈希。第一块全部一致；第二块 embedding 输入一致，最早差异是第 0 层 GDN 输出投影之后。内部控制 h/g/beta、qkv/z、卷积输出、递归输出、归一化输出均一致，GDN0 的 recurrent/conv 状态也一致。`Variant::gdn_output_projection → ops::linear_add → q5_linear_add_split2_exact_launch → launch_residual_exact` 对 `[N=5120,K=6144,T=4]` 走 `Q5SmallTMmaResidualAtomicEpilogue`。固定输入和原始残差，每次先恢复残差，在同一 CUDA stream 上调用 32 次：两次独立试验各得到 **32/32 个不同哈希**。仅临时改此处 `kSplits=1`，两次试验的各 32 次输出变为同一个哈希，且两次间也一致。

两个 K 分片对已有 BF16 残差原子写回，分片到达次序决定 `round(round(residual+p0)+p1)` 或反序，浮点加法不满足结合律。此处是已证实的数值非确定性，不是丢写或未初始化；单分片不能消除整个模型的波动。Compute Sanitizer **13.0.85** 在同形状 Q5 的临时最小 CTest 上执行 racecheck/synccheck/initcheck，全部退出 0、各自 0 个错误，FP64 oracle 通过。归档文件为 `q5-t4-v13-*.log`；真实输入诊断为 `dense-layer-trace-512-v4/`（内部文件前半段还保留原始双运行，固定算子记录是之后追加的一组）、`dense-layer-trace-512-q5-nosplit/`。建议另行评估固定次序 FP32 分片归约、最终只做一次残差加法；不能把关闭拆分直接当作经过质量/性能验证的正式修复。

**第二处：量化 small-T 注意力共享内存竞争，属于正确性 bug。** Q5 单分片后，第 0–2 层一致，首个不同张量转到第 3 层（第一个全注意力层）注意力输出；其输入归一化、Q/gate/K/V、RoPE 后 Q/K 以及写入 KV 字节均一致（`dense-layer-trace-512-q5-nosplit-attn/`）。对真实 rk4v4-e8 的固定 Q/K/V、位置和 KV 内容连续调用注意力 32 次，两次进程分别出现 **2 和 4 个不同输出哈希**（`dense-layer-trace-512-gqa-fixed/`）。

Compute Sanitizer 13.0.85 在同一个 `gqa_attention_decode_i8_tiled_kernel` 的 Qwen27B `[QHeads=24,KVHeads=4,T=4,keys=512]` INT8 最小 oracle 用例上，racecheck 退出 **99**，报告 **4 类 error hazards、0 warnings**；synccheck/initcheck 的 CUDA 检查各为 0 errors，但目标程序 FP64 oracle 失败，退出 **1**。一个实际失败点为 index 473：输出 **0.00430298**，参考 **0.00231169**。不能把 synccheck/initcheck 的 0 errors 写成测试通过。

源码与 racecheck 行号共同定位了两处缺同步：

1. `src/ops/kernel/gqa_attention_decode_i8.cuh:406–409` 全线程清零 `q_i8/q_scale_tmp`，紧接着 `411–430` 其他 warp 量化并写回同一存储，中间没有 CTA barrier；报告清零与 `gqa_small_t_i8_store_swz`/scale 写回的 WAW。
2. `q_scale_tmp` 在第 115 行与 `p_s` 复用同一 shared buffer；producer warp 在 `459–460` 把 scale 读入寄存器后直接进入循环，在 `592–595` 写概率。别的 producer warp 可能还在读 scale，中间没有 CTA barrier；报告 scale 读取与概率写回的 RAW。

这属于内核内部同步缺失，不是合法的原子求和顺序波动。公共 INT8 实测复现；rk4v4-e8 的固定输入实測复现且使用同一段 Q 初始化/scale 暂存逻辑。源码影响范围为走该量化 small-T 变体的 INT8、packed/rotated/E8 类型，可能覆盖短 prefill 尾块、普通 decode 和 MTP 验证；其他 T/布局是否必现尚未穷举，BF16 路径不由这份证据覆盖。建议分别在清零与写入之间、全部 producer scale 读入寄存器与 shared buffer 复用之间加入 CTA 同步，再做数值、sanitizer、性能验证；**本轮未执行该修复，也未改变 dense 分派**。旧 CUDA12 检查的部分长 racecheck 被终止，结论以完成的 13.0.85 检查为准。工具包来自 NVIDIA 官方 redistributable，SHA-256 已校验，放在仓库外；没有新增项目依赖。

完整证据目录：`D:\deeplearning\NInfer\logs\kvmem-stage0-baseline`；文件 `gqa24-t4-v13-{racecheck,synccheck,initcheck}.log` 与对应 stdout。诊断在每个阶段读回并同步，会改变时序，所以不同哈希数量只是本次实测，不是日常请求发生概率。后续若要修 dense 正确性 bug，须另获用户授权；当前按用户指令继续阶段 3 第 1、2 步，不进入新注意力内核。

临时源码和最小 CTest 改动撤回后完整构建退出 0；全量 CTest **96 项，92 通过、4 因其他模型制品缺失跳过、0 失败**，耗时 73.46 s（`dense-diagnosis-restore-build.log`、`ctest-dense-diagnosis-restored.log`）。现有完整套件没有覆盖上述新缩出的 T=4、base=508 用例，套件通过不否定 sanitizer 已证实的 bug。

### 第 1 步：主机归档与回写（2026-09-30，基于 `445dcfd0`）

- 新增 `src/core/kvmem/host_kv_archive.{h,cpp}`。归档布局由真实 `PagedKVPool::page_bytes()` 推导，按层 → plane → 原始逻辑页排列；每个 plane 的 OS 页对齐和所有容量运算有溢出检查。`max_context` 可以大于设备视图的物理容量，只要求逻辑页表容量覆盖它。既有 pool 增加一个只读 `plane_order()` getter，未修改 dense 内核或分派。
- 加载期支持 auto/pinned/pageable 与实际路径日志。完整归档 + 4 GiB 的物理内存准入；auto 成功整块锁页且剩余物理内存 ≥4 GiB 才保留锁页，否则释放后回退；显式 pinned 失败退出。pageable 使用 Windows `VirtualAlloc` 只预留地址，写入时按 plane 提交，trim 时撤销完整 OS 页的提交。运行期不申请新的 CUDA 锁页缓冲。VirtualLock 的可选 CLI 接线仍属第 7 步。
- PageMajor 的连续物理页合并调用既有 copy 接口；HeadMajor 按单页形成规范字节，避免既有 bulk copy 的「head → 一段页」打包次序改变归档的「逻辑页 → head」。归档按完整量化页字节搬运，不重新量化。每层 D2H 回写和恢复 H2D 有加载期创建的 CUDA 事件；完成后才发布主机可读 frontier，覆盖回写前与 trim 前等待已有读写。trim 增加 generation、保持精确 token frontier、保留部分尾页，尾部不作为有效 key；不允许 trim 增长或回写产生逻辑空洞。
- 第 1 步的 copy 方法是整视图镜像的基础接口：pageable 尚直接调用既有 CUDA copy；第 2 步将增加显式四槽中转与独立传输线程。尚未在 NInfer 产品运行时启用 tiered 模式，完整视图的镜像由真实 PagedKVPool 测试验证；产品运行时和模式参数按既定第 4/7 步接入。
- 新 CTest **`test_host_kv_transfer`**：三种请求模式 × PageMajor/HeadMajor，共 6 组；每组两层、每层四个不同 page-byte plane，归档容量 **417792 B**，8 个逻辑页。独立生成逐字节 pattern，碎片物理映射 `{5,2,3,7}` → `{1,6,4,0}` 的全部 plane/page 往返和覆盖回写一致；frontier **250 → 130 → 0 → 12**，保留第 2 页前 2 token 的 frontier 语义，越界/空洞/无效物理页被拒绝。pageable 初始提交量和 trim 到 0 后提交量均为 0，中间 trim 能实际减少提交量；auto 在本次小归档上选择 pinned。4 GiB 准入边界和失败 pin 的策略另有纯函数断言，不宣称在本测试人为制造了实际大块 pin 失败。
- 新增延迟 H2D 的 trim 用例：先观察到失败 `trim must drain outstanding archive H2D reads`，再增加读取完成事件保护，6 组全部通过。初次测试构建在新接口未实现处失败；实现后基础及延迟用例均通过。外部数据为 `host-kv-red-build.log`、`host-kv-trim-red-build.log`、`stage3-step1-byte-test.log`。
- 全量构建退出 **0**；全量 CTest **97 项：93 通过、4 因缺少其他模型制品跳过、0 失败**，耗时 **66.85 s**。新测试在完整套件中耗时 1.53 s（与其他测试并行）；另一次单独运行退出 0。记录：`stage3-step1-build.log`、`ctest-stage3-step1.log`。数值全部为实测或明确的测试配置，未提供未经测量的推理性能结论。

### 第 2 步：传输引擎与真实多 plane 吞吐（2026-09-30，基于 `7eb5655a`）

- 新增 `src/core/kvmem/host_kv_transfer.{h,cpp}`。加载期固定单工作线程、一个非阻塞 CUDA staging stream、每个 ticket 的 ready/consumed 事件、每层生产/回写事件；pageable 固定 **4×64 MiB** cacheable 锁页环，pinned **0 B** 环且 H2D/D2H 直接用归档。运行期不申请新的锁页、设备存储或 CUDA 事件。普通 CPU 任务描述/队列可分配，不参与 CUDA Graph 地址。归档与调用方 staging、生产/消费 stream 必须比引擎活得久；C=1 调度入口由调用方串行调用。
- `plan_host_kv_staging()` 从真实归档 plane 字节求最大单层流式量，再加 64 MiB；容量覆盖低于最小值会报错。16 个 Main 全注意力层、INT8、262K 逻辑容量、128K 驻留预算时，流式层 **264 MiB**、staging **328 MiB**。设备 span 由外部 workspace 提供，引擎不自行申请 staging；与 `build_workspace_plan()` 的统一求解/扣减视图、GDN 提前调度、ViewTable 的 DeviceOnly/Both/HostOnly 状态在后续第 4 步接线，不修改现有 dense 预算。
- `prefetch(layer,plane,first,count,device_offset)` 保留原始逻辑号，按最多 64 MiB 的连续 plane 段排入 FIFO；跨层可以提前排队。CPU 等到 worker 记录 ready 后才向消费流提交 `cudaStreamWaitEvent`，不在 CPU 等 DMA 完成；`release` 在实际消费后记录 consumed，覆盖 live range 会被拒绝，重用已释放范围必须等待对应 consumed。事件槽引用计数防止依赖尚未被传输流捕获时重录事件。访问顺序不由传输完成时间决定，split/LSE 顺序仍留给第 3 步新内核。
- 引擎对归档取得唯一传输所有权：直接归档写入/恢复/trim 被拒绝，须由 owner 排序。异步回写通过生产流事件等待 KV 写完；同一传输流保证此前读取先于归档覆盖。pageable D2H 经固定环、CPU memcpy 回归档，pinned 直接 D2H；工作线程只写字节并兑现 completion future，主线程读取该层时才发布完成 frontier，避免元数据数据竞争。`trim` 先 drain 队列、DMA 和消费事件，重置 ticket，再截断精确 frontier、增加 generation。snapshot/restore 运行时尚未接线，不声称已实现完整恢复。
- 新 CTest **`test_host_kv_transfer_engine`**：pinned/pageable × PageMajor/HeadMajor 四组，每组两层、四种 plane；独立字节 pattern 校验全部页。覆盖四槽/事件反复复用（48 次）、其他层提前排队、延迟消费者 50 ms 后跨流覆盖、异步回写、延迟生产者 50 ms 后回写、归档覆盖前已排队读取的旧字节、过小预算/live 重叠/旧 generation 拒绝与 trim 到 **130**。真实 INT8 几何另断言 **264/328 MiB**。先写测试时新接口未实现，构建失败（`host-transfer-red-build.log`）；实现后的独立基础测试和完整套件中的扩展测试均通过。

**真实多 plane 手动微基准。** 新增 `bench/host_kv_transfer_bench.cpp`，启用 `NINFER_BUILD_BENCHMARKS=ON` 后构建目标 `ninfer_host_kv_transfer_bench`，分别执行 `pinned`、`pageable`。真实 INT8 几何：64 token/page，K/V 各 `I8 [256,4 heads]`，K/V scale 各 `FP16 [4,4 heads]`；单层每页 135168 B，16 层、4096 页归档 **8858370048 B = 8.25 GiB**。先从真实 PagedKVPool 建立完整主机镜像，然后释放整个设备源；计时期间只从归档读。字节为确定性的 layer/plane pattern，不是模型实际生成的 KV 值，不影响传输布局。

每个正式层读取后半 **2048 页（128K token）**：K/V 各 128 MiB、两个 scale 各 4 MiB，分 6 个 ticket 连续放入 **328 MiB** staging 的前 **264 MiB**。预热 2 层，正式 16 层覆盖各层归档一次，共 **4429185024 B = 4.125 GiB**。`steady_clock` 计时包括 CPU 入队、pageable memcpy、异步 H2D、GPU 事件等待和每层最终同步；不含归档初始化、归档写回、末尾 D2H 校验。正式循环后独立比对最后一层全部 **276824064 B**；每次均通过。前面各 layer/page 的内容正确性由上述 CTest 覆盖。本基准没有 GDN/注意力/LSE，因此未实测计算与跨层传输的重叠收益，也不代表模型推理吞吐。

| 模式 | 第 1 次 GiB/s（耗时 s） | 第 2 次 GiB/s（耗时 s） | 两次平均 GiB/s | 锁页环 |
|---|---|---|---|---|
| pinned | **8.7644（0.4706542）** | **7.6075（0.5422307）** | **8.1859** | 0 |
| pageable | **2.7551（1.4972063）** | **3.2208（1.2807308）** | **2.9880** | 256 MiB |

四次独立进程按 pinned/pageable 交替运行。测量前整卡占用 **1930 MiB**、后 **1934 MiB**，PCIe **3.0×16**；构造归档前可用物理内存约 **17.62–17.88 GiB**，引擎就绪时 pinned 约 **9.29–9.32 GiB**、pageable 约 **9.09–9.10 GiB**。这是无模型服务加载的传输微基准状态，不替代阶段 0′ 的「262K 模型就绪后」锁页上限测量。可分页数字低于阶段 0′ 整块基准 4.19 GiB/s：这次有多 plane、4.125 GiB 读取足迹、任务/消费事件和每层同步，测试口径不同，尚未隔离各因素的贡献，不据此断言唯一原因。D4 仍按用户决定保留 auto 优先完整锁页。

数据全部在仓库外 `D:\deeplearning\NInfer\logs\kvmem-stage0-baseline`：`stage3-transfer-{pinned,pageable}-{1,2}.{json,log}`、`stage3-transfer-summary.json`（命令、基准/源码 SHA-256、基线提交）、`stage3-transfer-gpu-{before,after}.csv`。复现实验使用 CMake **3.31.10**、CUDA **13.3**、MSVC；系统 VS 自带 CMake 3.25.1 不能配置本项目，实际使用研究目录中现有的 CMake。

最终完整构建退出 **0**；**构建结束后**运行全量 CTest，**98 项：94 通过、4 缺少其他模型制品跳过、0 失败**，耗时 **63.30 s**（`stage3-step2-build.log`、`ctest-stage3-step2.log`）。一次误提前启动测试造成 Windows 正在运行的 exe 被锁、链接 LNK1104 与测试 BAD_COMMAND；保留 `stage3-step2-build-interrupted.log` / `ctest-stage3-step2-premature.log`，不把那轮作为验证结果。无新增 dense 内核/分派修改；第 2 步提交后停下等待审阅，不实现第 3 步。

### small-T INT8 同步修复（2026-10-01）

用户授权仅在 `gqa_attention_decode_i8.cuh` 加两处 CTA barrier；Q5 残差原子累加的合法求和顺序波动暂不修改。修复分支从 `221290ba` 切出，提交 **`81869152`** 已 push 到 `fix/i8-small-t-attn-sync`；以 **`b68009f7`** 合并并 push 到 `feat/kvmem`。没有合并其他分支。完整源码核对、测试判据和可复现命令见 `docs/maintainer/i8-small-t-attn-sync.md`。

- ordinary INT8、packed/rotated K/V、E8 lattice/root 都经同一个量化 small-T kernel，两个 barrier 对所有非空 CTA 无条件执行。审查 BF16 small-T、INT8/BF16 prefill 的清零、异步复制和 shared 复用，没有找到相同的缺 barrier 写法；这是源码审查，不是所有配置都经过 sanitizer 的结论，未修改其他 dense 内核或分派。
- 新 CTest `ninfer_gqa_small_t_sync_test`：Qwen27B 24Q/4KV、T=1/2/4、keys=512/8199（后者跨 8198 split 策略边界）、INT8/rk4v4-e8，共 12 组。独立 FP64 dot/softmax/V oracle，并对原始 BF16 输出逐位比较连续 **64** 次。未修复版 rk4v4-e8 8199-key 的三种 T 在第二次调用出现差异；修复后全部通过。固定数值门槛未放宽；实测相对 L2：INT8 **0.00279735–0.00285466**，rk4v4-e8 **0.00301066–0.00319963**。
- Compute-sanitizer **13.0.85**：原版完整形状矩阵 racecheck 返回 **99**、**64 displayed hazards / 64 errors**；修复后 racecheck、synccheck、initcheck 三进程均返回 **0**，racecheck **0 hazard / 0 error / 0 warning**，后两项 **0 error**，同进程 FP64/重复检查均通过。instrumentation 为每组 2 次；正常 CTest 的 64 次仍必做。
- 修复分支完整构建返回 0；全量 CTest **97 项：93 通过、4 缺少其他模型制品跳过、0 失败，62.58 s**。日志与测量全部在仓库外新目录 **`D:\deeplearning\NInfer\logs\kvmem-stage3-attention`**，旧阶段 0′ 数据保留。

修复前后各独立进程一次，固定原 stage-0 synthetic JSON、rk4v4-e8、64 greedy token、默认 CUDA Graph、不带 vision；MTP 使用 `--spec mtp --draft-tokens 3 --lm-head-draft`，其余配置相同：

| 上下文 / 模式 | 修复前 tok/s | 修复后 tok/s | 差值 |
|---|---:|---:|---:|
| 32768 decode | 48.54 | 48.61 | +0.144% |
| 32768 MTP-3 | 154.19 | 153.87 | -0.208% |
| 131072 decode | 44.02 | 44.00 | -0.045% |
| 131072 MTP-3 | 137.63 | 137.55 | -0.058% |
| 262144 decode | 39.17 | 39.16 | -0.026% |
| 262144 MTP-3 | 120.05 | 112.71 | -6.114% |

262K MTP 的 64 token ID 相同，但接受率从 **100%（47/47）降为 93.88%（46/49）**，验证轮数 **16→17**。以 CLI 的 63 decode token 和速度折算，每轮约 **32.80→32.88 ms（+0.25%）**；多一轮解释了几乎全部速度差，单样本不能分离剩余 Q5 波动与修复后的注意力值变化。32K/128K 接受率前后均 100%。Windows 桌面挂在 4090 上，无人为额外负载，测前占用原系列 **1558 MiB**、修复后 **1156 MiB**。此为单次回归核对，不能当作统计性能估计。

证据：`sync-before-regression.log`、`sync-before-racecheck.{log,stdout.txt}`、`sync-after-regression.log`、`sync-after-{racecheck,synccheck,initcheck}.{log,stdout.txt}`、`sync-full-build.log`、`ctest-sync-fix.log`、`before/`、`after/`、`performance-comparison.json`。JSON 保存完整命令、提交号、exe/kernel/prompt SHA-256 和 GPU 前后状态。临时未修复 exe 也保存在外部。sanitizer 复现命令（从 repo 执行，runtime DLL 目录加入 PATH）：

```powershell
$san = 'D:\deeplearning\NInfer\logs\kvmem-stage0-baseline\sanitizer-tools\cuda_sanitizer_api-windows-x86_64-13.0.85-archive\compute-sanitizer\compute-sanitizer.exe'
$data = 'D:\deeplearning\NInfer\logs\kvmem-stage3-attention'
$env:NINFER_OP_REPORT_STATS = '1'
foreach ($check in @('racecheck', 'synccheck', 'initcheck')) {
    & $san --tool $check --error-exitcode 99 `
        --log-file "$data\sync-after-$check.log" `
        '.\build-vision-integration\tests\ninfer_gqa_small_t_sync_test.exe' `
        --sanitizer *> "$data\sync-after-$check.stdout.txt"
    if ($LASTEXITCODE -ne 0) { throw "$check failed: $LASTEXITCODE" }
}
```

**合并后的九组未插桩 dense 金标准（`b68009f7`）。** 原 JSON SHA-256 和全部参数保持一致，各自独立进程，均无前缀复用、生成 64 token；九例的全部 token ID 均与旧 `78989fc0` 基线相同，**0 个用例发生变化**，六例 needle 全部答对，MTP 接受率全部 100%。旧文件不覆盖，新记录在上述外部目录的 `baseline/`。

| 用例 | 文本 prefill | 就绪后 cudaMemGetInfo 空闲 | MTP tok/s | token 变化 |
|---|---:|---:|---:|---:|
| synthetic-32k | 14.428 s | 4.50 GiB | 154.07 | 0 |
| needle-32k-10 | 14.464 s | 4.50 GiB | 154.10 | 0 |
| needle-32k-90 | 14.476 s | 4.50 GiB | 153.94 | 0 |
| synthetic-128k | 77.491 s | 2.81 GiB | 137.60 | 0 |
| needle-128k-10 | 78.049 s | 2.81 GiB | 137.60 | 0 |
| needle-128k-90 | 78.479 s | 2.81 GiB | 137.39 | 0 |
| synthetic-262k | 211.808 s | 560.72 MiB | 120.20 | 0 |
| needle-262k-10 | 212.959 s | 560.72 MiB | 120.19 | 0 |
| needle-262k-90 | 213.269 s | 553.67 MiB | 119.69 | 0 |

32K/128K/262K 的真实输入仍为 32704/131008/262080。RTX 4090 同时驱动桌面；测前占用约 1127–1137 MiB，未主动增加负载。CPU 视觉参数开启，但提示词全部为纯文本。修复内核在这轮 262K 合成用例获得 100% 接受率和 120.20 tok/s，先前另一轮 93.88%/112.71 tok/s 不构成稳定性能下降的证据。单次采样不保证未来所有 token 逐位一致，Q5 顺序波动仍保留。

**512-token、保留 checkpoint 分块的双运行。** 不带视觉、关闭 MTP，`--no-cuda-graph --greedy --no-thinking --kv-dtype rk4v4-e8 --prefill-chunk 1024 --max-context 32768 --kv-capacity 32768 --max-new 1`。使用旧矩阵同一份固定 `prompt-512.json`；两次实测均为 512 token，真实边界 **`[0,508), [508,512)`**，没有禁用 checkpoint。最后位置导出 248320 个 BF16 原值，有效 248077 项。两次**仍不逐位相同**：有效项 **246134/248077** 不同，计入物理词表尾部共 **246377/248320** 不同，有效向量相对 L2 **0.1248612403**，top-1 均为 **ID 16**。这是剩余模型波动的实测，不能因注意力独立重复测试稳定就宣称整个 dense 模型稳定，也不能仅凭两次输出把全部差异定量归因于 Q5；本轮按决定不再修 dense 的其他代码。原始二进制、哈希、真实分块、完整命令及双运行结果在 `checkpoint-512/summary.json` 和 `.logits/.chunks/.stdout.txt/.stderr.txt`。

**每步 logits 诊断重录也已完成。** 九例各 64 步的 top-8 与完整 BF16 target logits 均保存在新目录 `diagnostic/`（`*.top8.json`、`*.target-logits.bf16.bin`、`*.accept.bin`）；**9/9** 的诊断输出 token 与该次未插桩基线完全一致，每步 top-1 与诊断实际生成 ID 一致，六例 needle 也均答对。与旧数据相同，这些 `logits_fp32` 是 BF16 原值转成 FP32 表示，不是舍入前的 FP32。诊断禁用 CUDA Graph、增加读回同步，耗时不作为性能数据，也不能从本次一致推断 Q5 已确定性。脚本、SHA-256、命令和摘要在外部目录；所有临时 target 读回/分块钩子均已撤回，未改 Q5 或其他 dense 内核。传输修正与新部分注意力尚未完成。

### 第 2 步审阅修正：回写与预取独立推进（2026-10-01，基于 `1fff2bb6`）

- 新回归先在旧引擎上运行：第 0 层生产者延迟 `sleep_for(50ms)`，回写后马上排第 1 层预取。旧引擎 **退出 1**：预取下发 **62.7265 ms**、完成 **62.7946 ms**，回写完成 **62.5122 ms**，复现同 worker/同流的队头阻塞（`transfer-sync-red-test.log`）。Windows 调度使实际延迟约 62 ms，不把 sleep 请求值当作实际完成时间。
- 两条非阻塞 CUDA 流、最多 **2 个**主机工作线程：预取 worker/FIFO 只提交 H2D；D2H 回写与其完成事件等待在第二个 worker，pageable 的环→归档 memcpy 也在第二个 worker。pinned 直接 D2H 后事件完成再发布 future。仍是 pageable **4×64 MiB** cacheable 环、pinned **0 B** 环；没有运行期锁页/设备/事件分配。
- 两 worker 通过独占 lease 共用四槽，每槽带 DMA 事件，异常时先 drain 再释放以免另一 worker 改写尚在传输的字节。回写捕获此前最新 H2D ticket 的 ready 事件（先等其确已 record，再 CUDA wait），H2D FIFO 保护所有此前读归档的 DMA；引用计数保护 ready generation 不被重录，覆盖 ticket 已复用的情形。后续的无关层预取没有回写/生产者依赖。同层读取新归档仍须等待对应回写 future；这是数据依赖，未绕过它。reset/trim/destructor 排空两个 worker 与两条流。
- 新旧逐字节检查全部通过，四种 mode/order；新增测试覆盖 50 ms 延迟回写与后续无关预取、延迟消费者之后读取旧归档再覆盖、ticket 复用后的旧读保护。测试额外的时间 observer thread 只用于测试记录，不是生产引擎的第三个 worker。独立绿色运行退出 0，时间均为 `steady_clock` 从提交前开始的毫秒；预取下发取 worker 已记录 ready 的观察时刻，完成含字节读回同步，回写完成由 future observer 记录：

| 归档 / 源布局 | 预取下发 ms | 预取完成 ms | 回写完成 ms |
|---|---:|---:|---:|
| pinned / PageMajor | 0.2086 | 0.2364 | 62.2248 |
| pinned / HeadMajor | 0.1506 | 0.1859 | 61.3671 |
| pageable / PageMajor | 0.1464 | 0.1776 | 62.3957 |
| pageable / HeadMajor | 0.1283 | 0.1509 | 62.4526 |

源池及生产流必须活到 completion future 完成，源 KV 字节在此之前不改动，已写入 API 注释。原有归档往返、异步回写、事件多次复用、live range 与旧 generation 拒绝、trim 语义保留。证据在 `D:\deeplearning\NInfer\logs\kvmem-stage3-attention` 的 `transfer-sync-{red,green}-{build,test}.log`；完整构建退出 **0**（`transfer-sync-full-build.log`）。构建完成后全量 CTest **99 项：95 通过、4 缺少其他模型制品跳过、0 失败，62.87 s**（`ctest-transfer-sync.log`）。没有修改 dense 内核/分派或接入 tiered 运行时。

## 阶段 4：稀疏 decode

- 状态：未开始

## 与设计的偏差

| 日期 | 决策编号 | 偏差 | 原因与证据 | 用户是否同意 |
|---|---|---|---|---|
| 2026-09-30 | D3 前提 | 后续分层视图保留原始逻辑页号，不直接以紧凑视图槽号代替位置；阶段 0′ 不实现 | `paged_kv_address.cuh:16` 和 prefill/decode 内核按 `position >> 6` 查块表并以原始 key 位置做掩码，详见上文静态核对 | 是，用户明确选择“保留原始页号” |
| 2026-09-30 | D4 | 默认 `auto`：加载期整块 CUDA 锁页且剩余物理内存 ≥4 GiB 则归档直传，否则释放部分分配后退回可分页归档与 4×64 MiB 中转环；显式 `pinned` 失败即退出，`pageable` 保留旧路径。两者统一「层 → plane → 逻辑页」布局 | 262K 就绪后整块可锁页 15488 MiB；锁页 H2D 9.83 GiB/s，可分页经环端到端 4.19 GiB/s。旧「4090 WDDM 锁页限额更紧」的推断不成立，4060 仍须实测 | 是，用户明确同意修订 |
| 2026-10-01 | D1，用户授权例外 | 仅修复已知的 dense 量化 small-T 注意力共享内存竞争，独立修复分支合入后重录九组金标准，旧数据保留；Q5 和 dense 分派不改 | 原版 racecheck 64 hazards，修复后 0；12 组各 64 次逐位一致且 FP64 oracle 通过，详见修复记录 | 是，本轮用户明确要求现在修复，并限定文件与两处 barrier |
