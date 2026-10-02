# KVMem 4090 实施进度

记录方式见 [README](README.zh-CN.md) 第 7 节。每条记录写明日期、提交号，并注明数字是实测还是估算。

## 待用户决定的问题

第 4 步已审阅通过（提交 `6466a166`、`d33157a9`、`f5da0049` 已推送）。本轮按用户授权完成阶段 3 第 5、6 步，停在第 6 步审阅点，不开始第 7 步。D11 的 sink 语义已由用户再次确认：固定 MTP 窗口包含 sink，剩余环容量留给最近页；没有修订为仅近期页。第 5 步独立提交/push `4da637a4`；第 6 步源码、规格/质量审阅、128K/262K 正式门禁、最终全量构建/CTest 和 sanitizer 均通过，随本步独立提交发布。本轮没有尚待决定的问题。

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

**每步 logits 诊断重录也已完成。** 九例各 64 步的 top-8 与完整 BF16 target logits 均保存在新目录 `diagnostic/`（`*.top8.json`、`*.target-logits.bf16.bin`、`*.accept.bin`）；**9/9** 的诊断输出 token 与该次未插桩基线完全一致，每步 top-1 与诊断实际生成 ID 一致，六例 needle 也均答对。与旧数据相同，这些 `logits_fp32` 是 BF16 原值转成 FP32 表示，不是舍入前的 FP32。诊断禁用 CUDA Graph、增加读回同步，耗时不作为性能数据，也不能从本次一致推断 Q5 已确定性。脚本、SHA-256、命令和摘要在外部目录；所有临时 target 读回/分块钩子均已撤回，未改 Q5 或其他 dense 内核。该次基线提交时传输修正与新部分注意力尚未完成，后续结果见下面各步。

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

### 第 3 步：部分注意力与固定顺序 LSE（2026-10-01，基于 `fdfe3217`）

**实现范围。** 新增独立 `gqa_attention_partial.{h,cpp,cu}` API/launcher、BF16 与量化 prefill/small-T 内核、LSE merge/finalize；原 dense API、分派、内核和 Q5 均不改。覆盖 C=1、24Q/4KV、BF16 / INT8-G64 / rk4v4-e8。每次只读明确发布的 `{logical_page, physical_page}` 列表，物理 ID 分别指向 resident 与 staging plane，位置始终由原始逻辑页号求出。`attention_access_prefix()` 在执行边界验证排序、frontier、物理容量；设备 split 按实际可见 key 前缀等分，与传输到达顺序无关。空列表、空 split、因果掩码全部为空时输出 `O=0,m=-inf,l=0`。

- `O` 为未归一化 FP32 分子，`m` 为自然对数域的最大 score，`l` 为 FP32 分母；每个 split 独占输出。固定升序合并、不用原子操作，支持多个 pass 的 FP32 状态继续合并，最后才 BF16 finalize。rk4v4-e8 在 FP32 部分结果上做 V 的 H64 逆旋转。
- Q 的量化由一个 warp 独占完整 `(row,group)`，含 padding；scale 和概率有独立共享存储，不复用旧竞争写法。量化写入和首次读取之间有 CTA barrier。BF16 Q/K 的共享存储在 fragment 读取完成后有 CTA barrier 才复用。两种 small-T tile 支持尾块，prefill 为 10 token / 60 GQA 行的 tile；支持范围和 CUDA grid 上限写在新 API 注释里。
- 输入 prefix 必须原样来自 host helper：每项是整页，只有 frontier 的末页可能少于 64 key。因此 `ordinal/64` 可以 O(1) 查**列表条目**，再读条目的原始逻辑页号求绝对位置；不把紧凑列表序号当成位置。仍由 prefix 和 query frontier 算有效 key 数及 split；不接受任意截短的中间页。

**独立数值验证。** 新 CTest `test_attention_partial_lse_merge` 有三种格式各 12 个注意力用例，共 **36 例**；原 `ninfer_gqa_attention_test` 另加六个部分注意力 oracle 用例。FP64 oracle 从真实 BF16/量化缓存字节解码，不调用 GPU attention/encoder 求参考。覆盖第 **3000 页（192000）**、页间空洞、打散的物理映射、不同 K/V/page/head/group/offset scale、末页 13 key、跨 32/64-key tile 与 split、300 split 中的空 split、T=1/2/4/9 decode、T=16/17 prefill、混合/全驻留/全 staging、空列表和全未来页。输出先 NaN poison，检查所有部分状态已写且有限（仅中性 m 为 -inf），检查缓存字节未改变；同一输入与划分各跑两次，**部分 O/m/l 与最终 BF16 输出逐位一致**。独立 LSE FP64 用例另覆盖 ±75 的最大值、NaN 中性载荷、全部中性行、两级合并、别名拒绝及坏列表校验。

| 格式 | 对理想 FP64 oracle 的最大相对 L2（含 BF16 finalize） | 三个 streamed pass 对一次计算的最大相对 L2（归一化 FP32） |
|---|---:|---:|
| BF16 | 0.00166783337 | 2.32558334e-6 |
| INT8 | 0.00187258759 | 4.34224891e-5 |
| rk4v4-e8 | 0.00169578193 | 3.45517982e-5 |

注意力 oracle 判据在首次测试前固定为相对 L2 `1/256`、gross `1.1e-3 + 3.9e-3×max|ref|`；多 pass 对一次计算的判据独立为相对 L2 **1e-3**、gross `1e-4 + 1e-3×max|ref|`。初版 BF16 PV 概率暂存造成 decode 多 pass 相对差 **0.00157320683**，未通过后没有放宽判据：新 BF16 PV 额外保留概率的 BF16 低位残差，做第二次 MMA；quantized prefill 仍用 native FP16 PV。改后差异通过，属于新算子的精度改善，**不是改 Q5 或 dense**。初版失败日志 `partial-expanded-test.log` 保留。算子 oracle、跨 pass 的误差与 README 的整模型门禁 A/B 口径不同；影子验证和模型三次包络测量须在第 4 步接线后执行，本轮不宣称已通过。

**264 MiB 计算微基准。** `ninfer_attention_partial_bench` 使用真实 PageMajor INT8 K/V/FP16-scale 四 plane，**131072 key、276824064 B**，全部位于 staging，resident pool 为空；dense 使用同一份字节、同一份 Q、同一缓存位置和 key 数。不计 H2D、归档、模型层或 GDN；CUDA event、eager、5 次预热/20 次计时，RTX 4090 同时驱动桌面。手动扫描 split=32/64/128/256，完整曲线见外部 `partial-direct-page-bench.log`；下表每个 T 取本轮测得的最低 total，选择不依赖传输完成顺序：

| T / 路径 | split | dense ms | partial kernel ms | 独立 merge+finalize ms | 新路径整体 ms | 新/dense |
|---|---:|---:|---:|---:|---:|---:|
| 1 / decode | 64 | 0.328894 | 0.311808 | 0.0315216 | 0.327571 | 0.995977 |
| 4 / decode | 32 | 0.342589 | 0.384816 | 0.0198576 | 0.394635 | 1.15192 |
| 64 / prefill | 32 | 7.19003 | 2.38846 | 0.0273856 | 2.47229 | 0.343851 |
| 1024 / prefill | 32 | 21.7745 | 35.9727 | 1.00157 | 36.8224 | 1.69108 |

单独计时与整体计时是不同批次，受时钟和 eager 下发影响，不能要求三个数严格相加。T=64 的 dense grid 仅 24 CTA，split 增加并行度；T=1024 下新路径**仍慢 69.1%**，不能用小 T 的结果代表正常 prefill。原逐 key 二分查列表的版本测得 47.9529 vs dense 24.9162 ms（1.92457 倍）；O(1) 查条目后的相对差改善，但两批 GPU 时钟/桌面状态不同，不能将绝对时间全部归因于代码优化。量化 PV 的共享加载、同步和大量 FP32 partial 写出仍是后续优化候选，尚未用 profiler 定量归因。

**接线前需审阅的资源与性能限制。** T=1024、32 split 时 partial O/m/l 为 **774 MiB**，单份 merge O/m/l 再需 **24.1875 MiB**，不含 Q/output、staging 和驻留 KV；256 split 的 O 单 plane 达 **6 GiB**。第 4 步必须在加载期预算、按预算选择固定的 query 子块和 split，不能直接把微基准参数放进真实 24 GiB 模型运行时。当前仍是功能验证版本；prefill 性能/partial 容量是明确的审阅项，不涉及 D1–D15 改动，也没有以这些数字预测整模型速度。

**最终验证及证据。** `partial-direct-page-test.log` / `partial-numerical-summary.json` 保存最终数值；compute-sanitizer **13.0.85** 对最终 36 例与 merge 测试跑 `racecheck`、`synccheck`、`initcheck`，全部退出 **0**：racecheck **0 hazards / 0 errors / 0 warnings**，另两项各 **0 errors**。新 bench 独立构建退出 **0**；恢复原 `NINFER_BUILD_BENCHMARKS=OFF` 的 product 配置后，完整构建退出 **0**（`partial-full-build.log`）。构建结束才执行全量 CTest：**100 项，96 通过、4 缺少其他模型制品跳过、0 失败，62.56 s**（`ctest-partial.log`）；新 CTest 为 1.46 s。全部检查完成后按授权单独提交第 3 步并推送 `feat/kvmem`；到此停下等审阅，不实现视图状态机或运行时接线。所有初版失败、构建、微基准和 sanitizer 日志留在仓库外 `D:\deeplearning\NInfer\logs\kvmem-stage3-attention`，未提交临时代码、模型或数据。

复现命令（repo 下，先加载 VsDevCmd 的 x64 环境；使用 CMake 3.31.10）：

```powershell
$cmake = 'D:\VSCodeProjects\ninfer-4090-research\toolchain\python-packages\cmake\data\bin\cmake.exe'
$ctest = 'D:\VSCodeProjects\ninfer-4090-research\toolchain\python-packages\cmake\data\bin\ctest.exe'
$data = 'D:\deeplearning\NInfer\logs\kvmem-stage3-attention'
$env:PATH = 'D:\deeplearning\NInfer\runtime\ninfer-rtx4090-windows-x64-vision-modes-v3;' + $env:PATH
$env:NINFER_OP_REPORT_STATS = '1'
& $cmake -S . -B build-vision-integration -DNINFER_BUILD_BENCHMARKS=ON
& $cmake --build build-vision-integration --target test_attention_partial_lse_merge ninfer_gqa_attention_test ninfer_attention_partial_bench --parallel 4
& '.\build-vision-integration\tests\test_attention_partial_lse_merge.exe'
& '.\build-vision-integration\bench\ninfer_attention_partial_bench.exe' *> "$data\partial-direct-page-bench.log"
$san = 'D:\deeplearning\NInfer\logs\kvmem-stage0-baseline\sanitizer-tools\cuda_sanitizer_api-windows-x86_64-13.0.85-archive\compute-sanitizer\compute-sanitizer.exe'
foreach ($check in @('racecheck', 'synccheck', 'initcheck')) {
    & $san --tool $check --error-exitcode 99 --log-file "$data\partial-direct-page-$check.log" `
        '.\build-vision-integration\tests\test_attention_partial_lse_merge.exe' `
        *> "$data\partial-direct-page-$check.stdout.txt"
    if ($LASTEXITCODE -ne 0) { throw "$check failed: $LASTEXITCODE" }
}
# 单独构建/测量新 bench 后恢复原 product 配置；先构建结束，再跑测试。
& $cmake -S . -B build-vision-integration -DNINFER_BUILD_BENCHMARKS=OFF
& $cmake --build build-vision-integration --parallel 4
$env:NINFER_QWEN3_8_27B_WEIGHTS = 'D:\deeplearning\NInfer\models\qwen3_8_27b.ninfer'
$env:NINFER_TEST_VISION_GGUF = 'D:\deeplearning\kvmem-v0.16.0-rc3-windows-x86_64-cuda13.2.86\models\Qwen3.8-27B\Qwen3.8-27B\mmproj-BF16.gguf'
& $ctest --test-dir build-vision-integration --output-on-failure -j 4
```

### 审阅后：同步修复合回指定分支（2026-10-01）

按本轮授权只更新默认分支、CPU 视觉分支及后续工作的 `feat/kvmem`。`81869152` 无冲突 cherry-pick 到 `feat/rtx-4090-sm89-native`，新提交 **`c3d228ea`**；`fix/i8-small-t-attn-sync` 用 `--no-ff` 合并进 `feat/vision-cpu-ggml`，合并提交 **`333f68d0`**。两条分支均在各自完整构建及全量 CTest 通过后 push，其他分支未更新。

| 分支 / 提交 | CTest 总数 | 通过 | 跳过 | 失败 | 全量测试用时 |
|---|---:|---:|---:|---:|---:|
| `feat/rtx-4090-sm89-native` / `c3d228ea` | 85 | 81 | 4 | 0 | 44.73 s |
| `feat/vision-cpu-ggml` / `333f68d0` | 97 | 93 | 4 | 0 | 61.63 s |

四项跳过仍是缺少其他真实模型制品的 prefix / 35B / DFlash 测试，不作为已通过。主线首次构建遇到旧版 XGrammar UTF-8 字符串被 MSVC 按 CP936 解析的问题，仅在本地 CMake cache 的 `CMAKE_CXX_FLAGS` 增加 `/utf-8` 后完整重建，未改源码或提交构建配置。视觉分支的私有 GGML 补丁哈希因 Git checkout 的 CRLF 与原缓存 LF 不同而被拒绝；仅将本地补丁脚本恢复为缓存对应的 LF 字节（SHA-256 `8319b456b2b4ec75818eb2507bd056c07482a7f0ed95e2c8a36393ddd09c01e2`），Git 没有内容差异，再完整构建通过。构建与测试顺序执行，未并发重链接测试二进制。

证据保存在仓库外 **`D:\deeplearning\NInfer\logs\kvmem-stage3-optimization`**：`main-sync-build.log`（首次失败）、`main-sync-build-utf8.log`（完整构建成功）、`main-sync-ctest.log`；`vision-sync-build.log`（缓存校验失败）、`vision-sync-build-lf.log`（完整构建成功）、`vision-sync-ctest.log`。两个分支上的生产 dense 变更均仅为原定的两处 CTA 同步。

### 波动来源收尾：Q5 residual 单 split（2026-10-01，临时代码已撤回）

正常修复版 CLI 构建完成后，从 **02:49:26 UTC** 开始仅临时将 `launch_residual_exact` 的 `kSplits=2` 改为 **1**，加真实分块与最后 prefill logits 的诊断读回；没有改 W8、其他 dense 算子或分派。固定 `prompt-512.json`，不带视觉、MTP 关闭、`--no-cuda-graph --greedy --no-thinking --kv-dtype rk4v4-e8 --prefill-chunk 1024 --max-context 32768 --kv-capacity 32768 --max-new 1`。两次均保留 checkpoint 的 **`[0,508)、[508,512)`** 分块。

结果：两次完整 **248320 个 BF16 logits 逐位相同**，含有效 248077 项；不同项 **0**，相对 L2 **0**，top-1 均为 **ID 16**。对比上一轮相同配置下 Q5 split2 的 246134/248077 个有效项不同、相对 L2 0.1248612403，此结果支持 **Q5 residual 原子累加是此最小用例在注意力同步修复后剩下的波动来源**。关闭 MTP 的路径未执行 W8 的 MTP split launch；Q5 split2 向已有 residual 添加两份 BF16 部分和，三个项的舍入顺序可变，split1 后只有一份残差写回。两次运行的结论不推广为所有长度与 MTP 场景都已证明唯一来源，也不把原子求和顺序波动改称同步正确性 bug。

到 **02:51:34 UTC** 已撤回 Q5、target 诊断钩子和临时头文件，总计约 **2 分 8 秒**，低于 15 分钟限时；源码检查仅进度文档有改动。没有提交临时代码，Q5 的原生产设置保持 split2。数据位于上述外部目录的 `q5-single-split-512/`：双份 `.logits/.chunks/.stdout.txt/.stderr.txt`、完整命令与统计 `summary.json`、临时二进制及 SHA-256 `evidence.json`；构建和执行日志为 `q5-single-probe-build.log`、`q5-single-probe-run.log`。

### 新部分注意力性能优化（2026-10-01，主基准达标；profile 与扩展限制待审阅）

限时起点 **2026-10-01 02:52:31 UTC**，保守按连续 48 小时在 **2026-10-03 02:52:31 UTC** 截止。约 5 小时内完成 12 轮候选、最终复测及全量验证。只修改新 partial/LSE API、kernel、launcher、独立测试和 benchmark，dense 内核、codec helper、分派及 target 运行时没有变化，不开始第 4 步。全部日志、原版/候选二进制、工具与测量留在仓库外 **`D:\deeplearning\NInfer\logs\kvmem-stage3-optimization`**。

**Profile 未完成。** Nsight Compute 2026.2.1 对原版 INT8 dense、T1024/131072 keys 的实际 profile 退出 1，报 **ERR_NVGPUCTRPERM**；metrics 查询及仅 LaunchStats 同样受限。最终复查 `ncu-accepted-permission-recheck.txt` 仍为同一错误，注意 query 返回 0 也不能表示拿到了数据。一次只读管理员启动在 Windows UAC 被取消，未启动管理员进程、未更改全局设置，不重试提权。GPU 性能计数器权限待用户开启；**没有 tensor 管线利用率、共享 bank conflicts、实际 occupancy 或 DRAM 流量报告**，本项不能算完成。旧版和最终 benchmark 已保留，权限就绪后可补同配置 before/after。

**源码/编译证据的归因，尚非硬件 profile。** 原量化 prefill 是每 CTA 的 10 token×6 head（60 行），现在为单 Q head 的 64 行/64 key，与 dense 相同深度的异步 K/V 流水线、query-tile-major 栅格。不能把 10→64 误解为 K/V 复用增加 6.4 倍：原 CTA 已复用六个 head。新路径减少逐 key 元数据/掩码处理，P fragment 跨 D slice 复用，全因果/边界 producer 共用一个 PV/预取消费端，rk4 的 FP32 逆 H64 融入 epilogue；低 split 显著减少 partial 写出和 merge 流量。编译资源为 prefill 16 warp、128 寄存器、26880 B 静态 +65536 B 动态共享；与 dense 的 92672 B 很接近，两者的资源上限均为一 CTA/SM，不能仅由 theoretical occupancy 解释原性能差距。仍有寄存器 spill，未用 profile 证明它的动态成本。

官方 CUDA 13.3.73 portable cuobjdump/nvdisasm 经 SHA-256 校验，仅用于离线反汇编。T4 RK4 从 batch8 到 batch11 的静态指令数 **5522→4461**、PRMT **218→40**、HMMA **64→32**；这是代码消除/向量打包的编译证据，不是动态执行次数。早期标量 nibble 写入已被编译器向量化为 STS.128，所以不把 i4x16 改动的收益错误归因于 store 事务减少。证据为 `common-pv-batch11-build.log`、`batch*-rk4-t4-13.sass.txt`、`rk4-t4-static-sass-summary.json`。

**最终单 pass 主基准。** 真实 PageMajor four-plane INT8 **264 MiB**、rk4 **136 MiB**，131072 key，全部位于 staging；同一 Q/KV 字节、同一因果位置，CUDA event/eager，5 次预热。split **1、2、4、8、16、32** 重新扫描，3 格式×5 个 T×6 个 split 共 **90 行**（`accepted-split-sweep.csv`）。参数在最终三轮之前固定：量化大 T 用 S=1；T4 INT8 S=32、rk4 S=64。T4 每轮 600 次、大 T 每轮 40 次；每组 dense 在新路径前后各测一次，最终 18 行全部保留，dense 前后最大漂移 **1.753%**，没有因速度不理想剔除轮次。以下为三次中位数及最差比例，整体含 partial、固定 LSE fold 和 BF16 输出；比例是各轮比例的中位数，不要求等于两个耗时中位数之比。

| 格式 | T | split | dense ms | 新整体 ms | 新/dense 中位数 | 最差轮比例 | scratch+state MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| int8 | 4 | 32 | 0.342281 | 0.326313 | 0.953535 | 0.953642 | 3.117920 |
| int8 | 1024 | 1 | 22.638400 | 20.964600 | 0.924420 | 0.930262 | 48.375000 |
| int8 | 2048 | 1 | 45.578900 | 42.169700 | 0.921302 | 0.927111 | 96.750000 |
| rk4v4-e8 | 4 | 64 | 0.249395 | 0.226442 | 0.908830 | 0.910526 | 6.141357 |
| rk4v4-e8 | 1024 | 1 | 26.494100 | 23.958300 | 0.904088 | 0.904770 | 48.375000 |
| rk4v4-e8 | 2048 | 1 | 52.963200 | 47.088800 | 0.888386 | 0.889262 | 96.750000 |

**六项均三次达标**：T1024/2048 ≤1.10；T4 ≤1.05。这里是一个全注意力层的计算微基准，不含 H2D/归档/GDN/模型；没有据此宣称整模型 prefill 或 tok/s 已通过门禁。原数据和失败候选均保留，不采用已发现错误的 FMA 候选速度。

**BF16 对原实现的配对复测。** 注意力主体不改，固定原来的 T4/S32、T1024/S2、T2048/S1；旧、新交替执行三轮，第二轮反向排序，计时次数与上表相同。T2048 原始中位数比例有约 **+0.078%** 的差别，dense 归一化后约 **-0.014%**，没有可分辨的性能退步；不把计时噪声说成严格每次都更快。BF16 本身的大 T 相对 dense 仍约 2 倍，本轮并未承诺把它优化到量化目标。

| T | 旧整体 ms 中位数 | 新整体 ms 中位数 | 配对新/旧比例中位数 | 配对 dense 归一化比例中位数 |
|---|---:|---:|---:|---:|
| 4 | 0.627442 | 0.624242 | 0.994919 | 0.994387 |
| 1024 | 47.734900 | 47.565200 | 0.998347 | 0.999341 |
| 2048 | 95.532300 | 95.601700 | 1.000782 | 0.999862 |

**FP32 carry 与加载期预算。** `attention_partial_lse_accumulate()` 用唯一 FP32 `(O,m,l)` state 加当前 S 份 scratch；reset 不读旧 state，之后总是 prior-first，再升序 split，空 pass 保持已生成状态的位模式。末次可融合 BF16 finalize。权重 exp 独立并行，最大值扫描、分母/O 求和仍固定顺序；三处 CTA barrier 保护共享暂存，容量 parts+3，不依赖 DMA 完成顺序。没有为每个 pass 留一份 partial，也没有浮点原子。无分配的 `plan_attention_partial_workspace()` 先按预算与 launch 域降低 preferred split，再验证实际容量；预算为 **258×24×T×(S+1)×4 B**。T1024/S1 **48.375 MiB**、S2 **72.5625 MiB**、S3 **96.75 MiB**；T2048/S1 **96.75 MiB**。第 4 步才由 `build_workspace_plan()` 与视图/staging/输出统一预算。

**多 pass 扩展及未达标项。** 保持总 key 数不变，按逻辑顺序分成 2/4 段，复用上述 scratch/state，计时含每次 fold 与最后输出（`accepted-multipass-2/4.csv`）：

| 格式 | T | 两 pass 整体/dense | 四 pass 整体/dense |
|---|---:|---:|---:|
| int8 | 4 | 1.035370 | 1.134190 |
| int8 | 1024 | 0.927295 | 0.945993 |
| int8 | 2048 | 0.927823 | 0.945296 |
| rk4v4-e8 | 4 | 1.031430 | 1.201420 |
| rk4v4-e8 | 1024 | 0.905230 | 0.911093 |
| rk4v4-e8 | 2048 | 0.893607 | 0.906319 |

两 pass 的 T4 ≤1.05，四 pass 的大 T ≤1.10；**四 pass T4 的 INT8 1.134190、rk4 1.201420 未达到 1.05**。额外 launch、重复 Q/旋转处理与逐 pass fold 是源码/计时所示候选开销，未拿到硬件 profile，不能量化各项占比。单 pass 六项达标不代表任意 pass 数都达标；在第 4 步采用多遍 small-T 前需审阅这项限制。本轮保留正确确定性，不删概率残差、不降低精度或改 dense 以凑速度；停在审阅点。

**独立数值与正确性。** 原 36 例完整保留，新增 T64/65 producer/tile 边界 6 例、巨大有限公共 logits 12 例、穷举全部 256 packed byte 的 6 例，共 **60 个 FP64 用例**。raw FP32 O/m/l、最终 BF16 和 carry 重复运行逐位一致，原 oracle `relL2≤1/256`、多 pass 对单次 `relL2≤1e-3` 与 gross 阈值不变。各格式 oracle 最大相对 L2 为 BF16 **0.00166783337**、INT8 **0.00187258759**、rk4 **0.00169578062**；carry 对单次 FP32 的最大相对 L2 为 **2.32439e-6 / 1.84725e-4 / 1.42141e-4**。融合输出另直接对独立 FP64，合并的 reset=false、中性 NaN payload、空 pass 字节不变和融合/独立输出均覆盖。

一次候选 FMA 概率写法在巨大公共 logits 上被只读审阅发现风险，新增测试先在错误版本失败（`large-logits-red-test.log`，relL2=1/非有限），再仅在新内核用 `__fsub_rn` 后 `__fmul_rn` 修正；无放宽判据。预算降低 split 的边界问题也先以 `budget-cap-red-test.log` 复现。只读终审确认 i4x16 sign/PRMT/对齐、BF16 pair 舍入、PV 共享生命周期及固定顺序 LSE，无必须修问题。

每轮改动后均重跑 FP64/重复检查与 compute-sanitizer **13.0.85** 三项，失败性能候选撤回。最终 `batch12-oracle.log` / `batch12-gqa.log` 退出 **0**；racecheck **0 hazards / 0 errors / 0 warnings**，synccheck/initcheck 各 **0 errors**。恢复 `NINFER_BUILD_BENCHMARKS=OFF` 后完整 product 构建退出 **0**；全量 CTest **100 项，96 通过、4 因缺少其他真实模型制品跳过、0 失败，69.04 s**（`accepted-full-build.log`、`accepted-full-ctest.log`）。跳过不算已通过。所有临时 Q5/诊断代码已撤回，性能代码按授权单独提交到 `feat/kvmem` 并 push，不接线第 4 步。

复现脚本均在外部数据目录：`run-batch12.ps1`（60 例/旧 GQA/sanitizer/候选计时）、`run-accepted-benchmarks.ps1`（90 行扫描、固定三轮、2/4 pass、BF16 旧新配对）、`summarize-accepted.py`（统计）、`run-accepted-full-checks.ps1`（恢复产品配置/完整构建/全量 CTest）。例如 repo 下加载 x64 VsDevCmd 后：

```powershell
$data = 'D:\deeplearning\NInfer\logs\kvmem-stage3-optimization'
$env:PATH = 'D:\deeplearning\NInfer\runtime\ninfer-rtx4090-windows-x64-vision-modes-v3;' + $env:PATH
$env:NINFER_OP_REPORT_STATS = '1'
$san = 'D:\deeplearning\NInfer\logs\kvmem-stage0-baseline\sanitizer-tools\cuda_sanitizer_api-windows-x86_64-13.0.85-archive\compute-sanitizer\compute-sanitizer.exe'
foreach ($check in @('racecheck', 'synccheck', 'initcheck')) {
    & $san --tool $check --error-exitcode 99 --log-file "$data\batch12-$check.log" `
        '.\build-vision-integration\tests\test_attention_partial_lse_merge.exe' `
        *> "$data\batch12-$check-stdout.log"
    if ($LASTEXITCODE -ne 0) { throw "$check failed: $LASTEXITCODE" }
}
# benchmark 需先打开 NINFER_BUILD_BENCHMARKS，测量完成后恢复 OFF；步骤不可并发。
& '.\build-vision-integration\bench\ninfer_attention_partial_bench.exe' `
    --format int8 --tokens 1024 --splits 1 --repeats 40
# --passes 2/4 复用 state，整体计时包含所有 fold；默认 passes=1。
```

### 阶段 3 第 4 步：运行时接线（2026-10-02，修订门禁全部通过）

从 `7f01b0d5` 接入 C=1 `tiered-exact`，本步实现和规定门禁已完成，后续范围止于第 4 步审阅：Main 的 16 个全注意力层使用原始逻辑页号视图、页状态与完成 epoch；16 层回写完成后页才由 DeviceOnly 变 Both，最旧非 sink Both 页才可驱逐。块表、混合访问列表和 key 前缀和在同一计算边界上传；每层一遍注意力，流式页索引设备 staging，因果掩码仍由原始逻辑位置决定。下一层预取在上一层 release 后排队，设备 consumed 事件排序；首全注意力层在前三个 GDN 层前排预取。MTP pool 保持 dense，turn checkpoint、retained resume 和磁盘状态缓存按本步授权关闭并记录日志。

新增 CPU 视图状态机、预算和独立运行时 CTest。运行时测试以固定均匀注意力的独立参考覆盖 BF16/INT8、实际主机页换入、trim 后的 HostOnly 尾页前缀恢复、reset、T4 与有效列不足四列时的零填充，以及锁页模式下 16 层连续异步提交与延迟计算。原事件槽策略在层 13 出现 `capacity=26 references=26` 耗尽；修正为优先复用 references=0 的已 release 槽，再选择未用槽，保留 consumed 依赖，避免所有历史 staging 消费槽持续占用引用。`stress-reuse-ctest.log` 中传输和运行时测试 **2/2 通过，3.38 s**。

同层回写与历史预取回归先在原阻塞读取上失败（延迟约 62 ms），再由 `prefetch_completed` 读取已经发布的归档范围通过；读取与正在回写的页重叠时仍拒绝。基础传输/状态机/预算测试首次 **3/3 通过，1.84 s**。量化预算按四个真实 plane 计算，INT8 128K 流式层 **264 MiB**，rk4 **136 MiB**，再加 64 MiB；small-T S64 scratch 与 state 的独立最大尺寸均计入 partial，而不归入元数据。固定 staging override 会先提高视图最小页数再建立容量曲线，不能在较小视图的探测候选上错误拒绝可行配置。

CLI/serve 本步最小配置为 `--kv-mode tiered-exact`、`--kvmem-view-tokens`/`--kvmem-view`、`--kvmem-sink-tokens`/`--kvmem-sink`、`--kvmem-host-archive`、`--kvmem-staging-mib`。普通 tiered 以 view 为物理上限、`max-context` 为逻辑容量，旧 `kv-capacity` 仅用于 dense/影子完整设备池；帮助和加载日志明确此语义。dense 默认分派/内核未改。BF16 帮助注明仅保证功能、不承诺性能。

数据位于仓库外 **`D:\deeplearning\NInfer\logs\kvmem-stage4-runtime`**，旧基线不覆盖。INT8 4K 容量/2K 视图/MTP-3 的算术冒烟答案为 **5**，正常运行退出 0，text prefill **0.710 s**。32K INT8/8K 视图影子比对退出 0，共 32 块、512 个层输出；真实流式字节 **10,378,739,712**，全层最大相对 L2 **0.000269830**、最大绝对误差 **0.25**。影子模式逐层读回、同步与 CPU 比较，**142.450 s** 仅为验证开销，不能代替普通模式性能结果；该例仅生成 1 token，未计为 needle/decode 门禁通过。

rk4v4-e8 的 32K/8K 视图影子第一次运行在首全注意力层、frontier=1024、尚无流式页时失败：相对 L2 **0.00250436**、最大绝对误差 **0.03125**。日志 `shadow-32k-rk4.stderr.txt` 与失败结果保留；门禁 `1e-3` 不放宽。该次失败已在新 finalizer 的 BF16 输出边界修正，32K 后续 GPU 影子通过；dense 未改，262K 后续失败的另一根因见下文。

128K INT8/32K 视图影子第一次运行也通过：128 块、2048 个层输出，相对 L2 最大 **0.000589935**、最大绝对误差 **0.25**，实际流式字节 **161,109,442,560**；逐层读回验证的 text prefill **605.497 s** 不作为性能数值。正常模式的 32K/8K 与 128K/32K INT8、MTP-3 needle 均答对 **73184269**，均生成 64 token，text prefill 分别 **14.633 s / 73.609 s**，decode **41.89 / 11.93 tok/s**；普通运行的 GPU ready 累计等待分别约每层 **56.67–66.81 ms / 277.74–303.70 ms**。详见 `needle-32k-int8.result.json`、`needle-128k-int8.result.json`。

**旋转格式的最终 BF16 边界。** 捕获同一层 dense/partial 输出及原始坐标 FP32 `(O,l)`，在 CPU 用 FP32 H64 复现 `O/l → H64 → BF16 RNE → H64 → BF16 RNE`，相对 L2 从 **0.00250436328** 降为 **0.000109229787**（最大绝对误差 **0.0078125**）。这对应现有 dense launcher 在旋转坐标中先 BF16 写出、再逆旋转的两次舍入；新 partial 原本在 FP32 中直接逆旋转，最后才 BF16 写出。证据位于 `rotation-diagnostic/analysis.json`，不是主机页或访问列表寻址差异。新增可选 rotated finalizer 只对齐最终输出边界，FP32 原始坐标 O/m/l 与固定顺序 LSE carry 完全保留，旧 finalizer/融合合并和所有 dense kernel/helper 不改。新测试独立 FP64 H64 矩阵、直接 FP64→BF16 RNE oracle（相对 L2 **1e-3**）、精确 tie 逐位、空行 NaN payload、输入状态不改和 64 次重复；CUDA 构建、独立 FP64/64 次重复和三个 sanitizer 已实跑通过；32K RK4 GPU 影子最大相对 L2 为 0.000620063，最大绝对误差 0.25。CPU 复现本身不算门禁通过，262K 后续结果见下文。

**自动预算预留修正。** 最初普通 tiered 将旧 Explicit KV 策略转 Auto 时，误沿用 Explicit 的零自动预留值；262K INT8/MTP-3/128K 视图运行后空闲显存约 **276 MiB**，超过 **606.1 s** 尚未完成 prefill，已停止该候选而不冒充完整计时。NVML 采样 TX **2,852,099–4,450,146 KiB/s**、RX **1,757,031–2,439,746 KiB/s**，提示 WDDM 迁移，但没有 Nsight profile，不能仅凭流量确定原因。日志与 `needle-262k-int8-c1024.aborted.json` 保留。普通 tiered 现在对忽略的 Explicit 策略使用 Auto 默认 **1 GiB** 预留；用户显式 Auto 的预留覆盖仍保留，dense/影子策略不变。CPU 回归先实跑失败退出 1（`headroom-red-test-v2.log`），修正后构建/运行通过。修正后按统一预算重测，262K 两档性能均通过，实际预算和结果见下文。

**整页回写的初始化缺陷。** 运行时 initcheck 首次报告 **129024 errors**：尾页只 append 了有效 token，归档回写整个 plane 页时读取未写入的 suffix 字节。因果掩码不读取这些 token，但整页 DMA 仍必须有定义的源。新增完整 suffix 字节回归先退出 1（`page-tail-red-test.log`）；现在只在新逻辑页首次分配时、按 producer compute stream 顺序清零该物理页的全部层/plane，然后 append。已有尾页保留前缀，HostOnly 尾页继续 restore，不触碰普通 dense 的池或内核。回归 GREEN，修正后的三个检查全部通过：initcheck/synccheck **0 errors**，racecheck **0 hazards/0 errors/0 warnings**（`page-tail-green-*.log`）。本次全 attention 数值测试的三 sanitizer 也全部通过：racecheck **0 hazards/0 errors/0 warnings**，synccheck/initcheck **0 errors**；7 项选定 CTest **7/7 通过，8.73 s**，当时最终端到端和全量 CTest 尚未完成，后续完整结果见本节下文。

**普通模式整模型性能已通过。** 32K/8K 视图、128K/32K 视图和 262K INT8 的 MTP-3/64-token needle 均答对 73184269；可分页归档的 32K 也答对，prefill 17.822 s。262K 1024/2048 分块的数据如下，旧数据全部保留：

| 分块 | 实际 GPU 视图 token | Main MiB | staging MiB | partial MiB | text prefill s | 对 dense 212 s 的比例 | 16 层 GPU ready 累计等待 ms | 整卡峰值/最低空闲 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 108800 | 3506.250 | 372.859 | 48.375 | 226.582 | 1.06878 | 11448.705 | 23896 / 247 |
| 2048 | 101120 | 3258.750 | 388.328 | 96.750 | 216.716 | 1.02225 | 10040.328 | 23592 / 551 |

两档均低于 243.8 s（1.15×212）。数据是一次采样，不承诺不同桌面负载下严格复现；1 GiB 是加载期预算预留，不是整卡运行期间最低空闲保证。GPU ready 等待含 prefill 与后续 64-token decode，不等于仅 prefill 或总 DMA 时间，不能据此声称精确传输覆盖率。普通每层每块只有一次 partial：272×16=4352 / 144×16=2304，没有四遍 decode 或容量后备路径触发。逐层 ready 与独立 CPU enqueue 计时在各 `.stderr.txt`/`.result.json`，完整命令、二进制 SHA、NVML 1 s 整卡采样及原始输入路径均保留。

上述两次运行的加载日志给出以下实际设备预算（B，不是最高运行占用估算）。Main、MTP KV pool payload 与独立 tiered backing 分开；表不包含模型权重，也不能把它的和当作整卡显存峰值：

| 项目 | chunk 1024，B | chunk 2048，B |
|---|---:|---:|
| Main 视图 | 3676569600 | 3417047040 |
| MTP 完整 dense pool | 553783296 | 553783296 |
| staging（层全部流式量 + 64 MiB） | 390971392 | 407191552 |
| partial FP32 scratch + state | 50724864 | 101449728 |
| 元数据/固定区，含对齐 | 164608 | 164608 |
| 通用 workspace | 180953088 | 361906176 |

两档逻辑页容量均 4096；访问列表为加载期三个独立的 `{logical_page, physical_page}` buffer，**3×32768=98304 B**；前缀和 **3×16388=49164 B**；逻辑块表 **16384 B**，三项有效字节共 **163852 B**，元数据固定区其余 **756 B** 为区域对齐。普通模式未分配 shadow 输出。锁页主机归档 **8858370048 B（8.25 GiB）**，直传、中转环 **0 B**；可分页 32K 验证则使用固定 **4×64 MiB** 锁页环，二者不是设备 staging 的一部分。

本步 tiered 明确关闭 **turn checkpoint、retained resume、磁盘 prompt/state 缓存与 CUDA Graph**（eager 执行）；相关配置、发布及恢复入口均不产生可复用 tiered 快照，日志说明关闭。MTP 本身仍开启，MTP pool 保持 dense，未做第 5 步窗口化；Main 的页 trim/reset 仍受测试覆盖，不能把 checkpoint 关闭等同于关闭这些页生命周期操作。

**262K RK4 原影子判据失败的确定性复现与根因（2026-10-01，历史记录）。** `shadow-262k-rk4-fixed` 在全注意力层 6、frontier **146432** 失败：相对 L2 **0.0010094**、该层最大绝对误差 **0.125**，门禁仍为 `1e-3`。完整 dense KV 和测试资源实际可放下，不能按显存不足跳过。失败数据、原二进制 SHA 与日志均保留。捕获同一份 Q、位置、四个 resident/staging plane、访问列表/前缀及完整输出后，仓库外 `alpha-repro.cpp` 重算 dense 与 partial 均和捕获结果**逐位相同**；一 split 的误差仍为 **0.0010094**，2/4/8 split 为 **0.00269704 / 0.00313621 / 0.00330881**，增加 split 不能解决。

根因已由算术消融坐实：dense prefill 的 alpha 为 `exp2(fma(m, scale_l2, -RN(nm*scale_l2)))`，当 `m==nm` 时仍残留乘积舍入误差；新 partial 先减再乘，指数严格为零。仅在临时新 partial 的 prefill alpha 中照搬 dense FMA（概率仍先减再乘），同一固定输入的误差下降到 **0.0000645685**；dense 输出仍逐位不变。该改法**不能提交**：既有独立 FP64 大公共 logit 测试退出 **1**，例如 RK4/T64/common=1e10 的第一项实际 **0.75**、数学参考 **0.257692**。也对未修改的 dense A3 API 直接跑了独立均匀注意力参考（两个页相同 K、前 64 key 的 V code=1、后 64 code=3）：RK4/common=1e10 的相对 L2 **0.808230**、第一项 **0.730469** 对 **0.257692**；INT8/common=1.2e10 产生 NaN，RK4 同档产生 Inf。极端公共值是诊断，不替代模型代表范围的质量基准，但证明这种 FMA 写法破坏了公共 score 平移不变性。模型固定输入的消融直接说明长历史的实际影子超限主要由 dense alpha 累积舍入引起，不是主机寻址/页传输不一致。

独立捕获数据的 FP64 native-Q8 参考另保留在 `alpha-capture/profile-oracle.json`，选择八个最大逐行差异与四个固定 head 的全部结果，不挑选有利样本；它包含 Q8/FP16 V 计算边界，不等同于主 BF16-Q 数值合同，不能单独据此宣称所有输出比 dense 更精确。实际最大值对应的稳定 alpha 舍入残差约 ±9.54e-7，24576 行中 21892 行的 alpha 不等于 1；连续 2288 块的示意旧项倍率约 **0.998501–1.001638**，这里只是累计量级分析，不是完整时序乘积实测。

证据在外部 `alpha-capture/`、`alpha-repro-{baseline,alpha-only,alpha-only-confirm}-run.log`、`alpha-only-oracle.log`、`alpha-oracle.log`，固定数据捕获运行人为停止且跳过前面各层的 CPU 比较，**不计门禁通过**。临时 alpha 改动和全部源码捕获/完整 logits 钩子均已撤回；不放宽门禁，不把错误 FMA 引入新路径。尚未找到经过完整门禁验证的一般性新路径修正，也未证明所有新路径修正都不可能。当时按用户“不要改 dense 内核/分派”和 D1 的限制，请求用户决定是否另行授权修复 dense prefill 的 alpha/概率指数（先相减再乘）并重录相关金标准，阶段第 4 步未提交/push。**后续用户已明确不修改 dense，并批准仅对 262K 改用 FP64 判据；本段不再是待授权请求，修订与新实测见后文。**

**262K RK4 门禁 B 已通过。** 固定 `needle-262k-90`，dense/tiered-exact 各独立运行三次，MTP-3、贪心、无 vision、无 CUDA graph，每次生成 64 token。六次均答对 73184269，全部 token ID 完全一致；每次 draft/accepted 均为 **47/47（100%）**。对九个 dense/tiered 配对逐步比较有效词表（248077 项）的 BF16 完整 logits，共同前缀均为 64，无分歧点。dense 三次两两最大相对 L2 为 **0.3937184153**，按既定规则得到包络 **0.7874368307**；dense/tiered 九配对的最大值 **0.4103691463**，均在包络内。这是带现有模型 logits 波动包络的门禁通过，**不表示逐位 logits 一致，也不抵消原门禁 A 的超限；262K 的用户修订后 FP64 判定另见后文**。本轮未进一步定位或修改 Q5 波动。

临时完整 logits 钩子只存在于仓库外归档诊断二进制 `stage4-diagnostic-ninfer.exe`（SHA-256 `03e6e0305e911904253b10758142cbb0c24d87c97f3cd7ed4667b016a5394789`），源码已撤回；本组未开启 alpha 捕获/旋转输出捕获。完整命令、输入、64 个 token ID、词表 logits 和 MTP 接受元数据保留于 `gate-b-{dense,tiered}-{1,2,3}.*`。仓库外 `run_gate_b.py`/`check_gate_b.py` 顺序执行并验算，结果 `gate-b-summary.json`，退出 0。最后一次 tiered prefill 228.529 s 与前两次 208.347/209.643 s 的差别如实保留；门禁 B 时序不替代上述 INT8 性能门禁。

**FP64 修订前清理版本的构建与 CTest（历史验证）。** 全部临时源码诊断已撤回后的产品配置（benchmarks OFF）完整构建退出 **0**；随后再次完整构建确认 `ninja: no work to do`。全量 CTest **103 项：99 通过、4 跳过、0 失败，233.42 s**，串行 `-j 1`，为真实 Qwen3.8-27B 权重和视觉 GGUF 设置路径。四项跳过为 Qwen3.6-27B prefix real、35B-A3B real、35B-A3B DFlash real 与 DFlash load-plan，缺少对应测试制品；不计为通过。日志为 `clean-full-build.log`、`final-full-build.log`、`final-full-ctest.log`，复现脚本 `run-production-full-checks.ps1`，均在仓库外数据目录。最终产品 `build-vision-integration/apps/ninfer.exe` 的 SHA-256 为 `5f3763ac9181985341ccf1253cf6ea763841811061c5b6c932c41f11ced396ed`。最终检查 `git diff --check` 退出 0，源码无临时 capture/logits 钩子，dense prefill/decode 内核、原 launcher/wrapper 及新 partial 主体与基线相比无修改；正常 LF→CRLF 提示不表示空白错误。当时分支与 origin 均在 `7f01b0d5`，本步改动留在工作区。规格与质量审阅已通过，但原门禁 A 超限，因此当时未提交/push。后续用户批准 FP64 判据并完成三层验证，结果见下文；没有开始 MTP 窗口化或 checkpoint 恢复。

**用户修订后的 262K FP64 验证（2026-10-02，通过）。** 原始 0.0010094 超限和消融证据保留；用户明确不修 dense，只修订 262K 档门禁 A。临时诊断完整跑到 262080 输入 token，不因旧 1e-3 提前终止，逐层逐块仍计算并记录 tiered/dense 的相对 L2 和最大绝对误差，下游始终用 dense。整个影子过程最大相对 L2 为 **0.00143766**；按各层全程最大值取前三名：第 **7（0.00143766）、6（0.00143742）、8（0.00135455）** 层，含原超限第 6 层。原始完整日志、排名和抓取 SHA 均在 `shadow-262k-rk4-fp64-capture.*`、`fp64-layer-selection.json`、`fp64-capture-binary.json`。影子 prefill 1410.467 s 含诊断读回与落盘，不是普通路径性能。原判据下退出 0 的诊断捕获本身不表示新门禁通过。

在三层分别取最后一个 prefill 块（位置 **261120–262079，T=960**），抓取原始 BF16 Q、位置、dense/tiered BF16 输出及量化 K/V/FP16 scales。抓取时每个 logical page 的混合 resident/staging 四个 plane 与完整 dense pool 原字节逐字节核对，每层 **285143040 B** 全部一致；仅落盘一次规范逻辑页布局，参考不依赖 CUDA 解码。仓库外 `fp64-full-oracle.cpp` 直接包含并调用 `tests/ops/attention_reference.h` 的 FP64 `hadamard()` 与 `oracle()`，signed nibble×原 FP16 scale 后做独立 H64 解码；原始 BF16 Q、因果掩码、double scores/softmax/加权输出，无 Q8 重量化或 HALF V staging 近似。每层完整 **960×24×256=5898240** 个输出聚合，报告三组相对 L2 与最大绝对误差，FP64 参考不做 BF16 最终舍入。六个 CPU 线程只并行独立 token，不改变每行参考的求和顺序；原单元测试 oracle 源码未改。临时抓取源码已从前置完整备份恢复，诊断二进制保留仓库外；任何一个完整层失败即停，不选择有利行或再改判据。

隐藏影子模式的正式判定策略也按修订更新：**仅 262144 logical context 的 RK4（packed K/V、rotate K/V、E8 非 root）** 不因 tiered/dense 的旧 1e-3 先退出，仍检查有限值、逐层记录误差，并在构造/汇总明确打印 `criterion=offline_fp64 status=not_evaluated` 与“退出 0 不是门禁结论”；不在产品中运行巨大 CPU oracle，也不加入临时捕获钩子。32K、128K 和其他格式仍保留严格 1e-3，正常 shadow 关闭时不增加拷贝、算子或同步。预算 CTest 对该策略正反例先 RED（`fp64-policy-red-test.log`，退出 1），实现后 GREEN（退出 0）；只读审阅参考工具与该小改动均通过。

该正式策略与临时抓取清理完成后再次全量构建退出 **0**，CTest **103 项，99 通过、4 制品缺失跳过、0 失败，327.93 s**（`fp64-rule-final-full-build.log`、`fp64-rule-final-full-ctest.log`，脚本 `run-fp64-rule-full-checks.ps1`）。本轮 CTest 与独立 CPU oracle 同时运行，时间不作为性能指标，GPU 上没有并行模型测量。最终产品 `ninfer.exe` SHA-256 为 `e4f7e13433952b0a46f4b48e48b484e1b80239db9ba558d0fef1794c892442eb`，临时诊断 marker/capture 环境钩子已清除；`git diff --check` 退出 0，dense 内核/分派不改。FP64 三层判定独立完成且通过，结果见下表；CTest 全通过不代替门禁 A。


**完整末段块的 FP64 结果。** 相对 L2 用对应参考的完整块范数，两个 FP64 比较使用同一个参考范数；最大绝对误差独立记录，不用于改写相对 L2 判据。

| FA 层索引（日志 0 基） | dense/FP64 相对 L2 | tiered/FP64 相对 L2 | tiered/dense 相对 L2 | dense/FP64 最大绝对误差 | tiered/FP64 最大绝对误差 | tiered/dense 最大绝对误差 | 新判据 |
|---|---:|---:|---:|---:|---:|---:|---|
| 7 | 0.00543534623365 | 0.00542739712411 | 0.00138025890506 | 0.575143965979 | 0.575143965979 | 0.125 | 通过 |
| 6 | 0.00516397129311 | 0.00516044643686 | 0.00143742396675 | 0.341367090559 | 0.353200096628 | 0.125 | 通过 |
| 8 | 0.00425544989244 | 0.00424812821871 | 0.00115082877968 | 0.438819035622 | 0.438819035622 | 0.125 | 通过 |

三层各 5898240 个输出均满足新判据，没有删行或再调标准。第 6 层的最大绝对误差 tiered 为 0.353200096628，高于 dense 的 0.341367090559；如实记录，此项不属于用户批准的相对 L2 判据。CPU oracle 耗时分别为层 6 **2092.581 s**、层 7 **1678.537 s**、层 8 **1559.988 s**，是独立参考的计算成本，不是 NInfer 推理性能。CPU 参考原始输出（FP64 全数组）、完整 JSON 和日志在 `fp64-capture-262k/layer-{7,6,8}/oracle-full.*`、`fp64-layer-{7,6,8}-full.log`；汇总 `fp64-gate-a-summary.json`。

复现：先在 x64 MSVC 环境执行外部 `build-fp64-full-oracle.ps1`，再执行 `fp64-full-oracle.exe <数据目录>/fp64-capture-262k/layer-6`；通过后依次计算 layer-7、layer-8，任一非零退出即停止，`run-remaining-fp64.ps1` 实现这一顺序。此程序直接包含仓库单元测试 reference header；构建不启用 fast-math，六线程只并行独立 token。

**本步代码提交（2026-10-02，feat/kvmem）。**

- `6466a166` — `fix(ops): preserve rotated BF16 boundary in partial attention`：新 partial 的旋转 BF16 输出边界与独立 FP64/64 次重复回归；不修改 dense。
- `d33157a9` — `feat(kvmem): integrate exact tiered KV runtime and budgets`：Main 视图状态机、主机归档/跨层预取接线、统一预算、产品参数、显式关闭的功能、影子策略与对应 CTest。
- README、设计、实施清单和本测量记录随独立文档提交发布；推送目标仅 `origin/feat/kvmem`，没有合并其他分支。本步达到审阅条件，范围止于第 4 步，未开始第 5 步。

## 阶段 4：稀疏 decode

- 状态：未开始

## 与设计的偏差

| 日期 | 决策编号 | 偏差 | 原因与证据 | 用户是否同意 |
|---|---|---|---|---|
| 2026-09-30 | D3 前提 | 后续分层视图保留原始逻辑页号，不直接以紧凑视图槽号代替位置；阶段 0′ 不实现 | `paged_kv_address.cuh:16` 和 prefill/decode 内核按 `position >> 6` 查块表并以原始 key 位置做掩码，详见上文静态核对 | 是，用户明确选择“保留原始页号” |
| 2026-09-30 | D4 | 默认 `auto`：加载期整块 CUDA 锁页且剩余物理内存 ≥4 GiB 则归档直传，否则释放部分分配后退回可分页归档与 4×64 MiB 中转环；显式 `pinned` 失败即退出，`pageable` 保留旧路径。两者统一「层 → plane → 逻辑页」布局 | 262K 就绪后整块可锁页 15488 MiB；锁页 H2D 9.83 GiB/s，可分页经环端到端 4.19 GiB/s。旧「4090 WDDM 锁页限额更紧」的推断不成立，4060 仍须实测 | 是，用户明确同意修订 |
| 2026-10-01 | D1，用户授权例外 | 仅修复已知的 dense 量化 small-T 注意力共享内存竞争，独立修复分支合入后重录九组金标准，旧数据保留；Q5 和 dense 分派不改 | 原版 racecheck 64 hazards，修复后 0；12 组各 64 次逐位一致且 FP64 oracle 通过，详见修复记录 | 是，本轮用户明确要求现在修复，并限定文件与两处 barrier |
| 2026-10-01 | D1 与门禁 A | 262K RK4 的新路径与当前 dense 在 146432 token 处相对 L2 0.0010094，固定输入消融确认 dense prefill alpha 的 FMA 舍入累积；照搬它会破坏现有 FP64 公共 logit 测试。保持门禁和正确新路径，是否另行修 dense 须用户决定 | 固定输入 dense/partial 重算均逐位一致；临时 alpha-only FMA 将两者误差降到 0.0000645685，但独立数值测试失败，未修改 dense 的均匀参考直接出现大误差/非有限输出，详见上文 | 已决定不修 dense；262K 门禁 A 后续按 FP64 修订见下一行，原失败保留 |
| 2026-10-02 | 门禁 A，D1 保持 | 32K/128K 的 tiered/dense 1e-3 不变；262K RK4 按影子误差选至少三个层（包括原超限层）的末段完整块，采用同一量化 KV 字节的 CPU FP64 oracle，要求逐层 tiered/FP64 相对 L2 不超过 dense/FP64；同时记录 tiered/dense 数值 | **在测得 0.0010094 之后**依据固定输入 alpha 消融作出的事后修订，保留原失败与所有数据；不修改 dense、不放宽任何其他门禁 | 是，用户明确同意；新判据实测通过 |

## 阶段 3 第 5、6 步：已完成，停在审阅点（2026-10-02）

- 基线 `f5da0049`；实现清单见 [stage5-6-runtime-plan](stage5-6-runtime-plan.zh-CN.md)。本轮数据在仓库外 `D:/deeplearning/NInfer/logs/kvmem-stage5-6`。第 4 步生产二进制已归档为 `stage4-baseline-ninfer.exe`，SHA256 为 `e4f7e13433952b0a46f4b48e48b484e1b80239db9ba558d0fef1794c892442eb`。
- 用户明确选择继续遵循 D11：默认 32768 token 的 MTP 物理页环预算包含 sink，近期页使用剩余容量。跨页临时草稿的保护页也在固定预算内，不能额外增加池大小。
- 第 6 步 public Engine 短对话基线（尚未实现时）：追加与回滚均 `reused_prompt_tokens=0`，实际计算完整提示词，符合复用入口关闭的当前状态；输出回滚/冷启动 ID 相同，不能以此代替已实现复用。原始记录 `reuse-baseline-red.json/.log`。这份短提示词的后续回答没有重述 needle，故不算 needle 门禁通过，也不作为正式长上下文质量数据。
- 第 5 步源代码接线完成，规格与代码质量审阅无阻塞项；dense 内核与分派未改。参数、预算、窗口页和 masked append 的回归先失败后通过。最终全量构建通过，CTest **105 项，101 通过、4 缺少其他模型制品跳过、0 失败，199.95 s**，见 `stage5-full-build-r3.log` / `stage5-full-ctest-r3.log`。
- compute-sanitizer **13.0.85** 对 `ninfer_test_mtp_window` 三项复测：racecheck **0 hazards / 0 errors / 0 warnings**，synccheck、initcheck 各 **0 errors**。首轮 initcheck 指向测试整块 D2H 字节比较的未初始化空闲页/尾部；测试夹具改为在已有初始化同步之后、计算流上清零该存储，未修改生产算法。首轮失败日志及 r2 复测日志均保留在外部目录。最终 CLI SHA256 `b9712bdf9a50c12499ad619300004c7e0c114135ef7159adf33795430ef235bc`。
- 正式 128K/262K 第 5 步门禁均通过；该步提交时第 6 步尚未实施，后续结果见下文。配置：INT8-G64 归档、chunk 1024、贪心、无视觉、无 CUDA Graph、Main view 上限 131072、MTP-3 + optimized draft head；逐组串行执行，二进制相同。

### 第 5 步：正式门禁与显存

| 上下文 | 关闭 MTP prefill（s） | 窗口 MTP prefill（s） | 64 token ID | needle（双方） | 接受率 | 对第 4 步下降 | 窗口 ready prefill / decode（ms） |
|---|---:|---:|---|---|---|---:|---:|
| 128K | 83.292 | 84.301 | 完全相同 | 正确 | 47/47 = 100% | 0 个百分点 | 3.159360 / 0.764704 |
| 262K | 222.939 | 224.297 | 完全相同 | 正确 | 47/47 = 100% | 0 个百分点 | 262.134150 / 6935.293000 |

两档都生成完整 64 token，没有分歧，因此无需启用门禁 B 的并列例外。第 4 步完整 MTP 的两档接受率均为 47/47。关闭 MTP 的 ready 等待分别为 128K：3.331072 / 3.022752 ms，262K：128.994970 / 26442.340000 ms（prefill / decode）。各层明细保存在四份 stderr 和 `stage5-gates-summary.json`。

| 上下文 | 原 full-MTP payload（MiB） | 窗口 payload（MiB） | 窗口 auxiliary（MiB） | 原二进制实际 Main view（token） | 窗口实际 Main view（token） | 增量（token） |
|---|---:|---:|---:|---:|---:|---:|
| 128K | 264.128906 | 66 | 1.551270 | 126272 | 131072 | 4800 |
| 262K | 528.128906 | 66 | 1.559082 | 108800 | 124032 | 15232 |

原二进制用相同 context/view/chunk/MTP/预算策略补测加载预算（`baseline-128k-budget`、`baseline-262k-budget`，短算术提示词），此处不把短提示词的时间当作长 prefill 基线。窗口启动日志中的同一已确定预算估算为 124544 / 108736 token，和跨运行实际值有区别，保留估算标签及两套原始数值，不能把它解释为固定桌面负载下的严格对照。

128K 窗口 Main 4224 MiB、staging 64 MiB、partial 48.375 MiB、metadata/fixed 82688 B、general workspace 180953088 B；262K 分别为 Main 3997.125 MiB、staging 342.179688 MiB、partial 48.375 MiB、metadata/fixed 164608 B、general workspace 180953088 B。两档 pinned host archive 分别为 4429185024 / 8858370048 B。窗口测试的整卡峰值分别为 23761 / 23737 MiB，包含桌面及其他程序，不能当作进程独占占用。资源汇总见 `stage5-resource-summary.json`。

复现：在外部目录使用 `run_full_checks_retry3.ps1 -Stage stage5`，随后 `python run_stage5_gates_r2.py`（三项 sanitizer → 四组正式模型测量 → 原版两组加载预算 → 门禁比较），`python summarize_stage5.py`。sanitizer 命令为版本 13.0.85 的 `compute-sanitizer --tool racecheck|synccheck|initcheck --error-exitcode 1 build-vision-integration/tests/ninfer_test_mtp_window.exe`，三项各自运行。临时代码、模型和测量数据均未进入仓库。

第 5 步独立提交并 push：`4da637a4`（`feat(kvmem): window MTP cache within a fixed sink and recent ring`）。第 6 步随后按精确 frontier 与 generation 恢复契约完成，正式结果如下；不开始第 7 步。

### 第 6 步：实现与验证记录（已完成）

Main 快照只持有 bundle 身份、精确 archive/view frontier 和 generation、有效性引用，不复制 KV，也不保存过时的物理槽映射。实际 checkpoint 分块结束时捕获 Main 元数据与已有的 GDN/hidden 状态，MTP 保存 `F−1` 的桥接边界及标签；retained resume 保存已提交 continuation 与真实 RoPE 坐标。restore 排空两条传输流和两个 worker、截断归档/视图、重新换入 sink/recent/当前部分页，最后发布新的块表、访问列表和前缀和。MTP 只取快照标签与存活标签交集，丢失历史不补算，日志明确 `replay=0`。磁盘状态缓存继续关闭。

定向 6 项 CTest **全部通过、0 失败、无跳过，34.25 s**（`step6-targeted-green-r2.log`），实际 Engine 测试 29.97 s，覆盖可分页+MTP、锁页+MTP、可分页无 MTP 的追加、重复 checkpoint 恢复、冷启动 ID 比较和单输出 token exact-hit；其他测试覆盖精确 `F=129`、部分页续写、非连续 lease、foreign bundle、reset/trim-below 后重新增长仍拒绝旧快照、旧 wait/staged/release ticket 拒绝及活跃 MTP guard。初始 CPU/Main/MTP RED/GREEN 的原始执行未重定向到文件，只有工具输出观察记录，见 `step6-tdd-observed.txt`；Program 有实际 RED 日志 `step6-program-red-r2.log`。最早 `step6-program-red.log` 是无效 chunk 配置失败，不作为功能 RED 证据。配置问题及历次 GREEN 失败日志保留，未覆盖。

聊天测试显式 `preserve_thinking=true`：现有模板在非 thinking 生成头中仍包含空 thinking 块；默认 false 会在后续轮删除历史块、改变缓存前缀，因而选旧 checkpoint 而非 retained append。没有改模板或 dense。仓库外首次 4K smoke 的原「access code」输入还触发第二轮拒答，needle 失败，回滚/冷启动轨迹也不同（`stage6-smoke-clean.*`）；这不是通过的质量数据。正式测试把合成数据字段改成「reference number」，数字仍为 `73184269`，原 fixture、失败记录和新旧 SHA 均保留于 `stage6-fixture-manifest.json`。中性数字 fixture 的 4K smoke（`stage6-smoke-reference.*`）追加、checkpoint、needle 和回滚/冷启动完整 64 token ID 全部通过。

128K/262K 正式 fixture 在原合成 archive 的末段少放 640 个 filler token，给两轮输入和 64 输出留出容量；实际 token 数以测量结果为准，不把容量配置误写成提示词恰好等于容量。配置为 INT8-G64、MTP-3 optimized head、chunk 2048、无视觉/无 CUDA Graph、无磁盘缓存、贪心、`preserve_thinking=true`。每组顺序执行 first（16 输出）、append（64）、checkpoint rollback（64）、相同聊天输入的 cold reset（64）。两档各三组抓取完整 BF16 词表 logits，按共同前缀包络及并列规则验算；cold 是同精度 tiered-exact 从头计算，在 262K 不用无法放下的 dense INT8 替代参考。正式数据已完成，判定如下，不以定向 CTest 或 smoke 代替门禁。

**128K 正式复用门禁通过。** 三组 first 输入各 130375 token；第二轮各 130441 token，retained append 复用 130390，只计算 51；checkpoint 回滚复用 130437，只计算 4；cold 从头计算 130441。first 的最后一个已输出 token 尚未反馈进入 Main 执行 frontier，因此 retained frontier 是 `130375+15`，追加后缀也包含这个待处理 token，没有把未计算的 KV 计入命中数。三组 append/checkpoint 都命中正确路径，needle 全正确，9 个 rollback/cold 配对 64 token ID 全部相同。三次 cold 两两最大相对 L2 **0.3351892057014107**，按既定门禁得到包络 **0.6703784114028214**；完整词表比较通过，不声称 logits 逐位一致。完整逐步误差见 `stage6-128k-gates-summary.json`，没有分歧点，无需并列例外。

| 128K 组 | first prefill（s） | append prefill（s） | rollback prefill（s） | cold prefill（s） | ready prefill（ms） | ready decode（ms） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 73.080866 | 0.598555 | 0.428131 | 74.024215 | 3.590368 | 8.785344 |
| 2 | 74.552613 | 0.640813 | 0.426275 | 75.093473 | 3.595552 | 8.350210 |
| 3 | 75.112074 | 0.601518 | 0.428285 | 75.653568 | 3.371264 | 6.385118 |

ready 为每个 Engine 的四次调用（first/append/rollback/cold）在 16 层上的累计 GPU 等待，prefill/decode 分别统计，不是单次请求值；各层 CPU 下发等待也保存在 stderr 与资源 JSON。三组四阶段的 MTP draft/accepted 分别均为 first 11/11，其余各 47/47（全部 100%）。原始 `stage6-128k-{1,2,3}.*`、输入 SHA、二进制 SHA 和整卡采样保存在仓库外，三组整卡峰值均为 23607 MiB，含桌面占用。

**262K 正式复用门禁通过。** 三组 first 输入各 261447 token；第二轮各 261513 token，append 复用 261462、只计算 51；checkpoint 回滚复用 261509、只计算 4；cold 从头计算 261513。needle 四阶段每次均正确；9 个 rollback/cold 配对的 64 token ID 全部相同。三次 cold 两两最大相对 L2 **0.21513618054751757**，门禁包络 **0.43027236109503514**，rollback/cold 九配对最大 **0.40204128237988074**，通过。128K 九配对最大为 **0.3417149724698084**，也通过其 0.6703784114028214 包络。两档都没有分歧点，不使用并列例外；不能把这些结论写成固定 1e-3 或 logits 逐位一致。完整有效词表为 248077 项，逐步数据保存在 `stage6-gates-summary.json`。

另用同一批抓取核对 retained append：两档各 9 个 append/cold 配对也都是 64 token 相同，最大相对 L2 分别 **0.3238738598588876 / 0.2951652530896267**，都在同一个已确定包络内，见 `stage6-append-gates-summary.json`；没有增加 GPU 运行、扩大参考样本或重算更宽的包络。

| 262K 组 | first prefill（s） | append prefill（s） | rollback prefill（s） | cold prefill（s） | ready prefill（ms） | ready decode（ms） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 196.288366 | 1.530497 | 0.807864 | 196.284459 | 1983.223100 | 22673.410000 |
| 2 | 223.591163 | 1.601274 | 0.955930 | 223.911307 | 2144.800200 | 24299.730000 |
| 3 | 223.315711 | 1.614789 | 0.910472 | 218.000986 | 2127.249900 | 22432.670000 |

同样是四次调用的 phase 累计等待，不将它冒充单次请求或隐藏比例。262K 三组 first 均为 draft/accepted 11/11，其余阶段各 47/47，接受率全部 **100%**。rollback 日志各记录 **510** 个 surviving pages、**1** 个 missing snapshot page，`draft_history=reduced acceptance_may_decline=1 replay=0`；缺页没有触发 MTP KV 补算。三组整卡峰值 **23832 / 23904 / 23540 MiB**，包含 WDDM 桌面与其他程序。后两组 prefill 比首组慢，原始数字全部保留，不选最快结果代替重复数据。

**本次 chunk 2048 的实测加载预算。** 两档各三组计划相同；与第 5 步 chunk 1024 的实际视图不同，不能混用。MTP payload 均 66 MiB，固定 512 页仍含 sink 4/recent 507/guard 1；旧完整 MTP payload 分别 264.128906 / 528.128906 MiB。

| 项目 | 128K | 262K |
|---|---:|---:|
| Main 视图（token） | 125120 | 116416 |
| Main payload（MiB） | 4032.187500 | 3751.687500 |
| staging（MiB） | 75.988281 | 357.519531 |
| partial workspace（MiB） | 96.750000 | 96.750000 |
| block/access/prefix 等 metadata 与 fixed（B） | 82688 | 164608 |
| general workspace（B） | 361906176 | 361906176 |
| MTP auxiliary（B） | 1626624 | 1634816 |
| CUDA pinned archive（B） | 4429185024 | 8858370048 |

资源、各层 GPU/CPU 等待和分阶段时间汇总 `stage6-resource-summary.json`，两条归档路径的恢复回归见定向 CTest；长测 auto 实际选 pinned，没有把锁页长测当成可分页长测。正式测量没有并行模型/测试/构建；桌面仍接 4090，系统占用会漂移。时间是带临时完整 logits 读回的 Engine 报告，不替代第 4 步产品性能门禁。

外部测量程序 `stage6-diagnostic-reuse-measure.exe` SHA256 **5ab214fb6657179270c60ab0cf9df0e635c01e33adbbf9b630de1f073be34dc3**；同时归档的诊断 CLI 为 `stage6-diagnostic-ninfer.exe`，SHA256 **9955499396e7aee15245a06dbe8a010c9664fdd005f73ae6bba39748ba106124**。临时 header 和三处 logits 钩子已按 SHA 清单恢复生产源码原始字节、移除临时文件，随后才进行最终产品构建。抓取/验算脚本、输入 SHA、原始完整词表数据都留在仓库外，没有提交临时代码或模型。

**最终产品验证与交付。** 完整构建退出 **0**（207 个更新动作），随后全量 CTest **106 项：102 通过、4 缺少其他模型制品跳过、0 失败，230.25 s**，`-j 1`，真实 Qwen3.8-27B 和视觉 GGUF 环境路径已设置。日志为 `stage6-full-build.log`、`stage6-full-ctest.log`，复现 `run_full_checks.ps1 -Stage stage6`。跳过的是 Qwen3.6-27B prefix real、35B-A3B real、35B-A3B DFlash real 和 DFlash load-plan，不计为通过。既有 dense 算子/前缀相关 CTest 通过；本步没有重新录制九组 dense 基线，dense 注意力内核和原 launcher/wrapper 分派没有修改。

compute-sanitizer **13.0.85** 分别对 `ninfer_test_mtp_window` 和 `ninfer_test_tiered_runtime` 运行 racecheck、synccheck、initcheck，六项均退出 **0**；两组 racecheck 都 **0 hazards / 0 errors / 0 warnings**，其余四项各 **0 errors**。后者包含可分页 BF16/INT8、锁页 INT8、影子 BF16 与延迟生产者的十六层回写。原始六份 `stage6-ninfer_test_{mtp_window,tiered_runtime}-{racecheck,synccheck,initcheck}.log` 和 driver 日志保留在外部目录。命令是版本 13.0.85 的 `compute-sanitizer --tool <racecheck|synccheck|initcheck> --error-exitcode 1 build-vision-integration/tests/<ninfer_test_mtp_window|ninfer_test_tiered_runtime>.exe`，按测试与工具逐项运行；复现脚本 `run_stage6_sanitizers.py`。

产品 CLI `build-vision-integration/apps/ninfer.exe` SHA256 **093c014acb6899cc3628f1970ac97e64aad7f278da0e8e2e817eaaf00d791adf**，外部另保留 `stage6-product-ninfer.exe`。规格与代码质量独立审阅通过；源码无临时 trace/capture 钩子，`git diff --check` 无错误。实施、测试和文档以 `feat(kvmem): restore tiered resume and turn checkpoints` 独立提交并推送 `origin/feat/kvmem`；磁盘状态缓存仍关闭，没有新增 pinned 申请或第三个工作线程，没有进入第 7 步。

正式门禁复现：外部 `run_stage6_gates.py` 顺序执行六组测量（使用归档诊断程序），`check_stage6_logits.py` 验算 rollback/cold 包络与 token，`check_stage6_append_logits.py` 检查 append/cold，`summarize_stage6.py` 汇总预算、各 phase 时间、MTP 接受率与等待。原源码备份/SHA 在 `stage6-trace-source-backup`，`manage_stage6_trace.py` 的撤回结果已核对；这些诊断辅助工具没有进入生产源码。
