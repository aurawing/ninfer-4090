# KVMem 4090 实施进度

记录方式见 [README](README.zh-CN.md) 第 7 节。每条记录写明日期、提交号，并注明数字是实测还是估算。

## 待用户决定的问题

第 5、6 步（`4da637a4`、`db8e9641`）已由用户审阅通过；第 7 步已提交发布 `685cae7c`，第 8 步已提交发布 **`c4f145e5`**。dense 组合验收三项全部通过，阶段 3 的退出标准已逐条关闭。首轮严格金标准 **8/9** 的失败、后续四次匹配和用户在失败后批准的修订均保留，原金标准不替换。36 次确定性诊断的完整词表逐位差异为 0；生产计划一致、dense 默认1024；两版生产算法各三次的换行分歧和相对 L2 包络按原门禁 B 通过。阶段4设计已按2026-10-02用户批准的D8/D9修订，新增顺序4.1–4.4实施清单；文档与4.1–4.3基础已分别完成门禁并commit/push，4.4 eager接线的规格与质量审阅、终版dense三项组合、72个合成needle与12个参考、12份复用/回滚及最终全量CTest已完成；完整owner synccheck/initcheck各0error。用户于2026-10-06批准停止约26小时的完整owner racecheck并改用内核级小shape门禁，内核级覆盖补缺、独立规格/质量复审和终版115项全量CTest现已完成；4.4实现/测试与验收文档已提交并push至origin/feat/kvmem，当前停在4.4审阅点。Graph/capture性能门禁及真实多文件/工具矩阵仍未验收，详情见本文末尾。

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

- 状态：4.1–4.3 已完成并分别提交；4.4 eager 全部门禁已完成，实现/测试 `9404c865` 与验收文档 `fb0d12ba` 已push，停在4.4审阅点；4.5/Graph与真实质量矩阵未开始。

## 与设计的偏差

| 日期 | 决策编号 | 偏差 | 原因与证据 | 用户是否同意 |
|---|---|---|---|---|
| 2026-09-30 | D3 前提 | 后续分层视图保留原始逻辑页号，不直接以紧凑视图槽号代替位置；阶段 0′ 不实现 | `paged_kv_address.cuh:16` 和 prefill/decode 内核按 `position >> 6` 查块表并以原始 key 位置做掩码，详见上文静态核对 | 是，用户明确选择“保留原始页号” |
| 2026-09-30 | D4 | 默认 `auto`：加载期整块 CUDA 锁页且剩余物理内存 ≥4 GiB 则归档直传，否则释放部分分配后退回可分页归档与 4×64 MiB 中转环；显式 `pinned` 失败即退出，`pageable` 保留旧路径。两者统一「层 → plane → 逻辑页」布局 | 262K 就绪后整块可锁页 15488 MiB；锁页 H2D 9.83 GiB/s，可分页经环端到端 4.19 GiB/s。旧「4090 WDDM 锁页限额更紧」的推断不成立，4060 仍须实测 | 是，用户明确同意修订 |
| 2026-10-01 | D1，用户授权例外 | 仅修复已知的 dense 量化 small-T 注意力共享内存竞争，独立修复分支合入后重录九组金标准，旧数据保留；Q5 和 dense 分派不改 | 原版 racecheck 64 hazards，修复后 0；12 组各 64 次逐位一致且 FP64 oracle 通过，详见修复记录 | 是，本轮用户明确要求现在修复，并限定文件与两处 barrier |
| 2026-10-01 | D1 与门禁 A | 262K RK4 的新路径与当前 dense 在 146432 token 处相对 L2 0.0010094，固定输入消融确认 dense prefill alpha 的 FMA 舍入累积；照搬它会破坏现有 FP64 公共 logit 测试。保持门禁和正确新路径，是否另行修 dense 须用户决定 | 固定输入 dense/partial 重算均逐位一致；临时 alpha-only FMA 将两者误差降到 0.0000645685，但独立数值测试失败，未修改 dense 的均匀参考直接出现大误差/非有限输出，详见上文 | 已决定不修 dense；262K 门禁 A 后续按 FP64 修订见下一行，原失败保留 |
| 2026-10-02 | 门禁 A，D1 保持 | 32K/128K 的 tiered/dense 1e-3 不变；262K RK4 按影子误差选至少三个层（包括原超限层）的末段完整块，采用同一量化 KV 字节的 CPU FP64 oracle，要求逐层 tiered/FP64 相对 L2 不超过 dense/FP64；同时记录 tiered/dense 数值 | **在测得 0.0010094 之后**依据固定输入 alpha 消融作出的事后修订，保留原失败与所有数据；不修改 dense、不放宽任何其他门禁 | 是，用户明确同意；新判据实测通过 |
| 2026-10-02 | D8 | 将CPU/AVX2 scorer改为确定性GPU两遍：每个layer/qhead/query token全局max/denominator，再固定顺序累加页概率，无atomic；主机Mean-K固定stride，archive pinned时额外cacheable pinned索引计入准入并直传，否则pageable经既有环；设备alias空闲staging/partial，运行期不分配，标量FP64仅测试 | 用户批准阶段4架构修订；原50–150ms CPU值只是估算，无GPU实测通过声明。GPU相对FP64 L2≤1e-4，集合一致只允许明确FP64第k阈值epsilon并列；分别记H2D与compute | 是，用户明确同意，批准日期2026-10-02；仅文档已修订，实现未验收 |
| 2026-10-02 | D9 | hard为sink/recent完整页/query span及图像闭包，另计reserve/guard；全部current闭包union若fits `V-G-H`则全部hard，否则较早非hard current页与历史同域评分竞争、图像原子；记录`current_input_softened_pages`，hard溢出拒绝并报告各项页数 | 用户批准容量规则修订，既有单条长input needle现在可合法软化，结构化历史/短query fixture为额外测试；不以文档修订冒充质量实测 | 是，用户明确同意，批准日期2026-10-02；仅文档已修订，实现未验收 |
| 2026-10-02 | D1，阶段 3 dense 退出标准 | 首轮 8/9 匹配后，改为三项组合验收：同一确定性临时补丁下旧版/终版九例各两次完整 logits 逐位一致；dense 加载计划一致；生产算法分歧步通过既有门禁 B 的并列/包络规则。以后各阶段沿用 | **在看到 needle-262k-10 第 9 token 271→198 的失败之后作出的事后修订**。首次失败没有 logits，四次后续匹配不证明根因，原失败和严格汇总断言失败保留，不替换原金标准 | 用户已明确批准；三项都通过才提交第 8 步，诊断差异则定位并停下；dense 永久代码不修改 |
| 2026-10-06 | sanitizer 门禁，D1–D15 不变 | racecheck 改为仅内核级小 shape；owner/e2e 仅完整 synccheck/initcheck，增加实际 kernel/config → 单元测试 racecheck 覆盖对照，有缺口补小 shape | **在完整 owner 插桩约26小时后作出的事后修订**：13 fixture 完成、日志未报告 hazard、停在 fragmented_next_exact，无最终summary，原中止/超时保留；e2e 规模插桩成本不可接受；同一 owner sync/init 已完整0error | 是，2026-10-06 用户明确批准；小 shape 数值与0 hazard判据不放宽 |
| 2026-10-06 | dense 组合门禁第三项，D1 不变 | **审阅决定、措辞澄清，在 4.5 本次结果之后作出**：只判断本轮旧版/终版各三次的九配对，完整词表相对 L2 不超过固定旧版包络；实际输出分歧须满足原门禁 B 的并列规则。后续阶段不重新证明阶段 3 首轮历史换行翻转 | 4.5 六次输出全同，最大相对 L2 `0.4037602120<0.8199316780`；确定性九例已逐位证明数值未变。Q5 求和顺序决定能否观察历史翻转，要求每轮复现会使门禁随机通过或失败；历史翻转在阶段 3 和 4.4 已两次判为并列。原失败凭据原样保留，另存本轮范围判定，无模型重跑 | 是，2026-10-06 用户明确要求；不放宽包络、实际分歧并列规则或其他两项门禁 |

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

## 阶段 3 第 7 步：准入与 CLI（2026-10-02）

基于 `db8e9641`。CLI/serve 共用 kvmem 参数语义：解析时拒绝 `--kv-mode kvmem`，说明阶段 4 Mean-K/选块/稀疏 decode 未完成；`--kvmem-prefill` 只接受 exact，window 明确拒绝。serve 的 tiered C>1 在启动前拒绝。Engine API 仍有相应目标校验。

增加 `--kvmem-lock-archive`：明确要求 pageable 传输，auto 在该开关下选择 pageable，pinned 冲突报错；加载期整份提交、提高 Windows 工作集并 VirtualLock，trim 保留全份锁定容量，析构/异常释放并恢复原工作集。它不是 cudaMallocHost 的 pinned 模式，仍用既有四槽环和两线程。默认普通 pageable 的按需提交/trim decommit 不变，运行期不新增锁页申请。主机物理准入与 pinned 失败消息分别给出 pageable、max-context 和更小 KV 的建议，同时说明 pageable 不能绕过容量加 4 GiB 的物理准入。

日志补全实际 Main view tokens/payload、staging、partial、访问列表与其他固定元数据、一般 workspace、MTP 窗口/pool、请求归档模式/容量；归档实际选择、OS lock 和加载后可用物理内存/4 GiB 余量单独打印。新增使用者文档 `tiered-exact.zh-CN.md`，改写 maintainer 的 offload non-goal，明确 dense 契约不变、C=1、BF16 仅功能、Graph/disk 关闭、decode 仅验证。

实测验证（仓库外 `D:\deeplearning\NInfer\logs\kvmem-stage7-8`）：选项 RED 两项均因缺失功能失败，GREEN 2/2 通过；完整构建退出 0（298 更新动作），全量 **106 项：102 通过、4 制品相关跳过、0 失败，251.04 s**。额外补强 parsed lock/order 和准入提示测试后，三项定向 CTest **3/3，0.76 s**，覆盖 pageable/auto OS lock、逐字节往返、回写、trim 保留锁定提交以及普通 pageable decommit。原四项 skip 与前一步一致，没有把跳过计为执行通过。构建/RED/GREEN/full CTest 原始日志均保留；MSVC 的既有 /Ob2→/Ob3 warning 不影响退出码。独立规格与质量审阅均 approve。dense 内核与分派无改动，九组 dense 金标准按第 8 步在终版重跑。


## 阶段 3 第 8 步：差异恢复与收尾（2026-10-02，dense 组合门禁通过）

普通 tiered 自动分块先创建 2048 的完整 planner，包含 Main/MTP、workspace、staging、partial、列表与预算预留；仅容量相关 `TieredPrefillCapacityError` 回退 1024。显式 CLI/serve 值和 API 非默认值原样使用；API 显式 1024 可设置 `prefill_chunk_explicit=true`。dense 与 shadow 保持原默认 1024 和预算流程。启动打印实际选择及原因，LoadSummary 记录实际分块。

Main 恢复按当前 Both 页与 slot owner 保留仍有效的目标前缀原物理槽，包括末页有效前缀；只为被覆盖或驱逐的目标页生成 hydration 列表。连续逻辑范围合并传输、设备非连续 lease 按真实 ID 分段复制；drain/归档 trim、generation 递增、旧 ticket 拒绝及原子发布保持。`[kvmem-restore]` 输出保留/缺失页、**实际调度 H2D 字节**和恢复耗时；该字节不是 PCIe 硬件计数器。MTP 缺页仍不补算，磁盘缓存和 CUDA Graph 仍关闭。

源/测试从 `685cae7c` 开发，无 dense 内核/attention 分派改动。接口 RED 因不存在新 API 失败；运行时有效 RED 为 `stage8-red-runtime-test-r2.log` 的“surviving current partial page restore must schedule zero hydration”。初次测试 fixture 的 narrowing 与 pool entitlement/bad allocation 问题也保留记录，不算功能 RED。MSVC 在新 planner 转移所有权时暴露原默认 move ctor 缺链接符号（27B/35B）；仅改为显式移动 unique_ptr 的实现，增加 moved-from/moved-to 生命周期断言，不改 dense 算法。

目标 GREEN **8/8、37.08 s**，覆盖 pinned/pageable、全部 16 层各 plane 有效前缀、非连续 lease、覆盖页重新换入、重复零 H2D、generation/stale plan，以及完整预算边界和不吞无关错误。完整构建 **189 更新动作，退出 0**；全量 CTest **106 项：102 通过、4 对应制品缺失跳过、0 失败，244.53 s**。日志为仓库外 `D:/deeplearning/NInfer/logs/kvmem-stage7-8/stage8-{full-build,full-ctest,green-target-test}.log`。规格、代码质量独立只读审阅通过；整模型测量结果随后补充，不以 CTest 代替质量/性能门禁。


### 第 8 步：262K 差异恢复配对实测

对照使用第 6 步正式聊天 fixture `kvmem-stage5-6/chat-262k-reference-user.txt`（SHA256 `6effaa5e1de5b1924b95b3d0c006c2fb2b804cc6dd8fe2ce693a5f3deec0436b`），262144 容量、INT8-G64、MTP-3 optimized、chunk 2048、无视觉/Graph/磁盘缓存、greedy、preserve_thinking=true。before 正式数据为 `stage8-before-262k-r3.*`，after 为 `stage8-after-262k.*`；两者串行且无重构建/其他 GPU 测试重叠。此前 r1 的 fixture 文件名错误和 r2 与构建重叠的探索记录保留，**不用于正式性能结论**。外部静态链接 harness 仅用 Engine API，before/after SHA256 分别为 `d24a890fb74a5a06b4c82ea0c4188fe0df47dbf6d122095cafe8fdc89344978f` / `133338b062640781096ed8be3c8cb9d3353e0aa163e38dde77ec0f99da8ea284`。

两轮均为 first 261447 token、append/rollback/cold 261513 token；append 复用 261462、计算 51，rollback 复用 261509、计算 4。after 四次 needle 均正确；rollback/cold 全部 64 token 相同，before/after 四阶段 ID 也相同。本步未再导出完整 logits，不能将 token 相同声称为 logits 逐位相同；此前第 6 步完整词表三次包络证据仍保留。MTP first accepted/drafted 11/11，其余阶段各 47/47，全部 100%。

| 恢复点 | prefill before s | prefill after s | 完整调用 before s | 完整调用 after s | 调度 H2D before B | 调度 H2D after B | after 保留/缺失页 | after Main restore ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| retained append | 1.666338 | 1.207654 | 10.573266 | 9.623647 | 3933929472 | 0 | 1819 / 0 | 0.3990 |
| turn checkpoint | 0.962624 | 0.546770 | 9.424440 | 9.397402 | 3933929472 | 2162688 | 1818 / 1 | 5.8729 |

before 字节按旧版“全部驻留页×全部层/plane”的实际规划及旧无条件 hydrate 调用计算；after 字节由实际调度计数日志取得。两者均是**调度传输量而非硬件 PCIe counter**。3933929472 B≈3.664 GiB；新 rollback 2162688 B=2.0625 MiB。完整调用用相同外部 marker 观察器计时，轮询间隔 5 ms，包含 prepare/generate 与输出解码，不只恢复；after Main restore_ms 是内部计时，旧版无相同内部计数，不能把旧完整调用与新 restore_ms 直接比。append 完整调用约降 8.98%，rollback 仅约降 0.29%；该次 rollback decode 8.618 s 高于 before，抵消 prefill 收益。一次配对不是稳定性能保证；first prefill 222.768→209.451 s、cold 222.973→223.826 s 的波动不归因于恢复优化。

两轮实际预算完全一致：Main view **116416 token / 3933929472 B**；staging **374886400 B**；partial **101449728 B**；访问列表/前缀/块表及对齐 **164608 B**；general workspace **361906176 B**；MTP window32768 pool **69206016 B**；CUDA pinned host archive **8858370048 B**，中转环0 B。这不含模型权重、桌面或整卡运行峰值。ready GPU 累计等待分 phase（四次调用累计，非单次或传输隐藏比例）：before prefill/decode **2326.4165 / 24645.5400 ms**，after **2210.3227 / 24707.0700 ms**，逐层日志与摘要 `stage8-reuse-summary.json` 保留。

真实产品 CLI 自动选择验证也通过：32K 配置、view8192、无显式 chunk 选 **2048/automatic**；view2048 选 **1024/fallback**，明确日志 `tiered view cannot fit sink, prefill chunk and replacement page`，两次返回0。容量边界与显式值拒绝/保持规则另外由预算 CTest 覆盖。恢复 GPU 测试补跑 compute-sanitizer **13.0.85**：racecheck **0 hazards/0 errors/0 warnings**，synccheck/initcheck **0 errors**，全部退出0。命令：`compute-sanitizer.exe --tool racecheck|synccheck|initcheck --error-exitcode 99 build-vision-integration/tests/ninfer_test_tiered_runtime.exe`（三个独立串行进程）；原始日志 `stage8-restore-*.log`、精确命令 `stage8-restore-sanitizers.json`，均在仓库外。


### 第 8 步：262K MTP 长生成（1024 token）

同一固定摘要输入，262144 容量、INT8-G64归档、chunk2048、MTP-3 optimized、greedy、无视觉/Graph/磁盘缓存，两个独立进程顺序测量。任务为120个合成项目备忘录的交付/库存/复核/风险/建议摘要；实际 prompt **260768 token**，各生成完整 **1024 token**，内容连贯、无EOS后填充。测量请求关闭默认stop并按固定token预算截断；任务要求至少1500词以避免自然提前结束。decode速率定义为1024/result.timings.decode_seconds。

首次fixture为261280 token、只剩864输出预算，外部harness在prefill之前拒绝并返回2；该失败 `stage8-long-32768.*`、原fixture与hash均保留。只额外移除512个filler token，120备忘录与问题不变，正式两轮使用同一 `summary-262k-user-v2.txt`，SHA256 `5a2902ead7b4c0db9a049f716984a6796bfcfd58254c81282d657e39959c6675`；校准/源fixture记录 `long-generation-fixture-v2.json`。该校准失败不计成1024生成或接受率通过，也不覆盖旧记录。

| MTP物理窗口 | Main视图token | accepted/drafted | 接受率 | prefill s | decode s | decode tok/s | ready prefill ms | ready decode ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 32768 | 116416 | 659/1089 | 60.514233% | 222.422207 | 186.987679 | 5.476297 | 565.521312 | 169722.7000 |
| 262144（完整窗口） | 101120 | 648/1121 | 57.805531% | 222.622604 | 195.262946 | 5.244211 | 665.412580 | 177692.9000 |

接受率下降定义为完整窗口减32768窗口：**−2.708702百分点≤5**，本轮长度/接受率门禁通过。一次对照不能声称短窗口稳定更好；两次target验证批/接受轨迹不同，未定位造成输出分歧的原因。**两次1024生成ID并非完全一致**：共同前缀54token，第55个token（0基54）32768为**561**、完整窗口为**19592**，之后生成轨迹不同。本轮未抓这一用例的完整logits，不能证明并列、包络通过或归因于Q5，也没有MTP-off的1024对照；不将接受率通过扩写成长生成贪心一致性通过。第5/6步已通过的64-token贪心/包络结果只对应既有needle/聊天fixture，不证明本摘要输入前64 token一致。本摘要输入的贪心一致性未验证，且已在第55 token观察到两窗口分歧，作为审阅项保留；不在本轮调整D11或数值判据。

32768的预算与上述复用实测相同；完整窗口Main **3417047040 B**，staging **407191552 B**，MTP **553648128 B**，partial **101449728 B**，metadata **164608 B**，general workspace **361906176 B**，host archive仍 **8858370048 B**。完整窗口4096物理页=4sink+4092recent（无需单独guard页）；32768为4sink+507recent+1guard，共512页。统一预算确实将短MTP的空间给Main；旧完整dense MTP **553783296 B**的一个额外页不能与完整window4096页混为一谈。

正式数据 `stage8-long-32768-r2.*` / `stage8-long-262144.*`；摘要 `stage8-long-generation-summary.json`，包含首分歧、命令/二进制/fixtureSHA及NVML采样。外部harness **stage8-long-generation.exe SHA256 d69fe15a783d309bcd4e082442443b07bc725adf7431aaeb0d51929ad37de7af**，只有Engine公开API与fixture容量预检，源码/构建日志均在仓库外，不提交临时测量代码。ready累计GPU事件时间按prefill/decode分开，是一层层等待之和，不等同于端到端停顿或PCIe传输隐藏比例。Nsight未补测：此前计数器权限限制未解除，不以NVML替代tensor/occupancy profile。

### 第 8 步：终版 dense 九例重录（门禁暂未通过）

原金标准是 `1fff2bb6` 提交留存、修复合入后的 `b68009f7` 未插桩实测，保留在仓库外 `kvmem-stage3-attention/baseline`；本轮新目录 `kvmem-stage7-8/dense-final` 不覆盖旧数据。终版产品二进制 SHA256 **fb00037aae6c83d21e4ef561e1edba45f8f42c7007a937267f53fd158e7754e0**，全部用原 fixture 和命令，rk4v4-e8、chunk1024、MTP-3、greedy、CPU vision 参数，CUDA Graph 沿用金标准默认值；关闭临时诊断环境变量。每项记录输入/金标准/二进制哈希、命令、源码 patch 身份、整卡显存和原始 stdout/stderr。

| 用例 | prefill s | decode tok/s | 金标准 token 比较 | needle |
|---|---:|---:|---|---|
| synthetic-32k | 14.958 | 149.58 | 64/64 相同 | — |
| needle-32k-10 | 15.200 | 144.45 | 64/64 相同 | 正确 |
| needle-32k-90 | 14.959 | 147.87 | 64/64 相同 | 正确 |
| synthetic-128k | 81.557 | 131.10 | 64/64 相同 | — |
| needle-128k-10 | 90.408 | 117.17 | 64/64 相同 | 正确 |
| needle-128k-90 | 91.011 | 118.64 | 64/64 相同 | 正确 |
| synthetic-262k | 244.387 | 103.27 | 64/64 相同 | — |
| needle-262k-10 | 244.889 | 102.91 | 第 9 token 271→198，其余 63 相同 | 正确 |
| needle-262k-90 | 231.552 | 119.39 | 64/64 相同 | 正确 |

九项都退出 0、生成 64 token，MTP 接受率均 100%，六项 needle 均正确，但这些不能代替 **9/9 金标准一致**。第 9 token（0 基 step8）的实际文本是答案后的双换行变为单换行，后续 token 再次一致。当前没有该失败运行的 logits，不能宣称门禁 B 并列或包络通过。历史 `kvmem-stage3-attention/diagnostic/needle-262k-10.top8.json` 的另一运行在此步 top-2 是 271/198、logits23.75/23.625，相差 0.125（该量级一个 BF16 ulp）；旧诊断禁用 Graph 并有读回同步，**不能将其作为本次分歧的 logits 或证实原因**。

性能限制：本轮较旧 dense prefill 数字偏慢，测量期间曾观察到 GPU 软件温度降频标志 Active。仓库外 `dense-thermal-observation.json` 保存了 86°C、2535 MHz、388.55 W 的时间点采样；没有同时间序列的旧基线温度数据，不能将所有差异归因于降频，也不能据本次时间宣称 dense 性能不变。未调整风扇、频率、功率或桌面设置。

源代码 `src/ops` 无 diff。复测采用原第 6 步二进制 **093c014acb6899cc3628f1970ac97e64aad7f278da0e8e2e817eaaf00d791adf** 与终版交替各两次，相同 fixture/参数，目录 `dense-repeat-control`。

| 交替次序 | 二进制 | prefill s | decode tok/s | 64 token 与金标准 |
|---|---|---:|---:|---|
| 1 | 第 6 步原版 | 206.852 | 120.47 | 完全一致 |
| 2 | 第 8 步终版 | 210.235 | 120.35 | 完全一致 |
| 3 | 第 6 步原版 | 212.606 | 120.21 | 完全一致 |
| 4 | 第 8 步终版 | 217.006 | 119.63 | 完全一致 |

四次均退出 0、needle 正确、MTP 接受率 100%。这证明同一终版产品二进制在相同参数下有一次分歧、两次匹配，不证明原版也会出现分歧，或本次唯一原因就是 Q5；原版仅复测两次不足以排除第 8 步改动对运行时序/非确定性分布的影响。测试未增加插桩，均未抓分歧步 logits。时间顺序及温度不同，此四次不能作为独立的密集性能回归通过证明。

门禁摘要 `dense-final-gate-result.json` 明确记录 `dense_exit_passed=false`；严格聚合脚本在金标准断言处退出 1，保留 `stage8-summary-gate-failure.log`，没有绕过断言或生成全通过摘要。在这一项解决前，第 8 步不提交/push，阶段 3 不宣告完成；阶段 4 草稿仅保存在仓库外 `stage4-sparse-decode-design.draft.zh-CN.md`，已核对参考 commit/许可、实际接口和 JSONL 格式，尚未安装或编写代码。

### 用户批准后的 dense 组合补验收（2026-10-02，三项均通过）

以上 `dense_exit_passed=false` 和首次 8/9 是原严格判据的历史失败，保留不覆盖。用户随后明确批准三项组合判据，README 已写入以后各阶段沿用的要求；这是看到失败后的事后修订，不是原标准已经通过。

**构建身份与恢复。** 旧版是实际录制金标准的 `b68009f7952741e4d52fa538de51fcd769cf96a3`，终版是父提交 `685cae7c` 加当前第 8 步实现。诊断两边均仅将 Q5 `launch_residual_exact` 的 `kSplits` 从 2 改为 1，并加入相同的完整 logits 读回和计划日志；原 INT8 同步修复本来就在两版中，不重复修改。历史上修复后的 512-token 确定性试验也只需这项 Q5 补丁。两版诊断补丁逻辑 SHA256 均为 `5f044c89afacd7f778434ee12071d4bffbead2a4d25747710855e4be982cd4e1`；诊断旧/终版二进制 SHA256 分别为 `b47e1e9aa57738e622d6deac02e1ebc43394745cfe1950c60d76166a62ad70a6` / `f1c56dd9549ffe59512e470f44622724c4351a97f8eb736803a3a6c2235d39d0`。另存生产算法读回构建，Q5 仍为 2、CUDA Graph 默认开启，无确定性数值修改；两边读回补丁逻辑 SHA 相同。

所有构建、临时补丁、1043 文件的原始字节与 SHA、实际命令和数据都在仓库外 `D:\deeplearning\NInfer\logs\kvmem-stage7-8\dense-composite-gate`。临时源码已撤回，`last-restoration-sha256.json` 核对 **1043/1043 原字节恢复、0 mismatch、临时头文件不存在**，之后才改动本次验收文档。恢复后的产品完整构建退出 **0**（388 个更新动作），全量 CTest **106 项：102 通过、4 缺其他模型制品跳过、0 失败，247.20 s**，日志 `stage8-composite-restored-full-{build,ctest}.log`。构建过程中一次哈希报告工具缺失、一次 CRLF/LF 导致 GGML 适配依赖校验拒绝的失败均保留；后者经确认内容相同后保留已有 checkout 字节重新构建，没有绕过依赖 SHA 校验。

**执行顺序。** 先九例各版各两次（36 个串行运行），每步比较有效完整词表 248077 个 BF16 原值，物理填充 243 项单独记录；任一重复或跨版本差异立即停止并定位。其次核对九组实际 dense 计划、Graph 准备路径与 split 输入，并额外移除显式分块参数确认默认 1024。最后旧/终版生产算法各三次复现分歧用例，抓取完整第 9 步 logits，按既定门禁 B 公式判定。当前尚无三项通过结论，不提交第 8 步或阶段 4 草稿。

**阶段性数据（未完成全部门禁）。** 32K 与 128K 各 synthetic、needle-10、needle-90 共六例，已完成每版两次、共 24 次诊断运行。每例旧版重复、终版重复、两组跨版本比较的有效词表差异均为 **0/15876928**，物理填充差异也为 0；六例实际计划与有序分派日志相同，64 token 全部匹配原金标准，needle 正确。`diagnostic-progress-32k-128k.json` 保存此阶段性快照，仍需 262K 三例、生产计划与并列门禁。另有 `fixture-identity-check.json` 证明九份 fixture SHA 与原金标准记录全部一致；文档修订后的 `post-doc-source-sha-check.json` 确认 **1008 个非文档文件原始字节不变**，临时头文件不存在。launch 策略核对明确区分实际 host split capacity 日志和按相同策略/输入推导的 device active split 范围，不声称取得硬件 profile。

**门禁 1 已完成，通过。** 九例 × 两版 × 两次，共 **36 次独立诊断运行**。每例都逐步比较 64 个生成位置的 248077 项有效词表 BF16 原值；下面的“跨版本”包含第 1/2 次各一组配对，填充 243 项另检，全部为零差异。36 次的 token ID 均匹配原未插桩金标准，六个 needle 每次都答对。诊断用 Q5 单分片、无 CUDA Graph，不作为产品性能数据。

| 用例 | 旧版重复差异项 | 终版重复差异项 | 跨版本两组差异项 | 计划与有序分派日志 |
|---|---:|---:|---:|---|
| needle-32k-10 | 0 | 0 | 0 / 0 | 相同 |
| needle-32k-90 | 0 | 0 | 0 / 0 | 相同 |
| synthetic-32k | 0 | 0 | 0 / 0 | 相同 |
| needle-128k-10 | 0 | 0 | 0 / 0 | 相同 |
| needle-128k-90 | 0 | 0 | 0 / 0 | 相同 |
| synthetic-128k | 0 | 0 | 0 / 0 | 相同 |
| needle-262k-10 | 0 | 0 | 0 / 0 | 相同 |
| needle-262k-90 | 0 | 0 | 0 / 0 | 相同 |
| synthetic-262k | 0 | 0 | 0 / 0 | 相同 |

证据：`dense-composite-gate/diagnostic-progress.json`、`diagnostic-run.log` 及 `diagnostic/` 下完整 BF16、JSONL、命令、SHA 和原始日志；`dense-launch-policy-comparison.json` 已覆盖九例。量化 prefill 为 `grid=(ceil(T/64),24,1)`、512 threads、92672 bytes shared memory、无 split-K；small-T 比较实际 host capacity/执行 envelope，再按两版相同的 Q24/KV4、split 上限 64 与固定 SM128 策略推导活动 split 范围。未取得 GPU profile，不把推导当设备读回。门禁 2、3 的后续完整结果见下文。

**门禁 2 已完成，通过。** 保持生产 Q5 双分片和原默认 CUDA Graph，用相同临时只读计划日志构建，在模型构造和 Graph 准备后退出，九组原配置各版一次，另外移除 synthetic-32k 的显式分块参数各版一次，共 **20 次启动**。全部整数预算、Main/MTP 每个 plane 的类型/shape/offset/bytes、block table、持久状态、七种 workspace recipe、内核 route/envelope/host split capacity 与 Graph 准备时的有序分派记录完全一致。额外默认检查两版实际 `prefill_chunk=1024`。永久代码的自动 2048 选择受 `normal_tiered` 条件保护，dense 未进入该选择器。九例实际 prefill/decode 的路径和 split 输入另由门禁 1 的运行日志验证；Graph 日志代表准备/捕获调用，不声称每次 replay 的设备 trace。

| dense 上下文 | Main payload bytes / 物理页 | MTP payload bytes / 物理页 | workspace bytes | Graph allowance bytes | runtime reservation bytes |
|---|---:|---:|---:|---:|---:|
| 32768 | 570425344 / 512 | 35721216 / 513 | 180969472 | 268435456 | 1395721472 |
| 131072 | 2281701376 / 2048 | 142675968 / 2049 | 180969472 | 268435456 | 3213964544 |
| 262144 | 4563402752 / 4096 | 285282304 / 4097 | 180969472 | 268435456 | 5638288640 |

表中两版相同，reservation 是模型权重之外的执行预算，不是整卡显存。每档三个用例一致；各 pool metadata 分别为 2048、8192、16384 bytes。证据 `dense-plan-runtime-comparison.json`、`production-plans/`、`plan-comparison-run.log` 与 `dense-plan-static-audit.{md,json}`。门禁 3 的后续完整结果见下文。

**门禁 3 的方法核对（六次生产运行未完成时记录）。** 收集脚本最初额外把“六次运行的所有候选差距都不超过旧版差距波动”作为聚合条件；这不是既有 `kvmem-stage4-runtime/check_gate_b.py` 的规则。正式验算器 `evaluate_production_gate_b.py` 逐配对沿用原规则：在实际分歧步检查该次参考 dense 的 top-1/top-2 差距是否不超过同前缀参考 dense 各次差距的 `max-min`，并在共同输入前缀（含分歧预测）逐步检查相对 L2 包络。最终版本的差距不加入参考范围，不增大参考包络；收集脚本的额外统计保留并注明不用于门禁。`gate-b-method-correction.json` 记录检查时序、原门禁脚本 SHA 与正式验算器 SHA，这次是撤回未要求的附加条件，原门禁 B 阈值和分歧规则未变。

旧版生产首轮复现同一变化：仅第 9 token **271→198**，其余 63 token 与金标准相同，needle 正确；该步 271/198 的 BF16 logits 均为 **23.75**，差距 **0**。第二轮选 271，分别为 **23.75/23.625**，差距 **0.125**。这些是补测观察，不回填首次终版失败缺失的 logits，也不能仅凭它们断言唯一波动来源。六次完整结果如下。

**门禁 3 已完成，通过。** 两版各三次，Q5 仍为双分片、默认 Graph，只有临时只读日志/完整 logits D2H；不是未插桩性能运行。六次均生成 64 token、needle 正确、MTP 接受率100%；两版各两次选271、一次选198，仅这一步与金标准有变化。greedy `argmax_better()` 对数值相等时选较小 token ID，因此 271/198 精确并列时选择198，规则本身没有改变。

| 版本 / 次数 | 第9 token | logit(271) | logit(198) | top-1/top-2差距 |
|---|---:|---:|---:|---:|
| 旧版1 | 198 | 23.750 | 23.750 | 0 |
| 旧版2 | 271 | 23.750 | 23.625 | 0.125 |
| 旧版3 | 271 | 23.750 | 23.625 | 0.125 |
| 终版1 | 271 | 23.750 | 23.625 | 0.125 |
| 终版2 | 198 | 23.625 | 23.625 | 0 |
| 终版3 | 271 | 23.625 | 23.500 | 0.125 |

旧版同前缀差距波动 `max-min=0.125`，每个实际分歧配对的参考 dense 差距为0或0.125，全部满足既定并列规则。旧版三次两两完整词表最大相对 L2 **0.3180724314**，包络 **0.6361448628**；旧/终版九配对最大相对 L2 **0.3735287798**，均通过。第9步本身九配对最大相对 L2 **0.0506294616**。共同前缀为8时仍比较第9步预测，因为其输入前缀一致；之后不同轨迹不比较。

| 旧版 / 终版 | 共同生成前缀 | 共同输入上的最大相对L2 | 包络 / 并列 |
|---|---:|---:|---|
| 1 / 1 | 8 | 0.0903826434 | 通过 / 通过 |
| 1 / 2 | 64 | 0.3127665575 | 通过 / 无分歧 |
| 1 / 3 | 8 | 0.1740525604 | 通过 / 通过 |
| 2 / 1 | 64 | 0.3153209565 | 通过 / 无分歧 |
| 2 / 2 | 8 | 0.0784037083 | 通过 / 通过 |
| 2 / 3 | 64 | 0.3589242144 | 通过 / 无分歧 |
| 3 / 1 | 64 | 0.3735287798 | 通过 / 无分歧 |
| 3 / 2 | 8 | 0.1719530001 | 通过 / 通过 |
| 3 / 3 | 64 | 0.3078849435 | 通过 / 无分歧 |

翻转频率只作补充记录：两版各 **198=1/3、271=2/3**，样本不足以估计稳定概率。单次约237–265 s，未追加至每版10次；需要再跑14次约一小时，用户将此项设为短耗时情况下的可选记录。正式门禁由 `evaluate_production_gate_b.py` 验算，`production-gate-b-summary.json` 保存全部逐步L2与候选；收集脚本附加的全六次差距统计在本组**也通过**，修正验算器不影响本组结论。原始完整logits、命令/输入/二进制SHA、stdout/stderr、许可事件在 `production/`。

**组合结论与源码恢复。** `dense-combined-gate-summary.json` 三项均为true，同时仍记录原首轮严格标准 `dense_exit_passed=false`；不是改写旧失败。九例原fixture SHA不变，临时补丁未提交，1043原文件撤回核对后再次确认1008非文档文件零字节变化、临时头不存在。恢复后产品二进制SHA256为 `295a194dc1bb4b13c43cdfa4f04e3ba3207f5d19a3f8c59212ae8e4d76993372`，与测量二进制的不同构建身份分别保存；实现源码未变，full build/106项CTest的证据及SHA都绑定到恢复后的源码。阶段3退出标准已逐条关闭；追加恢复调度传输 **0 bytes**、回滚 **2162688 bytes=2.0625 MiB**，32768/262144 MTP接受率 **60.514233%/57.805531%**、decode **5.476297/5.244211 tok/s**，长摘要第55token分歧且未额外抓取logits的限制仍保留。第8步提交发布后，阶段4只提交设计文档并停下等审阅。

## 阶段 3 收尾发布与阶段 4 文档审阅点

第 8 步 **`c4f145e565eab32fb20a0cac38265e2c37211a30`**（`perf(kvmem): plan tiered chunks and restore missing pages`）已推送 `origin/feat/kvmem`，包括实现、CPU/GPU回归测试、全部验收证据索引和用户批准的 dense 组合判据。测试为106项：102通过、4制品缺失跳过、0失败；代码没有临时读回/确定性补丁，也没有 dense 内核/分派改动。

阶段4草案已安装到 [stage4-sparse-decode-design.zh-CN.md](stage4-sparse-decode-design.zh-CN.md)，原安装为独立纯文档提交 `a78b533a`。当时内容包括两线程CPU全局softmax、整段current hard、图像引用检测提案及动态Graph契约；这些是原草案记录，不再作为后续实施契约，D8/D9修订与本轮边界见下文。实际源码入口行号807/839、参考六个文件的blob及Apache许可已复核；原JSONL示例经解析验证。原CPU150ms只是软目标估算，未测量；真实工具语料仍需用户提供，阶段4实现/性能/质量未验收。

## 阶段4文档修订与4.1–4.4实施入口

2026-10-02用户批准D8/D9修订（执行日期进入2026-10-03，批准日期不改）：GPU确定性两遍评分取代CPU scorer；固定stride主机Mean-K按归档类型选择pinned直传或pageable经现有环，额外索引字节计入准入，设备评分alias空闲staging/partial；D9允许较早current页容量软化并与历史共同竞争，基础hard与图像闭包仍超预算拒绝。对应偏差表已记录，README与设计5.6.5/5.6.6同步。

指定本地参考 `D:/deeplearning/NInfer/logs/kvmem-stage7-8/kvmem-qw3-reference` 的HEAD复核为 `1cf3b2f83bfc071ada9c57491a7d121723051ac0`。`src/qwen_executor.cpp`约23750–23790只在`nb>budget>0`且中间非空时mask sink/recent bands，`QW3_KVMEM_MASK_KEPT=0`恢复全页；评分kernel位于`src/kernels_cuda.cu`约5340。本项目默认保留仅两band语义，其他hard页仍参与denominator，软化current不另行排除；隐藏`NINFER_KVMEM_SCORE_ALL_PAGES=1`测试全部已提交页。非页齐hard recent覆盖可比固定recent band多一页。首版无历史图像引用检测，历史图像原子候选、本轮图像依D9，协议识别后续。这两项为设计审阅意见，不新增D表决策。

新增 [stage4-sparse-implementation-plan.zh-CN.md](stage4-sparse-implementation-plan.zh-CN.md)：全部文档先于实现，4.1 Mean-K/FP64/部分prefix、4.2 GPU score/CPU selector、4.3 query span/三槽Q/Main snapshot、4.4完整exact-prefill到sparse eager。每步新CTest、当时全量CTest、progress和证据完成后，由主任务分别commit/push；4.4终版对新增GPU owner/kernel跑 compute-sanitizer 13.0.85 的 racecheck/synccheck/initcheck，并保持三项dense组合门禁。4.4仅C=1/Graph=off，完成后停审，Graph及ordinary/MTP capture性能门禁下一轮。

4.4合成矩阵固定为128K/262K contexts×128K/32K views×至少3位置×2 denominators×3次独立运行；既有单条长input needle依D9合法soften，另加结构化历史/短query fixture，记录页统计、TTFT分解、H2D/compute、hydrate bytes、decode与MTP。128K视图对dense rk4，32K对tiered-exact INT8。文档提交时尚未开始阶段4代码/构建/GPU验收，后续实施进度见下文。**真实多文件/工具矩阵未验收，Graph/capture性能门禁未验收，合成测试不能替代。**

### 4.1 算子与主机索引基础验收（2026-10-03）

- 文档修订 `8e5f0b59` 已独立提交并推送到 `origin/feat/kvmem`。实际代码基线仍为阶段3终版 `c4f145e5`；当前先实现 Mean-K GPU 数学和主机索引 continuation，frontend/运行时接线按后续4.3、4.4执行，产品入口暂未解除拒绝。
- 实施前静态核对：`qn/kn` 分别为连续 BF16 `[256,24,T]` / `[256,4,T]`，捕获位于 RMSNorm 后、原地 RoPE 前。普通解码也使用 `Phase::Verify`，故不得由该枚举决定 provisional；必须使用显式事务种类，并按 Program 最终 Main accepted frontier 提交。ledger 的待生成 token 不进入索引；全接受的 verify 也必须显式提交，不能依赖 trim 才提交。多页 GPU 累加的旧尾页 seed 与新尾页 sum 输出必须分离，避免不同 CTA 共享地址读写竞争。
- 后续4.4需要扩展现有视图的任意 selection 规划与保护驱逐，并补传输 poison 恢复：现有传输 worker 错误会停止，`reset()` 经 `synchronize()` 仍重抛错误，不能直接当作 DMA 失败恢复。静态研究记录在仓库外 `D:/deeplearning/NInfer/logs/kvmem-stage4-sparse/runtime-research.md`；这不是运行验收结果。
- 新数据和原始日志目录为仓库外 `D:/deeplearning/NInfer/logs/kvmem-stage4-sparse`。已保存阶段3既有 `src/ops` 的 SHA-256 清单，供最终核对 dense 内核与分派未修改；所有既有数据保留。
- 待测合成输入清单已冻结到该目录的 `synthetic-matrix-inputs.json`：6份单条长user输入、72次规划运行，插入位置10%/50%/90%。10%和90%沿用原fixture字节及SHA，50%使用相同生成方法；终版frontend仍须核对各输入实际token数。**尚未执行这些稀疏运行**，结构化历史加本轮query样例另行补充。
- 新增 `test_meank_accumulate` 的初始RED：构建exit0、CTest exit8；后续“trim后未来snapshot仍有效”的回归RED也保留。独立FP64→FP16 RNE oracle覆盖1/63/64/65/129、跨chunk、偶/奇halfway、8次逐位重复、非阻塞流；owner测试包含两份拥有型snapshot、16层统一发布、部分45→48→46重放、旧/foreign/重复ticket拒绝、provisional接受0/2/16及63→79完整跨页接受，后者不依赖generic trim。
- 两轮全量构建/CTest均exit0，107项：103通过、4跳过、0失败，分别259.45 s与239.56 s。跳过为缺少Qwen3.6-27B专用prefix制品、35B-A3B制品及其DFlash制品，未用Qwen3.8-27B替代。终版代码质量与规格审阅通过，建议后续增加各层不同数值及带符号/消去敏感输入。既有313个`src/ops`文件SHA逐字节不变；这不是4.4终版dense组合门禁。
- **提交前首次sanitizer失败保留**：`4.1-final-v1-racecheck.log`为0 hazard，synccheck为0 error；initcheck exit99、4032 errors。错误为测试夹具`provisional()`的cudaMemcpy D2H source：读131072 bytes全tail，但只写2048 bytes一个有效tail token；余下129024 bytes恰为4032个32-byte未初始化段。修正为仅回读有效prefix，另外已初始化的整tail保留/拒绝后缀检查不删除，没有增加全缓冲清零来隐藏错误，生产算子未改。定向测试及两项独立审阅复核均通过。
- **终版门禁**：`full_checks.ps1 -Stage 4.1-final-v3`构建exit0（无待编译文件）、全量CTest exit0，107项中103通过、4制品缺失跳过、0失败，233.94 s。`run_sanitizers.py --stage 4.1-final-v3 --test D:/VSCodeProjects/ninfer-4090/build-vision-integration/tests/test_meank_accumulate.exe`使用compute-sanitizer 13.0.85，依次以`--tool racecheck|synccheck|initcheck --error-exitcode 99`执行，三项均exit0，0 hazard/0 error；终版测试二进制SHA256 `c08630741654775df875ea6e259b109e363e0df96469b744754f6176766370a6`。原始日志、完整命令、8个新源码SHA、现有313文件零变更核对及测试数量在`4.1-final-evidence.json`。
- 262K、L16/H4/D256、最大分块2048的**布局计算**：固定主机索引128 MiB；设备active/base/seed/output各64 KiB，共256 KiB；BF16 tail2 MiB、provisional512 KiB；最大33页FP16 patch1.03125 MiB、counts64 bytes。主机active/base128 KiB、两份snapshot最大128 KiB、待发布mean/sum patch1.09375 MiB；准入helper合并这些主机payload与4 GiB余量，4 GiB只应在统一准入时计一次。§8账本已同步。该数值不是生产模型的实际加载分配；统一Main加载准入、DMA协调器、模型pre-RoPE hook及最终accepted提交在4.3–4.4接线，`kvmem`入口当前仍拒绝，不把本步独立算子验收写成端到端完成。
- 4.1独立提交 `8267badd`（`feat(kvmem): add exact Mean-K accumulation and prefix restore`）已推送到`origin/feat/kvmem`；发布时HEAD与origin差异0/0。随后进入4.2，新增GPU scorer、纯CPU selector及索引传输基础，尚未执行模型稀疏验收。

### 4.2 GPU评分、纯CPU选块与索引传输基础验收（2026-10-03）

- 新增闭合数学算子 `ops::kvmem_score_layer()`：第一遍每个Q head/query token计算整个指定评分域的max与softmax分母，第二遍每个页独占写入，以固定layer/head/token次序累加。按层与Q head平均、query token求和，总概率质量为M；没有浮点atomic、CPU产品scorer或新增打分线程。非有限Q/Mean-K（包括排除域中的页）使整个capture无效，后续层保持错误状态；只有新capture显式reset才清除。设备输入、五个scratch区域要求互不重叠；大偏移使用size_t，planner和wrapper统一拒绝超过tensor索引域的布局。
- `make_scoring_domain()`与指定参考默认行为一致：只有`nb>budget>0`且中间域非空才排除配置sink/recent bands，recent使用参考的`R/64`，不使用D9硬必选union替代mask。纯参数`all_pages`覆盖全部已提交页；隐藏环境变量、真实模型评分协调器及每轮耗时日志在4.3–4.4接线，当前未解除`kvmem`入口拒绝。
- CPU纯selector实现D9 current fits/soften、sink/recent完整页/query硬保护、生成reserve/guard独立计费、硬溢出各项计数、较新logical ID精确tie及升序retained/add/remove。历史图像按页最大分数原子竞争，预算不足整组跳过。**规格审阅发现并修正一个实际缺口**：旧平铺span接口不能表达同一图像的多个不连续跨度。现在`ImageGroup`显式持有同一图像的全部spans，先组内union，再处理跨图像共享页的传递闭包；新增hard闭包、溢出、候选整组选入/跳过与空组拒绝测试。修复前RED为`4.2-image-red-ctest.log`（exit8），未放宽图像判据。
- `HostKVTransferEngine::prefetch_index()`复用现有ticket、H2D stream、两个copy workers和消费事件。pageable模式复用4×64 MiB环；pinned模式环为0，直接复制cacheable CUDA-pinned索引，明确拒绝pageable或write-combined源。源必须保持不可变及存活直至DMA/消费完成，`wait()`只下发GPU依赖，不代表主机源可立即释放。新增`test_meank_index_transfer`以真实262K每层8 MiB stride测试两层逐字节传输、50 ms延迟消费后的重叠复用、live覆盖/旧ticket/空输入/超64 MiB/越界拒绝。
- 来源notice列明`https://github.com/kvmem/kvmem-qw3`、commit `1cf3b2f83bfc071ada9c57491a7d121723051ac0`、`src/kvmem_store.cpp`和`pick_topk_blocks`、`pick_semantic_groups`、`set_selection`，说明纯规划、预算和原子闭包等改写。指定源无作者/年份版权头，未杜撰版权；参考与本仓库根LICENSE均保留完整Apache-2.0文本。没有移植原scorer的atomic。
- 新增`test_kvmem_scoring`使用独立CPU FP64点积/global softmax及不转换为FP32的候选排序。覆盖Qwen27B几何L16/QH24/KVH4/D256/M16、37页和513页跨线程迭代几何，两分母变体、极大有限logit、质量M、逐位重复、NaN/跨层sticky/reset、stale generation、D9/图像/tie/diff/预算。六个评分用例最大相对L2为 **6.84327e-7**（门槛1e-4）；GPU与FP64六组入选集合全部相同。测试仍显式执行固定`epsilon=1e-6×max(1,abs(FP64第k阈值))`规则；本次没有需要边界例外的差异。
- **首次racecheck失败保留**：`4.2-initial-v1-racecheck.log` exit99、32 hazards，定位新增scorer的共享reduction暂存：部分warp读取max之前，其他warp已把同一块内存覆盖为sum。已在所有线程将max读入寄存器后、sum写入前加入CTA barrier；只修改新增scorer，没有改dense内核。后续两个独立审阅分别规格PASS、代码质量APPROVED。
- **终版门禁**：`full_checks.ps1 -Stage 4.2-final-v3`全量构建exit0，CTest exit0：109项、105通过、4制品缺失跳过、0失败，**227.52 s**。跳过原因与4.1相同，没有替换模型掩盖缺失。对`test_kvmem_scoring.exe`及`test_meank_index_transfer.exe`分别执行`run_sanitizers.py --stage 4.2-final-v3-scoring|4.2-final-v3-index-transfer --test <对应二进制绝对路径>`，compute-sanitizer **13.0.85**使用`--tool racecheck|synccheck|initcheck --error-exitcode 99`，六次均exit0、0 hazard/0 error。两份二进制SHA256分别为`44d7dbbd29761bcf4a95c51ae8e0d49c438a48c107306b00faf7c636a167eceb`与`4a34b17bb1bcf836d49ca6cf83d83edc86deb234f15e7484533cb54fa8506b68`。
- 262K/M16的**评分借用布局计算**：单层Mean-K 8 MiB、Q192 KiB，共借staging **8.1875 MiB**；logits借partial O **6 MiB**；stats/row status/scores/status借另一partial区域 **20996 bytes**。planner不分配CUDA/锁页内存，不能把该计算写成生产模型加载实测。真实Main统一准入、borrow排空、模型hook、GPU打分/选块/hydrate和TTFT记录在4.3–4.4完成；当前没有端到端稀疏测量。
- 证据位于仓库外`D:/deeplearning/NInfer/logs/kvmem-stage4-sparse/4.2-final-evidence.json`，包含原始命令、CTest数量、六次sanitizer输出、14个相关源码/测试/notice SHA和阶段3既有313个ops文件**逐字节零变更**。该静态核对不代替4.4终版三项dense组合门禁。
- 4.2独立提交 `b3235b76`（`feat(kvmem): add deterministic GPU scoring and atomic page selection`）已推送到`origin/feat/kvmem`，发布时HEAD与origin差异0/0；随后进入4.3前端query/span、三槽Q capture及Main派生snapshot恢复。

### 4.3–4.4 验收准备（尚非端到端结果）

- 仓库外 `stage4.4-validation-queue.json` 已固定72个稀疏运行坐标，核对六份单条长user输入的SHA、消息形状与needle存在；实际token数仍须终版frontend验证。另有 `structured-history-inputs.json` 中六份“历史文档＋assistant确认＋本轮短user query”输入，本轮消息索引为2，历史为0/1；它们只是额外合成样例，不替代72次矩阵，也不表示真实多文件/工具质量验收。
- 终版dense组合门禁使用新的独立目录 `C:/Users/aurawing/AppData/Local/NInfer/kvmem-stage4-sparse/dense-composite-gate-4.4`，不覆盖阶段3原始数据。八个Python辅助脚本已做语法检查；旧金标准提交 `b68009f7` 与九份64步fixture存在性已核对。临时补丁恢复清单改为同时包含tracked和本轮新增untracked源码，防止沿用旧清单漏掉新文件；**尚未创建终版源码snapshot、安装临时补丁、构建诊断版或运行组合门禁**。
- `stage4.4-owner-review-notes.md` 为静态检查补充：现有tiered注意力已按`block_tokens`切片K/V/Q，MTP实际计数为`current_extents+1`，新Mean-K必须使用最终application accepted计数。已有transfer错误为sticky且`reset()`不能重启失败worker；4.4仍需显式failure-aware排空、poison、恢复及注入中途plane失败的CTest，不能以未发布新表代替覆写后的旧物理槽恢复。这些是实施约束，尚无稀疏模型测量结果。

### 4.3 Frontend、三槽Q与Main派生恢复验收（2026-10-03）

- 首份实现已连接真实Main/TieredContext的pre-RoPE hook、拥有型三槽Q、accepted-only Mean-K与派生snapshot/restore。`4.3-final-directed-v4.log`五项定向测试通过，32.71 s；`4.3-owner-final-v3-sanitizers.json`记录13.0.85三项全exit0、0 hazard/error，capture测试二进制SHA为`986859cddcb35fab63da7985863d91f96e17bccba973293325a745af41122b12`。`4.3-candidate-v1`完整构建和CTest均exit0，111项为107通过、4制品缺失跳过、0失败，227.98 s；这些是**修正前**证据，不作为修正后终版门禁。
- **独立规格审阅FAIL，未因已有测试通过而提交**：显式`current_input_message=0`、两条user文字分别为20个`x`与1个`x`时，renderer只从最新一条消息取query。独立fixture中前条text ordinal为2–21，后条为26；按本轮所有新user文字的last16应为7–21及26，而原实现只有26。这不涉及D表修订，属于既定§3要求的实现缺口。现已汇总本轮所有user文字跨度，最后一条仅附件时也保留本轮较早文字。运行时RED为`4.3-correction-frontend-red-v1.log`（CTest exit8、两项断言失败）；已知ordinal辅助API的编译RED另行保留，不把编译失败写成运行时RED。
- 前端跨度跟踪按功能开启，dense默认关闭；current-input事件与实际prefill delta分别表达。query只含本轮user文本，排除模板、role、媒体与其他角色；无新文本时fallback取最近源user。`source_query_messages`、`all_user_text`及已知ordinal校验支持合法多user capture续接tool：1–16个严格升序ordinal逐一核对角色/媒体及有界源前缀。raw token输入不虚构角色；原始完整编码ID、NFC/Hangul/BPE/added token及媒体M-RoPE路径测试保持一致。首版没有历史图像引用识别。
- Q使用加载期固定的三个pageable BF16槽与不可变拥有型句柄；借用排空后才复用。`SparseCaptureOwner`固定设备/锁页资源连接真实RMSNorm→RoPE hook；Exact/Ordinary/Speculative区分明确，Mean-K仅提交最终accepted列，16层D2H事件完成后才发布。Main/Tiered snapshot保存精确sum/count、frontier/generation与Q句柄；restore重新生成FP16索引、清除无效tail、递增generation并拒绝旧绑定。旧capture epoch可合法配新index generation；无文本续接、补捕获策略及真实Main重复restore均有测试。
- **终版门禁**：`full_checks.ps1 -Stage 4.3-final-v1`完整构建exit0、CTest exit0，111项中107通过、4制品缺失跳过、0失败，**225.43 s**；跳过原因与4.2相同。修正后五项定向测试全部通过，32.58 s。`run_sanitizers.py --stage 4.3-owner-correction-v1 --test D:/VSCodeProjects/ninfer-4090/build-vision-integration/tests/ninfer_test_query_capture.exe`使用compute-sanitizer 13.0.85，racecheck/synccheck/initcheck均exit0、0 hazard/0 error，分别31.84/2.84/2.97 s，二进制SHA256 `65a0e5e87cb5d655afdad94a2f671ad9747265264c4f70da6d566b2e1739a7d8`。规格复审PASS、质量审阅APPROVED；`4.3-final-evidence.json`核对27个源码/测试SHA、全量及sanitizer二进制一致性、既有313个ops文件逐字节零变更。
- **布局计算，非模型实测**：N16三槽Q主机9 MiB、设备3 MiB、锁页bounce3 MiB，共15 MiB。262K/chunk2048的Mean-K设备资源3964992 bytes，合计Q设备7110720 bytes；owner固定锁页Mean/sum patch与Q bounce为4292608 bytes。主机索引128 MiB及有界active/base/snapshot资源单列。统一加载准入与预算、Program事务协调、评分/hydrate及产品入口属于4.4，当前`kvmem`仍拒绝；本步不宣告稀疏矩阵或终版dense组合门禁完成。
- 质量审阅的非阻断建议留给4.4核对：多消息跨度投影的CPU耗时、伪造图像跨度的溢出边界、恢复后FP16索引值的额外断言。修正前失败及证据保留，未放宽判据。
- 4.3独立提交`e696eb02`（`feat(kvmem): capture query provenance and restore derived Main state`）已推送到`origin/feat/kvmem`，发布时差异0/0，仓库外记录`4.3-published.json`。

### 4.4 接线实施与验收准备（尚未完成）

- 先实现纯视图selection/install、检索进入次序的reserve安全驱逐，以及传输故障后的排空/恢复基础，再接入Main运行时。H2D覆写victim后旧mapping不可继续执行；D2H失败可能破坏归档权威性，必须分别处理，不把普通`reset()`当作已经恢复。
- 仓库外`stage4.4-runtime-integration-handoff.md`记录固定加载资源、组合主机准入、capture实际accepted边界、评分scratch借用生命周期、任意resident后的完整exact prefill及机器可读计时要求。`run_sparse_validation.py`仅完成语法检查，需终版frontend计数证据后才运行；目前仍没有4.4模型测量，72次合成矩阵和真实语料质量均未验收。
- Core基础新增任意selection/install、检索进入次序保护及failure-aware传输排空/恢复。首次四项定向CTest通过3.82 s，transfer测试13.0.85三项sanitizer均为0 hazard/error；`4.4-foundation-api-handoff.md`与source manifest保留这份候选证据，独立规格审阅正在进行。随后root的`4.4-foundation-candidate-v1`**全量构建失败，CTest未运行**：CLI链接引用原三参数transfer构造函数，而新core提供四参数默认fault签名。当前核对对象文件/增量依赖的根因，不将定向通过写为全量通过，也不删除失败日志。
- 已确认该链接失败来自旧对象的缺失头文件依赖：Tiered对象的Ninja compiled deps为0，但scanner有39项；旧engine引用三参数symbol，而新core只有四参数symbol。保留显式三参数转发构造函数后，CLI链接通过仅证明symbol兼容，不证明旧table/snapshot布局可执行。root的仓库外`refresh_msvc_objects.py`据scanner依赖与header/object时间戳核对，仅刷新5个过期MSVC `.obj`（两variant、TieredContext、tiered/capture测试），不修改源文件、不清理CUDA对象。`4.4-foundation-repair-v1-msvc-dependency-refresh.json`保留路径、原对象SHA及相关header源码零变化核对；随后重跑全量构建/CTest，尚待结果。
- **独立规格审阅复现P1并修正**：released槽排队复用后，fault取消该job，旧消费event被新ticket状态遮蔽，quiesce可在另一consumer仍借用staging时返回。首次延迟callback复现因提交序列串行化未失败，保留该记录；强化pending前提后的`4.4-foundation-borrow-red2-test.log`确实exit1。修正用独立`consumed_recorded`追踪，分别排空旧recorded event与当前waited/unreleased stream，新增故障取消复用和健康复用未release两分支，pinned/pageable都有逐字节oracle。修正后4项定向CTest全通过5.47 s，transfer测试三项13.0.85 sanitizer全exit0、0 hazard/error，6.664/6.373/7.132 s，二进制SHA `75c35e39581f33ef662aed05398ed254e4c2227fc710edb34425c3cf4514d81a`。规格复审PASS，质量审阅正在进行；终版source清单`4.4-foundation-correction-source-manifest.json`，不以此替代完整4.4验收。
- `4.4-foundation-candidate-v2`全量构建通过；111项CTest为**106通过、4制品缺失跳过、1失败**，246.41 s。失败为`ninfer_test_kvmem_resume_checkpoint`的pinned/MTP回滚路径：`startup KV transfer ticket capacity exhausted: capacity=30 references=30`。保留完整build/CTest日志，先区分引用泄漏与有界队列背压，不能通过扩大容量掩盖。本次候选检查不作为终版通过。
- **独立质量审阅未通过，正在修正两处P2异常安全缺口**：constructor先扩展consumed事件数组、后扩展消费标记数组，后一分配失败会使清理越界；recover在两线程成功启动前清除sticky错误，第二线程创建失败会留下部分worker与错误的健康状态。修正要求部分初始化可安全清理、重启作为fail-closed事务、失败后可合法再次恢复，并有默认关闭的故障注入回归。尚未发布4.4，也没有将这两处修正写成通过。
- 两处P2的运行时RED已记录：constructor隔离子进程在注入bad_alloc后发生访问异常`-1073741819`，第二worker启动失败后健康标记错误的用例exit1；日志分别为`4.4-foundation-constructor-unwind-only-red.log`和`4.4-foundation-startup-recovery-only-red.log`。启动顺序/事务修正后的回归通过，但同批定向检查再次复现pinned/MTP ticket耗尽，故仍未宣告foundation全量通过。
- ticket问题已得到确定性RED：预热30个released槽、15个延迟回写和15个排队H2D，旧回写ready-event引用加H2D消费依赖耗尽固定30槽，`4.4-foundation-ticket-red-test.log`exit1。正在改为H2D FIFO中的回写读栅栏，并显式确认该代event已record后D2H才wait，解除对可复用ticket代的占用；不扩大容量，不新增CUDA/锁页资源，不让预取等待GPU生产者。尚待修正后的完整测试与复审。
- **修正后的冻结候选证据**：`4.4-foundation-quality-green-ctest.log`五项定向CTest全通过37.31 s，其中实际resume/checkpoint用例30.04 s；30槽压力用例exit0、1.694 s。消费标记数组先于事件数组分配，worker启动门仅在两个设备握手都成功后发布健康状态。FIFO fence使用原有每层`writeback_done`事件，record确认与最后D2H完成分别排序，每层归档future保证代间串行；fence不持有完成promise，取消时由配对回写唯一发布异常。pending-fence故障/排空/恢复覆盖pinned与pageable。
- `4.4-foundation-quality-final-transfer-sanitizers.json`三项13.0.85均exit0、0 hazard/error，8.493/8.201/8.855 s；transfer二进制SHA `2ce2a6b9f88e8277d342a75ff19bdfcdc9ad64ae50b14dd495add31237a6ded3`。`4.4-foundation-quality-source-manifest.json`冻结8个源码/测试，root逐字节核对未变。压力用例staging仍67600384 bytes、ticket仍30，固定事件仍pinned92/pageable96、环仍0/268435456 bytes。规格复审与`4.4-foundation-final-v1`完整构建/CTest正在运行；这仍是接线基础，尚非整个4.4验收。
- 上述冻结候选规格复审PASS、质量复审APPROVED；但随后`4.4-foundation-final-v1`全量构建通过，111项CTest仍为**106通过、4制品缺失跳过、1失败**，242.92 s。实际resume/checkpoint已通过；新的失败为`ninfer_test_tiered_runtime`：pageable路径在F257→129→129恢复后，注意力与独立uniform-attention oracle不符（0.43 s）。完整日志及`4.4-foundation-final-v1-last-test.log`保留。重新打开实现调查，先无插桩重复复现，再定位最早错误字节/算子；不放宽数值判据，不把两项复审通过写为全量通过，仍未开始产品稀疏接线。
- 数值调查：原二进制无插桩10次均通过；仅失败时打印的诊断构建第55次在**pinned INT8冷启动第一块**复现（base0/count64/layer15/t61/d1536，observed0.490234、FP64 expected0.5），此时没有HostOnly换入，不能仅归因于恢复或新fence。测试通过`DeviceBuffer::copy_from_host`在Stream0提交pageable H2D，计算流为nonblocking；前一层计算sync不能发布下一次H2D。临时event记录显示多次上传在下一consumer提交时尚未完成，仍在抓取错误时真实KV/scale以坐实数据来源，并准备同流上传的确定性顺序回归；不修改生产数值路径。
- **规格追审新增P1，前次PASS不覆盖真实CUDA错误传播**：`CUDA_CHECK`到`device.cu`会`std::abort`，transfer worker和quiesce的try/catch无法处理真实API错误；D2H间接调用的pool copy也沿同一abort路径。现有`std::runtime_error`注入只证明模拟故障恢复。必须在core中加入携带status的局部throwing检查及共享的status-returning pool-copy路径，旧wrapper和全局宏保持原行为；sticky poison/排空后仅非fatal且drain成功者可恢复，fatal context明确unrecoverable，不做device reset。需真实非fatal CUDA拒绝及后续逐字节恢复测试。此为既定§6契约实现缺口，不是D表变更；数值测试调查完成后顺序修正，再复审与全量验收。
- 数值同步控制证据：仓库外`4.4-numeric-stream-ordering-witness.*`用pinned async控制源（**不是声称原pageable API必然延迟**）保持Stream0 producer未完成，nonblocking consumer读到65536个旧BF16值0.46875，event等待后全部为新值0.5。真实tiered第一块/layer15控制复现中，未等待producer时输出0.46875而非0.5；缓存64个key均code127、FP16 scale0.00369072，独立解码0.468721，确为上一层值。相同控制加入event等待后完整fixture通过。日志`4.4-numeric-runtime-control-unordered.log`/`ordered.log`保留；将positions/V/shadow oracle输入改为测试所属compute流上传，并补源重用与发布顺序回归，生产core/数值路径不因该测试缺口修改。
- 测试所属上传修正已冻结：先核对原始源码SHA恢复，再去掉临时诊断；7个positions/V/shadow输入上传点使用所属compute流。确定性回归的临时变异（helper改回默认流copy）exit1，提示前一consumer的源字节被覆盖；同流helper GREEN exit0。原独立FP64数学及容差均不变。修正后完整fixture无插桩连续**100/100通过**，定向CTest与memcheck通过；racecheck/synccheck/initcheck及独立审阅仍在完成，不以此替代foundation全量或4.4产品验收。
- 上述候选静态规格与质量审阅通过，完整fixture的racecheck为0 hazard（约420 s）、synccheck为0 error；**initcheck exit1，0 memory errors但控制夹具10秒gate超时**，不能记为三项sanitizer通过。待验证线索为initcheck拦截D2D copy时进行同步校验，与关闭的compute gate互相等待。控制回归将改用预先分配的真实KV append内核作为前后consumer，再做helper变异RED、重复和全部sanitizer；不增加超时、不跳过检查、不修改生产算子及数值容差。
- 控制回归改为真实BF16 KV append后，预分配两个物理页及两套原位置，前后consumer分别保存旧/新输入；初始化和内核预热在gate之前完成。新源码SHA `7c9fb3f0aad1eef4b3be09fe739542ef4c5604c2eed24b250be9ab2d96d45c75`，helper默认流变异仍RED，恢复后的GREEN及单独initcheck通过；完整fixture新一轮100/100通过（250.717 s）、定向CTest通过，规格与质量复审通过。完整fixture终版sanitizer仍在运行，因此仍未记为foundation全量通过。
- 仓库外终版验收脚本的独立审阅发现并修正了准备阶段的误通过风险：计数资料须含完整12份输入及当前production源码/CLI/serve身份；dense输入须匹配原金标准hash；生产并列判定保留所有OLD分歧而非挑选成功witness；另外修正旧label、dense reference graph元数据、补充统计阻断正式门禁和指标字段不一致。`4.4-harness-review-verification.json`的11个坏证据拒绝、9份金标准输入hash及72坐标未变检查均通过。**这些是CPU脚本检查，不是实际token计数、dense组合门禁或稀疏质量验收**。临时诊断的外层finally恢复与原始字节核对执行脚本已准备，尚未运行。
- 测试同步修正终版证明文件为`4.4-numeric-upload-append-evidence-manifest.json`（SHA `74a7b150658040563f7492de3e47da736640d58a401947459a0ad8aa8ac9dca7`）；二进制SHA `2679c8332d1c5e261c583c55523a37a34f5f651aa03a5430ee5974c87b9eec9c`。定向CTest1/1通过2.55 s；Compute Sanitizer13.0.85完整fixture的memcheck/racecheck/synccheck/initcheck全部exit0、0 error，其中racecheck0 hazard/0 warning、413.29 s。与原始fixture容差保持一致，临时变异恢复的SHA与冻结源码一致，所有失败资料保留。此仅完成测试同步子任务；core真实CUDA错误传播修正已开始，foundation全量和产品4.4验收仍待完成。
- core真实CUDA拒绝RED已坐实：有效H2D完成后，使用有效固定src/dst/stream但非法`cudaMemcpyKind(99)`，API返回`cudaErrorInvalidMemcpyDirection`，旧检查直接abort、exit `-1073740791`。命令与日志保留为`4.4-cuda-recovery-red1-build.log`/`test.log`。此测试不使用非法GPU地址或损坏context；正在实现局部typed status、pool共享status-copy、sticky错误和fail-closed恢复，尚未记为GREEN或全量通过。
- CUDA修正freeze1已有4项定向CTest通过，transfer二进制SHA `dcf1f2093312be3eee9c02122f343fe2bdf77d96b431ba5065c22c4eafc925f6` 的racecheck/synccheck/initcheck全部exit0、0 hazard/error（racecheck也0 warning）。真实H2D/D2H拒绝及公开event非法flag覆盖pinned/pageable；pool共享status-copy保持PageMajor合并和HeadMajor逐页布局，旧wrapper仍保留原检查行为。随后自查发现两处待收尾：prefetch归档同步异常需主动排空并取消排队future；构造期pinned释放和fatal分配错误必须走checked/fail-closed路径。正在生成freeze2，freeze1证据保留，不把它宣告为最终foundation通过。
- CUDA修正freeze2的4项定向CTest通过，transfer二进制SHA `842307dc95c19750c2546e192b38f41cd3937217517b137b49b4e0a228d4fbc1` 的racecheck/synccheck/initcheck全部exit0、0 hazard/error；规格复审PASS。随后质量复审发现一项P2：`producer_streams`先扩容、`producer_borrowed`标记后扩容，后者分配失败时析构可能越界访问。尚未批准该候选；补构造分配失败回归并修正初始化顺序后，重新审阅与全量验证。`4.4-foundation-final-v2`完整构建已通过，CTest仍在运行，不把中间结果写为完整4.4验收；产品稀疏接线尚未开始。
- `4.4-foundation-final-v2`随后完整构建与全量CTest均exit0：111项中107通过、4专用制品缺失跳过、0失败，258.61 s。此结果绑定到P2修正之前的源码；构造失败路径未被这轮测试覆盖，质量复审仍未批准，不能据此关闭该缺陷。新增分配失败回归和修正后的全量结果将另存，不覆盖本轮资料。仓库外测量脚本另补32项ready事件记录的prefill/decode分开汇总；CPU检查拒绝缺层、重复、非有限/负值、非法层/phase及dense混入tiered计时等8种坏记录，这不是实际运行计时或产品验收。
- producer标记修正：保持完全相同的测试/header，只将标记分配放在`producer_streams`索引域之前，旧顺序的分配失败回归RED以`0xC0000005`访问违规退出，新顺序GREEN exit0；pinned/pageable均覆盖两个标记分配失败并在同一归档上重新构造。定向传输CTest2/2通过，binary SHA `e0a61df6d2efc55fa3fb836c018af37d1835c6bc2a4f0f5cae11af1acdc94940` 的三项sanitizer均exit0、0 hazard/error，规格复审PASS，质量复审进行中。证据为`4.4-producer-flags-repair-source-manifest.json`。随后核对归档析构时发现同类P2：析构按`done.size()`访问`read_done[i]`，而后者扩容在前者之后，分配失败也可能越界；正在补归档构造回归并修正该顺序，不把前述定向结果当作整个foundation已通过。
- producer标记修正规格与质量复审均通过。归档事件数组修正也已取得独立RED/GREEN：完全相同的回归/header，旧`done`先分配顺序访问违规`0xC0000005`，新`read_done`先分配顺序exit0，覆盖pinned/pageable/auto并用同一有效布局重新构造。定向归档/引擎CTest2/2通过；archive binary SHA `49cf58061090443778339d26e7b65312fc811c3943fa0164d334805d69ccd875` 的racecheck/synccheck/initcheck均exit0、0 hazard/error。`4.4-archive-events-repair-source-manifest.json`保留完整命令/源码/二进制SHA；引擎重链接后的身份为`86902ba5d89cc2ee8a082600862e5b6eea3f615c654c44eed4f15a621a35f0f0`，旧引擎sanitizer不能冒充此身份的验收。归档规格复审与`4.4-foundation-final-v3`全量验证进行中，尚未关闭整个foundation。
- **4.4接线基础验收完成**：归档修正规格/质量复审通过，`4.4-foundation-final-v3`完整构建和全量CTest均exit0，111项中107通过、4专用制品缺失跳过、0失败，248.59 s；当前引擎binary `86902ba5…`补跑的三项sanitizer全部exit0、0 hazard/error。严格核对13个相关源码/测试SHA、两份当前owner二进制与各自三项sanitizer身份，313个阶段3既有`src/ops`文件字节不变，汇总为`4.4-foundation-final-evidence.json`。原数值oracle失败、真实CUDA拒绝abort与三类构造访问违规RED均保留。这个结论仅关闭core基础子任务；kvmem产品入口仍拒绝，最终dense组合门禁、token计数、72次合成矩阵与新增运行时owner门禁仍未运行，整个4.4继续实施、不提交/推送未验收代码。
- 4.4运行时owner继续接线：新增显式exact-prefill/sparse-decode阶段、绑定调用者bundle的Q接口、实际valid extent的ordinary/MTP事务、无条件accepted提交、串行16层评分与固定scratch借用、当前逻辑视图的snapshot/restore及任意驻留集合后的exact refill。硬必选/preflight与碎片页有界gather基础定向CTest2/2通过。独立`ninfer_test_kvmem_runtime_owner`首轮通过（14.39 s），同组传输/预算/冻结tiered数值fixture通过；既有query-capture测试失败（`Main capture accepted frontier`），原因为对手工调度的TieredExact夹具也启用了自动capture，正在限定到KVMem模式。`4.4-owner-fixture-first-ctest.log`保留原失败，不作为最终全量或sanitizer门禁。
- 接线中的统计口径明确为：`actual_view_tokens`是预算物理V×64，选中的有效key数另列；capture等待是prefill总墙钟时间的子项，不重复相加到TTFT；评分H2D/D2H包含主机/worker等待，只有评分compute使用CUDA event。前端准备还核对了Responses历史/指令偏移，以及Anthropic单个user事件拆成tool+user后仍须保留原事件起点。相关工作说明在仓库外`4.4-product-task-preparation.md`和`4.4-program-integration-notes.md`，不是已运行的产品证据；产品入口仍拒绝，终版测量未开始。
- 运行时owner的TieredExact手工capture兼容问题已限定自动调度到KVMem后修正；后续五项边界CTest全部通过（38.54 s）。另补非均匀Q/Mean-K的实际16层评分oracle，以及RK4码字/scale各plane的逐字节检查和独立FP64解码，RK4 owner定向CTest通过（28.26 s）。此前首次重链接因旧测试进程仍持有exe而出现LNK1104，日志保留；后续构建/测试串行执行。这些仍是候选定向证据，不替代终版全量或产品门禁。
- Query borrow异常安全RED：显式finish失败后析构再次抛出，隔离进程以`0xC0000409`退出；修正为保留失败源直到成功排空重试，析构不再次抛出，CPU回归与owner/query定向检查通过。恢复路径的临时变异移除`invalidate_device_ownership`后，真正发生hydrate覆写的用例在16层原KV字节oracle处exit1；随后源码恢复到原始SHA `ce9cd2de5a7953ad5881b66da4ef5e03cc721510096bf28bebdcec1884b3fe06`，变异不提交。
- owner收尾自查又发现通用源异常可能被恢复逻辑吞掉：真实H2D之后的新增回归RED（15.22 s）提示`generic score source failure remains fail-closed`。正在收紧为只有已知typed非fatal CUDA拒绝、排空成功且归档权威性明确时才能恢复；通用异常继续poison并传播，fatal/failed-drain明确不可恢复。Pinned/Pageable均补旧generation不发布、注意力被阻止和显式排空cold reset覆盖。修正后的完整构建、定向检查与三项sanitizer仍在运行；产品接线、终版dense组合门禁及模型矩阵尚未验收。
- 通用异常修正候选的完整构建通过，11项定向CTest全部通过、无跳过（73.56 s）；owner二进制SHA `33e94f401c92635152d8f3433f0beee3d70796235c388529ad22936dd40c96cf`。独立规格审阅随后发现两项缺口，因而未批准该冻结候选：capture owner仅保存poison标记，没有持久保存fatal/unknown/failed-drain分类，后续成功排空可能误允许reset；新增owner fixture只有BF16/RK4，没有把INT8 owner选页/恢复/ordinary/MTP与独立字节及FP64 oracle串起来。修正这两项属于既定失败处理与验证契约，不调整D表。此身份已启动的racecheck拟安全终止并保留为未完成，不能写成三项sanitizer通过；修正后另行冻结、复审与验收。
- 上述racecheck在517.10 s后仅终止本轮所属进程树，确认SAN/测试/构建进程全部退出再修改源码；`4.4-owner-finalize-sanitizer-interrupted.json`明确为未完成/非通过。capture持久分类RED（28.00 s）证明后来排空成功时旧Main reset仍错误获准；修正后包括failed-drain、fatal与unknown的元数据注入回归，以及新增INT8字节/scale/FP64路径，owner GREEN通过32.12 s，不损坏真实GPU context、不放宽`.0041`数值判据。冻结v3规格复审确认这两项已修正，但注意力和late restore的异常处理仍可能将fatal capture误改成`KVMemColdResetRequired`；继续补原始typed错误传播，未批准v3为最终owner。
- 补齐上述两处typed传播后，冻结v4的32个源码SHA均由独立规格复审核对，**SPEC SOURCE PASS**，没有新增D表冲突。`4.4-owner-finalize-source-freeze-v4.json` SHA `9b01e26516d5585018026cef0117a3cc260bf3a8cd2193793183c1de9e15894c`；完整构建通过，11项定向CTest全通过、无跳过（77.70 s），owner二进制SHA `e3adc567de91c5cd9e2157c17596d9f87807ede7b7b2ecdf09758f95a5b361d1`。代码质量审阅、终版三项sanitizer与当前全量CTest仍待完成；这里的规格通过仅关闭owner源码范围，不提前宣告产品入口、dense组合门禁或合成矩阵验收。
- v4代码质量审阅为**NEEDS FIXES**，发现两个相关Important故障入口：`healthy()`未检查传输worker的异步失败，故Q捕获/KV append可能在已失败的预取之后继续提交；`begin_append`已修改表后，partial-prefix恢复/发布/预取没有异常守卫，合法R=0且稀疏集合未选中部分frontier页时，失败H2D会留下表frontier已前进而Main仍报告healthy。已有测试覆盖选页hydrate/attention/restore故障，没有覆盖这个begin边界。正在补两种RED并修正为post-commit失败poison、排空所有借用、显式cold reset；纯准入/容量拒绝仍保持原视图和healthy，不要求自动恢复不完整append。未运行v4完整SAN，不宣告质量通过或发布。
- 两种QUALITY RED均坐实：合法R=0、非均匀评分且第61部分页HostOnly的prefix恢复，真实H2D拷贝后失败，旧实现用例exit1（2.08 s）；受控暂停/恢复的实际预取worker异步失败，旧实现仍允许capture/append，用例exit1（5.54 s）。加入统一`fail_incomplete_append`、engine失败health gate及post-commit守卫后，完整owner GREEN通过33.57 s；原子容量拒绝仍通过，固定GPU/锁页账本不增加。冻结v5 manifest SHA `f91f6de91b3976ccf5fec5f1e2eaf69142a4a1e68a73dbc134c03cfc91889d30`的32项源码由规格和质量复审分别核对，**SPEC SOURCE PASS / QUALITY APPROVED**，没有新增D表冲突。全量CTest及终版三项SAN接着运行；这里仍仅完成owner源码审阅，产品完整接线和最终模型验收未完成。
- 冻结v5随后完整构建及全量CTest均exit0：**112项中108通过、4专用制品缺失跳过、0失败，274.56 s**；11项定向检查全部通过、无跳过（79.28 s）。当前owner二进制SHA `149ea714cb36a31074bd80c400b222accac9c4d7f37242c955e32768bf0900aa`，全量CTest日志SHA `efd69394a1705a13af688afc7ee086be55a62fec6d0dc8c7cf3e692cb55f9538`。完整owner主入口的13.0.85 racecheck/synccheck/initcheck仍在运行；监督器使用Windows JOB排空所属进程树，每工具外层上限10800 s，不缩减夹具、内核范围或内部判据，超时不算通过。与此同时仅准备冻结32路径以外的产品源码，不构建/运行GPU或改动冻结源码。尚未解除产品`kvmem`拒绝，也未执行终版dense组合门禁和合成模型矩阵。
- **上述完整racecheck超时，非通过**：`4.4-owner-finalize-final-v5-sanitizers.json`记录10800.052 s外层监督超时、无完成summary、所属JOB进程树已排空；日志SHA `0608a4f07959ed788d2c60a16fd870a0b77de248b3b91e03f69db25f3c1eec5b`。只读分析发现21个场景累计约14份3905-token等效全历史prefill，每个64-token块均运行16层真实部分注意力；racecheck还插桩每个shared-memory K/V tile的cp.async和MMA读写。文件写入时间推测约2小时进入第二个场景，这只是间接估计，不能将超时误报为0 hazard。完整fixture和默认检查范围不缩减；在产品终版及其他GPU门禁完成后，用终版对应的不可变二进制安排最长48小时的外层监督，内部判据不变。synccheck/initcheck先在已审阅v5的不可变副本上独立完成，不把该候选检查冒充最终产品验收。
- v5的32份源码、二进制与来源manifest已逐字节归档到仓库外`C:/Users/aurawing/AppData/Local/NInfer/kvmem-stage4-sparse/owner-v5-immutable`，manifest SHA `c8976d7bfb0b3259d250faced03a7afb99fe5bd8fe19281e97d029ea797030a3`，二进制仍为`149ea714…900aa`。随后产品接线开放源码/构建通道，GPU检查仍串行；变更后须重新审阅并绑定最终源码/二进制身份。正式计数和合成测量脚本也改用同一已验证Windows JOB helper，异常/超时时排空整个所属进程树；CPU检查证明正常、exit7、超时的环境恢复与子进程排空，以及服务后代不能逃逸。34项坏数值/计时记录和3项未完成/未排空计数资料均被拒绝，证据`4.4-sparse-measurement-contract-verification-v3.json`及`4.4-measurement-owned-process-verification-v1/evidence.json`。这些是监督器/脚本检查，**不是模型计数、sanitizer通过或合成质量验收**。
- 归档版v5的完整synccheck与initcheck随后分别**exit0、0 error，85.71 s / 194.97 s**；均执行未过滤的21场景、出现owner PASS标记、所属JOB排空、32个归档源码及二进制SHA不变。原始日志SHA分别为`f54bbd53fb5737ceb3dbe2a732b560bde75ad3141bc92ed6fcd8f4fbf0b8fd76`、`c5899abd7c943a56f2a02e15a47bfeb9f9ce2b835857665f3594233b67aade5f`；范围汇总`4.4-owner-finalize-archived-evidence-v6.json` SHA `e5cd53d6a87065f05686233378f093e4d9c998bf9d19e4d0783325405a298fa0`明确保留`ALL_SAN=false`和race pending。GPU通道已转交产品接线的构建/测试；当前源码与归档版有区别时，后续终版证据须重新绑定，不能沿用归档SHA冒充终版。
- 产品接线候选的实模检查已进入真实27B Program：`4.4-product-real-20261004-e.log`覆盖ordinary/pageable INT8与MTP/pinned INT8、取消、零新增token的拥有型Q复用、checkpoint重试、tool-only原始Q续接和冷启动对照。此前两个“构建成功”实际没有刷新MSVC header依赖，造成旧二进制仍拒绝kvmem和ABI不一致；依赖刷新后才有真正运行证据，相关构建及失败日志保留。零新增token恢复后仍处于SparseDecode的运行RED保存在`4.4-product-real-20261004-d.log`，现已在重新选页前显式进入exact-prefill阶段，不重复捕获Q。
- `4.4-product-real-20261004-f.log`进一步通过真实Program失败边界：有效指针上的D2H拒绝使归档权威性不确定，显式cold reset后Main/index frontier、当前槽GDN、MTP页标签、GPU position及hidden/ledger/checkpoint/resume/query状态全部恢复冷态，随后完整exact prefill与冷启动token对照通过。fatal Q借用错误只注入typed分类、不损坏真实CUDA context；后来借用排空成功仍不能解除原始错误，下一请求继续被阻止。内部测试访问桥没有增加生产配置、fault flag或运行时分支。这仍是产品候选定向结果；全量CTest、独立规格/质量审阅和终版模型门禁尚未关闭。
- 合成汇总器的CPU契约检查`4.4-sparse-summary-contract-verification-v1.json`用虚构数据验证72坐标、24组三次重复、12对照齐全，拒绝缺失/重复/额外坐标，保留失败needle，并确认capture等待不重复计入TTFT。`4.4-dense-patch-anchor-contract-v1.json`只在临时文件副本上验证旧版与当前候选使用完全相同的诊断补丁；没有修改仓库、构建或运行dense。两者都不是实际模型或最终门禁的通过证据。
- 产品候选全量`4.4-product-fullctest-20261004-a.log`有一项旧ABI失败（request_log）；不完整但非零的MSVC依赖漏记新request header，原对象早于header，scanner列表提供了确切依赖。已扩展仓库外刷新器，只移除经路径/mtime/SHA验证的本地MSVC对象，排除CUDA及third_party，原helper与失败日志保留。随后`4.4-product-fullctest-20261004-b.log`为**114项、110通过、4专用制品缺失跳过、0失败，323.03 s**；这是j候选身份，不冒充后续修订身份。新事务断言首次误访问ordinary省略的row_counts，夹具访问违规RED保留；修正为遵循implicit-one契约后，真实full/partial/zero接受及MTP有效列检查通过，不修改生产数学或放宽判据。CLI/serve补齐设计规定的`--kvmem-gen-reserve`及`--kvmem-gen-reserve-tokens`别名。
- **进入规格审阅前发现外层恢复缺口，尚未批准产品终版**：真实Program冷重建成功，但ConcurrentExecutor在start/prefill/decode错误外层仍会fail_all并退出，public Engine不能处理下一请求。正在补仅适用于“目标确认完整cleanup成功”的窄恢复分类，失败请求仍收到原始exception，generic/fatal/failed-drain保持终止；同时新增实际executor的“失败请求→下一请求成功”回归。这实现已有DMA恢复要求，不修改D表或dense数学。`4.4-product-root-source-freeze-v1.json`（60路径，SHA `1e06dc1e8925663f1aecac48076d37927b47116d43b956fff174eeb7ac5821e4`）因该缺口不能作为最终冻结或发布证据；dense组合、正式计数、合成矩阵及终版完整racecheck继续待验收。
- 上述执行器恢复候选已补真实回归：`4.4-product-executor-20261004-m.log`与完整无过滤`4.4-product-real-20261004-j.log`均exit0，后者99.93 s、六个实际27B Program owner，二进制SHA `85106c1bf6363b84adcc25017e84d7d0127a5830231f983e33a0023c613e3e19`。ordinary/pageable与MTP/pinned均覆盖start外层、后续prefill、decode失败后的下一请求，以及fatal取消仍永久拒绝。start用仅测试调度适配器执行同一实际Program的前两块，在第一块后注入真实D2H拒绝，使第二块错误到达执行器start catch；不宣称错误发生于第一条prefill指令。后续prefill的失败前已排队请求保留且完成；fatal清理保留原始typed cause，即使之后借用排空成功也不恢复。
- 定位过程保留夹具RED及操作失败：n/i和p/k在准入前安装的fault被正常restore/reset撤销，不能作为“真实可恢复失败仍终止”的实测证据，原缺口来自源码审查；q/l的后续prefill与decode已通过，但取消安装时机有竞争，改为内部worker锁协调后r/m通过。o构建LNK1104由提前启动旧测试持有exe造成，已排空且不计入通过。恢复判定不比较`exception_ptr`地址，因为MSVC多次`current_exception()`可复制异常；改为完整异常cleanup标记、实际冷态和typed分类，标记在每次新准入前清除。较早c全量114项、110通过、4跳过、0失败（326.50 s）属于修复前候选；修复后全量d正在完成，独立规格/质量与最终模型门禁仍未批准。
- d全量随后**114项、110通过、4专用制品缺失跳过、0失败，382.25 s**。但最终源码自查另发现fatal顺序缺口：部分Program catch和clear尾部仍先调用通用同步，真实CUDA sticky错误可能经global CUDA_CHECK直接abort，早于checked owner清理并掩盖原始typed cause。正在仅对kvmem路径改为checked同步与错误清理顺序，global helper及dense既有同步保持不变。因此d也不是最终冻结验收，修复后须重新完整构建、默认Product与全量CTest，再交独立审查。
- w候选的完整默认Product k随后exit0（125.10 s），含真实返回status21送入计算排空检查的模拟失败位置，确认`drain_failed=true`、原始原因、永久拒绝及generation保留；没有破坏实际CUDA context。日志42条turn、224条transfer（7个owner×16层×prefill/decode），严格计数/有限值检查通过。e全量**114项、110通过、4专用制品缺失跳过、0失败，426.80 s**，仍只属于该候选。
- **独立规格审查v2不通过，保留失败记录**：冻结64路径manifest SHA `5b5c38399b490349978d0eb25e842bacc2deb35a2b4ec86b17a0380908e31ff4`前后逐字节一致。确认四项缺口：TextContext内部prefill末尾仍先调用通用同步；无关短历史的tool-only冷请求可误用旧original Q并拒绝；capture/Main reset有主机状态早于检查过的设备清理发布；turn日志缺返回的索引generation/capture ID/owner epoch。另要求实测最后全注意力层的迟发D2H失败，不能以首层plane1/129覆盖代替。报告`4.4-product-spec-final-v2.md`，同一实施者继续修复与RED/GREEN，未进入质量审查、未提交4.4、未开始4.5。这里不修改D表或放宽判据。
- 最后一层迟发D2H疑点已取得真实产品RED：`4.4-product-late-20261004-g-controlled-red.log`中，实际checkpoint frontier为2101，最终真实prefill块为4 token（prompt/index frontier2105）；内部传输测试桥将第16个全注意力层的最后plane64回写暂停至选块入口后释放。真实`cudaMemcpyKind(99)`返回status21，选块初始drain传播nonfatal typed错误，但旧Program把已poison的owner判为终止，后续执行器无法继续。正在补明确的首次观察位置标记并重录有界RED，再仅对这一“不完整append且归档权威性未知”的路径使用既有whole-bundle cold恢复；generic/fatal/failed-drain仍停止。此前b/c初步GREEN未证明首次观察位置，d为暂停落在checkpoint之前造成的夹具死锁，e/f为夹具frontier断言错误，均保留，不能据此关闭疑点或宣称产品失败。
- 最终日志契约增加实际返回的`index_generation/capture_id/owner_epoch/owner_id/selection_generation`，拒绝缺失、非法整数及generation不一致；仓库外CPU契约检查`4.4-sparse-measurement-contract-verification-v4.json`通过47项坏记录拒绝，旧产品日志继续按原v3契约保留。终版owner归档监督器的CPU检查`4.4-final-owner-archiver-contract-v1.json`验证源码/二进制身份、全源码复制、已有证据不覆盖，拒绝路径越界及复制期间源码改变。这些是脚本检查，不是产品模型或SAN验收；三项dense、正式计数、合成矩阵及终版完整racecheck仍待完成。
- 迟发D2H的有界重录h明确记录`observer=selection-drain exact_frontier=2105 paused=1 after_exact=1 status=21 drain_failed=0`，日志SHA `ad2de01e9e30696f090f90bcbeb0680b795d366252c0a6d11c1ab91cba86d9a1`。修正后i定向GREEN在ordinary/pageable与MTP/pinned两种实际executor上通过：仅当存在未完成append回写且D2H失败使归档权威性未知时，走已有all-lane排空后的ColdRequired；完整bundle冷重建、保留排队请求和冷启动输出对照均通过，log SHA `3377c91640bd75bd7f49a0061fb16597aa346423dbf2b1ad334c08e1aba428c6`。这是af候选的定向结果，不替代默认产品测试、全量CTest或复审；generic评分/hydrate、fatal/unknown与failed-drain分类未放宽。
- af候选的默认无过滤产品测试l通过（186.96 s），运行12个真实27B Program owner；新增不相关短历史与同范围变更来源的tool-only查询、Program及块内checked drain、GPU清理完成后最终drain失败且accepted Main/index/Q及相关valid/checkpoint状态未提前发布、以及迟发最后plane回写后的排队请求与冷启动对照。`4.4-product-real-20261004-l.log` SHA `78bf7729ee08e1a82b96874f4bc4d5db3519054ece70fe2b1d45aa0bd5d61e9f`；当前v4 parser独立复核65条turn与384条transfer（12×16×2 phases），313个阶段3既有ops字节不变，见`4.4-product-root-verification-v3.json`。
- 该候选全量f为**114项、110通过、4专用制品缺失跳过（40/45/46/47）、0失败，477.42 s**，所属进程树排空。源码冻结v3的64路径manifest SHA `2f10fdf2a6be07e01b2f69820741db049a2a659120ca9e9a121d2a74362ce1ed`；产品/owner二进制分别`bef5121f409589277474dc9db8f1fd8122a58e6f54f2ac35be2a1b504fc88288`、`074941e3afc9e5e3a21409e9a2fca294f3ca7ffac3d6512f6b5d5be0f1f06886`。独立规格复审继续检查完整剩余范围，已发现传输错误优先级缺口：最初generic/nonfatal错误可能掩盖随后更强的typed fatal/failed-drain原因；原全量通过不关闭这项源码证据，待安全status-feed RED与修复。未进入质量审查，未提交/推送4.4。
- reset证据范围进一步澄清：defer的是accepted Main表/frontier、Mean/Q身份及相关valid/checkpoint元数据；传输recover可在排空后退役ticket/推进archive epoch、使不确定的archive frontier失效，准入也会先撤销retained标记，因此不声称“所有主机字段都逐位不变”。迟发nonfatal回写的请求收到owner产生的`KVMemColdResetRequired`字符串分类，没有携带底层status21/operation字段；执行器保留该Program分类再冷重建。fatal/failed-drain仍须保留准确的原始typed原因。上述行为的安全性继续由独立复审核对，不当作新D表决策或终版验收。
- v3完整规格复审结束，**NEEDS FIXES**：manifest及64份源码前后不变；剩余两组源码冲突已完整列于`4.4-product-spec-final-v3.md`。一是FullReset跳过合法的original多消息Q来源，导致同历史tool-only关闭复用时只捕获最后历史user的短文本；须在cold模式也验证原ordinals/source-prefix、仅在无效时回退，并冷重捕获而非复用旧buffer。二是generic/nonfatal先发生时，传输worker/drain及多个Q borrower的后续typed fatal/failed-drain会被较早异常掩盖，recover_prior_view还会替换成generic错误；须排空所有资源后保留第一个更强的typed不可恢复原因及status/operation/drain_failed，后来成功排空不能清除。尚未取得这两组的实际RED/GREEN，继续同一实施者修复；不以477.42 s全量候选通过代替源代码契约验收。
- 本次复审覆盖v2早停后余下的布局、准入、API/frontend/executor、所有新增文件与变更测试、任意sparse→exact、full/partial/zero事务、snapshot资源上限、评分/选页/日志和Dense守卫，没有未审阅的冻结源码要求。archive epoch/不确定frontier失效及retained撤销仅是安全失效记账，不是额外阻塞项；失败最终cleanup仍禁止可用snapshot/attention/后续准入。质量审阅和终版模型/SAN门禁仍未开始，4.4保持未提交、未发布。
- v3两组缺口已取得真实RED（ag回归构建，生产修正尚未应用）：`4.4-product-multi-query-20261004-a-red.log`记录同一source prefix下，warm查询ordinals `[2118,2119,2120,2139]`，tool-only FullReset却变成`[2139]`，断言失败；`4.4-product-stronger-query-20261004-a-red.log`中两个borrower清理把后来的typed fatal隐藏为`first generic borrower`；`4.4-product-stronger-transfer-20261004-b-red.log`先实际完成健康drain，再将真实有效指针kind99返回的status21送入failed-drain分类位置，仍抛`earlier generic`而失败。转移测试a使用了错误exe名、没有执行，不算RED证据。全部所属进程排空后才应用共享错误优先级与FullReset来源修正；没有损坏实际CUDA context，status-feed不冒充真实执行故障或SAN门禁。

- ai候选的默认完整Product m通过（183.30 s），71条turn与384条transfer均通过严格解析；6项owner/传输/查询与既有tiered/Mean索引定向CTest通过（48.89 s）。全量g为**114项、110通过、4专用制品缺失跳过、0失败，459.15 s**，所有所属进程树已排空。日志SHA `e89d42e5e5653993ffc15586c16cb450bda5ef47f1253e48aac4c3681a37342f`；这些仍是候选结果，不是产品终版批准。
- **独立规格复审v4仍未通过**：冻结65路径manifest SHA `bdab2204d8d9258c5d64ca7d5a34646fe8163d187303a2597280852383bab814`前后不变，53个未变化路径沿用v3完整审阅，12个变更/新增路径已复核。剩余一项原始原因顺序缺口：Main已锁存typed不可恢复A后，reset的局部collector先看到后来的Q borrower typed B，最终可报告B而非A。prior-view/attention/restore中的跨owner清理也须覆盖历史transfer/Sparse错误先于新compute错误。报告`4.4-product-spec-final-v4.md`明确这是源码坐实、诊断RED待补，复用仍被阻止，不声称发现可恢复安全性绕过。全量g排空后已交同一实施者补安全RED并修正历史原因预先收集，仍须尝试所有资源清理；不修改D表、不放宽门禁，4.4保持未提交。

- v4历史原因顺序疑点随后取得安全实测RED：`4.4-product-historical-cause-20261004-a-red.log`（SHA `3340ca5f437bc887e8b0653784160357babd1dc2dcdbe7732d78d4f9b9b183f2`）先用有效指针kind99的真实status21在Main诊断入口锁存A（`historical Main original A`、`drain_failed=true`），后续真实Q borrower清理抛出typed B（status700、`drain_failed=false`）；reset报告B而断言失败，exit1。实际CUDA流均健康且所属进程已排空；不冒充硬件损坏、复用绕过或SAN结果。仅补测试夹具的aj owner二进制SHA `b9b81e23e617de5a1ee264732760db29c75c0de32fb497bd18c41157a7d8807c`，生产代码仍为v4。正在同一实施者修正历史原因先收集及跨owner回归，仍未进入质量审阅或终版模型门禁。

- 历史原因修正后的ak构建已完成，但b夹具运行不能算GREEN：故意kind99调用的预期status21留在CUDA last-error槽中，下一份独立夹具的既有launcher读到该旧错误而abort。实际CUDA context没有损坏；已确认先前MainA→QB路径继续到下一夹具。后续只在测试夹具取得status21后消费并断言该预期last-error，保留已捕获的status/operation/drain_failed，不修改global CUDA_CHECK或dense helper。该操作失败与日志保留，所属测试进程排空后再修改和重测。

- al修正版的历史原因定向c为GREEN（exit0、所属进程已排空），日志SHA `e33acde332d1bdfd51d70c2018ac2910648ffadc02011d9641e3ddf5f0ae0cdb`。Main先存A(status21、`historical Main original A`、drain_failed=true)后遇Q B，仍报告A；Sparse/transfer先存A(status700、各自static operation、false)后遇模拟compute failed-drain B(status21)，源码断言收到A且后续reset仍为A，accepted generation/frontier不变。cross-owner日志中的`returned_status=21`是kind99实际API返回并送入诊断位置的输入，不是最终传播的异常status；不据此宣称传播21。夹具消费并断言预期last-error后不影响下一独立夹具。新增内部bridge复用同一个真实清理collector，不加生产配置或破坏context。冻结v5为66路径，SHA `c05e07375df9f071a8ee0f27389fd656fbdf4a058ffd0ea150801ac5d03611cc`，相比v4只有Main .cpp/.h、内部bridge及owner测试四处变化；独立SPEC复审和完整产品/全量CTest继续，仍未关闭终版门禁。

- **v5独立SPEC仍为NEEDS FIXES**：66份源码/测试SHA前后不变，62个未改路径沿用v4完整范围；v4“既存Main A→Q B”已关闭。但同一共享清理collector在transfer.quiesce前收集历史原因，quiesce本身首次记录的新typed A尚未被收集，就先检查compute的新typed B，仍可丢失原始A的operation身份。core quiesce真实代码逐个record_drain且明确允许排空成为首次错误，故这不是假想接口；诊断RED待补，仍fail-closed，不声称复用绕过。报告`4.4-product-spec-final-v5.md`，完整默认产品n排空后交同一实施者修正post-quiesce原因收集并补回归；已知缺口未关闭前不启动全量h，避免将候选通过冒充契约通过。质量、正式计数/矩阵、dense及完整SAN均待验收，4.4未提交。

- v5默认产品n随后exit0（183.28 s），12个实际Program owner、所属进程排空；root独立解析71条turn及384条prefill/decode传输记录均通过，原始日志SHA `d4faebe5e7095c6aef52f73778f4c7b49c3750ffc1b0d4d0c38b3caf04b9c08d`。这属于fresh-quiesce缺口修复前的候选，不算终版批准。实施者先将post-quiesce preparation/finish以等价旧语义分离到同一实际collector，保留66份RED源码及manifest `4.4-product-fresh-quiesce-an-red-source.json`（SHA `cd26b55f0d20bc438b0850ea55dc7b1b3f12480d42aa734fda55958c8411391a`），再通过真实健康quiesce/synchronize及status21诊断输入复现顺序，不能用仅在入口前存A的既有夹具代替本次边界。

- fresh-quiesce边界取得有界实测RED/GREEN：an旧等价语义的a日志（SHA `48e41758c354950b049f370fae5feaa36bd910b5ee9bd0c42f909bbdb309df62`）错误报告`later checked compute B`；ao的b日志（SHA `c7115cae8ca2b27c34143bf81f51629ec2aabc0b38f9f027cdf93027290aa348`）保留`fresh quiesce original A`，status21/drain_failed=true、两个Q callback均排空，后来健康drain/reset仍被拒绝且frontier/generation不变。历史原因d亦GREEN，SHA `7ac0f162fdbfbae1f008a831662af05a8008b5221c4cbd164855a680b597763d`。诊断输入来自健康真实quiesce/synchronize和有效指针kind99返回status21，明确是模拟返回状态位置、不损坏CUDA context。修正在quiesce后、compute/Q前刷新Main与局部原始原因，内部bridge复用实际prepare/finish，无生产fault hook。冻结v6仍66路径，SHA `dd47bda7b8bf8d82660fff15209e1a33629748fc70643a557ae87586001b1b2e`；相比v5仅Main cpp、内部bridge与owner测试三处变化，独立SPEC/完整Product o/全量h继续。

- **独立SPEC v6 SOURCE PASS**：66个源码/测试SHA在复审前后完全一致，63个未变化路径沿用v5完整范围、三处变化已检查，无未审阅或剩余规格源码要求；报告`4.4-product-spec-final-v6.md`。quiesce新原因在compute/Q前先收集，finish仍执行全部排空并保留原始typed原因。root随后再次核对全部66份源码与freeze匹配。已按既定两阶段流程交新的独立QUALITY审阅，源码保持冻结；默认Product o/定向b/全量h继续。SOURCE PASS不等于4.4发布批准，完整SAN、终版dense三项、frontend计数及合成/对照/复用仍未验收，未提交或推送。

- **QUALITY v6为NEEDS FIXES，非批准**：66份源码与manifest前后一致。审阅发现实际Q borrower callback捕获Main Impl引用；永久排空失败时callback被保留，Main析构的成员顺序先销毁unrecoverable_cause，后销毁loading.capture；后者析构重试callback时可访问已销毁exception_ptr，属terminal teardown的未定义行为。现有检查后来健康drain会移除callback，因此不覆盖永久失败。报告`4.4-product-quality-final-v6.md`，实际context损坏未实验。当前全量h完成排空后，交同一实施者修正callback生命周期并补安全永久失败析构回归，保持live fatal锁存及原因顺序，不改dense/global helpers。QUALITY检查了生产变更及新bridge/helper，但在阻塞项处停止，只读到Product1–290/owner1–305；余下测试body/hunks无质量批准，修正后须完整复审。SPEC SOURCE PASS不替代此项，4.4保持未提交。


- v6源码通过后的候选完整Product o为exit0、183.79 s；六项定向b全通过49.54 s；全量h为**114项、110通过、4专用制品缺失跳过、0失败，453.75 s**，所属进程树全部排空。全量日志SHA `73a5e6fedbd084034f724f64581aadd5476065e3e2fd930b17a53340587ab386`，候选manifest `4.4-product-candidate-source-evidence-20261004-ao.json` SHA `f12f752a82bd954132a5a27bccd910c3ec924bea9c12228712aecdad762bc66a`。这些是生命周期修正前的候选，QUALITY失败仍保留，不能作为4.4终版批准。
- 生命周期修正采用同一个生产completion工厂，只按值捕获外部compute stream，用独立typed checker，不依赖Main成员寿命。显式live释放另由`finish_score_query_borrow`将异常交回仍存活的Main collector，保留原始status/operation/drain_failed和更强历史原因；析构重试仍可独立安全执行。安全定向b为GREEN（exit0、0.37 s、5次永久失败重试、所属进程排空），生产回调在Main/Sparse销毁后而外部stream仍存活时可执行，无法排空的借用按既有规则隔离；没有故意触发旧版未定义行为或损坏CUDA context，因此不宣称旧版运行时RED。
- 冻结v7仅覆盖live wrapper前的中间字节，后续发现三处差异后保留且明确作废，不用于当前批准。终版v8冻结66路径，SHA `d28795ab37487fde39801fe81cb8796b34935674a51a59914cbfe3ed5d179f7b`；与v6相比仅Main cpp、内部bridge和owner测试改变。SPEC复审已启动；须在通过后完成QUALITY上次早停余下测试body的完整复审。最终Product q、定向c和全量i待结果；正式计数、dense三项、合成矩阵/对照/复用及完整owner SAN仍未验收。

- v8默认完整Product q通过（exit0、203.65 s、所属进程排空），但**SPEC v8为NEEDS FIXES**，不能用这次产品通过关闭源码缺口。66份冻结范围已完整复审：63个未改路径沿用v6，三处新增变化检查完毕，callback生命周期本身通过；仅余一组live release原因优先级问题：transfer/Sparse已锁存更早typed A而Main尚未导入时，显式`finish_score_query_borrow`直接检查新callback B，先把B锁存到Main，后续清理会保留B而丢失A的身份。安全回归将调用实际live wrapper和同一生产callback工厂，实际同步健康CUDA流、仅在诊断位置送入kind99的status21，不冒充损坏context实测。已交同一实施者补transfer/Sparse两条旧A→新B的RED与修正；QUALITY尚未开始，正式模型和SAN门禁仍待验收。

- v8 live release疑点得到实际wrapper的RED/GREEN：av/a中transfer和Sparse两条路径均报告新B（21、`fresh live score borrower B`、true），而不是既存A（700、各自static operation、false），exit1；日志SHA `b859e89b0381f119b583e71f1e4da6cd17d1401b69d3a1ef6e0a869d5c93d1bb`。修正在fallible borrower.finish前导入历史原因，aw/b两条路径都保留精确A且后续reset仍拒绝，exit0、0.48 s、所属进程排空，日志SHA `c682da0644506bb3c060181791a5de55bb4083e2127b966774bbdb6860aab4b1`；永久失败teardown c亦GREEN，日志SHA `e6e7c894423f2c16d7df6c5c6b00a369d7ef5fb8fc25c2bc9e9e948d48ff8306`。诊断夹具调用实际live wrapper及生产completion工厂，真实CUDA同步健康、返回状态位置为模拟，未损坏context。RED66份源码归档manifest SHA `da70f6002ad228cc852e6d27db26c086990e383b91d66501e65768cc513ab5f6`。终版v9冻结66路径，SHA `aadd96f24fe3c3837d633b22ae1a79e8b0a2ea78a82e658e0118ef5dab05b0cb`，相比v8仅Main cpp/owner测试变化，完整SPEC复审、Product r/定向c/全量i继续；不以定向GREEN代替质量、模型或SAN验收。

- **SPEC v9 SOURCE PASS**：manifest及66份源码前后匹配，64个未变化路径沿用v8完整范围，两处变化检查完毕，无剩余源码要求或未审阅范围；root亦重新核对全部66份字节。报告`4.4-product-spec-final-v9.md`。已交同一QUALITY审阅者完成修正与上次早停后的全部剩余测试body/hunks复核，源码冻结，Product r/定向c/全量i继续。源码规格通过不等于发布批准，终版dense三项、计数/合成/对照/复用和完整owner SAN仍待验收，4.4未提交或推送。

- **QUALITY v9 SOURCE PASS**：51个production/source与15个test/CMake共66路径完整复核，三处变化重新检查、其余63份与已检查v6字节一致；Product291–713、owner306–965及所有相关test/CMake hunk、完整新prompt-input测试均已读完，无未审阅范围。Critical/Important/实质Minor均无，v6生命周期阻塞关闭；实际factory/wrapper回归保留。manifest及全部66份前后SHA匹配，root亦重核，`git diff --check`通过，dense默认chunk1024及既有ops不变。报告`4.4-product-quality-final-v9.md`。Product r独立解析71条turn/384条transfer/12owners通过；六项定向c通过51.71 s、0跳过/失败，进程排空；全量i继续。源码双审通过仍非发布批准，所有正式外部门禁待完成后方可提交。

- **v9完整构建/CTest通过并排空，独占lane交还root**：全量i为114项、110通过、4制品缺失跳过（40/45/46/47）、0失败，478.66 s，日志SHA `e393c5ebb5965f870f752ebf85e70fb19ba61ae3a775ce6a70c056c89e831352`。最终manifest `4.4-product-final-source-evidence-20261004-aw.json` SHA `73bd697f87edbcbed59ac60c41cadd7973913de7827cf10c71601fa3f25df1c6`；root核对该manifest/self-review/handoff/日志，以及66份源码、313份既有ops、旧tiered oracle字节和Product/owner exe均匹配。MSVC依赖dry check为0待刷新、0未知。Product/owner分别`d4bc38bf8df3b9b175a8465dcb7a614bfbeef9fd3388534b6085523184351b24`/`5a32dbe7026ac61ffc00a45fb482e6c16770eabd3b2b121517f99fc614fd5aa4`。实施与双审完成，但正式外部门禁尚未完成，4.4未提交；现开始root独占dense三项组合门禁，所有其他实施/审阅agent不再改源码或运行构建/GPU。


- **4.4 dense 三项组合门禁完成（2026-10-05）**：旧版取九例金标准录制源码 `b68009f7`，终版取 v9 冻结工作区。两版使用完全相同的临时 Q5 `kSplits=1` 与完整词表抓取补丁，九例各两次，共 36 次；所有 64 个生成步、248077 个有效词表项均逐位一致，两次运行自身也逐位一致，物理 padding 另行核对一致。`4.4-root-dense-diagnostic-completed-v1.json` 保留 root 对全部原始 logits、token、输入及二进制 SHA 的独立核对。九例 dense 启动计划加一例默认配置，共 10 对均一致；Main/MTP pool、workspace、prefill/decode 分派与 split 相同，dense 默认 chunk 仍为 1024、生产 CUDA Graph 开启，见 `4.4-root-dense-plan-completed-v1.json`。
- **组合门禁第三项生产复测**：未改 Q5 数值算法（`kSplits=2`），旧版、终版各三次 `needle-262k-10`，六次 needle 均正确。第 9 步 271/198 的次数分别为旧版 2/1、终版 1/2；旧版 top-1/top-2 差为 0.125、0.125、0，观察到的差距波动为 0.125。严格用三次旧版作固定参考，其共同前缀最大相对 L2 为 0.2959436474，既定包络 `max(1e-3, 2 × dense_max)` 为 0.5918872948；九对旧/终版的最大值为 0.3560183696，各对共同前缀包络与实际分歧步的并列规则均通过。此数字如实记录，不将较大的生产波动称为逐位确定；逐位证据来自第一项相同诊断构建。原始 logits 和 `production-gate-b-summary.json` 全部保留。额外 all-six gap 统计不是既定分歧步判据，未用来扩大旧版包络；旧版实际单换行翻转亦按既定规则通过，未作空条件通过。单次 262K 约四分钟，补充性每版十次频率统计未做。
- **诊断撤回和恢复生产构建**：1081 份源码全部恢复原始字节，临时头文件不存在，v9 的 66 份源码及原有 313 份 ops 字节均匹配。第一次外部清理构建因 Python→cmd 引号传递错误而未执行成功，`gate-session.json` 的失败与 `final-clean-build.log` 保留；随后通过独立 `.cmd` 文件完成生产 CLI/serve 构建，66.31 s、exit0、所属进程树排空，源码再次核对不变。`final-clean-build-recovery-v1.json` 绑定原失败记录，恢复日志 SHA `cb9421d2bcbdebb8b17c437bb1028a221acd539f23ec9162adf4d605e8622281`。root 汇总证据为仓库外 `4.4-root-dense-composite-completed-v2.json`；CLI/serve SHA 分别为 `575eb41c0a0034ca73d3b89a395a2e876eccf5921d0aaae7b65f6524d1234a14` / `52724ce0034ecb77813380c4a8334a60c6a6047dd0d75545f104592b097a3754`。该通过只覆盖 dense 组合门禁，最终前端计数、合成/对照/复用及终版完整 owner SAN 仍待完成，4.4 未提交或推送。


- **终版前端计数完成**：`4.4-final-frontend-counts.json` 的 12 份冻结输入均由实际 serve 计数，六个单文档输入分别为 131008/262080 token（max_context 减 64），六个历史/短 query 输入为 130485/261557 token；计数完成且服务所属进程树排空。生产身份为 586 份生产源码聚合 SHA `2bb5c9114e048ef51105bed78946710d4e4e1d5466832e196da4af67124d6296`，CLI/serve 与上项恢复构建匹配。外部复用 harness 构建完成，绑定 v9 的 66 份源码、全部实际 runtime libs 与生产身份，exe SHA `699daf28f98b1f0cb0f11566ea6f2d008ab807495113eccb6228fd63d80ab8c3`；尚未执行其模型复用测试。
- **72 组矩阵第一次启动被外部解析错误中止**：`final-v9-20261005-matrix72` 的首例模型 exit0、完成 64 token，原始输出正确；但脚本仅匹配小写 `graph=off`，未识别实际已有的 `kvmem CUDA Graph=off (eager), disk prompt cache=off`，记录为 failed 并停止。原始输出与错误记录保留；没有修改生产源码、日志文本或放宽门禁。外部 parser 修正为匹配这条实际完整启动日志，离线重读原始输出通过全部字段核对，并证明 Graph=on、其他模式及删掉日志的三种文本仍拒绝。该离线解析不替代新模型运行，`4.4-external-graph-log-parser-correction-v2.json` 记录旧/new 脚本 SHA 与原脚本归档。现以新文件夹 `final-v9-20261005b-matrix72` 重跑全部 72 例，随后是 `final-v9-20261005b-references12`；最终合成矩阵、参考比较、复用与完整 owner SAN 仍未验收，4.4 未提交或推送。


- **终版串行验收继续（中间状态，非完成）**：修正外部解析后，新矩阵前 7/72 例均已完成并答对 needle；其中 128K 上下文 / 32K 视图首例 prefill 78.884 s、decode 116.81 tok/s、MTP 45/51=88.235%，实际视图32768 token。请求128K视图的本机实际容量为124608 token，不能把请求上限写为实际分配。这些只是中间单例，不作为质量矩阵或性能门禁。`4.4-root-active-final-validation-20261005-v2.json` 记录当前 exec sessions 和排他 lane；`run_remaining_final_gates_v1.py` 在全部72组与12组参考完成后依次汇总、运行两种分母下的六份实际复用 fixture、重新全量构建/CTest、冻结完整终版源码/owner，再跑13.0.85完整owner三项SAN。任何必需检查失败即保留证据并停止；没有过滤测试/内核，没有放宽数值判据，racecheck使用48小时外部上限，超时不算通过。串行脚本不会修改源码、提交、push或开始4.5；真实多文件/工具回放与Graph/capture/performance继续明确未验收。


- **4.4 合成矩阵与对照全部完成**：`final-v9-20261005b-matrix72` 为 72/72 完成、needle 72/72 正确；`final-v9-20261005b-references12` 为 12/12 完成、needle 12/12 正确。128K/262K × 请求视图128K/32K × 10%/50%/90% × 两种分母 × 三次独立运行齐备，汇总为24组，每组页统计、TTFT分段、hydrate、decode、MTP、prefill/decode ready等待及参考结果均保留。`4.4-final-v9-synthetic-summary-v1.json` SHA `49323bba9f3f5d489d97f03e2444adaf141e82a0da991432a649f5d983ca0dcb` 绑定84份结果与原始stdout/stderr SHA、冻结输入和586份生产源码/CLI/serve身份。此为**合成**初步数据；真实多文件、工具回放与Graph/capture/performance仍未验收，不能称阶段4整体验收完成。
- **复用队列第一次停止及配置核对**：首例`history-128k-10`实际四轮已全部运行，append/checkpoint路径均正确、needle均正确、rollback/cold全部64 token逐位相同；但不可把裸探针pass当作整个测量pass。外部不可变harness明确`preserve_thinking=true`，初版count payload用默认false，导致首轮prompt为130489，而原count为130485；队列因严格计数检查中止，未跑后续CTest/SAN，原`4.4-root-remaining-final-gates-v1.json`与失败measurement/log保留。模板在保留历史assistant思考时新增空think包装；未改生产代码，未用固定加4放宽计数。
- **按实际相同策略重计数并恢复队列**：`validate_reuse_frontend_counts_v2.py`经实际serve显式设置preserve=true，12份输入全部完成且所属进程排空；单文档六份计数不变，历史六份分别为130489/261561，恰比默认策略多4且与不可变harness真实首轮一致。`4.4-reuse-final-frontend-counts-preserve-v2.json`记录完整prompt_policy和生产身份；`4.4-reuse-count-policy-correction-v2.json`保存前后计数、原失败及离线结构/完整token核对。离线核对不替代正式重跑。新`run_sparse_reuse_validation_v3.py`只用相同策略的实际count，并在每份measurement绑定count SHA与policy；`run_remaining_final_gates_v2.py`使用新目录`reuse-final-v9-20261005c-kept`/`all`重跑两种分母的六份fixture，后续保持完整CTest和13.0.85全owner三项SAN、48h外部racecheck上限。当前kept首例正式重跑已通过，其余继续；源码、原harness及所有数值判据不变，4.4仍未提交/push。


### 阶段 4.4 收尾与 sanitizer 策略修订（2026-10-06，用户批准）

- 终版正式复用／回滚已完成：`reuse-final-v9-20261005c-kept` 与 `all` 各六份实际模型 fixture，共 **12/12 通过**；覆盖 128K/262K、10%/50%/90% needle、两种 softmax 分母。每份依次生成 first 16、append 64、rollback 64、cold 64 个 token；第二轮确实复用前缀，追加只计算 51 个 token，checkpoint 回滚只计算 4 个 token；所有 needle 正确，rollback/cold 的完整 64 ID 逐位一致，无需放宽门禁 B。preserve_thinking=true 的实际前端计数证据单独绑定，原默认策略计数失败及修正记录保留。
- 最终全量构建与 CTest 完成：**114 项中 110 通过、4 专用模型缺失跳过、0 失败，447.81 s**。`4.4-final-v9-full-build-ctest-v2.log` SHA `42a934b3c9a1e8f6415b32a390d85d259e1db4a713cb491be0e7c4860ec667c2`。此前 v1 漏传 `NINFER_TEST_VISION_GGUF` 导致 CPU/CUDA 两个真实视觉测试额外跳过，外部队列按规则停止；v2 补齐同一 GGUF 路径后完整重跑，两个视觉测试通过，未修改源码或跳过集合判据。
- 终版 owner 及 747 份源码已不可变归档到 `C:/Users/aurawing/AppData/Local/NInfer/kvmem-stage4-sparse/owner-final-v9-20261005-immutable`；owner SHA `b36adb3a905213fdc6bf8c73f901abc708cdefcfa1bf50331b98e986cc73fe0e`，manifest SHA `4eed32b6338f89b4e692b3fad87c49ddce4d4e79dbac3d3bdad065629e9d4456`。同一程序的 compute-sanitizer 13.0.85 **synccheck 完整通过（87.300 s）、initcheck 完整通过（198.963 s），各 0 error**。
- 用户授权停止完整 owner racecheck：实测 **94627.434 s（约 26.29 小时）**，前 **13 个 fixture** 完成，停在 `fragmented_next_exact`。截至中止部分日志报告 **0 hazard**，但没有最终 summary，记录为**用户批准中止，非完整通过**。停止原因：**端到端规模的插桩成本不可接受**。只终止本次所属进程树，已确认 supervisor/child 全部排空；原 racecheck 日志、原失败退出 JSON、停止前后副本和 SHA 均保留在仓库外 `owner-racecheck-user-stop-20261006/`，不将中止状态改写为 PASS。
- **审阅决定、用户批准的策略修订**：racecheck 仅作内核级小 shape 门禁；owner/e2e 只作完整 synccheck/initcheck。逐项建立实际内核／模板配置与已通过 racecheck 的单元测试对照表，核对部分末页、碎片访问列表、T/split、BF16/INT8/rk4 plane；有缺口补小 shape 测试。可选 --only-* 子集限 1 小时、不作为门禁，本轮暂不执行。该修改是在约 26 小时插桩运行之后提出，不改变 D1–D15、数值 oracle 或 dense 组合门禁；不冒充原完整 racecheck 完成。
- 4.4 的 72 个合成单文档用例及 12 个参考用例全部 needle 正确，三个 dense 组合门禁均已完成；当前继续核对内核级 racecheck 对照与补缺，尚未提交/push，不开始 4.5。完整真实多文件／工具回放质量矩阵、CUDA Graph 与 capture 性能均仍未验收。

- 全部 24 组合的页数／字节、TTFT 每段、decode／MTP、prefill/decode ready wait，以及12份复用fixture详细数据已整理到 [合成 eager 实测](stage4.4-synthetic-eager-results.zh-CN.md)；仅为合成初步结果，不作为真实质量矩阵验收。

## 阶段4.4终版验收（2026-10-06，已提交发布）

- 当前终版保持 v9 生产源码、CLI、serve 与 owner 二进制身份不变；新增覆盖只修改4个测试/CMake文件。最终68路径冻结清单 `4.4-root-revised-kernel-test-freeze-v2-20261006.json` SHA `572d8ee3f4c42d0500e88b8019c55c77c9a901dbfc991d542c6a9a4bbb9cd17a`；原313个ops文件逐字节不变。没有提交临时确定性补丁、外部owner标记副本、测量脚本、模型或日志。
- 完整实现此前已通过v9规格及质量审阅；本轮测试增量再次通过独立SPEC→QUALITY，报告 `4.4-revised-kernel-test-spec-20261006.md` SHA `d4115871fbe08920730e7b5422c2fbee1c1e0391534d3d3c5c3eb849e6ae5de8`、`4.4-revised-kernel-test-quality-20261006.md` SHA `4a49eba06912525d1e1af536e6ae674dc02b603f55bd3a5fc5f9d57354a6af63`。首次SPEC发现测试尾缓冲在默认流填充后交给非阻塞流，已改为同流checked memset并重跑完整Mean单测；这是源码确认的测试排序缺陷，不伪称racecheck发现hazard。旧候选证据保留。
- 最终全量构建与无过滤CTest：**115项，111通过、4项缺少其他模型制品跳过、0失败**；构建+测试总墙钟 543.611 s。跳过仍只有27B prefix及35B A3B、35B DFlash real/load-plan四项；本机Qwen3.8模型与外挂视觉GGUF相关用例均执行。记录 `4.4-final-revised-policy-full-ctest-20261006.json`，完整日志SHA `bfea047d8501eb332ceb3ebb091297c75fcb86cfdced94fb0d9dc5e62189cec8`，受控进程树全部排空。此前114项完整通过和漏GGUF导致多跳过的失败记录均保留。
- 内核覆盖：完整外部标记副本采集891次成功owner调用，核对22个实际attention格式/T/split元组、66个append配置、7个实际评分域与30个Mean-K offset/有效行/恢复配置。新单测保留Qwen27B几何、实际T/split、逻辑空洞和13-token尾页，仅缩小key/物理页数量。三个完整无参数、无kernel过滤racecheck均为0 hazard / 0 error / 0 warning：kernel覆盖46.766 s、scoring27.422 s、修正后Mean7.640 s。每个内核/config的表与完整SHA/命令见 [覆盖对照](stage4.4-kernel-racecheck-coverage.zh-CN.md)。DMA明确为非kernel，owner的Main provisional fixture不冒充真实MTP/GDN覆盖。
- 同一原owner的完整synccheck/initcheck继续有效，分别87.300/198.963 s、0error；约26.29小时的原owner racecheck中止仍**不是完整PASS**，按用户批准的新门禁归类为非门禁诊断。原26小时部分日志SHA `4852ff13ae081d71ef4e1cfb3f36ba3a7aaa9f5c44767daf53728c002c23179b` 与停止证据保留；本轮不补跑可选owner子集。
- dense三项组合全部通过：旧实际录制源码b68009f7与终版使用同一临时Q5 kSplits=1诊断，9用例每版各两次，共36次；自身及新旧版所有64生成步的完整词表248077 logits逐位相同。10对加载计划包括默认chunk检查完全一致，dense仍默认1024。生产构建换行分歧按固定旧版包络/并列判据通过，九配对最大相对L2 `0.3560183696≤0.5918872948`，旧版差距波动0.125；不宣称生产构建逐位确定。1081份诊断相关源码已恢复，旧失败与事后修订记录未覆盖。
- 合成矩阵72/72 needle正确、12/12独立参考正确；12/12结构化复用fixture正确，追加只prefill51 token、回滚只prefill4 token，rollback/cold全部64 ID相同。完整24组合页数/diff/hydrate、TTFT各段、decode/MTP、prefill/decode ready wait及实际资源账本见 [合成eager实测](stage4.4-synthetic-eager-results.zh-CN.md)。合成短输出不代表真实多文件/工具质量或长生成性能。
- 根验收索引 `4.4-root-revised-policy-final-gates-20261006.json` SHA `41e5fc0d886a3b4df30dcd553f6341889dc4d1eb36ecd028351cdbb5133eb490` 绑定最终冻结、审阅、CTest、三项dense、模型与复用、完整owner sync/init、停止诊断及三个小unit racecheck原始身份；证据均在仓库外目录 `D:/deeplearning/NInfer/logs/kvmem-stage4-sparse`。
- 边界：仅C=1、eager、Graph=off，exact prefill后发布Main sparse selection；retained resume/checkpoint继续可用，磁盘缓存关闭。CUDA Graph descriptor、ordinary/MTP capture及性能门禁**未实施/未验收**，真实多文件/工具回放完整质量矩阵**未验收**；没有开始4.5，没有新的D1–D15冲突需决定。提交/push完成后在4.4停审。

### 4.4提交与推送结果（2026-10-06）

- 实现及测试：`9404c865ee32bed9d4cdd731245f5034c3101cc6` — `feat(kvmem): connect sparse eager execution and recovery`，68个源码/测试文件。
- 用户说明、验收实测、内核覆盖与进度：`fb0d12bac0f924b5e89053ae1e9b4992f4703b6b` — `docs(kvmem): record sparse eager gates and sanitizer policy`，8份文档。
- 两提交已成功push到 `origin/feat/kvmem`；首次push后独立 `ls-remote` 核对远端HEAD为 `fb0d12bac0f924b5e89053ae1e9b4992f4703b6b`，本地HEAD与tracking一致、工作区干净。原始发布凭据 `4.4-publication-first-push-20261006.json` SHA `9ecc12b722e91f9c219e0d76bc1630ad062ec7d307384ffaacf54dfcde61d523`。本段仅补记已完成的实际发布元数据，并随同分支提交；最终HEAD及再推送核对见仓库历史和仓库外 `4.4-final-publication-checkpoint-20261006.json`。
- 终版文档收尾再次通过独立SPEC→QUALITY；报告 `4.4-final-docdelta-spec-20261006.md` SHA `57bde4e7cfc4291a4ea27b7022acfd093a789b5357d5b96f617ec36e26556b95`、`4.4-final-docdelta-quality-20261006.md` SHA `61a92d0aa15d452cf8e17bbaef3b1b667c4c784c3e4a1e29a1a5371f1bffbeb0`。发布过程未改变已测的生产源码、原313个ops或受保护二进制。
- **4.4已完成并停审**。未开始4.5；CUDA Graph/capture性能与真实多文件/工具回放质量仍未验收。完整owner racecheck保留为约26小时的用户中止诊断，不作为完整PASS。

### 4.5 / 4.6 启动（2026-10-06）

- 4.4 用户审阅通过；授权顺序为合成质量 4.5、Graph 及性能 4.6，两步完成后停审，见 [本轮计划](stage4-quality-graph-plan.zh-CN.md)。
- 4.5 每类每档目标 10 case（5 seed × 两种输入形态），冻结完整输入及 truth/SHA；GPU 预算约 24 小时，按耗时从低优先级削减，保留跳过清单。
- 用户补充决定：信息性 rk4 档请求与 INT8 相同的 32K / 128K token 视图，记录实际分配值；不使用同显存预算的双倍请求。
- 本轮中间记录及证据统一放仓库外 `D:/deeplearning/NInfer/logs/kvmem-stage4-5-6`。当前是准备阶段，没有模型质量或性能结论，未修改默认分母、dense 内核或分派。

### 4.5 终版门禁结果（2026-10-06，未验收）

- 工具实现通过独立 SPEC v3 / QUALITY v2；120 份主矩阵输入及 3 份不重叠成本试跑输入已完成实际 tokenizer 长度校验，尚未生成主矩阵答案。
- 全量构建/CTest：117 项，113 通过、4 项其他模型缺失跳过、0 失败；三个小 shape 内核 racecheck 均 0 hazard，完整 owner synccheck/initcheck 均 0 error。证据分别见外部 `4-5-final-v5-full-ctest/completed.json` 与 `4-5-final-v5-sanitizers-v2/summary.json`。
- dense 确定性诊断：9 用例每版两次，共 36 次；自身重复与旧版/终版的完整词表 logits 全部逐位相同，0 不同有效元素。10 对生产加载计划相同，dense 默认 chunk 与分派保持不变。
- 生产算法旧版/终版各 3 次，所有 64 个输出 ID 相同；九配对最大相对 L2 为 `0.4037602120`，小于固定旧版包络 `0.8199316780`，本轮实际配对的包络与分歧检查通过。
- **历史换行分歧的并列证据未通过**：`needle-262k-10` 第 9 步六次都选双换行 ID 271，未选单换行 ID 198；两候选为 top-2，差距均为 `0.125`，旧版三次观察到的差距波动为 `0`，不能据此证明历史分歧属于并列。因此组合门禁总结果为失败；不将“本轮输出一致”替代用户批准的三项组合判据，不追加抽样直到通过，不放宽判据。
- 串行流水线已在该门禁停止；合成成本试跑、成本裁剪、正式质量矩阵及 4.6 尚未开始，4.5 未提交/push。此前 4.4 的通过记录保留，本轮失败不覆盖它。
- 临时诊断已撤回：1091 文件 SHA 核对全部恢复、临时头文件不存在，生产 CLI/serve 清理重建完成。所有原始运行与失败结果保留在外部 `dense-composite-gate-4.5/`；正式判定为 `production-gate-b-summary.json`，总状态为 `gate-session.json`，停止记录为 `final-v5-pipeline-v2/dense-composite-completed.json`。

### 4.5 dense 范围澄清与恢复（2026-10-06，审阅决定）

- 本段追加于上述失败之后，原失败与全部原始日志不改写。审阅明确第三项只判本轮九配对，不再要求重现/证明阶段 3 首轮的历史翻转，README 和偏差表已同步；数值阈值、实际分歧规则及前两项门禁不变。
- 不重跑模型，按已完成的旧版/终版各三次数据核对：六次 64 ID 全同，九配对完整词表最大相对 L2 `0.4037602120<0.8199316780`，本轮没有需要并列判定的实际分歧。按澄清后的范围，第三项及整个 dense 组合门禁通过。
- `needle-262k-10` 第 9 步：旧版三次的 `(ID271, ID198)` logits 为 `(23.75,23.625)`、`(23.875,23.75)`、`(23.625,23.5)`；终版三次均为 `(23.875,23.75)`。全部在 `[16,32)`，BF16 该区间 ulp=`2^(4-7)=0.125`，差距恰为 1 ulp；这是补充说明，不替代本轮包络与分歧规则。
- 新判定另存仓库外 `review-clarification-20261006/` 与 `dense-composite-clarified-receipt-20261006.json`，绑定原失败/诊断/计划/六次实际运行及恢复的 SHA；恢复队列只执行冻结、成本试跑、成本裁剪、质量主矩阵，不重跑 CTest、sanitizer 或 dense 测量。合成质量与 4.6 结果仍待后续执行，尚未提交/push。

### 实验版收口（2026-10-06，用户一小时时限决定）

- 本决定覆盖之前未完成的 4.5/4.6 指令；发布后停止开发，不继续矩阵、Graph 或长 GPU 验证。
- 正式矩阵已停止，受控 GPU 子进程排空；原因是用户时间与噪音约束，原日志、答案、冻结输入和 SHA 不变。
- 4.5 状态：**部分完成**。指定主汇总 **合成 8/240、提前终止、不构成质量验收**，冻结严格校验 4/8 通过、4/8 失败。
- 停止时实际已完成 12 次，额外 4 次单列保留，严格校验均失败；计划余下 228 次未运行，不丢弃任何已完成数据。
- 所有完成项仅为 262K/32K/INT8/all-committed/seed0；完整配对、参考、多 seed、128K 上下文/视图、rk4 信息档、摘要质量矩阵未完成。
- 默认分母未修改，部分答案不能给出参考质量差距；真实多文件与工具回放仍未测。
- 4.6 状态：**推迟**。本版 KVMem 仅 eager；Graph descriptor、ordinary/MTP capture 和性能门禁未实施/未验收。
- `kvmem` 与 `tiered-exact` 均标为**实验性**，默认 `--kv-mode dense`；C=1，稀疏 Graph/disk cache 关闭。
- 推荐 262K INT8 请求 32K/128K 视图；4.4 合成 64-token eager 实测分别 118.10–135.14 / 107.59–122.90 tok/s，128K 请求实际115840。
- 门禁源码 8 路径与生产树 SHA 均与 4.5 终版相同；生产 SHA `9fe2fea7abefa2b9447adf30676fbd7943eef1dc9f1bd43571326921a04168a2`。
- CLI/serve Release 与冻结产品 SHA 一致，复用原构建、不重新编译；117 CTest（113通过/4其他模型缺失跳过/0失败）和 dense 三项组合门禁继续有效。
- 收口仅新增文档与独立打包启动脚本，未改生产源码、dense 内核/分派或已测质量工具；启动脚本 PowerShell 语法检查通过。
- 不重跑全量 CTest、sanitizer 或 dense 门禁；原失败 gate-session 等继续保留，原历史失败不改写为通过。
- 包内依赖隔离路径冒烟仅两次：dense短生成 **13.25 s / OK**；KVMem262K/32K/INT8/MTP-3短生成 **14.813 s / OK**，均180秒硬超时。
- KVMem冒烟日志确认实际view32768、Graph=off、pinned归档约8.25GiB；仅验证配置容量与短生成，并未填满262K上下文。
- 已测质量工具提交 `faccb40f`；发布tag：`kvmem-v0.1-experimental`，分支 `feat/kvmem`；发布提交号与远端核对见 Git/tag 和外部发布凭据。
- 包路径：`D:/deeplearning/NInfer/releases/ninfer-rtx4090-kvmem-v0.1-experimental.zip`；SHA256见包旁 `.zip.sha256`，避免包内自引用校验值。
- 证据索引：`D:/deeplearning/NInfer/logs/kvmem-stage4-5-6/release-experimental-20261006/` 的 equivalence、partial、stop、smoke、release-receipt JSON。
- 文档：[发布用法](release-v0.1-experimental.zh-CN.md)、[部分合成结果](stage4.5-partial-synthetic-results.zh-CN.md)；模型、API KEY、原始语料及日志不入仓库或包。

## 2026-10-07：多图请求预算修复与本地重打包

- 用户授权修改和重新编译打包；基于 feat/kvmem 的 79808d9e，本轮未提交、未 push、未新建 tag。
- 将单个编码项容量与请求累计预算分开：`--vision-max-tokens 8192`；新增 CLI/serve 一致的 `--vision-request-max-tokens 32768`，包括历史图片。
- 保留媒体项数、解码像素、总上下文限制及单项检查；单图按原容量缩放，累计超限仍拒绝并指出对应参数。
- GPU 编码按项复用原工作区；未改 dense/视觉内核、dense 分派、MTP、Main/MTP pool 或 KVMem 选择算法。
- Release 重新编译；6 个相关 CTest 全通过、0 失败，合计 63.20 秒。未扩大到全量 GPU CTest、sanitizer 或 dense 组合门禁。
- 大图 CPU fixture：两张 4096×2048，每张 8192 token、累计 16384、raw patches 65536，最终 FP32 patch 缓冲 402653184 字节。
- GPU 冒烟使用用户的 Uncensored 模型、256K/INT8、请求128K视图、内嵌CUDA视觉、MTP-3；实际视图仍101248，预算未随累计上限增加。
- 两轮均回答正确：红图后带历史再发蓝图；第二轮 prompt16445、cached8223、reuse=append_frontier，说明历史 KV 正常复用。
- 本次大图 HTTP 耗时123.765/99.406秒；整组239.109秒。属于功能冒烟，未做性能门禁，处理时间波动不作性能结论。
- 首次冒烟漏用启动配置的600秒准备超时、返回503；原失败保留。补齐配置后上述两轮通过，不改写失败日志。
- 新包目录：D:/deeplearning/NInfer/releases/ninfer-rtx4090-kvmem-v0.1-experimental-multiimage-20261007；ZIP及SHA256旁置。
- 根目录 start-kvmem-256k-128k-gpu-vision.ps1 更新到新包，单张8192/累计32768；旧脚本备份在证据目录，原发布包未覆盖。
- 证据：D:/deeplearning/NInfer/logs/vision-multi-image-fix-20261007/（build-final、ctest、validation、smoke-results、package-receipt）。
- 用法：[多图视觉预算](../vision-multi-image-budget.zh-CN.md)。KVMem实验性、4.5未验收与4.6推迟状态保持原发布结论。
