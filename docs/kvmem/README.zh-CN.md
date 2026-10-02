# KVMem 分层 KV：RTX 4090 实施交接说明

本文是写给接手开发的 AI 助手的任务说明，内容自成一体。背景和推导见同目录的 [设计文档](design-4060-bonsai-kvmem.zh-CN.md)，与本任务直接相关的是 5.6、5.10、6、7、8 节。

设计文档是在另一台机器（RTX 4060 Laptop）上写的，里面形如 `D:\VideCoding\NInfer_KVMem\...` 的本地路径在这台机器上不存在。需要参考资料时，一律使用第 9 节的公开链接。

---

## 1. 任务

在本仓库（NInfer）里为 Qwen3.8-27B（groupwise-int 制品）实现 KVMem 式的分层 KV：

- 完整的 262K 上下文 KV 放在主机内存的归档里，GPU 上只保留一个"执行视图"（约 128K int8）。
- 新增 `tiered-exact` 模式：注意力对完整历史做精确计算，GPU 上没有的页流式送入，分块计算后用 LSE 合并。这是质量基准。
- 新增 `kvmem` 模式：prefill 精确计算，每轮按 Mean-K 打分选块，decode 只看视图。这是日常使用的模式。
- 默认的 `dense` 模式行为不变，输出逐位不变。

目的：现在 262K 只能用 rk4v4-e8（K、V 各 4 bit），满窗时仅剩约 300 MiB 显存。做完之后，262K 可以用 int8 KV。

分支：`feat/kvmem`，从 `221290ba`（`feat/vision-cpu-ggml`）切出，不带三值补丁。

### 1.1 范围

做：

- 阶段 0′：基线测量；
- 阶段 3：主机归档、传输、`tiered-exact`、MTP 窗口化、两个复用点；
- 阶段 4：Mean-K、选块、视图、`kvmem` 模式、质量门禁。

不做：

- 三值 Bonsai 与 jonj20 补丁、方言适配层、40 系通用的显存求解器；
- NVMe 层、window prefill 与 query replay；
- kvmem 相关模式下并发大于 1；
- 视觉模块本身的改动。

---

## 2. 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24 GB，sm_89，128 SM，Windows WDDM，桌面接在这张卡上 |
| CPU | i5-10400，6 核 12 线程，AVX2，无 AVX-512，没有大小核 |
| 内存 | 32 GB DDR4 |
| 总线 | PCIe 3.0 x16（Comet Lake 不支持 4.0），理论 15.75 GB/s |
| 工具链 | MSVC、CUDA 13.3、vcpkg、Ninja；构建命令见 `docs/vision-weight-modes.zh-CN.md` 的"构建"一节；llama.cpp 固定在 `b81c99b4`（`NINFER_GGML_SOURCE_DIR`） |
| 模型 | `D:\deeplearning\NInfer\models\qwen3_8_27b.ninfer`（groupwise-int，内嵌视觉） |
| 视觉 GGUF | `D:\deeplearning\kvmem-v0.16.0-rc3-windows-x86_64-cuda13.2.86\models\Qwen3.8-27B\Qwen3.8-27B\mmproj-BF16.gguf` |
| 测试环境变量 | `NINFER_QWEN3_8_27B_WEIGHTS`、`NINFER_TEST_VISION_GGUF` |

现有基线（出自 `docs/vision-cpu-gguf.md` 和 `docs/vision-weight-modes.zh-CN.md`）：

- 配置为 262K、rk4v4-e8、MTP-3、单并发。
- 满窗 prefill 用时 215.7 s，约 1218 tok/s。
- 整卡峰值占用 23809 MiB，最低空闲 300–340 MiB。整卡采样包含桌面，并且各轮基线有漂移。

---

## 3. 已定的设计决策

没有实测证据时，不要推翻这些决策。如果实测结果与某条冲突，先停下来记进 `progress.zh-CN.md`，等用户决定。

| 编号 | 决策 | 出处 |
|---|---|---|
| D1 | 用 `--kv-mode dense\|tiered-exact\|kvmem` 选择模式，默认 `dense`。dense 路径的输出、显存、性能都不变 | 5.6.1、5.6.11 |
| D2 | 只对 Main Text pool（16 个全注意力层）分层。GDN 始终处理完整序列，不做稀疏 | 5.6.9 |
| D3 | 262K 以内不做 re-RoPE。K 已经在写入时做过 RoPE，掩码按缓存序号计算，所以量化页可以逐字节换入换出。**前提要先核对**：注意力内核和 frontier 逻辑都不能假设"第 i 页覆盖位置 64i 到 64i+63" | 5.6.2 |
| D4 | `--kvmem-host-archive auto\|pinned\|pageable`，默认 `auto`。加载期按 `max_context` 和 KV 类型计算整份归档大小：若可整块锁页且锁页后可用物理内存仍不少于 4 GiB，选锁页归档，H2D/回写直接传输；否则 `auto` 释放已分配部分，退回可分页归档（`VirtualAlloc` 预留、按需提交）与加载期 4×64 MiB 锁页中转环，`pinned` 则报错退出；`pageable` 直接选后者。日志写明实际路径。两种归档均按「层 → plane → 逻辑页」布局，连续页在每个 plane 上只需一次传输；运行期不新申请锁页内存。此项依据 262K 就绪后整块可锁页 15488 MiB（约 15.1 GiB）、可分页环端到端 4.19 GiB/s 与锁页直传 9.83 GiB/s 的实测修订 | 5.6.4 |
| D5 | 页的状态机为 DeviceOnly → Both → HostOnly，只有 Both 状态的页可以驱逐。每个 prefill 分块结束、以及 decode 中每页写满时，都异步回写 | 5.6.3、5.6.4 |
| D6 | `tiered-exact`：部分注意力输出 `(O, m, l)`，对双缓冲的 tile 做在线 LSE 合并。合并的数学可以复用 decode 已有的 split-K 合并 | 5.6.7 |
| D7 | `kvmem`：精确 prefill，每轮选一次块，然后稀疏 decode。不做 query replay（`--kvmem-prefill window` 属于阶段 5，现在直接拒绝） | 5.6.6 |
| D8 | Mean-K：在 RoPE 之前，按页、层、kv 头求 K 的均值，以 FP16 存在主机，每 token 512 B。打分在 CPU 上做，用 KVMem 的全局 softmax，取本轮用户输入的最后 16 个 query token | 5.6.5 |
| D9 | 选块：sink、最近 R 个 token、本轮输入所在的页、被引用的图像跨度为必选，其余按打分取 top-k，按原始顺序排列，只换入差分部分。一张图的跨度要么整体入选，要么整体不选 | 5.6.6、5.6.8 |
| D10 | 视图表独立于逻辑页表，由它生成现有的块表 | 5.6.3 |
| D11 | MTP pool 不进归档，只保留 sink 加最近窗口（`--kvmem-mtp-window`，默认 32K）。贪心输出必须与关闭 MTP 时逐 token 相同 | 5.6.9、5.10.3 |
| D12 | resume frontier 和 turn checkpoint 两个复用点从阶段 3 起就要支持：归档按逻辑 frontier 截断，GDN 状态沿用现有 checkpoint。磁盘状态缓存在 kvmem 模式下先关闭 | 5.10.3 |
| D13 | kvmem 与 tiered-exact 模式只支持 C=1；`--max-concurrency` 大于 1 时启动报错 | 5.10.3 |
| D14 | 归档精度等于 `--kv-dtype`，4090 上推荐 int8；bf16 只作为显式选项。启动时做主机内存准入：可用物理内存不少于"归档上限 + 4 GiB"，否则拒绝，并提示改用 int8 或 rk8v4。可选 `--kvmem-lock-archive`（`SetProcessWorkingSetSizeEx` 加 `VirtualLock`） | 5.10.2 |
| D15 | 主机侧工作线程最多 2 个（CPU 只有 6 核）。CPU 视觉在 prefill 之前同步执行，与这些线程不重叠 | 5.10.3 |

`--kvmem-lock-archive` 是 D14 原有的 `VirtualLock` 选项，仅适用于可分页归档；它不等同于 D4 中可直接异步传输的 CUDA 锁页归档，不能与 `--kvmem-host-archive pinned` 同时使用。设备 staging 在加载期按「一个全注意力层每个 prefill 分块的最大流式量 + 64 MiB」求容量（int8、128K 视图约 264 + 64 MiB），由 `build_workspace_plan()` 从视图显存预算中让出；前面几个 GDN 层计算时预取下一全注意力层，不在切层时整层等待。split 划分和 LSE 合并顺序只由访问列表与 frontier 决定，不由传输完成先后决定。

新增 CLI 参数（设计值，实现时可以调整默认值，但要记录下来）：

```text
--kv-mode dense|tiered-exact|kvmem
--kvmem-view-tokens <N>        4090 上推荐 131072
--kvmem-prefill exact          window 在阶段 5 前直接拒绝
--kvmem-sink-tokens 256
--kvmem-recent-tokens 8192
--kvmem-gen-reserve 6144
--kvmem-query-tokens 16
--kvmem-mtp-window 32768
--kvmem-host-archive auto|pinned|pageable
--kvmem-lock-archive
```

目标配置：

```text
ninfer-serve qwen3_8_27b.ninfer --max-context 262144 --kv-mode kvmem --kv-dtype int8 ^
  --kvmem-view-tokens 131072 --kvmem-prefill exact --kvmem-mtp-window 32768 --prefill-chunk 2048 ^
  --spec mtp --draft-tokens 3 ^
  --vision --vision-device cpu --vision-max-tokens 1024 --vision-cpu-memory-mib 2048 ^
  --max-concurrency 1
```

---

## 4. 代码地图

行号是 `221290ba` 时的近似值。

| 关注点 | 位置 |
|---|---|
| KV 设计契约 | `docs/maintainer/paged-kv-cache.md`：§2 frontier 粒度，§3 pool 集合与容量向量（Main / MTP / DFlash 分开），§10 Prefix reuse（约 L741）。其中"KV offload 是 non-goal"的条款要改写 |
| 页池与分配 | `src/core/paged_kv_cache.h`：`PagedKVPool`（约 L107）、`PagedKVAllocation`（约 L177，含 `trim_tokens`）、`resize_paged_kv_bundle`；实现在 `.cu`/`.cpp` |
| 页寻址 | `src/ops/kernel/paged_kv_address.cuh` |
| 页 gather/scatter | `src/ops/kernel/kv_paged_staging.cuh`（`paged_kv_cache.cu` 在用） |
| KV 精度与存储面 | `include/ninfer/types.h` 的 `KvCacheStorage`（约 L21）；`src/targets/qwen3_6/impl/state/decoder_state.cpp` 的 `plan_cache`（约 L42–75） |
| KV 写入 | `text_context_impl.h` 的 `TextContext::attn_mix`（约 L797）：rmsnorm → rope → `ops::gqa_attention`；prefill 在 `gqa_attention_prompt_launch` 里先 append 再做 attention |
| 注意力 API | `include/ninfer/ops/gqa_attention.h`、`src/ops/wrapper/gqa_attention.cpp` |
| prefill 注意力内核 | `src/ops/launcher/gqa_attention_prefill.cu`；`src/ops/kernel/gqa_attention_prefill_{bf16,i8,common}.cuh`。目前没有 split 和 LSE 输出 |
| decode 注意力内核 | `src/ops/launcher/gqa_attention_decode.cu`；`src/ops/kernel/gqa_attention_decode.cuh`。已有 split-K，`partial_m` / `partial_l` 的合并约在 L150–260，合并数学可以复用 |
| 层调度 | `text_context_impl.h` 的 `run_layers`（约 L973）；`text_prefill_impl.h` 的 `prefill_text_chunk` / `prefill_multimodal_chunk` |
| MTP | 创建：`layouts_impl.h` 的 `plan_decoder_state`（约 L111–141），以及 `decoder_state.cpp` 约 L86–91 的 `layout.mtp_kv`；读取：`text_context_impl.h` 约 L404–537；轮次：`program_impl.h` 约 L2117–2208 |
| frontier、截断与复用 | `program_impl.h`：`trim_sequence_kv`（约 L1239）、`resize_sequence_kv_entitlement`（约 L1172）、恢复 turn checkpoint（约 L455–761）、快照（约 L2663+）；`program.h` 的 `TurnCheckpoint`（L114）；`linear_state_slots.h` 的 `turn_checkpoint_state_slot` |
| 容量与准入 | `src/runtime/engine/kv_capacity.cpp` 的 `resolve_kv_capacity`（L77）；`include/ninfer/types.h` 的 headroom（约 L35–50）、`EngineOptions`（约 L81）；`layouts_impl.h` 的 `build_workspace_plan`（约 L237，随 prefill 分块变化）、`build_sequence_candidate`（约 L616） |
| 锁页内存 | `src/artifact/materializer.cpp`：`kSlotBytes = 64 MiB`、`kMaximumSlotCount = 4`、`PinnedHostBuffer`；`src/core/arena.cu` 里的 `PinnedHostBuffer`（仓库中唯一的 `cudaMallocHost`） |
| CLI | `apps/cli/options.cpp`、`apps/cli/options.h`、`src/serve/serve_options.cpp` |
| 测试 | `tests/ops/test_gqa_attention.cpp`（FP64 oracle `ideal_attention`，约 L432）；`tests/CMakeLists.txt` 的 `ninfer_add_test`（约 L9）、`ninfer_add_op_test`（约 L270） |

放新代码的建议（最终以 `AGENTS.md` 的分层为准）：

- 通用部分放 `src/core/kvmem/`：归档、传输环、Mean-K 主机索引、打分与选块、视图表；
- 新内核放 `src/ops/`：部分注意力、LSE 合并、Mean-K 累加；
- 目标相关的接线放 `src/targets/qwen3_6/impl/runtime/`。

**给以后的三值合并留余地。** 下面这些位置是 jonj20 三值补丁要改的，KVMem 不需要改，请不要动：

- LM head 的 `ops::linear` 调用行：`text_prefill_impl.h` 约 L145，`text_context_impl.h` 约 L566、L671、L729、L1174，`dflash_impl.h` 约 L287；
- `src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp}`；
- `src/ops/linear/`、`src/ops/wrapper/{embedding,attn_input_proj,gdn_input_proj,linear_add,linear_swiglu}.cpp`；
- `src/artifact/{binder,reader,storage_layouts,typed_binding}.*`。

---

## 5. 阶段、任务与验收

### 阶段 0′：基线（1–2 天）

1. 从 `221290ba` 切出 `feat/kvmem`；按用户授权构建，跑全量 CTest，记录通过和跳过的数量（`221290ba` 时为 96 项：92 项通过，4 项跳过）。
2. 记录 dense 金标准。配置为 rk4v4-e8、MTP-3、C=1，上下文分 32K、128K、262K 三档，使用固定的合成长提示词和一组 needle 提示词。记录：
   - 前 64 个贪心 token；
   - 每步 logits 的摘要（top-8 id 与 FP32 值）；
   - prefill 时间；
   - 就绪后 `cudaMemGetInfo` 报告的空闲显存。

   数据放在仓库之外，路径写进进度文件。
3. 微基准，每项单独开进程，等 262K 配置的模型服务就绪后再测：
   - 最大能分配的单块锁页内存（二分查找，记录相邻的成功/失败边界）和此时的可用物理内存；
   - 锁页与可分页内存的 H2D/D2H 带宽；
   - 用 1 个和 2 个线程分别做 64 MiB "可分页 → 锁页"memcpy 的吞吐；
   - 64 MiB 块进入 4 × 64 MiB 锁页环、与异步 H2D 双缓冲重叠的端到端吞吐；
   - 典型负载下的可用物理内存。
4. **核对 D3 的前提**：通读 prefill 和 decode 注意力内核，以及 frontier 相关代码，列出所有隐含"页号 × 64 = 位置"的地方。
5. 退出标准：以上数据和核对结论写进 `progress.zh-CN.md`。

### 阶段 3：分层 + 精确（2.5–3.5 周）

保留原始逻辑页号的访问列表、设备 staging、split-K 与状态恢复方案见 [阶段 3 视图设计](stage3-view-design.zh-CN.md)。

建议按以下顺序推进，每一步都带测试：

1. **主机归档与回写**：先让 `tiered-exact` 在视图覆盖全部上下文时运行，这时归档只是设备页的镜像，和设备页比对，要求逐字节相同。
2. **传输引擎**：可分页归档的中转环、锁页归档的直传、`kv_stage_stream`、跨层预取与事件同步；按真实多 plane 布局分别实测两种路径的「归档 → 设备 staging」吞吐。
3. **内核**：prefill 注意力的"部分输出"变体，输出 `(O, m, l)`；合并内核。先覆盖 bf16、int8、rk4v4-e8，再补其余精度。
4. **接线 `tiered-exact`**：视图固定为 sink 加最近的 token，更早的页流式送入。decode 也走流式路径，只用于验证。按 32K（强制小视图）→ 128K → 262K 逐级验证。
   - 本步限定 C=1；每层每块按逻辑页升序发布一张 resident/staging 混合访问列表，prefill 一遍，decode 至多两遍。staging 在加载期按该层最大流式量加 64 MiB 规划；若后续使用容量不足的后备多遍路径，必须记录日志，不能让四遍 small-T 成为热路径。
   - 最小 CLI/serve 开关为 `--kv-mode tiered-exact`、`--kvmem-view-tokens`（别名 `--kvmem-view`）、`--kvmem-sink-tokens`（别名 `--kvmem-sink`）、`--kvmem-host-archive auto|pinned|pageable` 和可选 `--kvmem-staging-mib`。逻辑容量仍由 `--max-context` 决定，物理视图由 view 上限和统一显存预算确定；`--kv-capacity` 保留 dense/影子验证的原语义，不作为普通 tiered 的物理容量开关。
   - BF16 tiered 仅保证功能，不承诺性能。MTP pool 在本步保持 dense；turn checkpoint、retained resume、磁盘状态缓存显式关闭，分别待第 5、6 步实施窗口化和复用。
   - 加载期普通 tiered 的 Auto 预算默认预留 1 GiB，视图上限不保证全部分配。staging 必须容纳整层流式页加 64 MiB；固定覆盖不足时先提高最小视图，仍无可行预算则拒绝加载，不在 decode 热路径改成四遍。
   - 隐藏测试环境变量 `NINFER_KVMEM_SHADOW=1` 启用门禁 A；配合 `--kv-mode tiered-exact` 和强制小视图使用。`NINFER_KVMEM_TRANSFER_TIMING=1` 单独启用每层 ready 等待的 CUDA event 累计计时；默认关闭，无计时事件/同步开销。影子读回的 prefill 耗时不作为普通路径性能指标。
5. **MTP 窗口化**（D11）。
   - `--kvmem-mtp-window` 默认 32768，必须是正的 64 token 倍数。固定物理预算包含 sink、近期页和跨页临时草稿保护页；长上下文默认为 512 页（sink 4、recent 507、guard 1）。容量不足以容纳 sink、分块及保护页时加载前报错。MTP 不归档，保留原始位置，不做 re-RoPE。
   - 原始逻辑页标签构造访问列表，复用部分注意力内核；节省的 MTP 存储通过统一预算增加 Main 视图。启动打印新旧 MTP 字节及视图变化；旧视图估算明确标为 `old_main_view_tokens_estimate`，使用同一已确定显存预算。
   - 128K、262K INT8 各比较窗口 MTP 与关闭 MTP 的贪心输出，分歧按门禁 B 判定；needle 正确，接受率相对第 4 步完整池下降不超过 5 个百分点。`NINFER_KVMEM_TRANSFER_TIMING=1` 按 prefill/decode 分别累计每层 ready 等待。
6. **两个复用点**（D12）。
   - `tiered-exact` 恢复 retained resume 和 turn checkpoint。快照只记录 bundle 身份、精确 frontier、归档/视图 generation 以及 GDN、MTP、hidden、position continuation，不复制 KV 字节。checkpoint 在其实际 prefill 分块边界捕获；Main 与 MTP 各保存自己的 frontier，MTP 为后续桥接保留 `F−1` 边界。
   - restore 先排空旧传输，截断归档和视图，递增 generation；从归档重新换入 sink、最近页及当前部分写入页，再在同一执行边界发布块表、访问列表和前缀和。旧 ticket、staging 列表和异步完成通知不可再使用。
   - MTP 只保留快照标签与当前仍存活页的交集。被覆盖的历史不补算，日志说明可能影响草稿质量和接受率；Main 的完整历史仍由归档提供。磁盘状态缓存继续关闭。
   - 128K 和 262K 两轮对话记录 `reused_prompt_tokens` 与实际新增 prefill 数；checkpoint 回滚对相同输入的冷启动按门禁 B 判断。复用测试显式设 `preserve_thinking=true`，避免聊天模板删除历史 thinking 区域而改写已缓存前缀；前缀确实变化时继续由原有匹配规则选 checkpoint 或重算。
7. **准入与 CLI**（D13、D14），同时改写 `paged-kv-cache.md` 里的 non-goal 条款。
8. **性能**：262K int8 的 `tiered-exact` prefill。

退出标准：

- **A，同一次运行的影子注意力门禁**：新增仅用于测试、默认关闭的隐藏参数或环境变量；关闭时 dense 与 tiered 路径均不增加开销。开启时保留设备上的完整 dense KV，并把主机归档作为镜像。每个全注意力层对同一份 Q、同一份 KV 字节分别执行 dense 注意力和 `tiered-exact` 注意力；后者强制小视图，把视图外页视为未驻留，真正经过归档 → staging、部分注意力与 LSE 合并。下游始终使用 dense 输出，保证各层输入相同；此门禁不要求两次独立 dense 运行逐位相同。

  - **32K（约 8K 视图）与 128K（约 32K 视图）必做**：逐层输出相对 L2 `||tiered-dense||₂/max(||dense||₂, 1e-12)` 不超过 **1e-3**，同时记录最大绝对误差，原判据不变。
  - **262K `rk4v4-e8`**：完整 dense KV 加测试资源可放下时执行，否则记录实际显存缺口和原因。按影子相对 L2 选择误差最大的至少三个全注意力层，包含此前超限的层；每层至少取 prefill 末段一个完整块。用单元测试同一套 CPU FP64 oracle，从该块原始 BF16 Q 和同一份量化 KV 字节解码计算参考输出。每层分别报告 dense/FP64、tiered/FP64、tiered/dense 的相对 L2 和最大绝对误差；**各层的 tiered/FP64 相对 L2 必须不超过 dense/FP64，相对 L2 的参考范数均取 FP64 输出**。任何一层不满足就停止报告，不再调整判据。继续记录 tiered/dense 数值，不以它的 1e-3 为本档退出标准。用户在测得 0.0010094 之后，根据固定输入 alpha 消融证据同意此修订；dense 内核、分派和金标准不改，详见 progress 偏差表。

- **B，decode 门禁**：同一 262K 用例，MTP-3 分别运行 dense、`tiered-exact` 各 **3 次**，每次解码 64 个 token。在逐 token 的共同前缀上计算完整词表 logits 相对 L2，每步不超过 `max(1e-3, 2 × dense 三次两两比较所得的最大相对 L2)`；若出现分歧，记录首个分歧步、双方候选 ID 和对应运行的 logits，仅当该步 dense 的 top-1/top-2 差值不超过 dense 三次在该步观测到的最大差距波动时通过并列判据。每次 needle 都须答对，同时记录 MTP 接受率；分歧后不同生成轨迹的 logits 不作逐步误差比较。
- int8 归档能完整跑完 262K，prefill 时间不超过 dense 的 1.15 倍；
- 多轮复用正确；
- dense 模式的金标准不变。

### 阶段 4：稀疏 decode（2.5–3.5 周）

步骤：

1. Mean-K 累加，与 `ops::rope` 相邻或融合进去。
2. Q 捕获与 CPU 打分：主机侧逻辑可以从 kvmem-qw3 移植（`kvmem_store`、`kvmem_request_plan`、`pick_topk`）。
3. 视图表、计划差分与换入；页的状态机。
4. 图像跨度。
5. 稀疏 decode 与 gen_reserve 环。
6. 质量门禁：视图分 128K 和 32K 两档，32K 用来模拟 4060。

退出标准（设计文档 5.10.6）：

| 检查 | 判据 |
|---|---|
| kvmem int8（视图 128K）对 dense rk4v4-e8 | needle、多文件事实检索、工具回放的通过率不低于 dense |
| kvmem int8（视图 32K）对 tiered-exact int8 | 报告通过率，作为 4060 上的质量预期 |
| MTP 窗口化 | 贪心输出与关闭 MTP 时相同；接受率下降不超过 5 个百分点 |
| 多轮复用 | 第二轮只 prefill 新增 token；回滚后的输出与冷启动一致 |
| 主机内存 | 满窗期间硬页错误率不上升；内存不足时启动被明确拒绝 |
| decode | 满窗 decode 不慢于现在的 dense rk4v4-e8 |

质量评测集：needle 与多 needle 可以用脚本生成（放 `tools/`），深度覆盖 10%–90%。工具回放用的真实 agent 会话记录由用户提供（TODO）。

---

## 6. 测试要求

按仓库约定，每个新功能都要有 CTest，数值用独立的 FP32/FP64 oracle 核对。

| 测试 | 判据 |
|---|---|
| `test_host_kv_transfer` | 可分页 → 锁页 → 设备的往返、回写、并发，逐字节相等 |
| `test_attention_partial_lse_merge` | 多种 tile 划分、T 和掩码，与 FP64 全量注意力的相对 L2 不超过 1/256 |
| `test_meank_accumulate` | RoPE 之前的页内均值，包括不满的页，用 FP64 oracle |
| `test_kvmem_scoring` | 全局 softmax 打分与 top-k，包括必选页和图像跨度，与参考实现一致 |
| `test_kvmem_view_table` | 视图构建、计划差分、页状态机、只允许驱逐 Both 状态的页，不需要 GPU |
| `test_kvmem_mtp_window` | 窗口边界、页回收、provisional 草稿位置；贪心输出与关闭 MTP 时相同 |
| `test_kvmem_resume_checkpoint` | snapshot/restore 的 frontier/generation 一致、旧状态拒绝、追加只计算新增 token；回滚对冷启动的完整 logits 在独立端到端测量中按门禁 B 判定 |
| `test_host_archive_admission` | 内存不足时拒绝启动；提交量随 frontier 增长 |
| e2e `tiered_exact_vs_dense` | 同一 KV 精度，最长到 262K |
| 回归 | 全量 CTest 以及阶段 0′ 的 dense 金标准保持不变 |

---

## 7. 工作规范

- 遵守仓库根目录的 `AGENTS.md`。它要求未经明确指示不构建；构建和运行测试的授权以用户的指示为准。
- 只有用户要求时才提交，提交信息用 Conventional Commits，只提交到 `feat/kvmem`。
- 以下情况先问用户：
  - 要推翻 D1–D15 中的任何一条；
  - 要改变 dense 路径的行为；
  - 要动第 4 节列出的三值合并敏感位置；
  - 要新增第三方依赖；
  - 要 push 或 force-push。
- 每完成一个小步骤，就更新 `docs/kvmem/progress.zh-CN.md`：写清完成了什么、实测数据、与设计的偏差和原因、需要用户决定的问题。数字要写明是实测还是估算。
- 性能数据要记下命令行、提交号，以及当时是否有桌面负载。

---

## 8. 已知的风险与待实测项

| 项 | 说明 |
|---|---|
| prefill 分块 1024 还是 2048 | 2048 时流式流量减半，但 workspace 会变大，两者要实测后权衡 |
| 视图大小 | 131072 是估算值（4.2 GiB 预算）；MTP 窗口化后省下的约 238 MiB 能否加给视图，需要实测 |
| WDDM 锁页限额 | 常驻约 23 GB 显存时，锁页上限可能只有约 1.5 GiB，中转环必须在加载期确定 |
| 32 GB 内存 | int8 归档合计约 15–18 GB，宽裕；bf16 约 23–26 GB，偏紧，一旦换页会慢几个数量级 |
| D3 的前提 | 如果内核隐含"页号 × 64 = 位置"，视图的紧凑排列就要改成保留原始页号的块表，或者改内核 |
| 选块参数 | 在 groupwise-int 上调好的参数，以后到三值模型上要复测 |

---

## 9. 参考资料

- 完整设计文档：[design-4060-bonsai-kvmem.zh-CN.md](design-4060-bonsai-kvmem.zh-CN.md)
- KVMem 论文：[arXiv 2609.04852](https://arxiv.org/abs/2609.04852)
- KVMem 参考实现：[kvmem/kvmem-qw3](https://github.com/kvmem/kvmem-qw3)，提交 `1cf3b2f`，Apache-2.0。主机侧的 `kvmem_store`、`kvmem_request_plan`、`global_kv_page_pool`、`pinned_kv_tier` 可以借用；依赖 FlashInfer 和 Linux I/O 的部分要重写。
- jonj20 的 KVMem 评估：[jonj20/ninfer-ada-ternary-4060](https://github.com/jonj20/ninfer-ada-ternary-4060) 的 `docs/kvmem/4060-ninfer-KMEM-评估.md`
- WDDM 锁页的实测与限额：[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) 的 `src/core/host_kv_clamp.h`，以及 `src/models/qwen3_5/program/program_impl.cpp` 约 L363–387
