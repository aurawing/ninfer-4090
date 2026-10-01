# 阶段 3：保留原始逻辑页号的精确视图设计

状态：归档、传输引擎及独立部分注意力算子已实现，已通过第 3 步的全量验证，停在第 3 步审阅点；视图状态机及目标运行时接线尚未实现。适用于 `feat/kvmem` 上的 Qwen3.8-27B、单卡 RTX 4090、`C=1` 的 `tiered-exact`；`kvmem` 稀疏 decode 在阶段 4 复用视图和访问列表。本说明沿用 [README](README.zh-CN.md) 的 D1–D15。阶段 0′ 已确认当前内核把逻辑页 `i` 当成位置 `[64i,64i+63]`，所以不能把紧凑视图槽号直接送进现有 dense 注意力内核。

## 1. 页号、所有权与访问列表

`F` 是精确到 token 的 Main Text valid frontier；合法逻辑页为 `0..ceil(F/64)-1`。主机归档和视图表始终以**原始逻辑页号**为键。视图表记录每页的 DeviceOnly/Both/HostOnly 状态、驻留物理页号（若有）、有效 token 范围和未完成的传输事件。物理页号可以改变，逻辑页号、RoPE 坐标和缓存序号不随换入换出改变；已写入 K 的 RoPE 不重做。

视图表是 tiered 模式唯一的页映射来源，并在执行边界同时生成两种设备数据：

- 原有宽度为 `ceil(max_context/64)` 的块表仍按原始逻辑页号索引：驻留页写入其物理 ID，未驻留页写 `-1`。现有 KV append 只写已驻留的当前页，必须在发布执行前验证它有合法 ID。dense 模式仍由 `PagedKVAllocation::publish_mapping()` 发布原有连续块表，原有 dense 内核不改。
- 新注意力变体读取紧凑、按逻辑页号递增的 `{int32 logical_page, int32 physical_page}` 访问列表；列表只含本次确实可读的页。resident pass 和每个 streamed pass 各有自己的列表，彼此不重复；`tiered-exact` 在一次注意力计算内覆盖 `[0,F)` 的每个有效 key 恰好一次。缺页、重复页、超出 frontier 或未完成 H2D 都是执行错误，不能静默跳过。阶段 4 的稀疏模式才允许按选块计划省略 HostOnly 页。

新内核由 `logical_page * 64 + offset` 得到 key 的绝对缓存位置，而不由列表下标推导位置。例如 `{logical_page=1000, physical_page=7}` 的第 3 个 key 始终位于 `64003`，与它在访问列表中的次序无关。列表中的 `physical_page` 是新变体的统一寻址 ID：小于 resident pool 物理页数时指向当前层原有 K/V/scale plane；其余 ID 指向设备 staging 的页，按 `physical_page - resident_pool_pages` 求槽内偏移。dense 块表永不发布 staging ID。每次发布列表、块表和 frontier 都在同一个 GPU execution boundary 完成，运行中的 kernel 不见半更新状态。

当前 `PagedKVAllocation::page_ids()` 以向量下标表示连续逻辑页，不能直接代表有空洞的 tiered 视图。tiered 模式应新增独立的视图/归档 owner，并从 Main pool 取得与归还驻留物理页；不得把紧凑 `page_ids()` 冒充原始页序号。MTP 仍用独立 pool，Main 的换页不改变其逻辑 frontier 或物理容量；GDN 保持完整序列状态。

## 2. 按实际 key 数做部分注意力与 split-K

对一组 query 行，先以 `F` 与最大 query 位置裁掉不可见 key。列表中第 `i` 页贡献 `min(64, max(0, min(F, q_max+1) - 64*logical_page[i]))` 个候选 key；为列表建立累积有效 key 数的前缀和。split-K 在此前缀和的**紧凑 key 序号**上划分近似等长的 `[begin,end)`，必要时在页内切分；kernel 用前缀和定位列表项，而不是按 `last_pos+1` 或最高逻辑页号划分。空 split 输出 `m=-∞、l=0、O=0`。prefill 每个 query tile 和 small-T decode 均使用此规则；每行仍以自己的 `q_abs` 做最终因果掩码。

对列表项 `(logical_page, physical_page)` 和页内偏移 `j`，只有 `logical_page*64+j < F` 且 `logical_page*64+j <= q_abs` 时才读 K/V 并参与 softmax。末页的无效尾部、未来 key 和 staging 槽的填充位全被掩掉。原有 dense prefill 和 decode kernel、其 `block_table[position>>6]` 路径及 split 策略保持原样；新变体分别处理 BF16、INT8 与 `rk4v4-e8` 的 plane/scale 布局。

每个 resident/streamed pass 输出未归一化的 FP32 三元组 `(O,m,l)`，其中 `O=Σexp(score-m)V`、`l=Σexp(score-m)`。跨 pass 使用在线 LSE 合并：`m'=max(m,m_t)`，`l'=l·exp(m-m')+l_t·exp(m_t-m')`，`O'=O·exp(m-m')+O_t·exp(m_t-m')`，最后才算 `O/l` 并转成输出 BF16。可复用现有 `gqa_attention_decode.cuh` 的合并数学，但其 partial accumulator 当前是 BF16，不能把该存储格式直接作为阶段 3 的 FP32 精确中间结果。新 reducer 显式接收本 pass 的有效 split 数；不得再由 `last_pos+1` 推断。split 划分与 LSE 合并顺序必须仅由按原始逻辑页号排序的访问列表及 frontier 决定；传输完成先后只决定数据何时可读，不得改变 split 边界、pass 编号或合并顺序。

## 3. 分层传输与设备 staging

归档通过 `--kvmem-host-archive auto|pinned|pageable` 在加载期选定。`auto` 由 `max_context`、KV 类型求完整归档大小；整块锁页成功且锁页后可用物理内存仍不少于 4 GiB 时走锁页直传，否则释放已分配部分，走 `VirtualAlloc` 预留/按需提交的可分页归档和加载期 **4×64 MiB 锁页中转环**。显式 `pinned` 不能满足条件就报错，`pageable` 直接走中转环；日志记录选择和原因。两种模式的布局都为「层 → plane → 逻辑页」，K/V/scale 各 plane 内的连续逻辑页保持设备字节布局，可以一次传输；不重新量化。锁页模式 H2D 与 D2H 回写直接使用归档，可分页模式经中转环；运行期不新申请锁页内存。

设备 staging 在加载期按「一个全注意力层在单个 prefill 分块中最大的流式数据量 + 一个 64 MiB tile」确定容量；int8、128K 视图时约为 264 MiB + 64 MiB。容量由 `build_workspace_plan()` 与驻留视图一起求解，先从视图预算中让出 staging 所需显存；可提供加载期容量覆盖参数，但必须检查能容纳计算所得的最小需求，运行期不扩容。staging 是跨层复用的临时多 plane 区，不为 16 层各复制一套。在前面几个 GDN 层计算时，`kv_stage_stream` 预取下一全注意力层的页；层切换不做整层的主机等待，仅在具体页被消费前由执行流等待对应 H2D 事件。若 GDN 计算不能完全遮住传输，执行流只等待尚缺的那部分。

可分页模式的四个主机槽轮转：CPU 填下一槽可与上一槽 H2D 重叠，槽的复用受 H2D 完成事件保护；锁页模式跳过主机槽。设备 staging 中的每段以 H2D 完成事件标记可读，以注意力消费完成事件保护复用。跨层预取可提前填充未被当前层使用的段，但不能覆盖仍在消费的段；访问列表只在依赖的 H2D 完成后交给新注意力变体。逻辑顺序按访问列表固定，不能由哪个 H2D 先完成决定。GPU Graph 所需指针和 staging 容量在加载期固定；先用 tiered 专用 eager 路径验证，dense Graph 路径不变。满窗精确 decode 在阶段 3 也遍历全部页以验证数值，阶段 4 再用稀疏视图减少传输。

阶段 0′ 的独立 64 MiB 微基准测得 1/2 线程 `memcpy` 约 5.98/6.00 GiB/s，合成的可分页环流水线端到端约 4.19/3.86 GiB/s，锁页直传约 9.83 GiB/s；262K 配置就绪后整块可锁页 15488 MiB（约 15.1 GiB）。这些数字尚未计入真实多 plane 布局、逐层预取和 LSE kernel，不能直接当作模型吞吐预测。阶段 3 第 2 步要在两种模式下测真实「归档 → 设备 staging」吞吐。D15 限制主机工作线程最多 2 个；可分页路径首版以 1 个拷贝线程为默认。

第 2 步已实现 `HostKVArchive` 与 `HostKVTransferEngine`。审阅后修正为加载期固定两个拷贝工作线程及两条非阻塞 H2D/D2H 流，仍只分配可分页模式的四槽环。预取 worker 只提交 H2D；回写 worker 等生产事件、提交 D2H，pageable 再 memcpy 进归档，pinned 直接回写后通过完成事件兑现 future；等待回写不阻塞预取 worker。回写捕获此前最后一笔 H2D 的 ready 事件，依 H2D FIFO 顺序保护所有较早归档读取（包括 ticket 已复用的情况），但之后的预取没有回写依赖。两条流共用四槽时，每槽有独占 CPU lease 与 DMA 完成事件；D2H 等生产者之前不占槽，异常途中需先 drain DMA 再释放 lease，运行期不另申请锁页内存或事件。

每个 ticket 描述层/plane、原始逻辑页范围和 staging 字节偏移，单次不超过 64 MiB。ready/consumed 事件、引用计数与 generation 保护 DMA、消费者及槽复用；D2H 捕获 ready 前先在 CPU 确认它已记录，避免 CUDA 把未记录事件当作完成。跨层任务可提前排队，不按完成时间重排。引擎持有归档的唯一传输所有权，trim 先 drain 两条流和两个 worker。staging 使用调用方已有 `DeviceSpan`，`plan_host_kv_staging()` 已给出容量和覆盖参数校验；第 4 步才接入 `build_workspace_plan()`、GDN 调度和视图驱逐，产品 CLI 属第 7 步。因此目前不是可启动的 tiered 推理模式。

真实 16 层 INT8 K/V/FP16-scale 布局的手动微基准：262K 归档 **8.25 GiB**、每层流入 128K 页数据 **264 MiB**、staging **328 MiB**。两次独立进程的锁页直传为 **8.76 / 7.61 GiB/s**，可分页四槽为 **2.76 / 3.22 GiB/s**；计时含排队、主机复制、DMA 和每层末尾同步，完整 264 MiB 字节校验通过。该基准没有 GDN、注意力或 LSE，不证明模型吞吐或计算与传输的重叠效果。与阶段 0′ 的合成整块测量口径不同，细节及外部数据见 progress。

第 3 步已实现独立只读算子：`gqa_attention_partial_prefill/decode()` 接收 resident/staging 两组 PageMajor plane、访问列表、由 `attention_access_prefix()` 生成的 key 前缀和及 frontier；输出 FP32 未归一化 `(O,m,l)`，`attention_partial_lse_merge()` 固定按输入 part 升序合并，`attention_partial_finalize()` 最后归一化为 BF16。rk4v4-e8 的部分 O 在 FP32 上逆 H64 后才合并。Q scale、概率和概率低位残差均有独立共享存储，不沿用旧 small-T 的清零竞争或 scale/P alias。空列表用空 Tensor 与单个零 prefix 表示；空 split 是中性状态。列表每页完整、仅 frontier 末页可短，因此 compact ordinal 直接查列表条目；绝对位置仍读原始 logical_page。具体 API 形状、上限、独立 FP64 oracle、重复确定性、sanitizer 和 264 MiB 微基准见 progress 第 3 步。

性能优化后的量化 prefill 使用每个 Q head 的 64-query / 64-key tile、query-tile-major 栅格及与 dense 同深度的异步 K/V 流水线；BF16 注意力主体保持原实现。整页访问只读一次 logical/physical 映射，全因果 tile 与末页/因果边界分别实例化。rk4v4-e8 的逆 H64 融入 FP32 epilogue；Q scale 仍独立暂存并有明确 CTA 同步。概率必须先显式减去本 tile 的最大 score，再乘 scale，避免巨大有限公共 logits 在融合乘加时发生抵消误差。

跨 pass 不保留各遍的全部 partial：`attention_partial_lse_accumulate()` 把当前 S 个 split 固定按升序折叠进唯一的 FP32 `(O,m,l)` state，已有 state 总是第一项；首遍 reset 不读取旧字节，空遍保持已生成 state 的位模式。末遍可同时输出 BF16，省去独立 finalize launch；普通 out-of-place merge/finalize API 仍保留。split 权重的 exp 可以独立并行计算，但最大值扫描、prior-first 和 split 升序求和保持固定；共享 maximum/denominator 的复用必须经过 CTA barrier，不能按传输到达顺序归并。调用方在执行流上按固定 pass 编号串行调用，传输完成顺序不能改变折叠顺序。

`plan_attention_partial_workspace(max_tokens, preferred_splits, byte_budget)` 是无分配的加载期规划接口：预算只包含当前 pass scratch 与一份 state，按 `258 × 24 × T × (S+1) × sizeof(float)` 计算，与 pass 总数无关；preferred split 超预算就降低，连 S=1 都放不下则明确报错。100 MiB 预算下 T1024 最多 S=3（96.75 MiB），S=1/2 为48.375/72.5625 MiB；T2048 只能 S=1（96.75 MiB）。第 4 步再由 `build_workspace_plan()` 把该结果与输出、列表、设备 staging 和驻留视图统一预算，本步不接入目标运行时。

最终计算微基准的单 pass 六项（INT8/rk4，T4/1024/2048）三次均达到对应 1.05/1.10 目标；量化大 T 选择 S=1。两 pass T4 也在 1.05 内，四 pass T4 仍约 1.134/1.201 倍，不能推广为任意划分都达标。BF16 与原版配对无可分辨退步。真实流式 H2D/模型吞吐与影子门禁仍留给第 4 步；Nsight Compute 目前受计数器权限限制，硬件 profile 未完成，具体数据/限制见 progress。

当前 launcher 由调用方传入固定 split 数，再按实际可见 key 前缀划分。第 4 步在加载期结合 query 子块、视图和 staging 预算选 split，不能由 DMA 完成先后选择，也不能直接沿用 benchmark 的大 workspace。尚未新增产品参数、视图状态机或整模型影子接线；README 门禁 A/B 留在接线后的验证。

## 4. frontier、trim、restore 与 checkpoint

- 新 token 写入前，当前逻辑页必须驻留。每个 prefill 分块结束以及 decode 页写满时，按 D5 异步回写到主机归档；回写事件完成后才可从 DeviceOnly 变为 Both，且只有 Both 页可驱逐。staging 页只读，不取得归档所有权。
- `trim(F')` 先在执行边界等待所有仍可能访问旧页的 H2D、注意力与回写事件；把旧 generation 的异步完成通知标记为不可再发布，然后丢弃逻辑页号 `>= ceil(F'/64)` 的页映射和归档提交。若 `F'` 落在页内，保留该页及 `[0,F'%64)` 的有效字节，尾部标记无效并在续写前覆盖；Main、MTP 各按自己的 frontier 截断，不能把部分页当成完整 checkpoint。
- retained resume 和 turn checkpoint 只引用同一份独占 KV bundle。`restore` 先按 checkpoint 的精确 frontier 截断主机归档和设备视图，恢复对应的 GDN/MTP/hidden/position 状态，再把下一步必需的 sink、近期页及当前写入页换入，发布新块表和列表后才继续。前缀中未驻留的页仍有原始逻辑号；没有完整 continuation state 的任意短前缀仍是 miss。主机归档需保证 checkpoint 覆盖页的回写已完成；同页无效尾部不作为可读 key。
- `snapshot` 记录精确 frontier、页状态/映射 generation 和完整 target continuation state；turn checkpoint 只保存一个，不复制整份 KV。阶段 3 的 kvmem/tiered 模式关闭现有磁盘状态缓存路径，避免其按连续 `page_ids()` 序号 restore/snapshot；dense 的磁盘路径不变。任何 trim/restore 必须使旧 staging 列表和旧传输事件失效。

## 5. 同一次运行的影子验证

门禁 A 使用仅测试可开启的隐藏参数或环境变量，默认关闭。配置在加载期解析；关闭时不分配镜像/影子 workspace、不复制 KV、不执行额外算子或设备同步，dense 与 tiered 正常路径不增加开销。

开启时设备保留完整 dense KV，主机归档为这些量化字节的镜像。每个全注意力层在 KV append 完成后固定同一份 Q、同一份 KV 字节及 frontier：先执行 dense 注意力，再执行 tiered-exact 的影子注意力。影子视图只把选定页标为 resident；其余页即使仍存在于 dense pool 也不得读取，必须从主机归档经 staging 送入，产生部分 `(O,m,l)` 并按固定顺序合并。对最终注意力输出逐层记录相对 L2 和最大绝对误差；32K/128K 的 tiered/dense 相对 L2 上限保持 1e-3。下游只消费 dense 输出，影子输出不得改变后续隐藏状态、KV 或 GDN 状态。

32K 输入强制约 8K 视图、128K 输入强制约 32K 视图，使多数历史实际走流式路径；这两档必须执行。262K rk4v4-e8 需完整 dense KV、staging、影子输出和部分结果同时驻留，显存允许时执行，否则记录测得的预算缺口及原因。本档按影子相对 L2 选择误差最大的至少三个全注意力层，包括此前超限层；各取至少一个 prefill 末段完整块，从同一份原始 BF16 Q 与量化 KV 字节用单元测试同一套 CPU FP64 oracle 计算。记录 dense/FP64、tiered/FP64、tiered/dense 的相对 L2 和最大绝对误差，只有每层 tiered/FP64 相对 L2 均不超过 dense/FP64 才通过；任何一层失败即停止，不调整判据。此修订是测到 0.0010094 后、固定输入 alpha 消融定位之后用户同意的，不能描述为预先设定或仍满足旧 1e-3。测试资源仍在加载期预算，不能以省略流式传输或共享 dense 输出冒充通过。门禁 A 删除独立 dense 两次逐位相同的前提；README 的门禁 B 保留三次运行的波动包络、共同前缀、needle 和 MTP 接受率要求。

## 6. 后续实现位置与验证点

第 4 步接线采用 [运行时实施清单](stage4-runtime-plan.zh-CN.md)：每层一张逻辑页升序混合列表，staging 装下该层全部流式页，prefill 一遍、small-T decode 至多两遍；本步不会为热路径逐块缩小 staging。普通 tiered 的设备视图自动预算保留默认 1 GiB 显存余量，`--kvmem-view-tokens` 是上限而非保证分配量；dense 与影子完整设备池仍遵守原策略。影子模式逐层验证同一 Q/KV，单独记录每层最大相对 L2/绝对误差；它的同步读回耗时与普通运行性能分开。旋转格式的 FP32 partial/carry 保持原始 V 坐标，最终 BF16 输出需与 dense 的旋转基 BF16 舍入、逆 H64、再次 BF16 舍入边界一致；这不改变 dense 或把 FP32 carry 改为 BF16。

下表是阶段 3 的修改清单；前三步实现归档、传输和独立注意力算子，第 4 步实现 Main 运行时接线，MTP 窗口化与 checkpoint/resume 留给第 5、6 步。上文第 1–3 步的状态描述保留当时的实现范围；第 4 步的当前结果以 progress 和运行时实施清单为准。新逻辑按 core / ops / target 的仓库边界放置，避免 README 第 4 节列出的三值合并敏感位置。

| 位置 | 函数或新职责 |
|---|---|
| `src/core/kvmem/`（新增） | `HostArchive` 管理按原始逻辑页号索引的锁页/可分页归档和回写状态；`TransferRing` 管理可分页模式的 4 个锁页槽、锁页直传、按预算确定的设备 staging、跨层预取及 CUDA 事件；`ViewTable` 生成逻辑块表、resident/streamed 访问列表与 key 前缀和。 |
| `src/core/paged_kv_cache.{h,cu}` | 为 tiered owner 提供受容量约束的物理页取得/归还与 per-plane page bytes/copy 接口；保留 `PagedKVAllocation::publish_mapping()` 的 dense 语义，不把其连续 `page_ids()` 改成稀疏下标。 |
| `include/ninfer/ops/gqa_attention_partial.h`、`src/ops/wrapper/gqa_attention_partial.cpp`、`src/ops/launcher/gqa_attention_partial.h` | 增加显式 tiered access-list、staging plane、有效 key 数和输出 `(O,m,l)` 的新 API/参数检查；`gqa_attention()` / `gqa_attention_cached()` 的 dense 分派不变。 |
| `src/ops/launcher/gqa_attention_partial.cu`；`src/ops/kernel/gqa_attention_partial_{bf16,i8,common}.cuh` 与 `attention_partial_lse_merge.cuh` | 新 launcher 按实际 key 数决定 split；新 kernel 以逻辑页求绝对位置、以物理 ID 选 resident/stage plane，并对末页及因果条件掩码；不修改现有 dense kernel body。 |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h`、`src/targets/qwen3_6/impl/state/decoder_state.cpp`、`src/runtime/engine/kv_capacity.cpp` | `build_workspace_plan()` / `build_sequence_candidate()` 和 `plan_decoder_state()` 把视图、动态算出的设备 staging、访问列表、FP32 部分输出及主机归档准入纳入启动预算；staging 显存从视图预算中让出，Main/MTP pool 仍分开。 |
| `src/targets/qwen3_6/impl/runtime/program.h`、`program_impl.h` | `SequenceKVBundle` / `SequenceState` 持有 tiered owner；`start_prefill_lane()`、`advance_prefill()`、`decode_ordinary_batch()`、`decode_mtp_batch()` 在执行边界发布视图；`materialize_sequence_kv()` / `trim_sequence_kv()`、turn-checkpoint capture 与 restore 路径同步归档、frontier 和 generation；tiered 模式拒绝磁盘 snapshot/restore。 |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h`、`text_prefill_impl.h` | `TextContext::attn_mix()`、`run_layers()` 与 prefill chunk 调度选择新 Main 注意力路径；MTP/GDN 维持独立状态。前面几个 GDN 层计算时预取下一全注意力层的页，按段消费事件复用 staging，具体页读取前等待 H2D 事件。 |
| `include/ninfer/types.h`、`apps/cli/options.cpp`、`src/serve/serve_options.cpp` | 接入 README 已定义的模式和准入参数；默认 `dense`，`tiered-exact` / `kvmem` 仅 C=1。 |
| `tests/ops/test_gqa_attention.cpp` 与新增 `test_host_kv_transfer`、`test_attention_partial_lse_merge`、`test_kvmem_view_table`、`test_kvmem_resume_checkpoint` | FP64 全量注意力 oracle 覆盖远距离逻辑页、空洞、末页/因果掩码和 split 边界；逐字节 transfer 往返、事件复用、trim/restore；32K→128K→262K 与阶段 0′ dense 金标准比对，并跑全量 CTest。 |

后续接线的约束：所有列表与 staging 容量在启动时确定；容量不足或确切所需页缺失时清晰报错；split 和 LSE 顺序不能受异步传输完成顺序影响；新模式的数值门禁按 README 第 5 节的 prefill 与 decode 两层判据执行。dense 路径的输出、显存、性能基线不变。
