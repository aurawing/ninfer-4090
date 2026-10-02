# 阶段 3 第 5、6 步实施清单

基线 `f5da0049`。第 5 步完成验证并提交后实施第 6 步；第 6 步完成后停在审阅点。

## 第 5 步：MTP 窗口

`--kvmem-mtp-window` 默认 32768，固定预算包含 sink、最近页及跨页草稿保护页。默认长上下文使用 512 个物理页：sink 4 页、recent 507 页、guard 1 页；短上下文不需要 guard。用户确认遵循 D11，保留 sink，剩余容量用于最近页。

MTP 不归档，不做 re-RoPE；页标签和访问列表保留原始逻辑页号。逐 query 写入并读取窗口，避免后面的列提前覆盖前面 query 需要的页。无效草稿列不得写入任何 KV code、scale 或页标签。跨页保护页保存所有 plane，拒绝新页时恢复旧页，保护页包含在固定预算内。

复用部分注意力内核；masked append 算子由 ops 层拥有，窗口生命周期由目标运行时拥有。dense 内核和分派不改。统一预算把节省的 MTP 存储交给 Main 视图；启动打印新旧 MTP 字节、辅助空间和视图容量。`old_main_view_tokens_estimate` 是同一已确定预算下的旧布局估算，不代表历史桌面负载下的实测。

验证顺序：CLI 与预算回归；BF16、INT8、rk4v4-e8 的 sink/远逻辑页/环绕/T=1、2、4/无效列/跨页回滚测试；全量 CTest；128K 与 262K INT8 的窗口 MTP 与关闭 MTP 贪心比较、needle 和接受率门禁。prefill 与 decode 的 ready 等待分别记录。正式长上下文结果以 progress 和仓库外原始数据为准。

第 5 步已完成：全量 105 项 CTest 为 101 通过、4 缺少制品跳过、0 失败；三项 sanitizer 复测为零错误；两档各 64 token 完全一致、needle 正确、接受率均 100%。详见 progress 的实测表。

## 第 6 步：两个复用点（已完成，停在审阅点）

恢复 retained resume 和 turn checkpoint；磁盘状态缓存继续关闭。快照仅保存精确 frontier、generation、GDN、MTP、hidden 和 position continuation 状态，不复制 KV 字节。

在实际 checkpoint 分块结束的边界捕获状态。restore 先排空旧传输，截断归档和视图，递增 generation，使旧 ticket 和 staging 列表失效；从归档重新换入 sink、最近页及当前部分写入页，不重新安装可能已被覆盖的旧物理映射。

MTP 回滚仅保留仍存在的原始页标签交集；不补算丢失的窗口 KV，只记录草稿质量和接受率可能下降。generation 不回退。

CTest 覆盖 snapshot/restore 的 frontier 与 generation 一致性、部分页边界和旧 ticket 拒绝。128K、262K 两轮对话记录 reused prompt tokens，第二轮仅 prefill 新增后缀；checkpoint 回滚与冷启动按门禁 B 判定。每步全量 CTest 后单独提交并 push。

所有测量数据放在仓库外：`D:/deeplearning/NInfer/logs/kvmem-stage5-6`。

第 6 步实施记录：

- [x] `KVViewTable::plan_restore/install_restore` 在 CPU 上规划新的 sink/recent/部分尾页驻留映射；只能安装完整归档的前缀，generation 单调增加。
- [x] `TieredContext::capture/restore` 只保存精确 frontier、view/archive generation 与 bundle 身份；不保存物理映射或 KV 字节。restore 排空后通过既有 transfer owner 的 completed prefetch、ready/consumed 事件和固定额外 tile 从归档换入，按逻辑连续范围合并传输，并支持不连续 lease ID。
- [x] turn checkpoint 在实际分块边界捕获 Main 元数据及独立 GDN/hidden 状态；prefill 完成后只发布已捕获状态的有效性。MTP checkpoint frontier 为 Main 的 `F-1`，桥接位随新后缀重新写入。
- [x] retained resume 保存对应 bundle 的 continuation 元数据及 previous MRoPE position；相等 frontier 的复用仍执行 restore/generation 边界，MTP 桥接使用保存的坐标。
- [x] MTP snapshot/restore 只保存页标签，处理当前 guard 后取捕获标签与现存标签的交集。丢失历史不回放、不补算；日志记录 surviving/missing pages 和可能降低的接受率。sink/recent/guard 始终在原有固定物理预算内。
- [x] CPU frontier/generation/未完成回写拒绝；Main `F=129` 捕获后驱逐/恢复、部分页续写、重复恢复、foreign bundle、trim below 后重新增长仍失效、reset、非连续 leases；transfer wait/staged/release 的旧 ticket 拒绝；MTP 活跃 guard、标签交集、sink 保留和 KV 字节不变。
- [x] 新增 `ninfer_test_kvmem_resume_checkpoint`：实际 Engine 的 pageable+MTP、pinned+MTP、pageable 无 MTP；追加只计算后缀、重复 checkpoint 回滚与冷启动贪心 ID 相同，以及短上下文 max-new=1 exact-hit。聊天 fixture 使用 `preserve_thinking=true` 保留历史 assistant 的空 thinking 块，使同会话追加的 token 前缀一致。
- [x] 最终全量构建/CTest，包含既有 dense 相关回归：106 项，102 通过、4 制品缺失跳过、0 失败。dense 内核和注意力分派不改，本步不重新录制九组旧基线；两组 GPU owner 的三项 sanitizer 均通过，详见 progress。
- [x] 128K/262K 正式两轮对话：第二轮各只计算 51 新 token、checkpoint 各只重算 4 token；三组 cold/rollback 的完整 logits 包络、64 token 和 needle 全通过，接受率各阶段均 100%，prefill/decode ready 等待分别记录在 progress。

定向验证 6 项 CTest 全部通过、0 失败；仓库外日志为 `D:/deeplearning/NInfer/logs/kvmem-stage5-6/step6-targeted-green-r2.log`。正式长上下文门禁、最终全量测试和数据路径见 progress；此记录不推进第 7 步。
