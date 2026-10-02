# 阶段 3 第 5、6 步实施清单

基线 `f5da0049`。第 5 步完成验证并提交后实施第 6 步；第 6 步完成后停在审阅点。

## 第 5 步：MTP 窗口

`--kvmem-mtp-window` 默认 32768，固定预算包含 sink、最近页及跨页草稿保护页。默认长上下文使用 512 个物理页：sink 4 页、recent 507 页、guard 1 页；短上下文不需要 guard。用户确认遵循 D11，保留 sink，剩余容量用于最近页。

MTP 不归档，不做 re-RoPE；页标签和访问列表保留原始逻辑页号。逐 query 写入并读取窗口，避免后面的列提前覆盖前面 query 需要的页。无效草稿列不得写入任何 KV code、scale 或页标签。跨页保护页保存所有 plane，拒绝新页时恢复旧页，保护页包含在固定预算内。

复用部分注意力内核；masked append 算子由 ops 层拥有，窗口生命周期由目标运行时拥有。dense 内核和分派不改。统一预算把节省的 MTP 存储交给 Main 视图；启动打印新旧 MTP 字节、辅助空间和视图容量。`old_main_view_tokens_estimate` 是同一已确定预算下的旧布局估算，不代表历史桌面负载下的实测。

验证顺序：CLI 与预算回归；BF16、INT8、rk4v4-e8 的 sink/远逻辑页/环绕/T=1、2、4/无效列/跨页回滚测试；全量 CTest；128K 与 262K INT8 的窗口 MTP 与关闭 MTP 贪心比较、needle 和接受率门禁。prefill 与 decode 的 ready 等待分别记录。正式长上下文结果以 progress 和仓库外原始数据为准。

第 5 步已完成：全量 105 项 CTest 为 101 通过、4 缺少制品跳过、0 失败；三项 sanitizer 复测为零错误；两档各 64 token 完全一致、needle 正确、接受率均 100%。详见 progress 的实测表。

## 第 6 步：两个复用点（尚未实施）

恢复 retained resume 和 turn checkpoint；磁盘状态缓存继续关闭。快照仅保存精确 frontier、generation、GDN、MTP、hidden 和 position continuation 状态，不复制 KV 字节。

在实际 checkpoint 分块结束的边界捕获状态。restore 先排空旧传输，截断归档和视图，递增 generation，使旧 ticket 和 staging 列表失效；从归档重新换入 sink、最近页及当前部分写入页，不重新安装可能已被覆盖的旧物理映射。

MTP 回滚仅保留仍存在的原始页标签交集；不补算丢失的窗口 KV，只记录草稿质量和接受率可能下降。generation 不回退。

CTest 覆盖 snapshot/restore 的 frontier 与 generation 一致性、部分页边界和旧 ticket 拒绝。128K、262K 两轮对话记录 reused prompt tokens，第二轮仅 prefill 新增后缀；checkpoint 回滚与冷启动按门禁 B 判定。每步全量 CTest 后单独提交并 push。

所有测量数据放在仓库外：`D:/deeplearning/NInfer/logs/kvmem-stage5-6`。
