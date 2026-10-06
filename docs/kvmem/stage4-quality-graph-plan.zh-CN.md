# 阶段 4.5 / 4.6：合成质量与 CUDA Graph 实施计划

> 2026-10-06 用户收口决定覆盖本计划的未完成部分：4.5 部分完成并提前终止；4.6 推迟、不实施；发布实验性 eager 版本后停止开发。下文保留为计划记录，不表示已实现或获准继续运行。结果见 [部分合成报告](stage4.5-partial-synthetic-results.zh-CN.md)。

2026-10-06 用户已审阅通过 4.4，授权在 `feat/kvmem` 依次完成本文件两步、分别提交并 push；两步完成后停审。默认 dense 内核、分派和 D1–D15 保持原约束。真实多文件与工具回放质量仍未验收。

## 4.5 合成质量

本地固定种子生成多键、多值、多查询、变量追踪、标记频次聚合和 1024 token 摘要。每题型每档目标 10 个 case：5 份文档，各有单条长 user 与历史文档加短问题两种输入。成对配置使用完全相同的冻结输入、truth 和 SHA；校验器同时覆盖错误答案的负例。摘要记录十个植入事实的覆盖率、MTP 接受率和 decode tok/s。

按下表从高到低安排；先用小批量实测校准约 24 GPU 小时预算，超出时从低优先级削减，保留未执行 case 清单与削减原因，不依据答题结果选样。

| 优先级 | 上下文 / 请求视图 | 对照 |
|---|---|---|
| 1 | 262K / 32K，INT8，两个分母 | tiered-exact INT8 |
| 2 | 262K、128K / 128K，INT8，两个分母 | dense rk4v4-e8 |
| 3 | 128K / 32K，INT8，两个分母 | tiered-exact INT8 |
| 4，信息性 | 262K / 32K、128K，rk4v4-e8，两个分母 | 同 token 数 INT8；记录实际视图 |

信息档按用户补充决定请求相同 token 数，不改为同显存预算的双倍视图。仅使用公开 Engine API 的已有 checkpoint / retained resume；不能复用时单独 prefill，不为此改变运行时，也不以配置相关的生成答案改变冻结输入。每组报告通过率、相对参考差距、分母差异及默认值建议；默认分母是否调整由用户决定。结果明确标为“合成”。

`capture_wait_ms` 保留兼容字段，报告解释为等待 GPU 队列排空及捕获发布的墙钟时间，是 prefill 的子集，不能当作独立 Q 捕获算子耗时重复计入 TTFT。

## 4.6 CUDA Graph 与性能

按 [设计 §7](stage4-sparse-decode-design.zh-CN.md#7-单列表-sparse-eager-decode-与后续-cuda-graph) 实现固定地址的 device descriptor，覆盖列表、frontier、可见 key、有效 query、KV append mask/ordinal/物理页、RoPE/M-RoPE position 和 Mean-K provisional 有效列。固定 split 或有限 profile，padding 不写 KV、索引或 GDN；分别捕获 ordinary 与 MTP verify。评分、选块、hydrate、publish、回写完成管理留在 graph 外。

Graph/eager 测试覆盖列表增减、跨页、reserve 淘汰、选块变化、restore 和有效列；内核小 shape racecheck，owner 完整 synccheck/initcheck。最终 dense 三项组合门禁必须通过。

生成 512 token、MTP-3、贪心，每档三次取中位数：262K 上下文请求 128K 视图时，kvmem graph decode 不低于同一时段 dense rk4v4-e8 满窗 decode。128K 和 32K 视图档只记录，同时报告 eager/graph 差距。

## 验证与证据

迭代只跑定向 CTest；每步终版候选只跑一次全量 CTest、完整适用 sanitizer 和 dense 组合门禁，终版改变则重跑。每步进度约 20 行，只保留结论、数字、提交号和仓库外证据索引；中间候选及失败尝试放在 `D:/deeplearning/NInfer/logs/kvmem-stage4-5-6`。4.5 完成后继续 4.6，4.6 完成后停审。
