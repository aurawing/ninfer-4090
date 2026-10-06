# 4.5 部分合成结果：提前终止（2026-10-06）

**合成、主汇总 8/240、提前终止、不构成质量验收。** 用户要求一小时内实验版收口，并限制时间与噪音；正式矩阵已停止，之后不再运行长 GPU 任务。原始答案、输入、SHA、事件和失败结果不改写。

停止时实际已完成 12 次。按用户指定，以前 8 次作主汇总，额外 4 次单独附表；不丢弃已落盘数据。原计划裁剪后的 240 次中仍有 228 次未运行；更低优先级的预算削减另见外部冻结 plan。成本试跑不计入正式质量样本。

全部已完成项均为 `262144 / kvmem / int8 / 请求视图32768 / all-committed`，seed=0，各题型历史文档加短问题与单条长消息各一次。基础设施状态均为 ok，但这不表示答案通过。使用冻结 `kvmem-local-quality-v1` validator，未更改评分、truth 或输入。

## 指定的前 8 次主汇总

|用例|严格答案校验|decode tok/s|MTP 接受率|
|---|---|---:|---:|
|needle-c262144-n00-historical|通过|100.39|75.00%|
|needle-c262144-n00-single|通过|100.11|83.33%|
|repeat-c262144-n00-historical|失败|95.32|59.52%|
|repeat-c262144-n00-single|失败|93.95|62.50%|
|multi-c262144-n00-historical|通过|111.27|75.36%|
|multi-c262144-n00-single|失败|114.63|80.65%|
|chain-c262144-n00-historical|通过|122.24|86.67%|
|chain-c262144-n00-single|失败|102.40|76.67%|

仅 4/8 严格校验通过（50%），该比例仅描述这一局部子集，不是每档质量通过率或验收。多值两项失败；多查询单条消息失败；变量追踪单条消息失败。格式不符合严格 JSON 的输出同样保留为失败，不事后提取答案改判。

## 停止前额外完成的 4 次

|用例|严格校验|事实格式覆盖|decode tok/s|MTP 接受率|
|---|---|---|---:|---:|
|frequency-c262144-n00-historical|失败|不适用|126.23|90.48%|
|frequency-c262144-n00-single|失败|不适用|117.67|88.89%|
|summary-c262144-n00-historical|失败|0/10|95.21|60.77%|
|summary-c262144-n00-single|失败|0/10|98.26|63.72%|

额外 4 次严格校验均失败。摘要的 0/10 指冻结的完整、锚定、逐行事实格式校验，不把自然语言中可能存在的片段当成通过。原始输出可独立复核，但本轮不改 validator。

## 未完成与可得结论

- 当前档其余 seed 与全部严格配对未完成；kept-band 主矩阵与 tiered-exact INT8 参考未运行。
- 262K/128K 视图、128K 上下文的主矩阵、dense rk4v4-e8 参考、rk4 信息性归档档均未完成。
- 长生成摘要覆盖、多题型相对参考差距、两种分母差异与默认值建议无法给出；默认分母保持原值。
- 4.6 CUDA Graph descriptor、ordinary/MTP capture、eager/graph 差距与性能门禁推迟。
- 真实多文件与工具回放仍未测，合成结果不能替代真实质量验收。

`capture_wait_ms` 的含义是等 GPU 队列排空及捕获发布的墙钟时间，已在 prefill 中，不应作为独立捕获算子再加进 TTFT。

## 外部证据

- 目录：`D:/deeplearning/NInfer/logs/kvmem-stage4-5-6/release-experimental-20261006/`。
- `partial-synthetic-results.json`：全部 12 次冻结校验和原始 metrics，主/补充范围明确分开。
- `preserved-evidence-sha256.json`：原始 raw、日志、request、events、plan、manifest 身份。
- `matrix-user-stop-before.json` / `matrix-user-stop-after.json`：受控进程停止及 GPU 子进程排空证据。
- 原始数据：`../quality-final-v5/run/batch-0000/raw.jsonl`；SHA256 `a0220cf32ac41fe717fac75ce21852460bc245879c9f63ecd3d298de1ae6f979`。
