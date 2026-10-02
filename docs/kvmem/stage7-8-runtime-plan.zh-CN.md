# 阶段 3 第 7、8 步执行计划

目标：按用户已审阅方案收尾 tiered-exact，保持 D1–D15 与 dense 内核/分派不变；每步全量 CTest、独立提交并推送 feat/kvmem。阶段 4 仅写设计。

- [x] 第 7 步：CLI/serve 共同语义、exact-only、阶段 4 前拒绝 kvmem、C=1 校验；补帮助和主机准入提示。增加加载期 pageable VirtualLock，禁止 pinned 冲突；普通 pageable 按需提交不变。输出各项预算和主机余量，补测试与用户文档，改 maintainer non-goal。
- [x] 第 7 步全量构建与 CTest，审阅，单独提交并 push。
- [x] 第 8 步：自动分块先试 2048 的完整预算，预算不可行退 1024；显式设置与 dense 默认不变。恢复视图保留目标前缀仍存活的页，固定旧物理槽，只换入缺失页；generation 正常递增，旧 tickets 拒绝。
- [x] 新恢复 CPU/GPU CTest、262K 改动前后时间和实际调度传输字节；262K INT8 摘要生成 1024 token 比较 MTP 32768/262144 的接受率与 decode tok/s。完整结果及长生成未验证的输出一致性见 progress。
- [x] 终版九组 dense 首轮8/9的失败保留；用户事后批准的三项组合验收均通过：36次完整logits逐位零差异、九例生产计划一致且默认1024、生产算法六次的并列与L2包络均通过。阶段3关闭，未来各阶段沿用组合判据；证据见progress和dense-combined-gate-summary.json。
- [ ] 第 8 步全量构建与 CTest，审阅，单独提交并 push。
- [ ] 编写 stage4-sparse-decode-design.zh-CN.md，只设计 Mean-K/Q、CPU 打分、选块/diff、图像跨度、gen reserve、Graph、参考移植/许可和质量门禁；不写阶段 4 代码，停下等审阅。

模型与 GGUF 沿用 README，本轮所有测量、构建/测试日志及临时 harness 在仓库外 `D:\deeplearning\NInfer\logs\kvmem-stage7-8`，失败记录也保留。GPU 测量串行；构建不与性能测量重叠。
