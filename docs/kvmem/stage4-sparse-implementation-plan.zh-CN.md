# 阶段4：4.1–4.4 顺序实施清单

基线为 `feat/kvmem@a78b533a3080a7d2284a65aae234e5f748474e29`。本清单按2026-10-02用户批准的D8/D9修订编写；契约见[README D8/D9](README.zh-CN.md)、[阶段4设计](stage4-sparse-decode-design.zh-CN.md)及[progress](progress.zh-CN.md)。本轮先完成全部文档修订，再由主任务顺序实现4.1–4.4。此文件的勾选状态只反映实际完成的门禁，文档准备不等于实现通过。

每一步先新增对应CTest并验证其能检测该步关键错误，再完成实现、运行定向测试及当时全量CTest，更新progress与外部证据索引。全量清单以当前配置为准，不沿用AGENTS历史84项作为测试数量。只有检查通过、差异审阅完成后，主任务才为该步分别commit/push到`origin/feat/kvmem`，记录commit及push结果，再进入下一步；不得把四步合成一次提交或把未通过门禁的后续代码混入前一步。已获本轮逐步提交/推送授权，不重复询问常规许可。失败保留输入/日志/哈希，修正后重跑相关门禁，不能绕过。dense内核、dispatch和README三项组合门禁不变。

## 4.1 Mean-K 数学与部分前缀恢复

- [ ] 在Q/K RMSNorm后、RoPE前捕获归一化BF16 K，按原缓存ordinal累加固定顺序FP32 sum/count，页mean以FP16 RNE发布；完整页除64，部分页除实际有效count，禁止FP16 mean×count恢复sum。
- [ ] 实现固定最大stride的主机索引、加载期准入与D2H事件。archive pinned时额外cacheable pinned Mean-K计入准入并可直传；pageable经既有环。运行期不申请CUDA/锁页内存，不重解释F增长后的旧地址。
- [x] 有界tail BF16/provisional K、accepted-only commit、64 KiB精确prefix sum continuation及generation；普通verify回退与任意历史snapshot restore分开，禁止无snapshot trim到base之前，禁止用post-RoPE量化K补算。
- [x] 新增Mean-K CTest，用独立FP64 oracle覆盖1/63/64/65 token、跨chunk/轮次、FP16 RNE、部分封存、非64 frontier、provisional接受/拒绝/跨页、snapshot后尾页写满/驱逐及旧completion拒绝。
- [ ] 定向CTest与**全量CTest**通过；保存命令、输出、编译/二进制身份及实际测试数量/跳过原因，更新progress。主任务审阅后独立commit/push，记录结果再进入4.2。

本步只建立数学和恢复基础，不解除KVMem产品入口拒绝，不修改dense kernel/dispatch。

4.1基础门禁已通过：107项全量CTest为103通过、4制品缺失跳过、0失败，终版三项sanitizer为0 hazard/0 error，规格及代码质量审阅通过。前两条涉及真实模型hook、统一加载准入和DMA协调器的接线核验保留至4.3–4.4，不能仅由独立算子通过提前勾选。提交/推送记录见progress与仓库外证据索引。

## 4.2 确定性GPU评分与CPU纯选择器

- [ ] 两遍GPU scorer：每个(layer,qhead,query token)第一遍固定树求评分域全局max/denominator，第二遍固定layer/qhead/token顺序累加页概率，禁止atomic与按完成顺序归约。标量FP64只作独立测试oracle，不实现CPU/AVX2产品scorer或共享executor/暂停worker。
- [ ] 固定stride Mean-K pinned直接H2D，pageable复用既有环；pinned archive既有环为0 bytes，不能假设可借。必要的有界raw-index transfer ticket扩展现有engine，设备输入/统计/score加载期alias空闲staging/partial，排空原consumer后借用并在hydrate/下一prefill前交还。日志拆分H2D bytes/时间、两遍compute与score D2H。
- [ ] 默认只在`nb>budget>0`且`sink+recent<nb`时排除配置sink/recent bands；非页齐hard recent可多边界页，mask不能使用hard union。query/current/image等其他hard仍在denominator。隐藏`NINFER_KVMEM_SCORE_ALL_PAGES=1`覆盖全部已提交页；软化current页不能另行排除。
- [ ] CPU selector按D9构造sink/recent完整页/query span及图像闭包hard，另计reserve/guard。全current闭包union若fits `V-G-H`则全hard，否则较早非hard current页与历史同域竞争；记录`current_input_softened_pages`。hard溢出拒绝并报告各项页数；历史图像原子候选，本轮图像依D9，共享页传递闭包、整组不足跳过、稳定较新ID tie、升序输出。
- [ ] 移植固定参考的`pick_topk_blocks()`、`pick_semantic_groups()`、`set_selection()`思想：pure budget、原子组成本、overlap/retained/add/remove；保留[设计§9.1](stage4-sparse-decode-design.zh-CN.md)要求的具体repo/commit/path/function、修改说明、适用Apache-2.0文本/notices及原有注释，不引入global pool/re-RoPE/新依赖。
- [ ] 新增GPU scoring/CPU selector CTest：独立FP64、GQA/L/H/M归约、极大有限logit、两denominators及mask条件、NaN/缺失generation、重复稳定、current fit/soften、hard和图像闭包overflow、整组跳过、精确tie、预算/reserve/guard。GPU score完整向量相对L2≤`1e-4`；selected集合与FP64一致，唯一例外为固定 `epsilon=1e-6×max(1,abs(score_k))` 的FP64第k阈值±epsilon边界互换，记录阈值/页/两集合，阈值外差异失败。
- [ ] 定向CTest与**全量CTest**通过，更新progress及外部原始证据；主任务审阅后独立commit/push，再进入4.3。

## 4.3 Frontend query span、三槽capture与Main snapshot

- [ ] frontend显式提供current-input事件跨度、用户文本query跨度、图像token span/身份；lastN不取模板/生成头/历史assistant/system/tool/图像token，跨chunk按ordinal捕获16层Q。冷启动历史不变成本轮输入，current不等于实际prefill delta。
- [ ] 首版不做历史图像引用检测；保留图像ID/digest/grid/span供原子闭包，历史图像为候选，本轮图像按D9。协议引用识别留待后续，不猜自然语言引用。
- [ ] 最多active/checkpoint/retained三个不可变Q capture槽，拥有型句柄、完整16层有效性、prefix identity/lineage与covered frontier；相同句柄可共享槽，score/DMA只借用。替换/reset排空借用，不创建第四槽。
- [ ] 无新用户文本的tool续接仅沿用身份/源前缀/lineage匹配的有效capture；index epoch与capture epoch分别验证，旧capture可合法配新index。query被复用跳过时取已有有效capture，否则从合法continuation边界exact补捕获，必要时冷启动；不使用stale Q，不做window query replay。
- [ ] Main snapshot保存Mean-K精确prefix sum/count及index frontier/generation，并持有Q capture句柄；restore排空GPU/DMA，截断index、恢复sum、递增generation，拒绝旧score/plan/ticket，KV原量化字节和D12生命周期保持一致。
- [ ] 新增frontend/continuation CTest，覆盖跨chunk lastN、冷启动messages/单条长input、无文本tool续接、源prefix改写/恢复早于query、capture缺失补捕获、三槽高水位/借用、重复restore及旧状态拒绝。
- [ ] 定向CTest与**全量CTest**通过，更新progress及证据；主任务审阅后独立commit/push，再进入4.4。

## 4.4 Exact prefill到单列表sparse eager完整接线

- [ ] 完整接线 `exact prefill → capture ready → GPU score → CPU select → plan/diff → hydrate → publish → sparse eager decode/verify`，全部16层就绪再发布；intersection保留当前有效物理槽，新增页按差分hydrate，logical ID/原RoPE位置不变。only Both可驱逐；规划失败不改旧视图。
- [ ] 单logical升序列表resident attention，不为未选HostOnly页staging；有效prefix/frontier、原logical×64因果掩码、部分末页和MTP有效列正确。新生成页accepted-only索引提交/回写；reserve耗尽只淘汰安全非hard/recent/provisional Both页，不重新检索。
- [ ] 任意非连续resident选择之后，下一轮exact prefill仍读取完整归档历史，保持现有full-history流式路径；测试resident/staging混合、追加、checkpoint/retained回滚、缺失页只hydrate一次，不把sparse访问列表沿用为完整prefill列表。
- [ ] 注入DMA/plane中途失败：已覆写victim后Main标poisoned、阻止执行、排空地址借用，从权威归档重hydrate旧视图并验证发布，或reset bundle后exact重建。不能只“不发布新表”而继续使用旧mapping，不以device reset代替owner恢复。
- [ ] 产品入口仅在完整稀疏接线后启用`--kv-mode kvmem`，范围C=1/Graph=off，ordinary与MTP verify都eager；启动明确记录实际`graph=off`和阶段限制，沿现有tiered禁用方式处理默认`use_cuda_graph=true`，不要求用户额外传`--no-cuda-graph`，本轮不捕获Graph也不静默忽略配置。C>1/window明确拒绝，不能用tiered-exact冒充稀疏。
- [ ] 新增owner/eager/restore/error/admission CTest，包括arbitrary resident后的下一prefill、跨页/reserve淘汰、部分prefix恢复、旧generation/ticket、DMA poison恢复、pinned/pageable与MTP开关。定向CTest和**全量CTest**通过；终版对新增GPU owner/kernel跑Compute Sanitizer 13.0.85的 **racecheck、synccheck、initcheck**，记录命令、退出码和报告，任何错误必须处理；memcheck可额外补充，不能替代三项。
- [ ] 在终版保持README三项dense组合门禁：两版同一临时确定性补丁下完整词表逐位比较、加载计划一致、生产算法分歧步按既定门禁B并列/包络。三项都通过；dense kernel/dispatch不修改，金标准和原失败不覆盖。
- [ ] 合成验收冻结矩阵：`128K/262K contexts × 128K/32K views × 至少3位置 × 2 denominators × 3次独立运行`。既有单条长input needle依D9合法soften；另加结构化历史+短query fixture。每case核验实际token数/输入hash/模板/命令，128K view对dense rk4v4-e8，32K view对tiered-exact INT8，相同sampling/输出预算，不挑有利运行。
- [ ] 每case记录hard/current/image/softened/selected/retained/add/remove/reserve/guard页统计、denominator与mask范围、TTFT的prefill/capturewait/H2D/compute/D2H/selection/hydrate/publish分解、hydrate bytes、decode tok/s、MTP draft/accepted；保留失败和桌面负载。MTP/复用门禁按README已有规则。合成needle不能冒充真实多文件/工具通过率。
- [ ] 全部证据/progress更新，审阅差异及未验收清单；主任务为4.4独立commit/push，记录结果，**停止实施并交付审阅**。

## 4.4之后的停审清单

- [ ] 明确Graph descriptor、ordinary/MTP capture及对应性能门禁留到下一轮，当前**未实施/未验收**；本轮sparse eager结果不能替代。
- [ ] 明确真实脱敏多文件/工具矩阵**未验收**，语料和validator到位后再开展；格式/合成通过不能宣告完整阶段4质量验收。
- [ ] 报告实测与D8/D9/资源账本的偏差、残余矛盾和确需用户决定的问题；没有新问题时直接交付既定停审结果，不为已批准的常规步骤再次索要许可。

参考源码只使用本地 `D:/deeplearning/NInfer/logs/kvmem-stage7-8/kvmem-qw3-reference`，HEAD `1cf3b2f83bfc071ada9c57491a7d121723051ac0`；mask核对 `src/qwen_executor.cpp` 约23750，评分数学 `src/kernels_cuda.cu` 约5340。源码移植适用[设计§9.1](stage4-sparse-decode-design.zh-CN.md)的来源与Apache许可要求。
