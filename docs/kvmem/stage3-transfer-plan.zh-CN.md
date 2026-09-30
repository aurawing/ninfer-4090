# 阶段 3 第 1、2 步实施清单

目标：实现可独立验证的主机归档、回写和传输基础组件，在第 2 步完成后停下审阅。依据为已审核的 README D1–D15 和 stage3-view-design；不实现注意力变体，不接入产品 CLI（阶段 3 第 7 步），运行时调用点在第 4 步接线。采用现有 CUDA、PagedKVPool 和锁页缓冲组件，不新增项目依赖。

## 第 1 步：HostKVArchive

- [x] 在 `tests/core/test_host_kv_transfer.cpp` 写真实 PagedKVPool 的多层、多 plane 往返、碎片物理页、覆盖回写、部分页 trim 与续写测试；先验证失败。
- [x] 新增 `src/core/kvmem/host_kv_archive.{h,cpp}`：统一 layer → plane → logical page 布局，checked size 运算；auto/pinned/pageable 加载期选择，4 GiB 准入；可分页地址预留、按需提交/截断，锁页整块分配且不在运行期新增锁页。
- [x] 利用现有 pool copy 接口镜像整视图；回写由 CUDA 事件保护，完成前不得发布主机有效 frontier；trim 等待旧传输、增加 generation、保留部分尾页并屏蔽尾部。设备原始逻辑页号不改变。
- [x] 归档保存单页规范字节；PageMajor 连续页合并拷贝，HeadMajor 逐页保持规范，不改变现有 pool copy 的 dense 语义。
- [x] 注册 `test_host_kv_transfer`，执行全量构建和 CTest，更新 progress，独立提交并 push。

## 第 2 步：HostKVTransferEngine

- [ ] 先写 `tests/core/test_host_kv_transfer_engine.cpp`，覆盖四槽复用、锁页直传、跨层预取、生产/消费事件、generation 失效以及多 plane 字节校验。
- [ ] 新增 `src/core/kvmem/host_kv_transfer.{h,cpp}`：加载期分配 pageable 模式 4×64 MiB 缓冲，固定非阻塞 stage stream、固定事件；pinned 跳过中转槽。staging 使用调用方 workspace 的 DeviceSpan，容量按最大流式层 + 64 MiB 求解，后续由 build_workspace_plan 纳入视图预算。
- [ ] 预取任务显式传原始逻辑页范围与层/plane、设备偏移；消费者在自己的 CUDA stream 等 ready 事件，release 事件保护复用。下一层可以提前入队；完成顺序不改变逻辑访问顺序。trim 先 drain 引擎，再截断归档并使旧 ticket 失效。
- [ ] 回写路径 pageable 经固定环、pinned 直接归档；主机复制最多一个工作线程，满足 D15 两线程上限。不得运行期申请锁页内存。
- [ ] 加可手动运行的真实 INT8 K/V/scale plane 微基准；数据放仓库外，分别测直接锁页和 pageable 环的完整层传输、最后同步及逐字节校验。
- [ ] 全量 CTest、progress、独立提交并 push；到此停下，不做第 3 步内核。

