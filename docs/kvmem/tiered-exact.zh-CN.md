# 使用 tiered-exact

`tiered-exact` 把 Qwen3.8/3.6-27B 的完整 Main KV 历史归档到主机，在 GPU 上保留 sink 和最近页。注意力仍访问全部历史，非驻留页由归档流式送入，因此可以在 RTX 4090 上验证 262K INT8 上下文。它是精确质量基准，**decode 只用于验证**，满窗 decode 的 PCIe 开销较大；日常稀疏 decode 的 `--kv-mode kvmem` 在阶段 4 完成前明确拒绝。

CLI 与 serve 共用下列参数语义：

| 参数 | 默认与说明 |
|---|---|
| `--kv-mode tiered-exact` | 默认仍为 dense；需要显式启用 |
| `--max-context 262144` | 每个请求的逻辑容量，同时决定归档上限 |
| `--kv-dtype int8` | 归档与 GPU 视图相同精度；支持 bf16、int8、rk4v4-e8。BF16 仅保证功能，不承诺性能 |
| `--kvmem-view-tokens 131072` | 物理视图上限，实际容量由显存预算决定；别名 `--kvmem-view` |
| `--kvmem-sink-tokens 256` | 常驻前缀，别名 `--kvmem-sink` |
| `--kvmem-prefill exact` | 只接受 exact；window 与 query replay 尚未实现 |
| `--kvmem-mtp-window 32768` | MTP 总物理容量，包含 sink、近期页和草稿保护页；必须为正的 64 倍数。MTP 不归档、不补算被覆盖历史 |
| `--kvmem-host-archive auto` | 加载期尝试整块 CUDA 锁页且保留至少 4 GiB 可用物理内存；失败则释放并退回 pageable 与 4×64 MiB 锁页环 |
| `--kvmem-host-archive pinned` | 整块 CUDA 锁页，H2D/D2H 直传；无法满足就拒绝加载 |
| `--kvmem-host-archive pageable` | 虚拟地址预留，按需提交，使用固定中转环 |
| `--kvmem-lock-archive` | 可选 OS residency lock：auto 选择 pageable；加载时整份提交并 VirtualLock，trim 保留锁定内存。提高工作集失败或锁定失败则报错。它不是 CUDA 注册，传输仍经过中转环；不能与 pinned 同用 |
| `--kvmem-staging-mib N` | 加载期固定 staging 覆盖；不足以放下整层流式页时提高最小视图，仍无可行预算就拒绝 |
| `--prefill-chunk N` | 显式选择分块，必须为正的 128 倍数。不指定时，普通 tiered 先检查 2048 的完整预算，不可行再选 1024，日志记录选择及回退原因；显式值、dense 和影子验证的默认值不变 |

服务示例（PowerShell）：

```powershell
.\ninfer-serve.exe .\models\qwen3_8_27b.ninfer --kv-mode tiered-exact --kv-dtype int8 --max-context 262144 --max-concurrency 1 --kvmem-prefill exact --kvmem-host-archive auto --prefill-chunk 2048 --spec mtp --draft-tokens 3 --lm-head-draft --host 127.0.0.1 --port 8111
```

CLI 使用同一组选项，另加 `--prompt` 或 `--messages`。CPU 视觉可沿用 `--vision --vision-device cpu` 与既有视觉权重配置。

限制与日志：

- 只支持 C=1；serve 的 `--max-concurrency` 大于 1 在启动前拒绝。GDN 始终处理完整序列。
- CUDA Graph 自动关闭，磁盘状态缓存自动关闭；内存中的 retained resume 和 turn checkpoint 可用。历史 thinking 被模板删除时，前缀可能不再相同；希望保留原前缀可用 `--preserve-thinking`。
- Main snapshot 不复制 KV；恢复先排空传输，再截断归档、更新 generation 和视图。按当前视图保留回滚有效前缀中仍驻留且未被覆盖的页及原物理槽，包括 frontier 部分页的有效前缀；仅缺失页从归档换入。`[kvmem-restore]` 记录保留/缺失页、调度 H2D 字节及恢复耗时，旧 ticket 仍失效。MTP 缺失页不补算，日志明确说明草稿质量和接受率可能下降。
- 主机准入要求可用物理内存至少为归档容量加 4 GiB。CUDA 锁页失败可用 pageable；物理内存不足必须缩小 max-context 或选择更小 KV，pageable 不绕过准入。
- 加载日志打印实际视图 token 数、Main/MTP 字节、staging、partial、访问列表、一般 workspace、归档选择和主机可用物理内存。view 上限不是显存占用承诺。
- `NINFER_KVMEM_TRANSFER_TIMING=1` 开启逐层 ready 等待，prefill/decode 分开累计；默认关闭。影子验证环境变量仅用于开发，不用于服务性能测量。

当前门禁与实测见 [progress.zh-CN.md](progress.zh-CN.md)。
