# kvmem-v0.1-experimental 使用说明

本版面向 RTX 4090、Windows 11、Qwen3.8-27B `.ninfer`，发布于 2026-10-06。`kvmem` 与 `tiered-exact` 为**实验性**；程序与启动脚本默认 `--kv-mode dense`。原程序及启动脚本未覆盖。

阶段 3 与 4.4 已审阅通过；4.5 **部分完成**，合成主汇总 **8/240**，提前终止、不构成质量验收。停止时额外完成的 4 次单列保留。4.6 **推迟**：本版 sparse decode 只有 eager，无 KVMem Graph；dense 既有 Graph 保持原行为。收口原因是用户的一小时时间和噪音约束。

## 启动

解压后用 PowerShell，指定本机模型；包不含模型、视觉 GGUF 或 API KEY：

```powershell
# 默认 dense，262K，rk4v4-e8，MTP-3，端口 8111
.\launch.ps1 -Model 'D:\deeplearning\NInfer\models\qwen3_8_27b.ninfer'

# 实验性 sparse eager：262K 逻辑上下文，32K GPU 视图，INT8 主机归档
.\launch.ps1 -Model 'D:\deeplearning\NInfer\models\qwen3_8_27b.ninfer' -Mode kvmem -ViewTokens 32768

# 请求更大视图；实际值以加载日志为准
.\launch.ps1 -Model 'D:\deeplearning\NInfer\models\qwen3_8_27b.ninfer' -Mode kvmem -ViewTokens 131072
```

也可使用 `start-dense.cmd` / `start-kvmem-262k.cmd`，第一个参数是模型路径。不传模型时默认包内 `models/qwen3_8_27b.ninfer`。API KEY 默认读取**所选模型同目录**的 `api-keys.txt` 第一条非空行；可用 `-ApiKeyFile` 指定。文件不存在或没有非空行时不启用鉴权；KEY 不写入包或日志。默认监听 `127.0.0.1:8111`，需要局域网访问时显式传 `-BindAddress 0.0.0.0`。

CPU 视觉可传 `-CpuVision`，默认使用模型内嵌视觉权重；外部 GGUF 可另传 `-VisionMmproj '...\mmproj-BF16.gguf'`。CPU 视觉默认上限 1024 token。文本冒烟不验证视觉质量。

## 参数与实测

推荐先使用 32K 视图降低设备占用；希望保留更多历史时请求 128K。两档均设置 `--max-context 262144 --kv-dtype int8 --kvmem-prefill exact --kvmem-host-archive auto --kvmem-mtp-window 32768 --spec mtp --draft-tokens 3 --lm-head-draft`。只允许 C=1。

| 262K 配置 | 4.4 实际视图 | eager decode tok/s |
|---|---:|---:|
| KVMem，32K 请求 | 32768 | 118.10–135.14 |
| KVMem，128K 请求 | 115840 | 107.59–122.90 |

数据为本机 RTX 4090 的 **64-token 合成 needle**，MTP-3、INT8、chunk=2048，两种分母各位置三次中位数的范围。不能推断为 512/1024-token 长生成性能；未通过 Graph 性能门禁。完整配置与对照见 [4.4 实测](stage4.4-synthetic-eager-results.zh-CN.md)。实际视图受显存、staging、workspace、MTP 和桌面占用影响。

262K INT8 Main 归档约 8.25 GiB，另有索引与运行时资源；主机准入要求保留 4 GiB 可用物理内存。`auto` 选择 pinned 或 pageable，pageable 不能绕过物理内存准入。内存不足时缩小上下文或使用较小 KV。

## 已知限制与未完成项

- 合成质量门禁未完成。前 8 次冻结严格校验仅 4/8 通过；其余失败保留，不能据此比较相对参考差距。额外 4 次也保留；见 [部分结果](stage4.5-partial-synthetic-results.zh-CN.md)。
- 两种分母的完整配对、参考运行、128K 上下文和 128K 视图主矩阵、rk4 信息档、多 seed 与长摘要覆盖均未完成；默认分母未改。
- KVMem 无 CUDA Graph，性能门禁未验收；`tiered-exact` decode 仅用于验证。BF16 tiered 只保证功能、不承诺性能。
- 真实多文件、真实工具回放语料未测；合成数据不能替代它们。
- 磁盘缓存关闭；内存 retained resume / turn checkpoint 可用，MTP 回滚缺失历史不补算，可能影响接受率。
- 包复用已通过 117 项 CTest（113 通过、4 缺少其他模型跳过、0 失败）及 dense 三项组合门禁的 Release 源码/产物。本轮只作两次短生成冒烟，不重跑长门禁。

包内 `BUILD-MANIFEST.json` 记录源码、提交、tag 与各文件身份；包旁 `.sha256` 记录 ZIP 的 SHA256。包附 Apache-2.0 与第三方许可。需要支持 CUDA 13.x 的 NVIDIA 驱动，系统图形/驱动 DLL 不随包分发。
