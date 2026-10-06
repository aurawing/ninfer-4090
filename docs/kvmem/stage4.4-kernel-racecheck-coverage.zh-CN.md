# 阶段 4.4：owner 内核级 racecheck 覆盖对照（2026-10-06）

按用户批准的 sanitizer 策略：owner/e2e 只运行完整 synccheck/initcheck；racecheck 在内核级小 shape 单元测试上运行。完整 owner racecheck 在约26.29小时、前13个fixture完成后停止于 `fragmented_next_exact`，部分日志未报告hazard但无最终summary，**不作为完整通过**。停止原始记录、README偏差及来源SHA均保留。

本轮补齐了 **22 个实际 attention format/T/split 元组**（完整、无sanitizer的外部标记插桩owner副本采集891次成功调用），**66 个碎片化append配置**、**7 个实际评分域**，另有 **30 个 Mean-K offset/有效行/零行恢复配置**。原始owner源码不改，使用仓库外临时副本采集；所有16层共用相同配置，layer变化只改变值/plane地址。小单测保留D=256、QHeads=24、KVHeads=4和实际T/split，仅缩小物理页/key数量，保留逻辑页空洞与13-token尾页；**不是对完整历史作racecheck**。

最终完整、无过滤的小单测racecheck结果：

|单元测试|覆盖|耗时|最终结果|
|---|---|---:|---|
|`test_kvmem_kernel_racecheck`|120个可见case：44个attention、66个append、3个额外gather/scatter、7个精确评分域|46.766 s|0 hazard / 0 error / 0 warning|
|`test_kvmem_scoring`|既有FP64评分与重复一致性，加M=2的容量补充|27.422 s|0 hazard / 0 error / 0 warning|
|`test_meank_accumulate`|既有FP64/字节oracle，加30个owner精确与边界配置|7.640 s|0 hazard / 0 error / 0 warning|

同一原始终版owner（SHA `b36adb3a905213fdc6bf8c73f901abc708cdefcfa1bf50331b98e986cc73fe0e`）的完整synccheck/initcheck分别87.300/198.963 s，各0 error。生产源码/应用/owner身份保持不变；只有4个测试文件增加覆盖。独立审阅及最终全量CTest结果见 [progress](progress.zh-CN.md)，本表的单测结果不提前声明整体4.4发布。

下列完整对照保留英文内核模板名称与技术说明，所有case编号为零起始。CUDA checked memcpy/memset、Q槽copy与主机归档DMA不是自定义kernel，明确列为N/A；owner中的mtp fixture标志仅控制Main provisional事务，不会构造真实MTPWindow/GDN，不能将owner覆盖冒充完整模型路径。

独立源证据目录：`D:/deeplearning/NInfer/logs/kvmem-stage4-sparse`；报告 `4.4-kernel-racecheck-coverage-20261006-impl-v2.md` SHA `ab8cd01b6140ca08542f07b31aac307f9160ede58fc3229d64367a0f2c26a5fc`；完整JSON SHA `b36fe6459034ac32beb4032feec49fdd889d4f4fe332ac62ff214e30b0cfc50d`。完整JSON逐条包含891条owner记录的双向case映射、每份source/executable/log SHA及真实命令；历史证据和最初测试夹具RED都保留。

独立审阅发现新增 Mean-K 测试先在默认流填充尾缓冲、再在非阻塞流调用 retain-tail，缺少显式依赖。该项是源码核对发现的测试顺序缺陷；未宣称 racecheck 发现了 GPU hazard。现改为同一流上的 checked `cudaMemsetAsync`，完整功能测试和 racecheck 重跑通过；旧候选日志保留，新证据使用 `e` 记录，旧 Mean-K 结果不充当终版门禁。

## 来源、身份与完整证据

Machine-readable exhaustive records and bidirectional case mappings: `D:\deeplearning\NInfer\logs\kvmem-stage4-sparse\4.4-kernel-racecheck-coverage-20261006-impl-v2.json`; SHA256 `b36fe6459034ac32beb4032feec49fdd889d4f4fe332ac62ff214e30b0cfc50d`.
Actual owner inventory: `D:\deeplearning\NInfer\logs\kvmem-stage4-sparse\4.4-kernel-owner-actual-configurations-20261006.json`; SHA256 `62ec8d49dc343287fec7d6de402f13cbe62198aa5fd6d10163d2c0b3c9343a5d`.
Full functional inventory ran the complete owner main with no arguments and no sanitizer, using a separate executable built from an external source clone. It does not replace sanitizer evidence. The original owner test source was never edited; byte restoration was unnecessary, and its mtime was not touched.
Original owner test SHA256 `a353381b85410c56e78061d54afeb78b2f365494878b44e806dc226a6a1ebabe`. External inventory executable SHA256 `f67ebbe570db70db19ec6e28981a664d7cc329f444c6c1faa55651f3eb388d04`.
The external marker runs after layer-0 attention returns; the full functional run subsequently synchronizes and checks all 16 layers. The shared begin_block sets the same valid T/frontier/access list for every layer; partial() chooses splits from that shared prefix, cache format and scratch capacity. Cache geometry, dtype/packed flags and scratch extents are identical across all 16 layers. Layer changes alter K/V values and plane pointers, not launch dimensions. Therefore successful layer-0 markers plus complete 16-layer oracle completion represent each layer’s T/split configuration. Splits are calculated from the actual public plan/prefix using the unchanged production formula, not inferred from a nominal chunk. This is a compile-time test marker, not CUDA profiler instrumentation.

582 production source paths match both the pre-build guard and final immutable v9 source archive. This includes all src/ops and public ops headers, hence the protected old 313 ops paths. No production, kernel, dispatch, Dense default, application or owner-test source changed.

| Protected binary | SHA256 |
|---|---|
| `build-vision-integration/apps/ninfer.exe` | `575eb41c0a0034ca73d3b89a395a2e876eccf5921d0aaae7b65f6524d1234a14` |
| `build-vision-integration/apps/ninfer-serve.exe` | `52724ce0034ecb77813380c4a8334a60c6a6047dd0d75545f104592b097a3754` |
| `build-vision-integration/tests/ninfer_test_kvmem_runtime_owner.exe` | `b36adb3a905213fdc6bf8c73f901abc708cdefcfa1bf50331b98e986cc73fe0e` |

## Exact owner attention configurations

891 successful f.run records; 22 unique format/T/split tuples. Each tuple is executed by two new cases: resident-only (4 physical pages) and resident+staging (2+2). Both use a fragmented list `{logical 0 → physical 1, logical 1000 → physical 3}`, frontier 64013, 64 full sink keys and 13 valid final-page keys. D=256, H=24, K=4 and actual T/split are preserved. Only physical page count/key-domain size is reduced. Thus this is structural small-shape coverage, not a racecheck of the owner’s full history. Each split is executed exactly as listed, including neutral splits. FP64 stored-code softmax/inverse-H64 oracle and immutable code/scale byte guards are checked.

| Format | Exact `(T, split)` tuples | New case indices (0 based) |
|---|---|---|
| bf16 | (1, 2), (1, 12), (1, 18), (1, 32), (2, 2), (3, 12), (4, 12), (16, 1), (32, 1), (64, 1) | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 |
| int8 | (1, 12), (3, 13), (4, 13), (32, 1), (63, 1), (64, 1) | 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 |
| rk4 | (1, 12), (3, 13), (4, 13), (32, 1), (63, 1), (64, 1) | 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43 |

## Every owner CUDA kernel/template family

G = Gqa27Geometry (head dimension 256, Q heads 24, KV heads 4). All listed production sources retain immutable v9 SHA; the JSON contains every launcher/kernel SHA. Reset=true is the actual owner merge branch; reset=false second-pass carry is additional unit coverage.

| Owner kernel / instantiated configuration | Actual launch | New complete racecheck coverage | Prior evidence and gap |
|---|---|---|
| `gqa_attention_partial_bf16_kernel<G,1,4>` | grid(4,split,1), block128, smem0; T1 splits2/12/18/32 | exact tuples in cases0–7; FP64, two pool modes | Stage3/4 unit evidence existed, but did not establish all these exact splits |
| `gqa_attention_partial_bf16_kernel<G,2,4>` | grid(4,2,1), block128; T2 | cases8–9 | old T2/split7 is not exact T2/split2 proof |
| `gqa_attention_partial_bf16_kernel<G,4,4>` | grid(4,12,1), block128; T3/T4 | cases10–13 | old T4/split7 did not cover T3/12 and T4/12 |
| `gqa_attention_partial_bf16_kernel<G,10,4>` | grid(4,1,ceil(T/10)), block128; T16/32/64 | cases14–19 | old prefill64/65 split2 was not exact split1/T16/T32 evidence |
| `gqa_attention_partial_i8_kernel<G,1,4,32,false,false,false>` | INT8 T1/split12; grid(4,12,1), block128, smem0 | cases20–21 | signed-code unit existed; exact split12 added |
| `gqa_attention_partial_i8_kernel<G,4,4,32,false,false,false>` | INT8 T3/T4 split13; grid(4,13,1), block128 | cases22–25 | exact T3/13 and T4/13 added |
| `gqa_attention_partial_i8_kernel<G,64,16,64,true,false,true>` | INT8 T32/63/64 split1; grid(1,24,1), block512, smem65536 | cases26–31 | exact partial Tile64 tail/filled cases added |
| `gqa_attention_partial_i8_kernel<G,1,4,32,false,true,false>` | RK4 T1/split12; block128 | cases32–33 | packed K/V/H64/E8 storage covered with original logical IDs |
| `gqa_attention_partial_i8_kernel<G,4,8,32,false,true,false>` | RK4 T3/T4 split13; block256 | cases34–37 | actual packed Tile4/8-warp configuration added |
| `gqa_attention_partial_i8_kernel<G,64,16,64,true,true,true>` | RK4 T32/63/64 split1; block512/smem65536 | cases38–43 | packed prefill partial final tile and full tile added |
| `attention_partial_lse_merge_kernel<true,true>` | BF16/INT8; grid24*T, block256, smem4*(split+3); actual reset=true | every BF16/INT8 tuple; also reset=false second pass | historical LSE unit covered general merge, not every actual T/split |
| `attention_partial_lse_merge_kernel<true,false>` | RK4 FP32 state; same grid/block/smem; reset=true | every RK4 tuple plus second pass carry | historical rotated/LSE test, exact tuples now added |
| `attention_partial_finalize_rotated_kernel` | RK4 grid24*T, block128, smem0; BF16 boundary then inverse H64 | cases32–43, exact T1/3/4/32/63/64 | historical rotated-finalizer T1/17/65 did not establish all owner T |
| `gqa_attention_prefill_fill_bf16_kernel<G,GqaPrefillDirectMetadata>` | BF16 grid=T, block128, smem0 | fragmented append, every owner T/offset pair; independent zero FP64/code-byte oracle and untouched-byte guards | new exact offset/T evidence |
| `gqa_attention_prefill_fill_i8_kernel<G,false,false,false,false,false,false,Metadata>` | INT8 T<32; grid2*T, block256, smem0 | T1/2/3/4/16/31 and offsets0/63; exact owner offsets included | old page-tail evidence was not a complete tuple registry |
| `gqa_attention_prefill_fill_i8_page_kernel<G,false,false,false,false,false,false,Metadata>` | INT8 T>=32; grid(ceil((T+8)/8),4,4), block256 | T32/63/64 at offsets0/63, T32 offset3; byte oracle all K/V and FP16 scale planes | covers both sides of threshold and final partial tile |
| `gqa_attention_prefill_fill_i8_kernel<G,true,true,true,true,true,false,Metadata>` | RK4 T<32, fused H64 rotation/E8 K projection and packed V; grid2*T/block256 | same exact+boundary cases, expected zero codes/scales | no standalone E8/rotate kernel is launched by owner append |
| `gqa_attention_prefill_fill_i8_page_kernel<G,true,true,true,true,true,false,Metadata>` | RK4 T>=32, grid/page warp layout as above | T32/63/64, offsets0/63 and T32 offset3 | page/tail E8 projection uses same production specialization |
| `meank_accumulate_kernel` | all formats use BF16 pre-RoPE K; rows1024; grid(4,max(1,touched_pages)), block256 | existing full Mean unit plus30 visible exact-offset/valid-count/restore cases; independent FP64→FP16 means and exact FP32 tail sum/count | historical4.1 PASS covered kernel but missed several actual offset/T pairs |
| `meank_retain_tail_kernel` | rows1024; n=min(valid,(first+valid)%64); grid4*n, block256, no launch for n0 | exact same-page/cross-page pairs and accepted2; full tail byte oracle, untouched slots; existing full accepted16/provisional tests retained | zero acceptance does not launch this kernel; zero restore tested separately |
| `score_rows` (dot/logits/global max+sum) | D256/H24/K4/M2; grid48, block256, smem0 |7 exact P/domain cases, 16 layers, repeated bitwise output, independent FP64 softmax | historical4.2 M16/P37 covered kernel but not actual M2 domains |
| `score_pages` (second pass/fixed row order) | actual P17/18/32/62/63/128; grid1, block128 | same7 cases; FP64 whole-vector relativeL2<=1e-4, mass error<1e-3, original criteria | supplemental score unit also covers P64/128 M2 plus original P513/37 stress cases |

## Non-kernel operations and intentionally N/A owner families

| Owner operation / candidate kernel | Actual execution and evidence scope |
|---|---|
| Q slot capture | QueryCapturePool receives 16-layer pre-RoPE rows through checked cudaMemcpyAsync D2D. No Q-capture CUDA kernel exists in this owner path. Borrow callbacks synchronize streams/events; they are not kernels. |
| Mean-K export, seed upload, raw-K provisional stage, tail restore, snapshot/reset | checked cudaMemcpyAsync H2D/D2H/D2D and cudaMemsetAsync; not custom kernels. Prefix restoration does invoke meank_accumulate with valid0, explicitly covered. |
| Archive gather, eviction, hydration and recovery/prefix H2D | HostKVTransferEngine copies contiguous per-plane runs using checked CUDA DMA; no gather/scatter CUDA kernel is called by this owner. Full owner sync/init verifies those paths. |
| `gather_paged_kv_kernel`, `scatter_paged_kv_kernel` | N/A actual owner launch. Supplemental small unit cases72/92/112 exercise both directions: 3 selected IDs{3,0,2}, block256, heads4, BF16 two code planes / INT8 four code+scale planes / RK4 packed code+scale planes. Independent byte pattern oracle, not gather/scatter mutual agreement. |
| MTP/window masked append/restore, GDN and full dense GQA kernels | N/A this owner executable: mtp fixture flag selects speculative Main transactions and padded width8; it never constructs or invokes MTPWindow or real GDN/MTP model kernels. Invalid padded suffix is zeroed by cudaMemsetAsync. Product/e2e evidence is separate and is not mislabeled as owner inventory. |
| Dense inverse-rotate output, plain partial finalize, LSE merge<false>, E8Root, 35B geometry, INT8/RK Tile2/Tile8 decode | N/A actual successful owner launch list. Compiled template availability does not establish execution. Existing standalone tests may exercise some; no owner PASS claimed. |

## Exact scoring domains

P is ceil(committed frontier/64), not maximum planner capacity. The exact source-confirmed successful selection shapes are M2, L16 and: P17[1,15), P18[1,16), P32[1,24), P62[1,60), P62[1,62) for R0, P63[0,63) for all-committed, P128[1,126). Cases113–119 keep P and domain exactly. Initial supplemental P64/P128 additions in the score unit are not substituted for this exact mapping.

## Case mapping and numerical checks

JSON contains all120 kernel-unit markers, all30 Mean markers, owner→small and small→owner attention maps, all66 append→owner maps and the reverse map for every one of891 owner records. Append indices are44–71,73–91,93–111; page staging indices are72/92/112;113–119 are exact score domains. Extra append boundary cases have no matching owner record and are labeled additional threshold/cross-page cases. Each actual owner append tuple has at least one matching small case.

Mean exact offset/valid pairs: (0,1/2/16/32/63/64), (1,1/3/4/32/64), (2,1), (3,32), (5,3), (16,16), (32,16), (48,16), (63,1/3/4). Accepted partial2 at offsets1 and63 is added. Zero-row restore offsets0/1/3/5/16/32/48/63 are added; no dropped hard keys, altered production arithmetic or same-implementation numerical oracle.

## Stream-ordering source issue and v2 repair

SPEC found the new Mean tail guard fill used default-stream DeviceBuffer.fill before a nonblocking-stream retain_tail, without a dependency. The original complete d functional/race logs were GREEN and remain unchanged; this is a source-confirmed test ordering defect, not an executed GPU RED. Only Mean test line471 changed to checked cudaMemsetAsync on g.stream. All new kernel-unit setup uses nullptr/default stream. Scorer explicitly checks cudaDeviceSynchronize after default uploads before nonblocking work. Existing Mean provisional/full-accept fills similarly precede checked device drains. No core fill helper or production path changed.

All68 frozen source paths were compared with v1; only Mean changed. Mean-only build f succeeded, followed sequentially by complete no-argument Mean functional/racecheck e. Other kernel/scorer d executable identities and evidence remain valid and unchanged. Full v1 report/manifest/d logs are preserved. The v2 JSON records the repair command, source hashes, logs and owned-tree drain.

## Final build and complete unit runs

修正后 Mean-K 单独重建命令：`VsDevCmd.bat -arch=x64 -host_arch=x64; cmake --build build-vision-integration --target test_meank_accumulate --parallel 4`。日志 `4.4-kernel-unit-build-20261006-f.log`，SHA `05eb3cb08502ac3c7abfb6035d7031623b36772ea18cc814ad818a6f6232efad`，exit 0。下列三 target 的 `e` 构建是修正前的共同构建；终版 Mean-K 绑定 `f` 构建及新的 `e` 功能／racecheck 运行，原记录保留。

Build command: `VsDevCmd.bat -arch=x64 -host_arch=x64; cmake --build build-vision-integration --target test_kvmem_kernel_racecheck test_kvmem_scoring test_meank_accumulate --parallel 4`. Log `D:\deeplearning\NInfer\logs\kvmem-stage4-sparse\4.4-kernel-unit-build-20261006-e.log` SHA256 `4044c60045b21bfea4133e78f7a9494138c9410834ff8dd255a9f7f27760b6d3`. All builds target only these unit executables; application/owner were not relinked.
Compute Sanitizer13.0.85 SHA2567fe67e0b987e956b99bb2d443ecd7776cf4456c5161d385cf7e25116feb91820. Each race command is exactly `compute-sanitizer.exe --tool racecheck --error-exitcode 99 UNIT.exe`, no test arguments, no kernel-name/launch filtering.

| Complete unit | Functional sec | Racecheck sec | Final executable SHA256 | Race log SHA256 |
|---|---|---|---|---|
| test_kvmem_kernel_racecheck | 1.687 | 46.766 | `14c6ad49a0ea082510404beeb3dd8c4c4f04058fd978454c4ed304a182d79a63` | `4bf0b5f4ff547a915f42ba9a9daad046d828a93fa8f4a43a76f96fbe2f84c589` |
| test_kvmem_scoring | 2.078 | 27.422 | `955f69adc00d3509b077e72dbe170bcd568cf43f796d662a22fe857d3a38e23f` | `f89f05dcce2d71710dc931790d2fa8535231f8bf2eec9ce9eec2981cabb6f727` |
| test_meank_accumulate | 0.265 | 7.640 | `c049c8686951401e0c4cb011810b98777e373a4f8b48c75cc30d7d002925ac23` | `52e42b83254c1ec6bb7ef50267e8c361db8b34aff2fc33eef8e83753830236c1` |

Full commands/log paths/record SHA are in the JSON; each final race log ends with `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` and exit0. Final functional logs and executable identities match these race runs. All owned compiler/test/sanitizer trees were checked drained after completion. Full CTest is reserved for Root after test-delta SPEC→QUALITY; implementer did not run full CTest or owner/e2e racecheck.

## Historical evidence and preserved RED

Historical complete Mean and score unit racechecks were already PASS. Historical partial-direct-page and rotated-LSE/page-tail logs report zero hazards but are not credited as every current exact tuple. Fresh complete units close that precision gap. Every historical evidence path/SHA is recorded in JSON.

Test-only RED: first kernel unit reached all44 FP64 partial cases successfully, then append validation rejected the BF16 fixture view because the test had incorrectly attached scale tensors. Original log `4.4-kernel-test_kvmem_kernel_racecheck-functional-20261006-a.log` SHA256df00aca85d1b487383e04c9749f8e7784f1745428d2d6c2363183ab6a3c82b5b; executable298282853c637107ab74b180db9044a12497d0434a0f1d09e2edfcbc0a501d7d. Fixture corrected by leaving BF16 scales absent. This was not a production failure, and no production code changed. Operational external helper/path-regex and missing cmake-PATH failures are preserved separately, not counted as GPU numerical RED.

Owner complete immutable-v9 synccheck/initcheck evidence remains `4.4-owner-final-v9-20261005-sanitizers.json`, bound to b36adb3...fe0e and immutable archive manifest4eed32...4456. This task did not rerun or alter it. Partial owner racecheck remains incomplete/user-stopped; absence of hazards is not a full PASS.

## Test delta and handoff

Exact v9-relative four-file unified patch: `D:\deeplearning\NInfer\logs\kvmem-stage4-sparse\4.4-kernel-racecheck-tests-delta-20261006-v2.patch` SHA256 `5d7e6ac059aec51422ce24eac0cbf8c1832fb75e025f1e56a16c8f1c2b70852a`.

| Test path | Final SHA256 |
|---|---|
| `tests/CMakeLists.txt` | `77084b018c98514a0f3126e11833b5e0301a574706385167666f744d6d1641bf` |
| `tests/ops/test_kvmem_kernel_racecheck.cpp` | `ae55a504778870ef908432cd34af7fcf3f15a680051c4acd6d0547192c1b49a5` |
| `tests/ops/test_kvmem_scoring.cpp` | `88ddbf4cc5d3303ee016a94a7b6cf5dbe34f6d54a3e5cf1bf7dd6e9d5f2cb3e9` |
| `tests/ops/test_meank_accumulate.cpp` | `57b8fdc8da66bfb1f233ca21e508beee121ac8dfd8e90955e0931917d08667f0` |

Self-review: new fixtures use ordinary public ops APIs, no production hooks/fault flags or Dense changes. Existing numerical thresholds and old cases remain; zero append uses exact known FP64/code/scale bytes, page staging compares independently generated expected bytes, Mean compares independent FP64 conversion and tail bytes, score compares independent global FP64 softmax. Temporary inventory lives outside repo and is not part of the test delta. No new optional test subsets. Remaining acceptance work: independent test-delta SPEC and QUALITY, then Root full CTest. Exclusive source/build/GPU lane is returned to Root once this report and manifest are delivered.
