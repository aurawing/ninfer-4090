# Local synthetic KVMem quality measurements

This suite measures six seeded synthetic tasks. It does not establish real-corpus
quality acceptance. Tooling owns fixtures, strict validation and measurements;
the public Engine owns prompt preparation, generation and normal prefix reuse.

All generated inputs, truth, frozen identities, raw outputs, logs and reports
belong outside the repository. The Stage 4.5 destination is
`D:/deeplearning/NInfer/logs/kvmem-stage4-5-6`; use new named subdirectories and
never overwrite a freeze or raw batch. Generation is once per frozen case/profile.
An incorrect model answer is a quality failure, not a request to retry.

## Lifecycle and commands

Run `python tools/quality/kvmem_quality.py --help`. CPU-only commands:

1. `draft --output EXTERNAL/draft.json` creates the six tasks at context caps
   131072 and 262144. `--cases 5` means five paired documents/questions, each with
   a single and historical form: **ten cases total per task/context/profile**.
   Additional controls are `--contexts`, `--tasks` and `--filler-lines`.
2. `fit-request --draft EXTERNAL/draft.json --model MODEL --context 131072
   --output EXTERNAL/fit-131072.json` prepares a count-only fitting batch. Repeat
   for the other cap. These commands do not execute the Engine.
3. In the exclusive GPU lane, run `ninfer_kvmem_quality_bench REQUEST.json
   NEW_OUTPUT.jsonl`. A request has operation `fit`, `count`, or `generate`, one
   profile, fixed policies, and cases. Fitting loads an Engine/tokenizer but
   generates **no answers**. Only the unit count of the frozen seeded neutral
   background stream changes; both full chat
   forms of each document fit together. Headroom is the fixed 256-token short
   budget or 1024-token summary budget plus 128 guard tokens. The harness emits
   actual tokenizer counts and finalized messages. There is no character-count
   estimate in a freeze.
4. `freeze --counted EXTERNAL/fitted-131072.jsonl EXTERNAL/fitted-262144.jsonl
   --identity EXTERNAL/identity.json --output EXTERNAL/frozen` stamps all message,
   policy, document, truth and case-file SHAs. Freeze audits the actual document
   against independently parsed reference entries, graph, counters and facts.
   `audit --frozen EXTERNAL/frozen` verifies every hash and re-runs the input audit.
   Case IDs must be safe single filename components. Every intended case path
   is resolved and checked against the new freeze root before any directories or
   files are created; unknown input forms are rejected during audit.
5. `plan --frozen EXTERNAL/frozen --costs EXTERNAL/pilot-costs.json
   --budget-seconds SECONDS --output EXTERNAL/plan.json` fixes run order and skips
   before suite inference. Deduct final gates from the GPU allowance first.
   `cold_case_seconds` is a required conservative fallback cost; optional costs
   are keyed by profile ID, then `task/form`. Inputs never depend on correctness.
   High priorities fill first. Within each priority, seed-major ordering spreads
   cases across tasks and context caps. Once a comparison bundle exceeds remaining
   budget, all later bundles are skipped. Both forms, denominators and shared reference
   stay paired, and identical reference case/profile requests appear only once.
   Normal-suite planning requires exactly one single/historical pair per semantic
   task/context/index, unique IDs, and identical document, question, truth and
   blueprint identities. It rejects missing, duplicate or mismatched partners
   before cost selection. Isolated historical reuse/cost pilot freezes remain
   supported; they do not enter this normal-suite planner boundary.
6. `run --frozen EXTERNAL/frozen --plan EXTERNAL/plan.json --runner HARNESS
   --model MODEL --output EXTERNAL/run --timeout SECONDS` runs hidden owned
   processes, preserving raw JSONL and stdout/stderr for each persistent Engine
   batch. Python validates outputs after generation; it never feeds them back.

`identity.json` must contain `tokenizer_artifact_sha256` (SHA of the exact `.ninfer`
artifact), `counter_binary_sha256` (SHA of the exact harness), and
`production_identity` with `source_tree_sha256` and `files`. The source tree value
comes from `kvmem_quality.production_source_sha()`; `files` maps absolute paths to
SHAs of the measured CLI/server/runtime binaries and DLLs as applicable. Supply
the complete production identity from the orchestrator. Run and resume verify
these paths, runner source SHA map, harness SHA, model SHA, frozen manifest and
plan before dispatch. Finish code changes and reviews before the final freeze.

The ordered ledger is append-only and fsynced. Re-running `run` with the identical
identity resumes completed batches without repeating any case/profile. A crash
leaves an exclusive lock. After its orchestrator has exited, `recover --output
EXTERNAL/run` reconciles completed raw rows; started rows lacking completed output
become terminal infrastructure errors and are never retried. Recovery then removes
only that exact owned lock file. A malformed final raw JSONL line remains on disk
and is ignored during reconciliation; corruption earlier in the file fails closed.

An `OwnedTreeFatalError` means ownership setup or confirmed process-tree drain
failed. The dispatcher records `fatal-owned-tree`, preserves its exclusive lock,
raw files and unfinished rows, and immediately stops without a normal report or
another batch. Failed JOB handles remain held by the owning Python process.
Automatic `recover` and `run` reject this incident even if its lock is removed;
external inspection must establish process ownership/drain before any separately
authorized recovery. Ordinary child nonzero exits or timeouts after confirmed
tree drain remain infrastructure results, distinct from this fatal condition.

## Reuse pilot

`reuse-pilot-draft --output EXTERNAL/pilot-draft.json --context 32768` creates three
different short questions about one exact historical document with the fixed
assistant acknowledgement. Count/fit, freeze and audit it before generating a
public Engine `generate` batch. The pilot must preserve its identity/output and
record `reused_prompt_tokens`, `computed_prefill_tokens` and `prefix_reuse_path`.
It establishes whether ordinary cross-question reuse works with the existing
saved checkpoints. Zero reuse means budget with honest cold prefill costs.
No configuration-specific model answer is appended to another input. Pilot cases
are separate identities from the final quality suite.

The 2026-10-06 pilot at context cap 32768 measured three distinct historical
questions sharing the same frozen document/acknowledgement prefix; each complete
prompt contained 32370 tokens. All three used
`FullReset`: reused prompt tokens were **0**, and computed prefill tokens were
32370 each. Prefill times were 15.548, 16.666 and 15.703 seconds, with all three
answers correct. Its preserved evidence is
`D:/deeplearning/NInfer/logs/kvmem-stage4-5-6/reuse-pilot-v3-20261006/reuse-result.json`.
This supports cold-prefill budgeting for these public checkpoint semantics. Final
fixtures keep their existing per-task documents; no cross-question amortization
or composite-document extension is claimed.

The neutral background uses a frozen seeded pool of 128 sentences with varied
subjects, actions, locations and grammar patterns, rather than a repeated single
sentence. Each draft contains the pool, insertion recipe and a blueprint SHA;
neutral count fitting preserves that SHA. Background text cannot contain answer
records, variable assignments, marker words or planted fact values. This remains
a controlled sparse synthetic fixture: planted record pages may all fit in a
32K view, the specified target X chain has predictable labels, and the background
pool cycles. Those limitations must be considered when interpreting the results.

## Task and validation contracts

- Needle: eight exact key/reference records, including four lookalike distractor
  keys; query one target and require the exact eight-digit string in JSON.
- Repetition: one key occurs at four separated locations with four distinct
  references; require all four in document order, with no missing or duplicates.
- Multi-key: retrieve four exact keys together, requiring the entire JSON map.
- Chain: dispersed X1 through X5 assignments, interleaved with two independent
  distractor chains; resolve the graph and require all five target variables.
- Frequency: dispersed occurrences of six specified marker words with unique
  frozen frequencies; require the top three in descending frequency order.
- Summary: a long document with ten planted project facts. Measure exact whole
  lines `Entity | field | value`; each fact must occur once with its correct
  anchor and value. Wrong anchors, duplicate links, or missing facts fail strict
  all-ten completion. Report average ten-fact coverage separately. Generation
  must produce exactly 1024 token IDs through the existing request policy
  `stop.include_model_defaults=false`, never through a runtime change.

Short validators parse the entire response as JSON, reject duplicate object keys,
trailing text and partial/substr-only answers. Summary validators do not give
credit merely because a value appears elsewhere. Fixed greedy sampling overrides
all penalties and truncation settings; thinking is disabled while
`preserve_thinking=true`. C=1, MTP3 and optimized proposal head apply throughout.
Non-dense MTP window stays at the existing 32768-token default. Images and disk
prompt caching are disabled. Single form has one long USER and current boundary 0;
historical form is USER document, fixed ASSISTANT acknowledgement, USER short
question with explicit current boundary 2. Compared profile inputs are identical.

The selected JSON-only and document-order prompts measure structured answer
adherence as well as retrieval. A correct answer in a different format can fail.
Summary coverage counts literal anchored facts and can undercount correct
paraphrases. A malformed-format or exact-match failure is not, by itself, proof of
a memory or page-selection failure. Reports state this scope, aggregate failure
reasons, and preserve raw outputs for interpretation; scoring stays frozen and
there are no answer-selected retries or manual pass overrides.

## Profiles, accounting and reports

P1 compares KVMem INT8 view32768 at ctx262144 against tiered-exact INT8. P2 compares
KVMem INT8 requested view131072 at ctx262144/131072 against dense rk4v4-e8. P3 is
ctx131072/view32768 against tiered-exact INT8. Both KVMem scoring denominators are
explicit profiles: kept-band and `NINFER_KVMEM_SCORE_ALL_PAGES=1` (all-committed).
P4 is informational rk4 archive at ctx262144, **the same requested token counts**
32768 and 131072, both denominators, compared to the already measured matching
INT8 profiles. No requested token doubling or silent view adjustment is allowed.

The JSON report includes task/context/form success rates and strict planned-case
rates, paired reference gaps, paired denominator differences, actual budget skips,
summary mean coverage and all-ten completion. Infrastructure failures are counted
separately and never become quality passes. Each result retains exact identities,
requested/actual view, actual prompt/output count, raw outputs, validator version,
reuse path/counts, MTP accepted/drafted numerator/denominator, decode committed
tokens per second, public TTFT timings and original sparse turn logs.

`capture_wait_ms` in the original runtime log is **GPU QUEUE DRAIN WAIT**, not Q
DMA. The report alias `gpu_queue_drain_wait_ms` makes this explicit; it is a subset
of prefill and is never added again to TTFT. The original log schema is preserved.
Sparse logs provide authoritative actual selected view; for other modes the public
resolved Main KV capacity is recorded with that source label.

Targeted CPU CTests: `ctest -R '^ninfer_kvmem_quality_(contract|render)_test$'
--output-on-failure`. Targeted build: configure benchmarks ON, then build only
`ninfer_kvmem_quality_bench`. The orchestrator owns the final full gates, GPU lane,
commits and pushes.

The harness also supports `render` to verify exact C++/Python background and
injection placement parity. This operation does not construct an Engine or load
a model; its targeted CTest covers all six tasks and both forms at several unit
counts, including a partial pool cycle.
