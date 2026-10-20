# DeepSeek 1017 resident profile (experimental)

**Status:** upstream-based source port, deployed locally with partial GPU
qualification. The base is
`Niko1221/Strata@e88ee39d1d6be6e3364183af3b39b62a7ed6f211`.
Deployment wrappers pin the engine separately. Generated settings and model
packs stay local.
Backups stay local; the public repository contains only the `main` branch.

## Scope and source ownership

This fork adds an explicit, opt-in DeepSeek backend to upstream Strata. It
retains the core DeepSeek/EXL3 implementation needed by the pinned checkpoint,
our resident two-GPU execution and the native configure-only installer. Upstream
Qwen, Terminal and desktop behavior stays on the upstream path by default.

The imported core comes from `Jackwwg83/Strata-DS` at
`a46eeba2905de9759a580f830f9377b8ef27ca9b`, with our resident changes at
`afbd39b0dd13911be7ad325359ae606354623546` and
`ccd8cf28e4d72095fef96e00114105b52afc55d4`. Licenses and vendor notices accompany
the imported dependencies. The native installer is carried from the local
integration work; it is not attributed to either published engine commit.

The laptop SAGE installer/recipe, optimized CPU MoE implementation, benchmark
dumps, design-kit snapshots and automatic downloads are not imported. The core
still links some shared low-memory plumbing, but CPU expert calls throw rather
than silently dropping work. Construction rejects single-device mode before
opening a pack or calling CUDA. Both stages' backbone experts must fit in VRAM.

## Invariants

- Only `sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw` at
  `22205b9aff07ae7b999d466aaf96446c20f4ea97` is the deployment target.
- No pretrained-weight requantization, model download, model-sized host arena,
  automatic service, or unrequested GPU process is part of this port.
- Default placement runs layers 0–17 and 18–39 on two CUDA-visible devices.
  Optional TP2 shards projections across both devices. Backbone experts remain
  resident in either mode. Engram shards stay independently available on disk.
- Capacity is at most 524288; published RoPE remains unchanged. Ordinary decode
  is the engine default. The deployment selects layer split plus DSpark after
  matched short-workload measurements; this is not a general speed guarantee.
- Setup writes private configuration and launch files only on explicit
  `--configure-only`; startup reads the installed configuration. Logs, machine
  paths and device identifiers are not committed.
- Bind is `127.0.0.1`; an optional API key is supplied through `STRATA_API_KEY`.

## Build and configure

With an existing CMake/Ninja/CUDA toolchain (no dependency fetch):

```sh
cmake -S . -B build-sm120-resident -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_DS41=ON \
  -DSTRATA_DS41_STANDALONE=ON -DSTRATA_DS41_STAGE_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build build-sm120-resident --parallel 2
STRATA_PYTHON=/path/to/existing/python bash setup.sh \
  --family deepseek --model 1017-EXL3-2.7BPW --configure-only \
  --data-dir /path/to/strata-data --gpus 0,1 --telemetry-gpus 0,1 \
  --context 524288 --port 8081 --spec none --parallel 1 --no-start --no-browser
```

The resident backend is off by default. Standalone builds omit the upstream
Qwen targets; they do not install a llama.cpp tree. The binaries are under
`build-sm120-resident/bin/`. Configure-only writes private `strata-deepseek-1017.json`
and `run-deepseek-1017.sh`; settings change only on explicit reconfiguration.
Unrelated files are not overwritten, generated changes get private backups, and
the launcher/config digest must match before startup. `--log-dir` selects the
shared `service.lock` directory; keep it shared with any other deployment using
these GPUs. Starting the generated launcher is a separate, manual operation.

## Acceptance and limitations

### Tensor parallelism and non-replay verification

The requested extension adds an opt-in tensor-parallel mode alongside the
existing layer-split mode. In TP2, both ranks execute every layer, hold disjoint
output-channel shards of the quantized projections and routed experts, and
exchange intermediate activations. Replicated normalization and routing, and
peer-shared KV state, are not claims of tensor sharding. Published EXL3 bytes are sliced and
copied, never dequantized/requantized into a replacement checkpoint.

Resident verification evaluates multiple tentative token rows per layer and
commits the accepted prefix's rings, compressor state, logits and draft inputs
without replaying a backbone token. Rejected rows never become committed state.
The existing ordinary-decode path remains the default until same-workload GPU
measurements justify a change. Functional DSpark support alone is not a speedup.

Acceptance requires bounded-memory loading, two-rank collective/shard tests,
real-model PP/TP numerical and greedy-output comparisons, accepted/rejected
prefix and continuation tests, cancellation/failure checks, and matched
speculation-on/off timings. No competing model-sized process, automatic model
download, service installation or implicit publication is part of this work.
The implementation is opt-in with `--parallelism tensor` on the native setup,
`ds41_serve` or `ds41_generate` command. Do not combine it with `--split-layer`.
The default `--parallelism layer` remains a layer split, not tensor parallelism
and not an overlapping pipeline scheduler. Neither mode changes the default
`--spec none`.

TP2 uses a two-rank, intra-process output-channel decomposition. Each rank owns
half the large EXL3 projections' output tiles, their output scales, and half the
embedding channels. Query heads stay local through sparse attention; each rank
owns four complete grouped attention-output projections. Their combined output
is exchanged once before the sharded final attention projection. Small backbone
query-down, KV and shared-expert projections are replicated, adding approximately
426 MiB per rank compared with sharding those projections. Input scales, small BF16 matrices, normalizations, router weights and
sliding-window/compressor state remain replicated. The append-only compressed
KV/index cache has one owner, selected on the device with more free memory at
load; its peer reads it directly. Only the owner writes or restores cache rows.
Stream collectives order index reads, and explicit barriers order compressed-KV
reads after the owner writes. No model-sized host cache is introduced. This is
not expert parallelism: both ranks
hold a shard of every routed expert and compute every selected expert. A peer
all-gather reconstructs activations only at shard boundaries, including between
an expert's hidden and down projections. FP16 expert activation and FP32 expert
summation order are preserved. There is no NCCL dependency or reduction across
partial dot products.

This decomposition deliberately prioritizes numerical parity and the existing
EXL3 representation over communication minimization. It requires bidirectional
CUDA peer access and refuses unsupported topologies rather than silently becoming
a layer split. Collectives are CUDA-graph-replayable; a rank failure signals its
peer to fail rather than wait indefinitely. Weight loading uses bounded pinned
staging and column copies of the original tiles. No replacement checkpoint or
model-sized host allocation is created.

Peer exchanges push activations into double-buffered mailboxes, then copy locally
into the receiving activation. Stream-ordered epochs protect each mailbox from
reuse, including graphs replayed at different rates. The mailboxes use a bounded
64 MiB arena per rank; graph-captured addresses stay stable when a later shape
needs a larger slot. Cache publication uses separate ordering barriers.

Verification batches dense projections in native small-row MMA kernels that
decode each shared quantized tile once for up to eight rows. Verification windows
remain at most four input rows. The layer-split path transfers the tentative hidden
rows, hyperconnection mixes and the layer-14 compressor/indexer boundary into the
last stage; no backbone token is replayed on commit. It commits only the accepted
prefix's window, compressor, logit and DSpark input state. TP2 runs the same
transaction on both ranks. TP2 prefill currently uses bounded four-row windows,
not the layer split's larger GEMM prefill chunks; it is not a long-prefill speed
claim. Both ranks still need room for their weight shards and per-rank scratch.

`--spec dspark` uses the published MTP tensors. Its dense work is batched too.
Only MTP experts may stream from disk, through bounded GPU staging. An MTP-only
GPU LRU avoids repeated uploads: `--draft-cache-mib 1024` is the ceiling per
participating draft rank (default), `0` disables it, and `8192` is the maximum.
It grows on demand, limited by free VRAM minus the configured reserve, and never
evicts a backbone expert or consumes model-sized host RAM. Cache hits count
actual routed uses served without uploading again (in TP2, both shards must hit).
The current stage's cached working set is pinned while batched experts execute;
smaller cache ceilings use the bounded single-expert fallback. Size it for the desired
context, not just a short benchmark. The setup flag is available only on this
explicit 1017 profile; installed defaults are never changed implicitly.

Snapshots also preserve the optional draft rings and pending main-hidden rows,
using about 4.1 MiB of bounded host state per snapshot per draft rank. This avoids
changing draft acceptance after restoring a conversation snapshot; greedy
correctness alone would not catch that performance regression. The compressed
cache saved by single ownership is a computed 1600 MiB at 524288 capacity.

Qualification is recorded separately below. Small tests, capacity allocation and
timing experiments do not establish independent full-model numerical correctness
or actual-524K input qualification.

### Reproducing the new checks

Component CTests use synthetic weights and never load the model. Run them serially:
`ctest --test-dir build-sm120-resident --output-on-failure --parallel 1`.
They cover native shared-row arithmetic, strided peer collectives and graph
replays, mixed-rate expert shards, refusal of unsupported residency policies,
and input-thread lifecycle. The Linux pipe reader is owned and joined: QUIT no
longer waits for an open writer to close stdin, and fatal exits can stop an idle
reader. STOP delivery, CRLF, chunked long lines and a final unterminated line
remain supported.

The following programs are explicitly model-sized, not automatic CTests. Stop the
model service first, use the same shared `service.lock`, and run only one at a
time. Supply the existing lossless `--pack` and a comma-separated `--ids-file`.

- `ds41_stage_serving_model_test --dspark 1`: full-logit prefix commits 1–4,
  ring/index/compressor boundaries 127/128/511/512, session isolation, draft
  replay and cancellation. Add `--tensor 1 --max-seq 524288` for TP2 with full
  configured capacity. `--reference-out` / `--reference-in` compare PP/TP logits
  after the same sequential input arithmetic; they are not a separate model oracle.
- `python tools/ds41/test_resident_protocol.py --exe ... --pack ... --ids-file ...
  --parallelism tensor --spec dspark`: the real native pipe, greedy replay,
  cancellation/recovery, invalid IDs/context overflow and clean shutdown.
- `ds41_speculation_model_test --tensor 0 --spec 0 --reference-out ...`, then
  `--spec 1 --reference-in ...` and the two `--tensor 1` cases with that same
  reference: fixed greedy output, one warm-up and three timed runs. All use
  sequential prefill, outside decode timing. Larger layer-split GEMM prefill and
  small-row GEMV prefill need not be bit-identical; mixing their initial states
  would confound a speculation comparison. Timings include draft, verification,
  direct commit, graph/host dispatch and accepted-output bookkeeping, not loading
  or prompt processing. Test output is not a coding-quality score.

The source port is checked with a source/build dependency audit, offline installer
tests, DeepSeek tokenizer/parser/API tests and selected regressions of touched
upstream interfaces. Compilation does not load weights. GPU and real-model tests
run separately from the model server, never as competing model-sized processes.

Historical measurements in the old fork qualify that old binary only. They are
not performance or full-model correctness evidence for this port. The local
deployment uses a reviewed source pin; passing startup and short-input checks
does not qualify the full context or every API path.

### Initial source-port checks — 2026-10-20

- CUDA 13.4 / GCC 15 / SM120 build: `ds41_serve`, `ds41_generate`, component
  binaries and the opt-in model-test binaries compile. This does not load weights.
- Four host CTests pass: stage planning, suffix drafting, request parsing and
  resident-policy/CPU-fallback refusal. GPU component tests are compiled, not run.
- Fifteen native profile tests, 29 lossless packer tests, 101 selected upstream
  setup tests and 442 selected upstream serving tests pass.
- Twenty DeepSeek tests pass using the actual tokenizer files and CPU stand-ins:
  reference rendering, token IDs, parsing, OpenAI streaming/nonstreaming,
  Responses, Anthropic Messages, tool calls, overflow/image refusal and startup
  through the native server's config/pipe-engine path. No real model is used.
- Configure-only checks the existing pinned pack and creates ignored private
  launch files. The launcher's shared lock refuses a second model process while
  the original deployment runs. No original config, pack or service is replaced.
- A failed early stand-in test used an invalid `DONE` fixture; the fixture is
  corrected to the documented native protocol and the final run passes. Private
  failed/follow-up receipts remain outside this repository.

### Local GPU qualification — 2026-10-20

- All 11 component CTests pass, including GPU transfer/upload, MCG decoding/pack
  checks, EXL3 linear operations, staged experts and bounded prefill experts.
  The initial MCG pack test expected the old MCG-specific refusal message; its
  assertion now checks the resident device-policy guard, which rejects earlier.
  Production engine behavior is unchanged.
- A 513-token real-model retrieval case generates `BRONZE-73921` and passes
  batched/sequential predictive agreement, snapshot restore, deterministic replay
  and cancellation recovery. Final prefill logits are byte-identical to the
  historical binary on that fixture. This is not an independent model reference.
- The native deployment builds from this source and starts loaded/healthy with
  524288 configured capacity. The actual 524256-token rerun and queued follow-up
  checks were cancelled at the user's request, not counted as passes.

Still unverified: complete actual 524K input, fresh real-model API qualification,
broad numerics/quality/performance, and container GPU runtime. No full
upstream-suite or cross-backend build coverage is claimed. Private failed and
cancelled receipts remain local.

### TP2 / non-replay DSpark experiment — 2026-10-20

#### TP latency redesign — implemented and qualified

The performance path keeps the 32 owned query heads local through sparse
attention and assigns four complete grouped output projections to each rank.
It exchanges the combined grouped output once, rather than gathering the query
and eight independent group outputs. The small backbone query-down/KV and shared
expert projections are replicated (computed additional weight ceiling about
426 MiB per rank); routed experts remain tensor-sharded and resident. No
pretrained bytes change. Source-level counting gives 170 collective/barrier calls
per ordinary token, down from 686. Posted peer stores fill private ping-pong inboxes and local
notifications avoid polling the peer's memory. Stream epochs protect reuse;
mutable draft projections retain the optional consumption acknowledgement.
Index-key and compressed-KV publication have explicit ordering barriers.

A diagnostic run with blocking CUDA host waits eliminates measured cgroup CPU
throttling but reduces TP decode throughput; it is not adopted as engine policy.
The benchmark-only `--blocking-sync 1` switch reproduces that comparison without
changing launcher defaults. `--profile 1` brackets one measured round with the
CUDA profiler API. Profiler-injected failures are kept separately from successful
uninstrumented timings; they do not establish kernel-level attribution.

Final checks retain exact PP/TP and sequential/batched logits, wrong-proposal
rollback, both device orders, full configured capacity, native cancellation and
matched timings. All 15 component CTests pass, including poisoned peer-head
attention tests and changing/skewed mailbox epochs at rows 1/4/8/32/64, stable
layouts across growth, strided padding and allocation-budget refusal. Real-model
checks pass at 524288 configured capacity and in reversed device order. Three
new TP native-pipe runs pass with speculation off/on and the MTP cache disabled.
The layer split's 513-token prefill and sequential logits remain byte-identical
to the previous published binary. No actual-524K input run is claimed.

The final matched run uses the same 63-token coding prompt, 96 greedy output
tokens, sequential prefill outside timing, one warm-up and three measured runs,
1024 MiB MTP-cache ceilings, CUDA 13.4 / GCC 15 / SM120, two RTX PRO 6000 GPUs
over PCIe P2P, 16 GiB host-memory limit, no cgroup swap and a two-CPU quota:

| Placement | Speculation | Median tokens/s |
| --- | --- | ---: |
| Layer split 18/22 | None | 68.090 |
| Layer split 18/22 | DSpark | 80.808 |
| TP2 | None | 41.991 |
| TP2 | DSpark | 43.026 |

Every mode produces the same greedy sequence; DSpark uses 33 windows and accepts
2.909 inputs/window, including warm-up. Layer DSpark gains **18.7%** over ordinary
layer decode on this fixture. TP improves over the original candidate but still
loses to the layer split. The chosen deployment profile is **layer split + DSpark**;
TP remains opt-in and experimental. Engine CLI defaults remain layer/none so
existing configurations do not silently change. These are native short-fixture
measurements, not HTTP, long-context or coding-quality qualification.

Full-model runs under Nsight injection exit with code 139 before a report, both
with CUDA event tracing enabled and disabled; the cause remains unresolved. A
synthetic collective trace succeeds, but it does not establish full-model
kernel-level cost attribution. Failed and pilot receipts remain private.

#### Original candidate, before the TP latency redesign

These are candidate-source results, not a change to the installed service or its
defaults. CUDA 13.4 / GCC 15 / SM120, two RTX PRO 6000 Blackwell GPUs, PCIe P2P,
48 GB host RAM. Model-sized cases run serially, with a 16 GiB host-memory limit,
no cgroup swap, and a two-CPU quota. No pretrained weights were changed.

Matched native-engine decode experiment: one 63-token chat-mode prompt asking
for a Python thread-safe LRU cache, 96 greedy output tokens (no EOS in that
prefix), identical token sequence in every mode. Sequential prompt processing is
outside timing. One warm-up plus three timed runs; median below. The MTP cache
ceiling is 1024 MiB per draft rank. Rates are not HTTP request throughput, long
context performance, or the earlier OpenCode screenshot's workload.

| Placement | Speculation | Median tokens/s |
| --- | --- | ---: |
| Layer split 18/22 | None | 69.489 |
| Layer split 18/22 | DSpark | 82.655 |
| TP2 | None | 18.949 |
| TP2 | DSpark | 19.999 |

DSpark improves the layer-split case by **18.9%** on this fixture, and TP2 by
**5.5%** relative to TP2 alone. All DSpark runs accept 96 inputs in 33 windows
(2.909 inputs/window), including the warm-up: restoring a snapshot no longer
changes draft acceptance. This does not predict the acceptance rate or speedup
on another conversation. The current output-channel/all-gather TP implementation
is substantially slower than the layer split on this PCIe topology. At this
earlier stage, layer split and ordinary decode remained the defaults. The later
deployment decision above selects layer DSpark; TP2 is not recommended for speed.

Earlier failed/pilot receipts are retained: the first TP capacity load exhausted
headroom with duplicated compressed caches; an early timing comparison mixed
GEMM and GEMV prefill; and draft-ring omission from snapshots reduced acceptance
on repeated runs. The first native-pipe run also exposed a detached-reader QUIT
hang, now fixed and covered by the input lifecycle test. These attempts are not
counted as final qualification.

Completed qualification:

- All **14 component CTests** pass. Shared-row dense/head outputs match separate
  GEMVs byte-for-byte for varied inputs, rows 1/4/8 and padded strides. Two-rank
  expert shards match unsplit FP32 outputs byte-for-byte, including repeated
  routes, mixed bit rates, rows 1/4/8, top-k 1/6 and graph replay.
- **65** profile/tokenizer/mock-HTTP/packer Python tests pass, plus **95**
  additional selected upstream setup regressions (the 111-test setup run includes
  the same 16 profile tests). HTTP adapter tests use CPU stand-ins, not the model.
- Real-model layer-split and TP2 checks pass prefix commits 1–4, deliberately
  wrong rejected proposals, exact full-logit continuation across positions
  127/128/511/512, isolated session slots, published drafting and cancellation.
  PP/TP controlled sequential decode matches all logits for 16 generated tokens.
  Pinned-host layer handoff is covered too. Both device orders were exercised
  during development; the final TP rollback run uses the deployment order.
- TP2 loads with **524288 configured capacity**, all backbone expert shards
  resident, and passes the above short/boundary checks at that capacity. This is
  **not** an actual-524K input pass; that longer qualification was not resumed.
- Five real native-pipe cases pass: layer/TP2, speculation off/on, and TP2 with
  the draft cache disabled. They cover READY, fresh greedy replay, STOP and
  recovery, invalid IDs/context overflow, and QUIT with stdin still open.
- The default layer-split path's 513-token fixture still retrieves
  `BRONZE-73921`. Both its batched-prefill and sequential logits are byte-identical
  to the previous published binary; snapshot, reset and cancellation checks pass.

The persistent service remains stopped. No fresh real-model HTTP, full-524K input, independent
full-model reference, broad coding-quality or container qualification is claimed.
