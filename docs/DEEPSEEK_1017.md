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

### Prefill dispatch and handoff candidate

**Status: implemented opt-in; local native/HTTP E2E and 2.7-bpw long-input requalification pass.**
`DS41_PF_BULK_HANDOFF=1` enables grouped event-ordered peer handoff during
layer-split GEMM prefill, with bounded pinned-host fallback. Completion is fenced
before source reuse; there is no cross-chunk pipeline overlap.
`DS41_PF_HC_BATCHED=1` enables two-launch, grid-batched exact-order K15 with
caller-owned scratch. Both default to off, preserve decode K7 and configured
capacity, and do not accelerate token-wise TP prefill. Per-call attribution
separates stage compute wall time, handoff, setup/cleanup and HC launch counts.
The local E2E gates pass; sustained timing for these two prefill controls and a
matched Maya comparison have not been established.

Local 2026-10-25 validation passes all 18 component CTests, including HC shapes
1..16384 with partial tiles, independent concurrent scratch and graph replay,
plus both transfer directions, eager/graph producers and consumers, bounded
forced-host copies larger than staging, and malformed-group refusal. The 63
Python packer/installer/frontend/launcher checks also pass. CUDA-only changes are
not a cross-backend portability claim.

`ds41_prefill_e2e_model_test` checks all four combinations of the new flags with
unchanged chunk/batch 512/256 and capacity 524288. It passes for 2.7-bpw grouped
and 2.9-bpw reconstructed prefill, with actual 4097-token peer inputs and
513-token forced-host inputs. All four modes have byte-identical full logits at
the same actual shapes. Snapshot/reset, cancellation/recovery, odd-position
751-token follow-ups and exact warm replay pass. The stronger existing two-turn,
96-draft/verify warm DSpark test also passes for 2.7; a 2.9 token-wise TP
regression passes without claiming the new layer-split optimizations affect TP.

On a real 16384-token 2.7-bpw fixture, both old and new paths pass exact replay
and byte-identical cross-path logits. Actual dispatch counters are:

| Work | Existing path | Both candidates enabled |
| --- | ---: | ---: |
| Peer copy calls | 17472 | 128 |
| Host handoff waits | 34944 | 32 |
| HC-pre kernel launches | 327680 | 10240 |
| Logical handoff bytes | 713293824 | 713293824 |

These are dispatch counts, not an Nsight kernel trace or a speedup claim. Timing
observations use profiling and only one timed round after warm-up; they are not
a qualified throughput comparison. Profiling deliberately fences stage boundaries
to attribute completed wall time; setup includes scratch/cache handoff and reader
initialization, not model loading or tokenization. Without profiling, wall
subfields stay unset and the extra attribution fence is absent.

Real local HTTP chat E2E passes for both checkpoints with both candidates on:
arithmetic, 2848-token retrieval, prompt replay, a 4038-token follow-up reusing
2854 tokens (1184 new tokens), and disconnect/slot-release/recovery. Native
callback cancellation is checked separately; a client disconnect alone is not
proof of the engine's exact stopping point. Tests use temporary loopback-only
servers, then stop them; the installed service/configuration remain untouched.
The supported 2.7-bpw candidate also passes actual 65536- and 524256-token
retrieval at 524288 capacity: both return `BRONZE-73921`, retain chunk/batch
512/256, and have byte-identical full logits to the respective historical grouped
path. Both keep every backbone expert resident, pass in-capacity snapshot
continuation, and record zero swap/OOM events. Peak observed cgroup memory is
about 9.47 GiB under the 16-GiB limit. These are one-shot correctness runs, not
repeated speed benchmarks. The 2.9 profile is qualified only on the smaller E2E
inputs above; its actual 64K/524K inputs remain untested. Device-fatal CUDA fault
injection is not performed.

The explicit model runner is `ds41_prefill_e2e_model_test --pack PACK --ids-file
IDS --out DIR --experimental-29 0 --force-host 0`. Create `DIR` first and use
513..16384 frozen input IDs. It checks all four candidate flag combinations at
524288 capacity. Set `DS41_PF_PROFILE=1`; for the supported profile use
`DS41_PF_GROUPED=1`. For the experimental profile use `--experimental-29 1`,
`DS41_EXPERIMENTAL_PROFILE=libertaidai-29`, `DS41_GRAPH=0` and
`DS41_PF_GROUPED=0`. `--force-host 1` exercises the bounded host fallback. These
are model-sized tests: hold the shared service lock and run only one at a time.

### Bounded 2.9-bpw GPU decode-expert reuse candidate

**Status: implemented opt-in; local native/HTTP/default-regression checks pass.**
`DS41_EXPERIMENTAL_DECODE_CACHE_MIB` requests an additional per-device GPU cache
for immutable streamed experts (0 by default, integer 0..8192 MiB). It applies
only to the separately gated 2.9 profile; no host expert cache, CPU MoE,
requantization, production promotion, or automatic evaluation restart is included.
The static resident tier and its reserve remain unchanged. Cache allocation is
strictly bounded; a nonzero budget too small for one complete expert fails closed.
For isolated timing comparisons, `DS41_EXPERIMENTAL_RESIDENT_SLOTS=A,B` fixes the
static layer-major prefix counts by logical stage/rank (not physical GPU ID).
Both counts must be positive and fit the current VRAM budget; initialization
fails instead of silently reducing either. Unset keeps automatic sizing. This
separate experimental control avoids attributing a desktop/free-VRAM fluctuation
to the decode cache; it does not change the supported 2.7 policy.

An LRU lookup retains complete stored expert bytes, protects every selected
cached expert from eviction within a top-k group, and falls back to the existing
six-slot staging arena if all cache entries are protected. Before reuse/eviction,
the nonblocking consumer stream is fenced. Descriptors and selections are
published on that stream before the caller can consume them. Failed reads never
publish a valid cache entry. Cache contents are immutable weight state, not
conversation state, so reset/snapshot need not clear them. Prefill continues to
use its separate ring and is not accelerated by this candidate.

Acceptance requires synthetic hit/eviction/duplicate-route/protected-slot and
read-error tests, unchanged cache-off outputs, exact cache-on logits and greedy
tokens across reset/snapshot/warm-prefill/cancellation, and serial same-fixture
cache-off/on timing rounds with resources and actual streaming counts retained.
Layer and TP compatibility are tested separately. A faster short fixture does
not establish cliecy's 89–112 tok/s or broad quality equivalence. The user's
paused quality suite and installed 2.7 fallback stay stopped and unchanged.

Matched local validation (2026-10-25) fixes the static resident sets at 6200 and
6800 experts, uses the same frozen 63-token coding prompt and 96 greedy decode
steps, one warm-up plus three measured rounds, 524288 capacity, layer split
18/22, no speculation/graphs, 16-GiB host-memory limit, zero swap and two-CPU
quota. All modes use the same binary. Timing is the sum of wall intervals around
`Engine::step`; logit validation and serialization are outside those intervals.
The actual prefill shapes are 63/63 on both stages. There is no EOS in the
96-token output, and every greedy token and full logit vector is byte-identical
across modes and repeats.

| Additional cache per GPU | Median decode tok/s | Uploaded experts per 96-step timed round |
| --- | ---: | ---: |
| Off | 9.765 | 3683 |
| 2048 MiB | 20.414 | 1313 |
| 4096 MiB | 29.641 | 635 |

The 4096-MiB candidate improves this native short-fixture median by **3.04x** and
reduces streamed uploads by 82.8%. Its warm-up is slower than its timed rounds
because weight-cache contents survive conversation reset; cache warm-up is not
counted as a timed round. This is not HTTP throughput or a cliecy-matched result:
their 89–112 tok/s uses different weights, prompts, placement, large host-pinned
tables/experts and DSpark. Prefill is unchanged by this cache. The exact static
count control is for measurement, not a recommended production preset.

All 18 component tests pass, including protected entries, duplicate routes,
cross-layer LRU eviction, malformed budgets/fixed counts, allocation-budget
refusal and short-read recovery. Native real-model tests retain the historical
cache-off 2.9 logits and exact cache-on replay, snapshot/reset, warm prefill and
cancellation/recovery. A separate 128-MiB-cache TP2 retrieval test passes at
2-GiB reserve without claiming TP acceleration. Observed peak cgroup memory is
about 9.09 GiB across native cases, with no swap/OOM events. An earlier timing
attempt changed one static resident slot and is retained but rejected as an
isolated speed comparison; its correctness results do not override that gate.

Bounded HTTP checks with a 4096-MiB cache pass arithmetic, retrieval and an
actual cached-prefix follow-up, then stop the temporary authenticated loopback
server. The supported 2.7 default's 513-token full logits remain byte-identical
to the historical reference with both new controls unset. Compute Sanitizer
memcheck on the synthetic cache matrix reports zero errors and zero leaked
allocations. Python packer/installer/frontend/launcher regression selection
collects 63 cases: 54 pass and nine frontend cases are skipped (the separate real
HTTP checks above do execute the model). These checks do not establish broad
2.9 quality, actual 64K/524K 2.9 inputs, graph/speculative support, multi-session
behavior, or long-context throughput. No production defaults change.

`ds41_decode_cache_model_test --pack PACK --ids-file IDS --out DIR` runs the
bounded native check (create `DIR` first): 96 generated steps, warm-up plus three
timed rounds, exact per-position logits, reset/snapshot/warm-prefill/cancellation.
`--reference-in FILE` checks every output vector against the cache-off
`decode-logits.bin`. Use the same fixed resident sets and requested/selected
shapes for every timing case. Cache-off remains the default. Model-sized checks
require the shared service lock and no competing GPU model workload.

### Experimental LibertAIDAI 2.9-bpw compatibility

#### Bounded RAM expert reuse

**Status: implemented opt-in; native/HTTP correctness and bounded RAM-cache sanitizer checks pass.**
`DS41_EXPERIMENTAL_RAM_CACHE_MIB` requests a bounded pinned-host cache for
complete, unchanged cold expert payloads. Unset/0 disables it; the total budget
is an even integer in 0..16384 MiB, divided equally between the two logical
stages/ranks unless `DS41_EXPERIMENTAL_RAM_CACHE_SPLIT=A,B` supplies their MiB
budgets. The split must sum exactly to the total, and either stage can receive
zero. Allocation is also capped at that stage's number of nonresident experts.
It applies only to experimental 2.9 decode. The lookup order is
static GPU residency, GPU decode cache, RAM cache, then the existing file reader.
RAM hits still transfer weights to GPU for computation. A TP rank caches full
source payloads before the existing lossless shard extraction; the two ranks'
combined allocations still stay within the total budget. Prefill keeps its
separate reader/ring and does not populate or evict this cache.

Entries publish only after a complete read. LRU eviction cannot invalidate an
in-flight transfer; uploads complete before a slot is reused. Reset/snapshot
preserve immutable cached weights, not conversation state. Budget/allocation
failure refuses startup rather than silently consuming unbounded RAM or dropping
experts. No CPU MoE, weight conversion, model download, or quality-suite restart
is included. Acceptance requires synthetic hit/eviction/read-error/TP checks,
byte-identical full model logits, serial same-binary timing with matched GPU
resident sets and cache sizes, plus explicit host-memory and swap accounting.
This is a 2.9-only performance experiment, not a claim of cliecy-equivalent speed.

`DS41_EXPERIMENTAL_RESIDENT_FASTPATH=1` separately selects device-only route lookup
for layers whose entire expert set is statically resident. Default/0 keeps the
existing route-copy/fence path; other values fail closed. Partial layers retain
the protected upload/cache path. This removes host waits only where no descriptor
or payload can be evicted. It does not enable whole-model CUDA graphs. Validation
compares the same residency and all full logits with this control off/on.

An initial matched 63-token/96-step ablation rejects an equal 4-GiB total RAM
split as a performance recommendation: 28.937 versus 29.854 tok/s cache-off on
the same binary, with zero timed RAM hits. Equal 8 GiB gives 31.196 tok/s, only
92 RAM hits among 635 timed GPU uploads. Both retain exact full logits. The
first GPU's existing GPU cache already covers its repeated decode working set;
equal host allocation wastes budget there while the first stage's RAM LRU
cycles. The explicit split is tested next rather than assuming more RAM alone
establishes an improvement. These are short-fixture native rates, not HTTP,
long-context or cliecy-matched results.

Correcting the placement to the second logical stage (`0,8192`) gives 45.849
tok/s with the 4096-MiB/device GPU cache, versus 30.151 tok/s in that binary's
cache-only baseline with the resident fast path enabled in both. All 635 GPU uploads per measured 96-step round come from
RAM, with zero expert-file cache-fill bytes. Peak observed cgroup memory is
13.27 GiB, with no swap. A 12-GiB host budget does not improve this fixture and
is not preferred. These observations are from an exploratory placement run;
the final same-binary pipeline qualification is recorded separately below.

#### Asynchronous cached-route read

**Implemented opt-in; not recommended for speed.** `DS41_EXPERIMENTAL_ASYNC_CACHED_ROUTES=1` starts the
verifier's bounded route read into owned pinned memory before submitting the
independent shared-expert work. It still fences before reading the routes,
publishing any selection, changing cache ages or reusing descriptors. A miss
falls back to the existing row-by-row path; cancellation drains the outstanding
read. If a CUDA drain fails, the tier refuses reuse and retains the pinned target
until process exit. The default path remains synchronous. The option requires
experimental 2.9 layer-split verification and cached-row admission.

At 32768 allocated capacity and 6200/6800 residency, the same binary measured
112.868 / 115.339 / 113.893 / 110.512 tok/s for an off/on/on/off sequence, and
112.717 / 115.105 / 115.842 / 111.376 tok/s for an on/off/off/on sequence.
Each value is the median of seven timed 96-token responses after a warmup, using
the 63-token frozen coding prompt, projection batching, dense occupancy candidate
and confidence 0.5. The first sequence favors the option, but the reverse order
favors the control. Rate drift prevents a reliable speedup claim; leave it off.
All 56 timed responses reproduce the frozen output and the per-window draft
tokens/confidence bits. Each has 109 verified rows, 273 cached batches and no
timed uploads; enabled runs issue 273 asynchronous reads.

All 19 component tests pass. Synthetic cache tests compare synchronous and
asynchronous selections, payloads, LRU and counts through hits, misses and
evictions, and cover cancellation, invalid IDs and capture refusal. Full-leak
cache memcheck reports zero errors and leaks. Two real-weight fixtures pass 108
full-logit and rollback cases at 32768 allocation, including wrong drafts and
graph replay. The CUDA-drain failure branch was not fault-injected.

Two separately instrumented responses show why an API-duration reduction alone
is misleading: summed route-copy calls fall from 670.534 to 0.555 ms, while the
next same-thread stream fences rise from 0.208 to 692.064 ms. Those times include
previously queued GPU work and are not removable overhead or a speed estimate.
The option is not qualified for a 32K-token input, HTTP, multi-session behavior
or broad quality, and the runs are not a controlled cross-engine comparison.

The 6200/6800 Nsight trace contains 273 cached-route copies per response. Their
701.022 ms of summed API durations mostly waits for previously queued GPU work;
it is not removable host overhead. The copies themselves sum to 0.289 ms, and
copy completion to the next same-stream selection starts sums to 4.202 ms,
including overlap and instrumentation effects. Completing another static layer
would require 6912 stage-1 slots, but that request was refused by the existing
12-GiB reserve. The guard was not relaxed and no speed comparison is available.

#### Batched draft embedding handoff

`DS41_EXPERIMENTAL_DRAFT_BULK_EMBEDDING=1` prepares the five DSpark embedding
rows in one kernel and transfers the resulting contiguous buffer as one group.
It is off by default and requires experimental 2.9 layer split and DSpark. The
source scratch is bounded and allocated before residency sizing; the grouped
transfer retains peer or bounded forced-host behavior. `Engine::Draft::ms`
measures the DSpark stage, while `embedding_handoff_ms` and its counters measure
the split embedding and transfer separately.

On the 32768-capacity short coding fixture at 6200/6800 residency, an
off/on/on/off sequence measured 115.629 / 115.615 / 115.610 / 112.803 tok/s
median. The isolated embedding/handoff median fell from 2.412/2.479 ms to
0.986/0.913 ms per response; peer transfer groups, copies and host waits fell
from 195/195/390 to 39/39/39. The approximately 1.5 ms saving per response did
not produce a reliable end-to-end gain against run variation; leave it off.

All rounds match the 96-token output and the saved per-window draft tokens and
confidence bits. A forced-host run matches the same trace. The default 19
component tests and native embedding full-leak memcheck pass, including exact
batched-versus-sequential graph replay. This does not qualify a full 32K-token
input, serving, multi-session behavior or broad quality.

#### Grouped verifier handoff

`DS41_EXPERIMENTAL_VERIFY_BULK_HANDOFF=1` groups the hidden rows, HC mix,
mirrored compressed rows and index slices at the split verifier boundary. A
peer transfer waits once for the group; the forced-host path remains bounded
by the pinned staging buffer. Mirrored rows are backed up before the group is
copied, and rejected rows retain the existing rollback order. Default off;
requires experimental 2.9 layer split and verification. `Engine::Timing` counts
the verifier-boundary transfer groups, copies and host waits.

The 32768-capacity peer trial uses the original 6200/6800 resident split,
batched output projections, dense occupancy candidate and confidence 0.5. Its
off/on/on/off medians are 115.198 / 113.493 / 106.685 / 109.670 tok/s, seven
timed rounds after warm-up. For each 96-token response, the grouped path reduces
verifier-boundary groups from 295 to 39, copies from 295 to 225, and host waits
from 590 to 39; this did not produce a throughput improvement. Leave it off.

All rounds match the frozen output. At 32768 allocation, short and boundary
fixtures pass 108 exact-logit/partial-rollback cases with graph replay. All 19
default component tests and transfer full-leak memcheck pass, including peer
and forced-host transfer cases. This is not a 32768-token input or a serving or
broad-quality qualification.

#### Per-rate expert GEMV dispatch

`DS41_EXPERIMENTAL_EXPERT_RATE_KERNEL=1` uses a per-rate small-row GEMV for
eligible K10 expert jobs. It is off by default and requires the experimental
2.9 layer split. At startup, each layer and MTP stage is eligible only if every
expert's three projections have the same integer K1..K6 rate and codebook 2,
and their trellis byte lengths match that rate and the actual projection shape.
Mixed rates, noninteger rates, MCG and inconsistent descriptors retain the
existing mixed-rate kernel. The startup diagnostic reports eligible counts.
Grouped-expert execution takes precedence where its separate gate is active.

The tested backbone has K2/K3 layers and the MTP stages use K4; runtime tracing
confirmed the corresponding per-rate kernels execute. Synthetic GPU tests
compare the two paths exactly for K1..K6 at four rows and K1..K4 at one and eight
rows, with inactive routes, changing graph replays, and native FP16 and QDQ
activations. The grouped and per-rate gates also pass a component test together;
this does not qualify their combined serving performance.
Host tests reject fractional/NaN rates, mixed experts, mixed codebooks and
incorrect trellis sizes. All 19 default component tests and enabled expert
full-leak memcheck pass (zero errors and leaks). Two verifier fixtures pass
108 exact-logit and partial-rollback cases, including graph replay. A separate
96-position non-speculative run matches full logits, prefix logits and tokens
from the earlier cache-off reference byte for byte.

The fixed resident split for this trial was explicitly 6000/6800, rather than
the previous 6200/6800, to accommodate an existing GPU service. The original
allocation was refused without shrinking and the failed receipt was retained.
The following are same-binary 32768-capacity medians of seven measured rounds
after warm-up; projection batching and confidence 0.5 were enabled, and the
same resident split and cache budgets were used on both sides:

| Dense occupancy | Per-rate off A | Per-rate on A | Per-rate on B | Per-rate off B |
|---|---:|---:|---:|---:|
| Off | 105.601 | 106.217 | 106.266 | 104.261 |
| On | 107.083 | 108.245 | 108.422 | 106.973 |

The adjacent-pair gain is approximately 0.6% to 1.9%; treat this as a modest
experimental result, not a demonstrated production speedup or a comparison to
earlier runs with different resident placement. Every run reproduces the frozen
96-token response and records 109 verified rows with zero timed expert uploads.
No long-input, serving or broad-quality qualification follows from this trial.
All gates remain off by default; no production configuration was promoted.

A subsequent 32768-capacity run fit the original 6200/6800 split without
interrupting the background GPU service. With dense occupancy and projection
batching on, an off/on/on/off per-rate sequence produced 107.585 / 108.098 /
108.997 / 109.675 tok/s median. The reversed pair does not favor per-rate
dispatch, so the earlier small gain is not reliable across these trials; leave
the flag off. This run does not establish that the 524288-capacity verifier
fits the original split under the same background allocation.

#### Dense GEMV register-allocation candidate

`DS41_EXPERIMENTAL_DENSE_OCCUPANCY1=1` relaxes the K5/K6 small-row kernel's
compiler launch bound from two blocks to one. Block dimensions, trellis decode,
MMA, folding and reduction order remain unchanged. This is a register-allocation
hint, not a runtime guarantee of one resident block. Default off; read once per
process. Combining it with `DS41_EXPERIMENTAL_REG56=1` is rejected. Expert-job
dispatch and other bit rates retain their existing kernels.

On SM120, compiled K5/K6 register counts rise from 64 to 117 and stack usage
drops from 32 to 8 bytes. Isolated synthetic geometry tests on both devices
show improvement on the tested dense shapes with byte-identical raw FP32
outputs. The alternative CFG1 geometry changes K-splitting/fold cadence, changes
raw output bits, and is shape-dependent in speed; it was not integrated.

With projection batching enabled at 32768 capacity, an off/on/on/off sequence
measures medians 109.318 / 113.097 / 107.852 / 105.487 tok/s. Local pairs favor
the relaxed bound by 3.46% and 2.24%, but absolute speed drifts between runs.
This is a modest candidate gain, not evidence of surpassing earlier 117.816
tok/s peaks. All four runs reproduce the frozen 96-token coding response;
19 component tests, 108 exact full-logit/rollback cases and native-only full-leak
memcheck pass. The verifier fixtures retain their 524288 capacity. Production
and broad quality evaluation are unchanged.

The batching-enabled node trace assigns 35.3% of summed kernel time to expert
GEMV jobs, 20.5% to K5 dense GEMV and 6.0% to K6 dense GEMV. Across its 959.8 ms
kernel span, device kernel intervals do not overlap: device 0 is active for
307.8 ms, device 1 for 541.9 ms, and neither has a kernel active for 110.1 ms.
These include spin-waits and exclude copy engines; they are not useful-compute
utilization or a complete causal critical-path analysis. Serial layer staging
and expert execution remain substantial targets after dense batching.

#### Batched attention output projections

Native decode qualified on the short coding fixture. `DS41_EXPERIMENTAL_BATCH_PROJECTIONS=1` batches the
verifier's equal-shape/rate EXL3 attention output slices into three launches:
input transform, multi-projection small-row GEMV, and output transform. The
per-projection Hadamards, MMA folds, cross-warp reductions and BF16 output
rounding retain the existing arithmetic. This combines independent projections;
it does not combine their sums or reconstruct weights.

Default off. Requires experimental 2.9 layer split. Descriptors travel by value
in kernel arguments, so graph replay needs no temporary host pointer or upload.
The bounded workspace is sized before execution and owned by the verifier.
The API accepts at most 16 unsharded projections and 1..8 rows, validates shape,
rate, strides and workspace before launching, and rejects incompatible batches.
Grouped expert execution remains a separate switch.

The paired 32768-capacity trial measures 115.481 tok/s enabled versus 101.474
disabled (+13.8%), medians of seven timed rounds after one warm-up. Best rounds
are 117.816 and 101.731 tok/s. All rounds match the frozen 96-token output from
the 63-token prompt. Confidence threshold is 0.5, cached-row admission is on,
and grouped experts, register decoding and tail graphs are off.

All 19 component tests and 108 exact full-logit/rollback cases pass. Native-only
full-leak memcheck reports zero errors and leaks; shared-memory racecheck also
passes. The whole dense test's full-leak run reports existing process-lifetime
cuBLAS handles from reconstruction; that failed receipt is retained. New test
fixtures explicitly order default-stream uploads and nonblocking-stream work.
Rollback fixtures retain 524288 capacity. This result qualifies short native
decode, not a 32K input, HTTP serving or broad quality. The option remains off
by default and production configuration is unchanged. It has not reached Maya's
historical 153.78 tok/s peak.

A subsequent same-binary 6200/6800 test with dense occupancy on and per-rate
experts off produced projection batching off/on/on/off medians of 95.822 /
111.421 / 109.876 / 96.926 tok/s at 32768 capacity. Each run preserved all 96
reference output tokens. This confirms a substantial batching benefit for this
short fixture under the later conditions, not a long-input or serving claim.

#### Grouped expert verification candidate

Implemented opt-in; not recommended for speed. `DS41_EXPERIMENTAL_GROUPED_EXPERTS=1` groups up to
eight expert uses with identical trellis pointers, dimensions, rate and codebook.
The bounded device plan is rebuilt before each gate/up or down GEMV, including
on graph replay. Only multi-row K10 calls with at most six experts per row use
the new path. Other calls retain the existing dispatch. Default off, read once
per process. This does not change weights or cache admission.

Each group decodes the weights once for its rows. Input and output addresses
remain per-use: Hadamards, weighted hidden rounding, MMA folding and expert
addition order retain the existing arithmetic. Groups larger than eight split
into separate chunks; inactive entries are skipped. Scratch storage is bounded
by the existing K10 workspace and reclaimed with it. The grouped mixed-rate
kernel shares a dynamic reduction buffer across rate branches.

The initial 32K coding trial is slower: 100.381 tok/s enabled versus 102.259
disabled, medians of seven timed rounds after one warm-up. Best rounds are
105.638 and 102.716 respectively; the enabled runs vary more, so the best round
does not establish an improvement. Every round reproduces the frozen 96-token
output. All 19 component tests, 108 full-logit/rollback cases, and full-leak
Compute Sanitizer checks pass (zero errors and leaks). Component checks include
grouped versus single-token arithmetic, MCG, mixed rates, inactive routes,
groups larger than eight, and changing routes under graph replay. Rollback
fixtures use their existing 524288 allocation; decode timing uses 32768.

This first implementation retains the maximum-sized grid and early-returns
nonleader jobs; it also adds grouping kernels and a larger reduction buffer.
Those are possible overhead sources, not yet measured explanations. Leave the
switch off. Explicit expert reuse alone has not closed the Maya performance gap.

#### K5/K6 register decoder candidate

Implemented opt-in; not recommended for speed. `DS41_EXPERIMENTAL_REG56=1` uses warp shuffles for
K5/K6 packed-word reads in the small-row dense GEMV path. The existing dq4 bit
indices, funnel shifts, codebook, MMA, folding and reduction remain unchanged.
The option is read once per process; default off. This targets the node-level
32K Nsight trace, where K5 dense GEMV accounts for 26.9% of summed kernel time.
Expert jobs and other rates retain the existing dispatch. Require component
equality, full-logit/rollback checks and same-binary timing before recommending.

All 19 component tests, eight-round exact token replay and 108 byte-exact
verifier/rollback cases pass. Compute Sanitizer reports zero memory-access
errors for the dense-linear component (leak scanning disabled). The paired
32K run measures 89.818 tok/s enabled versus 89.295 disabled: only 0.6%, with
both below earlier sessions' absolute rates. This does not establish a useful
speedup; leave the flag off. The unmodified-kernel node-level trace attributes
31.3% of summed GPU kernel time to expert GEMV jobs, 26.9% to K5 dense GEMV and
5.4% to K6 dense GEMV. These percentages include overlapping streams and are
not wall-time fractions or predictions of achievable speedup. No 154-tok/s
result was reached by the 32K experiments.

#### 32K verifier tail-graph candidate

Implemented but not recommended for speed. `DS41_EXPERIMENTAL_VERIFY_LAYER_GRAPHS=1` captures
attention through routing for verifier layers beyond the static resident prefix.
Expert admission, uploads and expert evaluation retain their existing eager
path. The graph keeps the same kernels and arithmetic; replay restores the host
KV/index/Engram cursor as well as executing device operations. Keys include the
layer, row count, parity and both index-capacity buckets. At most 256 graphs per
stage are retained; new shapes fall back to eager execution at that bound.
Graphs are destroyed with the verifier workspace. Default off; requires
experimental 2.9 layer split and static graphs. Measure at 32768 capacity against
the unchanged draft-confidence path; require exact continuation and rollback
checks before recommending this candidate.

The matched 32K coding trial measures 101.390 tok/s enabled versus 101.361
disabled (seven timed rounds after warm-up), with all 96 tokens exact. All 19
component tests and 108 full-logit/rollback cases pass; each verifier fixture
records 369 graph replays. The observed difference is negligible, so leave this
gate off. The separate 32K confidence sweep measured 96.704 at cumulative 0.5
with four inputs, 78.635 at 0.2 with six, 96.224 at 0.5 with six, and 97.020 at
0.7 with six. None establishes an advantage over Maya's observed warm peak.

#### Decode locality and confidence candidates

Both controls default off. On experimental 2.9 layer
split, `DS41_EXPERIMENTAL_DRAFT_CUM=0.5` returns only the longest DSpark draft
prefix whose cumulative confidence is at least 0.5. Zero disables trimming.
The target still verifies every returned draft. All native/serving draft callers
share this policy; no pretrained data or target arithmetic changes.
`DS41_EXPERIMENTAL_CACHE_FREQUENCY=1` selects bounded GPU-cache victims by
observed route frequency, with LRU ties. A new route no more frequent than the
coldest unprotected cached owner uses existing staging instead of evicting it.
Counts live for the engine lifetime and survive snapshot/reset, just like cache
bytes. Static residency, budgets and the pinned RAM cache are unchanged.
Acceptance requires exact replay and verifier checks plus matched timing and
actual uploaded-expert/verified-row counters. Neither candidate implies a
154-tok/s result.

The eight-round coding fixture (one warm-up, seven measured repeats, 96 exact
greedy outputs each) measures 76.233 tok/s disabled and **102.953 tok/s with
cumulative confidence 0.5**, a 35.1% same-binary gain. Static residency remains
6200/6800, capacity 524288, GPU cache 8192 MiB/device and RAM split 0/12288 MiB.
Verified rows fall from 143 to 109 per response; timed expert uploads fall from
319 to zero. The confidence-only run is the preferred candidate. Frequency-only
measures 90.562 tok/s; combined measures 102.296, and combined six-input
verification measures 101.466. Neither frequency nor wider windows improve on
confidence-only here. All eight rounds in each configuration reproduce the
frozen reference; all 19 component tests pass.

The attempted 19/21 layer balance is unsupported by the shared KV/index layout.
The supported 20/20 split cannot fit the fixed resident set with these budgets;
it was refused before generation. Neither failed attempt changed the fixed
placement contract. At 103 tok/s with zero warm expert uploads, remaining decode
cost is verifier/drafter execution and scheduling, not cache churn on this
fixture. The 154-tok/s gap remains open.

Final checks pass 108 byte-exact verifier/rollback cases across short and
512-boundary inputs, both parities and wrong drafts; cache-stream memcheck with
full leak checking reports zero errors. Authenticated HTTP with both controls
and prefill reuse passes arithmetic, retrieval, cached follow-up and repeated
coding responses; the temporary server is stopped. A separate confidence-only
32768-capacity trial measures 95.600 tok/s with exact output, so reducing reserved
capacity did not close the gap in this trial. Broad evaluation remains paused.
Use `DS41_EXPERIMENTAL_DRAFT_CUM=0.5` with cached rows and the existing four-input
verifier for the preferred measured candidate; leave frequency admission off.

#### Maya-inspired bounded prefill reuse

**Implemented opt-in; native numerical/lifecycle checks pass.**
`DS41_EXPERIMENTAL_PREFILL_CACHE=1` admits prefill reuse of the existing GPU/RAM
expert cache budgets. Before readers start, empty GPU cache slots are filled in
stage/layer/expert order, then empty pinned RAM slots are filled with remaining
experts. Existing valid decode entries are retained. Priming never evicts a valid
entry, exceeds a configured budget or changes static residency. Those tables are
read-only until the expert stream is joined; decode may use their bytes afterward.
Cached GPU experts stay in their original **cold-expert reduction positions**;
only their redundant SSD/ring copy is removed. Pinned RAM hits use the ordinary
stream and consumer fences. Uncached bytes still use the bounded SSD ring.
The default is off. It requires experimental 2.9 layer split and separate
reconstructed prefill, not TP or packed MMA. Acceptance requires byte-identical
full logits/continuation, warm/reset/cancellation tests, source counters, same
static residency/cache budgets/capacity and separately measured priming overhead.

Source trace: Maya `4615f346d9311dd1ce9f5738ab04446f023a6e42`,
`src/ds4/ds4_model.cu:652–699` owns chunk/host expert reuse;
`:4146–4292` runs resident/streamed sets in bounded pieces. Its CUDA EXL3 prompt
path calls packed `set_gate_up`/`set_down` (`src/kernels/cuda/exl3.cu:2308–2457`),
instead of reconstructing every full matrix. That kernel path remains a separate
numerical question here: Strata retains its established routing-weight/cast and
reduction order, and does not enable the previously rejected packed candidate.

The matched 4097-token test measures **394.268 tok/s enabled versus 244.995
disabled** in the same binary, a **60.9%** improvement (preserved build: 246.629).
Both use static residency 6200/6800, 524288 capacity, 512/256 shapes,
8192-MiB/device GPU cache, pinned RAM split `0,12288` MiB, an 800% CPU cap and
a 24-GiB/no-swap host guard. One warm-up precedes two timed reset rounds; the
reported duration is the conservative upper-middle time for those two rounds.
Each round starts with empty KV/history: **zero prompt tokens are reused**.
Only immutable expert bytes survive reset. Full logits are byte-identical.

The first call includes 3284 ms priming and completes in 13.774 s
(297.436 tok/s); later calls prime no new payload. Each call reuses 5565 cold
expert groups directly from the GPU cache and serves 4515 stream jobs from RAM;
only 492 stream jobs still read SSD. The disabled path streams 10572 jobs from
SSD. Priming fills 644 existing GPU slots per device and 967 existing RAM slots
on stage 1, without an additional allocation or a static-expert move.
The 96-position full-logit continuation, snapshots, warm follow-up, cancellation
recovery and four bulk/HC combinations also pass. All 19 component CTests pass.
No pretrained byte, cast, route, shape or expert-addition order changes. These
measurements are not a matched Maya/Strata speed comparison or a long-context
quality claim. Installed serving configuration and defaults remain unchanged.
The cache/stream component passes Compute Sanitizer with **full leak scanning**
and zero errors. With both new gates enabled and the four-input DSpark verifier,
authenticated loopback HTTP passes arithmetic, retrieval, cached follow-up and
two identical 96-token coding responses. The temporary server stops afterward.

To opt in alongside the already-qualified 2.9 cache/DSpark controls:

```sh
export DS41_EXPERIMENTAL_PREFILL_CACHE=1
export DS41_EXPERIMENTAL_CACHED_ROWS=1
# Keep the GPU/RAM budgets above, verification at four inputs,
# and packed prefill, expert-prefill graphs and reconstructed waves off.
```

#### Maya-inspired cache-hit row batches

**Implemented opt-in; exact native verifier checks pass.**
`DS41_EXPERIMENTAL_CACHED_ROWS=1` replaces per-row host route/publication fences
with one bounded route read for a fully cached verification layer. All selected
experts must already be static residents or members of the existing GPU LRU.
One by-value selection kernel publishes the whole window; the existing K10
kernel evaluates its rows without changing arithmetic or expert reduction order.
A miss selects the original per-row staging path. No CPU expert, residency
migration, extra cache allocation or changed pretrained byte is introduced.
The default is off, and the option requires experimental 2.9 layer split.

The transfer comes from the observed warmed expert locality in Maya
`4615f346d9311dd1ce9f5738ab04446f023a6e42`: `src/ds4/ds4_model.cu`
`run_set`/`run_vram` batches resident expert sets, while its adaptive decode places
frequently routed experts on GPU. Our prior trial's adaptive configurations
change one greedy replay; that mechanism is not copied. Our cache already keeps
immutable GPU bytes, so removing repeated host scheduling is the first bounded
experiment. Acceptance requires exact full-logit/rollback tests for every
1..6-row window, mixed static/cached and missing selections, safe queued consumers,
actual batch counters, sanitizer checks and matched 6200/6800, 524288-capacity
sampling loops. Faster decode is not assumed from source inspection.

The first hit-only batch follow-up passes 108 byte-exact native verifier cases,
including wrong drafts, every partial commit, both parities and the 512-position
indexer boundary. A 96-output matched sampling loop measures 76.458 tok/s enabled
versus 74.844 disabled in the same binary (preserved build: 75.998). Each timed
four-input round performs 131 actual hit-only batch admissions. This is a small
short-run difference, not an explanation of Maya's 154-tok/s exploratory result
or a universal speed recommendation. Six-input windows still lose on this fixture
(65.683 tok/s). Static residency, cache limits, output tokens and defaults stay
unchanged; full-model quality evaluation remains paused.

#### Wider resident verification candidate

**Native numerics qualified; wider windows are not preferred for speed.**
`DS41_EXPERIMENTAL_VERIFY_ROWS=6` raises the
experimental 2.9 layer-split verifier's bound to seed plus five drafts. The
default stays four inputs. Values 4..6 are accepted; larger windows require
`DS41_EXPERIMENTAL_VERIFY=1`. CPU, TP and multi-session row bounds stay unchanged.
`Engine::verify_capacity()` reports the actual bound used by serving and tests.
Verifier, rollback tails, Engram staging and DSpark main-hidden storage use that
bound, not the unrelated CPU batch limit. Acceptance requires full-logit equality
for every row/partial commit, wrong drafts, both compressor parities and an indexer
capacity boundary, followed by a matched k=3 versus k=5 sampling-loop measurement.
An invalid window is rejected before tentative state or reader jobs change.

The native six-input suite passes 108 byte-exact cases across a short prompt and
positions 510/511: all 1..6-row windows, every partial commit, non-greedy tentative
inputs, both parities, snapshot continuation and the 512-position indexer boundary.
There are 82 actual verifier-prefix replays per fixture. The same-binary coding
comparison keeps the prior 63-token input, 96 greedy outputs, residency 6200/6800,
524288 capacity, 8192-MiB/device target cache, host split `0,12288`, requested
4096-MiB drafter cache and 800% CPU cap. One warm-up precedes three measured rounds.

| Proposed drafts | Verify inputs | Median native output rate | Emitted inputs/window |
| --- | ---: | ---: | ---: |
| 3 | 4 | 77.534 tok/s | 2.595 |
| 5 | 6 | 69.142 tok/s | 2.909 |

All rounds match the same non-speculative 96-token reference. Six-input rounds
actually exercise six rows, with 66 prefix replays and 1485/1485 drafter-cache hits.
Higher acceptance does not pay for the extra verification: timed verify/commit
work rises from about 1.14 s to 1.30 s per 96 outputs. Do not equate this result
with cliecy's workload or claim a k=5 speedup. Keep four inputs as the default;
the wider option is for explicit experiments. `ds41_generate` also needs
`--spec-max 6` to use the larger bound; serving uses the configured capacity.
All 19 component CTests pass in the six-input build. Authenticated loopback
serving with the six-input control passes arithmetic, retrieval, cached follow-up
and two identical 96-token coding responses, with zero swap/OOM events. The
temporary server stops afterward. These are functional checks, not a paired
HTTP speed or broad-quality claim. Installed serving configuration is unchanged.

#### Packed hot/cold prefill candidate

**Exact expert graphs: numerically qualified, no measured speedup.**
`DS41_EXPERIMENTAL_PREFILL_GRAPHS=1` replays the existing reconstructed expert
pipeline in bounded per-row-count CUDA graphs. It retains the reference's exact
cuBLAS shapes, three reconstructions, activation casts and expert scatter order.
Graphs share one ordinary reconstruction workspace; expert descriptors and row
offsets are passed by value through an updated preparation node, not a reused
asynchronous host staging buffer. A dedicated nonblocking capture stream permits
replay on prefill's default stream. At most 128 graphs per stage are retained;
unseen shapes beyond that bound and oversized row tiles use the eager reference.
The owner destroys graphs before prefill scratch or expert-ring storage is freed.
Default/0 remains eager. It refuses TP, packed MMA and reconstructed wave options.
Acceptance requires exact full logits, actual replay counters, changing expert/
offset tests without host fences, cancellation/warm/reset lifetime checks and
matched 512/256, 6200/6800 timing. The 4097-token paired run measures 241.711
tok/s on versus 241.940 tok/s off in the same binary (preserved earlier build:
243.494 tok/s). Each pass records 238 captures and 66,571 actual expert-graph
replays across both stages. Full logits are byte-identical, as are 96 subsequent
greedy positions, warm-prefill and cancellation recovery. All 19 component tests
pass, including queued descriptor/offset updates without host fences and the
128-shape bound. This does not qualify as a prefill improvement; keep it off.
The final build also passes the four-way bulk/HC lifecycle check with graph
prefill enabled: snapshots, odd-position follow-up, warm replay, reset and
cancellation recovery. Its expert component has zero Compute Sanitizer memory
access errors. Leak scanning is disabled for that check because of the known
process-lifetime cuBLAS retention; it is not a leak-clean claim.

**Exact batched alternative: numerically qualified, rejected for speed.**
`DS41_EXPERIMENTAL_PREFILL_BATCH_GROUPS=0..16` (default 0) batches reconstruction
and Hadamard/activation launches for bounded expert waves on experimental 2.9
layer-split prefill. It retains the reference's FP16 reconstructions, per-expert
cuBLAS GEMM shapes, routing weights and ordered scatter. It does not requantize
weights or select the rejected packed MMA kernel. Scratch is bounded by the
explicit wave size and the pass's route count, and is released after prefill.
The option refuses TP, other profiles and simultaneous packed-prefill selection.
Acceptance requires exact component and full-model logits, default-off replay,
peer/host handoff, warm/reset/cancellation checks and same-binary, fixed-residency
timing. The 4097-token experiment is byte-identical, including 96 subsequent
greedy positions, but slower: 201.254/201.057 tok/s for waves 8/16 versus
249.759 tok/s off. A fused reconstruction launch also remains slower at
211.069/199.120 tok/s for waves 1/2 versus 244.340 tok/s off. These opt-in
experimental paths are not recommended settings and do not count as a prefill
speedup. Source/binaries and all receipts remain in ignored validation storage.

**Status: rejected by the initial full-model numerical gate; not recommended.**
`DS41_EXPERIMENTAL_PACKED_PREFILL=1` selects grouped packed expert GEMM for the
2.9 layer-split path. It groups active static experts together, then processes
cold experts through the existing bounded ring in ordered waves. It keeps the
checkpoint bytes, FP16 activation semantics and hot-then-cold expert reduction
order. Tile accumulation differs from reconstructed cuBLAS GEMM, so this candidate
requires explicit full-logit error and greedy-output checks, not a claim of
byte-identical prefill. Decode kernels and defaults stay unchanged. The same
512/256 shapes, residency and capacity are required for isolated timing; larger
chunks and TP/speculation are separate experiments.
The packed wave's grids use its actual assignment count rather than launching
the whole chunk's maximum route capacity for every small cold wave.
The initial 4097-token trial retains the final greedy token but has relative
logit L2 0.088699 and KL 0.030021 against reconstructed prefill, exceeding the
accepted 0.01/0.002 gates. The source/binary and failed receipt are preserved.
Small synthetic component agreement does not override that full-model failure.
An independent 1024/256 reconstructed-prefill shape trial is also rejected:
relative logit L2 0.100669 and KL 0.094899 versus 512/256. The usable candidate
keeps the original 512/256 shapes. Neither faster trial is promoted as a
qualified prefill improvement.

#### Experimental multi-token verification

**Status: opt-in native and HTTP checks pass; not promoted to production.**
`DS41_EXPERIMENTAL_VERIFY=1` admits the existing 1..4-token verifier on the 2.9
layer-split backend, with the native FP16 router and bounded GPU/RAM expert
uploads. Partial-residency layers prepare and consume one row before reusing
staging slots; fully resident layers use the existing multi-row kernel. The
streamed suffix stays eager (`DS41_GRAPH=0`); tentative cache writes roll back
through the existing commit protocol. With `DS41_EXPERIMENTAL_STATIC_GRAPHS=1`,
the verifier also captures each fully resident prefix after an eager warm-up,
keyed by row count, parity and indexer capacity. Its graph owns no conversation
state and restores the prefix's source pointers before executing the suffix.
Reset/snapshot reuse the workspace; prefill destroys it and its captured graphs
before releasing buffers. Actual capture/replay counts are exposed in timing.
Each of two native fixtures passes 22 reference cases with 38 actual prefix graph replays, including
each 1..4-row window, every partial commit, wrong drafts, snapshot continuation
and both parities. The second fixture starts at positions 510/511 and crosses
the 512-position indexer-capacity boundary. Target logits match ordinary decode
byte-for-byte in all 44 cases.
All 19 component CTests pass. The native-FP16 DSpark component is clean under
Compute Sanitizer with full leak checking. The reconstructed-wave component has
zero memcheck errors with leak scanning disabled; the known process-lifetime
cuBLAS retention described below is not claimed fixed or leak-clean.
Default/unset keeps verification rejected. TP and
DSpark are not enabled by this control alone. Acceptance requires every accepted-row
logit to match ordinary decode, partial acceptance/rollback, both position
parities, snapshots, recovery and a separately measured speculation pipeline.

`DS41_EXPERIMENTAL_DSPARK=1` additionally admits `EngineOptions::dspark` with
that verifier. The published MTP router, Markov head and confidence head use
their stored FP16 weights directly, not rounded BF16 replacements. Native-FP16
component tests include values distinguishable from BF16. The candidate pack
adds only the already-downloaded, unchanged MTP expert bytes; the backbone pack
remains untouched. Initial verification windows keep the existing four-input
limit (seed plus at most three proposed tokens). This is not cliecy's k=5
verification configuration. The first eager-verifier pipeline preserves all 96
greedy tokens and accepts 2.595 inputs/window, but at 31.428 tok/s is slower than
its 50.312 tok/s no-speculation control. Prefix-graph/cache ablations are
measured separately; functional support alone is not a speed recommendation.
Default-off behavior is unchanged.

#### Measured 2.9 DSpark follow-up

The 2026-10-25 matched experiment enables the native FP16 drafter, bounded expert
verification and verifier-prefix graphs. Same binary, frozen 63-token coding
input, token-wise untimed prompt setup, 96 generated tokens without EOS, one
snapshot/warm-up and three timed repeats, 524288 capacity, static residency
6200/6800, 8192 MiB/device target GPU cache, 12288 MiB host cache on logical stage
1 (`0,12288`), and an 800% CPU cap are common to both modes. No GPU-sized jobs or
compilation overlap these final runs. GPU clocks are not fixed.

| Mode | Median native output rate |
| --- | ---: |
| No speculation | 55.321 tok/s |
| DSpark, up to three proposed inputs per window | 74.274 tok/s |

This is a **34.3%** same-budget gain. It is a native greedy sampling-loop wall
measurement including draft, verification, commit and per-window output handling;
it is not HTTP speed, nor the earlier timing that sums isolated `Engine::step`
calls. All 96 greedy tokens match the no-speculation reference byte-for-byte in
every round. There are 37 windows, 2.595 accepted inputs/window, 74 actual
verifier-prefix replays, zero captures and 1665/1665 drafter expert-cache hits in
each timed round. The drafter's requested GPU-cache ceiling is 4096 MiB and is
clipped to available memory while retaining the caller's reserve. The 8-GiB host
variant separately passes at 72.698 tok/s; these short runs do not establish a
universal best cache split or speed on other traffic.

Observed peak native cgroup memory is 22.970 GiB under a 24-GiB, zero-swap guard;
maximum observed per-device GPU memory is 95186 MiB. Only the 6.35-GiB MTP expert
payload is added in a **new** candidate pack: all 4608 components are compared
byte-for-byte with the already-downloaded 2.9 source. Existing backbone files
are reused unchanged. There are no downloads, quantization or changed logits
used to obtain these results.

In addition to the qualified layer-split settings, the new controls are:

```sh
export DS41_EXPERIMENTAL_RAM_CACHE_MIB=12288
export DS41_EXPERIMENTAL_RAM_CACHE_SPLIT=0,12288
export DS41_EXPERIMENTAL_VERIFY=1
export DS41_EXPERIMENTAL_DSPARK=1
# Use the separate MTP-enabled candidate pack, --spec dspark,
# --draft-cache-mib 4096, and the original 512/256 prefill shapes.
```

The backbone-only pack is insufficient: it lacks `mtp_experts.bin` and its
index. Never append to the original pack just to enable a setting. The local
validation receipt identifies the separate candidate pack and its source-byte
verification. Default controls and installed serving configuration are unchanged.

Authenticated loopback HTTP passes arithmetic, 1308-token retrieval, a 1653-token
follow-up reusing 1314 cached tokens, and two identical 96-token coding responses
(including a warm-prefix repeat). Observed HTTP cgroup peak is 22.866 GiB with
zero swap under the explicit 28-GiB guard. Both the temporary server and engine
stop afterward. These are functional HTTP checks, not a paired HTTP speed claim.

This closes part of the decode gap, **not** the prefill or 1M-context work. The
packed-MMA candidate still fails its numerical gate; the exact reconstructed
wave alternatives are slower and remain off. The later six-input follow-up above
qualifies wider k=5 numerics but finds it slower on the matched coding fixture.
Efficient TP sharding, actual 1M 2.9 input and broad-quality comparison remain unqualified.
The quality suite remains stopped at 30 saved answers.

#### Static-prefix CUDA graphs

**Status: implemented opt-in; native/HTTP and peer/host-handoff E2E checks pass.**
`DS41_EXPERIMENTAL_STATIC_GRAPHS=1` captures only each stage's fully resident
prefix, including its parameter/embedding/Engram setup. It requires experimental
2.9 layer split and `DS41_EXPERIMENTAL_RESIDENT_FASTPATH=1`; streamed layers stay
eager with the existing lifetime fences. Two eager steps prime kernel/library
state before capture (the native runner explicitly uses positions 0 and 1).
Graph keys retain the existing parity/indexer-capacity
buckets; CPU KV-source pointers are restored to the captured prefix's exit
state before the eager tail. Reset/snapshot/warm prefill reuse immutable graph
bindings, not conversation state. The accepted gate is byte-identical logits at
every generated position, with actual replay counters, cancellation/snapshot
checks and matched-residency timing. This is not whole-model graph or DSpark
support.
The GPU cache ceiling is extended to 8192 MiB for the next matched experiment:
the measured 4096-MiB layout leaves enough GPU headroom to test more reuse
without changing static residency. Allocation still requires 512 MiB free after
the cache, and prefill must retain its measured shapes; this is not permission
to silently shrink resident sets or consume an unbounded budget.

#### Qualified 2.9 decode pipeline

The final local source passes matched native checks on 2026-10-25. It combines
an 8192-MiB/device GPU decode cache, 8192 MiB of pinned RAM on logical stage 1,
device-only routing for fully resident layers, and static-prefix graph replay.
The 1017 2.9-bpw pretrained bytes and arithmetic kernels are unchanged. No CPU
experts, dropped experts, DSpark, whole-model graphs, or production promotion
are involved. The paused quality evaluation is not resumed.

Both modes use the same final binary, frozen 63-token coding input, 96 greedy
steps without EOS, capacity 524288, fixed static resident sets 6200/6800 and
requested prefill 512/256 (actual 63/63 for this short input). Two unmeasured
steps initialize both parity paths, then one warm-up round precedes three timed
rounds. Timing sums wall intervals around `Engine::step`, excluding logit
validation/serialization. The baseline has a 4096-MiB/device GPU cache, no RAM
cache, the resident fast path on and prefix graphs off.

| CPU quota cap | Baseline native decode | Combined candidate | Ratio |
| --- | ---: | ---: | ---: |
| 200% | 29.499 tok/s | 51.472 tok/s | 1.74x |
| 800% | 30.904 tok/s | 54.863 tok/s | 1.78x |

These are separate paired runs, not proof that raising the CPU cap causes the
difference: neither short run records cgroup CPU throttling. Clocks are not
pinned. Every generated token and all 129280 logits at every generated position
are byte-identical to the frozen cache-only reference, across warm resets and
all measured rounds. Snapshots, odd-position warm prefill, cancellation and
recovery also pass. Candidate rounds record 192 actual prefix replays, zero new
captures and zero expert uploads per 96 steps. Immutable weight caches remain
warm across conversation reset; startup/warm-up is not included in these rates.

A separate actual 4097-token input retains exact reference logits/tokens and
measures 58.456 native decode tok/s over 96 steps, again with three timed rounds,
zero expert uploads and 192 graph replays per round. It does not establish
sustained long-context throughput or 1M execution. Observed host cgroup peak is
19.60 GiB under a 24-GiB/no-swap guard; observed GPU peaks are at most 94259 MiB
per device on this fixture. The RAM cache itself is at most 8192 MiB total.

The explicitly tested layer-split controls are:

```sh
export DS41_EXPERIMENTAL_PROFILE=libertaidai-29
export DS41_GRAPH=0 DS41_PF_GROUPED=0
export DS41_PF_BULK_HANDOFF=1 DS41_PF_HC_BATCHED=1
export DS41_EXPERIMENTAL_DECODE_CACHE_MIB=8192
export DS41_EXPERIMENTAL_RAM_CACHE_MIB=8192
export DS41_EXPERIMENTAL_RAM_CACHE_SPLIT=0,8192
export DS41_EXPERIMENTAL_RESIDENT_FASTPATH=1
export DS41_EXPERIMENTAL_STATIC_GRAPHS=1
export DS41_EXPERIMENTAL_PACKED_PREFILL=0
```

Keep requested prefill at 512/256. The fixed resident-count control is for
isolated measurement, not a production default. If the GPU budget does not fit,
reduce the cache explicitly; initialization refuses rather than shrinking the
matched static set. A 4096-MiB/device GPU cache plus the correctly placed 8-GiB
RAM cache is the separately measured lower-VRAM alternative above.

Prefix graphs are not universally faster: the 4096-MiB GPU-cache-only trial
passes exactness but measures only 26.355 tok/s, slower than its eager baseline.
Graph replay is selected here only with the larger cache after measuring the
combined pipeline. Reconstructed prefill remains around 244 tok/s on the 4097-
token fixture; the rejected packed/1024-row trials are not counted as progress.
Cliecy's speculative 89–112 tok/s, approximately 5K prefill, actual 1M context,
multi-session behavior and broad quality equivalence remain unqualified.

All 18 component CTests pass, including strict budgets/splits, unowned-layer
refusal, RAM/GPU hits and eviction, duplicate/protected routes, short reads and
graph-capturable static route lookup. Native 2.9 E2E passes all four bulk-handoff/
batched-HC combinations with both peer transfer and forced host fallback, exact
logits, snapshots, cancellation and odd-position 751-token follow-ups. A separate
512-MiB total RAM/128-MiB-per-device GPU-cache TP2 retrieval check preserves the
historical per-token/final logits; it does not enable static-prefix graphs in TP.

Authenticated loopback HTTP checks pass arithmetic, 2588-token retrieval, a
3577-token follow-up reusing 2594 prefix tokens, and arithmetic after reuse. A
separate 36-token coding prompt emits 128 tokens per request: one warm-up plus
three timed repeats produce identical greedy text. Warm median end-to-end
throughput is 57.376 completion tok/s, including the request's first-content
latency of about 0.127 s and reusing 35 prompt tokens. The SSE content-interval
estimate is 60.449 tok/s; this is not an exact native token-boundary measurement.
The first cold coding request is only 28.611 completion tok/s end-to-end. Do not
present the warm replay result as cold/new-session or cliecy-matched throughput.
HTTP host cgroup peak is 21.75 GiB under the 24-GiB/no-swap guard. Temporary
servers stop after checks; no installed launcher or model selection changes.

Compute Sanitizer reports zero errors and zero leaked allocations for the
synthetic RAM/GPU-cache matrix. The separate prefill component reports three
process-exit allocations totaling 67240960 bytes, all from `cublasCreate` in
K12's intentionally process-lifetime `BlasRegistry`. The identical report
reproduces in the untouched pre-task binary. This existing retention is recorded
as a sanitizer finding, not concealed or called leak-clean; the rejected packed
candidate's component checks do not override its full-model numerical failure.
The stopped quality suite still has exactly 30 saved answers; no broad-quality
or production qualification is claimed.

#### Compatibility baseline

**Status: implemented experimental compatibility, not serving-qualified.** This
is not a replacement deployment target. The 2.7-bpw executable, model selection
and restart configuration remain unchanged; that service is stopped at the user's
request during these tests. The experiment uses the already downloaded
`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` revision
`c178a01284fc5c0e859d8668cffaac81ca8f4042` and the same verified 1017 Engram
shards. No pretrained weights are requantized. Pack files preserve their source
tensor bytes; CPU RAM has no model-sized expert cache.

The explicit `DS41_EXPERIMENTAL_PROFILE=libertaidai-29` gate enables K8 indexer
projections, EXL3 compressors, native FP16 routing/index weights and FP32 Engram
gating. `pack.py --profile libertaidai-29` records that separate pack profile.
The gate and pack marker must agree. Ordinary 2.7-bpw behavior is unchanged.

Because both current layer splits fail actual weight-only CUDA allocations,
the experiment retains a bounded static subset of backbone experts in VRAM and
streams misses through small GPU staging buffers. All selected experts still
execute on the GPU; none are skipped or evaluated by a CPU MoE implementation.
Eager decode and reconstructed-GEMM layer-split prefill are the first paths.
Experimental TP2 shards the stored expert and dense bytes across ranks, with
replicated small compressor/index-key projections and the same bounded staging.
Compressed KV/index caches retain the existing single-owner, peer-readable TP
layout. Expert descriptor and route-selection publication is explicitly ordered
on the nonblocking consumer stream, with a completion fence before stack-owned
host staging expires. Default-stream pageable-host copies are not a substitute
for that dependency.
Its initial prefill uses token-by-token TP steps, not speculative verification
or batched GEMM. CUDA graphs, grouped prefill, verification, batch slots and DSpark
are rejected rather than silently changing their contracts. The experimental
cache keeps at least 12 GiB per GPU free for layer split, or 2 GiB for token-wise
TP2, before selected-expert staging and prefill scratch. A larger configured
reserve is honored. Decode stages at most six missing experts per rank, and layer
prefill uses a 16-slot device ring and eight host buffers. Configured capacity remains 524288 in model
qualification; the initial real inputs are short, not a new actual-524K test.

Acceptance requires pack byte preservation and malformed-layout rejection,
K8 CPU-reference and replay checks, native FP16/FP32 component checks, bounded
streaming tests, then isolated real-weight arithmetic/retrieval, prefill versus
sequential predictive comparison, snapshot/reset/continuation and warm follow-up
tests. Model tests hold the shared service lock. Earlier trials restore and
smoke-test the original service; later trials leave it stopped as requested.
Any partial failures stay in private validation
storage. No same-model Maya comparison or broad quality claim precedes these gates.

Local validation on 2026-10-25 builds a separate experimental binary and passes
all 17 component CTests plus 63 Python packer/installer/frontend/launcher checks.
All 3301 packed dense tensors and 120 sampled complete experts (three per layer)
are byte-compared with the source; this is not a full expert-pack checksum check.
The original 2.7-bpw reconstructed-prefill fixture retains byte-identical logits.

Both layer split and TP2 produce `323` and the expected identifier on frozen
17-token arithmetic and 164-token retrieval inputs, at 524288 configured capacity.
The latest short-input run passes snapshot, reset, cancellation and warm-prefill
checks; sequential PP/TP final logits match exactly on these two fixtures.
Token-wise TP reset replay additionally compares every input token's full-logit
fingerprint, not only the final answer. Layer GEMM prefill is predictive rather
than bit-identical to sequential decode (retrieval relative L2 0.16287,
KL about 6.99e-7).

TP2 with a 12-GiB reserve holds about 13004/15360 expert shards on the more
constrained rank. With a 2-GiB reserve it holds 14614/15360 there (95.1%), with
14677 on the other rank in the post-fix retrieval trial. Each rank still stages some
misses from the unchanged pack; this is **not** full weight residency. The smaller
reserve retains roughly 10 GiB more expert bytes per GPU, rather than reducing
total checkpoint bytes. Comparing this TP setting with layer split's 12-GiB
reserve is not an isolated tensor-parallelism memory or speed comparison.

Historical pre-fix TP retrieval trials fail exact replay intermittently at both
12-GiB and 2-GiB reserves; the latter first diverges at input position 22. The new
streaming path had published pageable-host descriptors and selections using the
default stream, without ordering their nonblocking consumer. Publication now
uses the consumer stream and an explicit completion fence. A synthetic immediate
consumer check exercises 128 changed-route/descriptor publications. Three fresh
post-fix TP retrieval trials pass exact per-token reset, snapshot, cancellation
and warm-prefill replay checks, with reserves of 2, 2 and 12 GiB respectively.
The aborted rank-local-cache isolation is not counted as a pass or a result about
cache sharing. Every historical failed receipt remains retained.

These initial bounded checks support short-input experimental compatibility, not
a general stability guarantee or promotion to the installed deployment. The
separately documented tests above extend 2.9 coverage to 4097 input tokens, local
HTTP chat with warm reuse, and bounded DSpark and static-prefix graph cases.
Actual 64K/524K input, sustained production serving, grouped prefill, batch slots,
a matched Maya comparison and broad quality remain unqualified for 2.9.
The tiny one- or six-step decode timings in the model test are diagnostics, not
performance benchmarks.

### Prefill optimization — opt-in grouped EXL3 experiment

The experiment retains the pinned source bytes, full 524288 configured capacity,
resident backbone experts and the chosen layer/DSpark mode. It runs in a
separate development checkout; no installed defaults or service state change.
Phase profiling is wired on both resident stages. GPU-grouped EXL3 expert prefill
uses bounded scratch, FP32 projection outputs and the existing Hadamard/activation
rounding. The candidate is opt-in with `DS41_PF_GROUPED=1` in the development build.
The existing reconstructed-GEMM path remains the baseline and default.

The grouped path takes routing counts and row/job tables directly from the GPU,
handles independent per-projection bit rates and separate gate/up sign vectors,
and reduces expert contributions in a deterministic order. It does not import
another model, a host expert tier, lossy int8 activation arithmetic or a new KV
cache format. Gate, up and down use separate packed-weight GEMM launches, with
64 rows per job and no reconstructed full matrices. Their Hadamards remain
separate, preserving independent gate/up sign vectors. The gate/up input arena
is reused for down outputs only after both input consumers finish. Grouped-kernel
scratch grows with at most 1024 tokens, not sequence capacity or expert-weight size.
Existing attention/indexer scratch still depends on configured capacity.
One aggregate active-expert count is read per stage/chunk; routes and weights
remain on the GPU. This is a layer-split mul1 path, not TP2, MCG or expert streaming.

The tiled MMA loop adapts Diffbot's MIT-licensed
[`xmoe_do/do_gemm.cu`](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/blob/58799d945d0f836fdb575e3a39a662b0a4d1543c/recipe/kernels/xmoe_do/do_gemm.cu).
Its notice is retained in `third_party/diffbot_exl3/LICENSE`. No alternate model
or converted deployment pack is imported. SGLang's asynchronous chunk scheduling
and FlashInfer's indexed-cache attention remain design references, not dependencies.

Synthetic tests compare with the current K12 path before model-sized timing.
Acceptance covers predictive distributions, continuation, snapshot/cancel/reset
and unchanged default-path logits. Optimized GEMM is not expected to be
bit-identical to reconstructed cuBLAS GEMM. Timings use identical prompt IDs,
configured capacity and speculation settings, with warm-up separate. Report the
actual selected chunk/batch: extra scratch or lazy allocations can reduce them.
No speedup is assumed for unmeasured prompts or devices.

`DS41_PF_PROFILE=1` prints each resident stage/chunk's phase timeline, including
host gaps between GPU events. It is not a kernel-only profiler or complete TTFT:
use external wall-clock timing for allocation, stage transfers and cleanup too.
`ds41_prefill_benchmark_test` is an explicit model benchmark, not an automatic
CTest. It requires local `--pack` and `--ids-file` arguments, resets before every
round (zero prefix reuse), separates warm-up and timed repeats, and can write/read
full-logit references with `--reference-out` / `--reference-in`. The reference
gate requires equal argmax, KL <0.02 and relative L2 <0.25; it is a regression
check, not an independent model-quality oracle. Default capacity is 524288.

The next stages, after this numerical/performance gate, adapt asynchronous
chunked pipeline scheduling and study direct indexed KV access. They are not
implemented or qualified by the grouped-kernel experiment alone.

#### Local grouped-prefill qualification — 2026-10-24

CUDA 13.4 / GCC 15 / SM120 builds and all 15 component CTests pass. The grouped
tests cover independent K1..K6 projections, both activation-rounding profiles,
empty rows, remapped resident slots, multi-job tails, 1024-token skewed routing,
short-workspace refusal, and poisoned-scratch CUDA graph replay. Maximum observed
relative L2 against existing K12 is 0.000528. At 1024 tokens the grouped workspace
is about 255 MiB; it is about 128 MiB at 512 tokens.

Real-weight checks retain 524288 configured capacity. The 513-token retrieval
returns `BRONZE-73921`, agrees predictively with sequential execution (KL
3.54e-8), and passes snapshot continuation, deterministic reset and cancellation
recovery. With DSpark enabled, serving-state checks pass full-logit verification
and prefix commits 1..4, rejected-tail rollback, boundaries 127/128/511/512,
isolated slots, published drafting and cancellation/retry. These checks compare
the engine with itself, not an independent model oracle. The 36 CPU
installer/tokenizer/mock-HTTP tests, six lossless dense-packer tests and six
deployment-wrapper tests also pass; no real HTTP claim follows.

Matched native prefill uses the same frozen 16384-token source-code fixture,
layer split 18/22, DSpark enabled, 524288 capacity, one snapshot, requested and
selected chunk/batch 512/256, zero prefix reuse, one warm-up and three measured
rounds. Two RTX PRO 6000 GPUs, 16 GiB host-memory limit, no cgroup swap and a
two-CPU quota are unchanged. Timing surrounds `Engine::prefill`, including its
scratch allocation, transfers and cleanup, but excludes model load, tokenization
and HTTP. This is a warm-engine/cold-prefix test, not cold service startup.

| Native prefill path | Median seconds | Tokens/s |
| --- | ---: | ---: |
| Default reconstructed GEMM | 32.064 | 510.976 |
| Opt-in grouped packed GEMM | 12.363 | 1325.218 |

This fixture improves 2.59x (61.4% less prefill time). The default path's logits
remain byte-identical to the pre-optimization reference on the same fixture.
The default path also reproduces the published build's saved full logits for
both batched and sequential execution on the historical 513-token retrieval.
Grouped logits are deterministic but not byte-identical: equal next token,
relative L2 0.05923 and KL 0.01216 against that reference. The scratch-sharing
revision preserves the earlier grouped prototype's logits byte for byte. These
results do not establish general coding quality or long-context throughput.

Observed cgroup memory peaks are 7.20 GiB default / 7.85 GiB grouped, including
file cache; process peak RSS is about 0.83 / 0.82 GiB. GPU total-use peaks sampled
every 0.5 s are about 94.5/93.3 GiB default and 94.3/93.3 GiB grouped, including
unrelated existing desktop processes. No cgroup swap or expert streaming occurs.

The **requested 512-batch trial fails exact replay**: it initially selects
512/512, then the second measured round selects 256/256 on the tail stage and
produces different logits. Its faster partial timing is not a qualified result.
The failure is retained, the replay gate is not relaxed, and batch 256 stays the
measured choice. No installer settings or generated configuration are changed.

Separate profiled 16K runs put the routed-expert phase at about 23.70 s default
versus 4.48 s grouped. In the grouped GPU-event timeline, routed experts now
account for 36%, HC mixes/norms 23% (2.82 s), and Engram 17% (2.16 s). These
rounded phase totals include host gaps and exclude inter-stage transfer gaps;
they are not Nsight kernel attribution. They motivate measuring HC work, Engram
overlap and stage handoffs next rather than assuming attention dominates.

Long-input qualification passes with actual 65536- and
524256-token retrieval inputs, each in a single 524288-token conversation with
32 or more tokens of generation headroom. Fresh default/grouped runs use the
same frozen prompt IDs, layer/DSpark mode and selected chunk/batch; each must
return the exact needle, retain resident backbone experts, stay within capacity,
and reproduce snapshot continuation. Full-logit comparisons retain the existing
equal-argmax, KL <0.02 and relative-L2 <0.25 gates. Resource receipts retain
memory/swap/oom observations and failed attempts. No sequential decode of the
entire long input or independent-model quality claim is implied.

Each row below is one actual prefill in a fresh engine, with DSpark enabled and
no prefix reuse. All four runs use the same test binary, 524288 capacity, layer
split 18/22, two snapshots and selected chunk/batch 512/256. Prefill timing
excludes model load, tokenization and HTTP. This repeated-background retrieval
fixture is not the 16K source-code performance workload; these are single-run
qualification timings, not warm-repeat medians.

| Actual input tokens | Path | Prefill seconds | Tokens/s |
| ---: | --- | ---: | ---: |
| 65536 | Default | 65.291 | 1003.757 |
| 65536 | Grouped | 34.411 | 1904.510 |
| 524256 | Default | 628.750 | 833.807 |
| 524256 | Grouped | 378.557 | 1384.878 |

Grouped prefill improves 1.90x at 64K and 1.66x on the actual 524256-token input.
Every run returns exactly `BRONZE-73921`, passes full-logit snapshot continuation,
and uses only resident backbone experts. Final positions are 65542 and 524262,
respectively, within the single-conversation 524288 limit. The 524256-token
fixture is byte-identical to the historical full-input fixture; its baseline is
a fresh run of the current default path, not a reused historical timing.

Predictive comparisons retain equal argmax and the original gates: at 64K,
KL is 6.18e-7 and relative L2 is 0.13284; at 524256, KL is 7.93e-8 and relative
L2 is 0.16324. Optimized logits are not bit-identical to the default path.
No expert streaming, cgroup swap or OOM event occurs. Maximum observed cgroup
memory across the four runs is about 9.36 GiB under the 16 GiB limit, including
file cache. These checks establish actual-input retrieval and greedy continuation,
not independent broad quality or full-context speculative-decode throughput.

Broad model quality, current container GPU runtime and
cross-backend portability remain unqualified for this experiment. The installed
deployment and its stopped-service state remain unchanged.

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
decode each shared quantized tile once for up to eight rows. CPU and TP verification
windows remain at most four input rows; the separate experimental 2.9
`DS41_EXPERIMENTAL_VERIFY_ROWS` control admits up to six on the layer split.
The layer-split path transfers the tentative hidden
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

Decode-to-prefill memory handoff: resident layer-split
prefill releases inactive verification workspaces/graphs and the optional MTP
expert LRU before choosing its bounded scratch plan. It preserves conversation
KV, snapshots, draft rings and pending rows, and all backbone experts. Verification
workspaces and the MTP LRU rebuild on demand during decode. The reserve and 524288
capacity stay unchanged. Acceptance requires two warm draft/verify-to-prefill
turns, exact verification replay, snapshot continuation and no backbone streaming;
fresh-engine prefill alone does not cover this transition.

Local validation on 2026-10-24 reproduced `insufficient bounded scratch VRAM`
after 96 draft/verify calls on the earlier build. The fix passes both turns with
grouped and reconstructed-GEMM prefill at 524288 capacity, with 4320 routed MTP
uses per turn, exact committed-logit and snapshot replay checks, and every backbone
expert resident. The affected stage reclaims about 0.992 GiB of MTP cache plus its
verification workspace. All 15 component CTests pass. A real two-turn HTTP check
also passes: 38021 then 38865 prompt tokens, 128 output tokens each, with 38148
cached tokens on the second turn. This is a warm-transition regression, not a new
actual-524K-input qualification or an independent model-quality evaluation.

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

- `ds41_stage_prefill_model_test --out ... --max-seq 524288 --chunk 512
  --batch 256 --sequential 0 --dspark 1`: one actual batched prefill, saved full
  logits, exact snapshot continuation and up to 24 greedy output tokens. Create
  a fresh output directory first and independently check the input token count.
  `--sequential 0` skips feeding the entire long prompt token by token; it does
  not skip prefill or continuation. `DS41_PF_GROUPED=1` selects the candidate.
- `ds41_stage_serving_model_test --dspark 1`: full-logit prefix commits 1–4,
  ring/index/compressor boundaries 127/128/511/512, session isolation, draft
  replay and cancellation. Add `--tensor 1 --max-seq 524288` for TP2 with full
  configured capacity. `--reference-out` / `--reference-in` compare PP/TP logits
  after the same sequential input arithmetic; they are not a separate model oracle.
- `ds41_warm_prefill_model_test`: two resident layer-split turns at 524288
  capacity, using 1K–16K input IDs. Each warms the 1 GiB MTP LRU with 96
  draft/verify calls, checks every committed row against sequential logits, then
  feeds a 751-token extension through both bulk and final-token prefill. Snapshot
  replay and a rebuilt draft/verify workspace must preserve exact logits, with
  every backbone expert still resident. Run with `DS41_PF_GROUPED=1` to cover
  grouped prefill; unset it to cover reconstructed-GEMM prefill.
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

At this historical checkpoint, unverified: complete actual 524K input, fresh real-model API qualification,
broad numerics/quality/performance, and container GPU runtime. No full
upstream-suite or cross-backend build coverage is claimed. Private failed and
cancelled receipts remain local.
The later grouped-prefill qualification above closes actual-524K input for the
layer-split default/grouped paths; it does not change the cancelled run's status.

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

The persistent service remains stopped. These earlier checks do not establish
fresh real-model HTTP, full-524K input, an independent full-model reference,
broad coding quality or container qualification. Later layer-split actual-input
evidence is recorded in the grouped-prefill qualification above; TP2's actual
full-524K input remains unqualified.
