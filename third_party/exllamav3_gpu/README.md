# exllamav3 GPU GEMV and reconstruct subset

Port note: the implementation and reversible vendor patch are retained from the
Strata-DS source identified in `docs/DEEPSEEK_1017.md`. Historical task/history
references below refer to that source repository; task reports, benchmark data
and design snapshots are not imported into this upstream-based branch.

Source: https://github.com/turboderp-org/exllamav3, commit
`16a49792a3c93d8432d72e6c4bce800841566577` (v1.5.4). MIT license;
see LICENSE (copyright Turboderp).

The paths in UPSTREAM.sha256 are copied from `exllamav3/exllamav3_ext/`
except LICENSE, which comes from the repository root. They were copied and
byte-verified before adaptation. The original history contains the unmodified
vendor import followed by the raw-pointer adaptation; the historical staging
plan is retained in ds41/tasks/K10.COMMITS.md.

This subset uses the small-row GEMV path, so no GEMM compilation units are
needed. `quant/exl3_gemv.cu` is included by the K10 unity translation unit.
The adaptation replaces its PyTorch host dispatch with a raw device-job API,
and moves the GEMV kernel's input/output Hadamards to surrounding launches.
The trellis decoder, codebook, MMA loop, FP16-to-FP32 folds, and reduction
remain upstream code. K10 calls the unchanged Hadamard helpers directly.
The initial adapter instantiated only 3-bit mul1, single-row, narrow GEMV.
The mixed-rate extension is described below.

`strata.patch` records all changes to upstream files. Apply it in reverse to
recover the unmodified vendor tree; verify against UPSTREAM.sha256. This makes
the first commit reproducible without relying on an untracked temporary copy.
The patch itself belongs to the subsequent adaptation commit.

K10-01 also applies the upstream narrow two-block launch bound to the integer
3-bit instance. This changes compiler register allocation, not the decoding,
MMA, FP16 fold cadence, reduction, or surrounding Hadamards/activation. The
updated reversible patch includes this change. See ds41/tasks/K10.REPORT.md
for compile-only resource evidence and the still-pending GPU measurements.

K10-03 keeps the exact K10-01 two-block bound and four-slot prefetch ring.
Only the narrow integer packed-weight loads change from `__ldcs`
(`ld.global.cs.u32`) to `__ldcg` (`ld.global.cg.u32`). The latter caches in L2
without filling L1. The scalar width, original 4-byte alignment requirement,
lane guards and addresses are unchanged; wide and half-integer templates keep
the upstream `__ldcs` fallback. No wider vector load is introduced. Hadamard,
input and codebook loads, decode arithmetic, FP16 folds and reductions remain
unchanged. The host check normalizes exactly this explicit policy block before
its whole-loop comparison; the reversible patch and all pristine hashes are
rechecked. Timing and GPU acceptance remain pending.

K10-08 combines the exact K10-03 narrow integer cache policy with K10-02's
register schedule: two prefetch slots, retaining the original four-slice
narrow arithmetic unroll and FP16 fold boundaries. The two-CTA bound stays.
Wide/half-integer load policies, guards, widths and addresses remain unchanged.
The reversible patch contains both changes; the host audit normalizes only
those explicit substitutions before comparing upstream arithmetic. An actual
source-extracted C++ schedule model checks ring tails and fixed upstream fold
boundaries, including mutation rejection. K10-08's own GPU acceptance and
performance are pending; individual controls' gains are not assumed additive.

## K12 reconstruct subset

K12 adds `quant/reconstruct.cu` and `quant/reconstruct.cuh` from the same
upstream commit. Both were copied unmodified and SHA-256 checked before
adaptation; their pristine digests are appended to UPSTREAM.sha256.
The reviewer stages the pristine import separately, as described in
`ds41/tasks/K12.COMMITS.md` (the agent cannot write .git).

The initial K12 adapter used the trellis-only `reconstruct_tile<3, 2, false>` and its upstream
batched wrapper, with a one-entry device pointer table per launch. It keeps
the tile decoder, shuffle, layout and stores unchanged. The adapter removes
Torch dispatch, unused reconstruct/Hadamard entry points and instance tables,
and exposes one raw-pointer host launch. The complete change is appended to
strata.patch. K10's existing sources and patch sections remain unchanged.
K12 includes reconstruct.cu through its unity translation unit; no GEMM
compilation units are needed for the cuBLAS path. Input/output Hadamards stay
in the activation pipeline to retain K10's rounding locations.

## Mixed integer rates (SAGE 1.59bpw)

K10 now accepts K1..K6 in each device job. A single 512-thread launch switches
on the job's rate before any barrier. K2/K3/K4 retain the narrow register
codebook extraction, two-slot prefetch ring, four-step FP16 fold, L2 loads,
and two-CTA launch bound. K1/K5/K6 instantiate the same GEMV template with
padded shared-memory staging and the existing upstream `dq_dispatch<K, 2>`.
Each warp loads two tiles. A tile uses one or two warp loads, with guarded
padding. Decode, MMA, folds, and reduction use the existing templates.
No new upstream files are needed. The public Strata interfaces are unchanged.

K12 prepares three device jobs with independent rates. Each reconstruction
launch switches on that job's K and calls `reconstruct_tile<K, 2, false>`.
The upstream tile bodies remain byte-identical. The job table fits in the
existing 256-byte workspace prefix. Neither path copies descriptors to the
host, allocates, synchronizes with the host, or launches once per rate.

`strata.patch` remains reversible against every `UPSTREAM.sha256` entry.
Host checks audit the K3 source path and the new packed-load addresses.
GPU compilation, numerical parity, register allocation, and 3-bit timing
still require the reviewer box. Source preservation is not timing evidence.

## MCG K2 resident split adapter

The device job additionally selects codebook 1 for MCG K2. Codebook 2 remains
the default for existing mul1 descriptors. The MCG instantiation uses the
already-vendored `decode_3inst<1>` and paired decoder without changing their
arithmetic or the trellis layout; it does not create a different quantization.
The loader checks the stored `0xCBAC1FED` marker and restricts this input to
resident split decode. K12 and the CPU fallback do not admit MCG packs.

`strata.patch` includes the descriptor/dispatch changes and still reverses to
all pristine hashes. `src/ds41/tests/mcg_decode_test.cu` checks all 65,536
codebook entries bitwise and K2 GEMV against a separate scalar reconstruction.
See `docs/MULTI_GPU.md` for measured synthetic evidence and full-model limits.
