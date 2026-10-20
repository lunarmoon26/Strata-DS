# DeepSeek 1017 resident profile (experimental)

**Status:** upstream-based source port, not the deployed engine. The base is
`Niko1221/Strata@e88ee39d1d6be6e3364183af3b39b62a7ed6f211`.
The pre-migration history is retained on `legacy/pre-upstream-1017`. Local
installer edits, model packs and the running server remain unchanged. Updating
the code's ancestry does not change GitHub's displayed fork-parent metadata.

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
- Layers 0–17 and 18–39 run on two CUDA-visible devices. Backbone experts remain
  resident. Engram shards stay independently available on the data disk.
- Capacity is at most 524288; published RoPE remains unchanged. Ordinary decode
  is default. Optional speculation and isolated slots are functional paths, not
  speed claims.
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

The source port is checked with a source/build dependency audit, offline installer
tests, DeepSeek tokenizer/parser/API tests and selected regressions of touched
upstream interfaces. The CUDA binaries compile without loading weights. GPU component
tests and a new full-model cold start/long-context run are separate qualification
gates; they are not run alongside the existing model server.

Historical measurements in the old fork qualify that old binary only. They are
not performance or full-model correctness evidence for this port. Publication of
the source is not deployment approval; real-model qualification and a reviewed
deployment pin are still required before replacing the existing service.

### Validation of this port — 2026-10-20

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

Still unverified: real-model startup/API behavior, 524K input, new-binary
numerics, performance, GPU component execution, and container GPU runtime. No
full upstream-suite or cross-backend build coverage is claimed.
