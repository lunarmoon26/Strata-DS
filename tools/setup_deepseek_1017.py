"""Offline setup and preflight for the pinned, resident two-GPU 1017 profile.

Configuration is written only through setup.py. This module's CLI provides
read-only checks and a small API smoke test; it never loads model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = "1017-EXL3-2.7BPW"
MODEL_ID = "deepseek-v4.1-flash-1017"
REVISION = "22205b9aff07ae7b999d466aaf96446c20f4ea97"
SOURCE = "sudoingx-DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw"
PACK = "sudoingx-1017-22205b9a-EXL3-2.7bpw-dspark"
CONFIG_NAME = "strata-deepseek-1017.json"
LAUNCHER_NAME = "run-deepseek-1017.sh"
GENERATOR = "Strata/setup.py:deepseek-1017"
LEGACY_GENERATOR = "deepseek-v4.1-flash/install.sh"
SUPPORTED_FLAGS = {
    "--family", "--model", "--configure-only", "--data-dir", "--ds41-engine",
    "--gpus", "--telemetry-gpus", "--log-dir", "--context", "--port", "--parallel",
    "--spec", "--host", "--no-start", "--no-browser", "--yes", "--setup", "--backend",
}


def is_profile(cfg):
    return isinstance(cfg, dict) and cfg.get("format") == "deepseek_v41" and cfg.get("model_name") == MODEL_ID


def configuration(a, root=ROOT):
    """Build the one supported resident configuration from setup's parsed args."""
    if not sys.platform.startswith("linux") or a.backend not in (None, "cuda"):
        raise ValueError("1017 resident serving supports Linux + NVIDIA only")
    if a.host not in (None, "127.0.0.1"):
        raise ValueError("the 1017 profile keeps the listener on 127.0.0.1")
    context = a.context if a.context is not None else 524288
    parallel = a.parallel if a.parallel is not None else 1
    port = a.port if a.port is not None else 8081
    if not 1 <= context <= 524288:
        raise ValueError("1017 context must be in 1..524288; no extra RoPE scaling")
    if not 1 <= parallel <= 4:
        raise ValueError("1017 parallel must be in 1..4 (serial GPU scheduling)")
    if not 1 <= port <= 65535:
        raise ValueError("port must be in 1..65535")
    spec = a.spec or "none"
    if spec not in ("none", "suffix", "dspark"):
        raise ValueError("1017 spec must be none|suffix|dspark")
    gpus = (a.gpus or "0,1").split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(
            not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", x) for x in gpus):
        raise ValueError("1017 --gpus needs two distinct CUDA indices or GPU UUIDs")
    telemetry = (a.telemetry_gpus or "0,1").split(",")
    if len(telemetry) != 2 or any(not x.isdecimal() for x in telemetry):
        raise ValueError("--telemetry-gpus needs two distinct physical GPU indices")
    telemetry = list(map(int, telemetry))
    if len(set(telemetry)) != 2:
        raise ValueError("--telemetry-gpus needs two distinct physical GPU indices")
    data = Path(a.data_dir or root.parent / "Strata-data").expanduser().resolve()
    model = (data / "models" / SOURCE / REVISION).resolve()
    pack = (data / "packs" / PACK).resolve()
    candidates = [root / directory / "bin/ds41_serve" for directory in (
        "build-sm120-resident", "build-sm120-serving", "build-sm120-split")]
    built = next((p for p in candidates if p.exists()), candidates[0])
    exe = Path(a.ds41_engine or built).expanduser().resolve()
    logs = Path(a.log_dir or root / "logs").expanduser().resolve()
    return {
        "_generated_by": GENERATOR, "format": "deepseek_v41", "model_name": MODEL_ID,
        "aliases": ["strata"], "tokenizer": str(model), "exe": str(exe), "cwd": str(root),
        "args": ["--pack", str(pack), "--devices", "0,1", "--split-layer", "18",
                 "--max-context", str(context), "--threads", "2", "--ram-budget-gib", "0",
                 "--adapt-every", "0", "--vram-slots", "-1", "--vram-reserve-mib", "1024",
                 "--prefill-chunk", "1024", "--prefill-batch", "256", "--snapshots", "4",
                 "--batch", str(parallel if parallel > 1 else 0), "--spec", spec],
        "parallel": parallel, "sampling": {"temperature": 0}, "telemetry_gpus": telemetry,
        "host": "127.0.0.1", "port": port, "open_browser": False,
        "log": str(logs / "engine.log"), "cuda_visible_devices": ",".join(gpus),
        "python": str(Path(sys.executable).absolute()),
    }


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if not is_profile(cfg) or cfg.get("_generated_by") != GENERATOR or cfg.get("host") != "127.0.0.1":
        raise ValueError("Unrecognized 1017 configuration; rerun setup with --configure-only")
    return cfg


def check(cfg):
    """Check small metadata/tokenizer files, never allocate or map model tensors."""
    model = Path(cfg["tokenizer"])
    pack = Path(cfg["args"][cfg["args"].index("--pack") + 1])
    if not Path(cfg["exe"]).is_file() or not os.access(cfg["exe"], os.X_OK):
        raise ValueError("1017 engine is missing or not executable; build ds41_serve first")
    if model.name != REVISION or model.parent.name != SOURCE:
        raise ValueError("configuration does not select the pinned sudoingx checkpoint")
    info = dict(line.split(None, 1) for line in (pack / "pack_info.txt").read_text().splitlines() if line.strip())
    if info.get("finished") != "1" or Path(info.get("source", "")).resolve() != model:
        raise ValueError("pack is unfinished or not from the pinned sudoingx checkpoint")
    for name in ("dense.bin", "experts.bin", "mtp_experts.bin", "engram.txt", "engram_hash.txt", "index.txt"):
        if not (pack / name).is_file():
            raise ValueError(f"pack file missing: {name}")
    for name in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
        p = model / name
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"Engram shard is missing or symlink-dependent: {name}")
    sys.path.insert(0, str(ROOT))
    from serve.deepseek import DeepSeekTokenizer
    if DeepSeekTokenizer(model).eos_id != 1:
        raise ValueError("wrong tokenizer EOS")
    print(f"Checked: {MODEL_ID}, pinned pack metadata, two-GPU layer-18 split", flush=True)


def preflight(cfg):
    check(cfg)
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", cfg["port"]))


def launcher_text(cfg, config_path):
    root = Path(cfg["cwd"])
    logs = Path(cfg["log"]).parent
    digest = hashlib.sha256((json.dumps(cfg, indent=2) + "\n").encode()).hexdigest()
    helper = shlex.join([cfg["python"], str(root / "tools/setup_deepseek_1017.py"),
                         "preflight", "--config", str(config_path), "--config-sha256", digest])
    server = shlex.join([cfg["python"], str(root / "serve/server.py"), "--engine", "strata",
                         "--config", str(config_path), "--host", "127.0.0.1", "--port", str(cfg["port"])])
    command = (f"set -euo pipefail\nexport CUDA_VISIBLE_DEVICES={shlex.quote(cfg['cuda_visible_devices'])}\n"
               "export DS41_ROUTE_TAIL_SKIP=0 DS41_SSD_TAIL_SKIP=0 DS41_GRAPH=1\n"
               f"{helper}\ncd {shlex.quote(str(root))}\nexec {server}\n")
    return f'''#!/usr/bin/env bash
# Generated by {GENERATOR}; rerun setup --configure-only to change settings.
set -euo pipefail
if (( $# )); then
  echo 'Set options through setup --configure-only, not the generated launcher.' >&2
  exit 2
fi
mkdir -p {shlex.quote(str(logs))}
exec flock --no-fork -n -E 73 {shlex.quote(str(logs / 'service.lock'))} bash -c {shlex.quote(command)}
'''


def write_generated(root, cfg):
    outputs = [(root / CONFIG_NAME, json.dumps(cfg, indent=2) + "\n", 0o600),
               (root / LAUNCHER_NAME, launcher_text(cfg, root / CONFIG_NAME), 0o700)]
    for path, _, _ in outputs:
        if path.is_symlink():
            raise ValueError(f"Refusing generated-file symlink: {path.name}")
        if path.exists():
            text = path.read_text()
            if path.name == CONFIG_NAME:
                saved = json.loads(text)
                owned = isinstance(saved, dict) and saved.get("_generated_by") in (GENERATOR, LEGACY_GENERATOR)
            else:
                owned = any(text.startswith(f"#!/usr/bin/env bash\n# Generated by {owner};")
                            for owner in (GENERATOR, LEGACY_GENERATOR))
            if not owned:
                raise ValueError(f"Refusing to overwrite non-generated file: {path.name}")
    for path, content, mode in outputs:
        if path.exists() and path.read_text() == content:
            path.chmod(mode)
            continue
        if path.exists():
            fd, backup = tempfile.mkstemp(prefix=path.name + ".bak-", dir=root)
            os.close(fd)
            shutil.copyfile(path, backup)
        fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=root)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def configure(a, root=ROOT):
    cfg = configuration(a, root)
    check(cfg)
    write_generated(root, cfg)
    print(f"Installed {CONFIG_NAME} and {LAUNCHER_NAME}; no server was started.")
    return 0


def start(config_path):
    cfg = load_config(config_path)
    launcher = Path(cfg["cwd"]) / LAUNCHER_NAME
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError("1017 launcher missing; rerun setup with --configure-only")
    return subprocess.call([str(launcher)])


def smoke(cfg, port=None):
    port = cfg["port"] if port is None else port
    def request(path, body=None):
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("STRATA_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     None if body is None else json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
    health = request("/health")
    context = int(cfg["args"][cfg["args"].index("--max-context") + 1])
    if not health.get("loaded") or health.get("max_context") != context:
        raise ValueError("server not loaded with the installed context")
    if not any(m["id"] == MODEL_ID for m in request("/v1/models")["data"]):
        raise ValueError("server does not advertise the installed model")
    result = request("/v1/chat/completions", {
        "model": MODEL_ID, "messages": [{"role": "user", "content": "What is 17 * 19? Reply with only the integer."}],
        "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}})
    if result["choices"][0]["message"]["content"].strip() != "323":
        raise ValueError("arithmetic smoke check failed")
    print(json.dumps({"status": "pass", "health": health, "answer": "323"}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["check", "preflight", "smoke"])
    ap.add_argument("--config", type=Path, default=ROOT / CONFIG_NAME)
    ap.add_argument("--port", type=int, help="smoke-test endpoint override only")
    ap.add_argument("--config-sha256", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.config_sha256 and hashlib.sha256(a.config.read_bytes()).hexdigest() != a.config_sha256:
        raise ValueError("1017 config and launcher differ; rerun setup --configure-only before starting")
    cfg = load_config(a.config)
    if a.action == "smoke":
        smoke(cfg, a.port)
    elif a.action == "preflight":
        preflight(cfg)
    else:
        check(cfg)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, IndexError) as e:
        raise SystemExit(str(e))
