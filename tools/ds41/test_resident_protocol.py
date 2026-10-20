"""Opt-in resident native-pipe checks. One GPU engine, no download or HTTP listener.

Run after building, with --exe, --pack and --ids-file (comma-separated IDs).
--parallelism layer|tensor and --spec none|dspark select the path to qualify.
"""
import argparse
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ds41.test_serve_protocol import Engine


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--parallelism", choices=("layer", "tensor"), default="layer")
    parser.add_argument("--spec", choices=("none", "dspark"), default="none")
    parser.add_argument("--draft-cache-mib", type=int, default=1024)
    a = parser.parse_args()
    prompt = [int(item) for item in Path(a.ids_file).read_text().split(",") if item.strip()]
    require(8 <= len(prompt) < 1024, "use a bounded, nontrivial prompt")
    command = [a.exe, "--serve", "--pack", a.pack, "--devices", "0,1",
               "--parallelism", a.parallelism, "--max-context", "4096", "--threads", "2",
               "--ram-budget-gib", "0", "--adapt-every", "0", "--vram-reserve-mib", "1024",
               "--prefill-chunk", "512", "--prefill-batch", "256", "--snapshots", "2",
               "--spec", a.spec, "--draft-cache-mib", str(a.draft_cache_mib)]
    if a.parallelism == "layer":
        command += ["--split-layer", "18"]
    engine = Engine(command)
    try:
        require(engine.ready[1] == "4096", "wrong READY capacity")
        _, expected, done, error = engine.request(prompt, 32)
        require(error is None and done is not None and expected, "initial request failed")
        # Change the session, then repeat from a fresh prefix.
        other = [2 if prompt[0] != 2 else 3, *prompt[1:8]]
        engine.request(other, 1)
        lines, actual, done, error = engine.request(prompt, 32)
        require(error is None and done is not None and actual == expected, "fresh greedy replay differs")
        require(lines[0].startswith("RESUME "), "missing reuse protocol")
        _, stopped, done, error = engine.request(prompt, 256, stop_after_t=3)
        require(error is None and done is not None and done[5] == "cancel" and 3 <= len(stopped) < 32,
                "decode cancellation failed")
        engine.request(other, 1)
        _, actual, done, error = engine.request(prompt, 32)
        require(error is None and done is not None and actual == expected, "post-cancel continuation differs")
        for request in ("GEN 2 999999999", "GEN 4096 1,2"):
            engine.send(request)
            require(engine.get().startswith("ERR"), "invalid request was accepted")
        engine.send("QUIT")
        require(engine.p.wait(timeout=120) == 0, "QUIT did not exit cleanly")
        print(f"RESULT pass resident protocol parallelism={a.parallelism} spec={a.spec}: "
              "READY, greedy replay, STOP/recovery, invalid IDs/overflow, QUIT")
    finally:
        if engine.p.poll() is None:
            engine.p.terminate()
            try:
                engine.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                engine.p.kill()
                engine.p.wait()


if __name__ == "__main__":
    main()
