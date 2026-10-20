"""tools/ds41/test_serve_protocol.py - ds41_serve end to end over its pipes, as serve/server.py drives it.

    python tools/ds41/test_serve_protocol.py --exe build/ds41_serve --pack PACK [--expert-profile F] [--threads T]

Needs a GPU and the model pack. Checks: READY and INFO; a request's RESUME, PP, T and DONE lines; the session reuse
(a prompt that continues the session reads only the new tokens, and gives the tokens a fresh start gives); a prompt
that does not continue it starts over; STOP while the prompt is read and while tokens are written; the ERR answers;
QUIT ends the engine with code 0; with static residency (--adapt-every 0) a seeded sampled request repeats.
"""
from __future__ import annotations

import argparse
import queue
import random
import subprocess
import sys
import threading
import time

failures = 0


def check(ok: bool, what: str) -> None:
    global failures
    print(f"{'ok' if ok else 'FAIL'}: {what}", flush=True)
    if not ok:
        failures += 1


class Engine:
    def __init__(self, cmd: list[str]):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.info = {}
        while True:
            line = self.get(1800)
            if line.startswith("INFO "):
                self.info = dict(kv.partition("=")[::2] for kv in line.split()[1:])
            if line.startswith("READY"):
                self.ready = line.split()
                return

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def get(self, timeout=600.0) -> str:
        line = self.q.get(timeout=timeout)
        if line is None:
            raise RuntimeError(f"the engine ended (code {self.p.wait()})")
        return line

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def request(self, ids, max_new, keys="", stop_after_pp=None, stop_after_t=None):
        """One GEN: returns (lines, tokens, done fields or None, err or None)"""
        self.send(f"GEN {max_new}{' ' + keys if keys else ''} {','.join(map(str, ids))}")
        lines, toks, pps = [], [], 0
        while True:
            line = self.get()
            lines.append(line)
            if line.startswith("PP "):
                pps += 1
                if stop_after_pp is not None and pps == stop_after_pp:
                    self.send("STOP")
            elif line.startswith("T "):
                toks.append(int(line[2:]))
                if stop_after_t is not None and len(toks) == stop_after_t:
                    self.send("STOP")
            elif line.startswith("DONE"):
                return lines, toks, line.split(), None
            elif line.startswith("ERR"):
                return lines, toks, None, line[4:]


def prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [0] + [rng.randrange(3, 128000) for _ in range(n - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--expert-profile", default="")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--max-context", default="16384")
    a = ap.parse_args()
    cmd = [a.exe, "--serve", "--pack", a.pack, "--max-context", a.max_context, "--threads", a.threads]
    if a.expert_profile:
        cmd += ["--expert-profile", a.expert_profile]
    t0 = time.time()
    e = Engine(cmd)
    print(f"READY after {time.time() - t0:.0f} s: {' '.join(e.ready)}; INFO {e.info}", flush=True)
    check(e.ready[1] == a.max_context and "stop" in e.ready[2:], "READY <context> stop")
    check("engine" in e.info and "context" in e.info, "INFO carries engine= and context=")

    # a plain request
    P = prompt(300, 1)
    lines, out, done, err = e.request(P, 8)
    check(err is None and done is not None, f"a GEN ends with DONE ({err})")
    check(lines[0] == "RESUME 0", "a fresh session reuses nothing: RESUME 0")
    check(any(l.startswith("PP ") for l in lines), "the prompt sends PP lines")
    pp_last = [l for l in lines if l.startswith("PP ")][-1].split()
    check(pp_last[1] == pp_last[2] == str(len(P)), "the last PP reports the whole prompt")
    check(len(out) == 8 and done[1] == "8" and done[2] == str(len(P)) and done[5] == "length",
          "8 tokens, DONE 8 <prompt> ... length")
    check(len(done) >= 16, "DONE has upstream's 15 fields")

    # the next turn continues the session: only the new part is read
    P2 = P + out + prompt(40, 2)[1:]
    lines, out2, done, err = e.request(P2, 6)
    reused = len(P) + len(out) - 1   # the last output was never fed
    check(err is None and lines[0] == f"RESUME {reused}", f"a continuing prompt reuses the session ({lines[0]})")
    check(done is not None and done[8] == str(reused) and done[14] == str(len(P2) - reused),
          "DONE reports the reused and the read tokens")

    # the same prompt from a fresh start gives the same tokens
    e.request(prompt(50, 9), 1)       # another conversation: the session starts over
    lines, out3, done, err = e.request(P2, 6)
    check(lines[0] == "RESUME 0", "a prompt that does not continue the session starts over")
    print(f"  continued {out2} fresh {out3}", flush=True)
    check(out2[0] == out3[0], "the reused session gives the first token a fresh start gives")

    # a snapshot before the last prompt token: the next turn changes the last answer's start (DeepSeek drops its
    # reasoning), so it differs from the session at the old prompt's last token - for a pass (400) and windows (60)
    for size, seed in ((400, 21), (60, 22)):
        e.request(prompt(30, 900 + seed), 1)               # another conversation: the session starts over
        Pa = prompt(size, seed)
        lines, outa, done, err = e.request(Pa, 4)
        Q = Pa[:-1] + prompt(31, 100 + seed)[1:]           # the old prompt but its last token, then 30 new tokens
        lines, outq, done, err = e.request(Q, 3)
        check(lines[0] == f"RESUME {len(Pa) - 1}", f"{size}: the next turn goes back to the snapshot ({lines[0]})")
        check(done is not None and done[14] == "30", f"{size}: it reads only the 30 new tokens ({done and done[14]})")
        e.request(prompt(30, 950 + seed), 1)
        _, fresh, _, _ = e.request(Q, 3)
        print(f"  {size}: from the snapshot {outq}, fresh {fresh}", flush=True)

    # STOP while the prompt is read: cancel, nothing generated, the next request works
    long_p = prompt(6000, 3)
    lines, out, done, err = e.request(long_p, 4, stop_after_pp=2)
    check(done is not None and done[5] == "cancel" and done[1] == "0", "STOP during the prompt: DONE 0 ... cancel")
    check(done is not None and int(done[14]) < len(long_p), "fewer prompt tokens read than the prompt")
    lines, out, done, err = e.request(P, 2)
    check(done is not None and lines[0] == "RESUME 0" and done[5] == "length", "the engine works after the cancel")

    # STOP while a short prompt part is read in verify windows: the windows read so far stay in the session
    lines, out, done, err = e.request(P, 2)
    P3 = P + out + prompt(101, 4)[1:]   # 100 new tokens: the window path
    lines, out, done, err = e.request(P3, 4, stop_after_pp=3)
    check(done is not None and done[5] == "cancel" and 0 < int(done[14]) < 100,
          f"STOP during the windows: DONE ... cancel after {done[14] if done else '?'} tokens")
    lines, out, done, err = e.request(P3, 2)
    check(done is not None and int(lines[0].split()[1]) > len(P), f"the next request reuses what was read ({lines[0]})")

    # STOP while tokens are written
    lines, out, done, err = e.request(P + [5, 6, 7], 200, stop_after_t=3)
    check(done is not None and done[5] == "cancel" and len(out) < 10, f"STOP during decode ends it ({len(out)} tokens)")

    # the ERR answers; the engine keeps running
    for line, what in [("BGEN 0 4 1,2", "batch slots"), ("VRAM", "VRAM"), ("GENI 4 x.bin 1,2", "images"),
                       ("GEN 4", "no ids"), (f"GEN 4 1,{10**9}", "an id past the vocabulary"),
                       (f"GEN {a.max_context} 1,2", "past the context")]:
        e.send(line)
        reply = e.get()
        check(reply.startswith("ERR"), f"{what}: {reply}")

    # sampled requests run (with the adaptive tier a seeded result does not repeat run to run: upstream's limit too)
    keys = "temperature=0.8 top_p=0.95 top_k=40 seed=7"
    _, s0, done, _ = e.request(prompt(20, 13), 12, keys)
    check(done is not None and len(s0) >= 1, "a sampled request runs")

    e.send("QUIT")
    code = e.p.wait(timeout=120)
    check(code == 0, f"QUIT ends the engine with code 0 ({code})")

    # static residency (--adapt-every 0, upstream: --adapt-every 100000): the same experts on the same path in every
    # run, so a turn resumed from a snapshot gives exactly the tokens of a fresh start, and a seeded sampled request
    # repeats (with the adaptive tier, which experts the CPU computes depends on the run)
    e = Engine(cmd + ["--adapt-every", "0"])
    for size, seed in ((400, 31), (60, 32)):
        e.request(prompt(30, 900 + seed), 1)
        Pa = prompt(size, seed)
        e.request(Pa, 4)
        Q = Pa[:-1] + prompt(31, 100 + seed)[1:]
        lines, outq, _, _ = e.request(Q, 3)
        e.request(prompt(30, 950 + seed), 1)
        _, fresh, _, _ = e.request(Q, 3)
        print(f"  static, {size}: from the snapshot {outq}, fresh {fresh} ({lines[0]})", flush=True)
        check(lines[0] == f"RESUME {len(Pa) - 1}" and outq == fresh,
              f"static residency, {size}: the snapshot gives the tokens of a fresh start")
    e.request(prompt(20, 11), 1)
    _, s1, _, _ = e.request(P, 12, keys)
    e.request(prompt(20, 12), 1)   # another conversation: both runs below start fresh
    _, s2, _, _ = e.request(P, 12, keys)
    print(f"  sampled {s1} / {s2}", flush=True)
    check(s1 == s2 and len(s1) >= 1, "with static residency a seeded sampled request repeats")
    e.send("QUIT")
    check(e.p.wait(timeout=120) == 0, "the second engine ends with code 0")
    print(f"RESULT {'fail' if failures else 'pass'} serve_protocol", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
