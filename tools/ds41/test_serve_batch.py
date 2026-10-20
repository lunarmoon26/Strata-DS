"""tools/ds41/test_serve_batch.py - ds41_serve --batch over its pipes, as serve/server.py drives it (upstream's batch
protocol, docs/BATCHING.md).

    python tools/ds41/test_serve_batch.py --exe build/ds41_serve --pack PACK [--expert-profile F]

Needs a GPU and the model pack. Checks: INFO batch_slots; a BGEN admission answers PP, T, DONE, then BADM 1; the slots
then write BT lines between requests and BDONE <slot> <n> length at max_new; with static residency the tokens of three
slots decoded together equal each request decoded alone (GEN); BSTOP ends a slot with one more BT and BDONE cancel;
a BGEN for a busy or missing slot answers ERR and BADM 0; a solo GEN after the slots finished reuses a slot's
conversation (slot_cache); QUIT ends the engine with code 0.
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
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.info = {}
        while True:
            line = self.get(1800)
            if line.startswith("INFO "):
                self.info = dict(kv.partition("=")[::2] for kv in line.split()[1:])
            if line.startswith("READY"):
                return

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def get(self, timeout=900.0):
        line = self.q.get(timeout=timeout)
        if line is None:
            raise RuntimeError(f"the engine ended (code {self.p.wait()})")
        return line

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def until(self, pred, slots_out=None):
        """lines until pred(line); BT/BDONE lines on the way go to slots_out (slot -> list of lines)"""
        seen = []
        while True:
            line = self.get()
            if line.startswith(("BT ", "BDONE ")) and slots_out is not None:
                slots_out.setdefault(int(line.split()[1]), []).append(line)
            else:
                seen.append(line)
            if pred(line):
                return seen

    def gen(self, ids, max_new):
        self.send(f"GEN {max_new} {','.join(map(str, ids))}")
        lines = self.until(lambda l: l.startswith(("DONE", "ERR")))
        return [int(l[2:]) for l in lines if l.startswith("T ")], lines

    def admit(self, slot, ids, max_new, slots_out):
        self.send(f"BGEN {slot} {max_new} {','.join(map(str, ids))}")
        return self.until(lambda l: l.startswith("BADM "), slots_out)


def prompt(n, seed):
    r = random.Random(seed)
    return [0] + [r.randrange(3, 128000) for _ in range(n - 1)]


def slot_tokens(lines):
    return [int(l.split()[2]) for l in lines if l.startswith("BT ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--expert-profile", default="")
    ap.add_argument("--threads", default="16")
    a = ap.parse_args()
    cmd = [a.exe, "--serve", "--pack", a.pack, "--max-context", "8192", "--threads", a.threads, "--batch", "3",
           "--adapt-every", "0"]
    if a.expert_profile:
        cmd += ["--expert-profile", a.expert_profile]
    e = Engine(cmd)
    check(e.info.get("batch_slots") == "3" and e.info.get("slot_cache") == "1", f"INFO batch_slots=3 slot_cache=1 ({e.info})")
    prompts = [prompt(200, 1), prompt(57, 2), prompt(130, 3)]
    N = 12
    alone = [e.gen(p, N)[0] for p in prompts]
    for k, t in enumerate(alone):
        print(f"  alone {k}: {t}", flush=True)

    out = {}
    lines = e.admit(0, prompts[0], N, out)
    check([l.split()[0] for l in lines if l.split()[0] in ("RESUME", "T", "DONE", "BADM")] == ["RESUME", "T", "DONE", "BADM"]
          and lines[-1] == "BADM 0 1", f"an admission answers RESUME, PP, T, DONE, BADM 0 1 ({lines[-1]})")
    first = {0: int(next(l for l in lines if l.startswith("T "))[2:])}
    for slot in (1, 2):
        lines = e.admit(slot, prompts[slot], N, out)
        first[slot] = int(next(l for l in lines if l.startswith("T "))[2:])
        check(lines[-1] == f"BADM {slot} 1", f"slot {slot} admitted ({lines[-1]})")
    # the slots decode while nothing is sent: wait for every BDONE
    while sum(1 for k in out for l in out[k] if l.startswith("BDONE")) < 3:
        line = e.get()
        if line.startswith(("BT ", "BDONE ")):
            out.setdefault(int(line.split()[1]), []).append(line)
    for slot in range(3):
        toks = [first[slot]] + slot_tokens(out[slot])
        done = out[slot][-1].split()
        print(f"  slot {slot}: {toks} ({' '.join(done)})", flush=True)
        check(done[0] == "BDONE" and done[2] == str(N) and done[3] == "length", f"slot {slot}: BDONE {slot} {N} length")
        check(toks == alone[slot], f"slot {slot}: the tokens of the request decoded alone")

    # BSTOP: one more BT, then BDONE cancel
    out = {}
    e.admit(1, prompts[1], 200, out)
    while len(slot_tokens(out.get(1, []))) < 3:
        line = e.get()
        if line.startswith(("BT ", "BDONE ")):
            out.setdefault(int(line.split()[1]), []).append(line)
    e.send("BSTOP 1")
    while not any(l.startswith("BDONE") for l in out.get(1, [])):
        line = e.get()
        if line.startswith(("BT ", "BDONE ")):
            out.setdefault(int(line.split()[1]), []).append(line)
    done = out[1][-1].split()
    check(done[3] == "cancel" and len(slot_tokens(out[1])) <= 6, f"BSTOP ends the slot: {' '.join(done)}")

    # a BGEN for a busy slot, and for a slot that does not exist: ERR, then BADM 0
    out = {}
    e.admit(2, prompts[2], 200, out)
    lines = e.admit(2, prompts[0], 4, out)
    check(any(l.startswith("ERR") for l in lines) and lines[-1] == "BADM 2 0", f"a busy slot: ERR, BADM 2 0 ({lines[-1]})")
    lines = e.admit(7, prompts[0], 4, out)
    check(any(l.startswith("ERR") for l in lines) and lines[-1] == "BADM 7 0", f"no slot 7: ERR, BADM 7 0 ({lines[-1]})")
    e.send("BSTOP 2")
    while not any(l.startswith("BDONE") for l in out.get(2, [])):
        line = e.get()
        if line.startswith(("BT ", "BDONE ")):
            out.setdefault(int(line.split()[1]), []).append(line)

    # slot_cache: a solo GEN that continues slot 0's conversation (prompt, its tokens but the last) reuses the slot
    cont = prompts[0] + alone[0][:N - 1] + prompt(11, 9)[1:]
    toks, lines = e.gen(cont, 3)
    resume = next(l for l in lines if l.startswith("RESUME"))
    check(resume == f"RESUME {len(prompts[0]) + N - 1}", f"a later turn continues the slot's conversation ({resume})")

    e.send("QUIT")
    check(e.p.wait(timeout=120) == 0, "QUIT ends the engine with code 0")
    print(f"RESULT {'fail' if failures else 'pass'} serve_batch", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
