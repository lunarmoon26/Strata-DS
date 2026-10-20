"""Tests for tools/ds41/pack.py on small synthetic EXL3 checkpoints. No GPU, no downloads.

Run: python tools/ds41/test_pack.py   (or python -m pytest tools/ds41/test_pack.py)
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pack as P  # noqa: E402

SHAPES = {3: {"w1": ([320, 144, 48], [5120], [2304]), "w3": ([320, 144, 48], [5120], [2304]),
              "w2": ([144, 320, 48], [2304], [5120])},
          2: {"w1": ([320, 144, 32], [5120], [2304]), "w3": ([320, 144, 32], [5120], [2304]),
              "w2": ([144, 320, 32], [2304], [5120])}}


def write_checkpoint(d, layers=(0, 1), n_exp=3, bits_of=lambda L, e: 3):
    """Two shards. Expert tensors are split across them; one expert per layer has another bitrate."""
    shards, index, truth = [{}, {}], {}, {}

    def put(name, t, k):
        shards[k][name] = t
        index[name] = f"model-0000{k + 1}-of-00002.safetensors"
        truth[name] = t

    for L in layers:
        for e in range(n_exp):
            for w, (tr, su, sv) in SHAPES[bits_of(L, e)].items():
                base = f"layers.{L}.ffn.experts.{e}.{w}."
                put(base + "trellis", torch.randint(-32768, 32767, tr, dtype=torch.int16), (e + L) % 2)
                put(base + "suh", torch.randn(su).half(), e % 2)
                put(base + "svh", torch.randn(sv).half(), (e + 1) % 2)
                put(base + "mul1", torch.tensor(-2082680531, dtype=torch.int32), 0)
        wo = (torch.randn(64, 96) * 8).clamp(-448, 448).to(torch.float8_e4m3fn)
        put(f"layers.{L}.attn.wo_a.weight", wo, 1)
        put(f"layers.{L}.attn.wo_a.scale",
            (2.0 ** torch.randint(-4, 2, (2, 3)).float()).to(torch.float8_e8m0fnu), 1)
        put(f"layers.{L}.attn.wq_a.weight", (torch.randn(64, 32)).to(torch.float8_e4m3fn), 0)
        put(f"layers.{L}.attn.wq_a.scale", (torch.ones(2, 1)).to(torch.float8_e8m0fnu), 0)
        put(f"layers.{L}.attn.attn_sink", torch.randn(64), 1)
        put(f"layers.{L}.ffn.gate.weight", torch.randn(384, 32).bfloat16(), 0)
    put("embed.weight", torch.randn(100, 32).bfloat16(), 0)
    put("vision.blocks.0.attn.wo.weight", torch.randn(8, 8).bfloat16(), 1)
    for k in range(2):
        save_file(shards[k], os.path.join(d, f"model-0000{k + 1}-of-00002.safetensors"))
    json.dump({"weight_map": index}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        open(os.path.join(d, f), "w").write("{}")
    return truth


def read_index(path):
    rows = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        name, dtype, ndim = f[0], f[1], int(f[2])
        dims = [int(x) for x in f[3:3 + ndim]]
        off, nb = int(f[3 + ndim]), int(f[4 + ndim])
        rows[name] = (dtype, dims, off, nb)
    return rows


class PackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.out = os.path.join(self.tmp.name, "pack")
        os.makedirs(self.src)
        mixed = lambda L, e: 2 if (L, e) == (1, 2) else 3
        self.truth = write_checkpoint(self.src, bits_of=mixed)
        P.build_pack(self.src, self.out, n_layers=2, n_experts=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dense_bytes_round_trip_except_vision_and_experts(self):
        idx = read_index(os.path.join(self.out, "index.txt"))
        blob = np.fromfile(os.path.join(self.out, "dense.bin"), dtype=np.uint8)
        self.assertNotIn("vision.blocks.0.attn.wo.weight", idx)
        self.assertFalse(any(".ffn.experts." in n for n in idx))
        for name in ("layers.0.attn.wq_a.weight", "layers.1.attn.wq_a.scale", "layers.0.attn.attn_sink",
                     "layers.1.ffn.gate.weight", "embed.weight"):
            dtype, dims, off, nb = idx[name]
            want = self.truth[name].contiguous().view(torch.uint8).numpy().reshape(-1)
            self.assertEqual(off % P.DENSE_ALIGN, 0, name)
            self.assertEqual(dims, list(self.truth[name].shape), name)
            np.testing.assert_array_equal(blob[off:off + nb], want, err_msg=name)

    def test_wo_a_stays_fp8_with_its_scales(self):
        # the default: the engine dequantizes wo_a on the fly (half the bytes of BF16 per decode step)
        idx = read_index(os.path.join(self.out, "index.txt"))
        blob = np.fromfile(os.path.join(self.out, "dense.bin"), dtype=np.uint8)
        for name, dtype in (("layers.0.attn.wo_a.weight", "f8e4m3"), ("layers.0.attn.wo_a.scale", "e8m0")):
            got_dtype, dims, off, nb = idx[name]
            self.assertEqual((got_dtype, dims), (dtype, list(self.truth[name].shape)), name)
            want = self.truth[name].contiguous().view(torch.uint8).numpy().reshape(-1)
            np.testing.assert_array_equal(blob[off:off + nb], want, err_msg=name)

    def test_dense_only_rewrites_dense_and_keeps_the_experts(self):
        experts = os.path.join(self.out, "experts.bin")
        before = (os.path.getsize(experts), os.stat(experts).st_mtime_ns)
        P.build_pack(self.src, self.out, n_layers=2, n_experts=3, wo_a_fp8=False, dense_only=True)
        self.assertEqual((os.path.getsize(experts), os.stat(experts).st_mtime_ns), before)
        idx = read_index(os.path.join(self.out, "index.txt"))
        self.assertEqual(idx["layers.0.attn.wo_a.weight"][0], "bf16")
        info = open(os.path.join(self.out, "pack_info.txt")).read()
        self.assertIn("finished 1", info)
        self.assertIn(f"dense_bytes {os.path.getsize(os.path.join(self.out, 'dense.bin'))}", info)

    def test_wo_a_is_dequantized_to_bf16_like_convert_py(self):
        P.build_pack(self.src, self.out, n_layers=2, n_experts=3, wo_a_fp8=False)
        idx = read_index(os.path.join(self.out, "index.txt"))
        self.assertNotIn("layers.0.attn.wo_a.scale", idx)
        dtype, dims, off, nb = idx["layers.0.attn.wo_a.weight"]
        self.assertEqual((dtype, dims), ("bf16", [64, 96]))
        blob = np.fromfile(os.path.join(self.out, "dense.bin"), dtype=np.uint8)
        got = torch.from_numpy(blob[off:off + nb].copy()).view(torch.bfloat16).view(64, 96)
        w = self.truth["layers.0.attn.wo_a.weight"].float()
        s = self.truth["layers.0.attn.wo_a.scale"].float()
        want = (w * s.repeat_interleave(32, 0).repeat_interleave(32, 1)).bfloat16()
        self.assertTrue(torch.equal(got, want))

    def test_every_expert_component_round_trips_and_is_aligned(self):
        rows = P.read_experts_txt(os.path.join(self.out, "experts.txt"))
        self.assertEqual(len(rows), 2 * 3)
        blob = np.fromfile(os.path.join(self.out, "experts.bin"), dtype=np.uint8)
        for (L, e), r in rows.items():
            self.assertEqual(r["offset"] % P.EXPERT_ALIGN, 0)
            self.assertEqual(r["bytes"] % P.EXPERT_ALIGN, 0)
            for comp, (coff, cnb) in r["components"].items():
                self.assertEqual(coff % P.COMPONENT_ALIGN, 0)
                w, part = comp.split(".")
                want = self.truth[f"layers.{L}.ffn.experts.{e}.{w}.{part}"].contiguous().reshape(-1).view(torch.uint8).numpy().reshape(-1)
                a = r["offset"] + coff
                np.testing.assert_array_equal(blob[a:a + cnb], want, err_msg=f"{L}.{e}.{comp}")
        # the 2-bit expert is smaller than its 3-bit neighbours, and the file records that
        self.assertLess(rows[(1, 2)]["bytes"], rows[(1, 1)]["bytes"])
        self.assertEqual(rows[(1, 2)]["bits"], {"w1": 2, "w3": 2, "w2": 2})

    def test_pack_is_marked_finished_last(self):
        info = open(os.path.join(self.out, "pack_info.txt")).read()
        self.assertIn("finished 1", info)
        for f in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            self.assertTrue(os.path.exists(os.path.join(self.out, f)))


class MtpExpertsTest(unittest.TestCase):
    """DSpark's routed experts (mtp.*) from exllamav3/requant_mtp_part*.safetensors into mtp_experts.bin."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.out = os.path.join(self.tmp.name, "pack")
        os.makedirs(os.path.join(self.src, "exllamav3"))
        write_checkpoint(self.src)
        P.build_pack(self.src, self.out, n_layers=2, n_experts=3)
        self.truth = {}
        parts = [{}, {}]
        for S in range(2):
            for e in range(3):
                for w, (tr, su, sv) in SHAPES[3].items():
                    base = f"mtp.{S}.ffn.experts.{e}.{w}."
                    for k, (part, t) in enumerate((("trellis", torch.randint(-32768, 32767, tr, dtype=torch.int16)),
                                                   ("suh", torch.randn(su).half()), ("svh", torch.randn(sv).half()),
                                                   ("mul1", torch.tensor(-2082680531, dtype=torch.int32)))):
                        parts[(S + e + k) % 2][base + part] = t
                        self.truth[base + part] = t
            # requantized attention lives in the same files and is not an expert
            parts[S][f"mtp.{S}.attn.wkv.trellis"] = torch.zeros(320, 32, 48, dtype=torch.int16)
        parts[0]["__align_pad__.requant_mtp_part00.0"] = torch.zeros(60, dtype=torch.uint8)
        for k in range(2):
            save_file(parts[k], os.path.join(self.src, "exllamav3", f"requant_mtp_part0{k}.safetensors"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_mtp_expert_component_round_trips(self):
        n = P.add_mtp_experts(self.src, self.out, n_stages=2, n_experts=3)
        self.assertEqual(n, 6)
        rows = P.read_experts_txt(os.path.join(self.out, "mtp_experts.txt"))
        self.assertEqual(sorted(rows), [(S, e) for S in range(2) for e in range(3)])
        blob = np.fromfile(os.path.join(self.out, "mtp_experts.bin"), dtype=np.uint8)
        for (S, e), r in rows.items():
            self.assertEqual(r["offset"] % P.EXPERT_ALIGN, 0)
            self.assertEqual(r["bits"], {"w1": 3, "w3": 3, "w2": 3})
            for comp, (coff, cnb) in r["components"].items():
                w, part = comp.split(".")
                want = self.truth[f"mtp.{S}.ffn.experts.{e}.{w}.{part}"].contiguous().reshape(-1).view(torch.uint8)
                a = r["offset"] + coff
                np.testing.assert_array_equal(blob[a:a + cnb], want.numpy().reshape(-1), err_msg=f"{S}.{e}.{comp}")

    def test_the_pack_stays_finished_and_records_the_mtp_experts(self):
        experts = os.path.join(self.out, "experts.bin")
        before = os.stat(experts).st_mtime_ns
        P.add_mtp_experts(self.src, self.out, n_stages=2, n_experts=3)
        self.assertEqual(os.stat(experts).st_mtime_ns, before)
        info = open(os.path.join(self.out, "pack_info.txt")).read().splitlines()
        self.assertEqual(info[-1], "finished 1")
        self.assertIn("mtp_stages 2", info)
        self.assertIn("mtp_experts 3", info)
        self.assertIn(f"mtp_expert_bytes {os.path.getsize(os.path.join(self.out, 'mtp_experts.bin'))}", info)
        with self.assertRaises(SystemExit): # source-equal artifacts must not be overwritten accidentally
            P.add_mtp_experts(self.src, self.out, n_stages=2, n_experts=3)
        info = open(os.path.join(self.out, "pack_info.txt")).read().splitlines()
        self.assertEqual(sum(l.startswith("mtp_stages") for l in info), 1)

    def test_needs_a_finished_pack(self):
        os.remove(os.path.join(self.out, "pack_info.txt"))
        with self.assertRaises(SystemExit):
            P.add_mtp_experts(self.src, self.out, n_stages=2, n_experts=3)


class EngramLocationTest(unittest.TestCase):
    """Engram tables in engrams/*.safetensors (the 3bpw pack) or in a shard of the index (SAGE 1.59bpw)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.out = os.path.join(self.tmp.name, "pack")
        os.makedirs(self.src)
        os.makedirs(self.out)
        write_checkpoint(self.src)
        self.tables = {L: (torch.randn(rows, 16).to(torch.float8_e4m3fn), torch.ones(rows, 1).to(torch.float8_e8m0fnu))
                       for L, rows in ((1, 7), (14, 5))}

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, paths):
        lines = [l.split() for l in open(os.path.join(self.out, "engram.txt")) if not l.startswith("#")]
        self.assertEqual([int(f[0]) for f in lines], [1, 14])
        for f in lines:
            L, rows, dim, woff, soff, path = int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), f[5]
            w, s = self.tables[L]
            self.assertEqual((rows, dim), tuple(w.shape))
            self.assertEqual(path, paths[L])
            blob = open(path, "rb").read()
            self.assertEqual(blob[woff:woff + w.numel()], w.view(torch.uint8).numpy().tobytes())
            self.assertEqual(blob[soff:soff + s.numel()], s.view(torch.uint8).numpy().tobytes())

    def test_tables_in_the_engrams_directory(self):
        d = os.path.join(self.src, "engrams")
        os.makedirs(d)
        paths = {}
        for L, (w, s) in self.tables.items():
            paths[L] = os.path.abspath(os.path.join(d, f"engram-layer-{L:02d}.safetensors"))
            save_file({f"layers.{L}.engram.embed.weight": w, f"layers.{L}.engram.embed.scale": s}, paths[L])
        self.assertEqual(P.write_engram(self.src, self.out), 2)
        self.check(paths)

    def test_tables_in_index_shards(self):
        idx_path = os.path.join(self.src, "model.safetensors.index.json")
        idx = json.load(open(idx_path))
        paths = {}
        for k, (L, (w, s)) in enumerate(self.tables.items()):
            fname = f"model-0000{k + 3}-of-00004.safetensors"
            paths[L] = os.path.abspath(os.path.join(self.src, fname))
            save_file({f"layers.{L}.engram.embed.weight": w, f"layers.{L}.engram.embed.scale": s}, paths[L])
            idx["weight_map"][f"layers.{L}.engram.embed.weight"] = fname
            idx["weight_map"][f"layers.{L}.engram.embed.scale"] = fname
        json.dump(idx, open(idx_path, "w"))
        self.assertEqual(P.write_engram(self.src, self.out), 2)
        self.check(paths)
        # the dense arena does not take the tables
        P.build_pack(self.src, os.path.join(self.tmp.name, "pack2"), n_layers=2, n_experts=3)
        names = read_index(os.path.join(self.tmp.name, "pack2", "index.txt"))
        self.assertFalse(any(".engram.embed." in n for n in names))


class FakeBackend:
    """Just enough of a tokenizers backend for build_compressed_token_map."""

    def __init__(self, words):
        self.words = words

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.words[i] for i in ids)

    def id_to_token(self, i):
        return self.words[i]


class FakeTokenizer:
    def __init__(self, words):
        self.backend_tokenizer = FakeBackend(words)
        self.words = words

    def __len__(self):
        return len(self.words)


class EngramHashTest(unittest.TestCase):
    """The exported tables plus engram_hash_reference must reproduce DeepSeek's NgramHashState exactly."""

    def test_export_reproduces_official_hashes(self):
        sys.path.insert(0, os.path.join(HERE, "..", "..", "ds41", "proto", "ref"))
        import engram as EG
        words = ["<pad>", " The", "the", "THE", " cat", "Cat", "\n", "  ", "猫", "貓", "ｃａｔ", "x"] * 3
        tok = FakeTokenizer(words)
        _, vocab = EG.build_compressed_token_map(tok)

        class Args:
            engram_layer_ids = (1, 14)
            engram_num_embeddings = (1000003, 1000033)
            engram_max_ngram_size = 4
            engram_vocab_size = 10007
            engram_n_heads = 8
            engram_head_dim = 256
            engram_pad_id = 0
            engram_compressed_vocab_size = vocab
            max_batch_size = 1
            max_seq_len = 64

        layout = EG.EngramLayout.from_args(Args)
        official = EG.NgramHashState(Args, layout, tok)
        with tempfile.TemporaryDirectory() as d:
            P.write_engram_hash(Args, tok, d)
            tables = P.read_engram_hash(d)
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, len(words), (1, 40), generator=g)
        want = official(ids, 0)[0].numpy()                       # [L, n_engram_layers, 24]
        got = P.engram_hash_reference(ids[0].tolist(), tables)
        np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
