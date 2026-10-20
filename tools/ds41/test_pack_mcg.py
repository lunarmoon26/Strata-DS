"""Byte-preserving EXL3 pack tests using synthetic safetensors, without torch/GPU."""
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import pack as P


def checkpoint(directory, codebook="mcg", bits=2, bad_marker=False, both=False, bad_marker_layout=False, missing_marker=False,
               prefix="layers"):
    tensors = {}
    for projection in P.PROJ:
        base = f"{prefix}.0.ffn.experts.0.{projection}."
        tensors[base + "trellis"] = ("I16", [2, 1, bits * 16], bytes(range(bits * 64)))
        tensors[base + "suh"] = ("F16", [32], bytes(range(64)))
        tensors[base + "svh"] = ("F16", [16], bytes(range(32)))
        marker = 0 if bad_marker else {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}[codebook]
        tensors[base + codebook] = ("I32", [], struct.pack("<I", marker))
        if bad_marker_layout:
            tensors[base + codebook] = ("I16", [2], struct.pack("<I", marker))
        if missing_marker:
            del tensors[base + codebook]
        if both:
            tensors[base + "mul1"] = ("I32", [], struct.pack("<I", 0x83DCD12D))
    tensors["embed.weight"] = ("U8", [(4 << 20) + 19], b"\x7b" * ((4 << 20) + 19))
    tensors["layers.0.attn.wq_a.weight"] = ("F8_E4M3", [2, 32], bytes(range(64)))
    tensors["layers.0.attn.wq_a.scale"] = ("F8_E8M0", [1, 1], b"\x7f")
    header, data = {}, bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        offset = len(data)
        data.extend(payload)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, len(data)]}
    encoded = json.dumps(header).encode()
    shard = "model-00001-of-00001.safetensors"
    (directory / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: shard for n in tensors}}))
    return tensors


class PackMcgTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ds41-mcg-", dir=os.environ.get("TMPDIR"))
        self.directory = Path(self.temp.name)
        self.sources = []

    def tearDown(self):
        for source in self.sources:
            for fd in source.fds.values():
                os.close(fd)
        self.temp.cleanup()

    def source(self, **kwargs):
        tensors = checkpoint(self.directory, **kwargs)
        source = P.Source(str(self.directory))
        self.sources.append(source)
        return source, tensors

    def test_mcg_components_round_trip_without_renaming_or_conversion(self):
        source, tensors = self.source()
        P.write_experts(source, str(self.directory), 1, 1)
        row = P.read_experts_txt(str(self.directory / "experts.txt"))[(0, 0)]
        packed = (self.directory / "experts.bin").read_bytes()
        self.assertEqual(row["bits"], dict.fromkeys(P.PROJ, 2))
        for component, (offset, size) in row["components"].items():
            expected = tensors["layers.0.ffn.experts.0." + component][2]
            self.assertEqual(packed[offset:offset + size], expected)
            self.assertEqual(offset % 256, 0)
        self.assertEqual(sum(c.endswith(".mcg") for c in row["components"]), 3)
        self.assertEqual(row["bytes"] % 4096, 0)

    def test_mul1_components_remain_supported_and_unchanged(self):
        source, tensors = self.source(codebook="mul1")
        comps, _, bits = P.expert_layout(source, 0, 0)
        self.assertEqual(sum(c[0].endswith(".mul1") for c in comps), 3)
        self.assertEqual(bits, dict.fromkeys(P.PROJ, 2))

    def test_unknown_marker_is_rejected(self):
        source, _ = self.source(bad_marker=True)
        with self.assertRaisesRegex(ValueError, "unsupported codebook marker"):
            P.expert_layout(source, 0, 0)

    def test_ambiguous_markers_are_rejected(self):
        source, _ = self.source(both=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            P.expert_layout(source, 0, 0)

    def test_mcg_non_k2_is_rejected(self):
        source, _ = self.source(bits=3)
        with self.assertRaisesRegex(ValueError, "K2 only"):
            P.expert_layout(source, 0, 0)

    def test_invalid_marker_storage_is_rejected(self):
        source, _ = self.source(bad_marker_layout=True)
        with self.assertRaisesRegex(ValueError, "invalid codebook marker layout"):
            P.expert_layout(source, 0, 0)

    def test_missing_marker_is_rejected(self):
        source, _ = self.source(missing_marker=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            P.expert_layout(source, 0, 0)

    def test_mcg_mtp_export_is_rejected(self):
        source, _ = self.source(prefix="mtp")
        with self.assertRaisesRegex(ValueError, "backbone experts only"):
            P.expert_layout(source, 0, 0, prefix="mtp")

    def test_dense_upload_source_streams_at_most_four_mib(self):
        source, tensors = self.source()
        requests = []
        pread = os.pread

        def bounded(fd, count, offset):
            requests.append(count)
            return pread(fd, count, offset)

        with mock.patch.object(source, "read", side_effect=AssertionError("tensor-sized read")):
            with mock.patch.object(P.os, "pread", side_effect=bounded):
                P.write_dense(source, str(self.directory))
        self.assertTrue(requests)
        self.assertLessEqual(max(requests), 4 << 20)
        self.assertTrue((self.directory / "dense.bin").read_bytes().startswith(tensors["embed.weight"][2]))

    def test_copy_rejects_zero_chunk(self):
        source, _ = self.source()
        with self.assertRaises(ValueError):
            source.copy_to("embed.weight", io.BytesIO(), 0)

    def test_truncated_source_is_not_silently_packed(self):
        source, _ = self.source()
        os.truncate(self.directory / "model-00001-of-00001.safetensors", 0)
        with self.assertRaisesRegex(IOError, "short read"):
            source.copy_to("embed.weight", io.BytesIO())


if __name__ == "__main__":
    unittest.main()
