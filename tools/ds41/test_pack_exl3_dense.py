"""Full-EXL3 dense bytes and fail-closed validation, no torch or GPU."""
import struct
import tempfile
import unittest
from pathlib import Path

import pack as P


class Source:
    def __init__(self, bits=4):
        self.payload = {
            'head.trellis': bytes((i % 256 for i in range(8 * 8 * 16 * bits * 2))),
            'head.suh': bytes(range(256)), 'head.svh': bytes(reversed(range(256))),
            'head.mul1': struct.pack('<I', 0x83DCD12D),
        }
        self.entries = {
            'head.trellis': ('f', 'I16', [8, 8, 16 * bits], 0, len(self.payload['head.trellis'])),
            'head.suh': ('f', 'F16', [128], 0, 256), 'head.svh': ('f', 'F16', [128], 0, 256),
            'head.mul1': ('f', 'I32', [], 0, 4),
        }

    def read(self, name):
        return self.payload[name]

    def copy_to(self, name, output):
        output.write(self.payload[name])


class DenseExl3Test(unittest.TestCase):
    def test_copy_all_components_unchanged(self):
        for bits in (1, 2, 3, 4, 5, 6):
            s = Source(bits)
            with tempfile.TemporaryDirectory() as directory:
                self.assertEqual(P.validate_exl3_dense(s), 1)
                count, _ = P.write_dense(s, directory)
                self.assertEqual(count, 4)
                data = (Path(directory) / 'dense.bin').read_bytes()
                for line in (Path(directory) / 'index.txt').read_text().splitlines()[1:]:
                    fields = line.split(); name = fields[0]; off, size = map(int, fields[-2:])
                    self.assertEqual(off % 256, 0)
                    self.assertEqual(data[off:off + size], s.payload[name])

    def test_reject_marker_changes(self):
        for value in (0, 0xCBAC1FED):
            s = Source(); s.payload['head.mul1'] = struct.pack('<I', value)
            with self.assertRaisesRegex(ValueError, 'codebook'): P.validate_exl3_dense(s)

    def test_k8_requires_explicit_profile_and_preserves_bytes(self):
        for bits in (7, 8):
            s = Source(bits)
            with self.assertRaisesRegex(ValueError, 'geometry'):
                P.validate_exl3_dense(s)
            self.assertEqual(P.validate_exl3_dense(s, 'libertaidai-29'), 1)
            with tempfile.TemporaryDirectory() as directory:
                P.write_dense(s, directory, profile='libertaidai-29')
                blob = (Path(directory) / 'dense.bin').read_bytes()
                for line in (Path(directory) / 'index.txt').read_text().splitlines()[1:]:
                    name, *_, off, size = line.split()
                    self.assertEqual(blob[int(off):int(off) + int(size)], s.payload[name])

    def test_profile_rejects_out_of_range_rate(self):
        with self.assertRaisesRegex(ValueError, 'geometry'):
            P.validate_exl3_dense(Source(9), 'libertaidai-29')

    def test_dense_only_cannot_strip_experimental_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'pack_info.txt'
            original = 'profile libertaidai-29\nfinished 1\n'
            marker.write_text(original)
            with self.assertRaisesRegex(ValueError, 'profile gate'):
                P.build_pack('/nonexistent-source', directory, dense_only=True)
            self.assertEqual(marker.read_text(), original)

    def test_reject_half_rates_and_bad_geometry(self):
        for shape in ([8, 8, 72], [8, 7, 64], [8, 8, 0], [8, 8], [-8, 8, 64]):
            s = Source(); e = list(s.entries['head.trellis']); e[2] = shape
            s.entries['head.trellis'] = tuple(e)
            with self.assertRaisesRegex(ValueError, 'geometry'): P.validate_exl3_dense(s)

    def test_reject_missing_component(self):
        s = Source(); del s.entries['head.svh']
        with self.assertRaisesRegex(ValueError, 'component'): P.validate_exl3_dense(s)

    def test_reject_ambiguous_marker(self):
        s = Source(); s.entries['head.mcg'] = s.entries['head.mul1']
        with self.assertRaisesRegex(ValueError, 'codebook'): P.validate_exl3_dense(s)

    def test_reject_wo_a_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'wo-a-bf16'): P.write_dense(Source(), directory, False)


if __name__ == '__main__':
    unittest.main()
