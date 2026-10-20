"""Resident 1017 setup contract. CPU-only; no GPU, downloads or conversions."""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup
from tools import setup_deepseek_1017 as profile


class Installer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "Strata with 'quotes'"
        self.root.mkdir()
        self.cfg = self.root / profile.CONFIG_NAME
        self.launcher = self.root / profile.LAUNCHER_NAME
        self.flags = ["--family", "deepseek", "--model", profile.MODEL, "--configure-only",
                      "--data-dir", str(self.root / "data"), "--gpus", "1,0", "--telemetry-gpus", "0,1"]

    def invoke(self, flags=None, check=True):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(setup, "ROOT", self.root))
            stack.enter_context(patch.object(sys, "argv", ["setup.py", *(self.flags if flags is None else flags)]))
            # Every accidental entry into bootstrap/conversion/update/discovery is
            # a test failure, rather than a harmless mock returning success.
            for name in ("pip_install", "download", "build_deepseek_engine", "setup_deepseek",
                         "deepseek_gpu", "gpus", "update_installed_engine", "update_deepseek_engine"):
                if hasattr(setup, name):
                    stack.enter_context(patch.object(setup, name, side_effect=AssertionError("unexpected " + name)))
            stack.enter_context(patch.object(setup, "data_folder", side_effect=AssertionError("data migration")))
            stack.enter_context(patch.object(setup, "load_settings", side_effect=AssertionError("shared settings")))
            if check:
                stack.enter_context(patch.object(profile, "check"))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(output))
            try:
                code = setup.main()
            except SystemExit as e:
                code = e.code
            self.output = output.getvalue()
            return code

    def install(self, *extra):
        with patch.object(profile, "start", side_effect=AssertionError("configure started a server")):
            self.assertEqual(self.invoke([*self.flags, *extra]), 0, self.output)
        return profile.load_config(self.cfg)

    def test_routing_and_qualified_configuration(self):
        cfg = self.install("--context", "524288", "--port", "8081", "--no-start", "--no-browser")
        self.assertEqual(cfg["model_name"], profile.MODEL_ID)
        self.assertEqual(cfg["cuda_visible_devices"], "1,0")
        self.assertEqual(cfg["telemetry_gpus"], [0, 1])
        args = cfg["args"]
        expected = {"--devices": "0,1", "--split-layer": "18", "--ram-budget-gib": "0",
                    "--max-context": "524288", "--adapt-every": "0", "--spec": "none", "--batch": "0"}
        for flag, value in expected.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn("--rope-scale", args)
        self.assertEqual(Path(cfg["tokenizer"]).name, profile.REVISION)
        self.assertEqual(cfg["host"], "127.0.0.1")
        self.assertNotIn("api_key", cfg)

    def test_smaller_context_and_opt_in_sessions(self):
        cfg = self.install("--parallel", "2", "--spec", "dspark", "--context", "131072")
        self.assertEqual(cfg["parallel"], 2)
        self.assertEqual(cfg["args"][-4:], ["--batch", "2", "--spec", "dspark"])

    def test_tensor_parallel_is_explicit_and_not_a_layer_split(self):
        cfg = self.install("--parallelism", "tensor", "--context", "4096", "--draft-cache-mib", "0")
        args = cfg["args"]
        self.assertEqual(args[args.index("--parallelism") + 1], "tensor")
        self.assertNotIn("--split-layer", args)
        self.assertEqual(args[args.index("--spec") + 1], "none")
        self.assertEqual(args[args.index("--draft-cache-mib") + 1], "0")

    def test_bad_settings_and_unsupported_flags_leave_no_files(self):
        cases = (("--context", "524289"), ("--context", "0"), ("--port", "0"),
                 ("--parallel", "5"), ("--parallel", "0"), ("--gpus", "0,0"),
                 ("--gpus", "all"), ("--telemetry-gpus", "1,1"), ("--telemetry-gpus", "0"),
                 ("--host", "0.0.0.0"), ("--backend", "hip"), ("--rope-scaling", "yarn"),
                 ("--draft-cache-mib", "-1"), ("--draft-cache-mib", "8193"),
                 ("--build",), ("--source", "huggingface"), ("--vision", "yes"), ("--browser",))
        for extra in cases:
            with self.subTest(extra=extra):
                self.assertNotEqual(self.invoke([*self.flags, *extra]), 0)
                self.assertFalse(self.cfg.exists())
                self.assertFalse(self.launcher.exists())

    def test_profile_flags_never_enter_qwen_or_sage(self):
        for flags in (["--configure-only"], ["--family", "qwen", "--configure-only"],
                      ["--family", "deepseek", "--model", "SAGE-1.59BPW", "--configure-only"],
                      ["--family", "qwen", "--spec", "none"],
                      ["--family", "qwen", "--parallelism", "tensor"],
                      ["--family", "qwen", "--draft-cache-mib", "0"]):
            self.assertNotEqual(self.invoke(flags), 0)

    def test_missing_pack_prevents_any_outputs(self):
        self.assertNotEqual(self.invoke(check=False), 0)
        self.assertFalse(self.cfg.exists())
        self.assertFalse(self.launcher.exists())

    def test_asset_metadata_and_independent_engram_checks(self):
        cfg = self.install()
        exe = Path(cfg["exe"])
        exe.parent.mkdir(parents=True)
        exe.touch()
        exe.chmod(0o700)
        model = Path(cfg["tokenizer"])
        model.mkdir(parents=True)
        pack = Path(cfg["args"][1])
        pack.mkdir(parents=True)
        for name in ("dense.bin", "experts.bin", "mtp_experts.bin", "engram.txt", "engram_hash.txt", "index.txt"):
            (pack / name).touch()
        shards = [model / name for name in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")]
        for shard in shards:
            shard.touch()
        info = pack / "pack_info.txt"
        info.write_text(f"source {model}\nfinished 1\n")
        with patch("serve.deepseek.DeepSeekTokenizer") as tokenizer:
            tokenizer.return_value.eos_id = 1
            profile.check(cfg)
            info.write_text("source /wrong/model\nfinished 1\n")
            with self.assertRaisesRegex(ValueError, "pinned sudoingx"):
                profile.check(cfg)
            info.write_text(f"source {model}\nfinished 0\n")
            with self.assertRaisesRegex(ValueError, "unfinished"):
                profile.check(cfg)
            info.write_text(f"source {model}\nfinished 1\n")
            shards[0].unlink()
            shards[0].symlink_to(shards[1])
            with self.assertRaisesRegex(ValueError, "symlink-dependent"):
                profile.check(cfg)

    def test_mismatched_config_and_launcher_fail_before_asset_checks(self):
        self.install()
        self.cfg.write_text(self.cfg.read_text() + "\n")
        with patch.object(profile, "check", side_effect=AssertionError("asset check should not run")), \
                patch.object(sys, "argv", ["helper", "preflight", "--config", str(self.cfg), "--config-sha256", "bad"]), \
                self.assertRaisesRegex(ValueError, "config and launcher differ"):
            profile.main()

    def test_private_idempotent_outputs_backups_and_git_ignores(self):
        self.install()
        before = (self.cfg.stat().st_mtime_ns, self.launcher.stat().st_mtime_ns)
        self.install()
        self.assertEqual(before, (self.cfg.stat().st_mtime_ns, self.launcher.stat().st_mtime_ns))
        self.assertEqual(self.cfg.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.launcher.stat().st_mode & 0o777, 0o700)
        self.install("--port", "8082")
        backups = list(self.root.glob(profile.CONFIG_NAME + ".bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text())["port"], 8081)
        self.assertEqual(backups[0].stat().st_mode & 0o777, 0o600)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        shutil.copyfile(ROOT / ".gitignore", self.root / ".gitignore")
        for path in (self.cfg, self.launcher, backups[0]):
            subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=self.root, check=True)

    def test_migrates_only_recognized_legacy_outputs_with_backups(self):
        self.cfg.write_text(json.dumps({"_generated_by": profile.LEGACY_GENERATOR}))
        self.launcher.write_text(f"#!/usr/bin/env bash\n# Generated by {profile.LEGACY_GENERATOR}; old\n")
        cfg = self.install()
        self.assertEqual(cfg["_generated_by"], profile.GENERATOR)
        self.assertEqual(len(list(self.root.glob("*.bak-*"))), 2)
        self.assertNotIn("deployment/local_serve.py", self.launcher.read_text())

    def test_unowned_files_and_symlinks_are_preserved(self):
        self.launcher.write_text("# user work\n")
        self.assertNotEqual(self.invoke(), 0)
        self.assertEqual(self.launcher.read_text(), "# user work\n")
        self.assertFalse(self.cfg.exists())
        self.launcher.unlink()
        target = self.root / "user-file"
        target.write_text("user work")
        self.launcher.symlink_to(target)
        self.assertNotEqual(self.invoke(), 0)
        self.assertEqual(target.read_text(), "user work")

    def test_saved_start_and_update_do_not_use_sage_updater(self):
        self.install()
        before = self.cfg.read_bytes()
        with patch.object(profile, "start", return_value=73) as start:
            self.assertEqual(self.invoke([]), 73, self.output)
            start.assert_called_once_with(self.cfg)
        self.assertEqual(self.invoke(["--update"]), 0, self.output)
        self.assertEqual(self.cfg.read_bytes(), before)
        with patch.object(profile, "start", return_value=73):
            self.assertEqual(self.invoke(["--family", "deepseek", "--model", profile.MODEL]), 73, self.output)

    def test_start_overrides_are_not_silently_ignored(self):
        self.install()
        with patch.object(profile, "start", side_effect=AssertionError("unexpected start")):
            self.assertNotEqual(self.invoke(["--port", "8082"]), 0)

    def test_generated_launcher_delegates_and_preserves_lock(self):
        cfg = self.install()
        (self.root / "tools").mkdir()
        (self.root / "tools/setup_deepseek_1017.py").write_text(
            "import sys; assert sys.argv[1] == 'preflight'; assert sys.argv[2] == '--config'\n")
        (self.root / "serve").mkdir()
        receipt = self.root / "receipt.json"
        (self.root / "serve/server.py").write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            f"Path({str(receipt)!r}).write_text(json.dumps({{'args': sys.argv[1:], "
            "'cuda': os.environ['CUDA_VISIBLE_DEVICES'], 'cwd': os.getcwd()}))\n")
        before = self.cfg.read_bytes()
        subprocess.run(["bash", "-n", str(self.launcher)], check=True)
        result = subprocess.run([str(self.launcher)], env={**os.environ, "CUDA_VISIBLE_DEVICES": "9"}, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        received = json.loads(receipt.read_text())
        self.assertEqual(received["cuda"], "1,0")
        self.assertEqual(received["cwd"], str(self.root))
        self.assertEqual(received["args"], ["--engine", "strata", "--config", str(self.cfg),
                                            "--host", "127.0.0.1", "--port", "8081"])
        self.assertEqual(self.cfg.read_bytes(), before)
        with (Path(cfg["log"]).parent / "service.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(subprocess.run([str(self.launcher)], capture_output=True).returncode, 73)
        self.assertEqual(subprocess.run([str(self.launcher), "--host", "0.0.0.0"], capture_output=True).returncode, 2)

    def test_occupied_port_preflight_is_read_only(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            cfg = self.install("--port", str(listener.getsockname()[1]))
            before = self.cfg.read_bytes()
            with patch.object(profile, "check"), self.assertRaises(OSError):
                profile.preflight(cfg)
            self.assertEqual(self.cfg.read_bytes(), before)

    def test_shell_offline_route_does_not_repair_or_create_environment(self):
        shutil.copyfile(ROOT / "setup.sh", self.root / "setup.sh")
        (self.root / ".venv/bin").mkdir(parents=True)
        (self.root / ".venv/user-work").write_text("preserve")
        (self.root / "setup.py").write_text("raise SystemExit(42)\n")
        result = subprocess.run(["bash", "setup.sh", *self.flags], cwd=self.root,
                                env={**os.environ, "STRATA_PYTHON": sys.executable}, capture_output=True)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual((self.root / ".venv/user-work").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
