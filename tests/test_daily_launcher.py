import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "开始照片分类.cmd"


class DailyLauncherTests(unittest.TestCase):
    def test_launcher_is_ascii_crlf_without_bom(self):
        content = LAUNCHER.read_bytes()
        content.decode("ascii")
        self.assertTrue(content.startswith(b"@echo off\r\n"))
        self.assertNotIn(b"\n", content.replace(b"\r\n", b""))
        self.assertIn(b"%*", content)
        self.assertIn(b"pause\r\n", content)

    @unittest.skipUnless(os.name == "nt", "Windows cmd launcher")
    def test_real_cmd_plan_is_readonly_and_reaches_pause(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "待分类"
            output = root / "output"
            source.mkdir()
            (source / "photo.jpg").write_bytes(b"scan only")
            config = root / "config.json"
            config.write_text(json.dumps({"input_directory": str(source), "output_directory": str(output),
                                          "archive_policy": "cascade_prediction", "cpu_threads": 1}), encoding="utf-8")
            command = f'call "{LAUNCHER}" --config "{config}" --plan --input "{source}" --output "{output}"'
            # Windows list2cmdline's backslash quoting is not cmd.exe syntax.
            result = subprocess.run("cmd.exe /d /c " + command, input="", capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"images": 1', result.stdout)
            self.assertIn('"inference_started": false', result.stdout)
            self.assertIn("The window will remain open", result.stdout)
            self.assertFalse(output.exists())

    @unittest.skipUnless(os.name == "nt", "Windows cmd launcher")
    def test_real_cmd_failure_keeps_error_and_pause(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            command = f'call "{LAUNCHER}" --config "{missing}"'
            result = subprocess.run("cmd.exe /d /c " + command, input="", capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("workflow exited with code 1", result.stdout)
            self.assertIn("The window will remain open", result.stdout)
            self.assertIn("missing.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
