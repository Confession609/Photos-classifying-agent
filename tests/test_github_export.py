import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import export_github_source as publication


class GithubExportTests(unittest.TestCase):
    def test_all_selected_sources_pass_scan(self):
        files = publication.collect_files(publication.ROOT)
        self.assertEqual(len(files), len(set(files)))
        for path in files:
            publication.check_source(path, publication.ROOT)
        self.assertNotIn(publication.ROOT / "PROJECT_MEMORY.md", files)
        self.assertNotIn(publication.ROOT / "daily_workflow_config.json", files)

    def test_export_is_source_only_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "example.py").write_text("value = 1\n", encoding="utf-8")
            (root / "README.md").write_text("Example", encoding="utf-8")
            (root / "private.jpg").write_bytes(b"private photo")
            (root / "model.pt").write_bytes(b"weight")
            with patch.object(publication, "PUBLIC_FILES", ("README.md",)):
                result = publication.export(root, root / "github_upload")
                self.assertEqual(result["files"], 2)
                self.assertFalse((root / "github_upload" / "private.jpg").exists())
                with self.assertRaises(FileExistsError):
                    publication.export(root, root / "github_upload")
                with self.assertRaises(ValueError):
                    publication.export(root, root / "data")

    def test_privacy_scan_refuses_credentials_without_exposing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "unsafe.py"
            secret = "ghp_" + "x" * 24
            source.write_text('token = "' + secret + '"', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                publication.check_source(source, root)
            self.assertNotIn(secret, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
