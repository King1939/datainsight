"""储藏区：长期保存、额度、同名覆盖、文件归谁。

不碰真目录：每个用例换一个临时 VAULT_DIR，跑完连根删掉。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import vault
from app.readers import DocumentError


class VaultCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="vault-test-"))
        patcher = mock.patch.object(vault, "VAULT_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def add(self, name: str, data: bytes) -> vault.Entry:
        return vault.add(name, data)


class TestAdd(VaultCase):
    def test_document_text_is_extracted_once(self):
        entry = self.add("会议.txt", "今天定了三件事。".encode("utf-8"))

        self.assertEqual(entry.kind, vault.KIND_DOCUMENT)
        self.assertEqual(entry.chars, 8)
        self.assertEqual(entry.parts, 1)
        self.assertTrue(entry.path.exists())
        self.assertTrue(entry.text_path.exists())
        self.assertEqual(vault.usage(), entry.size)

        doc = vault.open_document(entry)
        self.assertEqual(doc.text, "今天定了三件事。")
        self.assertFalse(doc.owned)

    def test_table_is_not_extracted(self):
        entry = self.add("销量.csv", b"city,amount\nA,1\nB,2\n")

        self.assertEqual(entry.kind, vault.KIND_TABLE)
        self.assertFalse(entry.text_path.exists())
        self.assertIsNone(vault.open_document(entry))

        frame = vault.read_table(entry)
        self.assertEqual(list(frame.columns), ["city", "amount"])
        self.assertEqual(len(frame), 2)

    def test_gbk_csv_still_readable(self):
        entry = self.add("gbk.csv", "城市,金额\n上海,10\n".encode("gbk"))
        frame = vault.read_table(entry)
        self.assertEqual(list(frame.columns), ["城市", "金额"])

    def test_same_content_same_name_is_stored_once(self):
        first = self.add("a.txt", b"hello")
        second = self.add("a.txt", b"hello")

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(vault.entries()), 1)
        self.assertEqual(vault.usage(), 5)

    def test_new_version_replaces_old_one(self):
        old = self.add("报告.txt", ("旧" * 40).encode("utf-8"))
        old_path = old.path
        new = self.add("报告.txt", "新版".encode("utf-8"))

        self.assertNotEqual(old.id, new.id)
        self.assertEqual([e.name for e in vault.entries()], ["报告.txt"])
        self.assertEqual(vault.usage(), new.size)
        self.assertFalse(old_path.exists())

    def test_same_content_two_names_share_one_file(self):
        one = self.add("a.txt", b"same")
        two = self.add("b.txt", b"same")

        self.assertNotEqual(one.id, two.id)
        self.assertEqual(one.path, two.path)
        self.assertEqual(len(vault.entries()), 2)
        self.assertEqual(vault.usage(), 4)

    def test_name_cannot_escape_vault(self):
        entry = self.add("../../evil.txt", b"x")
        self.assertEqual(entry.name, "evil.txt")
        self.assertEqual(entry.path.parent, self.root / "files")

    def test_unsupported_suffix_rejected(self):
        with self.assertRaises(DocumentError):
            self.add("a.xyz", b"whatever")
        self.assertEqual(vault.entries(), [])

    def test_empty_file_rejected(self):
        with self.assertRaises(DocumentError):
            self.add("a.txt", b"")

    def test_oversized_file_rejected(self):
        with self.assertRaises(DocumentError) as ctx:
            self.add("big.txt", b"x" * (21 * 1024 * 1024))
        self.assertIn("20 MB", str(ctx.exception))
        self.assertEqual(vault.entries(), [])

    def test_entries_newest_first(self):
        self.add("first.txt", b"1")
        self.add("second.txt", b"2")
        self.assertEqual([e.name for e in vault.entries()][0], "second.txt")


class TestQuota(VaultCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(vault, "VAULT_QUOTA_MB", 1)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_full_vault_refuses_new_file(self):
        self.add("half.txt", b"x" * (600 * 1024))

        with self.assertRaises(vault.VaultFull) as ctx:
            self.add("half2.txt", b"y" * (600 * 1024))

        self.assertIn("1 MB", str(ctx.exception))
        self.assertEqual([e.name for e in vault.entries()], ["half.txt"])
        self.assertEqual(vault.usage(), 600 * 1024)

    def test_replacing_same_name_frees_its_room(self):
        self.add("a.txt", b"x" * (600 * 1024))
        entry = self.add("a.txt", b"z" * (600 * 1024))

        self.assertEqual(vault.usage(), entry.size)
        self.assertEqual(len(vault.entries()), 1)


class TestDrop(VaultCase):
    def test_drop_removes_files_and_index(self):
        entry = self.add("a.txt", b"hello")
        path, text_path = entry.path, entry.text_path

        self.assertTrue(vault.drop(entry.id))

        self.assertFalse(path.exists())
        self.assertFalse(text_path.exists())
        self.assertEqual(vault.entries(), [])
        self.assertEqual(vault.usage(), 0)
        self.assertFalse(vault.drop(entry.id))

    def test_shared_file_survives_dropping_one_name(self):
        one = self.add("a.txt", b"same")
        two = self.add("b.txt", b"same")

        vault.drop(one.id)

        self.assertTrue(two.path.exists())
        self.assertEqual([e.name for e in vault.entries()], ["b.txt"])

    def test_dropping_last_reference_removes_file(self):
        one = self.add("a.txt", b"same")
        two = self.add("b.txt", b"same")

        vault.drop(one.id)
        vault.drop(two.id)

        self.assertFalse(two.path.exists())
        self.assertEqual(vault.entries(), [])


class TestPersistence(VaultCase):
    def test_index_survives_process_restart(self):
        entry = self.add("长期.txt", "留着".encode("utf-8"))

        vault._DOCS.clear()
        again = vault.entries()

        self.assertEqual(len(again), 1)
        self.assertEqual(again[0].id, entry.id)
        self.assertEqual(vault.open_document(again[0]).text, "留着")

    def test_broken_index_does_not_crash(self):
        self.add("a.txt", b"hello")
        (self.root / "index.json").write_text("{not json", encoding="utf-8")

        self.assertEqual(vault.entries(), [])
        self.add("b.txt", b"world")
        self.assertEqual([e.name for e in vault.entries()], ["b.txt"])


if __name__ == "__main__":
    unittest.main()
