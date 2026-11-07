"""对话历史：存、读、列表、删除、索引自愈。

不碰真目录：每个用例换一个临时 CHAT_DIR，跑完连根删掉。
重点是"索引坏了历史不能跟着没"和"id 来自 URL，不能让它跳目录"这两条。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import chats


def _turn(question: str, answer: str = "答案") -> dict:
    return {"question": question, "answer": answer, "trace": "", "charts": []}


class ChatCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="chats-test-"))
        for name, value in (("CHAT_DIR", self.root),):
            patcher = mock.patch.object(chats, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def save(self, chat_id: str, **kwargs) -> chats.Chat:
        kwargs.setdefault("turns", [_turn("问一句")])
        return chats.save(chat_id, **kwargs)


class TestSaveLoad(ChatCase):
    def test_roundtrip(self):
        self.save(
            "c1",
            title="销售额排名",
            table={"kind": "vault", "id": "abc", "name": "sales.csv"},
            turns=[_turn("各城市排名"), _turn("画个图")],
            messages=[{"role": "user", "content": "各城市排名"}],
            stats={"turns": 2, "steps": 3, "tokens": 900, "tool_calls": 1},
        )
        chat = chats.load("c1")
        self.assertIsNotNone(chat)
        self.assertEqual(chat.title, "销售额排名")
        self.assertEqual(chat.count, 2)
        self.assertEqual(chat.table["id"], "abc")
        self.assertEqual(chat.messages[0]["content"], "各城市排名")
        self.assertEqual(chat.stats["tokens"], 900)

    def test_load_missing_returns_none(self):
        self.assertIsNone(chats.load("nope"))

    def test_created_at_kept_updated_at_moves(self):
        first = self.save("c1", title="一")
        second = self.save("c1", title="二")
        self.assertEqual(first.created_at, second.created_at)
        self.assertGreaterEqual(second.updated_at, first.updated_at)
        self.assertEqual(chats.load("c1").title, "二")

    def test_empty_title_gets_placeholder(self):
        chat = self.save("c1", title="")
        self.assertTrue(chat.title)

    def test_save_overwrites_not_appends(self):
        self.save("c1", turns=[_turn("一"), _turn("二")])
        self.save("c1", turns=[_turn("三")])
        self.assertEqual(chats.load("c1").count, 1)

    def test_broken_content_file_is_ignored(self):
        self.save("c1")
        (self.root / "c1.json").write_text("{ 不是 json", encoding="utf-8")
        self.assertIsNone(chats.load("c1"))


class TestList(ChatCase):
    def test_summary_has_no_payload(self):
        self.save("c1", title="标题", table={"kind": "vault", "id": "x", "name": "a.csv"})
        item = chats.list_chats()[0]
        self.assertEqual(item["title"], "标题")
        self.assertEqual(item["count"], 1)
        self.assertEqual(item["table"], "a.csv")
        self.assertNotIn("turns", item)
        self.assertNotIn("messages", item)

    def test_newest_first(self):
        self.save("c1", title="旧")
        self.save("c2", title="新")
        titles = [i["title"] for i in chats.list_chats()]
        self.assertEqual(titles[0], "新")
        self.assertEqual(len(titles), 2)

    def test_limit(self):
        for index in range(5):
            self.save(f"c{index}", title=f"对话{index}")
        self.assertEqual(len(chats.list_chats(limit=3)), 3)

    def test_missing_index_is_rebuilt(self):
        self.save("c1", title="还在")
        (self.root / "index.json").unlink()
        items = chats.list_chats()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "还在")
        # 重建之后索引文件应该真的写回去了
        self.assertTrue((self.root / "index.json").exists())

    def test_corrupt_index_is_rebuilt(self):
        self.save("c1", title="还在")
        (self.root / "index.json").write_text("not json at all", encoding="utf-8")
        self.assertEqual(chats.list_chats()[0]["title"], "还在")

    def test_entry_without_file_is_dropped(self):
        """索引说有、文件没了（手动删的/删到一半）—— 列表里不能留着。
        留着的话用户点进去是一张空对话，比不显示更让人困惑。"""
        self.save("c1", title="还在")
        self.save("c2", title="文件没了")
        (self.root / "c2.json").unlink()

        self.assertEqual([i["title"] for i in chats.list_chats()], ["还在"])
        index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(index["chats"]), 1)

    def test_empty_vault(self):
        self.assertFalse(chats.list_chats())


class TestDelete(ChatCase):
    def test_delete_removes_file_and_entry(self):
        self.save("c1")
        self.assertTrue(chats.delete("c1"))
        self.assertFalse((self.root / "c1.json").exists())
        self.assertIsNone(chats.load("c1"))
        self.assertEqual(chats.list_chats(), [])

    def test_delete_missing_is_false(self):
        self.assertFalse(chats.delete("nope"))

    def test_delete_one_of_two_keeps_other(self):
        self.save("c1", title="A")
        self.save("c2", title="B")
        chats.delete("c1")
        self.assertEqual([i["title"] for i in chats.list_chats()], ["B"])


class TestIds(ChatCase):
    def test_url_style_id_cannot_escape(self):
        """id 来自 ?c=，是用户能改的。不能让它写到目录外面去。"""
        saved = self.save("../../etc/passwd", title="恶意")
        self.assertNotIn("..", saved.id)
        self.assertNotIn("/", saved.id)
        self.assertTrue((self.root / f"{saved.id}.json").exists())
        self.assertFalse((self.root.parent.parent / "etc").exists())

    def test_load_with_traversal_returns_none(self):
        self.assertIsNone(chats.load("../../.env"))

    def test_new_id_is_unique(self):
        ids = {chats.new_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)


class TestPrune(ChatCase):
    def test_keeps_only_limit(self):
        for index in range(chats.KEEP + 3):
            self.save(f"c{index}", title=f"对话{index}")
        items = chats.list_chats()
        self.assertEqual(len(items), chats.KEEP)
        files = [p for p in self.root.glob("*.json") if p.name != "index.json"]
        self.assertEqual(len(files), chats.KEEP)

    def test_prune_keeps_newest(self):
        for index in range(chats.KEEP + 2):
            self.save(f"c{index}", title=f"对话{index}")
        titles = {i["title"] for i in chats.list_chats()}
        self.assertIn(f"对话{chats.KEEP + 1}", titles)
        self.assertNotIn("对话0", titles)


class TestIndexShape(ChatCase):
    def test_index_has_version(self):
        self.save("c1")
        data = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(data["version"], chats.INDEX_VERSION)
        self.assertEqual(len(data["chats"]), 1)


if __name__ == "__main__":
    unittest.main()
