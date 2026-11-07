"""对话历史的界面测试：URL 里带 ?c=<id>，页面能不能把历史读回来。

为什么单独一个文件：这一组只碰"刷新之后历史还在"这一条链路。
模块层的存取逻辑在 tests/test_chats.py，这里验的是接线有没有接对——
磁盘上有文件、URL 里有 id，但页面不显示，功能照样是坏的。

AppTest 不能驱动 file_uploader，所以对话直接写进临时 CHAT_DIR，
不经过上传框。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _isolation import isolate_uploads

ROOT = Path(__file__).resolve().parent.parent

try:
    from streamlit.testing.v1 import AppTest

    from app import chats

    HAS_APPTEST = True
except Exception:  # noqa: BLE001  没有 streamlit 就整组跳过
    HAS_APPTEST = False

TURN = {
    "question": "各城市销售额排名前五",
    "answer": "上海 24.2 万，北京 18.2 万……",
    "trace": "第 1 步：要调用 run_sql",
    "charts": [],
}


def setUpModule() -> None:
    """AppTest 会把 app.py 整个跑一遍，入口里有一次"清掉上个进程遗留"的上传区扫描。

    不隔开的话，跑一次测试就会把用户真实的临时上传删掉——测试不能有这种行为。
    """
    isolate_uploads()


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class ChatHistoryCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="chats-ui-"))
        patcher = mock.patch.object(chats, "CHAT_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def app(self, chat_id: str | None = None) -> AppTest:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        if chat_id:
            at.query_params["c"] = chat_id
        at.run()
        return at


class TestRestore(ChatHistoryCase):
    def test_saved_turns_show_up_after_reload(self):
        chats.save(
            "20261103-120000-abcd",
            title="城市排名",
            turns=[TURN],
            messages=[{"role": "user", "content": TURN["question"]}],
        )

        at = self.app("20261103-120000-abcd")

        self.assertEqual(len(at.exception), 0, list(at.exception))
        texts = [m.value for m in at.markdown]
        self.assertTrue(any(TURN["question"] in t for t in texts), texts)
        self.assertTrue(any("24.2 万" in t for t in texts), texts)

    def test_title_shows_up_in_sidebar(self):
        chats.save("c-title", title="城市排名", turns=[TURN])
        at = self.app("c-title")
        texts = [m.value for m in at.markdown]
        texts += [c.value for c in at.sidebar.caption]
        texts += [b.label for b in at.sidebar.button]
        self.assertTrue(any("城市排名" in t for t in texts), texts)

    def test_unknown_id_opens_empty_page(self):
        at = self.app("20200101-000000-xxxx")
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertFalse(any(TURN["question"] in m.value for m in at.markdown))

    def test_history_lists_other_chats(self):
        chats.save("old-1", title="旧对话甲", turns=[TURN])
        chats.save("old-2", title="旧对话乙", turns=[TURN])
        at = self.app("old-2")
        labels = [b.label for b in at.sidebar.button]
        self.assertTrue(any("旧对话甲" in x for x in labels), labels)


class TestFreshVisit(ChatHistoryCase):
    def test_first_visit_puts_id_in_url(self):
        """第一次来没有 ?c=，也要生成一个并写进 URL——否则刷新就找不回来了。"""
        at = self.app()
        self.assertEqual(len(at.exception), 0, list(at.exception))
        self.assertTrue(at.query_params.get("c"), "URL 里没有对话 id")

    def test_first_visit_does_not_write_a_file(self):
        self.app()
        files = [p for p in self.root.glob("*.json") if p.name != "index.json"]
        self.assertEqual(files, [], "还没有对话内容，不该先占一个文件")


if __name__ == "__main__":
    unittest.main()
