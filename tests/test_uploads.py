"""临时上传区的回收：sweep / discard / drop_table。

判据就一条：**会话结束后，这个目录里不该留下任何还有人要读的东西**。

这里最容易写错的两个地方，所以单独有测试：
  - sweep 用 mtime 判断遗留，一不小心就把本进程刚写的文件也删了；
  - discard 删得太狠，把别的会话或储藏区的文件连带清掉。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import uploads  # noqa: E402
from app.tools import data_tools as dt  # noqa: E402


class _Base(unittest.TestCase):
    """把 UPLOAD_DIR 换成临时目录。

    data_tools 是通过 `uploads.table_path()` 拼路径的，所以这里改的
    `uploads.UPLOAD_DIR` 对生产代码同样生效——这也顺带证明了路径规则只有一份。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._patch = mock.patch.object(uploads, "UPLOAD_DIR", self.root)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def make_csv(self, key: str, body: str = "城市,销售额\n甲,10\n乙,20\n") -> Path:
        path = self.root / f"{key}.csv"
        path.write_text(body, encoding="utf-8")
        return path

    def make_folder(self, key: str, name: str = "笔记.txt") -> Path:
        folder = self.root / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("现场记录", encoding="utf-8")
        return folder

    def age(self, path: Path) -> None:
        """把 mtime 推到"比本进程启动还早"，模拟上一个进程留下的。"""
        old = time.time() - 3600
        os.utime(path, (old, old))

    def names(self) -> set[str]:
        return {p.name for p in self.root.iterdir()}


class TestDiscard(_Base):
    def test_removes_csv_and_folder(self):
        self.make_csv("aaaa1111")
        self.make_folder("aaaa1111")

        items, freed = uploads.discard("aaaa1111")

        self.assertEqual(items, 2)
        self.assertGreater(freed, 0)
        self.assertEqual(self.names(), set())

    def test_unknown_key_is_silent(self):
        items, freed = uploads.discard("never-existed")
        self.assertEqual((items, freed), (0, 0))

    def test_other_sessions_untouched(self):
        self.make_csv("aaaa1111")
        self.make_csv("bbbb2222")
        self.make_folder("bbbb2222")

        uploads.discard("aaaa1111")

        self.assertEqual(self.names(), {"bbbb2222.csv", "bbbb2222"})

    def test_folder_only_session(self):
        """只传了资料没传表：删目录，不该因为缺 csv 就报错。"""
        self.make_folder("cccc3333")
        items, _ = uploads.discard("cccc3333")
        self.assertEqual(items, 1)
        self.assertEqual(self.names(), set())

    def test_safe_key_strips_separators(self):
        """带路径成分的 key 要落到自己名下，不能删到目录外面去。"""
        self.make_csv("aaaa1111")
        uploads.discard("../../aaaa1111")
        # 过滤后是 "______aaaa1111"，和真实 key 不是同一个，所以原文件还在
        self.assertEqual(self.names(), {"aaaa1111.csv"})


class TestDropTable(_Base):
    def test_keeps_documents(self):
        self.make_csv("dddd4444")
        self.make_folder("dddd4444")

        items, _ = uploads.drop_table("dddd4444")

        self.assertEqual(items, 1)
        self.assertEqual(self.names(), {"dddd4444"})

    def test_no_csv_is_noop(self):
        self.make_folder("eeee5555")
        self.assertEqual(uploads.drop_table("eeee5555"), (0, 0))
        self.assertEqual(self.names(), {"eeee5555"})


class TestSweep(_Base):
    def test_removes_orphans_from_previous_process(self):
        csv = self.make_csv("aaaa1111")
        folder = self.make_folder("bbbb2222")
        self.age(csv)
        self.age(folder)

        items, freed = uploads.sweep()

        self.assertEqual(items, 2)
        self.assertGreater(freed, 0)
        self.assertEqual(self.names(), set())

    def test_keeps_live_key_even_if_old(self):
        """活会话的文件一定留，哪怕 mtime 看着很旧（沙箱子进程写的内容不会刷新目录）。"""
        csv = self.make_csv("aaaa1111")
        self.age(csv)

        items, _ = uploads.sweep(keep={"aaaa1111"})

        self.assertEqual(items, 0)
        self.assertIn("aaaa1111.csv", self.names())

    def test_keeps_files_written_by_this_process(self):
        """本进程刚写的不能删：另一个标签页可能正在用它。"""
        self.make_csv("aaaa1111")
        items, _ = uploads.sweep()
        self.assertEqual(items, 0)
        self.assertIn("aaaa1111.csv", self.names())

    def test_ignores_foreign_files(self):
        """不是 <key>.csv 也不是目录的东西，是别人放的，一个都不动。"""
        (self.root / "README.md").write_text("手写的说明", encoding="utf-8")
        (self.root / "leftover.tmp").write_text("x", encoding="utf-8")
        self.age(self.root / "README.md")
        self.age(self.root / "leftover.tmp")

        uploads.sweep()

        self.assertEqual(self.names(), {"README.md", "leftover.tmp"})

    def test_empty_dir_is_fine(self):
        self.assertEqual(uploads.sweep(), (0, 0))


class TestUsage(_Base):
    def test_counts_entries_and_bytes(self):
        self.make_csv("aaaa1111")
        self.make_folder("bbbb2222")

        items, total = uploads.usage()

        self.assertEqual(items, 2)
        self.assertGreater(total, 10)

    def test_empty(self):
        self.assertEqual(uploads.usage(), (0, 0))


class TestWiring(_Base):
    """生产代码写盘的位置必须和回收时找的位置一致，否则 sweep 永远扫不到。"""

    def test_bind_dataset_lands_on_table_path(self):
        import pandas as pd

        frame = pd.DataFrame({"城市": ["甲", "乙"], "销售额": [10, 20]})
        try:
            dt.bind_dataset(frame, key="ut_sweep", label="上传.csv")
            self.assertTrue(uploads.table_path("ut_sweep").exists())

            items, _ = uploads.discard("ut_sweep")
            self.assertEqual(items, 1)
            self.assertFalse(uploads.table_path("ut_sweep").exists())
        finally:
            dt.unbind_dataset("ut_sweep")

    def test_add_document_lands_in_doc_folder(self):
        try:
            doc = dt.add_document("现场笔记.txt", "温度 23 度".encode("utf-8"), key="ut_sweep")
            self.assertEqual(Path(doc.path).parent, uploads.doc_folder("ut_sweep"))

            items, _ = uploads.discard("ut_sweep")
            self.assertEqual(items, 1)
            self.assertFalse(uploads.doc_folder("ut_sweep").exists())
        finally:
            dt.clear_documents("ut_sweep")


class TestIsolationGuard(unittest.TestCase):
    """跑测试不能碰真实的 data/uploads——这靠每个模块自己记得隔离。

    这条是给"以后又加了一个界面测试"准备的：新模块用了 AppTest 却没调
    isolate_uploads()，跑一遍就会删掉用户真实的临时上传，而且**不报错**，
    肉眼完全看不出来。同类守卫见 test_tool_discovery.py。
    """

    def test_upload_writers_are_isolated(self):
        here = Path(__file__).resolve().parent
        missing = []
        for path in here.glob("test_*.py"):
            text = path.read_text(encoding="utf-8")
            writes = "AppTest.from_file" in text or "bind_dataset(pd.DataFrame" in text
            if writes and "isolate_uploads" not in text:
                missing.append(path.name)
        self.assertEqual(missing, [], f"这些模块会写/删上传区却没隔离：{missing}")


class _State:
    """够用的假 session_state：属性读写 + get/pop。"""

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)

    def get(self, key: str, default: object = None) -> object:
        return self.__dict__.get(key, default)

    def pop(self, key: str, default: object = None) -> object:
        return self.__dict__.pop(key, default)


class TestChatSwitch(_Base):
    """界面上的接线：换对话要把旧 sid 的副本删掉。

    这条不测 uploads 本身，测的是"有人真的调它"——单测全绿但界面忘了接，
    这个 bug 一模一样会回来（一开始就是这样：unbind_dataset 只关了连接）。
    """

    def test_switching_chat_discards_old_session(self):
        from app import ui_streamlit as ui

        sid = "aaaa1111"
        uploads.table_path(sid).write_text("城市,销售额\n甲,1\n", encoding="utf-8")
        uploads.doc_folder(sid).mkdir()
        (uploads.doc_folder(sid) / "笔记.txt").write_text("x", encoding="utf-8")

        state = _State(
            sid=sid,
            chat_id=None,
            turns=[],
            temp_table=None,
            table_id=None,
            bound=None,
            doc_sig=None,
            vault_docs=[],
            saved_sig=None,
            restored=None,
            missing_table=None,
            session=None,
            agent=None,
            conv_title=None,
        )
        fake = mock.Mock()
        fake.session_state = state

        with mock.patch.object(ui, "st", fake):
            ui._forget_local_state("20261105-120000-abcd")

        self.assertFalse(uploads.table_path(sid).exists())
        self.assertFalse(uploads.doc_folder(sid).exists())
        self.assertNotEqual(state.sid, sid)


if __name__ == "__main__":
    unittest.main()
