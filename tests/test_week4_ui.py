"""界面层冒烟测试：真的执行一遍 app.py，断言页面能渲染出来。

为什么值得有：单元测试只能证明"函数返回对不对"，证明不了"页面打开没报错"。
Streamlit 的报错是显示在页面上的，HTTP 仍然返回 200，所以 curl 健康检查查不出来。

需要 streamlit（Anaconda 自带）。没装就整组跳过，不影响其他测试。

文件都从储藏区进去，不从上传框进：AppTest 驱动不了 file_uploader，
而储藏区正是网页重启后唯一的文件来源，走这条路等于顺带验了"重开页面文件还在"。
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _isolation import isolate_embeddings, isolate_uploads

ROOT = Path(__file__).resolve().parent.parent

try:
    import pandas as pd
    from streamlit.testing.v1 import AppTest

    from app import vault

    HAS_APPTEST = True
except Exception:  # noqa: BLE001  没装 streamlit，或者版本太老没有 testing 模块
    HAS_APPTEST = False


def fresh_vault(case: unittest.TestCase) -> Path:
    """每个用例类换一个空储藏区，别碰用户真实的那份 data/vault。"""
    root = Path(tempfile.mkdtemp(prefix="vault-ui-"))
    patcher = mock.patch.object(vault, "VAULT_DIR", root)
    patcher.start()
    case.addClassCleanup(patcher.stop)
    case.addClassCleanup(shutil.rmtree, root, ignore_errors=True)
    return root


_CHAT_ROOT: Path | None = None


def setUpModule() -> None:
    """整个模块的对话历史都写到临时目录。

    这个文件里有几条用例会真的跑完一轮问答（假 Agent，答案是"假答案"）。
    对话历史接上以后，不隔开就等于每次跑测试都往 data/chats/ 里塞几份假对话，
    用户的真实历史被挤到列表后面去。

    上传区同理，而且更狠：AppTest 会把 app.py 整个跑一遍，入口里有"清掉上个
    进程遗留"的扫描——不隔开就会删掉用户真实的临时上传。
    """
    global _CHAT_ROOT
    isolate_uploads()
    # 上传处理函数会在后台给资料算向量。本机 .env 配了真实向量模型，
    # 不隔离就等于测试期间发真实 HTTP 请求。
    isolate_embeddings()
    if not HAS_APPTEST:
        return
    from app import chats

    _CHAT_ROOT = Path(tempfile.mkdtemp(prefix="chats-ui-"))
    patcher = mock.patch.object(chats, "CHAT_DIR", _CHAT_ROOT)
    patcher.start()
    # 模块级 patch 用 addModuleCleanup 管收尾（unittest 保证 tearDownModule 后也执行）
    unittest.addModuleCleanup(patcher.stop)
    unittest.addModuleCleanup(shutil.rmtree, _CHAT_ROOT, ignore_errors=True)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestEmptyState(unittest.TestCase):
    """储藏区空着、也没上传时的页面：能打开、不报错、不给提问框。"""

    @classmethod
    def setUpClass(cls) -> None:
        fresh_vault(cls)
        cls.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        cls.app.run()

    def test_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_title(self):
        self.assertEqual(self.app.title[0].value, "DataInsight")

    def test_no_how_it_works_text(self):
        """空状态不再解释"CSV 会变成表、PDF 会变成资料"。

        那是实现细节，用户只需要知道"传文件"这一件事。界面到了应用阶段，
        文案要像产品，不像教程。
        """
        texts = [m.value for m in self.app.sidebar.markdown]
        texts += [c.value for c in self.app.sidebar.caption]
        texts += [c.value for c in self.app.caption]
        for hint in ("会变成可查询的表", "可检索的资料", "还没有文件"):
            self.assertFalse(any(hint in t for t in texts), (hint, texts))
        self.assertFalse(any("sales.csv" in t for t in texts), texts)

    def test_two_uploaders_by_lifetime(self):
        """两个框，按"留多久"分：临时上传 vs 储藏区。

        不是按文件类型分（CSV 一个框、资料一个框）——那会逼用户猜该放哪个，
        放错了还什么提示都没有。类型由后缀判断，不该是用户的选择题。
        AppTest 读不到 file_uploader，就退一步看侧边栏的分区标题。
        """
        headers = [h.value for h in self.app.sidebar.subheader]
        self.assertIn("临时上传", headers)
        self.assertIn("文件储藏区", headers)
        self.assertNotIn("数据表", headers)
        self.assertNotIn("资料文件", headers)

    def test_chat_input_hidden(self):
        self.assertEqual(len(self.app.chat_input), 0)

    def test_no_suggestions(self):
        labels = [b.label for b in self.app.sidebar.button]
        self.assertNotIn("各城市销售额排名前五", labels)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestAfterUpload(unittest.TestCase):
    """储藏区里有一份 CSV：自动拿来当当前表，示例问题跟着列名走。"""

    @classmethod
    def setUpClass(cls) -> None:
        fresh_vault(cls)
        vault.add("我的数据.csv", "城市,销售额\n甲,10\n乙,20\n".encode("utf-8"))
        cls.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        cls.app.run()

    def test_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_badge_shows_uploaded_name(self):
        texts = [m.value for m in self.app.sidebar.markdown]
        self.assertTrue(any("我的数据.csv" in t for t in texts), texts)

    def test_suggestions_follow_real_columns(self):
        labels = [b.label for b in self.app.sidebar.button]
        self.assertTrue(any("城市" in x for x in labels), labels)

    def test_chat_input_present(self):
        self.assertEqual(len(self.app.chat_input), 1)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestDocumentOnly(unittest.TestCase):
    """储藏区里只有一份资料、没有表：也该能提问。

    这条对应一个真实的坑：空状态守卫原来写的是"没有表就 return"，
    结果用户传了 PDF 却看不到输入框，还以为程序坏了。
    """

    @classmethod
    def setUpClass(cls) -> None:
        fresh_vault(cls)
        vault.add("说明.txt", "这是一份测试资料，讲的是季度复盘。".encode("utf-8"))
        cls.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        cls.app.run()

    def test_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_chat_input_present(self):
        self.assertEqual(len(self.app.chat_input), 1)

    def test_sidebar_lists_document(self):
        texts = [m.value for m in self.app.sidebar.markdown]
        self.assertTrue(any("说明.txt" in t for t in texts), texts)

    def test_suggestions_are_about_document(self):
        labels = [b.label for b in self.app.sidebar.button]
        self.assertTrue(any("说明.txt" in x for x in labels), labels)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestVaultSection(unittest.TestCase):
    """侧边栏的储藏区：用量、清单、删除。

    每条用例现开一个空储藏区：删除是会改状态的，共用一份数据时，
    谁先跑决定谁看见什么，这种测试迟早会咬人。
    """

    def setUp(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="vault-ui-"))
        patcher = mock.patch.object(vault, "VAULT_DIR", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)

        vault.add("旧报表.txt", ("数据" * 100).encode("utf-8"))
        vault.add("销量.csv", "城市,销售额\n甲,1\n".encode("utf-8"))
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        self.app.run()

    def test_header_and_usage_present(self):
        headers = [h.value for h in self.app.sidebar.subheader]
        self.assertIn("文件储藏区", headers)

        captions = " ".join(c.value for c in self.app.sidebar.caption)
        self.assertIn("/ 200.0 MB", captions)

    def test_lists_every_entry(self):
        texts = " ".join(m.value for m in self.app.sidebar.markdown)
        self.assertIn("旧报表.txt", texts)
        self.assertIn("销量.csv", texts)

    def test_delete_removes_from_disk(self):
        before = len(vault.entries())
        button = next(b for b in self.app.sidebar.button if b.label == "删除")
        button.click().run()

        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))
        self.assertEqual(len(vault.entries()), before - 1)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestTempDocument(unittest.TestCase):
    """临时上传的资料：进会话、能问，但不进储藏区。

    AppTest 驱动不了 file_uploader，所以直接调数据层登记一份"本次会话自有"的资料，
    等价于临时上传走过的那条路（Doc.owned=True，文件在 data/uploads/<sid>/）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        from app.tools import data_tools as dt

        fresh_vault(cls)
        cls.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        cls.app.run()
        cls.sid = cls.app.session_state["sid"]
        dt.add_document(
            "现场笔记.txt", "临时记的两句：渠道 A 转化 3%。".encode("utf-8"), key=cls.sid
        )
        cls.app.run()

    @classmethod
    def tearDownClass(cls) -> None:
        from app.tools import data_tools as dt

        dt.clear_documents(cls.sid)

    def test_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_listed_in_sidebar(self):
        texts = " ".join(m.value for m in self.app.sidebar.markdown)
        self.assertIn("现场笔记.txt", texts)

    def test_marked_as_temporary(self):
        captions = " ".join(c.value for c in self.app.sidebar.caption)
        self.assertIn("临时资料", captions)

    def test_not_written_into_vault(self):
        self.assertEqual(vault.entries(), [])

    def test_chat_input_present(self):
        self.assertEqual(len(self.app.chat_input), 1)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestTempTableWins(unittest.TestCase):
    """临时传的表要顶掉储藏区那张，而不是被自动绑定的储藏区表盖回去。"""

    @classmethod
    def setUpClass(cls) -> None:
        fresh_vault(cls)
        vault.add("储藏区的表.csv", "城市,销售额\n甲,1\n".encode("utf-8"))
        cls.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        cls.app.run()
        cls.app.session_state["temp_table"] = (
            "临时表.csv",
            pd.DataFrame({"城市": ["甲", "乙"], "销售额": [7, 9]}),
        )
        cls.app.run()

    def test_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_badge_shows_temp_table(self):
        texts = " ".join(m.value for m in self.app.sidebar.markdown)
        self.assertIn("临时表.csv", texts)

    def test_suggestions_follow_temp_columns(self):
        labels = [b.label for b in self.app.sidebar.button]
        self.assertTrue(any("城市" in x for x in labels), labels)


class _FakeFile:
    """够 file_uploader 用的最小替身：只用 name / size / getvalue。"""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestTempUploadHandler(unittest.TestCase):
    """临时上传的处理函数本身：不写储藏区，表直接顶掉当前那张。

    AppTest 喂不进 file_uploader，所以把 st 换成假的、直接调这个函数。
    验的是它有没有把文件放错地方——放错的话，用户以为存进储藏区了，
    下次打开却什么都没有。
    """

    def setUp(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="vault-temp-"))
        patcher = mock.patch.object(vault, "VAULT_DIR", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)

        from app import ui_streamlit as ui
        from app.tools import data_tools as dt

        self.ui, self.dt = ui, dt
        self.sid = "tempsid"
        state = SimpleNamespace(
            sid=self.sid,
            temp_sigs={},
            vault_sigs={},
            temp_table=None,
            table_id=None,
            bound=None,
            doc_sig=None,
            vault_docs=[],
            temp_gen=0,
            vault_gen=0,
            turns=[],
        )
        self.state = state
        stub = mock.Mock()
        stub.session_state = state
        stub.column_config = {}
        stub_patcher = mock.patch.object(ui, "st", stub)
        stub_patcher.start()
        self.addCleanup(stub_patcher.stop)
        self.addCleanup(dt.clear_documents, self.sid)
        self.addCleanup(dt.unbind_dataset, self.sid)

    def test_csv_becomes_current_table(self):
        self.ui._handle_temp_upload([_FakeFile("临时.csv", "城市,销售额\n甲,1\n".encode())])

        self.assertIsNotNone(self.state.temp_table)
        self.assertEqual(self.state.temp_table[0], "临时.csv")
        self.assertEqual(self.state.table_id, "")  # 别让储藏区那张在下一轮抢回来
        self.assertEqual(vault.entries(), [])

    def test_document_goes_into_session_only(self):
        self.ui._handle_temp_upload([_FakeFile("现场笔记.txt", "渠道 A 转化 3%。".encode())])

        docs = self.dt.documents(self.sid)
        self.assertEqual([d.name for d in docs], ["现场笔记.txt"])
        self.assertTrue(docs[0].owned)  # 会话自己拥有 -> 撤下时连带删盘
        self.assertEqual(vault.entries(), [])

    def test_same_file_twice_registered_once(self):
        file = _FakeFile("现场笔记.txt", "渠道 A 转化 3%。".encode())
        self.ui._handle_temp_upload([file])
        self.ui._handle_temp_upload([file])

        self.assertEqual(len(self.dt.documents(self.sid)), 1)


class _ScriptedAgent:
    """假 Agent：不碰网络，只把统计数字推进去，顺便发两个事件。

    用它是因为这里要验的是"数字有没有送到侧边栏"，不是模型回答得对不对。
    真 Agent 一调就会发 HTTP 请求，测试既慢又不稳定。
    """

    def __init__(self, session) -> None:
        self.session = session
        self.on_event = None

    def update_system_prompt(self, text: str) -> None:  # noqa: ARG002
        pass

    def ask(self, question: str) -> str:
        self.session.add_user(question)
        for _ in range(2):
            self.session.steps += 1
            self.session.tokens += 60
            if self.on_event:
                self.on_event("step", {"step": self.session.steps, "tokens": 60, "latency": 0.1})
        self.session.tool_calls += 1
        if self.on_event:
            self.on_event("tool_call", {"name": "run_sql", "args": "{}"})
        return "假答案"


class _SlowAgent(_ScriptedAgent):
    """比标题线程慢半秒。

    标题是后台线程起的，如果假 Agent 也瞬完，两者谁先跑完说不准，
    断言就会时绿时红。慢半秒把先后关系钉住。
    """

    def ask(self, question: str) -> str:
        time.sleep(0.5)
        return super().ask(question)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestSidebarStatsDuringTurn(unittest.TestCase):
    """一轮提问结束后，侧边栏的统计必须已经变了，不需要再点一次。"""

    @classmethod
    def setUpClass(cls) -> None:
        from app import ui_streamlit as ui

        fresh_vault(cls)
        vault.add("我的数据.csv", "城市,销售额\n甲,10\n乙,20\n".encode("utf-8"))
        cls.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        cls.app.run()
        cls.app.session_state["agent"] = _ScriptedAgent(cls.app.session_state["session"])

        # 标题走真模型，这里换成固定值：既不联网，也不会让后台线程影响断言
        patcher = mock.patch.object(ui, "_make_conv_title", lambda question: "测试标题")
        patcher.start()
        cls.addClassCleanup(patcher.stop)

        cls.app.chat_input[0].set_value("各城市销售额排名前五").run()

    def test_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_stats_visible_without_extra_interaction(self):
        texts = " ".join(c.value for c in self.app.sidebar.caption)
        self.assertIn("轮次 1", texts)
        self.assertIn("步数 2", texts)
        self.assertIn("工具调用 1", texts)
        self.assertIn("累计 tokens 120", texts)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestConversationTitle(unittest.TestCase):
    """第一轮提问时就把标题写进会话状态，标签页不必等回答写完。"""

    @classmethod
    def setUpClass(cls) -> None:
        from app import ui_streamlit as ui

        fresh_vault(cls)
        vault.add("我的数据.csv", "城市,销售额\n甲,10\n乙,20\n".encode("utf-8"))
        cls.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        cls.app.run()
        cls.app.session_state["agent"] = _SlowAgent(cls.app.session_state["session"])

        patcher = mock.patch.object(ui, "_make_conv_title", lambda question: "城市销售排名")
        patcher.start()
        cls.addClassCleanup(patcher.stop)

        cls.app.chat_input[0].set_value("各城市销售额排名前五").run()

    def test_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))

    def test_title_follows_first_question(self):
        self.assertEqual(self.app.session_state["conv_title"], "城市销售排名")

    def test_title_stays_on_second_turn(self):
        """标题只在第一轮起一次：后面每问一句都换名字，标签页就没法当书签了。"""
        self.app.chat_input[0].set_value("那北京呢").run()

        self.assertEqual(len(self.app.exception), 0, list(self.app.exception))
        self.assertEqual(self.app.session_state["conv_title"], "城市销售排名")


if __name__ == "__main__":
    unittest.main()
