"""本地文件读写 + 模式开关的测试。

最要紧的一条是 `test_read_mode_blocks_to_csv`：早先 AST 只拦了 open()，
`df.to_csv("桌面/test.txt")` 能直接写出去。如果那条退回去，整个「只读」开关
就变成了一个装饰品，所以它必须是一条会失败的测试，而不是注释里的一句话。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import files, sandbox  # noqa: E402
from app.agent import make_system_prompt  # noqa: E402
from app.config import DATA_FILE, MAX_WRITE_KB  # noqa: E402
from app.tools import registry  # noqa: E402


class TempCase(unittest.TestCase):
    """每个用例一个临时目录，并且强制把模式复位——模式是进程级全局变量，
    漏复位会让后面所有用例莫名其妙地"能写文件"。"""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="files-")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        files.set_mode(files.MODE_READ)
        self.addCleanup(files.set_mode, files.MODE_READ)

    def path(self, name: str = "a.txt") -> str:
        return str(self.root / name)


class TestResolve(TempCase):
    def test_absolute_path(self):
        self.assertEqual(files.resolve(self.path()), self.root / "a.txt")

    def test_parent_traversal_is_normalized(self):
        raw = str(self.root / "sub" / ".." / "b.txt")
        self.assertEqual(files.resolve(raw), self.root / "b.txt")

    def test_backslash_path(self):
        if os.name != "nt":
            self.skipTest("反斜杠只在 Windows 上有意义")
        raw = str(self.root).replace("/", "\\") + "\\c.txt"
        self.assertEqual(files.resolve(raw), self.root / "c.txt")

    def test_relative_path_lands_in_home(self):
        with mock.patch.object(Path, "home", return_value=self.root):
            self.assertEqual(files.resolve("d.txt"), self.root / "d.txt")

    def test_desktop_alias(self):
        # 桌面目录现在走 desktop_dir()（问系统要真实路径），所以要连它一起换掉，
        # 只 patch Path.home 已经不够了。
        with mock.patch.object(files, "desktop_dir", return_value=self.root / "Desktop"):
            self.assertEqual(files.resolve("桌面/note.txt"), self.root / "Desktop" / "note.txt")
            self.assertEqual(files.resolve("Desktop/note.txt"), self.root / "Desktop" / "note.txt")
            self.assertEqual(files.resolve("DESKTOP/note.txt"), self.root / "Desktop" / "note.txt")
            self.assertEqual(files.resolve("桌面"), self.root / "Desktop")

    def test_tilde_expands(self):
        # ~ 走的是 os.path.expanduser，它认环境变量而不是 Path.home()，
        # 所以这里必须改环境变量——两个 API 用的是不同来源，这点很容易记错。
        env = {"USERPROFILE": str(self.root), "HOME": str(self.root)}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(files.resolve("~/e.txt"), self.root / "e.txt")

    def test_empty_path_rejected(self):
        with self.assertRaises(files.FileError):
            files.resolve("   ")


class TestBareFileName(TempCase):
    """只给文件名时别把文件写到看不见的地方去。

    用户说的是"桌面上的 test.txt"，模型经常只传 `test.txt`。按 resolve 的规则
    它会落到主目录：文件在桌面、它却在主目录新建一个同名文件，然后回报"已写入"。
    这类错误不报错、不显眼，用户只看到"桌面上没变化"，所以规则要写死。
    """

    def setUp(self) -> None:
        super().setUp()
        self.home = self.root / "home"
        self.desk = self.home / "Desktop"
        self.home.mkdir()
        self.desk.mkdir()
        patches = [
            mock.patch.object(Path, "home", return_value=self.home),
            mock.patch.object(files, "desktop_dir", return_value=self.desk),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_home_copy_wins_when_it_exists(self):
        (self.home / "a.txt").write_text("主目录", encoding="utf-8")
        (self.desk / "a.txt").write_text("桌面", encoding="utf-8")
        self.assertEqual(files.locate("a.txt", must_exist=True), self.home / "a.txt")

    def test_desktop_copy_found_when_home_empty(self):
        (self.desk / "a.txt").write_text("桌面", encoding="utf-8")
        self.assertEqual(files.locate("a.txt", must_exist=True), self.desk / "a.txt")
        text, _ = files.read_text("a.txt")
        self.assertEqual(text, "桌面")

    def test_missing_bare_name_lists_both_places(self):
        with self.assertRaises(files.FileError) as ctx:
            files.locate("nope.txt", must_exist=True)
        msg = str(ctx.exception)
        self.assertIn("桌面/", msg)
        self.assertIn("~/", msg)

    def test_write_to_new_bare_name_is_refused(self):
        """两边都没有的新文件：拒绝，不猜目录。猜错的代价是文件凭空出现在主目录。"""
        files.set_mode(files.MODE_WRITE)
        with self.assertRaises(files.FileError) as ctx:
            files.write_text("brand-new.txt", "x")
        self.assertIn("桌面/", str(ctx.exception))
        self.assertFalse((self.home / "brand-new.txt").exists())
        self.assertFalse((self.desk / "brand-new.txt").exists())

    def test_write_bare_name_lands_on_desktop_file(self):
        files.set_mode(files.MODE_WRITE)
        (self.desk / "a.txt").write_text("旧", encoding="utf-8")
        info = files.write_text("a.txt", "新")
        self.assertEqual(Path(info["path"]), (self.desk / "a.txt").resolve())
        self.assertEqual((self.desk / "a.txt").read_text(encoding="utf-8"), "新")
        self.assertFalse((self.home / "a.txt").exists())

    def test_path_with_directory_is_untouched(self):
        """有目录线索就按字面走，不做任何"帮忙"。"""
        files.set_mode(files.MODE_WRITE)
        files.write_text(str(self.home / "sub" / "x.txt"), "x")
        self.assertTrue((self.home / "sub" / "x.txt").exists())


class TestRead(TempCase):
    def test_reads_utf8(self):
        (self.root / "a.txt").write_text("第一行\n第二行", encoding="utf-8")
        text, meta = files.read_text(self.path())
        self.assertIn("第二行", text)
        self.assertFalse(meta["empty"])

    def test_reads_gbk(self):
        # 中文 Windows 上记事本另存的 txt 多半是 GB18030，
        # 用 utf-8 硬读会抛异常，所以探测顺序里必须有它。
        (self.root / "gbk.txt").write_bytes("营收增长 18%".encode("gb18030"))
        text, _ = files.read_text(str(self.root / "gbk.txt"))
        self.assertIn("营收增长", text)

    def test_empty_file_is_not_an_error(self):
        (self.root / "empty.txt").write_bytes(b"")
        text, meta = files.read_text(self.path("empty.txt"))
        self.assertEqual(text, "")
        self.assertTrue(meta["empty"])

    def test_binary_file_rejected(self):
        (self.root / "bin.dat").write_bytes(bytes(range(32)))
        with self.assertRaises(files.FileError):
            files.read_text(str(self.root / "bin.dat"))

    def test_missing_file(self):
        with self.assertRaises(files.FileError):
            files.read_text(self.path("nope.txt"))

    def test_directory_gives_actionable_message(self):
        with self.assertRaises(files.FileError) as ctx:
            files.read_text(str(self.root))
        self.assertIn("list_files", str(ctx.exception))

    def test_truncates_long_content(self):
        (self.root / "long.txt").write_text("x" * 500, encoding="utf-8")
        text, meta = files.read_text(self.path("long.txt"), limit=100)
        self.assertEqual(len(text), 100)
        self.assertTrue(meta["clipped"])


class TestWrite(TempCase):
    def test_read_mode_refuses(self):
        with self.assertRaises(files.FileError) as ctx:
            files.write_text(self.path(), "hello")
        self.assertIn("只读", str(ctx.exception))
        self.assertFalse((self.root / "a.txt").exists())

    def test_creates_file(self):
        files.set_mode(files.MODE_WRITE)
        info = files.write_text(self.path(), "你好")
        self.assertFalse(info["existed"])
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "你好")

    def test_overwrites(self):
        files.set_mode(files.MODE_WRITE)
        files.write_text(self.path(), "旧的")
        info = files.write_text(self.path(), "新的")
        self.assertTrue(info["existed"])
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "新的")

    def test_appends(self):
        files.set_mode(files.MODE_WRITE)
        files.write_text(self.path(), "第一行\n")
        files.write_text(self.path(), "第二行\n", append=True)
        self.assertEqual(
            (self.root / "a.txt").read_text(encoding="utf-8"), "第一行\n第二行\n"
        )

    def test_creates_parent_dirs(self):
        files.set_mode(files.MODE_WRITE)
        files.write_text(str(self.root / "x" / "y" / "z.txt"), "deep")
        self.assertTrue((self.root / "x" / "y" / "z.txt").is_file())

    def test_no_backup_file_created(self):
        """用户明确要求不备份。这条测试是为了防止"好心"加回 .bak——
        加了之后 files/ 目录里会多出一堆用户没要求的文件。"""
        files.set_mode(files.MODE_WRITE)
        files.write_text(self.path(), "第一版")
        files.write_text(self.path(), "第二版")
        self.assertEqual([p.name for p in self.root.iterdir()], ["a.txt"])

    def test_size_limit(self):
        files.set_mode(files.MODE_WRITE)
        with self.assertRaises(files.FileError):
            files.write_text(self.path(), "x" * (MAX_WRITE_KB * 1024 + 1))

    def test_utf8_bytes(self):
        files.set_mode(files.MODE_WRITE)
        info = files.write_text(self.path(), "中文")
        self.assertEqual(info["after"], len("中文".encode("utf-8")))


class TestList(TempCase):
    def test_lists_files_and_dirs(self):
        (self.root / "sub").mkdir()
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        rows = files.list_dir(str(self.root))
        names = [r["name"] for r in rows]
        self.assertIn("a.txt", names)
        self.assertIn("sub", names)
        self.assertTrue(any(r["is_dir"] for r in rows))

    def test_file_path_rejected(self):
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(files.FileError):
            files.list_dir(self.path())

    def test_missing_dir(self):
        with self.assertRaises(files.FileError):
            files.list_dir(str(self.root / "nope"))


class TestSandboxPolicy(TempCase):
    def test_read_mode_blocks_to_csv(self):
        target = self.root / "leak.csv"
        result = sandbox.run(f"df.head(2).to_csv(r'{target.as_posix()}')", DATA_FILE)
        self.assertFalse(result.ok)
        self.assertIn("只读模式", result.output)
        self.assertFalse(target.exists())

    def test_write_mode_allows_to_csv(self):
        files.set_mode(files.MODE_WRITE)
        target = self.root / "out.csv"
        result = sandbox.run(
            f"df.head(2).to_csv(r'{target.as_posix()}')", DATA_FILE, allow_write=True
        )
        self.assertTrue(result.ok, result.output)
        self.assertTrue(target.exists())

    def test_read_mode_blocks_open(self):
        result = sandbox.run(f"open(r'{self.path()}', 'w')", DATA_FILE)
        self.assertFalse(result.ok)
        self.assertIn("open", result.output)

    def test_write_mode_allows_open(self):
        files.set_mode(files.MODE_WRITE)
        target = self.root / "written.txt"
        code = f"open(r'{target.as_posix()}', 'w', encoding='utf-8').write('hello')"
        result = sandbox.run(code, None, allow_write=True)
        self.assertTrue(result.ok, result.output)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello")

    def test_write_mode_allows_os_but_not_subprocess(self):
        files.set_mode(files.MODE_WRITE)
        self.assertTrue(sandbox.run("import os\nos.getcwd()", None, allow_write=True).ok)
        result = sandbox.run("import subprocess\n1", None, allow_write=True)
        self.assertFalse(result.ok)
        self.assertIn("subprocess", result.output)

    def test_write_mode_still_blocks_eval(self):
        files.set_mode(files.MODE_WRITE)
        result = sandbox.run("eval('1+1')", None, allow_write=True)
        self.assertFalse(result.ok)

    def test_no_dataset_gives_df_none(self):
        files.set_mode(files.MODE_WRITE)
        result = sandbox.run("df is None", None, allow_write=True)
        self.assertTrue(result.ok, result.output)
        self.assertIn("True", result.output)


class TestOpen(TempCase):
    """打开文件/程序。

    这里**必须**把 files._launch 换掉：真跑一遍会弹出记事本、微信窗口，
    甚至让测试卡在某个 GUI 上。用一个记录器替代，既能断言"传了什么路径"，
    又不会真的启动任何东西。
    """

    def setUp(self) -> None:
        super().setUp()
        self.launched: list[Path] = []
        patcher = mock.patch.object(files, "_launch", side_effect=self.launched.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_read_mode_refuses_to_open(self):
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(files.FileError) as ctx:
            files.open_path(self.path())
        self.assertIn("完全工作区", str(ctx.exception))
        self.assertEqual(self.launched, [])

    def test_write_mode_opens_file(self):
        files.set_mode(files.MODE_WRITE)
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        info = files.open_path(self.path())
        self.assertEqual(info["kind"], ".txt")
        self.assertEqual(self.launched, [(self.root / "a.txt").resolve()])

    def test_opens_directory(self):
        files.set_mode(files.MODE_WRITE)
        info = files.open_path(str(self.root))
        self.assertEqual(info["kind"], "dir")

    def test_missing_path_rejected(self):
        files.set_mode(files.MODE_WRITE)
        with self.assertRaises(files.FileError):
            files.open_path(str(self.root / "nope.txt"))
        self.assertEqual(self.launched, [])

    def test_launch_failure_becomes_friendly_error(self):
        files.set_mode(files.MODE_WRITE)
        (self.root / "a.xyz").write_text("x", encoding="utf-8")
        with mock.patch.object(files, "_launch", side_effect=OSError("没有关联程序")):
            with self.assertRaises(files.FileError) as ctx:
                files.open_path(self.path("a.xyz"))
        self.assertIn("没有关联程序", str(ctx.exception))

    def test_tool_says_requested_not_succeeded(self):
        """只能说"已请求系统打开"。它确实看不到那个窗口，说自己成功就是在编。"""
        files.set_mode(files.MODE_WRITE)
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        text = registry.call("open_file", {"path": self.path()})
        self.assertIn("已请求系统打开", text)
        self.assertNotIn("已成功打开", text)
        self.assertEqual(len(self.launched), 1)

    def test_tool_refused_in_read_mode(self):
        text = registry.call("open_file", {"path": self.path()})
        self.assertIn("错误", text)
        self.assertIn("只读", text)
        self.assertEqual(self.launched, [])


class TestFindPrograms(TempCase):
    """开始菜单查找。只翻 .lnk，不扫 Program Files。"""

    def setUp(self) -> None:
        super().setUp()
        self.menu = self.root / "StartMenu"
        (self.menu / "微信").mkdir(parents=True)
        (self.menu / "Google Chrome").mkdir()
        (self.menu / "微信" / "微信.lnk").write_text("", encoding="utf-8")
        (self.menu / "微信" / "卸载微信.lnk").write_text("", encoding="utf-8")
        (self.menu / "Google Chrome" / "Chrome.lnk").write_text("", encoding="utf-8")
        (self.menu / "NotALink.txt").write_text("", encoding="utf-8")
        patcher = mock.patch.object(files, "_start_menu_dirs", return_value=[self.menu])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_finds_by_chinese_name(self):
        rows = files.find_programs("微信")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["name"], "微信")  # 短名在前，排在"卸载微信"之前
        self.assertTrue(rows[0]["path"].endswith(".lnk"))

    def test_ignores_non_shortcut_files(self):
        rows = files.find_programs("NotALink")
        self.assertEqual(rows, [])

    def test_case_insensitive(self):
        self.assertEqual([r["name"] for r in files.find_programs("chrome")], ["Chrome"])

    def test_limit(self):
        self.assertEqual(len(files.find_programs("微信", limit=1)), 1)

    def test_no_match(self):
        self.assertEqual(files.find_programs("钉钉"), [])

    def test_tool_output_has_path_and_next_step(self):
        text = registry.call("find_program", {"name": "微信"})
        self.assertIn("微信", text)
        self.assertIn(".lnk", text)
        self.assertIn("open_file", text)

    def test_tool_no_match_is_actionable(self):
        text = registry.call("find_program", {"name": "钉钉"})
        self.assertIn("没有匹配", text)
        self.assertIn("list_files", text)


class TestTools(TempCase):
    def test_tools_registered(self):
        self.assertLessEqual(
            {"list_files", "read_file", "write_text_file", "open_file", "find_program"},
            set(registry.names()),
        )

    def test_write_tool_refused_in_read_mode(self):
        text = registry.call("write_text_file", {"path": self.path(), "content": "x"})
        self.assertIn("错误", text)
        self.assertIn("只读", text)
        self.assertFalse((self.root / "a.txt").exists())

    def test_write_tool_works_in_write_mode(self):
        files.set_mode(files.MODE_WRITE)
        text = registry.call("write_text_file", {"path": self.path(), "content": "三条待办"})
        self.assertIn("新建", text)
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "三条待办")

    def test_read_tool_mentions_empty(self):
        (self.root / "e.txt").write_bytes(b"")
        text = registry.call("read_file", {"path": str(self.root / "e.txt")})
        self.assertIn("空的", text)

    def test_list_tool_output(self):
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        text = registry.call("list_files", {"path": str(self.root)})
        self.assertIn("a.txt", text)
        self.assertIn("[文件]", text)

    def test_append_via_tool(self):
        files.set_mode(files.MODE_WRITE)
        registry.call("write_text_file", {"path": self.path(), "content": "A"})
        registry.call(
            "write_text_file", {"path": self.path(), "content": "B", "mode": "append"}
        )
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "AB")


class TestPrompt(unittest.TestCase):
    def test_write_mode_mentions_file_tools(self):
        text = make_system_prompt(write_mode=True)
        self.assertIn("完全工作区", text)
        self.assertIn("write_text_file", text)
        self.assertIn("不做备份", text)

    def test_read_mode_does_not(self):
        text = make_system_prompt(columns=["a"], rows=10, label="t.csv")
        self.assertNotIn("完全工作区", text)
        self.assertNotIn("write_text_file", text)


class TestChatGuard(unittest.TestCase):
    def test_read_mode_needs_content(self):
        from app.ui_streamlit import _can_chat

        files.set_mode(files.MODE_READ)
        self.addCleanup(files.set_mode, files.MODE_READ)
        self.assertFalse(_can_chat(False))
        self.assertTrue(_can_chat(True))

    def test_write_mode_allows_empty_page(self):
        from app.ui_streamlit import _can_chat

        files.set_mode(files.MODE_WRITE)
        self.addCleanup(files.set_mode, files.MODE_READ)
        self.assertTrue(_can_chat(False))


if __name__ == "__main__":
    unittest.main()
