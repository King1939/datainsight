"""工具自动发现：新写的工具模块不能在界面进程里"隐身"。

背景：注册靠 import 副作用，而 `app.tools` 只在第一次导入时执行一次。
界面进程跑起来之后新建一个 `*_tools.py`，那行 import 命中 sys.modules 缓存，
模块根本不会执行 —— 模型拿到的工具列表里就没有它，人是看着文件在、
模型却说"没有这个工具"。这组测试盯的就是这条路径。
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from app import tools
from app.tools import registry


class _Info:
    """pkgutil.iter_modules 返回值的替身（真身是 ModuleInfo 具名元组）。"""

    def __init__(self, name: str) -> None:
        self.name = name


class TestLoadMissing(unittest.TestCase):
    def test_returns_empty_when_all_imported(self):
        """正常运行的时候没有新模块，不该有任何副作用。"""
        self.assertEqual(tools.load_missing(), [])

    def test_skips_private_and_registry(self):
        """`__init__` 和 `registry` 不是工具模块，不能被当成插件导入。"""
        fake = [_Info("registry"), _Info("__init__"), _Info("data_tools")]
        with mock.patch.object(tools.pkgutil, "iter_modules", return_value=fake), mock.patch.object(
            tools.importlib, "import_module"
        ) as importer:
            self.assertEqual(tools.load_missing(), [])
        importer.assert_not_called()

    def test_imports_only_modules_not_yet_loaded(self):
        """已经加载过的一律跳过，只导新文件——重复导入会撞上注册表的重名检查。"""

        def fake_import(name):
            if name.endswith("new_tools"):
                @tools.tool
                def discovered_tool() -> str:
                    """这个工具是测试临时注册的。"""
                    return "ok"

            sys.modules[name] = mock.MagicMock()
            return sys.modules[name]

        fake = [_Info("chart_tools"), _Info("new_tools")]
        try:
            with mock.patch.object(tools.pkgutil, "iter_modules", return_value=fake), mock.patch.object(
                tools.importlib, "import_module", side_effect=fake_import
            ) as importer:
                added = tools.load_missing()

            self.assertEqual(added, ["new_tools"])
            importer.assert_called_once_with("app.tools.new_tools")
            self.assertIn("discovered_tool", registry.names())
        finally:
            registry._tools.pop("discovered_tool", None)
            sys.modules.pop("app.tools.new_tools", None)

    def test_tool_module_file_on_disk_is_registered(self):
        """eye-level 检查：目录里真实存在的工具模块都已经在注册表里。

        新增一个 `*_tools.py` 却忘了加 import，这条会失败——比等到模型
        说"没有这个工具"要早得多。
        """
        from pathlib import Path

        folder = Path(tools.__file__).parent
        on_disk = {
            p.stem
            for p in folder.glob("*_tools.py")
            if not p.stem.startswith("_")
        }
        loaded = {name for name in sys.modules if name.startswith("app.tools.")}
        missing = {m for m in on_disk if f"app.tools.{m}" not in loaded}
        self.assertEqual(missing, set(), f"这些工具文件没被导入：{sorted(missing)}")

    def test_agent_specs_expose_mindmap(self):
        """模型看到的是 specs()，不是文件。这里确认 schema 真的发出去了。"""
        names = [s["function"]["name"] for s in registry.specs()]
        self.assertIn("make_mindmap", names)

        spec = registry.get("make_mindmap").spec()["function"]
        self.assertIn("paths", spec["parameters"]["properties"])
        self.assertEqual(spec["parameters"]["required"], ["paths"])


if __name__ == "__main__":
    unittest.main()
