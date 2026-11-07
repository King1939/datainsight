"""思维导图工具：建树、校验、渲染、以及与界面的接口。

重点测三件事：
  1. 相同前缀要合并（否则模型把每条路径都当独立分支画出来）
  2. 各种超限要在画之前拦住，且错误信息说清"怎么办"
  3. 返回值第一行必须是「图表已生成：xxx.png」——界面靠这个正则认图
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.tools import mindmap_tools, registry
from app.tools.mindmap_tools import make_mindmap
from app.ui_render import extract_chart


class MindmapBase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        patch = mock.patch.object(mindmap_tools, "OUTPUT_DIR", self.dir)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    def png(self) -> list[Path]:
        return list(self.dir.glob("mindmap_*.png"))


class TestBuild(MindmapBase):
    def test_shared_prefix_is_merged(self):
        """三条路径共用「销售分析」「华东」，合并后是 6 个节点。"""
        text = make_mindmap(
            paths=[
                ["销售分析", "华东", "上海"],
                ["销售分析", "华东", "杭州"],
                ["销售分析", "华北", "北京"],
            ]
        )
        self.assertIn("6 个节点", text)
        self.assertEqual(len(self.png()), 1)

    def test_duplicate_path_is_deduped(self):
        text = make_mindmap(
            paths=[["根", "A", "x"], ["根", "A", "x"], ["根", "A", "x"]]
        )
        self.assertIn("3 个节点", text)

    def test_two_levels_is_enough(self):
        text = make_mindmap(paths=[["根", "分支一"], ["根", "分支二"]])
        self.assertIn("3 个节点", text)
        self.assertIn("2 层", text)

    def test_single_branch(self):
        """只有一个一级分支时不该出错，也不该画出半张空白画布。"""
        text = make_mindmap(paths=[["根", "唯一分支", "叶子"]])
        self.assertIn("图表已生成", text)


class TestValidation(MindmapBase):
    def test_empty(self):
        self.assertIn("paths 是空的", make_mindmap(paths=[]))

    def test_flat_string_instead_of_path(self):
        """模型偶尔会传 ["根", "A"] 这种扁平结构，要给出正确写法。"""
        text = make_mindmap(paths=["根", "分支"])
        self.assertIn("路径列表", text)
        self.assertIn("[[", text)
        self.assertEqual(self.png(), [])

    def test_path_too_short(self):
        text = make_mindmap(paths=[["只有一个中心"]])
        self.assertIn("至少要两层", text)

    def test_multiple_roots(self):
        text = make_mindmap(paths=[["甲", "x"], ["乙", "y"]])
        self.assertIn("同一个中心节点", text)
        self.assertEqual(self.png(), [])

    def test_too_deep(self):
        deep = ["中心"] + [f"第{i}层" for i in range(1, 7)]  # 7 层
        text = make_mindmap(paths=[deep])
        self.assertIn("最多画 5 层", text)

    def test_too_many_nodes(self):
        paths = [["中心", f"分支{i}"] for i in range(61)]
        text = make_mindmap(paths=paths)
        self.assertIn("太多了", text)

    def test_too_many_leaves(self):
        paths = [["中心", "唯一分支", f"叶{i}"] for i in range(25)]
        text = make_mindmap(paths=paths)
        self.assertIn("叶子", text)

    def test_blank_labels_are_skipped(self):
        text = make_mindmap(paths=[["根", "  ", "叶子"], ["根", "叶子"]])
        self.assertIn("图表已生成", text)


class TestOutput(MindmapBase):
    def test_first_line_is_chart_marker(self):
        """界面靠 extract_chart 找图，格式一变图就不显示了。"""
        text = make_mindmap(paths=[["根", "A"]])
        first = text.splitlines()[0]
        self.assertTrue(first.startswith("图表已生成："))
        self.assertIsNotNone(extract_chart(text))

    def test_summary_mentions_center(self):
        text = make_mindmap(paths=[["销售分析", "A"]])
        self.assertIn("中心「销售分析」", text)

    def test_long_label_wraps_to_two_lines(self):
        node = mindmap_tools._Node("这是一个超过十二个字的很长标签用来测试折行")
        mindmap_tools._measure(node)
        self.assertIn("\n", node.label)
        self.assertEqual(len(node.label.split("\n")), 2)

    def test_very_long_label_is_truncated(self):
        node = mindmap_tools._Node("很长" * 20)
        mindmap_tools._measure(node)
        self.assertIn("…", node.label)
        self.assertEqual(len(node.label.split("\n")), 2)

    def test_wrap_is_idempotent(self):
        """_measure 之后 _draw 还会再遍历到同一个节点，不能每过一遍再切一刀。"""
        once = mindmap_tools._wrap("这是一个超过十二个字的很长标签")
        self.assertEqual(mindmap_tools._wrap(once), once)


class TestSchema(unittest.TestCase):
    def test_paths_shape(self):
        params = registry.get("make_mindmap").parameters
        self.assertIn("paths", params["required"])
        self.assertEqual(params["properties"]["paths"]["type"], "array")
        self.assertEqual(
            params["properties"]["paths"]["items"], {"type": "array", "items": {"type": "string"}}
        )

    def test_title_is_optional(self):
        params = registry.get("make_mindmap").parameters
        self.assertNotIn("title", params.get("required", []))


if __name__ == "__main__":
    unittest.main()
