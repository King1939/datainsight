"""edit_pptx 与 read_file(pptx) 的回归测试。

这套测试真正要守住的不是"能改"，而是**改的范围就是用户说的那几页**：
  - 插一页，其余页的文字一个都不能变（这是"重建整份"和"增量编辑"的分界线）；
  - 插页之后，后面每页的页码都得跟着变，否则页脚全是错的；
  - 任何一步失败，文件必须原封不动——改坏用户的原稿是这里唯一不可逆的事故。
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pptx import Presentation

EMU = 914400

import app.tools.pptx_tools as pt
from app import files
from app.tools import registry
from app.schema import build_tool_spec


def _deck() -> dict:
    return {
        "title": "季度经营分析",
        "slides": [
            {"layout": "cover", "title": "季度经营分析", "subtitle": "2026-11-05"},
            {"layout": "bullets", "title": "核心结论", "lead": "营收继续增长", "items": ["营收 +12%", "华东 46%"]},
            {"layout": "kpi", "title": "关键指标", "items": ["1,284|营收|+12%", "8.7|客单价|-3%"]},
            {"layout": "table", "title": "分城市", "headers": ["城市", "销售额"], "rows": [["上海", "486"], ["杭州", "312"]]},
            {"layout": "closing", "title": "谢谢", "subtitle": "欢迎提问"},
        ],
    }


def _pages(path: Path) -> list[list[str]]:
    """把每页的文字按页取出来，用来比对"哪些页被动了"。"""
    prs = Presentation(str(path))
    return [
        [shape.text_frame.text for shape in slide.shapes if shape.has_text_frame and shape.text_frame.text.strip()]
        for slide in prs.slides
    ]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        patch = mock.patch.object(pt, "OUTPUT_DIR", self.tmp)
        patch.start()
        self.addCleanup(patch.stop)
        files.set_mode("write")
        self.addCleanup(files.set_mode, "read")

    def build(self, **overrides) -> Path:
        payload = _deck()
        payload.update(overrides)
        text = registry.call("make_pptx", payload)
        match = re.search(r"演示文稿已生成：(.+?\.pptx)", text)
        self.assertIsNotNone(match, text)
        return Path(match.group(1))

    def edit(self, path: Path, ops, **kwargs) -> str:
        return registry.call("edit_pptx", {"path": str(path), "ops": ops, **kwargs})


class TestInsert(_Base):
    def test_other_pages_untouched(self):
        """插一页之后，原来的五页文字必须逐字不变——这就是"增量"和"重建"的区别。"""
        path = self.build()
        before = _pages(path)

        text = self.edit(path, [{"op": "insert", "after": 1, "content": {"layout": "bullets", "title": "四季度重点", "items": ["控成本"]}}])

        after = _pages(path)
        self.assertIn("已更新：", text)
        self.assertEqual(len(after), 6)
        # 页码会被重排，所以比内容时把纯数字的页脚剔掉——它本来就该变
        strip = lambda page: [t for t in page if not re.fullmatch(r"\d{2}", t)]  # noqa: E731
        self.assertEqual([strip(p) for p in after[:1] + after[2:]], [strip(p) for p in before])
        self.assertIn("四季度重点", "".join(after[1]))

    def test_page_numbers_renumbered(self):
        """插页后后面每页的页码都要重排，不然页脚会跳号或重号。"""
        path = self.build()
        self.edit(path, [{"op": "insert", "after": 1, "content": {"layout": "bullets", "title": "插入页", "items": ["一条"]}}])

        prs = Presentation(str(path))
        numbers = []
        for index, slide in enumerate(prs.slides, 1):
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                text = shape.text_frame.text.strip()
                if re.fullmatch(r"\d{2}", text) and shape.top / EMU > prs.slide_height / EMU - 0.7:
                    numbers.append((index, text))
        # 封面和结束页没有页脚，其余页的页码必须等于页序
        self.assertEqual([text for _, text in numbers], [f"{i:02d}" for i, _ in numbers])

    def test_keeps_canvas_size(self):
        """4:3 的稿子插页之后还是 4:3，不能被新页悄悄改成 16:9。"""
        path = self.build(size="4:3", margin="wide")
        self.edit(path, [{"op": "insert", "after": 0, "content": {"layout": "bullets", "title": "首页之后", "items": ["一条"]}}])

        prs = Presentation(str(path))
        self.assertAlmostEqual(prs.slide_width / EMU, 10.0, places=2)
        # after=0 表示插到最前面，新页就是第一页；标题要沿用原来那 0.9 英寸的宽边距
        title = next(s for s in prs.slides[0].shapes if s.has_text_frame and "首页之后" in s.text_frame.text)
        self.assertAlmostEqual(title.left / EMU, 0.9, places=2)

    def test_content_is_validated(self):
        path = self.build()
        text = self.edit(path, [{"op": "insert", "after": 1, "content": {"layout": "kpi", "title": "缺指标"}}])
        self.assertIn("错误", text)
        self.assertIn("items", text)
        self.assertEqual(len(_pages(path)), 5)

    def test_after_out_of_range(self):
        path = self.build()
        text = self.edit(path, [{"op": "insert", "after": 99, "content": {"layout": "closing", "title": "结尾"}}])
        self.assertIn("after 要在 0~5", text)


class TestReplace(_Base):
    def test_text_replaced_and_font_kept(self):
        path = self.build()
        before = Presentation(str(path)).slides[1]
        original = next(s for s in before.shapes if s.has_text_frame and "营收 +12%" in s.text_frame.text)
        run = original.text_frame.paragraphs[0].runs[0]
        font, size = run.font.name, run.font.size.pt

        text = self.edit(path, [{"op": "replace", "page": 2, "block": 3, "text": "营收 +18%"}])

        self.assertIn("已替换", text)
        slide = Presentation(str(path)).slides[1]
        shape = next(s for s in slide.shapes if s.has_text_frame and "营收 +18%" in s.text_frame.text)
        new_run = shape.text_frame.paragraphs[0].runs[0]
        self.assertEqual(new_run.font.name, font)
        self.assertEqual(new_run.font.size.pt, size)

    def test_long_text_shrinks_instead_of_overflowing(self):
        """换成长文字后如果放不下，字号要降下来，而不是压出框外。"""
        path = self.build(size="4:3", margin="wide")
        long_text = "营收同比增长百分之十二点四，增量主要来自华东与华南的新开门店，以及客单价的回升" * 10

        text = self.edit(path, [{"op": "replace", "page": 2, "block": 3, "text": long_text}])

        self.assertIn("字号压到", text)
        shape = next(
            s for s in Presentation(str(path)).slides[1].shapes
            if s.has_text_frame and long_text[:10] in s.text_frame.text
        )
        sizes = [r.font.size.pt for p in shape.text_frame.paragraphs for r in p.runs if r.font.size]
        self.assertLess(max(sizes), 18)
        self.assertGreaterEqual(min(sizes), 11)
        # 压完还得真的放得下，不能只是"字号变小了"就算过
        self.assertLessEqual(pt._shape_text_height(shape, shape.width / EMU), shape.height / EMU)

    def test_block_that_cannot_fit_is_reported(self):
        """小框里塞长句：压到 11pt 也放不下时必须说出来，不能让用户自己去发现。"""
        path = self.build()
        long_text = "季度营业收入与去年同期相比增长百分之十二点四" * 4

        text = self.edit(path, [{"op": "replace", "page": 3, "block": 3, "text": long_text}])

        self.assertIn("仍然超出", text)
        self.assertIn("建议把文字缩短", text)

    def test_block_out_of_range(self):
        path = self.build()
        text = self.edit(path, [{"op": "replace", "page": 2, "block": 99, "text": "x"}])
        self.assertIn("block 收到 99", text)

    def test_table_cell(self):
        path = self.build()
        text = self.edit(path, [{"op": "replace", "page": 4, "block": 2, "cell": [1, 1], "text": "3,900"}])
        self.assertIn("表格 [1,1] 已替换", text)

        table = next(s.table for s in Presentation(str(path)).slides[3].shapes if s.has_table)
        self.assertEqual(table.cell(1, 1).text, "3,900")
        self.assertEqual(table.cell(0, 1).text, "销售额")  # 表头不能被顺手改掉

    def test_table_needs_cell(self):
        """整块换表格文字做不到，报错要说清改用 cell。"""
        path = self.build()
        text = self.edit(path, [{"op": "replace", "page": 4, "block": 2, "text": "整块换"}])
        self.assertIn("cell", text)
        self.assertIn("表格", text)


class TestDeleteMove(_Base):
    def test_delete(self):
        path = self.build()
        text = self.edit(path, [{"op": "delete", "pages": [2]}])
        self.assertIn("删除第 2 页", text)
        self.assertEqual(len(_pages(path)), 4)
        self.assertNotIn("核心结论", "".join("".join(p) for p in _pages(path)))

    def test_cannot_delete_everything(self):
        path = self.build()
        text = self.edit(path, [{"op": "delete", "pages": [1, 2, 3, 4, 5]}])
        self.assertIn("至少要留一页", text)
        self.assertEqual(len(_pages(path)), 5)

    def test_move(self):
        path = self.build()
        text = self.edit(path, [{"op": "move", "page": 5, "to": 2}])
        self.assertIn("挪到第 2 页", text)
        self.assertIn("谢谢", _pages(path)[1])


class TestSafety(_Base):
    def test_backup_written(self):
        path = self.build()
        self.edit(path, [{"op": "delete", "pages": [2]}])
        backups = list(path.parent.glob(f"{path.stem}.bak-*.pptx"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(len(_pages(backups[0])), 5)  # 备份是改之前的五页

    def test_backup_can_be_off(self):
        path = self.build()
        self.edit(path, [{"op": "delete", "pages": [2]}], backup=False)
        self.assertEqual(list(path.parent.glob(f"{path.stem}.bak-*.pptx")), [])

    def test_failed_op_leaves_file_untouched(self):
        """第二个操作不合法时，第一个操作也不能生效——不能只改一半。"""
        path = self.build()
        before = _digest(path)

        text = self.edit(path, [
            {"op": "delete", "pages": [2]},
            {"op": "replace", "page": 99, "block": 1, "text": "x"},
        ])

        self.assertIn("文件没有改动", text)
        self.assertEqual(_digest(path), before)
        self.assertEqual(list(path.parent.glob(f"{path.stem}.bak-*.pptx")), [])

    def test_read_only_mode_refused(self):
        path = self.build()
        before = _digest(path)
        files.set_mode("read")

        text = self.edit(path, [{"op": "delete", "pages": [2]}])

        self.assertIn("只读", text)
        self.assertEqual(_digest(path), before)

    def test_wrong_suffix(self):
        path = self.build()
        other = self.tmp / "笔记.txt"
        other.write_text("x", encoding="utf-8")
        self.assertIn("只能改 .pptx", self.edit(other, [{"op": "delete", "pages": [1]}]))

    def test_empty_ops(self):
        path = self.build()
        self.assertIn("ops 不能为空", self.edit(path, []))


class TestDescribe(_Base):
    def read(self, path: Path, **kwargs) -> str:
        return registry.call("read_file", {"path": str(path), **kwargs})

    def test_outline_has_pages_and_blocks(self):
        path = self.build()
        text = self.read(path)
        self.assertIn("5 页", text)
        self.assertIn("【第 1 页】cover", text)
        self.assertIn("[3] item", text)
        self.assertIn("block", text)  # 提示里要说清块号怎么用

    def test_theme_probed_from_file(self):
        path = self.build(theme="academic", size="4:3")
        text = self.read(path)
        self.assertIn("学术灰", text)
        self.assertIn("4:3", text)

    def test_part_paging(self):
        path = self.build()
        first = self.read(path, part=1)
        self.assertIn("第 1/", first)

    def test_missing_file(self):
        self.assertIn("错误", self.read(self.tmp / "没有这个.pptx"))


class TestSchema(unittest.TestCase):
    def test_ops_carry_field_names(self):
        spec = build_tool_spec(pt.edit_pptx)
        ops = spec["function"]["parameters"]["properties"]["ops"]
        item = ops["items"]
        self.assertEqual(item["properties"]["op"]["enum"], ["insert", "replace", "delete", "move"])
        self.assertIn("page", item["properties"])
        self.assertIn("block", item["properties"])
        # 插入页的字段要能展开到 layout 一级，否则模型只能猜
        self.assertIn("layout", item["properties"]["content"]["properties"])
