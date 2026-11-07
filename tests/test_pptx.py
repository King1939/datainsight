"""make_pptx 的回归测试。

重点不是"能不能生成文件"（能生成但排得乱没有意义），而是三件事：
  1. 七种版式每种都排得出来，页数对得上；
  2. 内容放不下时**降档**而不是溢出——这条用几何检查兜底，比肉眼看靠谱；
  3. schema 里真的带上了每页的字段名，否则模型只能猜。
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pptx import Presentation
from pptx.util import Inches

import app.tools.pptx_tools as pt
from app.tools import registry
from app.ui_render import extract_chart, extract_file

EMU = 914400


def _slides(**overrides) -> list[dict]:
    base = [
        {"layout": "cover", "title": "季度经营分析", "subtitle": "2026-11-05"},
        {"layout": "section", "number": "01", "title": "整体表现", "subtitle": "营收与结构"},
        {"layout": "bullets", "title": "核心结论", "lead": "营收继续增长", "items": ["营收 +12%", "华东 46%"]},
        {"layout": "kpi", "title": "关键指标", "items": ["1,284|营收|+12%", "8.7|客单价|-3%"]},
        {"layout": "table", "title": "分城市", "headers": ["城市", "销售额"], "rows": [["上海", "486"], ["杭州", "312"]]},
        {"layout": "compare", "title": "问题与对策", "left_title": "问题", "left_items": ["失败率 3.2%"],
         "right_title": "动作", "right_items": ["加自动重试"]},
        {"layout": "closing", "title": "谢谢", "subtitle": "欢迎提问"},
    ]
    for page in base:
        page.update(overrides.get(page["layout"], {}))
    return base


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        patch = mock.patch.object(pt, "OUTPUT_DIR", self.tmp)
        patch.start()
        self.addCleanup(patch.stop)

    def build(self, slides, **kwargs):
        text = registry.call("make_pptx", {"title": "测试文稿", "slides": slides, **kwargs})
        return text

    def opened(self, text: str) -> Presentation:
        match = re.search(r"演示文稿已生成：(.+?\.pptx)", text)
        self.assertIsNotNone(match, text)
        return Presentation(match.group(1))

    def shapes_of(self, slide):
        return list(slide.shapes)


class TestLayouts(_Base):
    def test_all_seven_layouts(self):
        text = self.build(_slides())
        self.assertIn("7 页", text)
        prs = self.opened(text)
        self.assertEqual(len(prs.slides), 7)
        self.assertEqual([p["layout"] for p in _slides()], list(pt.LAYOUTS))

    def test_footer_only_on_content_pages(self):
        """封面、章节页、结束页不加页码——这是规范里最容易被忽略的一条。

        只看"是不是两位数"会误判：章节页的「01」也是两位数，但它在页面中部。
        位置才是判据——页码必须贴着底边右侧。
        """
        prs = self.opened(self.build(_slides()))
        width, height = prs.slide_width / EMU, prs.slide_height / EMU
        numbered = []
        for slide in prs.slides:
            footers = [
                s
                for s in slide.shapes
                if s.has_text_frame
                and re.fullmatch(r"\d{2}", s.text_frame.text.strip())
                and s.top / EMU > height - 0.7
                and s.left / EMU > width / 2
            ]
            numbered.append(bool(footers))
        self.assertEqual(numbered, [False, False, True, True, True, True, False])

    def test_section_page_uses_deep_background(self):
        prs = self.opened(self.build(_slides()))
        slide = prs.slides[1]
        self.assertEqual(str(slide.background.fill.fore_color.rgb), pt.THEMES["business"].deep)

    def test_second_level_by_indent(self):
        """两个空格开头的是二级，缩进值必须比一级大。"""
        slides = [
            {"layout": "bullets", "title": "层级", "items": ["一级", "  二级"]},
        ]
        prs = self.opened(self.build(slides))
        box = next(s for s in prs.slides[0].shapes if s.has_text_frame and "一级" in s.text_frame.text)
        indents = []
        for paragraph in box.text_frame.paragraphs:
            pPr = paragraph._p.find(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}pPr"
            )
            indents.append(int(pPr.get("marL")))
        self.assertLess(indents[0], indents[1])


class TestGeometry(_Base):
    """排版不是"能生成就行"。这里查的是每个形状都在页面内、正文不压页脚。"""

    def _within_page(self, prs) -> list[str]:
        width, height = prs.slide_width / EMU, prs.slide_height / EMU
        problems = []
        for index, slide in enumerate(prs.slides, start=1):
            for shape in slide.shapes:
                left, top = shape.left / EMU, shape.top / EMU
                right, bottom = left + shape.width / EMU, top + shape.height / EMU
                if left < -0.01 or top < -0.01 or right > width + 0.01 or bottom > height + 0.01:
                    problems.append(f"第{index}页越界：{right:.2f}x{bottom:.2f}")
                elif bottom > height - 0.72 + 0.01 and index not in (1, 2, 7) and not self._is_footer(shape):
                    problems.append(f"第{index}页正文压到页脚区：bottom={bottom:.2f}")
        return problems

    @staticmethod
    def _is_footer(shape) -> bool:
        if not shape.has_text_frame:
            return False
        text = shape.text_frame.text.strip()
        return bool(re.fullmatch(r"\d{2}", text)) or text == "测试文稿"

    def test_dense_bullets_stay_inside(self):
        items = [f"第 {i} 条要点写得很长，用来把这一页塞满，看看文字会不会越过页脚跑出页面" for i in range(8)]
        prs = self.opened(self.build([{"layout": "bullets", "title": "满页", "items": items}]))
        self.assertEqual(self._within_page(prs), [])

    def test_dense_compare_stays_inside(self):
        items = [f"这是一条很长的对比内容第 {i} 条，用来压测两栏版式的降档逻辑是否可靠" for i in range(4)]
        prs = self.opened(
            self.build(
                [{"layout": "compare", "title": "对比", "left_title": "左", "left_items": items,
                  "right_title": "右", "right_items": items}]
            )
        )
        self.assertEqual(self._within_page(prs), [])

    def test_43_and_narrow_margin(self):
        prs = self.opened(self.build(_slides(), size="4:3", margin="narrow"))
        self.assertAlmostEqual(prs.slide_width / EMU, 10.0, places=2)
        title = next(s for s in prs.slides[2].shapes if s.has_text_frame and "核心结论" in s.text_frame.text)
        self.assertAlmostEqual(title.left / EMU, pt.MARGINS["narrow"], places=3)

    def test_dense_page_shrinks_font(self):
        """同样内容，塞得越满字号越小；不能出现"字号没降但页面溢出"。"""
        short = self.build([{"layout": "bullets", "title": "少", "items": ["一条"]}])
        long = self.build(
            [{"layout": "bullets", "title": "多", "items": [
                f"第 {i} 条：营收环比增长 12% 达到 1,284 万元，连续三个季度保持两位数增长，主要来自华东与华南的新增门店"
                for i in range(8)
            ]}]
        )

        def first_size(text):
            prs = self.opened(text)
            # 找要点框，不是标题框：标题固定 28pt，比它没意义
            box = next(s for s in prs.slides[0].shapes if s.has_text_frame and "•" in s.text_frame.text)
            return box.text_frame.paragraphs[0].runs[0].font.size.pt

        self.assertGreater(first_size(short), first_size(long))


class TestSpacing(_Base):
    """内容不满一页时，把富余高度摊进行间距；但不能摊到像双倍行距。"""

    def _gaps(self, text: str, slide_no: int = 0) -> list[float]:
        prs = self.opened(text)
        box = next(s for s in prs.slides[slide_no].shapes if s.has_text_frame and "•" in s.text_frame.text)
        return [p.space_after.pt for p in box.text_frame.paragraphs if p.space_after is not None]

    def test_sparse_page_gets_looser(self):
        sparse = self.build([{"layout": "bullets", "title": "少", "items": ["营收 +12%", "华东 46%"]}])
        dense = self.build(
            [{"layout": "bullets", "title": "多", "items": [
                f"第 {i} 条：营收环比增长 12% 达到 1,284 万元，连续三个季度保持两位数增长" for i in range(8)
            ]}]
        )
        self.assertGreater(min(self._gaps(sparse)), min(self._gaps(dense)))

    def test_gap_is_capped(self):
        """一条短要点独占一页时也不能拉到离谱——上限 24pt。"""
        text = self.build([{"layout": "bullets", "title": "一条", "items": ["营收 +12%"]}])
        self.assertLessEqual(max(self._gaps(text) or [0]), 24.0)

    def test_spread_gap_never_overflows_block(self):
        """摊间距之后整块高度仍然不能超过正文区，否则最后一行会压页脚。

        前提是内容本来就放得下：本来就超过正文区的，摊间距帮不上忙，
        该由降档处理，这里只要求它别把间距撑得更大。
        """
        for count in (2, 3, 5, 8):
            texts = [f"第 {i} 条短要点" for i in range(count)]
            width, height, size = 6.0, 4.0, 18.0
            gap = pt._spread_gap(texts, width, height, size)
            if pt._block_height(texts, width, size) > height:
                self.assertEqual(gap, 7.0)
            else:
                self.assertLessEqual(pt._block_height(texts, width, size, gap=gap), height + 1e-9)


class TestValidation(_Base):
    def test_missing_title(self):
        text = self.build([{"layout": "bullets", "items": ["a"]}])
        self.assertTrue(text.startswith("错误："))
        self.assertIn("缺少 title", text)

    def test_bullets_need_items(self):
        text = self.build([{"layout": "bullets", "title": "空"}])
        self.assertIn("缺少 items", text)

    def test_compare_needs_both_sides(self):
        text = self.build([{"layout": "compare", "title": "半截", "left_title": "左", "left_items": ["a"]}])
        self.assertIn("right_title", text)

    def test_unknown_field(self):
        """模型爱自己加字段（notes / content / heading），必须当场说清楚。"""
        text = self.build([{"layout": "cover", "title": "封面", "heading": "错的"}])
        self.assertIn("不认识的字段", text)

    def test_items_must_be_list(self):
        text = self.build([{"layout": "bullets", "title": "字符串", "items": "一条一条"}])
        self.assertIn("必须是列表", text)

    def test_kpi_count_limit(self):
        text = self.build([{"layout": "kpi", "title": "五个", "items": [f"{i}|标签" for i in range(5)]}])
        self.assertIn("1~4 个指标", text)

    def test_table_column_mismatch(self):
        text = self.build(
            [{"layout": "table", "title": "错列", "headers": ["a", "b"], "rows": [["只有一格"]]}]
        )
        self.assertIn("对不上", text)

    def test_title_too_long(self):
        text = self.build([{"layout": "cover", "title": "标题" * 25}])
        self.assertIn("超过", text)

    def test_bad_theme(self):
        text = self.build(_slides(), theme="dark")
        self.assertIn("theme 只能是", text)

    def test_empty_slides(self):
        text = self.build([])
        self.assertIn("至少要有一页", text)

    def test_too_many_pages(self):
        text = self.build([{"layout": "cover", "title": f"第{i}页"} for i in range(pt.MAX_SLIDES + 1)])
        self.assertIn("最多", text)

    def test_table_truncated_with_notice(self):
        rows = [[f"城市{i}", str(i)] for i in range(20)]
        text = self.build(
            [{"layout": "table", "title": "太多行", "headers": ["城市", "值"], "rows": rows}]
        )
        self.assertIn("注意", text)
        prs = self.opened(text)
        table = next(s for s in prs.slides[0].shapes if s.has_table).table
        self.assertLessEqual(len(table.rows) - 1, pt.MAX_ROWS)


class TestSchema(unittest.TestCase):
    def test_slides_items_carry_field_names(self):
        """模型看到的是 schema，不是文件。字段名没进 schema 就等于没定义。"""
        spec = registry.get("make_pptx").spec()["function"]["parameters"]
        item = spec["properties"]["slides"]["items"]
        self.assertEqual(item["type"], "object")
        self.assertEqual(sorted(item["properties"]), sorted(pt.Slide.__annotations__))
        self.assertEqual(item["required"], ["layout"])
        self.assertEqual(item["properties"]["layout"]["enum"], list(pt.LAYOUTS))

    def test_enums_for_theme_size_margin(self):
        spec = registry.get("make_pptx").spec()["function"]["parameters"]["properties"]
        self.assertEqual(spec["theme"]["enum"], ["business", "academic"])
        self.assertEqual(spec["size"]["enum"], ["16:9", "4:3"])
        self.assertEqual(spec["margin"]["enum"], ["narrow", "normal", "wide"])


class TestOutput(_Base):
    def test_return_text_and_file(self):
        text = self.build(_slides())
        self.assertTrue(text.startswith("演示文稿已生成："))
        self.assertIn("7 页 · 16:9 · 商务蓝", text)

    def test_filename_sanitized(self):
        text = registry.call("make_pptx", {"title": "a/b:c*?", "slides": [{"layout": "cover", "title": "x"}]})
        match = re.search(r"演示文稿已生成：(.+?\.pptx)", text)
        name = Path(match.group(1)).name
        for ch in '\\/:*?"<>|':
            self.assertNotIn(ch, name)

    def test_ui_extracts_pptx(self):
        text = self.build([{"layout": "cover", "title": "封面"}])
        path = extract_file(text)
        self.assertTrue(path and path.endswith(".pptx"))
        # 图片识别不能被 pptx 误伤（反过来也一样）
        self.assertIsNone(extract_chart(text))


if __name__ == "__main__":
    unittest.main()
