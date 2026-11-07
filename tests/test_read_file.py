"""read_file 的回归测试：一个入口读所有本地文件。

以前它叫 read_text_file，只能解码文本。模型拿着桌面上的 .pptx 去读，被判成
"二进制文件"，于是转头去 run_python 里翻 zipfile——那条路走不通，最后靠
sys.modules 才勉强解开内容。工具能读的格式越少，模型越会绕道去用通用工具硬扛。

所以这里测的不是"能不能解码"，而是**每种格式都被接住了，且接不住时说清了原因**。
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from app import files
from app.tools import registry

HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None
HAS_DOCX = importlib.util.find_spec("docx") is not None


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        files.set_mode("read")  # 读文件在只读模式下也必须能用

    def write(self, name: str, data: bytes | str) -> Path:
        path = self.dir / name
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
        return path

    def read(self, path: Path, **kwargs) -> str:
        return registry.call("read_file", {"path": str(path), **kwargs})


class TestText(_Base):
    def test_plain_text(self):
        path = self.write("note.txt", "第一行\n第二行\n")
        text = self.read(path)
        self.assertIn("第一行", text)
        self.assertIn("txt", text)

    def test_gbk_file(self):
        """中文 Windows 导出的 txt 多半是 GBK，读成乱码等于没读。"""
        path = self.dir / "gbk.log"
        path.write_bytes("订单异常：库存不足".encode("gbk"))
        self.assertIn("库存不足", self.read(path))

    def test_unknown_suffix_falls_back_to_text(self):
        """不认识的扩展名也值得一试，总比直接说"不支持"强。"""
        path = self.write("config.toml", 'name = "演示"\n')
        self.assertIn("演示", self.read(path))

    def test_binary_is_rejected_with_reason(self):
        path = self.write("blob.bin", bytes(range(256)) * 8)
        text = self.read(path)
        self.assertIn("错误", text)

    def test_missing_file_names_the_next_step(self):
        text = self.read(self.dir / "不存在.txt")
        self.assertIn("错误", text)
        self.assertIn("list_files", text)

    def test_long_text_is_paged(self):
        path = self.write("long.txt", ("这是一段用来把文件撑长的文字。" * 3000))
        first = self.read(path, part=1)
        self.assertIn("第 1/", first)
        self.assertIn("part=2", first)

        pieces = int(first.split("第 1/")[1].split(" 段")[0])
        last = self.read(path, part=pieces)
        self.assertIn(f"第 {pieces}/{pieces} 段", last)

    def test_part_out_of_range(self):
        path = self.write("short.txt", "一句话")
        self.assertIn("part 必须在 1 到 1 之间", self.read(path, part=9))

    def test_empty_file(self):
        path = self.write("empty.txt", "")
        self.assertIn("错误", self.read(path))


class TestExcel(_Base):
    @unittest.skipUnless(HAS_OPENPYXL, "需要 openpyxl")
    def test_sheet_content(self):
        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        sheet.title = "分城市"
        sheet.append(["城市", "营收"])
        sheet.append(["上海", 486])
        sheet.append(["杭州", 312])
        book.create_sheet("备注").append(["口径说明"])
        path = self.dir / "数据.xlsx"
        book.save(str(path))

        text = self.read(path)
        self.assertIn("【分城市】", text)
        self.assertIn("城市 | 营收", text)
        self.assertIn("上海 | 486", text)
        self.assertIn("【备注】", text)
        # 读 Excel 不是为了让模型算数，要算数得转成 CSV 走 SQL
        self.assertIn("CSV", text)

    @unittest.skipUnless(HAS_OPENPYXL, "需要 openpyxl")
    def test_rows_are_capped_with_notice(self):
        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        for index in range(120):
            sheet.append([f"第{index}行", index])
        path = self.dir / "长表.xlsx"
        book.save(str(path))

        text = self.read(path)
        self.assertIn("只列了前", text)
        self.assertNotIn("第99行", text)

    def test_legacy_xls_is_refused_with_hint(self):
        path = self.write("老表.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600)
        text = self.read(path)
        self.assertIn(".xlsx", text)


class TestWord(_Base):
    @unittest.skipUnless(HAS_DOCX, "需要 python-docx")
    def test_paragraphs_and_tables(self):
        from docx import Document as DocxDocument

        document = DocxDocument()
        document.add_paragraph("项目背景")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "城市"
        table.cell(0, 1).text = "营收"
        table.cell(1, 0).text = "上海"
        table.cell(1, 1).text = "486"
        path = self.dir / "报告.docx"
        document.save(str(path))

        text = self.read(path)
        self.assertIn("项目背景", text)
        self.assertIn("上海 | 486", text)


class TestDispatch(_Base):
    def test_directory_is_not_a_file(self):
        text = self.read(self.dir)
        self.assertIn("是文件夹", text)
        self.assertIn("list_files", text)

    def test_supported_suffixes_include_office(self):
        from app import readers

        for name in ("a.xlsx", "b.pptx", "c.docx", "d.pdf", "e.txt"):
            self.assertEqual(readers.route(name), "document", name)
