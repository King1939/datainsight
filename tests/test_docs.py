"""资料文件（txt / log / pdf / docx / pptx）的三层测试。

跟代码结构一一对应：
  1. 抽取层 readers.extract —— 后缀分发、编码探测、二进制拒绝、分块
  2. 登记层 data_tools      —— 落盘、同名去重、会话隔离、删除
  3. 工具层 registry.call   —— 给模型看的文本里有没有"错在哪、下一步怎么做"

第 3 层最容易被忽略，但它才是决定 Agent 能不能自己纠错的地方：
工具返回一句"失败"和返回"没有这个文件，已上传的是 A、B"，模型下一轮的表现完全不同。

docx / pptx / pypdf 都是可选依赖，没装就跳过对应用例，不影响其它测试。
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))   # 直接 -m unittest tests/test_docs.py 时也能 import _isolation

from app import readers, uploads  # noqa: E402
from app.tools import data_tools as dt  # noqa: E402
from app.tools import registry  # noqa: E402
from _isolation import isolate_embeddings  # noqa: E402

KEY = "ut_docs"


def setUpModule() -> None:
    # 本机 .env 配了真实向量模型。不隔离的话，search_document 会走混合检索真的发
    # HTTP 请求：测试变慢、要联网，还会往真实 data/vault/embeddings.db 写向量。
    # 隔离后 embedding 不可用 -> 自动退回纯 BM25，这些用例仍是纯离线确定性。
    isolate_embeddings()


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


HAS_DOCX = _has("docx")
HAS_PPTX = _has("pptx")
HAS_PDF = _has("pypdf")


def _make_pdf(text: str) -> bytes:
    """手搓一个最小 PDF。

    为什么不用 reportlab：只为验证"抽取链路通不通"，多装一个库不值得。
    这个 PDF 没有压缩、只有一页一行文字，pypdf 能正常解析。
    """
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


class TempFiles(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, name: str, data: bytes) -> Path:
        path = self.dir / name
        path.write_bytes(data)
        return path


class TestRouting(unittest.TestCase):
    """一个上传框收所有文件，按后缀决定走表还是走资料。

    这条规则必须稳定：同一个文件传两次要得到同一种处理，
    否则用户没法解释"为什么这次它变成了表、上次不是"。
    """

    def test_csv_goes_to_table(self):
        self.assertEqual(readers.route("销售.csv"), "table")
        self.assertEqual(readers.route("DATA.TSV"), "table")

    def test_documents_go_to_text(self):
        for name in ("a.pdf", "b.docx", "c.pptx", "d.txt", "e.log", "f.md", "g.json"):
            self.assertEqual(readers.route(name), "document", name)

    def test_unknown_suffix(self):
        self.assertEqual(readers.route("x.xyz"), "unsupported")
        self.assertEqual(readers.route("没有后缀"), "unsupported")

    def test_csv_is_in_upload_list(self):
        """CSV 也走同一个上传框，只是分派到"表"那条路。

        上传框的 accept 列表里必须包含它，否则用户在文件选择器里根本看不到自己的 csv。
        """
        self.assertIn("csv", readers.UPLOAD_SUFFIXES)
        self.assertIn("pdf", readers.UPLOAD_SUFFIXES)


# ---------------------------------------------------------------- 抽取层


class TestTextExtraction(TempFiles):
    def test_plain_txt(self):
        path = self.write("a.txt", "第一行\n第二行".encode("utf-8"))
        doc = readers.extract(path)
        self.assertIn("第一行", doc.text)
        self.assertEqual(doc.kind, "txt")

    def test_gbk_log(self):
        """中文 Windows 的日志多半是 GBK，用 utf-8 硬读会抛异常。"""
        path = self.write("run.log", "2025-01-01 启动失败\n原因：缺少配置".encode("gbk"))
        doc = readers.extract(path)
        self.assertIn("启动失败", doc.text)
        self.assertEqual(doc.kind, "log")

    def test_binary_disguised_as_txt(self):
        """二进制文件如果被 latin-1 硬解成乱码，模型会对着垃圾认真分析。"""
        path = self.write("fake.txt", b"\x00\x01\x02\x03" * 50)
        with self.assertRaises(readers.DocumentError) as ctx:
            readers.extract(path)
        self.assertIn("二进制", str(ctx.exception))

    def test_unknown_suffix(self):
        path = self.write("a.xyz", b"hello")
        with self.assertRaises(readers.DocumentError) as ctx:
            readers.extract(path)
        self.assertIn("不支持", str(ctx.exception))

    def test_empty_file(self):
        path = self.write("empty.txt", b"")
        with self.assertRaises(readers.DocumentError) as ctx:
            readers.extract(path)
        self.assertIn("空", str(ctx.exception))

    def test_crlf_normalized(self):
        path = self.write("a.txt", b"a\r\nb\r\n\r\n\r\n\r\nc")
        doc = readers.extract(path)
        self.assertNotIn("\r", doc.text)
        self.assertNotIn("\n\n\n", doc.text)

    def test_chunks_break_on_sentence(self):
        text = "这是一句话。" * 600  # 3000 字，应切成两块
        chunks = readers.split_chunks(text, size=1000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.endswith("。"), chunk[-10:])


class TestStructuredExtraction(TempFiles):
    @unittest.skipUnless(HAS_DOCX, "需要 python-docx")
    def test_docx_paragraphs_and_tables(self):
        from docx import Document as DocxDocument

        document = DocxDocument()
        document.add_paragraph("年度报告")
        document.add_paragraph("营收同比增长百分之十二")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "季度"
        table.cell(0, 1).text = "金额"
        table.cell(1, 0).text = "Q1"
        table.cell(1, 1).text = "120"
        path = self.dir / "report.docx"
        document.save(str(path))

        doc = readers.extract(path)
        self.assertIn("营收同比增长", doc.text)
        # 表格文字不在 paragraphs 里，必须单独取 —— 这条就是防这个坑的
        self.assertIn("Q1", doc.text)
        self.assertIn("120", doc.text)

    @unittest.skipUnless(HAS_PPTX, "需要 python-pptx")
    def test_pptx_slides_and_notes(self):
        from pptx import Presentation

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "第三季度复盘"
        slide.placeholders[1].text = "收入上升\n成本持平"
        slide.notes_slide.notes_text_frame.text = "这里要强调毛利下滑"
        path = self.dir / "deck.pptx"
        prs.save(str(path))

        doc = readers.extract(path)
        self.assertIn("第三季度复盘", doc.text)
        self.assertIn("收入上升", doc.text)
        self.assertIn("毛利下滑", doc.text)
        self.assertIn("第 1 页", doc.text)

    @unittest.skipUnless(HAS_PDF, "需要 pypdf")
    def test_pdf_text(self):
        path = self.write("brief.pdf", _make_pdf("Hello PDF"))
        doc = readers.extract(path)
        self.assertIn("Hello PDF", doc.text)
        self.assertIn("第 1 页", doc.text)

    def test_legacy_doc_rejected_with_advice(self):
        """老 .doc 是二进制格式，明确拒绝并告诉用户该怎么办。"""
        path = self.write("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 200)
        with self.assertRaises(readers.DocumentError) as ctx:
            readers.extract(path)
        self.assertIn("docx", str(ctx.exception))

    def test_doc_that_is_really_text(self):
        path = self.write("fake.doc", "其实是纯文本".encode("utf-8"))
        doc = readers.extract(path)
        self.assertIn("其实是纯文本", doc.text)
        self.assertIn("纯文本", doc.note)

    def test_truncation_note(self):
        original = readers.MAX_DOC_CHARS
        readers.MAX_DOC_CHARS = 100
        try:
            path = self.write("long.txt", ("字" * 500).encode("utf-8"))
            doc = readers.extract(path)
        finally:
            readers.MAX_DOC_CHARS = original
        self.assertEqual(len(doc.text), 100)
        self.assertIn("截断", doc.note)


# ---------------------------------------------------------------- 登记层


class TestDocumentRegistry(unittest.TestCase):
    def setUp(self) -> None:
        dt.clear_documents(KEY)
        dt.use_dataset(KEY)

    def tearDown(self) -> None:
        dt.clear_documents(KEY)

    def test_add_and_list(self):
        doc = dt.add_document("notes.txt", "会议记录：下周上线".encode("utf-8"), key=KEY)
        self.assertEqual(doc.name, "notes.txt")
        self.assertTrue(Path(doc.path).exists())
        self.assertEqual([d.name for d in dt.documents(KEY)], ["notes.txt"])

    def test_duplicate_name_kept_as_new_copy(self):
        dt.add_document("a.txt", b"v1", key=KEY)
        second = dt.add_document("a.txt", b"v2", key=KEY)
        self.assertEqual(second.name, "a(2).txt")
        self.assertEqual(len(dt.documents(KEY)), 2)

    def test_sessions_isolated(self):
        dt.add_document("a.txt", b"only in A", key=KEY)
        self.assertEqual(dt.documents("ut_docs_other"), [])
        self.assertTrue(dt.has_documents(KEY))
        self.assertFalse(dt.has_documents("ut_docs_other"))

    def test_remove_deletes_file(self):
        doc = dt.add_document("temp.txt", b"bye", key=KEY)
        path = Path(doc.path)
        self.assertTrue(dt.remove_document("temp.txt", KEY))
        self.assertFalse(path.exists())
        self.assertFalse(dt.has_documents(KEY))

    def test_size_limit(self):
        from unittest import mock

        with mock.patch.object(dt, "MAX_UPLOAD_MB", 0):
            with self.assertRaises(readers.DocumentError) as ctx:
                dt.add_document("big.txt", b"x" * 1024, key=KEY)
        self.assertIn("MB", str(ctx.exception))

    def test_path_traversal_neutralized(self):
        """名字里的路径成分必须被剥掉，不能落到 uploads 目录外面。"""
        doc = dt.add_document("../../evil.txt", b"x", key=KEY)
        self.assertEqual(doc.name, "evil.txt")
        self.assertEqual(Path(doc.path).parent.name, KEY)

    def test_unsupported_file_not_left_on_disk(self):
        folder = uploads.doc_folder(KEY)
        before = set(folder.glob("*")) if folder.exists() else set()
        with self.assertRaises(readers.DocumentError):
            dt.add_document("x.xyz", b"whatever", key=KEY)
        after = set(folder.glob("*")) if folder.exists() else set()
        self.assertEqual(before, after, "抽取失败的文件不能留在 uploads 里")


# ---------------------------------------------------------------- 工具层


class TestDocumentTools(unittest.TestCase):
    def setUp(self) -> None:
        dt.clear_documents(KEY)
        dt.use_dataset(KEY)
        body = "第一章 概述\n" + "本季度收入增长明显。" * 300 + "\n关键错误：ERROR 502 网关超时"
        dt.add_document("报告.txt", body.encode("utf-8"), key=KEY)

    def tearDown(self) -> None:
        dt.clear_documents(KEY)

    def test_list_when_empty(self):
        dt.clear_documents(KEY)
        text = registry.call("list_documents", {})
        self.assertIn("还没有上传", text)

    def test_list_shows_name_and_parts(self):
        text = registry.call("list_documents", {})
        self.assertIn("报告.txt", text)
        self.assertIn("段", text)
        self.assertIn("read_document", text)

    def test_read_first_part(self):
        text = registry.call("read_document", {"name": "报告.txt"})
        self.assertIn("第一章", text)
        self.assertIn("第 1/", text)
        self.assertIn("part=2", text)

    def test_read_out_of_range(self):
        text = registry.call("read_document", {"name": "报告.txt", "part": 99})
        self.assertIn("错误", text)
        self.assertIn("99", text)

    def test_read_missing_lists_alternatives(self):
        text = registry.call("read_document", {"name": "不存在.pdf"})
        self.assertIn("没有名为", text)
        self.assertIn("报告.txt", text)

    def test_search_hits(self):
        text = registry.call("search_document", {"name": "报告.txt", "keyword": "ERROR"})
        self.assertIn("命中", text)
        self.assertIn("502", text)

    def test_search_case_insensitive(self):
        text = registry.call("search_document", {"name": "报告.txt", "keyword": "error"})
        self.assertIn("502", text)

    def test_search_miss_gives_next_step(self):
        text = registry.call("search_document", {"name": "报告.txt", "keyword": "量子计算"})
        self.assertIn("没找到", text)
        self.assertIn("read_document", text)

    def test_bad_argument_rejected_by_registry(self):
        text = registry.call("read_document", {"file": "报告.txt"})
        self.assertIn("不认识参数", text)

    def test_tools_do_not_leak_other_session(self):
        dt.use_dataset("ut_docs_empty")
        self.assertIn("还没有上传", registry.call("list_documents", {}))


# ---------------------------------------------------------------- 检索（跨文件 / 多词 / 候选卡片）


class TestSearchScopeAndRanking(unittest.TestCase):
    """search_document 升级成"跨文件 + 多关键词 + 候选卡片"后的契约。

    这四条是这一版存在的理由，改坏了就等于退回旧版：
      1. 不给 name 能搜到第二份文件（省掉一轮 list_documents 猜测）
      2. 多关键词是 OR（"违约金 赔偿" 不要求原文连续出现这个长串）
      3. 命中词多的段落排在前面（否则候选卡片等于随机给三段）
      4. 返回的是片段不是整段（这是 token 下降的直接来源）
    """

    def setUp(self) -> None:
        # 先清盘再清登记：clear_documents 只删它认识的路径，上一轮如果留下孤儿文件，
        # 这一轮的同名上传就会被改叫 alpha(2).txt，断言里写死文件名会随机红。
        uploads.discard(KEY)
        dt.clear_documents(KEY)
        dt.use_dataset(KEY)
        # 第一章只提"赔偿"；第二章三个词全都有。用来验证排序不是按段号。
        # 末尾那句"是否存在"是给模糊降级的假命中测试准备的：它是个常见字面，
        # 但不该把"不存在的词甲乙丙"这种查询变成"有结果"。
        alpha = (
            "第一章 赔偿说明（资产是否存在需核实）\n"
            + "本节介绍背景。" * 200
            + "\n第二章 违约责任\n"
            + "甲方违反约定应承担责任。" * 150
            + "\n违约金为合同总额的百分之十，赔偿上限不超过总价。"
        )
        dt.add_document("alpha.txt", alpha.encode("utf-8"), key=KEY)
        dt.add_document(
            "beta.txt", ("量子计算入门\n" + "叠加态与纠缠。" * 200).encode("utf-8"), key=KEY
        )
        dt.add_document(
            "gamma.txt", ("区块链的创新应用\n" + "案例集。" * 100).encode("utf-8"), key=KEY
        )
        # 给"字散落各处"准备的：共识和机制各出现一次，中间隔两千多字，
        # 但"共识机制"这个词整串从未出现。
        dt.add_document(
            "delta.txt",
            ("共识需要多方确认。\n" + "本节介绍背景。" * 200 + "\n激励机制的设计。").encode("utf-8"),
            key=KEY,
        )

    def tearDown(self) -> None:
        dt.clear_documents(KEY)
        uploads.discard(KEY)

    def test_search_without_name_covers_all_documents(self):
        text = registry.call("search_document", {"keyword": "量子"})
        self.assertIn("beta.txt", text)
        self.assertIn("份资料", text, "不指定 name 时要说清搜了几份，模型才知道范围")

    def test_multiple_keywords_are_or(self):
        text = registry.call("search_document", {"keyword": "量子 违约金"})
        self.assertIn("beta.txt", text)
        self.assertIn("alpha.txt", text)

    def test_ranking_prefers_chunk_with_more_terms(self):
        text = registry.call("search_document", {"keyword": "违约金 赔偿 上限"})
        blocks = text.split("\n[")
        self.assertGreaterEqual(len(blocks), 3, "至少两段候选")
        top = blocks[1]
        self.assertIn("alpha.txt", top)
        self.assertIn("违约金", top, "命中三个词的那段必须排第一，而不是按段号")
        self.assertIn("3 个词", top)

    def test_returns_snippet_not_whole_chunk(self):
        text = registry.call("search_document", {"keyword": "赔偿"})
        self.assertIn("read_document", text)
        # 一段是 2000 字，填一句 7 字的话会重复约 285 次。卡片只给一百来字，
        # 所以这里允许十来次——要挡住的是"把整段倒回去"这种退化。
        self.assertLess(len(text), 1200)
        self.assertLess(text.count("本节介绍背景"), 30)

    def test_exact_miss_falls_back_to_fuzzy_and_says_so(self):
        # 原文是"区块链的创新应用"，问法少了"的"：整词一个都命中不了
        text = registry.call("search_document", {"keyword": "区块链创新应用"})
        self.assertIn("gamma.txt", text)
        self.assertIn("模糊", text, "降级结果必须标注，不能伪装成精确命中")

    def test_fuzzy_does_not_invent_hits_for_nonsense(self):
        # "存在"是极常见的字面，按覆盖率挡掉之前，这句会返回几十段候选，
        # 模型就会当成"文档里真在讲甲乙丙"。宁可说没找到。
        text = registry.call("search_document", {"keyword": "不存在的词甲乙丙"})
        self.assertIn("没找到", text)
        self.assertNotIn("候选", text)

    def test_fuzzy_rejects_scattered_bigrams(self):
        # 覆盖率能过（2/3），但"共识"和"机制"隔了两千字——不是同一个词。
        # 真实语料里踩到过：《区块链教程》里"共识机制"出现 0 次，
        # 却因为这两个词各出现几次被判成命中。
        text = registry.call("search_document", {"keyword": "共识机制", "name": "delta.txt"})
        self.assertIn("没找到", text)

    def test_unknown_name_lists_alternatives(self):
        text = registry.call("search_document", {"keyword": "x", "name": "不存在.pdf"})
        self.assertIn("没有名为", text)
        self.assertIn("alpha.txt", text)

    def test_keyword_still_required(self):
        text = registry.call("search_document", {"name": "alpha.txt"})
        self.assertIn("缺少必填参数", text)


# ---------------------------------------------------------------- prompt


class TestPromptWithFiles(unittest.TestCase):
    def test_files_only_does_not_mention_table(self):
        from app.agent import make_system_prompt

        prompt = make_system_prompt(
            [], None, "x", [{"name": "a.pdf", "kind": "pdf", "chars": 1200, "parts": 1}]
        )
        self.assertIn("a.pdf", prompt)
        self.assertIn("read_document", prompt)
        # 没有表的时候连"数据表"这一节都不能出现：模型看到一节空的表描述，
        # 就会去调 run_sql，然后连续几步都在报错。
        self.assertNotIn("## 数据表", prompt)
        self.assertNotIn("run_sql", prompt)

    def test_table_only_does_not_mention_documents(self):
        from app.agent import make_system_prompt

        prompt = make_system_prompt(["城市", "销售额"], 10, "d.csv")
        self.assertIn("城市", prompt)
        self.assertNotIn("read_document", prompt)

    def test_both_present(self):
        from app.agent import make_system_prompt

        prompt = make_system_prompt(
            ["城市"],
            10,
            "d.csv",
            [{"name": "a.pdf", "kind": "pdf", "chars": 1200, "parts": 1}],
        )
        self.assertIn("run_sql", prompt)
        self.assertIn("read_document", prompt)
        self.assertIn("a.pdf", prompt)

    def test_table_name_comes_from_config(self):
        """prompt 里的表名必须跟配置一致。

        以前这里是写死的 `sales`——改配置不改文案，模型就会去查一张不存在的表，
        而报错信息看起来像是 SQL 写错了。
        """
        from app.agent import make_system_prompt
        from app.config import TABLE_NAME

        prompt = make_system_prompt(["城市"], 10, "d.csv")
        self.assertIn(f"`{TABLE_NAME}`", prompt)
        self.assertNotIn("`sales`", prompt)

    def test_fallback_prompt_has_no_table(self):
        """没有数据时的兜底 prompt 不能假装有一张表。"""
        from app.agent import SYSTEM_PROMPT

        self.assertNotIn("sales", SYSTEM_PROMPT)
        self.assertNotIn("电商", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
