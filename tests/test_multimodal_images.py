"""文档内嵌图片与独立图片的多模态抽取、缓存与 Agent 提交测试。

全部纯离线运行：用内存中生成的 PNG / JPEG / BMP、python-docx / python-pptx / openpyxl
生成的真实文件，以及 ScriptedLLM 验证协议构造与降级逻辑。
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from PIL import Image  # noqa: E402

from _isolation import isolate_embeddings, isolate_uploads  # noqa: E402
from app import config, images, readers, vault  # noqa: E402
from app.agent import Agent, make_system_prompt  # noqa: E402
from app.llm import LLMError, LLMResponse  # noqa: E402
from app.session import Session  # noqa: E402
from app.tools import data_tools as dt  # noqa: E402
from app.tools import registry  # noqa: E402
from app.ui_render import build_item  # noqa: E402

KEY = "ut_mm_images"


def setUpModule() -> None:
    isolate_uploads("uploads-mm-")
    isolate_embeddings("emb-mm-")


def _png_bytes(size: tuple[int, int] = (40, 30), color=(20, 120, 220), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (40, 30), mode="RGB") -> bytes:
    buf = io.BytesIO()
    color = (10, 20, 30, 40) if mode == "CMYK" else (200, 80, 40)
    Image.new(mode, size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _pdf_with_jpeg(text: str, jpeg_bytes: bytes, w: int = 40, h: int = 30) -> bytes:
    """构造带真实 DCTDecode 图片 XObject 的单页最小 PDF（pypdf page.images 可直接抽取）。"""
    content = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET\nq {w} 0 0 {h} 72 600 cm /Im1 Do Q\n".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> "
        b"/Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
            f"/Length {len(jpeg_bytes)} >>\nstream\n"
        ).encode("latin-1")
        + jpeg_bytes
        + b"\nendstream",
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


class _ScriptedLLM:
    def __init__(self, script: list[dict | Exception]) -> None:
        self.script = list(script)
        self.seen: list[list[dict]] = []

    def __call__(self, messages, tools=None):
        self.seen.append(json.loads(json.dumps(messages, ensure_ascii=False)))
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return LLMResponse(message=nxt, prompt_tokens=15, completion_tokens=5, latency=0.01)


def _tool_call_msg(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


class TestImageNormalization(unittest.TestCase):
    def test_bmp_converted_to_png_and_oversized_scaled(self):
        buf = io.BytesIO()
        Image.new("RGB", (3000, 1500), (10, 200, 100)).save(buf, format="BMP")
        img = images.normalize_image(buf.getvalue(), name="chart.bmp", max_side=1000)

        self.assertIsNotNone(img)
        self.assertEqual(img.mime_type, "image/png")
        self.assertTrue(img.name.endswith(".png"))
        self.assertEqual(img.width, 1000)
        self.assertEqual(img.height, 500)
        self.assertTrue(img.to_data_url().startswith("data:image/png;base64,"))

    def test_cmyk_jpeg_converted_to_rgb_jpeg(self):
        raw = _jpeg_bytes((80, 60), mode="CMYK")
        img = images.normalize_image(raw, name="print.jpg")
        self.assertIsNotNone(img)
        self.assertEqual(img.mime_type, "image/jpeg")
        decoded = Image.open(io.BytesIO(img.read_bytes()))
        self.assertEqual(decoded.mode, "RGB")

    def test_tiny_spacer_pixel_ignored(self):
        tiny = _png_bytes((1, 1))
        self.assertIsNone(images.normalize_image(tiny, name="spacer.png"))

    def test_fallback_without_pillow_still_reads_png_and_jpeg(self):
        raw_png = _png_bytes((32, 24))
        raw_jpg = _jpeg_bytes((48, 36))
        with mock.patch.object(images, "_encode_with_pil", return_value=None):
            png = images.normalize_image(raw_png, name="a.png")
            jpg = images.normalize_image(raw_jpg, name="b.jpg")
            self.assertIsNotNone(png)
            self.assertEqual((png.mime_type, png.width, png.height), ("image/png", 32, 24))
            self.assertIsNotNone(jpg)
            self.assertEqual((jpg.mime_type, jpg.width, jpg.height), ("image/jpeg", 48, 36))

    def test_large_4mb_image_compressed_under_limit(self):
        import os

        # 构造一张 >4.1 MB 的随机像素高分辨率 PNG，验证能平滑缩放压缩到 2 MB 以内
        raw_pixels = os.urandom(1400 * 1100 * 3)
        big = Image.frombytes("RGB", (1400, 1100), raw_pixels)
        buf = io.BytesIO()
        big.save(buf, format="PNG")
        raw = buf.getvalue()
        self.assertGreater(len(raw), 4 * 1024 * 1024)

        img = images.normalize_image(raw, name="big_photo.png")
        self.assertIsNotNone(img)
        self.assertLessEqual(img.size, config.IMAGE_MAX_BYTES)
        self.assertIn(img.mime_type, ("image/png", "image/jpeg"))


class TestDocumentImageExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_standalone_image_file_supported(self):
        self.assertEqual(readers.route("趋势图.png"), "document")
        self.assertIn("png", readers.UPLOAD_SUFFIXES)

        path = self.dir / "趋势图.png"
        path.write_bytes(_png_bytes((120, 80)))
        doc = readers.extract(path)

        self.assertEqual(doc.kind, "png")
        self.assertEqual(doc.image_count, 1)
        self.assertEqual(doc.parts, 1)
        self.assertIn("[图片 1：", doc.chunks[0])
        self.assertEqual(doc.images[0].part, 1)

    def test_pdf_extracts_text_and_embedded_image(self):
        pdf_bytes = _pdf_with_jpeg("Revenue Chart", _jpeg_bytes((64, 48)))
        path = self.dir / "report.pdf"
        path.write_bytes(pdf_bytes)

        doc = readers.extract(path)
        self.assertIn("Revenue Chart", doc.text)
        self.assertEqual(doc.image_count, 1)
        self.assertEqual(doc.images[0].location, "第 1 页")
        self.assertEqual(doc.images[0].part, 1)

    def test_pdf_image_only_page_not_treated_as_empty(self):
        pdf_bytes = _pdf_with_jpeg("", _jpeg_bytes((64, 48)))
        path = self.dir / "scan.pdf"
        path.write_bytes(pdf_bytes)

        doc = readers.extract(path)
        self.assertEqual(doc.image_count, 1)
        self.assertIn("图片 1", doc.chunks[0])

    def test_docx_extracts_embedded_image_in_order_and_assigns_part(self):
        from docx import Document as DocxDocument
        from docx.shared import Inches

        document = DocxDocument()
        document.add_paragraph("第一段文字。" * 450)  # 超过 DOC_CHUNK_CHARS，撑到第 2 段
        img_path = self.dir / "chart.png"
        img_path.write_bytes(_png_bytes((60, 40)))
        document.add_paragraph("下面是附图：")
        document.add_picture(str(img_path), width=Inches(1.0))

        docx_path = self.dir / "with_img.docx"
        document.save(str(docx_path))

        doc = readers.extract(docx_path)
        self.assertEqual(doc.image_count, 1)
        self.assertGreaterEqual(doc.parts, 2)
        self.assertEqual(doc.images[0].part, doc.parts)
        self.assertIn("[图片 1：", doc.chunks[doc.images[0].part - 1])

    def test_pptx_extracts_slide_images_even_on_image_only_slide(self):
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()
        slide1 = prs.slides.add_slide(prs.slide_layouts[1])
        slide1.shapes.title.text = "概览"
        slide2 = prs.slides.add_slide(prs.slide_layouts[6])  # 空白版式纯图片页
        img_path = self.dir / "slide_pic.png"
        img_path.write_bytes(_png_bytes((90, 50)))
        slide2.shapes.add_picture(str(img_path), Inches(1), Inches(1))

        pptx_path = self.dir / "deck.pptx"
        prs.save(str(pptx_path))

        doc = readers.extract(pptx_path)
        self.assertEqual(doc.image_count, 1)
        self.assertEqual(doc.images[0].location, "第 2 页")
        self.assertIn("第 2 页", doc.text)

    def test_xlsx_extracts_embedded_images(self):
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XLImage

        wb = Workbook()
        ws = wb.active
        ws.title = "图表页"
        ws.append(["城市", "销售额"])
        ws.append(["上海", 500])
        img_path = self.dir / "xl_pic.png"
        img_path.write_bytes(_png_bytes((50, 50)))
        ws.add_image(XLImage(str(img_path)), "D2")

        xlsx_path = self.dir / "sheet.xlsx"
        wb.save(str(xlsx_path))

        doc = readers.extract(xlsx_path)
        self.assertIn("上海 | 500", doc.text)
        self.assertEqual(doc.image_count, 1)


class TestVaultImagePersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="vault-mm-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        patcher = mock.patch.object(vault, "VAULT_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_vault_caches_and_reloads_images_and_cleans_on_drop(self):
        from docx import Document as DocxDocument

        tmp = Path(tempfile.mkdtemp(prefix="docx-src-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pic = tmp / "pic.png"
        pic.write_bytes(_png_bytes((48, 32)))
        document = DocxDocument()
        document.add_paragraph("带图文档")
        document.add_picture(str(pic))
        docx_path = tmp / "report.docx"
        document.save(str(docx_path))

        entry = vault.add("report.docx", docx_path.read_bytes())
        self.assertEqual(entry.image_count, 1)
        self.assertTrue((entry.images_dir / "manifest.json").exists())

        vault._DOCS.clear()
        reloaded = vault.open_document(entry)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.image_count, 1)
        self.assertTrue(reloaded.images[0].to_data_url().startswith("data:image/png;base64,"))

        img_dir = entry.images_dir
        self.assertTrue(vault.drop(entry.id))
        self.assertFalse(img_dir.exists())

    def test_legacy_vault_entry_without_manifest_backfills_lazily(self):
        entry = vault.add("scan.pdf", _pdf_with_jpeg("Page One", _jpeg_bytes((40, 30))))
        shutil.rmtree(entry.images_dir, ignore_errors=True)
        vault._DOCS.clear()

        reloaded = vault.open_document(entry)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.image_count, 1)
        self.assertTrue((entry.images_dir / "manifest.json").exists())


class TestToolsAndAgentMultimodal(unittest.TestCase):
    def setUp(self) -> None:
        dt.clear_documents(KEY)
        dt.use_dataset(KEY)
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(dt.clear_documents, KEY)

    def test_read_document_attaches_only_current_part_images(self):
        from docx import Document as DocxDocument

        p1 = self.dir / "p1.png"
        p2 = self.dir / "p2.png"
        p1.write_bytes(_png_bytes((30, 30), (255, 0, 0)))
        p2.write_bytes(_png_bytes((30, 30), (0, 255, 0)))

        document = DocxDocument()
        document.add_paragraph("第一段内容。")
        document.add_picture(str(p1))
        document.add_paragraph("填充正文。" * 450)
        document.add_picture(str(p2))
        docx_path = self.dir / "multi.docx"
        document.save(str(docx_path))

        doc = dt.add_document("multi.docx", docx_path.read_bytes(), key=KEY)
        self.assertGreaterEqual(doc.parts, 2)

        with images.collect_step_images() as step_imgs:
            out1 = registry.call("read_document", {"name": "multi.docx", "part": 1})
        self.assertIn("已提交给模型查看", out1)
        self.assertEqual(len(step_imgs), 1)
        self.assertEqual(step_imgs[0].part, 1)

    def test_read_file_supports_png_and_pptx_images(self):
        from pptx import Presentation
        from pptx.util import Inches

        png_path = self.dir / "chart.png"
        png_path.write_bytes(_png_bytes((40, 40)))
        with images.collect_step_images() as png_imgs:
            out_png = registry.call("read_file", {"path": str(png_path)})
        self.assertIn("已提交给模型查看", out_png)
        self.assertEqual(len(png_imgs), 1)

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(str(png_path), Inches(1), Inches(1))
        pptx_path = self.dir / "deck.pptx"
        prs.save(str(pptx_path))

        with images.collect_step_images() as pptx_imgs:
            out_pptx = registry.call("read_file", {"path": str(pptx_path), "part": 1})
        self.assertIn("已提交给模型查看", out_pptx)
        self.assertEqual(len(pptx_imgs), 1)

    def test_agent_submits_image_url_without_polluting_session_messages(self):
        raw_png = _png_bytes((45, 35), (10, 80, 160))
        dt.add_document("chart.png", raw_png, key=KEY)

        llm = _ScriptedLLM(
            [
                _tool_call_msg("c1", "read_document", {"name": "chart.png", "part": 1}),
                _tool_call_msg("c2", "list_documents", {}),
                {"role": "assistant", "content": "这是一张蓝色图表。"},
            ]
        )
        events: list[tuple[str, dict]] = []
        session = Session()
        prompt = make_system_prompt(files=dt.document_digest(KEY))
        self.assertIn("含 1 张图片", prompt)

        agent = Agent(
            session,
            llm=llm,
            session_key=KEY,
            system_prompt=prompt,
            on_event=lambda k, p: events.append((k, p)),
        )
        answer = agent.ask("看看 chart.png 里有什么")

        self.assertEqual(answer, "这是一张蓝色图表。")
        # 第 2 次、第 3 次调用模型时，都应在 read_document 的 tool 结果后附带多模态 user 图片消息
        self.assertEqual(len(llm.seen), 3)
        for seen_msgs in (llm.seen[1], llm.seen[2]):
            img_msgs = [
                m
                for m in seen_msgs
                if m.get("role") == "user" and isinstance(m.get("content"), list)
            ]
            self.assertEqual(len(img_msgs), 1)
            blocks = img_msgs[0]["content"]
            img_parts = [b for b in blocks if b.get("type") == "image_url"]
            self.assertEqual(len(img_parts), 1)
            url = img_parts[0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            # Base64 可正常解码回图片
            decoded = base64.b64decode(url.split(",", 1)[1])
            self.assertGreater(len(decoded), 0)

        # 持久化 session.messages 绝不包含 list content 或 base64
        for msg in session.messages:
            self.assertIsInstance(msg.get("content", ""), str)
            self.assertNotIn("data:image/", msg.get("content", ""))

        # observation 事件携带图片元信息且 ui_render 能渲染
        obs_events = [p for k, p in events if k == "observation" and p.get("images")]
        self.assertEqual(len(obs_events), 1)
        item = build_item("observation", obs_events[0])
        self.assertIn("附带 1 张图片", item.title)

    def test_agent_falls_back_to_text_when_model_rejects_image_url(self):
        dt.add_document("chart.png", _png_bytes((40, 40)), key=KEY)
        llm = _ScriptedLLM(
            [
                _tool_call_msg("c1", "read_document", {"name": "chart.png", "part": 1}),
                LLMError("HTTP 400：unknown variant image_url", status=400, detail="image_url not supported"),
                {"role": "assistant", "content": "已按文本信息说明该文件包含一张 40x40 图片。"},
            ]
        )
        events: list[tuple[str, dict]] = []
        agent = Agent(
            Session(),
            llm=llm,
            session_key=KEY,
            on_event=lambda k, p: events.append((k, p)),
        )
        answer = agent.ask("看看这张图")

        self.assertIn("按文本信息", answer)
        self.assertEqual(len(llm.seen), 3)
        # 第三次调用（降级重试）中所有消息 content 均为纯字符串
        for msg in llm.seen[2]:
            self.assertIsInstance(msg.get("content", ""), str)
        warnings = [p["message"] for k, p in events if k == "warning"]
        self.assertTrue(any("不支持多模态" in w for w in warnings))

    def test_ui_preindex_in_background_works_when_embedding_ready(self):
        from types import SimpleNamespace
        from app import embeddings, ui_streamlit

        doc = dt.add_document("chart.png", _png_bytes((40, 40)), key=KEY)
        ready_cfg = config.EmbeddingConfig(
            base_url="https://example.test/v1",
            model="embedding-3",
            api_key="sk-test",
        )
        stub_st = SimpleNamespace(session_state=SimpleNamespace(hybrid_search=True))
        with (
            mock.patch.object(ui_streamlit, "st", stub_st),
            mock.patch.object(config, "embedding_current", return_value=ready_cfg),
            mock.patch.object(embeddings.VectorStore, "preindex_async", return_value=None) as mocked,
        ):
            ui_streamlit._preindex_in_background(doc)
            embeddings.preindex_async([doc], cfg=ready_cfg)

        self.assertEqual(mocked.call_count, 2)

    def test_system_prompt_and_read_file_steer_uploaded_image_to_read_document(self):
        dt.add_document("downloaded.jpg", _jpeg_bytes((40, 40)), key=KEY)
        prompt = make_system_prompt(files=dt.document_digest(KEY))
        self.assertIn("read_document", prompt)
        self.assertIn("不要调用 read_file", prompt)

        err = registry.call("read_file", {"path": "downloaded.jpg"})
        self.assertIn("已上传的资料中有 `downloaded.jpg`", err)
        self.assertIn('read_document(name="downloaded.jpg", part=1)', err)


if __name__ == "__main__":
    unittest.main()
