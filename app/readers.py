"""把各种文件读成纯文本。

为什么单独一层：
  1. 抽取逻辑和"给模型看什么"无关，可以脱离工具层单测；
  2. 解析库全部**延迟导入 + 可选**——别人 clone 下来只装了 pandas，
     传 CSV 照样能用，只是传 PDF 时收到一句"装一下 pypdf"，而不是 ImportError 崩掉；
  3. 工具层只关心"读第几块"，不需要知道 pdf 和 docx 有什么区别。

为什么不放进沙箱里读：
  子进程禁 open，也不该为了读文件往里面塞一堆解析库。文档解析放在**主进程**，
  属于工具能力；沙箱只管"执行模型写的计算代码"。两者职责不要混。

一个必须记住的取舍：这里抽的是**文字**，不是版面。
  - PDF 只有文字层里的内容，扫描件抽出来是空的（要 OCR，不在本项目范围）；
  - PPT 只取文本框和表格，图表里的数值抽不到；
  - Word 的文本框、页眉页脚、批注会被漏掉；
  - Excel 只取前几十行、前几列，公式给的是缓存值（没被 Excel 算过就是空的）。
这不是 bug，是"轻量抽取"的固有边界，返回结果里会写清楚，免得模型对着空文本胡说。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import DOC_CHUNK_CHARS, MAX_DOC_CHARS

# 后缀 -> 内部类型。日志和 txt 走同一条路，没必要分开写。
KIND_BY_SUFFIX: dict[str, str] = {
    ".txt": "text",
    ".log": "text",
    ".md": "text",
    ".markdown": "text",
    ".json": "text",
    ".yaml": "text",
    ".yml": "text",
    ".ini": "text",
    ".conf": "text",
    ".sql": "text",
    ".py": "text",
    ".csv": "text",  # 当作资料看的 CSV 走文本（当数据表用的是另一条路）
    ".tsv": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".doc": "doc-legacy",
    ".ppt": "ppt-legacy",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xls-legacy",
}

# 一次读几张工作表、每张读多少行。超过就截断并说明——
# 一个 5 万行的 Excel 全倒进上下文，既贵又没用，模型真正需要的是"有哪些列、大概什么形状"。
MAX_SHEET_ROWS = 40
MAX_SHEETS = 6
MAX_SHEET_COLS = 12

# 会被**当成表**处理的后缀：能解析成 DataFrame 就走 SQL / Python / 图表。
# 其余后缀一律当资料抽成文字。这个判断在程序里做，不该让用户在上传时选。
TABLE_SUFFIXES = {".csv", ".tsv"}

# 网页上传框接受的全部后缀（表 + 资料共用同一个框）
UPLOAD_SUFFIXES = sorted({s.lstrip(".") for s in KIND_BY_SUFFIX})


def route(name: str) -> str:
    """这个文件该走"表"还是"资料"：返回 "table" / "document" / "unsupported"。

    只按后缀判断，**不猜内容**：同一个文件传两次必须得到同一种处理，
    否则用户没法解释"为什么这次它变成了表、上次不是"。
    """
    suffix = Path(name).suffix.lower()
    if suffix in TABLE_SUFFIXES:
        return "table"
    if suffix in KIND_BY_SUFFIX:
        return "document"
    return "unsupported"

_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5")


class DocumentError(Exception):
    """读不了这份文件。消息是**给用户看的**，要写清楚下一步怎么办。"""


class MissingDependency(DocumentError):
    """缺解析库。单独一个类，是因为它和"文件坏了"是完全不同的两件事。"""


@dataclass
class Doc:
    """一份已抽取的资料。chunks 在构造时就切好，读的时候不用重算。"""

    name: str
    kind: str
    text: str
    note: str = ""
    path: str = ""  # 落盘位置，删除时要它；不参与文本内容
    owned: bool = True  # 文件归不归这里：储藏区的文件归储藏区，这里只能撤登记
    chunks: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.chunks = split_chunks(self.text) or [""]

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def parts(self) -> int:
        return len(self.chunks)


def read_table(raw: bytes) -> pd.DataFrame | None:
    """把 CSV 字节读成表。读不出来返回 None，让调用方决定怎么提示。

    按编码依次尝试：中文 Windows 上 Excel 导出的 CSV 多半是 GBK，用默认 utf-8
    会抛 UnicodeDecodeError；latin-1 兜底是因为它永远不会解码失败。
    """
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:  # noqa: BLE001  解析失败通常是分隔符不对，换编码也没用
            return None
        if frame.empty or not len(frame.columns):
            return None
        return frame
    return None


def split_chunks(text: str, size: int = DOC_CHUNK_CHARS) -> list[str]:
    """按 size 切块，尽量落在换行或句号上。

    为什么要挑断点：硬切会把一句话劈成两半，模型拿到"上半年销售额为"和
    "120 万"，很容易把后半句当成另一个意思。挑断点只多几行代码。
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + size, total)
        if end < total:
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind(". "))
            # 断点太靠前会让块碎成几句，还不如硬切
            if cut > size * 0.6:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end
    return chunks


def _looks_binary(raw: bytes) -> bool:
    """看前 4KB 有没有 NUL 或过多控制字符。

    没有这一步，`latin-1` 会把任何二进制都"成功"解码成一堆乱码，
    然后模型对着一屏垃圾认真分析——比直接报错糟糕得多。
    """
    head = raw[:4096]
    if not head:
        return False
    if b"\x00" in head:
        return True
    control = sum(1 for b in head if b < 9 or 13 < b < 32)
    return control / len(head) > 0.05


def decode_text(raw: bytes) -> str:
    """公开的编码探测入口。

    files.py 读本地文件要用同一套规则：两处各写一份，迟早会出现
    "上传能读、本地读不了"这种自相矛盾的行为。
    """
    return _decode(raw)


def _decode(raw: bytes) -> str:
    """依次试常见编码。中文 Windows 的 txt/log 多半是 GB18030。"""
    if _looks_binary(raw):
        return ""
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_text(path: Path) -> tuple[str, str]:
    text = _decode(path.read_bytes())
    if not text:
        raise DocumentError("这个文件看起来是二进制内容，不是文本。")
    return text, ""


def _read_pdf(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 装了就不会走到
        raise MissingDependency("读 PDF 需要 pypdf：`pip install pypdf`") from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # 很多"加密"PDF 只是设了空密码，先试一下
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise DocumentError("这份 PDF 有密码，读不了。请先解密再上传。") from exc

    pages: list[str] = []
    empty = 0
    for index, page in enumerate(reader.pages, 1):
        try:
            body = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001  单页坏了不该让整份文件失败
            body = ""
        if body:
            pages.append(f"【第 {index} 页】\n{body}")
        else:
            empty += 1

    text = "\n\n".join(pages)
    if not text:
        return "", "这份 PDF 没有可抽取的文字层（很可能是扫描件/图片），需要 OCR 才能读。"
    note = f"共 {len(reader.pages)} 页，其中 {empty} 页没有文字层。" if empty else ""
    return text, note


def _read_docx(path: Path) -> tuple[str, str]:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency(
            "读 Word(.docx) 需要 python-docx：`pip install python-docx`"
        ) from exc

    document = DocxDocument(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    # 表格文字不在 paragraphs 里，必须单独取，否则"数据全在表里"的文档读出来是空的
    for index, table in enumerate(document.tables, 1):
        rows = [
            " | ".join(cell.text.strip() for cell in row.cells) for row in table.rows
        ]
        rows = [r for r in rows if r.strip(" |")]
        if rows:
            parts.append(f"【表 {index}】\n" + "\n".join(rows))

    text = "\n\n".join(parts)
    if not text:
        return "", "这份 Word 里没抽到文字（可能内容都在图片或文本框里）。"
    return text, ""


def _read_pptx(path: Path) -> tuple[str, str]:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency(
            "读 PPT(.pptx) 需要 python-pptx：`pip install python-pptx`"
        ) from exc

    prs = Presentation(str(path))
    parts: list[str] = []
    empty = 0
    for index, slide in enumerate(prs.slides, 1):
        lines: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        lines.append(line)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))
        # 演讲备注常是真正的内容，正文只有标题时尤其明显
        notes = ""
        if slide.has_notes_slide:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()

        if lines:
            block = f"【第 {index} 页】\n" + "\n".join(lines)
            if notes:
                block += f"\n（备注）{notes}"
            parts.append(block)
        elif notes:
            parts.append(f"【第 {index} 页】（无正文，仅有备注）\n（备注）{notes}")
        else:
            empty += 1

    text = "\n\n".join(parts)
    if not text:
        return "", "这份 PPT 没抽到文字（可能整页都是图片）。"
    note = f"共 {len(prs.slides)} 页，其中 {empty} 页没有文字。" if empty else ""
    return text, note


def _read_xlsx(path: Path) -> tuple[str, str]:
    """Excel 抽成「每张表：列名 + 前几十行」。

    用 openpyxl 的 read_only + data_only：read_only 才不会把整本表读进内存
    （几万行的表能把内存吃满），data_only 让公式单元格给出缓存值而不是
    "=SUM(B2:B9)" 这串字符串——模型拿到公式文本没法用。

    代价是公式没被 Excel 算过的文件里取值是空的，这一点写进 note，
    免得模型把空白当成"这张表没数据"。
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - 装了就不会走到
        raise MissingDependency("读 Excel(.xlsx) 需要 openpyxl：`pip install openpyxl`") from exc

    try:
        book = load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001  损坏、加密、根本不是 xlsx 都会到这里
        raise DocumentError(f"这个 Excel 打不开（{exc}）。如果设置了密码，请先解密再读。") from exc

    blocks: list[str] = []
    truncated: list[str] = []
    try:
        for sheet in list(book.worksheets)[:MAX_SHEETS]:
            rows: list[str] = []
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                if index >= MAX_SHEET_ROWS:
                    break
                cells = ["" if value is None else str(value).strip() for value in row[:MAX_SHEET_COLS]]
                if any(cells):
                    rows.append(" | ".join(cells))
            if not rows:
                continue
            size = f"约 {sheet.max_row or 0} 行 × {sheet.max_column or 0} 列"
            blocks.append(f"【{sheet.title}】{size}\n" + "\n".join(rows))
            if (sheet.max_row or 0) > MAX_SHEET_ROWS:
                truncated.append(sheet.title)
    finally:
        book.close()

    if not blocks:
        return "", "这张 Excel 里没抽到任何单元格内容。"

    notes = []
    if truncated:
        notes.append(
            f"工作表 {', '.join(truncated)} 只列了前 {MAX_SHEET_ROWS} 行"
        )
    notes.append("要按 SQL 查询整张表，请另存为 CSV 再上传")
    return "\n\n".join(blocks), "；".join(notes)


def _read_xls(path: Path, raw: bytes) -> tuple[str, str]:
    """.xls 是 Excel 97-2003 的二进制格式，openpyxl 读不了。"""
    if not _looks_binary(raw):
        return _decode(raw), "这个文件其实是纯文本（只是后缀写成了 .xls），已按文本读取。"
    raise DocumentError(
        "Excel 97-2003 的旧版 .xls 读不了。请在 Excel / WPS 里另存为 .xlsx 再读。"
    )


def _read_legacy(path: Path, raw: bytes) -> tuple[str, str]:
    """.doc / .ppt：老二进制格式，明确拒绝，但两条常见的"伪装的"能救回来。"""
    if raw[:5] == b"{\\rtf":
        raise DocumentError(
            "这是 RTF 文件（后缀写成了 .doc）。请在 Word 里打开后另存为 .docx 再上传。"
        )
    if not _looks_binary(raw):
        return _decode(raw), "这个文件其实是纯文本（只是后缀写成了 .doc），已按文本读取。"
    raise DocumentError(
        "Word 97-2003 的旧版 .doc/.ppt 是二进制格式，没有可靠的纯 Python 解析方式。"
        "请在 Word / WPS 里另存为 .docx（或 .pptx）再上传。"
    )


def _clip(text: str) -> tuple[str, str]:
    """超长文档截断。不截的话一次读入几十万字，token 预算当场烧完。"""
    if len(text) <= MAX_DOC_CHARS:
        return text, ""
    head = text[:MAX_DOC_CHARS]
    return head, f"原文约 {len(text):,} 字，已截断为前 {MAX_DOC_CHARS:,} 字。"


def extract(path: Path, name: str | None = None) -> Doc:
    """按后缀分发。读不了就抛 DocumentError，消息直接可以显示给用户。"""
    path = Path(path)
    name = name or path.name
    suffix = path.suffix.lower()
    kind = KIND_BY_SUFFIX.get(suffix)
    if kind is None:
        supported = "、".join(UPLOAD_SUFFIXES)
        raise DocumentError(f"不支持 {suffix or '无后缀'} 文件。目前支持：{supported}")

    raw = path.read_bytes()
    if not raw:
        raise DocumentError("文件是空的（0 字节）。")

    if kind == "text":
        text, note = _read_text(path)
    elif kind == "pdf":
        text, note = _read_pdf(path)
    elif kind == "docx":
        text, note = _read_docx(path)
    elif kind == "pptx":
        text, note = _read_pptx(path)
    elif kind == "xlsx":
        text, note = _read_xlsx(path)
    elif kind == "xls-legacy":
        text, note = _read_xls(path, raw)
    else:
        text, note = _read_legacy(path, raw)

    text, clipped = _clip(_normalize(text))
    notes = "；".join(n for n in (note, clipped) if n)
    return Doc(
        name=name,
        kind=suffix.lstrip(".") or kind,
        text=text,
        note=notes,
        path=str(path),
    )


def _normalize(text: str) -> str:
    """统一换行、压掉多余空行。PDF 抽出来的文本常见三四个连续空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


__all__ = [
    "Doc",
    "DocumentError",
    "MissingDependency",
    "KIND_BY_SUFFIX",
    "TABLE_SUFFIXES",
    "UPLOAD_SUFFIXES",
    "extract",
    "read_table",
    "split_chunks",
]
