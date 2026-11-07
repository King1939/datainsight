"""生成 PowerPoint。

七种版式，几何和主题都是**参数**而不是写死的：同一份内容，换 theme 换配色，
换 size 换画布，换 margin 换留白。版式本身不开放自由定位——这是有意的，
理由写在 README 的「设计取舍」里：坐标交给模型，20 页里迟早有一页文字压着边框。

排版里唯一需要动脑的地方是**写之前的估字**。python-pptx 不给你渲染结果，
导出后才发现溢出就晚了，所以每个文本块都要按可用宽高算出「这个字号放不放得下」，
放不下就降一档，最低 11pt 仍放不下才截断，并在返回文本里说清哪一页被截了。
"""

from __future__ import annotations

import math
import re
import shutil
import time
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Any, Literal, TypedDict

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from .. import files
from ..config import OUTPUT_DIR
from . import registry

# 上限不是审美，是防呆：一页塞 15 条要点，字号降到 8pt 也没人看得清，
# 不如直接报错让人拆页。表格同理。
MAX_SLIDES = 30
MAX_ITEMS = 8
MAX_ROWS = 12
MAX_COLS = 6
MAX_TITLE_CHARS = 40

LAYOUTS = ("cover", "section", "bullets", "kpi", "table", "compare", "closing")

# 这三种是全屏深色页，不加页脚。目录/章节页上挂个页码会显得很不专业。
NO_FOOTER = {"cover", "section", "closing"}


class _SlideBase(TypedDict):
    layout: Literal["cover", "section", "bullets", "kpi", "table", "compare", "closing"]


class Slide(_SlideBase, total=False):
    """一页的内容。字段按 layout 取用，用不到的直接不传。"""

    title: str
    subtitle: str
    number: str
    lead: str
    items: list[str]
    note: str
    headers: list[str]
    rows: list[list[str]]
    left_title: str
    left_items: list[str]
    right_title: str
    right_items: list[str]


@dataclass(frozen=True)
class Theme:
    label: str
    primary: str  # 主色：强调条、KPI 数字、章节页编号
    deep: str  # 深色底：封面之外的章节页 / 结束页
    ink: str  # 正文
    muted: str  # 次要文字、页脚
    soft: str  # 浅底：表头、KPI 卡片
    rule: str  # 分隔线
    ghost: str  # 深色底上的大号编号，用中间调而不是主色
    up: str  # 涨（A 股习惯：红）
    down: str  # 跌（绿）
    head_font: str
    body_font: str


@dataclass(frozen=True)
class Geo:
    width: float
    height: float
    margin: float

    @property
    def content_w(self) -> float:
        return self.width - 2 * self.margin

    @property
    def body_bottom(self) -> float:
        """正文能画到的最低处。下面 0.72 英寸留给页脚，再往下就压页码了。"""
        return self.height - 0.72


THEMES: dict[str, Theme] = {
    "business": Theme(
        label="商务蓝",
        primary="185FA5",
        deep="0C447C",
        ink="2C2C2A",
        muted="5F5E5A",
        soft="E6F1FB",
        rule="D3D1C7",
        ghost="85B7EB",
        up="A32D2D",
        down="0F6E56",
        head_font="微软雅黑",
        body_font="微软雅黑",
    ),
    "academic": Theme(
        label="学术灰",
        primary="0F6E56",
        deep="2C2C2A",
        ink="2C2C2A",
        muted="5F5E5A",
        soft="E1F5EE",
        rule="D3D1C7",
        ghost="9FE1CB",
        up="A32D2D",
        down="0F6E56",
        head_font="宋体",
        body_font="微软雅黑",
    ),
}

SIZES: dict[str, tuple[float, float]] = {"16:9": (13.333, 7.5), "4:3": (10.0, 7.5)}
MARGINS: dict[str, float] = {"narrow": 0.5, "normal": 0.625, "wide": 0.9}


# ---------------------------------------------------------------- 基础排版


def _rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def _east_asian(run, name: str) -> None:
    """把中文字体同时写到 <a:latin> 和 <a:ea>。

    只设 font.name 时 python-pptx 只写 latin，中文在 PowerPoint 里会退回主题字体
    （一般是宋体），西文却是雅黑——同一行字两种字体，看得出是机器排的。
    """
    rpr = run._r.get_or_add_rPr()
    ea = rpr.find(qn("a:ea"))
    if ea is None:
        ea = rpr.makeelement(qn("a:ea"), {})
        latin = rpr.find(qn("a:latin"))
        if latin is not None:
            latin.addnext(ea)
        else:
            rpr.append(ea)
    ea.set("typeface", name)


def _font(
    run,
    *,
    name: str,
    size: float,
    color: str,
    bold: bool = False,
) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = _rgb(color)
    _east_asian(run, name)


def _frame(shape, *, anchor: MSO_ANCHOR = MSO_ANCHOR.TOP, pad: float = 0.0):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(pad)
    tf.margin_right = Inches(pad)
    tf.margin_top = Inches(0)
    tf.margin_bottom = Inches(0)
    return tf


def _text(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    _frame(box, anchor=anchor)
    return box.text_frame


def _rect(slide, left: float, top: float, width: float, height: float, *, fill: str, radius: bool = False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(width), Inches(height))
    if radius:
        shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(fill)
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def _rule(slide, left: float, top: float, width: float, color: str, pt: float = 0.75) -> None:
    bar = _rect(slide, left, top, width, pt / 72, fill=color)
    bar.name = "rule"


# ---------------------------------------------------------------- 估字


def _units(text: str) -> float:
    """按"一个中文字宽 = 1，一个西文字宽 = 0.55"折算成排版单位。"""
    return sum(1.0 if ord(ch) > 127 else 0.55 for ch in text or "")


def _lines(text: str, width_in: float, size: float) -> int:
    """这段文字在给定宽度下要占几行。

    _units 里中文记 1、西文记 0.55，都是「几个字号宽」，所以每行的容量就是
    宽度（pt）除以字号——不用再乘一个 0.5。乘过之后中文会少算一半行数，
    判断"放得下"，实际溢出。
    """
    per_line = max(1.0, width_in * 72 / size)
    return max(1, math.ceil(_units(text) / per_line))


# PowerPoint 的「单倍行高」不是字号的 1 倍，而是约 1.2 倍（ ascent + descent + 行间）。
# 漏掉这个系数，每页都会算出"放得下"，实际溢出——4:3 长文本对比页就是这么翻车的。
LINE_FACTOR = 1.2


def _block_height(texts: list[str], width_in: float, size: float, *, spacing: float = 1.45, gap: float = 7.0) -> float:
    """整块文字占多高（英寸），含段间距。"""
    lines = sum(_lines(t, width_in, size) for t in texts)
    return (lines * size * spacing * LINE_FACTOR + gap * max(len(texts) - 1, 0)) / 72


def _pick_size(texts: list[str], width_in: float, height_in: float, sizes: tuple[float, ...]) -> float:
    """从上往下试，第一个放得下的字号就是它。都放不下就用最小的那档。"""
    for size in sizes:
        if _block_height(texts, width_in, size) <= height_in:
            return size
    return sizes[-1]


def _spread_gap(
    texts: list[str],
    width_in: float,
    height_in: float,
    size: float,
    *,
    base: float = 7.0,
    cap: float = 24.0,
    spacing: float = 1.45,
) -> float:
    """把富余的竖直空间摊进段落间距，返回段后距（pt）。

    五条短要点用 18pt 排出来只占半页时，全挤在顶部会像没写完。把剩下的高度
    均摊到行间，整块内容就撑满了正文区。封顶是为了不出现「双倍行距」那种
    松散感——间距超过 cap 就宁可留着底部白，也不硬拉。乘 0.85 是给
    PowerPoint 实际行高与估算值之间的误差留余量，不然刚好填满会压到最后一行。
    """
    if len(texts) < 2:
        return base
    slack = height_in - _block_height(texts, width_in, size, spacing=spacing, gap=base)
    if slack <= 0:
        return base
    return min(cap, base + slack * 0.85 * 72 / (len(texts) - 1))


def _clip(text: str, width_in: float, size: float, max_lines: int = 3) -> tuple[str, bool]:
    """到最小字号还放不下时的兜底：按「最多几行」硬截，末尾加省略号。"""
    per_line = max(1.0, width_in * 72 / size)
    budget = int(per_line * max_lines)
    if _units(text) <= budget:
        return text, False
    out: list[str] = []
    used = 0.0
    for ch in text:
        cost = 1.0 if ord(ch) > 127 else 0.55
        if used + cost > budget - 1:
            break
        out.append(ch)
        used += cost
    return "".join(out).rstrip() + "…", True


# ---------------------------------------------------------------- 七种版式


def _title_bar(slide, page: dict, geo: Geo, theme: Theme, *, font_pt: float = 28) -> float:
    """统一页标题 + 分隔线，返回正文可用的起始 y。"""
    title = str(page.get("title") or "").strip()
    tf = _text(slide, geo.margin, 0.5, geo.content_w, 0.8)
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    _font(run, name=theme.head_font, size=font_pt, color=theme.ink, bold=True)
    _rule(slide, geo.margin, 1.34, geo.content_w, theme.rule)
    return 1.62


def _footer(slide, page_no: int, geo: Geo, theme: Theme, deck_title: str) -> None:
    y = geo.height - 0.52
    if deck_title:
        tf = _text(slide, geo.margin, y, geo.content_w * 0.7, 0.28)
        run = tf.paragraphs[0].add_run()
        run.text = deck_title
        _font(run, name=theme.body_font, size=10, color=theme.muted)

    tf = _text(slide, geo.width - geo.margin - 0.55, y, 0.55, 0.28)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = f"{page_no:02d}"
    _font(run, name=theme.body_font, size=10, color=theme.muted)


def _cover(slide, page: dict, geo: Geo, theme: Theme, deck_title: str) -> None:
    _rect(slide, 0, 0, geo.width, 0.16, fill=theme.primary)

    title_top = geo.height * 0.36
    # 竖条要和标题块等高：比标题长一截会像两件没对齐的东西
    bar = _rect(slide, geo.margin, title_top + 0.05, 0.07, 1.0, fill=theme.primary)

    title = str(page.get("title") or "").strip()
    size = 40 if _units(title) <= 24 else 32
    tf = _text(slide, geo.margin + 0.3, title_top, geo.content_w - 0.3, 0.9)
    run = tf.paragraphs[0].add_run()
    run.text = title
    _font(run, name=theme.head_font, size=size, color=theme.ink, bold=True)

    subtitle = str(page.get("subtitle") or "").strip()
    if subtitle:
        # 框给 0.9：副标题常常带一行括号说明，按 0.5 给会在两行时溢出框外
        tf = _text(slide, geo.margin + 0.3, title_top + 1.28, geo.content_w - 0.3, 0.9)
        run = tf.paragraphs[0].add_run()
        run.text = subtitle
        _font(run, name=theme.body_font, size=16, color=theme.muted)

    bar.name = "accent"


def _section(slide, page: dict, geo: Geo, theme: Theme, deck_title: str) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _rgb(theme.deep)

    # 编号贴着标题放，别隔着一整块空白——那样像两个不相干的元素
    number = str(page.get("number") or "").strip()
    if number:
        tf = _text(slide, geo.margin, geo.height * 0.5 - 1.15, geo.content_w, 1.1)
        run = tf.paragraphs[0].add_run()
        run.text = number
        _font(run, name=theme.head_font, size=56, color=theme.ghost, bold=True)

    title = str(page.get("title") or "").strip()
    tf = _text(slide, geo.margin, geo.height * 0.5, geo.content_w, 0.7)
    run = tf.paragraphs[0].add_run()
    run.text = title
    _font(run, name=theme.head_font, size=32, color="FFFFFF", bold=True)

    subtitle = str(page.get("subtitle") or "").strip()
    if subtitle:
        tf = _text(slide, geo.margin, geo.height * 0.5 + 0.78, geo.content_w, 0.6)
        run = tf.paragraphs[0].add_run()
        run.text = subtitle
        _font(run, name=theme.body_font, size=15, color="D3D1C7")


def _bullet_lines(items: list[str]) -> list[tuple[str, int]]:
    """把「两个空格开头」识别成第二级，其余是第一级。"""
    rows: list[tuple[str, int]] = []
    for raw in items:
        text = str(raw)
        if text.startswith("  ") or text.startswith("　　"):
            rows.append((text.strip(), 2))
        else:
            rows.append((text.strip(), 1))
    return rows


def _bullets(slide, page: dict, geo: Geo, theme: Theme, deck_title: str, notes: list[str], page_no: int) -> None:
    top = _title_bar(slide, page, geo, theme)

    lead = str(page.get("lead") or "").strip()
    if lead:
        tf = _text(slide, geo.margin, top, geo.content_w, 0.8)
        run = tf.paragraphs[0].add_run()
        run.text = lead
        _font(run, name=theme.head_font, size=19, color=theme.primary, bold=True)
        top += 0.85

    items = [str(i) for i in (page.get("items") or [])]
    rows = _bullet_lines(items)
    texts = [f"•  {t}" if level == 1 else f"–  {t}" for t, level in rows]

    height = geo.body_bottom - top
    size = _pick_size(texts, geo.content_w - 0.35, height, (18, 17, 16, 15, 14, 13, 12, 11))
    if _block_height(texts, geo.content_w - 0.35, size) > height:
        clipped = []
        for text in texts:
            kept, cut = _clip(text, geo.content_w - 0.35, size)
            clipped.append(kept)
            if cut:
                notes.append(f"第 {page_no} 页有要点过长，已截断到 {size:.0f}pt 三行以内")
        texts = clipped

    gap = _spread_gap(texts, geo.content_w - 0.35, height, size)

    tf = _text(slide, geo.margin, top, geo.content_w, height)
    for index, (text, (_, level)) in enumerate(zip(texts, rows)):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.line_spacing = 1.45
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        indent = 0.30 if level == 1 else 0.62
        pPr.set("marL", str(int(Inches(indent))))
        pPr.set("indent", str(-int(Inches(0.30))))
        run = p.add_run()
        run.text = text
        _font(
            run,
            name=theme.body_font,
            size=size if level == 1 else size - 1,
            color=theme.ink if level == 1 else theme.muted,
        )


def _parse_kpi(item: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in str(item).split("|")]
    value = parts[0] if parts else ""
    label = parts[1] if len(parts) > 1 else ""
    delta = parts[2] if len(parts) > 2 else ""
    return value, label, delta


def _delta_color(delta: str, theme: Theme) -> str:
    if delta.startswith("+"):
        return theme.up
    if delta.startswith(("-", "−", "–")):
        return theme.down
    return theme.muted


def _kpi(slide, page: dict, geo: Geo, theme: Theme, deck_title: str, notes: list[str], page_no: int) -> None:
    top = _title_bar(slide, page, geo, theme)

    items = [_parse_kpi(i) for i in (page.get("items") or [])]
    count = max(1, len(items))
    gap = 0.28
    box_w = (geo.content_w - gap * (count - 1)) / count
    box_h = 2.0

    note_text = str(page.get("note") or "").strip()
    # 垂直居中：只有两三块指标时贴着标题排，下面会空掉大半页
    slack = geo.body_bottom - top - box_h - (0.45 if note_text else 0)
    card_top = top + max(0.25, slack * 0.45)

    for index, (value, label, delta) in enumerate(items):
        left = geo.margin + index * (box_w + gap)
        _rect(slide, left, card_top, box_w, box_h, fill=theme.soft, radius=True)

        value_size = 40 if _units(value) <= 5 else 30
        tf = _text(slide, left + 0.28, card_top + 0.5, box_w - 0.5, 0.78)
        run = tf.paragraphs[0].add_run()
        run.text = value
        _font(run, name=theme.head_font, size=value_size, color=theme.primary, bold=True)

        if label:
            tf = _text(slide, left + 0.28, card_top + 1.32, box_w - 0.5, 0.34)
            run = tf.paragraphs[0].add_run()
            run.text = label
            _font(run, name=theme.body_font, size=12, color=theme.muted)

        if delta:
            tf = _text(slide, left + 0.28, card_top + 1.68, box_w - 0.5, 0.34)
            run = tf.paragraphs[0].add_run()
            run.text = delta
            _font(run, name=theme.body_font, size=12, color=_delta_color(delta, theme), bold=True)

    if note_text:
        tf = _text(slide, geo.margin, card_top + box_h + 0.28, geo.content_w, 0.5)
        run = tf.paragraphs[0].add_run()
        run.text = note_text
        _font(run, name=theme.body_font, size=12, color=theme.muted)


def _looks_numeric(text: str) -> bool:
    cleaned = str(text).strip().replace(",", "").replace("%", "").replace("¥", "").replace("$", "")
    if not cleaned:
        return False
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def _strip_table_style(table) -> None:
    """把表格的内置样式换成「无样式、无网格」。

    不换的话，默认的 Medium Style 2 会给每个单元格画一圈白边——表头那行浅底上
    就是一道道白色竖缝，看起来像表格被切碎了。边框我们自己用底线画。
    """
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(old)
    style = tblPr.makeelement(qn("a:tableStyleId"), {})
    style.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    tblPr.append(style)


def _cell_border(cell, color: str, pt: float = 0.75) -> None:
    """只画底线。python-pptx 没有边框 API，得手写 OOXML。"""
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("a:lnB")):
        tcPr.remove(old)
    ln = tcPr.makeelement(qn("a:lnB"), {"w": str(int(pt * 12700)), "cap": "flat", "cmpd": "sng", "algn": "ctr"})
    fill = ln.makeelement(qn("a:solidFill"), {})
    fill.append(ln.makeelement(qn("a:srgbClr"), {"val": color}))
    ln.append(fill)
    tcPr.insert(0, ln)


def _cell_text(cell, text: str, *, theme: Theme, size: float, color: str, bold: bool = False, align=PP_ALIGN.LEFT) -> None:
    cell.margin_left = Inches(0.09)
    cell.margin_right = Inches(0.09)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = str(text)
    _font(run, name=theme.body_font, size=size, color=color, bold=bold)


def _table(slide, page: dict, geo: Geo, theme: Theme, deck_title: str, notes: list[str], page_no: int) -> None:
    top = _title_bar(slide, page, geo, theme)

    headers = [str(h) for h in (page.get("headers") or [])]
    rows = [[str(c) for c in row] for row in (page.get("rows") or [])]
    cols = len(headers)

    if len(rows) > MAX_ROWS:
        notes.append(f"第 {page_no} 页表格只放了前 {MAX_ROWS} 行（共 {len(rows)} 行）")
        rows = rows[:MAX_ROWS]

    # 列宽按内容长度分配，但夹在 0.8~2.2 之间：否则一列长文本会把其它列压成一条缝
    weights = []
    for idx in range(cols):
        longest = max([_units(headers[idx])] + [_units(r[idx]) for r in rows if idx < len(r)])
        weights.append(min(2.2, max(0.8, longest / 6)))
    total = sum(weights) or 1
    widths = [geo.content_w * w / total for w in weights]

    # 行高按页面剩余空间撑开：4 行的表原来只占 1.9 英寸，下面空一大片。
    # 上限 0.62 是经验值，再高一行里的字就浮在中间显得松散；
    # 下限 0.34 是 13pt 文字加内边距的物理极限。低于它就减行，别硬塞。
    header_h = 0.45
    note_text = str(page.get("note") or "").strip()
    avail = geo.body_bottom - (top + 0.1) - (0.34 if note_text else 0)
    fit = max(1, int((avail - header_h) // 0.34))
    if len(rows) > fit:
        notes.append(f"第 {page_no} 页表格按页面高度只放了 {fit} 行（共 {len(rows)} 行）")
        rows = rows[:fit]
    body_h = min(0.62, max(0.34, (avail - header_h) / max(len(rows), 1)))
    height = header_h + body_h * max(len(rows), 1)

    shape = slide.shapes.add_table(len(rows) + 1, cols, Inches(geo.margin), Inches(top + 0.1), Inches(geo.content_w), Inches(height))
    table = shape.table
    table.first_row = True
    table.horz_banding = False
    _strip_table_style(table)

    for idx, width in enumerate(widths):
        table.columns[idx].width = Inches(width)
    table.rows[0].height = Inches(header_h)
    for idx in range(1, len(rows) + 1):
        table.rows[idx].height = Inches(body_h)

    numeric = [
        all(_looks_numeric(r[idx]) for r in rows) if rows and idx < len(rows[0]) else False
        for idx in range(cols)
    ]

    for idx, header in enumerate(headers):
        cell = table.cell(0, idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = _rgb(theme.soft)
        _cell_text(
            cell,
            header,
            theme=theme,
            size=13,
            color=theme.ink,
            bold=True,
            align=PP_ALIGN.RIGHT if numeric[idx] else PP_ALIGN.LEFT,
        )

    for row_idx, row in enumerate(rows, start=1):
        for col_idx in range(cols):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _rgb("FFFFFF")
            value = row[col_idx] if col_idx < len(row) else ""
            _cell_text(
                cell,
                value,
                theme=theme,
                size=13,
                color=theme.ink,
                align=PP_ALIGN.RIGHT if numeric[col_idx] else PP_ALIGN.LEFT,
            )
            _cell_border(cell, theme.rule)

    if note_text:
        tf = _text(slide, geo.margin, top + 0.1 + height + 0.12, geo.content_w, 0.3)
        run = tf.paragraphs[0].add_run()
        run.text = note_text
        _font(run, name=theme.body_font, size=11, color=theme.muted)


def _compare(slide, page: dict, geo: Geo, theme: Theme, deck_title: str, notes: list[str], page_no: int) -> None:
    top = _title_bar(slide, page, geo, theme)

    gap = 0.42
    col_w = (geo.content_w - gap) / 2
    text_w = col_w - 0.3
    columns = [
        [geo.margin, str(page.get("left_title") or ""), [f"•  {i}" for i in (page.get("left_items") or [])]],
        [geo.margin + col_w + gap, str(page.get("right_title") or ""), [f"•  {i}" for i in (page.get("right_items") or [])]],
    ]

    head_h = 0.52
    body_top = top + head_h + 0.28
    body_h = geo.body_bottom - body_top

    # 两栏字号取同一个小值：各降各的会出现左栏 16pt、右栏 13pt，一页两种正文大小
    size = min(_pick_size(c[2], text_w, body_h, (16, 15, 14, 13, 12, 11)) for c in columns)

    for column in columns:
        if _block_height(column[2], text_w, size) <= body_h:
            continue
        fixed = []
        for text in column[2]:
            kept, cut = _clip(text, text_w, size, max_lines=3)
            fixed.append(kept)
            if cut:
                notes.append(f"第 {page_no} 页对比栏有要点过长，已截断到 {size:.0f}pt 三行以内")
        column[2] = fixed

    # 内容不满一页时把富余高度摊进条目间距，别全挤在顶部；有上限，不然会松成
    # 双倍行距。按较高的那栏算——两栏共用同一个间距，短的那栏自然收在更上面。
    # 标题条不跟着动——跟着动会让标题和分隔线之间空出一块，更像漏了东西。
    taller = max(columns, key=lambda c: _block_height(c[2], text_w, size))
    gap_pt = _spread_gap(taller[2], text_w, body_h, size, base=6.0, cap=26.0, spacing=1.4)

    for left, header, texts in columns:
        _rect(slide, left, top, col_w, head_h, fill=theme.soft, radius=True)
        tf = _text(slide, left, top, col_w, head_h, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = header
        _font(run, name=theme.head_font, size=15, color=theme.primary, bold=True)

        tf = _text(slide, left + 0.1, body_top, col_w - 0.2, body_h)
        for idx, text in enumerate(texts):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.line_spacing = 1.4
            p.space_after = Pt(gap_pt)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.28))))
            pPr.set("indent", str(-int(Inches(0.28))))
            run = p.add_run()
            run.text = text
            _font(run, name=theme.body_font, size=size, color=theme.ink)


def _closing(slide, page: dict, geo: Geo, theme: Theme, deck_title: str) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _rgb(theme.deep)

    title = str(page.get("title") or "").strip()
    tf = _text(slide, geo.margin, geo.height * 0.4, geo.content_w, 1.0, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = title
    _font(run, name=theme.head_font, size=40, color="FFFFFF", bold=True)

    subtitle = str(page.get("subtitle") or "").strip()
    if subtitle:
        tf = _text(slide, geo.margin, geo.height * 0.4 + 1.0, geo.content_w, 0.5)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = subtitle
        _font(run, name=theme.body_font, size=15, color="D3D1C7")


PAINTERS = {
    "cover": _cover,
    "section": _section,
    "bullets": _bullets,
    "kpi": _kpi,
    "table": _table,
    "compare": _compare,
    "closing": _closing,
}

# 除了标题，每种版式最少还要哪些字段
NEEDS = {
    "cover": (),
    "section": (),
    "bullets": ("items",),
    "kpi": ("items",),
    "table": ("headers", "rows"),
    "compare": ("left_title", "left_items", "right_title", "right_items"),
    "closing": (),
}


# ---------------------------------------------------------------- 校验


def _validate(slides: list[Any], theme: str, size: str, margin: str) -> list[str]:
    errors: list[str] = []
    if theme not in THEMES:
        errors.append(f"theme 只能是 {'/'.join(THEMES)}")
    if size not in SIZES:
        errors.append(f"size 只能是 {'/'.join(SIZES)}")
    if margin not in MARGINS:
        errors.append(f"margin 只能是 {'/'.join(MARGINS)}")

    if not slides:
        errors.append("slides 不能为空，至少要有一页")
        return errors
    if len(slides) > MAX_SLIDES:
        errors.append(f"最多 {MAX_SLIDES} 页，现在是 {len(slides)} 页")

    for index, page in enumerate(slides, start=1):
        if not isinstance(page, dict):
            errors.append(f"第 {index} 页必须是对象")
            continue
        layout = page.get("layout")
        if layout not in LAYOUTS:
            errors.append(f"第 {index} 页 layout 不合法：{layout!r}，可选 {'/'.join(LAYOUTS)}")
            continue

        unknown = set(page) - set(Slide.__annotations__)
        if unknown:
            errors.append(f"第 {index} 页有不认识的字段 {sorted(unknown)}，可用字段：{sorted(Slide.__annotations__)}")

        # 类型也要挡：items 传成字符串时会被逐字拆开，页面上是 20 个单字，
        # 而错误信息只说"缺少 items"的话，模型根本看不出问题在哪。
        for field in ("items", "headers", "rows", "left_items", "right_items"):
            value = page.get(field)
            if value is not None and not isinstance(value, list):
                errors.append(f"第 {index} 页 {field} 必须是列表，现在是 {type(value).__name__}")
        for field in ("title", "subtitle", "number", "lead", "left_title", "right_title"):
            value = page.get(field)
            if value is not None and not isinstance(value, str):
                errors.append(f"第 {index} 页 {field} 必须是字符串，现在是 {type(value).__name__}")

        title = str(page.get("title") or "").strip()
        if not title:
            errors.append(f"第 {index} 页（{layout}）缺少 title")
        elif len(title) > MAX_TITLE_CHARS:
            errors.append(f"第 {index} 页 title 超过 {MAX_TITLE_CHARS} 字，请缩短")

        for field in NEEDS[layout]:
            value = page.get(field)
            if not value:
                errors.append(f"第 {index} 页（{layout}）缺少 {field}")

        items = page.get("items") or []
        if items and len(items) > MAX_ITEMS:
            errors.append(f"第 {index} 页有 {len(items)} 条要点，超过 {MAX_ITEMS} 条，请拆页")

        if layout == "kpi":
            count = len(items)
            if not 1 <= count <= 4:
                errors.append(f"第 {index} 页 kpi 需要 1~4 个指标，现在是 {count} 个")

        headers = page.get("headers") or []
        if headers and not 2 <= len(headers) <= MAX_COLS:
            errors.append(f"第 {index} 页表格需要 2~{MAX_COLS} 列，现在是 {len(headers)} 列")
        for row in page.get("rows") or []:
            if len(row) != len(headers):
                errors.append(f"第 {index} 页有 {len(row)} 个单元格，和 {len(headers)} 个表头对不上")
                break

    return errors


# ---------------------------------------------------------------- 读已有文件

# 1 英寸 = 914400 EMU。python-pptx 的坐标全是 EMU，估字和几何检查要换成英寸。
EMU = 914400


@dataclass(frozen=True)
class DeckInfo:
    """从既有文件里反推出来的排版参数。

    增量修改必须沿用原文件的样子，否则"插进去的那一页"一眼就看得出是外人做的。
    不要求模型再报一遍 theme/size/margin：它会记错，而且在新的会话里根本没有记忆。
    """

    theme: Theme
    geo: Geo
    title: str  # 页脚左侧那份文稿标题
    theme_key: str  # 命中内置主题时是 business / academic，否则为空
    size_key: str
    margin_key: str
    matched: bool  # 配色是否命中内置主题（没命中说明是外部文件或改过色）

    @property
    def label(self) -> str:
        size = self.size_key or f"{self.geo.width:.2f}×{self.geo.height:.2f}"
        return f"{size} · {self.theme.label} · 边距 {self.geo.margin:.3g} 英寸"


def _emu(value: Any) -> float:
    return (value or 0) / EMU


def _solid_fill(shape) -> str:
    """形状的纯色填充，取不到返回空串（主题色、渐变、图片填充都会失败）。"""
    try:
        if shape.fill.type is None:
            return ""
        return str(shape.fill.fore_color.rgb)
    except Exception:  # noqa: BLE001  主题色没有 .rgb，取不到就是取不到
        return ""


def _luminance(color: str) -> float:
    try:
        red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    except (ValueError, IndexError):
        return 1.0
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _mix(color: str, target: str, ratio: float) -> str:
    """把 color 朝 target 混 ratio 比例。派生"深一档/浅一档"用，够用且不引入依赖。"""
    try:
        a = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
        b = [int(target[i : i + 2], 16) for i in (0, 2, 4)]
    except (ValueError, IndexError):
        return color
    return "".join(f"{round(x + (y - x) * ratio):02X}" for x, y in zip(a, b))


def _derive_theme(primary: str) -> Theme:
    """主色认不出来时，用它的深浅派一套中性配色，至少保证插进去的页和封面同色系。"""
    base = THEMES["business"]
    return dataclass_replace(
        base,
        label="自定义",
        primary=primary,
        deep=_mix(primary, "000000", 0.5),
        soft=_mix(primary, "FFFFFF", 0.88),
        ghost=_mix(primary, "FFFFFF", 0.55),
        rule=_mix(primary, "FFFFFF", 0.8),
    )


def _prst(shape) -> str:
    """自选图形的预设形状名（roundRect 之类）。拿不到返回空串。"""
    try:
        sp_pr = shape._element.spPr
        geom = sp_pr.find(qn("a:prstGeom"))
        return geom.get("prst") if geom is not None else ""
    except AttributeError:
        return ""


def _probe_deck(prs) -> DeckInfo:
    """从既有幻灯片反推配色 / 画布 / 边距 / 页脚标题。

    信号源全部是**我们自己写在形状上的东西**：封面那条名为 accent 的竖条是主色，
    正文页那条 rule 是分隔线、它的 left 就是页边距。比读 theme1.xml 靠谱得多——
    我们生成时是直接给形状上色，主题部件里根本没有这些值。
    """
    width, height = _emu(prs.slide_width), _emu(prs.slide_height)
    primary = ""
    rule = ""
    margin: float | None = None
    fonts: list[str] = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.name == "accent" and not primary:
                primary = _solid_fill(shape)
            elif shape.name == "rule":
                if not rule:
                    rule = _solid_fill(shape)
                if margin is None:
                    margin = _emu(shape.left)
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        if run.font.name and run.font.name not in fonts:
                            fonts.append(run.font.name)

    theme_key = ""
    palette: Theme
    if primary:
        for key, candidate in THEMES.items():
            if candidate.primary.upper() == primary.upper():
                theme_key, palette = key, candidate
                break
        else:
            palette = _derive_theme(primary)
    else:
        theme_key, palette = "business", THEMES["business"]

    if fonts:
        palette = dataclass_replace(
            palette,
            head_font=fonts[0],
            body_font=fonts[0] if len(fonts) == 1 else palette.body_font,
        )

    size_key = next(
        (k for k, (w, h) in SIZES.items() if abs(w - width) < 0.02 and abs(h - height) < 0.02),
        "",
    )
    margin = margin if margin is not None else MARGINS["normal"]
    margin_key = next((k for k, v in MARGINS.items() if abs(v - margin) < 0.02), "")

    # 页脚左侧那份标题：多页重复出现的那句最可信，单页稿就用空的
    counts: dict[str, int] = {}
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame or _emu(shape.top) <= height - 0.7:
                continue
            text = shape.text_frame.text.strip()
            if text and not re.fullmatch(r"\d{1,3}", text):
                counts[text] = counts.get(text, 0) + 1
    deck_title = max(counts, key=lambda k: counts[k]) if counts else ""

    return DeckInfo(
        theme=palette,
        geo=Geo(width, height, margin),
        title=deck_title,
        theme_key=theme_key,
        size_key=size_key,
        margin_key=margin_key,
        matched=bool(theme_key),
    )


@dataclass
class Unit:
    """一处可寻址的文字：某个形状里的某一段，或者是整张表。

    定位用「第几页 + 第几块」，块按 z-order 数、形状内部按段落数。
    之所以细到段落而不是停在形状：一页要点是一个文本框里的 5 个段落，
    只到形状的话，改第 2 条要点就得重写全部 5 条。
    """

    shape: Any
    para: int | None  # None 表示这是表格
    tag: str
    text: str
    extra: str = ""

    @property
    def is_table(self) -> bool:
        return self.para is None


def _tag(shape, para, text: str, info: DeckInfo) -> str:
    """给一块文字贴个标签。标签既是给人看的，也是版式推断的依据。

    顺序有讲究：页脚和页码先认，否则 10pt 的页码会被当成 body；
    然后是项目符号；再才是按字号分。KPI 数字和章节编号都很大，
    但一个短（1.28 亿）、一个更短（01），所以只要"大且短"就算 value。
    """
    size = max((r.font.size.pt for r in para.runs if r.font.size), default=0.0)
    top = _emu(shape.top)
    bottom_zone = top > info.geo.height - 0.7
    if re.fullmatch(r"\d{1,3}", text) and bottom_zone and _emu(shape.left) > info.geo.width / 2:
        return "pageno"
    if text == info.title and bottom_zone:
        return "footer"
    if text.startswith(("•", "–", "-", "·")):
        return "item"
    if size >= 30 and len(text) <= 12 and re.search(r"\d", text):
        return "value"
    if size >= 22:
        return "title"
    if size >= 17:
        return "lead"
    if size >= 14:
        return "head"
    return "body"


def _units_of(slide, info: DeckInfo) -> list[Unit]:
    units: list[Unit] = []
    for shape in slide.shapes:
        if getattr(shape, "has_table", False):
            rows = list(shape.table.rows)
            header = " | ".join(cell.text.strip() for cell in rows[0].cells)
            units.append(Unit(shape, None, "table", header, f"{len(rows)} 行含表头"))
            continue
        if not getattr(shape, "has_text_frame", False):
            continue
        for index, para in enumerate(shape.text_frame.paragraphs):
            text = "".join(run.text for run in para.runs).strip()
            if text:
                units.append(Unit(shape, index, _tag(shape, para, text, info), text))
    return units


def _guess_layout(slide, index: int, info: DeckInfo) -> str:
    """按形状特征反推版式。只用于给人看的标签，写错不影响功能，所以用「≈」表示。"""
    shapes = list(slide.shapes)
    if any(getattr(shape, "has_table", False) for shape in shapes):
        return "table"
    try:
        fill = slide.background.fill
        dark = fill.type is not None and _luminance(str(fill.fore_color.rgb)) < 0.45
    except Exception:  # noqa: BLE001
        dark = False
    units = _units_of(slide, info)
    if dark:
        # 章节页和结束页都是深底，区别只有一点：章节页有个大号编号
        numbered = any(re.fullmatch(r"\d{1,3}", unit.text) and unit.tag != "pageno" for unit in units)
        return "section" if numbered else "closing"
    if index == 1:
        return "cover"
    if sum(1 for unit in units if unit.tag == "value") >= 3:
        return "kpi"
    if sum(1 for shape in shapes if _prst(shape) == "roundRect") >= 2:
        return "compare"
    return "bullets"


def _short(text: str, limit: int = 56) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _outline_page(slide, index: int, info: DeckInfo) -> str:
    units = _units_of(slide, info)
    layout = _guess_layout(slide, index, info)
    title = next((u.text for u in units if u.tag == "title"), "")
    lines = [f"【第 {index} 页】{layout}" + (f"：{_short(title)}" if title else "")]
    if not units:
        lines.append("  （这一页没有文字）")
    for number, unit in enumerate(units, 1):
        tail = f"（{unit.extra}）" if unit.extra else ""
        lines.append(f"  [{number}] {unit.tag}  {_short(unit.text)}{tail}")
    return "\n".join(lines)


def describe_pptx(path: Path, *, part_chars: int = 1600) -> tuple[str, list[str], str]:
    """把一份 .pptx 读成「结构化大纲」，返回 (概览行, 分片, 备注)。

    为什么不是纯文本：下一步要按页改，模型必须知道"第几页、第几块"。
    纯文本读十遍也数不出来第 6 页第 3 块是哪一段。
    """
    prs = Presentation(str(path))
    info = _probe_deck(prs)
    pages = [_outline_page(slide, index, info) for index, slide in enumerate(prs.slides, 1)]

    parts: list[str] = []
    buffer: list[str] = []
    size = 0
    for page in pages:
        if buffer and size + len(page) > part_chars:
            parts.append("\n\n".join(buffer))
            buffer, size = [], 0
        buffer.append(page)
        size += len(page) + 2
    if buffer:
        parts.append("\n\n".join(buffer))

    # 页码、块号是 edit_pptx 的定位依据，顺带告诉模型怎么用
    note = "块号就是 edit_pptx 的 block，页码就是 page。"
    if not info.matched:
        note += "这份文件不是本工具排的，版式是从形状推断的（仅供参考）。"
    return f"{len(pages)} 页 · {info.label}", parts, note


# ---------------------------------------------------------------- 增量修改

# 一次修改能做的事就这四件。没有"随便改样式"——那会逼着模型去描述坐标和字号，
# 又回到"第 10 页压边框"的老路上。
OPS = ("insert", "replace", "delete", "move")


class _OpBase(TypedDict):
    op: Literal["insert", "replace", "delete", "move"]


class Op(_OpBase, total=False):
    """一次修改。用哪些字段由 op 决定，其余的不用传。"""

    after: int  # insert：插到第几页之后，0 表示最前面
    content: Slide  # insert：要插入的那一页，字段和 make_pptx 的 slides 完全一样
    page: int  # replace / move：第几页
    block: int  # replace：第几块（见 read_file 的大纲）
    text: str  # replace：新文字
    cell: list[int]  # replace：表格块用 [行, 列] 指定单元格
    pages: list[int]  # delete：要删哪几页
    to: int  # move：挪到第几页的位置


class _OpError(Exception):
    """这一次改动做不了。消息是给模型看的，要说清怎么改才对。"""


def _slide_at(prs, page: Any) -> Any:
    total = len(prs.slides._sldIdLst)
    if not isinstance(page, int) or isinstance(page, bool):
        raise _OpError(f"page 必须是数字，收到的是 {page!r}")
    if not 1 <= page <= total:
        raise _OpError(f"page 越界：一共 {total} 页，收到 {page}")
    return prs.slides[page - 1]


def _unit_at(units: list[Unit], block: Any, page: int) -> Unit:
    if not isinstance(block, int) or isinstance(block, bool):
        raise _OpError(f"block 必须是数字，收到的是 {block!r}")
    if not 1 <= block <= len(units):
        raise _OpError(f"第 {page} 页一共 {len(units)} 块，block 收到 {block}")
    return units[block - 1]


def _set_para_text(unit: Unit, text: str) -> None:
    """改一个段落的文字，保留它的字体字号颜色。

    整段替换而不是拼接 run：一个段落里出现两种字号，八成是之前拼出来的，
    继续拼下去只会越来越乱。
    """
    para = unit.shape.text_frame.paragraphs[unit.para or 0]
    runs = list(para.runs)
    if not runs:
        para.add_run().text = text
        return
    runs[0].text = text
    for extra in runs[1:]:
        extra._r.getparent().remove(extra._r)


def _shape_text_height(shape, width: float) -> float:
    """估一个文本框里的文字占多高（英寸）。和生成时用的是同一套估字规则。"""
    total = 0.0
    for para in shape.text_frame.paragraphs:
        text = "".join(run.text for run in para.runs)
        if not text:
            continue
        size = max((r.font.size.pt for r in para.runs if r.font.size), default=16)
        spacing = para.line_spacing if isinstance(para.line_spacing, float) else 1.2
        total += _lines(text, max(width - 0.1, 0.5), size) * size * spacing * LINE_FACTOR / 72
        if para.space_after:
            total += para.space_after.pt / 72
    return total


def _body_size(shape) -> float:
    return max(
        (r.font.size.pt for p in shape.text_frame.paragraphs for r in p.runs if r.font.size),
        default=0.0,
    )


def _scale_runs(shape, ratio: float) -> None:
    for para in shape.text_frame.paragraphs:
        for run in para.runs:
            if run.font.size:
                run.font.size = Pt(max(11.0, run.font.size.pt * ratio))


def _refit(shape) -> tuple[float, bool]:
    """换字之后重新量一遍，放不下就把这块的字号往下压。

    返回 (压到多少 pt, 是否放得下)。没动过字号时第一个值是 0。
    生成时可以截断，改文件时不行——用户原来的字不能被悄悄截掉。
    压到 11pt 还放不下，就把"仍然超出"如实报回去，由人决定删字还是换页。
    """
    width, height = _emu(shape.width), _emu(shape.height)
    if height <= 0 or width <= 0:
        return 0.0, True
    original = _body_size(shape)
    if not original:
        return 0.0, True
    for _ in range(14):
        if _shape_text_height(shape, width) <= height:
            break
        if _body_size(shape) <= 11.0:
            break
        _scale_runs(shape, 0.94)
    final = _body_size(shape)
    fits = _shape_text_height(shape, width) <= height
    shrunk = 0.0 if abs(final - original) < 0.05 else round(final, 1)
    return shrunk, fits


def _move_slide(prs, source: int, destination: int) -> None:
    """把第 source 页（0 起）挪到第 destination 个位置。

    python-pptx 只能往末尾加页，插入位置只能靠调 sldIdLst 的顺序——
    好在幻灯片就是这一层引用，挪它不会碰到任何一页的内容。
    """
    sequence = prs.slides._sldIdLst
    ids = list(sequence)
    element = ids[source]
    sequence.remove(element)
    sequence.insert(destination, element)


def _delete_slide(prs, page: int) -> None:
    sequence = prs.slides._sldIdLst
    element = list(sequence)[page - 1]
    # 先断关系再摘引用：只摘引用的话，那一页的部件还留在包里，文件只会越来越大
    prs.part.drop_rel(element.rId)
    sequence.remove(element)


def _blank_layout(prs):
    """找占位符最少的版式。插入的页要完全由我们画，占位符只会多出空的"单击添加文本"。"""
    layouts = list(prs.slide_layouts)
    empty = [layout for layout in layouts if not len(layout.placeholders)]
    return empty[0] if empty else min(layouts, key=lambda layout: len(layout.placeholders))


def _drop_placeholders(slide) -> None:
    for placeholder in list(slide.placeholders):
        element = placeholder._element
        element.getparent().remove(element)


def _renumber(prs, info: DeckInfo) -> None:
    """重排页码。插一页之后，后面每页的页脚都是错的，必须整份刷一遍。"""
    for index, slide in enumerate(prs.slides, 1):
        for unit in _units_of(slide, info):
            if unit.tag == "pageno" and unit.text != f"{index:02d}":
                _set_para_text(unit, f"{index:02d}")


def _geometry_warnings(prs, info: DeckInfo) -> list[str]:
    """改完自己量一遍：有没有元素出页面、正文有没有压到页脚。

    "压页脚"这一条只对**本工具排出来的稿子**成立——外部 PPT 没有"底部 0.72 英寸
    留给页脚"这个约定，拿我们的尺子去量它，只会在一份本来正常的稿子上报假警。
    """
    problems: list[str] = []
    for index, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            left, top = _emu(shape.left), _emu(shape.top)
            right, bottom = left + _emu(shape.width), top + _emu(shape.height)
            if left < -0.02 or top < -0.02 or right > info.geo.width + 0.02 or bottom > info.geo.height + 0.02:
                problems.append(f"第 {index} 页有元素超出页面（{right:.2f}×{bottom:.2f} 英寸）")
            elif (
                info.matched
                and bottom > info.geo.body_bottom + 0.02
                and top < info.geo.height - 0.7
            ):
                problems.append(f"第 {index} 页正文压到页脚区")
    return list(dict.fromkeys(problems))


def _do_insert(prs, op: dict, info: DeckInfo) -> str:
    total = len(prs.slides._sldIdLst)
    after = op.get("after", total)
    content = op.get("content")
    if not isinstance(content, dict):
        raise _OpError("insert 缺少 content，要插入的那一页写在 content 里")
    if not isinstance(after, int) or isinstance(after, bool) or not 0 <= after <= total:
        raise _OpError(f"after 要在 0~{total} 之间（0 = 插到最前面），收到的是 {after!r}")

    errors = _validate(
        [content],
        info.theme_key or "business",
        info.size_key or "16:9",
        info.margin_key or "normal",
    )
    if errors:
        raise _OpError("；".join(errors))

    layout = content["layout"]
    slide = prs.slides.add_slide(_blank_layout(prs))
    _drop_placeholders(slide)
    notes: list[str] = []
    painter = PAINTERS[layout]
    if layout in ("bullets", "kpi", "table", "compare"):
        painter(slide, content, info.geo, info.theme, info.title, notes, after + 1)
    else:
        painter(slide, content, info.geo, info.theme, info.title)
    if layout not in NO_FOOTER:
        _footer(slide, after + 1, info.geo, info.theme, info.title)
    _move_slide(prs, total, after)
    return f"第 {after} 页后插入 {layout}「{_short(str(content.get('title') or ''), 24)}」"


def _do_replace(prs, op: dict, info: DeckInfo) -> str:
    page = op.get("page")
    slide = _slide_at(prs, page)
    units = _units_of(slide, info)
    unit = _unit_at(units, op.get("block"), page)
    text = op.get("text")
    if not isinstance(text, str):
        raise _OpError("replace 需要 text（新的文字）")

    if unit.is_table:
        cell = op.get("cell")
        if not isinstance(cell, list) or len(cell) != 2:
            raise _OpError(
                f"第 {page} 页第 {op.get('block')} 块是表格，改单元格要传 cell=[行, 列]"
                "（都从 0 开始，0 是表头行）；整块换文字做不到。"
            )
        row, column = cell
        table = unit.shape.table
        if not 0 <= row < len(table.rows) or not 0 <= column < len(table.columns):
            raise _OpError(
                f"cell 越界：这张表 {len(table.rows)} 行 × {len(table.columns)} 列（都从 0 开始）"
            )
        _set_para_text(Unit(table.cell(row, column), 0, "cell", ""), text)
        return f"第 {page} 页表格 [{row},{column}] 已替换"

    before = len(unit.text)
    _set_para_text(unit, text)
    shrunk, fits = _refit(unit.shape)
    message = f"第 {page} 页第 {op.get('block')} 块（{unit.tag}）已替换：{before} 字 → {len(text)} 字"
    if shrunk:
        message += f"，字号压到 {shrunk}pt 才放下"
    if not fits:
        message += "，但压到 11pt 仍然超出这一块的高度，建议把文字缩短"
    return message


def _do_delete(prs, op: dict, info: DeckInfo) -> str:
    pages = op.get("pages")
    total = len(prs.slides._sldIdLst)
    if not isinstance(pages, list) or not pages:
        raise _OpError("delete 需要 pages，比如 [3, 5]")
    for page in pages:
        _slide_at(prs, page)
    unique = sorted({int(p) for p in pages})
    if len(unique) >= total:
        raise _OpError(f"不能把 {total} 页全删了，至少要留一页")
    for page in reversed(unique):
        _delete_slide(prs, page)
    return f"删除第 {'、'.join(str(p) for p in unique)} 页"


def _do_move(prs, op: dict, info: DeckInfo) -> str:
    page = op.get("page")
    _slide_at(prs, page)
    total = len(prs.slides._sldIdLst)
    to = op.get("to")
    if not isinstance(to, int) or isinstance(to, bool) or not 1 <= to <= total:
        raise _OpError(f"to 要在 1~{total} 之间，收到的是 {to!r}")
    _move_slide(prs, page - 1, to - 1)
    return f"第 {page} 页挪到第 {to} 页的位置"


_HANDLERS = {
    "insert": _do_insert,
    "replace": _do_replace,
    "delete": _do_delete,
    "move": _do_move,
}


@registry.tool
def edit_pptx(path: str, ops: list[Op], backup: bool = True) -> str:
    """在**已经存在的** .pptx 上做增量修改：插页、改文字、删页、挪位置。

    什么时候用：用户说"往这份 PPT 里加一页 / 把第 5 页那个数字改成 1.42 亿 /
    删掉第 3 页"。**不要为了改几页就重新调用 make_pptx 重做整份**——重建会丢掉
    用户后来手动调过的东西，也会让每一页都变成新对象，改动范围说不清。

    定位方式：page 是第几页，block 是那一页的第几块。
    这两个号必须先用 read_file 读出来，不要自己数——数错了改的是别人的内容。

    四种操作，按 ops 顺序执行，后面的操作看到的是前面执行后的页码：
    - insert：after（插到第几页之后，0 = 最前面）+ content（要插入的那一页）
    - replace：page + block + text；如果那块是表格，改单元格再传 cell=[行, 列]（从 0 开始）
    - delete：pages（页码数组）
    - move：page + to（挪到第几页的位置）

    几何、配色、字体、页码都会沿用原文件；改完自动重排页码并做几何自检。
    写入前默认在同目录留一份 .bak-时间戳 备份，整个操作要么全成、要么文件不动。

    Args:
        path: 要改的 .pptx 路径，支持 ~ 和 桌面 这类写法
        ops: 要做的修改，按顺序执行
        backup: 是否在改之前留备份，默认 True
    """
    if not files.allow_write():
        return (
            "错误：当前是「只读」模式，不能改文件。"
            "请让用户在输入框上方把模式切到「完全工作区」再试。"
        )

    try:
        target = files.locate(path, must_exist=True)
    except files.FileError as exc:
        return f"错误：{exc}"
    if target.suffix.lower() != ".pptx":
        return f"错误：只能改 .pptx，`{target.name}` 不是。旧版 .ppt 请先另存为 .pptx。"
    if not ops:
        return "错误：ops 不能为空。要做什么改动写在 ops 里。"

    try:
        prs = Presentation(str(target))
    except Exception as exc:  # noqa: BLE001  文件损坏、加密都会到这里
        return f"错误：打不开 `{target}`（{exc}）。如果设了密码，请先解密。"

    info = _probe_deck(prs)
    before_pages = len(prs.slides._sldIdLst)
    messages: list[str] = []

    for index, op in enumerate(ops, 1):
        if not isinstance(op, dict) or op.get("op") not in _HANDLERS:
            return (
                f"错误：第 {index} 个操作不合法，op 只能是 {'/'.join(OPS)}。文件没有改动。"
            )
        try:
            messages.append(_HANDLERS[op["op"]](prs, op, info))
        except _OpError as exc:
            # 之前那些操作已经在内存里的对象上生效了，但没存盘 = 等于没发生
            return f"错误：第 {index} 个操作没做成（{exc}）。文件没有改动。"

    _renumber(prs, info)
    warnings = _geometry_warnings(prs, info)

    stamped = ""
    if backup:
        bak = target.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}.pptx")
        try:
            shutil.copy2(target, bak)
            stamped = str(bak)
        except OSError as exc:  # noqa: BLE001  备份失败就别改，免得不可逆
            return f"错误：备份失败（{exc}），为了不丢原文件这次没有写入。"

    try:
        prs.save(str(target))
    except OSError as exc:
        return f"错误：写入失败（{exc}）。" + (f"原文件备份在 {stamped}。" if stamped else "")

    after_pages = len(prs.slides._sldIdLst)
    lines = [
        f"已更新：{target}",
        f"{before_pages} 页 → {after_pages} 页 · {len(messages)} 处改动 · 沿用原文件 {info.label}",
    ]
    lines += [f"- {message}" for message in messages]
    if stamped:
        lines.append(f"备份：{stamped}")
    if warnings:
        lines.append("注意：" + "；".join(warnings))
    return "\n".join(lines)


# ---------------------------------------------------------------- 入口


def _safe_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]', "", str(name or "")).strip()
    return cleaned[:40]


@registry.tool
def make_pptx(
    title: str,
    slides: list[Slide],
    theme: Literal["business", "academic"] = "business",
    size: Literal["16:9", "4:3"] = "16:9",
    margin: Literal["narrow", "normal", "wide"] = "normal",
) -> str:
    """把结构化内容排成一份 .pptx 演示文稿，保存到 output/。

    什么时候用：要做多页汇报、方案、总结、答辩时用它。只想看一张图的趋势用
    make_chart，只想看层级关系用 make_mindmap，要成篇的页面才用这个。

    七种版式，每页用 layout 指定：
    - cover 封面：title, subtitle
    - section 章节过渡页（深色）：title, number, subtitle
    - bullets 要点页：title, lead, items（某条以两个空格开头表示二级）
    - kpi 指标页：title, items，每项写成「数值|标签|变化」，1~4 项
    - table 表格页：title, headers, rows
    - compare 两栏对比：title, left_title, left_items, right_title, right_items
    - closing 结束页（深色）：title, subtitle

    字号会按内容多少自动降档，放不下才截断并在返回里说明。表格最多 12 行，
    要点每页最多 8 条——超了请拆页，压到 8pt 也没人看得清。

    Args:
        title: 演示文稿标题，显示在页脚左侧
        slides: 每页内容，按播放顺序排列
        theme: 配色，business 商务蓝 / academic 学术灰
        size: 画布比例，16:9 宽屏或 4:3 传统投影
        margin: 页边距，narrow 窄 / normal 标准 / wide 宽
    """
    errors = _validate(slides, theme, size, margin)
    if errors:
        return "错误：\n" + "\n".join(f"- {e}" for e in errors)

    width, height = SIZES[size]
    geo = Geo(width=width, height=height, margin=MARGINS[margin])
    palette = THEMES[theme]

    prs = Presentation()
    prs.slide_width = Inches(width)
    prs.slide_height = Inches(height)
    blank = prs.slide_layouts[6]

    notes: list[str] = []
    for index, page in enumerate(slides, start=1):
        slide = prs.slides.add_slide(blank)
        layout = page["layout"]
        painter = PAINTERS[layout]
        if layout in ("bullets", "kpi", "table", "compare"):
            painter(slide, page, geo, palette, title, notes, index)
        else:
            painter(slide, page, geo, palette, title)
        if layout not in NO_FOOTER:
            _footer(slide, index, geo, palette, title)

    name = _safe_name(title) or "演示文稿"
    path = OUTPUT_DIR / f"{name}_{int(time.time() * 1000)}.pptx"
    prs.save(str(path))

    lines = [
        f"演示文稿已生成：{path}",
        f"{len(slides)} 页 · {size} · {palette.label} · 边距 {MARGINS[margin]:.3g} 英寸",
    ]
    if notes:
        lines.append("注意：" + "；".join(dict.fromkeys(notes)))
    return "\n".join(lines)
