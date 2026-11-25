"""前置查询改写（Pre-LLM Query Rewriting）。

位置在主循环**之前**：用户那句话还没发给主模型，先过一次改写层，决定原样透传，
还是补全指代、扩同义词、拆成多路或分解成有依赖的几步。

为什么值得单独一层：

    多轮追问里的"北京呢？"、口语化的"卖得最差的那个品类"、术语型的"违约怎么赔"，
    主模型并非答不了，而是**要多花几步去猜**。猜错一次就是一轮工具调用白烧。
    把这些活儿提前压成一句自包含的完整问题，主循环第一步就能直接动手。

三条硬边界（写在代码里，不只是口号）：

  1. **不碰 session.messages**。改写只影响本轮发给模型的 request_messages 浅拷贝，
     持久化历史里永远只有用户原话。否则多轮下来上下文会被模板撑爆，
     而且下一轮采集历史时会读到上一轮自己生成的模板。
  2. **失败一律静默降级**。超时、HTTP 报错、JSON 解析失败、字段乱填，全部退回透传，
     绝不因为"改写没做成"而阻塞主问答。
  3. **单次调用做两件事**。策略路由和改写执行合并成一次结构化 JSON 请求：
     前置改写是纯串行阻塞，拆成两次串行请求会让首字节时间翻倍。
"""

from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from . import files
from .config import (
    QUERY_REWRITE_HISTORY_TURNS,
    QUERY_REWRITE_MAX_DECOMPOSE,
    QUERY_REWRITE_MAX_EXPANSIONS,
    QUERY_REWRITE_MAX_MULTI,
    QUERY_REWRITE_MAX_TOKENS,
    QUERY_REWRITE_TIMEOUT,
)
from .llm import LLMCallable, chat
from .tools import data_tools


class RewriteStrategy(str, Enum):
    PASS_THROUGH = "pass_through"
    BASIC_REWRITE = "basic_rewrite"
    QUERY_EXPANSION = "query_expansion"
    WEB_SEARCH = "web_search"
    MULTI_QUERY = "multi_query"
    DECOMPOSITION = "decomposition"


STRATEGY_LABELS: dict[RewriteStrategy, str] = {
    RewriteStrategy.PASS_THROUGH: "直接透传",
    RewriteStrategy.BASIC_REWRITE: "基础改写",
    RewriteStrategy.QUERY_EXPANSION: "查询扩展",
    RewriteStrategy.WEB_SEARCH: "联网检索",
    RewriteStrategy.MULTI_QUERY: "多路查询",
    RewriteStrategy.DECOMPOSITION: "查询分解",
}


@dataclass(frozen=True)
class SubQuery:
    id: str                                # "Q1", "Q2", ...
    query: str                             # 自包含的子查询 / 子问题
    angle: str = ""                        # multi_query 使用：视角标签，如 "总量趋势"
    purpose: str = ""                      # decomposition 使用：该步目标 / 中间产出
    depends_on: tuple[str, ...] = ()       # decomposition 使用：前置依赖 ID，如 ("Q1",)
    target_source: str = "auto"            # "table" | "document" | "mixed" | "workspace" | "auto"
    # 这条子查询自己的纯英文检索词。
    # 为什么挂在子查询上而不是整份结果上：英文词的作用域必须等于**检索单元**的作用域。
    # decomposition 拆出来的是"SPV 与全节点的差异"和"防双花机制"两件事，共用一份词包
    # 等于让两路检索互相污染，检索结果趋同，分解这一步就白做了。
    english_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteContext:
    history: tuple[tuple[str, str], ...] = ()   # 最近 N 轮 (role, concise_text)
    has_table: bool = False
    table_label: str = ""
    columns: tuple[str, ...] = ()
    table_rows: int | None = None
    documents: tuple[dict[str, Any], ...] = ()  # [{"name", "kind", "chars", "parts"}]
    write_mode: bool = False
    web_search: bool = False


@dataclass(frozen=True)
class RewriteResult:
    original: str
    strategy: RewriteStrategy = RewriteStrategy.PASS_THROUGH
    reason: str = ""
    rewritten_query: str = ""              # 规范化后的主问题（pass_through 时等于 original）
    expanded_terms: tuple[str, ...] = ()   # query_expansion: 同义词 / 近义词 / 术语
    search_keywords: tuple[str, ...] = ()  # query_expansion: 可直接传给 search_document 的词组
    # 单查询策略（basic_rewrite / query_expansion）的英文检索词。
    # 不变量：sub_queries 非空时这里必须为空——有子查询时检索单元是子查询，
    # 英文词归 sub_queries[].english_terms 管，两个地方都放就会自相矛盾。
    english_keywords: tuple[str, ...] = () # query_expansion: 跨语言检索用的纯英文检索词组
    sub_queries: tuple[SubQuery, ...] = () # multi_query / decomposition 的子查询列表
    tokens: int = 0
    latency: float = 0.0
    bypassed: bool = False                 # True = 命中快速通道，未调用 LLM
    fallback: bool = False                 # True = 异常 / 解析失败降级透传
    error: str = ""

    @property
    def effective_query(self) -> str:
        return self.rewritten_query or self.original

    @property
    def changed(self) -> bool:
        return self.strategy != RewriteStrategy.PASS_THROUGH

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "strategy": self.strategy.value,
            "label": STRATEGY_LABELS[self.strategy],
            "reason": self.reason,
            "rewritten_query": self.effective_query,
            "expanded_terms": list(self.expanded_terms),
            "search_keywords": list(self.search_keywords),
            "english_keywords": list(self.english_keywords),
            "sub_queries": [
                {
                    "id": sq.id,
                    "query": sq.query,
                    "angle": sq.angle,
                    "purpose": sq.purpose,
                    "depends_on": list(sq.depends_on),
                    "target_source": sq.target_source,
                    "english_terms": list(sq.english_terms),
                }
                for sq in self.sub_queries
            ],
            "tokens": self.tokens,
            "latency": self.latency,
            "bypassed": self.bypassed,
            "fallback": self.fallback,
            "error": self.error,
        }


@dataclass(frozen=True)
class RetrievalUnit:
    """一个**检索单元**：一次 search_document 调用的输入，加上它自己的英文词。

    存在的意义是把"检索单元"从"改写结果"里拆出来。改写结果有五种策略，
    但检索单元只有两类形态：整条查询（单查询策略）或子查询（多路/分解）。
    下游（指令包、评测脚本）只认这个接口，就不会再出现"把全局词包硬塞给
    三条互不相干的子查询"这种错配。
    """

    id: str
    query: str
    english_terms: tuple[str, ...] = ()
    target_source: str = "auto"


def retrieval_units(result: RewriteResult) -> tuple[RetrievalUnit, ...]:
    """把改写结果摊平成检索单元列表。

    三条规则：
      - pass_through 不产生检索单元：原话直接透传，没做任何改写，也就没有英文词可说。
      - 有子查询（multi_query / decomposition）：一节子查询 = 一个单元，各带各的英文词。
      - 其余（basic_rewrite / query_expansion）：整条查询就是一个单元，英文词取结果级字段。
    """
    if result.strategy is RewriteStrategy.PASS_THROUGH:
        return ()
    if result.sub_queries:
        return tuple(
            RetrievalUnit(
                id=sq.id,
                query=sq.query,
                english_terms=sq.english_terms,
                target_source=sq.target_source,
            )
            for sq in result.sub_queries
        )
    return (
        RetrievalUnit(
            id="Q1",
            query=result.effective_query,
            english_terms=result.english_keywords,
        ),
    )


# ---------------------------------------------------------------- 上下文采集


def _flatten(text: Any, limit: int) -> str:
    """把一段消息压成一行，超长截断。

    历史里塞进整段表格或代码块的唯一后果，是改写请求的 prompt 比主请求还长，
    而改写器真正需要的只是"上一轮聊的是哪张表、哪个指标"。
    """
    if isinstance(text, list):
        text = " ".join(
            str(part.get("text") or "")
            for part in text
            if isinstance(part, dict) and part.get("type") == "text"
        )
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _history(
    messages: list[dict[str, Any]], *, turns: int, max_msg_chars: int
) -> tuple[tuple[str, str], ...]:
    """只留干净的对话轮次。

    跳过 system / tool / 带 tool_calls 的 assistant：这些是协议脚手架，
    夹在历史里既费 token 又容易让改写模型以为"用户说过"那些函数参数。
    """
    limit = max(0, int(turns)) * 2
    if not limit:
        return ()

    kept: list[tuple[str, str]] = []
    for message in messages or []:
        role = message.get("role")
        if role == "user":
            pass
        elif role == "assistant" and not message.get("tool_calls"):
            pass
        else:
            continue
        text = _flatten(message.get("content"), max_msg_chars)
        if text:
            kept.append((role, text))
    return tuple(kept[-limit:])


def build_context(
    messages: list[dict[str, Any]],
    *,
    session_key: str | None = None,
    write_mode: bool | None = None,
    web_search: bool = False,
    history_turns: int = QUERY_REWRITE_HISTORY_TURNS,
    max_msg_chars: int = 240,
) -> RewriteContext:
    """采集改写需要的轻量上下文：最近对话 + 表结构 + 文档目录 + 文件权限。

    文档**只取摘要**（名字/类型/字数/段数），绝不读正文：前置改写多做一次全文读取，
    首屏就多等半秒，而改写器判断"该不该拆成多路"根本用不上正文。

    这里任何一步失败都只丢元数据，不抛异常——没绑定数据的线程、表刚被撤下的瞬间，
    都不该让"改写"变成一次崩溃。
    """
    key = session_key or data_tools.current_key()
    if session_key and data_tools.current_key() != session_key:
        # 多标签页并发时，模块级 ContextVar 里可能是别的会话的值。
        data_tools.use_dataset(session_key)

    has_table = False
    label = ""
    columns: tuple[str, ...] = ()
    rows: int | None = None
    documents: tuple[dict[str, Any], ...] = ()
    try:
        if data_tools.has_dataset(key):
            meta = data_tools.dataset_meta()
            has_table = True
            label = str(meta.get("label") or "")
            raw_rows = meta.get("rows")
            rows = int(raw_rows) if raw_rows is not None else None
            columns = tuple(data_tools.column_names())
        if data_tools.has_documents(key):
            documents = tuple(dict(d) for d in data_tools.document_digest(key))
    except Exception:  # noqa: BLE001  上下文只是锦上添花，采集失败就当没有
        has_table, label, columns, rows, documents = False, "", (), None, ()

    return RewriteContext(
        history=_history(
            messages, turns=history_turns, max_msg_chars=max_msg_chars
        ),
        has_table=has_table,
        table_label=label,
        columns=columns,
        table_rows=rows,
        documents=documents,
        write_mode=files.allow_write() if write_mode is None else bool(write_mode),
        web_search=bool(web_search),
    )


# ---------------------------------------------------------------- Prompt


REWRITE_SYSTEM_PROMPT = """你是数据分析系统的前置查询改写器（Query Rewriter）。
你的唯一任务是：结合当前会话的上下文元数据与最近对话历史，分析用户的最新提问，选择最恰当的一种处理策略并输出改写结果。

## 可选策略（只能六选一，按以下优先级判断）
1. decomposition（查询分解）：
   - 适用：包含多个先后依赖步骤、对比归因（先找异常再查原因）、跨数据源协同（例如既要查表/文档又要联网查外部数据，或分析后再画图/生成PPT）的复杂任务。
   - 产出：2~5 个带依赖关系（depends_on）的有向无环子查询，且每条子查询必须自包含（消除指代），各自准确标注 target_source（联网子步骤标为 web）。
2. multi_query（多路查询）：
   - 适用：宽泛、开放、探索型问题，单一查询容易以偏概全（如"分析整体表现""总结亮点与风险""各区域情况怎么样"）。
   - 产出：2~4 条互补视角的并行子查询（depends_on 为空数组），并标明各自视角（angle）与 target_source。
3. web_search（联网检索）：
   - 适用：必须同时满足两个条件——① 上下文显示「联网搜索：已开启」；② 从用户提问与对话历史中能明确看出用户想获取互联网外部实时资讯、公开网页/URL 内容、行业外部数据或开源/公共知识，而不是在查询已上传的数据表或本地资料（例如明确说"联网查/搜一下网上的/看看这个网页/最新动态"，或问题主题与已有表、资料文件名明显无关且需要外部事实）。
   - 产出：消除指代后的完整搜索目标 rewritten_query，以及 1~4 组适合搜索引擎的简短检索词 search_keywords（如需检索英文网页可附 english_keywords）。
   - 严格限制：若「联网搜索：未开启」，严禁选择此策略；若用户只是在问已上传数据表或本地资料里的内容（即使联网开关已开启），也绝不能选择此策略。
4. query_expansion（查询扩展）：
   - 适用：面向**本地资料文档（search_document）或表内文本筛选**的检索型/概念型问题，用户用词单一、口语化或仅用缩写，字面匹配容易漏掉结果；也适用于本地资料的跨语言检索场景（如中文提问、资料列表中存在英文文档如 `*_en.pdf` 或英文文件名）。
   - 产出：规范化后的 rewritten_query、3~6 个高相关同义词/业务术语/中英等价词（expanded_terms），以及 1~4 组可直接用于空格 OR 关键词检索的词组（search_keywords），外加 1~2 组纯英文检索词（english_keywords，写法要求见硬性约束第 5 条）。
5. basic_rewrite（基础改写）：
   - 适用：多轮追问中的指代/省略补全（如"北京呢？""把它画成折线图"）、口语化或错别字纠正、将模糊表述对齐到上下文中的真实表列名或文件名。
   - 产出：单条完整、自包含的 rewritten_query。
6. pass_through（直接透传）：
   - 适用：单步清晰、无歧义、未依赖省略信息、且无需术语扩展、联网路由或多路拆解的问题。
   - 产出：无需修改，rewritten_query 填原问题或留空。

## 硬性约束
1. 必须仅输出一个合法的 JSON 对象，禁止输出 Markdown 围栏以外的任何说明文字。
2. 严禁编造元数据：只能引用上下文中真实列出的字段名与资料文件名；不确定时保留用户原始概念词。
3. 只要选择了非 pass_through 策略，必须在 rewritten_query 与所有 sub_queries[].query 中一并完成指代消解与实体对齐。
4. 保持克制：不要把简单明确的 SQL 统计问题强行拆成 multi_query 或 decomposition；不要因为联网已开启就把本来问本地表/本地文件的问题误判为 web_search。
5. 跨语言强制项：只要「当前可用数据源」里存在英文文件名的资料，且策略不是 pass_through 或 web_search，就必须产出纯英文检索词。
   - 单查询策略（basic_rewrite / query_expansion）：写在顶层的 english_keywords。
   - multi_query / decomposition：写在每条 sub_queries[].english_terms 里，**每条子查询只写它自己的主题词**，不要把别的子查询的术语塞进来，否则几路检索会互相污染、结果趋同，分解就失去意义。
   - 写法：空格分隔，仅含英文专业术语/专有名词/实词，不夹杂中文，不带 how/what/the/of 等虚词，避免堆砌过泛的词。
6. 输出 JSON 结构固定如下：
{
  "strategy": "pass_through | basic_rewrite | query_expansion | web_search | multi_query | decomposition",
  "reason": "不超过30字的决策理由",
  "rewritten_query": "改写后的完整清晰问题",
  "expanded_terms": ["同义词1", "术语2"],
  "search_keywords": ["关键词A 关键词B", "同义词C 术语D"],
  "english_keywords": ["English domain term A term B"],
  "sub_queries": [
    {
      "id": "Q1",
      "query": "自包含的子查询内容",
      "angle": "multi_query 时填视角标签，否则空串",
      "purpose": "decomposition 时填步骤产出目标，否则空串",
      "depends_on": [],
      "target_source": "table | document | web | mixed | workspace | auto",
      "english_terms": ["English terms for this sub-query only"]
    }
  ]
}"""


_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _has_english_doc(documents: tuple[dict[str, Any], ...]) -> bool:
    """资料列表中是否存在纯英文文件名的文档（不含汉字但含英文字母）。"""
    for doc in documents:
        stem = str(doc.get("name") or "").rsplit(".", 1)[0]
        if _ASCII_LETTER_RE.search(stem) and not _CJK_RE.search(stem):
            return True
    return False


def build_rewrite_user_prompt(question: str, ctx: RewriteContext) -> str:
    """把上下文摊成一段人类和模型都读得懂的说明。

    顺序刻意固定：先"有什么"（数据源），再"刚才聊到哪"（历史），最后"现在问什么"。
    模型判断策略时最需要的是前两段——没有它们，改写只能靠猜。
    """
    if ctx.has_table:
        table = f"- 数据表：`{ctx.table_label or '当前表'}`"
        detail: list[str] = []
        if ctx.table_rows is not None:
            detail.append(f"共 {ctx.table_rows:,} 行")
        if ctx.columns:
            detail.append("字段：" + "、".join(ctx.columns))
        if detail:
            table += "（" + "，".join(detail) + "）"
    else:
        table = "- 数据表：无数据表"

    if ctx.documents:
        listing = "\n".join(
            f"  - `{doc.get('name', '?')}`（{doc.get('kind', '?')}，"
            f"{int(doc.get('chars') or 0):,} 字，{doc.get('parts', 1)} 段）"
            for doc in ctx.documents
        )
        documents = f"{len(ctx.documents)} 份：\n{listing}"
        if _has_english_doc(ctx.documents):
            documents += (
                "\n  （检测到英文文件名资料：无论选择何种策略（pass_through 除外）都必须产出纯英文检索词"
                "——单查询写在 english_keywords，子查询写在各条 sub_queries[].english_terms）"
            )
    else:
        documents = "无资料文件"

    if ctx.history:
        history = "\n".join(
            f"{'U' if role == 'user' else 'A'}: {text}" for role, text in ctx.history
        )
    else:
        history = "（首轮提问，无历史）"

    permission = (
        "完全工作区（可读写/打开本机文件）" if ctx.write_mode else "只读"
    )
    web_status = (
        "已开启（仅当用户明确想查互联网外部信息或公开网页时才选择 web_search 或标注 web 数据源）"
        if ctx.web_search
        else "未开启（严禁选择 web_search 策略或 web 数据源）"
    )
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")

    return (
        "## 当前可用数据源\n"
        f"{table}\n"
        f"- 资料文件：{documents}\n"
        f"- 本地文件权限：{permission}\n"
        f"- 联网搜索：{web_status}\n"
        f"- 当前本地时间：{now_str}（涉及\"最近/今年/本月\"等相对时间时以此为准，勿猜过时年份）\n\n"
        "## 最近对话历史\n"
        f"{history}\n\n"
        "## 用户最新提问\n"
        f"{question}"
    )


# ---------------------------------------------------------------- 规则快速通道

# 归一化（去空白与常见中英文标点）后精确命中的控制 / 确认 / 礼貌短语。
# 只放"确定不需要理解上下文"的词：多一个字都不进这个集合，
# 因为短路掉一次真实追问，比多调一次模型贵得多。
_BYPASS_PHRASES = {
    "继续", "接着做", "继续做", "继续吧", "好的", "好", "ok", "okay",
    "谢谢", "感谢", "明白了", "收到", "不用了", "算了",
}

# 复合任务连接词：出现任意一个，就不再是"单步打开"，必须交给模型判断。
_COMPOUND_MARKERS = ("并", "并且", "然后", "同时", "顺便", "以及", "再", "之后")

_OPEN_RE = re.compile(r"^(打开|启动)\s*\S{1,24}$")

_TRAILING_PUNCT = " \t\r\n，。！？,.!?~～"


def _normalize_phrase(text: str) -> str:
    return str(text or "").strip().strip(_TRAILING_PUNCT).lower()


def _fast_bypass(
    question: str, ctx: RewriteContext, *, enabled: bool = True
) -> RewriteResult | None:
    """0 token、0 ms 的确定性短路。命中返回结果，否则返回 None。

    这里的每条规则都必须是"确定不需要理解上下文"的：开关关闭、空白输入、
    纯控制短语、单步打开程序。像"北京呢？""为什么？"这种短追问**绝不能**因为
    字数少就短路——它们恰恰是最需要结合历史补全的类型。
    """
    text = question or ""
    if not enabled or not text.strip():
        return RewriteResult(
            original=text,
            rewritten_query=text,
            bypassed=True,
            reason="未启用改写或问题为空",
        )

    if _normalize_phrase(text) in _BYPASS_PHRASES:
        return RewriteResult(
            original=text,
            rewritten_query=text,
            bypassed=True,
            reason="控制/确认短语，无需改写",
        )

    stripped = text.strip()
    if (
        ctx.write_mode
        and _OPEN_RE.match(stripped)
        and not any(marker in stripped for marker in _COMPOUND_MARKERS)
    ):
        return RewriteResult(
            original=text,
            rewritten_query=text,
            bypassed=True,
            reason="单步打开指令，无需改写",
        )

    return None


# ---------------------------------------------------------------- 容错解析

_ALLOWED_SOURCES = {"table", "document", "web", "mixed", "workspace", "auto"}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _dedupe(items: list[str]) -> list[str]:
    """去重且保序。用 set 做判重，但输出顺序仍按模型给的顺序走。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _squash(text: str) -> str:
    """去掉所有空白后比较。"把它改成折线图" 和 "把它改成 折线图" 是同一句话。"""
    return re.sub(r"\s+", "", str(text or ""))


def _clean_english_keywords(raw: Any, cap: int = 3) -> tuple[str, ...]:
    """把模型给的英文检索词洗成能直接塞进 search_document 的词组。

    四条清洗规则，每条都对应一种真实输出：
      - 压空白：模型爱写 "Simplified   Payment Verification"；
      - 去中文：中英混杂的组合在纯英文语料里反而会拖低 BM25 命中；
      - 去单字符：单个字母（"a"）在倒排表里命中的是一切，纯噪声；
      - 去重 + 截断：同一组词模型常常换大小写重复给。
    """
    out: list[str] = []
    for item in _as_list(raw):
        keyword = " ".join(str(item).split())
        if len(keyword) < 2:
            continue
        if _CJK_RE.search(keyword) or not _ASCII_LETTER_RE.search(keyword):
            continue
        out.append(keyword)
    return tuple(_dedupe(out)[:cap])


def _slice_braces(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else ""


def _load_json(raw: str) -> dict[str, Any] | None:
    """从模型输出里刨出一个 JSON 对象。

    三重容错：剥 ``` 围栏、直接 loads、再截首尾花括号重试。
    不做 `response_format={"type": "json_object"}` 的硬保证——不少国产和本地网关
    收到这个字段直接 400，用 prompt 约定 + 事后容错更稳。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    for candidate in (text, _slice_braces(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _source(value: Any) -> str:
    source = str(value or "").strip().lower()
    return source if source in _ALLOWED_SOURCES else "auto"


def _pass(original: str, reason: str, tokens: int, latency: float) -> RewriteResult:
    return RewriteResult(
        original=original,
        strategy=RewriteStrategy.PASS_THROUGH,
        reason=reason,
        rewritten_query=original,
        tokens=tokens,
        latency=latency,
    )


def _degrade(
    original: str,
    rewritten: str,
    reason: str,
    tokens: int,
    latency: float,
) -> RewriteResult:
    """子查询不够用时的退路：能当一句话用就降成基础改写，否则透传。

    "只有 1 条子查询"不是错误，只是模型把 multi_query 用过头了。
    把它降级成 basic_rewrite，这一轮仍然能拿到比原文更清楚的问题。
    """
    if rewritten and _squash(rewritten) != _squash(original):
        return RewriteResult(
            original=original,
            strategy=RewriteStrategy.BASIC_REWRITE,
            reason=reason,
            rewritten_query=rewritten,
            tokens=tokens,
            latency=latency,
        )
    return _pass(original, reason, tokens, latency)


def _raw_sub_queries(
    data: dict[str, Any], cap: int, *, web_search_enabled: bool = True
) -> list[dict[str, Any]]:
    """把 sub_queries 洗成干净的行：过滤空 query 与完全重复项，裁到上限。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _as_list(data.get("sub_queries")):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query or query in seen:
            continue
        seen.add(query)
        src = _source(item.get("target_source"))
        if not web_search_enabled and src == "web":
            src = "auto"
        out.append(
            {
                "id": str(item.get("id") or "").strip(),
                "query": query,
                "angle": str(item.get("angle") or "").strip(),
                "purpose": str(item.get("purpose") or "").strip(),
                "depends_on": [
                    str(dep).strip()
                    for dep in _as_list(item.get("depends_on"))
                    if str(dep).strip()
                ],
                "target_source": src,
                # 英文词跟着子查询走：这条子查询的主题词只服务它自己那一路检索
                "english_terms": _clean_english_keywords(item.get("english_terms")),
            }
        )
        if len(out) >= cap:
            break
    return out


def _multi(items: list[dict[str, Any]]) -> tuple[SubQuery, ...]:
    """多路查询：重编 ID、强制清空依赖（每一路都是并行的）。"""
    return tuple(
        SubQuery(
            id=f"Q{index + 1}",
            query=item["query"],
            angle=item["angle"],
            target_source=item["target_source"],
            english_terms=item["english_terms"],
        )
        for index, item in enumerate(items)
    )


def _decompose(items: list[dict[str, Any]]) -> tuple[SubQuery, ...]:
    """分解：过滤非法依赖 -> Kahn 拓扑排序 -> 破环 -> 按新顺序重编 Q1..Qn。

    为什么非要拓扑排序，而不是按模型给的顺序照抄：依赖关系是给主循环看的执行顺序，
    模型把 Q2 写在 Q1 前面是常事（它按重要性写，不按执行先后写）。排序一次，
    下游直接从上往下读就是合法的执行计划。

    环怎么办：真的出现 `Q1 -> Q2 -> Q1` 时，Kahn 排不完。此时把剩下的节点按原顺序
    接到末尾，并**只保留指向已输出节点的依赖**——这等于丢掉成环的那几条边，
    输出一定无环，且不会丢子查询本身。宁可少一条依赖边，也不能让下游拿到一个环。
    """
    count = len(items)
    ids: list[str] = []
    used: set[str] = set()
    for index, item in enumerate(items):
        candidate = item["id"] or f"Q{index + 1}"
        suffix = 2
        base = candidate
        while candidate in used:
            candidate = f"{base}#{suffix}"
            suffix += 1
        used.add(candidate)
        ids.append(candidate)

    known = set(ids)
    deps: list[set[str]] = [
        {dep for dep in item["depends_on"] if dep in known and dep != ids[index]}
        for index, item in enumerate(items)
    ]

    children: dict[str, list[int]] = {node: [] for node in ids}
    indegree: dict[int, int] = {}
    for index in range(count):
        indegree[index] = len(deps[index])
        for dep in deps[index]:
            children[dep].append(index)

    ready = sorted(index for index in range(count) if indegree[index] == 0)
    order: list[int] = []
    while ready:
        ready.sort()                       # 同层按原始顺序，输出稳定、可断言
        index = ready.pop(0)
        order.append(index)
        for child in children[ids[index]]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    emitted = {ids[index] for index in order}
    for index in range(count):             # 原顺序补齐成环节点，并丢掉指向未输出节点的边
        if index in order:
            continue
        deps[index] = {dep for dep in deps[index] if dep in emitted}
        order.append(index)
        emitted.add(ids[index])

    position = {ids[index]: rank for rank, index in enumerate(order)}
    renamed = {node: f"Q{rank + 1}" for node, rank in position.items()}

    return tuple(
        SubQuery(
            id=renamed[ids[index]],
            query=items[index]["query"],
            purpose=items[index]["purpose"],
            depends_on=tuple(
                renamed[dep] for dep in sorted(deps[index], key=lambda d: position[d])
            ),
            target_source=items[index]["target_source"],
            english_terms=items[index]["english_terms"],
        )
        for index in order
    )


def parse_rewrite_response(
    raw: str,
    original: str,
    *,
    tokens: int = 0,
    latency: float = 0.0,
    web_search_enabled: bool = True,
) -> RewriteResult:
    """模型输出 -> RewriteResult。**任何**异常输入都必须返回一个能用的结果。

    分策略的裁剪规则不是洁癖：模型很爱多给（6 条子查询、10 个同义词、
    指向不存在 ID 的依赖），下游 prompt 是照单全收的，脏数据会直接变成
    一串无意义的工具调用。所以这里是最后一道闸。
    """
    data = _load_json(raw)
    if data is None:
        return RewriteResult(
            original=original,
            strategy=RewriteStrategy.PASS_THROUGH,
            rewritten_query=original,
            tokens=tokens,
            latency=latency,
            fallback=True,
            error="改写结果 JSON 解析失败",
        )

    reason = str(data.get("reason") or "").strip()[:80]
    rewritten = str(data.get("rewritten_query") or "").strip() or original

    raw_strategy = str(data.get("strategy") or "").strip().lower()
    try:
        strategy = RewriteStrategy(raw_strategy)
    except ValueError:
        return RewriteResult(
            original=original,
            strategy=RewriteStrategy.PASS_THROUGH,
            reason=reason,
            rewritten_query=original,
            tokens=tokens,
            latency=latency,
            fallback=True,
            error=f"未知策略：{raw_strategy or '空'}",
        )

    if strategy is RewriteStrategy.PASS_THROUGH:
        return _pass(original, reason, tokens, latency)

    if strategy is RewriteStrategy.BASIC_REWRITE:
        if _squash(rewritten) == _squash(original):
            # 改了个寂寞：不要假装做过改写，否则每轮都会多出一条"改写"卡片。
            return _pass(original, reason, tokens, latency)
        return RewriteResult(
            original=original,
            strategy=strategy,
            reason=reason,
            rewritten_query=rewritten,
            # 指代补全后的仍是中文，语料是英文时照样零命中，所以这条策略也要带英文词
            english_keywords=_clean_english_keywords(data.get("english_keywords")),
            tokens=tokens,
            latency=latency,
        )

    if strategy is RewriteStrategy.WEB_SEARCH:
        if not web_search_enabled:
            return _degrade(original, rewritten, reason, tokens, latency)
        keywords = [
            keyword
            for keyword in _dedupe(
                [str(k).strip() for k in _as_list(data.get("search_keywords"))]
            )
            if keyword
        ][:4]
        en_keywords = _clean_english_keywords(data.get("english_keywords"))
        return RewriteResult(
            original=original,
            strategy=strategy,
            reason=reason,
            rewritten_query=rewritten,
            search_keywords=tuple(keywords),
            english_keywords=tuple(en_keywords),
            tokens=tokens,
            latency=latency,
        )

    if strategy is RewriteStrategy.QUERY_EXPANSION:
        terms = [
            term
            for term in _dedupe(
                [str(t).strip() for t in _as_list(data.get("expanded_terms"))]
            )
            if term and len(term) > 1 and _squash(term) != _squash(original)
        ][:QUERY_REWRITE_MAX_EXPANSIONS]
        keywords = [
            keyword
            for keyword in _dedupe(
                [str(k).strip() for k in _as_list(data.get("search_keywords"))]
            )
            if keyword
        ][:4]
        en_keywords = _clean_english_keywords(data.get("english_keywords"))

        if not terms and not keywords and not en_keywords:
            return _degrade(original, rewritten, reason, tokens, latency)

        return RewriteResult(
            original=original,
            strategy=strategy,
            reason=reason,
            rewritten_query=rewritten,
            expanded_terms=tuple(terms),
            search_keywords=tuple(keywords),
            english_keywords=tuple(en_keywords),
            tokens=tokens,
            latency=latency,
        )

    if strategy is RewriteStrategy.MULTI_QUERY:
        # 英文词不在这里收：多路/分解的检索单元是子查询，词包必须跟着各自那一路走。
        # 顶层 english_keywords 即使模型给了也一律丢弃，否则会出现"一份词包污染多路检索"。
        items = _raw_sub_queries(
            data, QUERY_REWRITE_MAX_MULTI, web_search_enabled=web_search_enabled
        )
        if len(items) < 2:
            return _degrade(
                original,
                items[0]["query"] if items else rewritten,
                reason,
                tokens,
                latency,
            )
        return RewriteResult(
            original=original,
            strategy=strategy,
            reason=reason,
            rewritten_query=rewritten,
            sub_queries=_multi(items),
            tokens=tokens,
            latency=latency,
        )

    items = _raw_sub_queries(
        data, QUERY_REWRITE_MAX_DECOMPOSE, web_search_enabled=web_search_enabled
    )
    if len(items) < 2:
        return _degrade(
            original,
            items[0]["query"] if items else rewritten,
            reason,
            tokens,
            latency,
        )
    return RewriteResult(
        original=original,
        strategy=strategy,
        reason=reason,
        rewritten_query=rewritten,
        sub_queries=_decompose(items),
        tokens=tokens,
        latency=latency,
    )


# ---------------------------------------------------------------- 下游包装


def _sub_english_hint(sub: SubQuery) -> str:
    """子查询行下面的英文检索词注脚。没有英文词就整行不出现，不留空括号。"""
    if not sub.english_terms:
        return ""
    return "\n   ↳ 英文检索词（搜英文资料时用）：" + " / ".join(sub.english_terms)


def format_for_agent(result: RewriteResult) -> str:
    """RewriteResult -> 本轮真正发给主模型的 user 消息。

    铁律：**永远带上 `【用户原始问题】`**。改写模型再准也可能改歪，
    主模型看得到原话才有机会自己纠回来。透传时则原样返回，不套任何模板——
    否则每轮都多几十个字，等于给每句话加一段没人看的抬头。
    """
    if result.strategy is RewriteStrategy.PASS_THROUGH:
        return result.original

    head = f"【用户原始问题】\n{result.original}\n\n"

    if result.strategy is RewriteStrategy.BASIC_REWRITE:
        # 指代补全后仍是中文，语料是英文时一样零命中，所以这条策略也要把英文词交出去
        en_block = ""
        if result.english_keywords:
            en_lines = "\n".join(f"- {kw}" for kw in result.english_keywords)
            en_block = (
                "建议英文检索词组合（资料含英文文档时，可直接用于 search_document 跨语言检索）：\n"
                f"{en_lines}\n"
            )
        return head + (
            "【查询改写（基础改写）】\n"
            f"完整意图：{result.effective_query}\n"
            f"{en_block}"
            "请直接按上述完整意图调用工具并回答用户，无需复述改写过程。"
        )

    if result.strategy is RewriteStrategy.WEB_SEARCH:
        kw_block = ""
        if result.search_keywords:
            keywords = "\n".join(f"- {kw}" for kw in result.search_keywords)
            kw_block = (
                "建议联网搜索词（可直接传入 web_search 的 search_queries）：\n"
                f"{keywords}\n"
            )
        en_block = ""
        if result.english_keywords:
            en_lines = "\n".join(f"- {kw}" for kw in result.english_keywords)
            en_block = (
                "英文搜索词（需要检索英文网页时使用）：\n"
                f"{en_lines}\n"
            )
        return head + (
            "【查询改写（联网检索）】\n"
            f"搜索目标：{result.effective_query}\n"
            f"{kw_block}"
            f"{en_block}"
            "已判定本问题需要互联网外部信息，请直接优先调用 `web_search`"
            "（或针对具体链接调用 `web_fetch`），无需先在本地资料文件中试探检索。"
        )

    if result.strategy is RewriteStrategy.QUERY_EXPANSION:
        terms = ", ".join(result.expanded_terms) or "无"
        keywords = "\n".join(f"- {keyword}" for keyword in result.search_keywords) or "- 无"
        if result.english_keywords:
            en_lines = "\n".join(f"- {kw}" for kw in result.english_keywords)
            en_block = (
                "建议英文检索词组合（资料含英文文档时，可直接用于 search_document 跨语言检索）：\n"
                f"{en_lines}\n"
            )
            tail = (
                "请优先结合上述扩展术语与检索词组合查找资料或筛选数据；"
                "若资料含英文文档可优先尝试上述英文检索词组合；"
                "若精确词未命中可尝试同义词，最终综合给出准确回答。"
            )
        else:
            en_block = ""
            tail = (
                "请优先结合上述扩展术语与检索词组合查找资料或筛选数据；"
                "若精确词未命中可尝试同义词，最终综合给出准确回答。"
            )
        return head + (
            "【查询改写（查询扩展）】\n"
            f"规范问题：{result.effective_query}\n"
            f"同义/扩展术语：{terms}\n"
            "建议检索词组合（适用于 search_document / 文本筛选）：\n"
            f"{keywords}\n"
            f"{en_block}"
            f"{tail}"
        )

    if result.strategy is RewriteStrategy.MULTI_QUERY:
        lines = []
        has_web = any(sub.target_source == "web" for sub in result.sub_queries)
        for index, sub in enumerate(result.sub_queries, start=1):
            angle = sub.angle or "未标注"
            lines.append(
                f"{index}. [{sub.id} | 视角:{angle} | 数据源:{sub.target_source}] {sub.query}"
                + _sub_english_hint(sub)
            )
        web_note = (
            "\n（注：标注 `数据源:web` 的子查询请直接调用 `web_search` / `web_fetch`，无需先搜本地资料。）"
            if has_web
            else ""
        )
        return head + (
            "【查询改写（多路查询）】\n"
            f"总目标：{result.effective_query}\n"
            "请从以下互补视角分别查询/检索（各路无依赖关系，可同一步并行调用多个工具），"
            "最后汇总去重给出全面结论：\n"
            + "\n".join(lines)
            + web_note
        )

    lines = []
    has_web = any(sub.target_source == "web" for sub in result.sub_queries)
    for index, sub in enumerate(result.sub_queries, start=1):
        depends = "、".join(sub.depends_on) or "无"
        line = (
            f"{index}. [{sub.id} | 依赖:{depends} | 数据源:{sub.target_source}] {sub.query}"
        )
        if sub.purpose:
            line += f"（目标：{sub.purpose}）"
        lines.append(line + _sub_english_hint(sub))
    web_note = (
        "\n（注：标注 `数据源:web` 的子步骤请直接调用 `web_search` / `web_fetch`，无需先搜本地资料。）"
        if has_web
        else ""
    )
    return head + (
        "【查询改写（查询分解）】\n"
        f"总目标：{result.effective_query}\n"
        "请按以下依赖顺序逐步求解（无前置依赖的步骤可并行；有前置依赖的步骤须基于前序"
        "工具返回结果再执行），最终汇总完整回答：\n"
        + "\n".join(lines)
        + web_note
    )


# ---------------------------------------------------------------- 改写器


def _llm_kwargs(llm: Callable[..., Any], *, max_tokens: int, timeout: int) -> dict[str, Any]:
    """探测模型回调认不认这些参数。

    测试里写 `lambda messages, tools=None: ...` 是最省事的假模型，
    直接硬塞 temperature/timeout 会当场 TypeError。所以只在对方签名里真的有
    这个参数（或吃了 **kwargs）时才传。和 agent._takes_on_delta 是同一种思路。
    """
    try:
        parameters = inspect.signature(llm).parameters
    except (TypeError, ValueError):
        return {}
    takes_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    wanted = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "timeout": timeout,
        # 改写层不重试：它挡在主模型前面，重试两次等于首屏多卡十几秒，
        # 还不如立刻透传——用户至少能马上看到回答。
        "retries": 0,
    }
    return {
        key: value
        for key, value in wanted.items()
        if takes_kwargs or key in parameters
    }


class QueryRewriter:
    """前置改写的对外入口。一个会话一个实例，UI 层持有它以便随时开关。"""

    def __init__(
        self,
        *,
        llm: LLMCallable | None = None,
        enabled: bool = True,
        timeout: int = QUERY_REWRITE_TIMEOUT,
        max_tokens: int = QUERY_REWRITE_MAX_TOKENS,
        history_turns: int = QUERY_REWRITE_HISTORY_TURNS,
    ) -> None:
        self.llm = llm or chat
        self.enabled = enabled
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.history_turns = history_turns

    def rewrite(
        self, question: str, context: RewriteContext | None = None
    ) -> RewriteResult:
        """跑一遍改写。**不抛异常**：最差的结果是"原样透传 + 一条错误摘要"。"""
        ctx = context if context is not None else build_context([])

        bypassed = _fast_bypass(question, ctx, enabled=self.enabled)
        if bypassed is not None:
            return bypassed

        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": build_rewrite_user_prompt(question, ctx)},
        ]
        started = time.time()
        try:
            response = self.llm(
                messages,
                None,
                **_llm_kwargs(self.llm, max_tokens=self.max_tokens, timeout=self.timeout),
            )
        except Exception as exc:  # noqa: BLE001  超时/网络/协议错误一律降级
            return RewriteResult(
                original=question,
                strategy=RewriteStrategy.PASS_THROUGH,
                rewritten_query=question,
                latency=time.time() - started,
                fallback=True,
                error=str(exc)[:160] or type(exc).__name__,
            )

        latency = time.time() - started
        content = (response.message or {}).get("content") or ""
        return parse_rewrite_response(
            content,
            question,
            tokens=int(response.total_tokens or 0),
            latency=latency,
            web_search_enabled=ctx.web_search,
        )


__all__ = [
    "QueryRewriter",
    "REWRITE_SYSTEM_PROMPT",
    "RewriteContext",
    "RewriteResult",
    "RewriteStrategy",
    "RetrievalUnit",
    "STRATEGY_LABELS",
    "SubQuery",
    "build_context",
    "build_rewrite_user_prompt",
    "format_for_agent",
    "parse_rewrite_response",
    "retrieval_units",
]
