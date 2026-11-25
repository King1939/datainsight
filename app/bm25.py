"""BM25 精确词匹配：把"出现了这个词"变成"这一段最可能是要找的"。

为什么换掉原来那套启发式打分（命中词数 × 3 + 出现次数 × 0.4）：
  1. 它把"命中几个不同的词"和"某个词出现几次"混在一个线性式里，权重只能拍脑袋；
     长段因为词多天然占便宜，"到处都提了一嘴"的段落会排在"整段在讲这一件事"前面。
  2. 没有长度归一化，一段两千字的背景介绍可以靠重复把分数堆上去。
  3. 没有 IDF，满篇都是"的、是、本、节"这种字，谁出现得多谁赢。

BM25 把这三件事都处理了：词频饱和（tf 涨到头会平）、长度归一化（b）、
稀有词加权（IDF）。

**IDF 用 Lucene 变体** `ln(1 + (N - df + 0.5)/(df + 0.5))`，不是教科书里那个。
原因很实在：用户只传一份两段的文档时，N=2、df=2，原始公式算出来是负数，
命中的段落反而被扣分，排到没命中的后面。加 `1 +` 之后 IDF 恒正，区分度不丢。

分词两套并存：装了 jieba 就用 `lcut_for_search`（"区块链" 不会被切成 "区块/链"），
没装就退回纯 Python 中文 2-gram。两条路都保留**防误召回闸门**——覆盖率 + 邻近窗口。
这是真实语料逼出来的：《区块链教程》里"共识机制"一次都没出现，但"共识"3 次、
"机制"12 次散在两千字里，不加闸门就会把整段当成命中。
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Sequence

from .config import BM25_B, BM25_K1, BM25_MIN_SCORE, BM25_USE_JIEBA, HYBRID_RECALL_TOP_K
from .readers import Doc

# 一个检索词里有没有汉字：只有中文才做 bigram 降级，英文按前缀切会误伤
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
# 英文/数字/编号整词：error、502、sha-256、err-9012
_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SPLIT = re.compile(r"[\s,;，；、/|]+")

# 单字停用词。jieba 会把"的/了/在"这类字单独切出来，留着只会让所有段落都命中
_STOP_SINGLE = set(
    "的了在是和与等中为对将被由及或之其于以从到把让给向按就都也还而但如若这那个一吗呢啊吧"
)

# 模糊那一遍的闸门（和原 doc_tools 里的取值一致，是照真实语料定的）：
# 覆盖率 0.6："区块链创新应用"/"区块链的创新应用" 4/6 能过，
#              "不存在的词甲乙丙" 只撞上 1/7 必须挡掉。
# 窗口 48 字：真改写几个字挨在一起；散落的命中相隔几百字。
FUZZY_COVERAGE = 0.6
FUZZY_WINDOW = 48

# 短中文词的判定上限：再长就不是"一个词"，而是自然语言问句了
_SHORT_TERM_MAX = 10


@dataclass(frozen=True)
class BM25Hit:
    doc: Doc
    doc_order: int
    part: int                     # 1-based，直接给 read_document 用
    score: float
    matched_terms: tuple[str, ...]
    fuzzy: bool = False           # True = 整词没命中，走字面 bigram 降级


@dataclass(frozen=True)
class BM25SearchResult:
    hits: tuple[BM25Hit, ...]
    fuzzy: bool = False
    query_terms: tuple[str, ...] = ()


# ---------------------------------------------------------------- 分词


_JIEBA = None
_JIEBA_TRIED = False


def _jieba():
    """懒加载并静音。jieba 首次 import 会往 stderr 打一屏字典加载日志。"""
    global _JIEBA, _JIEBA_TRIED
    if _JIEBA_TRIED:
        return _JIEBA
    _JIEBA_TRIED = True
    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        _JIEBA = jieba
    except Exception:  # noqa: BLE001  没装、或字典损坏，都退回纯 Python 方案
        _JIEBA = None
    return _JIEBA


_TOKEN_CACHE: "OrderedDict[tuple[bool, str], tuple[Counter, int]]" = OrderedDict()
_TOKEN_CACHE_LIMIT = 4096


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def tokenize(text: str, *, use_jieba: bool = BM25_USE_JIEBA) -> list[str]:
    """中英混合分词，返回**含重复**的 token 列表（BM25 要算词频）。"""
    text = (text or "").lower()
    if not text:
        return []

    out: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0)
        out.append(word)
        # sha-256 / err-9012 这类编号：整词要留（精确匹配），拆开也要留（部分匹配）
        if any(sep in word for sep in "._-"):
            out.extend(p for p in re.split(r"[._-]+", word) if len(p) >= 2)

    jieba = _jieba() if use_jieba else None
    for match in _CJK_RUN.finditer(text):
        seg = match.group(0)
        if jieba is not None:
            pieces = [p.strip() for p in jieba.lcut_for_search(seg)]
            kept = [
                p for p in pieces if p and not (len(p) == 1 and p in _STOP_SINGLE)
            ]
            # jieba 偶尔会把整段切成全是停用词，这时候原词本身还有意义，别丢
            out.extend(kept or ([seg] if len(seg) > 1 else []))
        elif len(seg) >= 2:
            out.extend(seg[i : i + 2] for i in range(len(seg) - 1))
        elif seg not in _STOP_SINGLE:
            out.append(seg)
    return out


def analyze(text: str, *, use_jieba: bool = BM25_USE_JIEBA) -> tuple[Counter, int]:
    """分词并计数，同一段文字只算一次（按内容哈希缓存）。"""
    key = (bool(use_jieba), _cache_key(text or ""))
    cached = _TOKEN_CACHE.get(key)
    if cached is not None:
        _TOKEN_CACHE.move_to_end(key)
        return cached

    tokens = tokenize(text, use_jieba=use_jieba)
    result = (Counter(tokens), len(tokens))
    _TOKEN_CACHE[key] = result
    if len(_TOKEN_CACHE) > _TOKEN_CACHE_LIMIT:
        _TOKEN_CACHE.popitem(last=False)
    return result


# ---------------------------------------------------------------- 打分


def _idf(n_docs: int, df: int) -> float:
    """Lucene 非负 IDF。df=0 时按最大区分度处理（不会真的走到，防御性写法）。"""
    if df <= 0:
        return math.log(1 + (n_docs + 0.5) / 0.5)
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass(frozen=True)
class _Segment:
    doc: Doc
    doc_order: int
    part: int
    chunk: str
    low: str
    length: int


def _segments(docs: Sequence[Doc]) -> list[_Segment]:
    out: list[_Segment] = []
    for order, doc in enumerate(docs):
        if not (doc.text or "").strip():
            continue
        for part_index, chunk in enumerate(doc.chunks, 1):
            if not (chunk or "").strip():
                continue
            out.append(
                _Segment(
                    doc=doc,
                    doc_order=order,
                    part=part_index,
                    chunk=chunk,
                    low=chunk.lower(),
                    length=max(1, len(chunk)),
                )
            )
    return out


def _raw_terms(query: str) -> list[str]:
    """按空白/标点切检索词。整串一起匹配，不在这里分词。

    "违约金 赔偿 上限" 是三个词做 OR；"区块链创新应用" 是一个词整串匹配。
    顺序很重要：整词这一遍最可信，绝不拿分词结果去污染它。
    """
    seen: set[str] = set()
    out: list[str] = []
    for piece in _SPLIT.split(query or ""):
        term = piece.strip().lower()
        if term and term not in seen:
            seen.add(term)
            out.append(term)
    return out


def _bm25_block(
    seg: _Segment,
    counts: Counter,
    df: dict[str, int],
    n_docs: int,
    avgdl: float,
    *,
    k1: float,
    b: float,
    term_filter: Sequence[str] | None = None,
) -> tuple[float, list[str]]:
    """在已分词的 Counter 上算 Okapi BM25。"""
    matched: list[str] = []
    total = 0.0
    norm = k1 * (1 - b + b * seg.length / avgdl)
    for term, tf in counts.items():
        if term_filter is not None and term not in term_filter:
            continue
        if tf <= 0:
            continue
        matched.append(term)
        total += _idf(n_docs, df.get(term, 0)) * (tf * (k1 + 1)) / (tf + norm)
    return total, matched


def _literal_score(
    seg: _Segment,
    terms: Sequence[str],
    df: dict[str, int],
    n_docs: int,
    avgdl: float,
    *,
    k1: float,
    b: float,
) -> tuple[float, list[str]]:
    """整词子串命中。词频是子串出现次数，IDF 的 df 按"包含该子串的段数"算。"""
    matched: list[str] = []
    total = 0.0
    norm = k1 * (1 - b + b * seg.length / avgdl)
    for term in terms:
        tf = seg.low.count(term)
        if tf <= 0:
            continue
        matched.append(term)
        total += _idf(n_docs, df.get(term, 0)) * (tf * (k1 + 1)) / (tf + norm)
    return total, matched


def _positions(low: str, terms: Sequence[str]) -> list[int]:
    spots = [low.find(t) for t in terms]
    return [s for s in spots if s >= 0]


def bm25_search(
    query: str,
    docs: Sequence[Doc],
    *,
    top_k: int = HYBRID_RECALL_TOP_K,
    k1: float = BM25_K1,
    b: float = BM25_B,
    min_score: float = BM25_MIN_SCORE,
    use_jieba: bool = BM25_USE_JIEBA,
) -> BM25SearchResult:
    """两阶段检索。

    Stage 1 整词精确：错误码、专有名词、多个关键词。这一遍不等分词冷启动，
    而且结果最可信——命中就是命中。

    Stage 2 只在 Stage 1 一个都没命中时才跑，分两支：
      2A 单一短中文词（3~10 字）：拆 2-gram 打分 + 覆盖率 + 邻近窗口两道闸门；
      2B 自然语言长问句：分词后对实词打 BM25，要求至少命中 2 个词且分数够。
    降级结果带 fuzzy 标记，让上层明确告诉模型"这是字面相近，不一定是同义词"。
    """
    query = (query or "").strip()
    segs = _segments(docs)
    if not query or not segs:
        return BM25SearchResult(hits=())

    n_docs = len(segs)
    avgdl = sum(s.length for s in segs) / n_docs
    raw_terms = _raw_terms(query)

    # Stage 1：整词
    if raw_terms:
        df: dict[str, int] = {t: 0 for t in raw_terms}
        for seg in segs:
            for term in raw_terms:
                if term in seg.low:
                    df[term] += 1

        hits: list[BM25Hit] = []
        for seg in segs:
            score, matched = _literal_score(
                seg, raw_terms, df, n_docs, avgdl, k1=k1, b=b
            )
            if score > 0 and matched:
                hits.append(
                    BM25Hit(
                        doc=seg.doc,
                        doc_order=seg.doc_order,
                        part=seg.part,
                        score=score,
                        matched_terms=tuple(matched),
                    )
                )
        if hits:
            hits.sort(key=lambda h: (-h.score, h.doc_order, h.part))
            return BM25SearchResult(
                hits=tuple(hits[: max(1, int(top_k))]),
                fuzzy=False,
                query_terms=tuple(raw_terms),
            )

    # Stage 2A：单一短中文词 -> 2-gram 模糊
    short_cjk = (
        len(raw_terms) == 1
        and bool(_CJK.search(raw_terms[0]))
        and 3 <= len(raw_terms[0]) <= _SHORT_TERM_MAX
    )
    if short_cjk:
        term = raw_terms[0]
        grams = list(dict.fromkeys(term[i : i + 2] for i in range(len(term) - 1)))
        need = max(2, math.ceil(len(grams) * FUZZY_COVERAGE))

        df = {g: 0 for g in grams}
        for seg in segs:
            for gram in grams:
                if gram in seg.low:
                    df[gram] += 1

        hits = []
        for seg in segs:
            score, matched = _literal_score(
                seg, grams, df, n_docs, avgdl, k1=k1, b=b
            )
            if len(matched) < need:
                continue
            spots = _positions(seg.low, matched)
            # 光有覆盖率不够：命中的字必须挤在一起，散落各处就是假命中
            if len(spots) >= 2 and max(spots) - min(spots) > FUZZY_WINDOW:
                continue
            hits.append(
                BM25Hit(
                    doc=seg.doc,
                    doc_order=seg.doc_order,
                    part=seg.part,
                    score=score,
                    matched_terms=tuple(matched),
                    fuzzy=True,
                )
            )
        if hits:
            hits.sort(key=lambda h: (-h.score, h.doc_order, h.part))
            return BM25SearchResult(
                hits=tuple(hits[: max(1, int(top_k))]),
                fuzzy=True,
                query_terms=tuple(grams),
            )

    # Stage 2B：自然语言长问句 -> 分词后打 BM25
    #
    # 注意这一步**只在 2A 不适用时**才跑。已经按"单一短词"处理过、被闸门挡下来的
    # 查询（比如"共识机制"），不能再拿分词结果打一遍——jieba 恰好会把它切成
    # "共识"+"机制"，闸门刚挡住的两个词换个入口又进来了，等于白设防。
    q_tokens = [] if short_cjk else list(dict.fromkeys(tokenize(query, use_jieba=use_jieba)))
    if q_tokens:
        wanted = set(q_tokens)
        df = {t: 0 for t in q_tokens}
        counts_by_seg: list[tuple[_Segment, Counter]] = []
        for seg in segs:
            counts, _ = analyze(seg.chunk, use_jieba=use_jieba)
            counts_by_seg.append((seg, counts))
            for token in q_tokens:
                if counts.get(token):
                    df[token] += 1

        need = min(2, len(q_tokens))
        hits = []
        for seg, counts in counts_by_seg:
            score, matched = _bm25_block(
                seg, counts, df, n_docs, avgdl, k1=k1, b=b, term_filter=wanted
            )
            if len(matched) < need or score < min_score:
                continue
            hits.append(
                BM25Hit(
                    doc=seg.doc,
                    doc_order=seg.doc_order,
                    part=seg.part,
                    score=score,
                    matched_terms=tuple(matched),
                )
            )
        if hits:
            hits.sort(key=lambda h: (-h.score, h.doc_order, h.part))
            return BM25SearchResult(
                hits=tuple(hits[: max(1, int(top_k))]),
                fuzzy=False,
                query_terms=tuple(q_tokens),
            )

    return BM25SearchResult(hits=(), fuzzy=False, query_terms=tuple(raw_terms))


__all__ = [
    "BM25Hit",
    "BM25SearchResult",
    "bm25_search",
    "analyze",
    "tokenize",
]
