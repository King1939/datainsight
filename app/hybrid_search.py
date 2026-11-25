"""混合检索：BM25 精确词 + 向量语义，两路各自排序后用 RRF 融合。

为什么不是"取两个分数的加权平均"：BM25 分数是无界的（同一段命中三个词能到 2~5），
余弦相似度是 [-1,1]，两者相加等于让 BM25 单方面决定排序。RRF 只看**名次**——
第 1 名 1/61、第 2 名 1/62——两路天然同权，也不用给分数做归一化。
双路都进前 20 的段落拿到两份分（1/61 + 1/61），严格高于只有一路第 1，
这就是"互补命中优先"的数学表达。

**静默降级**是这里最要紧的工程约束：向量接口超时、限流、断网、没配 Key，
都不能让"搜一下资料"这个动作失败。任何异常都被吞进 `vector_degraded`，
BM25 结果照常返回，界面只多一句"向量检索暂时不可用"。
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Literal, Sequence

import numpy as np

from . import config
from .bm25 import bm25_search
from .config import (
    HYBRID_RECALL_TOP_K,
    RRF_BM25_WEIGHT,
    RRF_K,
    RRF_VECTOR_WEIGHT,
    VECTOR_MIN_SCORE,
    EmbeddingConfig,
)
from .embeddings import EmbedderCallable, VectorStore, embed_query
from .readers import Doc

# 候选卡片里那一小截的取法，和改造前保持一致
SNIPPET_CHARS = 110
SNIPPET_LEAD = 24

_HYBRID_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "datainsight.hybrid_enabled", default=config.HYBRID_SEARCH_ENABLED
)


def is_hybrid_enabled() -> bool:
    return bool(_HYBRID_ENABLED.get())


def set_hybrid_enabled(enabled: bool) -> None:
    """切开关。用 ContextVar 而不是全局变量：一个标签页关掉，不影响另一个。"""
    _HYBRID_ENABLED.set(bool(enabled))


@dataclass(frozen=True)
class VectorHit:
    doc: Doc
    doc_order: int
    part: int
    score: float            # 父段 MaxSim 余弦相似度
    best_offset: int        # 胜出子块在父段里的起始偏移
    best_subchunk: str      # 胜出子块原文


def vector_search(
    query: str,
    docs: Sequence[Doc],
    *,
    top_k: int = HYBRID_RECALL_TOP_K,
    min_score: float = VECTOR_MIN_SCORE,
    store: VectorStore | None = None,
    embedder: EmbedderCallable | None = None,
    cfg: EmbeddingConfig | None = None,
    model_key: str = "",
) -> list[VectorHit]:
    """语义召回：子块余弦相似度 -> 父段取最大值。

    父段取最大值（MaxSim）而不是平均值：一段两千字里只有三百字在讲这件事，
    平均下来会被无关背景稀释到没有；最大值才是"这段里最相关的那一句有多相关"。
    同时记住胜出子块的位置，纯语义命中时卡片就能直接截这一段，而不是盲截开头。
    """
    query = (query or "").strip()
    if not docs or not query:
        return []

    resolved_cfg = cfg or config.embedding_current()
    resolved_key = model_key or (resolved_cfg.model_key if embedder is None else "custom")
    store = store or VectorStore()

    subchunks, vec_map, _ = store.ensure_indexed(
        docs, embedder=embedder, cfg=resolved_cfg, model_key=resolved_key
    )
    if not subchunks:
        return []

    q_vec = embed_query(query, cfg=resolved_cfg, embedder=embedder, model_key=resolved_key)
    if q_vec.size == 0 or float(np.linalg.norm(q_vec)) <= 1e-12:
        return []

    rows: list[np.ndarray] = []
    kept = []
    for sc in subchunks:
        vec = vec_map.get(sc.chunk_hash)
        # 维度对齐：换过模型/维度后库里可能混着旧向量，不校验就是 shapes not aligned
        if vec is None or vec.shape != q_vec.shape:
            continue
        kept.append(sc)
        rows.append(vec)

    if not rows:
        return []

    mat = np.vstack(rows).astype(np.float32, copy=False)
    sims = mat @ q_vec.astype(np.float32, copy=False)   # 都已归一化，点积即余弦

    best: dict[tuple[int, int], VectorHit] = {}
    for sc, sim in zip(kept, sims):
        score = float(sim)
        key = (sc.doc_order, sc.part)
        current = best.get(key)
        if current is None or score > current.score:
            best[key] = VectorHit(
                doc=docs[sc.doc_order],
                doc_order=sc.doc_order,
                part=sc.part,
                score=score,
                best_offset=sc.offset,
                best_subchunk=sc.text,
            )

    hits = [h for h in best.values() if h.score >= min_score]
    hits.sort(key=lambda h: (-h.score, h.doc_order, h.part))
    return hits[: max(1, int(top_k))]


def rrf_fuse(
    ranked_lists: Sequence[tuple[Sequence[Any], float]],
    key_fn: Callable[[Any], Hashable],
    *,
    k: int = RRF_K,
) -> dict[Hashable, float]:
    """加权 RRF。`sum(w_i / (k + rank_i))`，rank 从 1 开始。

    写成通用的 N 路，是因为后面多路改写（multi_query）融合可以原样复用。
    同一条路里出现重复 key 只算第一次（保留最高名次），否则同一段会被重复加分。
    """
    scores: dict[Hashable, float] = {}
    k = max(1, int(k))
    for items, weight in ranked_lists:
        if weight <= 0:
            continue
        seen: set[Hashable] = set()
        for rank, item in enumerate(items, start=1):
            key = key_fn(item)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + (float(weight) / (k + rank))
    return scores


@dataclass(frozen=True)
class HybridCandidate:
    doc: Doc
    doc_order: int
    part: int
    total_parts: int
    rrf_score: float
    bm25_score: float = 0.0
    bm25_rank: int | None = None
    vector_score: float = 0.0
    vector_rank: int | None = None
    matched_terms: tuple[str, ...] = ()
    fuzzy: bool = False
    snippet: str = ""

    @property
    def has_exact_lexical(self) -> bool:
        return bool(self.matched_terms) and not self.fuzzy

    @property
    def sources(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.bm25_rank is not None:
            out.append("bm25")
        if self.vector_rank is not None:
            out.append("vector")
        return tuple(out)


@dataclass(frozen=True)
class HybridSearchResult:
    candidates: tuple[HybridCandidate, ...]
    total_candidates: int
    used_bm25: bool
    used_vector: bool
    fuzzy: bool = False
    vector_degraded: bool = False
    degrade_reason: str = ""
    model_key: str = ""


def _flatten(text: str) -> str:
    """压掉换行和连续空白：卡片是一行显示，带换行会把排版搞乱。"""
    return " ".join((text or "").split())


def _build_snippet(
    chunk: str, matched_terms: Sequence[str], vec_hit: VectorHit | None
) -> str:
    """生成候选卡片里那一小截。

    优先级：有词命中就以词为锚（最能说明"为什么是这一段"）；
    纯语义命中则以胜出子块为锚（那才是真正相关的那几百字，不是段首的背景）。
    """
    if matched_terms:
        low = (chunk or "").lower()
        spots = [low.find(t) for t in matched_terms if low.find(t) >= 0]
        if spots:
            start = max(0, min(spots) - SNIPPET_LEAD)
            end = min(len(chunk), start + SNIPPET_CHARS)
            body = _flatten(chunk[start:end])
            return ("…" if start > 0 else "") + body + ("…" if end < len(chunk) else "")

    if vec_hit is not None and vec_hit.best_subchunk:
        piece = vec_hit.best_subchunk
        body = _flatten(piece[:SNIPPET_CHARS])
        lead = "…" if vec_hit.best_offset > 0 else ""
        tail = "…" if len(piece) > SNIPPET_CHARS or vec_hit.best_offset + len(piece) < len(chunk) else ""
        return lead + body + tail

    return _flatten((chunk or "")[:SNIPPET_CHARS])


def hybrid_search(
    query: str,
    docs: Sequence[Doc],
    *,
    top_k: int = 3,
    recall_top_k: int = HYBRID_RECALL_TOP_K,
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid",
    rrf_k: int = RRF_K,
    bm25_weight: float = RRF_BM25_WEIGHT,
    vector_weight: float = RRF_VECTOR_WEIGHT,
    vector_min_score: float = VECTOR_MIN_SCORE,
    store: VectorStore | None = None,
    embedder: EmbedderCallable | None = None,
    cfg: EmbeddingConfig | None = None,
    model_key: str = "",
) -> HybridSearchResult:
    """两路召回 + RRF 融合，返回 Top-K 候选卡片。

    `mode="bm25"` 强制只走词匹配（对照实验、排查向量问题用）；
    `embedder` 是给单测/评测注入假模型的口子，传了就不看配置。
    """
    query = (query or "").strip()
    run_bm25 = mode in {"hybrid", "bm25"}
    resolved_cfg = cfg or config.embedding_current()
    # 开关是**总闸**：关掉就一次向量调用都不发（包括注入的 embedder）。
    # 否则"我关掉了混合检索"和"它还在偷偷花钱算向量"会同时成立，用户没法解释。
    want_vector = (mode == "vector") or (
        mode == "hybrid"
        and is_hybrid_enabled()
        and (embedder is not None or resolved_cfg.ready)
    )

    bm25_res = bm25_search(query, docs, top_k=recall_top_k) if run_bm25 else None
    bm25_hits = list(bm25_res.hits) if bm25_res else []

    vec_hits: list[VectorHit] = []
    degraded = False
    reason = ""
    if want_vector:
        try:
            vec_hits = vector_search(
                query,
                docs,
                top_k=recall_top_k,
                min_score=vector_min_score,
                store=store,
                embedder=embedder,
                cfg=resolved_cfg,
                model_key=model_key,
            )
        except Exception as exc:  # noqa: BLE001  向量这一路绝不能让检索整体失败
            degraded = True
            reason = f"{type(exc).__name__}: {exc}"[:160]
            vec_hits = []

    if not bm25_hits and not vec_hits:
        return HybridSearchResult(
            candidates=(),
            total_candidates=0,
            used_bm25=False,
            used_vector=False,
            fuzzy=bool(bm25_res and bm25_res.fuzzy),
            vector_degraded=degraded,
            degrade_reason=reason,
            model_key=model_key or resolved_cfg.model_key,
        )

    bm25_rank = {(h.doc_order, h.part): i for i, h in enumerate(bm25_hits, 1)}
    vec_rank = {(h.doc_order, h.part): i for i, h in enumerate(vec_hits, 1)}
    by_key_bm25 = {(h.doc_order, h.part): h for h in bm25_hits}
    by_key_vec = {(h.doc_order, h.part): h for h in vec_hits}

    ranked_lists: list[tuple[Sequence[Any], float]] = []
    if bm25_hits:
        ranked_lists.append((bm25_hits, bm25_weight))
    if vec_hits:
        ranked_lists.append((vec_hits, vector_weight))

    rrf_scores = rrf_fuse(
        ranked_lists, key_fn=lambda h: (h.doc_order, h.part), k=rrf_k
    )

    merged: list[HybridCandidate] = []
    for key, rrf_score in rrf_scores.items():
        bm = by_key_bm25.get(key)
        vec = by_key_vec.get(key)
        doc = (bm.doc if bm else vec.doc)
        chunk = doc.chunks[(bm or vec).part - 1]
        merged.append(
            HybridCandidate(
                doc=doc,
                doc_order=key[0],
                part=key[1],
                total_parts=doc.parts,
                rrf_score=rrf_score,
                bm25_score=bm.score if bm else 0.0,
                bm25_rank=bm25_rank.get(key),
                vector_score=vec.score if vec else 0.0,
                vector_rank=vec_rank.get(key),
                matched_terms=bm.matched_terms if bm else (),
                fuzzy=bool(bm.fuzzy) if bm else False,
                snippet=_build_snippet(chunk, bm.matched_terms if bm else (), vec),
            )
        )

    # 排序必须**完全确定**：分数相同（同一名次）时按"有精确词 > BM25 分 > 语义分
    # > 文档顺序 > 段号"依次比较。否则 Python 的稳定排序会让结果随字典插入顺序漂移，
    # 同一句话问两次给出不同候选——这在评测里是致命的。
    merged.sort(
        key=lambda c: (
            -round(c.rrf_score, 12),
            -int(c.has_exact_lexical),
            -c.bm25_score,
            -c.vector_score,
            c.doc_order,
            c.part,
        )
    )

    return HybridSearchResult(
        candidates=tuple(merged[: max(1, int(top_k))]),
        total_candidates=len(merged),
        used_bm25=bool(bm25_hits),
        used_vector=bool(vec_hits),
        fuzzy=bool(bm25_res and bm25_res.fuzzy),
        vector_degraded=degraded,
        degrade_reason=reason,
        model_key=model_key or resolved_cfg.model_key,
    )


__all__ = [
    "HybridCandidate",
    "HybridSearchResult",
    "VectorHit",
    "hybrid_search",
    "is_hybrid_enabled",
    "rrf_fuse",
    "set_hybrid_enabled",
    "vector_search",
]
