"""离线文档检索评测脚本（MRR@K / Hit@K / Recall@K）。

仅用于开发期量化评估 `app/hybrid_search.py` 在真实语料上的召回与排序准确度，
不接入 Agent 运行时链路，也不依赖任何外部评测框架。

用法：
    python eval/eval_retrieval.py
    python eval/eval_retrieval.py --modes bm25,vector,hybrid --top-k 3 --recall-k 10
    python eval/eval_retrieval.py --show-all --json-out eval/last_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, embeddings, hybrid_search, vault  # noqa: E402
from app.query_rewrite import QueryRewriter, RewriteContext, RewriteResult  # noqa: E402
from app.readers import Doc  # noqa: E402

EvalMode = Literal["bm25", "vector", "hybrid"]
VALID_MODES: tuple[EvalMode, ...] = ("bm25", "vector", "hybrid")
DEFAULT_DATASET = Path(__file__).resolve().parent / "retrieval_dataset.json"


# ---------------------------------------------------------------- 指标计算


def _normalize_target(item: Sequence[Any]) -> tuple[str, int]:
    return (str(item[0]), int(item[1]))


def first_relevant_rank(
    retrieved: Sequence[tuple[str, int]],
    relevant: set[tuple[str, int]],
    k: int | None = None,
) -> int | None:
    """返回 retrieved[:k] 中首个命中 relevant 的 1-based 名次；未命中返回 None。"""
    if not relevant:
        return None
    if k is not None:
        if k <= 0:
            return None
        window = retrieved[:k]
    else:
        window = retrieved

    for rank, candidate in enumerate(window, start=1):
        if candidate in relevant:
            return rank
    return None


def reciprocal_rank(
    retrieved: Sequence[tuple[str, int]],
    relevant: set[tuple[str, int]],
    k: int,
) -> float:
    """单条查询的倒数排名 RR@K。"""
    rank = first_relevant_rank(retrieved, relevant, k=k)
    return (1.0 / rank) if rank is not None else 0.0


def hit_at_k(
    retrieved: Sequence[tuple[str, int]],
    relevant: set[tuple[str, int]],
    k: int,
) -> float:
    """单条查询在 Top-K 内是否至少命中 1 个相关段（0.0 或 1.0）。"""
    return 1.0 if first_relevant_rank(retrieved, relevant, k=k) is not None else 0.0


def recall_at_k(
    retrieved: Sequence[tuple[str, int]],
    relevant: set[tuple[str, int]],
    k: int,
) -> float:
    """单条查询在 Top-K 内的召回率 Recall@K。"""
    if not relevant or k <= 0:
        return 0.0
    hits = set(retrieved[:k]) & relevant
    return len(hits) / len(relevant)


def _aggregate(
    records: Sequence[dict[str, Any]], *, top_k: int, recall_k: int
) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "count": 0,
            "mrr_at_k": 0.0,
            "mrr_at_recall_k": 0.0,
            "hit_at_1": 0.0,
            "hit_at_k": 0.0,
            "hit_at_recall_k": 0.0,
            "recall_at_k": 0.0,
            "recall_at_recall_k": 0.0,
            "zero_hit_count": 0,
            "empty_count": 0,
            "degraded_count": 0,
            "avg_latency_ms": 0.0,
        }

    mrr_k_sum = 0.0
    mrr_r_sum = 0.0
    hit_1_sum = 0.0
    hit_k_sum = 0.0
    hit_r_sum = 0.0
    rec_k_sum = 0.0
    rec_r_sum = 0.0
    zero_hit = 0
    empty = 0
    degraded = 0
    latency_sum = 0.0

    for rec in records:
        rel = {_normalize_target(x) for x in rec.get("relevant", ())}
        ret = [_normalize_target(x) for x in rec.get("retrieved", ())]

        mrr_k_sum += reciprocal_rank(ret, rel, top_k)
        mrr_r_sum += reciprocal_rank(ret, rel, recall_k)
        hit_1_sum += hit_at_k(ret, rel, 1)
        hit_top = hit_at_k(ret, rel, top_k)
        hit_k_sum += hit_top
        hit_r_sum += hit_at_k(ret, rel, recall_k)
        rec_k_sum += recall_at_k(ret, rel, top_k)
        rec_r_sum += recall_at_k(ret, rel, recall_k)

        if hit_top == 0.0:
            zero_hit += 1
        if not ret:
            empty += 1
        if rec.get("degraded"):
            degraded += 1
        latency_sum += float(rec.get("latency_ms") or 0.0)

    return {
        "count": count,
        "mrr_at_k": mrr_k_sum / count,
        "mrr_at_recall_k": mrr_r_sum / count,
        "hit_at_1": hit_1_sum / count,
        "hit_at_k": hit_k_sum / count,
        "hit_at_recall_k": hit_r_sum / count,
        "recall_at_k": rec_k_sum / count,
        "recall_at_recall_k": rec_r_sum / count,
        "zero_hit_count": zero_hit,
        "empty_count": empty,
        "degraded_count": degraded,
        "avg_latency_ms": latency_sum / count,
    }


def summarize_results(
    records: Sequence[dict[str, Any]], top_k: int, recall_k: int
) -> dict[str, Any]:
    """聚合总体指标与按 category 分组的指标。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        category = str(rec.get("category") or "uncategorized")
        grouped.setdefault(category, []).append(rec)

    return {
        "top_k": int(top_k),
        "recall_k": int(recall_k),
        "overall": _aggregate(records, top_k=top_k, recall_k=recall_k),
        "by_category": {
            cat: _aggregate(items, top_k=top_k, recall_k=recall_k)
            for cat, items in grouped.items()
        },
    }


# ---------------------------------------------------------------- 数据集加载与校验


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """读取并校验 JSON 评测集格式。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("queries"), list):
        raise ValueError(f"评测集格式错误，缺少 queries 列表：{path}")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw["queries"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 条评测用例不是 JSON 对象")
        qid = str(item.get("id") or "").strip()
        category = str(item.get("category") or "").strip()
        query = str(item.get("query") or "").strip()
        rel_raw = item.get("relevant")
        if not qid or not category or not query:
            raise ValueError(f"第 {idx} 条评测用例缺少 id / category / query")
        if qid in seen_ids:
            raise ValueError(f"评测集存在重复 id：{qid}")
        seen_ids.add(qid)
        if not isinstance(rel_raw, list) or not rel_raw:
            raise ValueError(f"评测用例 {qid} 的 relevant 必须是非空列表")

        relevant: list[tuple[str, int]] = []
        for pair in rel_raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"评测用例 {qid} 的 relevant 项格式错误：{pair!r}")
            doc_name = str(pair[0]).strip()
            part = int(pair[1])
            if not doc_name or part < 1:
                raise ValueError(f"评测用例 {qid} 的标注段号非法：{pair!r}")
            relevant.append((doc_name, part))

        cases.append(
            {
                "id": qid,
                "category": category,
                "query": query,
                "relevant": relevant,
                "note": str(item.get("note") or "").strip(),
            }
        )

    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


def validate_against_docs(cases: Sequence[dict[str, Any]], docs: Sequence[Doc]) -> None:
    """确认每条标注的文档名与 1-based 段号在当前文档集中真实存在。"""
    by_name = {doc.name: doc for doc in docs}
    errors: list[str] = []
    for case in cases:
        for doc_name, part in case["relevant"]:
            doc = by_name.get(doc_name)
            if doc is None:
                errors.append(f"{case['id']}: 找不到文档 `{doc_name}`")
            elif part < 1 or part > doc.parts:
                errors.append(
                    f"{case['id']}: `{doc_name}` 段号 {part} 越界（共 {doc.parts} 段）"
                )
    if errors:
        raise ValueError("评测集与当前语料不匹配：\n  - " + "\n  - ".join(errors))


def load_vault_docs() -> list[Doc]:
    """按储藏区现有顺序加载全部已索引文档。"""
    docs: list[Doc] = []
    for entry in vault.documents():
        doc = vault.open_document(entry)
        if doc is not None:
            docs.append(doc)
    return docs


def filter_cases_by_category(
    cases: Sequence[dict[str, Any]], categories: Sequence[str] | None
) -> list[dict[str, Any]]:
    """按类别过滤评测用例；指定了不存在的类别时直接报错。"""
    wanted = [str(c).strip() for c in (categories or ()) if str(c).strip()]
    if not wanted:
        return list(cases)

    available = {str(c.get("category") or "") for c in cases}
    unknown = [c for c in wanted if c not in available]
    if unknown:
        raise ValueError(
            f"未知评测类别：{', '.join(unknown)}（可选：{', '.join(sorted(available))}）"
        )
    wanted_set = set(wanted)
    return [c for c in cases if c.get("category") in wanted_set]


def select_rewrite_query(rw_result: RewriteResult) -> str:
    """从改写结果中选出用于检索的单条查询词（优先英文检索词）。"""
    if rw_result.english_keywords:
        return rw_result.english_keywords[0]
    if rw_result.search_keywords:
        return rw_result.search_keywords[0]
    return rw_result.effective_query


def build_eval_rewrite_context(docs: Sequence[Doc]) -> RewriteContext:
    """用语料摘要构建改写上下文（只含文件名/类型/字数/段数，不含正文）。"""
    digests = tuple(
        {"name": doc.name, "kind": doc.kind, "chars": doc.chars, "parts": doc.parts}
        for doc in docs
    )
    return RewriteContext(documents=digests)


def rewrite_cases(
    cases: Sequence[dict[str, Any]],
    docs: Sequence[Doc],
    *,
    rewriter: QueryRewriter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """调用前置查询改写生成用于检索的新查询集与明细记录。"""
    rw = rewriter or QueryRewriter()
    ctx = build_eval_rewrite_context(docs)
    rewritten_cases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for case in cases:
        res = rw.rewrite(case["query"], ctx)
        selected = select_rewrite_query(res)
        updated = dict(case)
        updated["query"] = selected
        rewritten_cases.append(updated)
        traces.append(
            {
                "id": case["id"],
                "category": case["category"],
                "original_query": case["query"],
                "strategy": res.strategy.value,
                "rewritten_query": res.effective_query,
                "search_keywords": list(res.search_keywords),
                "english_keywords": list(res.english_keywords),
                "selected_query": selected,
                "tokens": res.tokens,
                "latency": res.latency,
                "fallback": res.fallback,
                "error": res.error,
            }
        )
    return rewritten_cases, traces


# ---------------------------------------------------------------- 执行评测


def evaluate_mode(
    cases: Sequence[dict[str, Any]],
    docs: Sequence[Doc],
    *,
    mode: EvalMode,
    top_k: int = 3,
    recall_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """在指定模式下逐条运行检索并计算指标。"""
    hybrid_search.set_hybrid_enabled(True)
    fetch_k = max(1, int(top_k), int(recall_k))
    pool_k = max(int(config.HYBRID_RECALL_TOP_K), fetch_k)

    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        result = hybrid_search.hybrid_search(
            case["query"],
            docs,
            mode=mode,
            top_k=fetch_k,
            recall_top_k=pool_k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        retrieved = [(c.doc.name, c.part) for c in result.candidates]
        relevant = set(case["relevant"])
        rank_top = first_relevant_rank(retrieved, relevant, k=top_k)
        rank_recall = first_relevant_rank(retrieved, relevant, k=recall_k)

        records.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "relevant": [list(pair) for pair in case["relevant"]],
                "retrieved": [list(pair) for pair in retrieved],
                "sources": [list(c.sources) for c in result.candidates],
                "rank_in_top_k": rank_top,
                "rank_in_recall_k": rank_recall,
                "rr_at_k": reciprocal_rank(retrieved, relevant, top_k),
                "rr_at_recall_k": reciprocal_rank(retrieved, relevant, recall_k),
                "degraded": bool(result.vector_degraded),
                "degrade_reason": result.degrade_reason,
                "latency_ms": elapsed_ms,
            }
        )

    summary = summarize_results(records, top_k=top_k, recall_k=recall_k)
    return records, summary


# ---------------------------------------------------------------- 报告渲染


def _fmt_pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _fmt_targets(pairs: Sequence[Sequence[Any]]) -> str:
    return ", ".join(f"{name}#{part}" for name, part in pairs)


def format_report(
    *,
    dataset_path: Path,
    docs: Sequence[Doc],
    modes: Sequence[EvalMode],
    results_by_mode: dict[EvalMode, tuple[list[dict[str, Any]], dict[str, Any]]],
    top_k: int,
    recall_k: int,
    show_all: bool = False,
) -> str:
    """格式化终端评测报告（绝不输出任何 API Key）。"""
    total_parts = sum(d.parts for d in docs)
    total_chars = sum(d.chars for d in docs)
    emb_cfg = config.embedding_current()
    store_stats = (
        embeddings.VectorStore().stats(emb_cfg.model_key) if emb_cfg.ready else {"count": 0}
    )

    lines: list[str] = [
        "=" * 78,
        "DataInsight 文档检索离线评测报告（MRR / Hit / Recall）",
        "=" * 78,
        f"评测集   : {dataset_path}（{len(next(iter(results_by_mode.values()))[0])} 条）",
        f"语料规模 : {len(docs)} 份文档 / {total_parts} 段 / {total_chars:,} 字",
        (
            f"向量配置 : {emb_cfg.provider}:{emb_cfg.model} "
            f"(dims={emb_cfg.dimensions or 'default'}, ready={emb_cfg.ready}, "
            f"cached_subchunks={store_stats.get('count', 0)})"
        ),
        (
            f"融合参数 : top_k={top_k}, recall_k={recall_k}, "
            f"recall_pool={config.HYBRID_RECALL_TOP_K}, rrf_k={config.RRF_K}, "
            f"weights=({config.RRF_BM25_WEIGHT}, {config.RRF_VECTOR_WEIGHT}), "
            f"vec_min_score={config.VECTOR_MIN_SCORE}"
        ),
        "",
        "## 1. 总体指标对比",
        (
            f"{'Mode':<8} | {'MRR@'+str(top_k):>7} | {'Hit@1':>6} | "
            f"{'Hit@'+str(top_k):>6} | {'Recall@'+str(top_k):>8} | "
            f"{'MRR@'+str(recall_k):>7} | {'Hit@'+str(recall_k):>7} | "
            f"{'Zero-Hit':>8} | {'Empty':>5} | {'Avg(ms)':>8}"
        ),
        "-" * 95,
    ]

    for mode in modes:
        _, summary = results_by_mode[mode]
        ov = summary["overall"]
        lines.append(
            f"{mode:<8} | {ov['mrr_at_k']:>7.4f} | {_fmt_pct(ov['hit_at_1']):>6} | "
            f"{_fmt_pct(ov['hit_at_k']):>6} | {_fmt_pct(ov['recall_at_k']):>8} | "
            f"{ov['mrr_at_recall_k']:>7.4f} | {_fmt_pct(ov['hit_at_recall_k']):>7} | "
            f"{ov['zero_hit_count']:>8d} | {ov['empty_count']:>5d} | {ov['avg_latency_ms']:>8.1f}"
        )
        if ov["degraded_count"]:
            lines.append(
                f"  [警告] {mode} 有 {ov['degraded_count']}/{ov['count']} 条查询发生向量降级！"
            )

    lines.extend(
        [
            "",
            f"说明：Zero-Hit 为 Top-{top_k} 内未命中标准答案的查询数；Empty 为 0 候选返回的查询数。",
            "      若按顺序依次跑 vector 与 hybrid，hybrid 会复用进程内 query 向量 LRU 缓存，因此耗时显著更低。",
            "",
            "## 2. 分类别对比（MRR@%d / Hit@%d）" % (top_k, top_k),
        ]
    )

    first_summary = results_by_mode[modes[0]][1]
    categories = list(first_summary["by_category"].keys())
    header = f"{'Category':<14} | {'N':>2}"
    for mode in modes:
        header += f" | {mode + ' MRR':>10} | {mode + ' Hit':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for cat in categories:
        n_cat = first_summary["by_category"][cat]["count"]
        row = f"{cat:<14} | {n_cat:>2d}"
        for mode in modes:
            cat_stats = results_by_mode[mode][1]["by_category"][cat]
            row += (
                f" | {cat_stats['mrr_at_k']:>10.4f}"
                f" | {_fmt_pct(cat_stats['hit_at_k']):>10}"
            )
        lines.append(row)

    lines.append("")
    lines.append(f"## 3. 检索细节与 Bad Case 诊断（Top-{top_k}）")
    for mode in modes:
        records, _ = results_by_mode[mode]
        interesting = [
            rec
            for rec in records
            if show_all or rec["rank_in_top_k"] is None or rec["rank_in_top_k"] > 1
        ]
        lines.append(f"\n### [{mode}] 需关注用例（{len(interesting)}/{len(records)}）")
        if not interesting:
            lines.append("  全部用例均以 #1 命中。")
            continue
        for rec in interesting:
            rank_k = rec["rank_in_top_k"]
            rank_r = rec["rank_in_recall_k"]
            if rank_k == 1:
                status = "HIT#1"
            elif rank_k is not None:
                status = f"HIT#{rank_k}"
            elif rank_r is not None:
                status = f"MISS@{top_k} (Top-{recall_k} 排 #{rank_r})"
            else:
                status = f"MISS@{recall_k}"
            if rec["degraded"]:
                status += f" [DEGRADED: {rec['degrade_reason']}]"
            top_shown = _fmt_targets(rec["retrieved"][:top_k]) or "(无候选)"
            lines.append(
                f"  - [{rec['id']}] ({rec['category']}) {status} | Q: {rec['query']}"
            )
            lines.append(f"      标准答案 : {_fmt_targets(rec['relevant'])}")
            lines.append(f"      实际Top-{top_k}: {top_shown}")

    return "\n".join(lines)


def _parse_modes(raw: str) -> list[EvalMode]:
    modes: list[EvalMode] = []
    for part in (raw or "").split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in VALID_MODES:
            raise ValueError(f"不支持的模式 `{mode}`，可选：{', '.join(VALID_MODES)}")
        typed: EvalMode = mode  # type: ignore[assignment]
        if typed not in modes:
            modes.append(typed)
    if not modes:
        raise ValueError("至少需要指定一个评测模式")
    return modes


def _fmt_rank(rec: dict[str, Any], top_k: int, recall_k: int) -> str:
    rank_k = rec["rank_in_top_k"]
    rank_r = rec["rank_in_recall_k"]
    if rank_k == 1:
        return "HIT#1"
    if rank_k is not None:
        return f"HIT#{rank_k}"
    if rank_r is not None:
        return f"MISS@{top_k}(#{rank_r})"
    return f"MISS@{recall_k}"


def format_compare_report(
    *,
    dataset_path: Path,
    docs: Sequence[Doc],
    modes: Sequence[EvalMode],
    before_by_mode: dict[EvalMode, tuple[list[dict[str, Any]], dict[str, Any]]],
    after_by_mode: dict[EvalMode, tuple[list[dict[str, Any]], dict[str, Any]]],
    traces: Sequence[dict[str, Any]],
    top_k: int,
    recall_k: int,
) -> str:
    """格式化前置查询改写改造前后对比报告。"""
    total_parts = sum(d.parts for d in docs)
    total_chars = sum(d.chars for d in docs)
    emb_cfg = config.embedding_current()
    store_stats = (
        embeddings.VectorStore().stats(emb_cfg.model_key) if emb_cfg.ready else {"count": 0}
    )

    lines: list[str] = [
        "=" * 95,
        "DataInsight 跨语言检索改造前后对比报告（Query Expansion 英文检索词）",
        "=" * 95,
        f"评测集   : {dataset_path}（筛选 {len(traces)} 条）",
        f"语料规模 : {len(docs)} 份文档 / {total_parts} 段 / {total_chars:,} 字",
        (
            f"向量配置 : {emb_cfg.provider}:{emb_cfg.model} "
            f"(dims={emb_cfg.dimensions or 'default'}, ready={emb_cfg.ready}, "
            f"cached_subchunks={store_stats.get('count', 0)})"
        ),
        "",
        "## 1. 前置改写明细",
    ]

    fallbacks = [t for t in traces if t["fallback"]]
    if fallbacks:
        for t in fallbacks:
            lines.append(
                f"  [警告] {t['id']} 改写降级（fallback=True）：{t['error'] or '未知错误'}"
            )
    for t in traces:
        en_str = ", ".join(t["english_keywords"]) or "(无)"
        lines.append(
            f"  - [{t['id']}] strategy={t['strategy']} | tokens={t['tokens']} | {t['latency']:.2f}s"
        )
        lines.append(f"      原问题    : {t['original_query']}")
        lines.append(f"      英文候选  : {en_str}")
        lines.append(f"      实际检索词: {t['selected_query']}")

    lines.extend(
        [
            "",
            "## 2. 改造前后指标对比",
            (
                f"{'Mode':<8} | {'Stage':<6} | {'MRR@'+str(top_k):>7} | {'Hit@1':>6} | "
                f"{'Hit@'+str(top_k):>6} | {'Recall@'+str(top_k):>8} | "
                f"{'MRR@'+str(recall_k):>7} | {'Hit@'+str(recall_k):>7} | "
                f"{'Zero-Hit':>8} | {'Empty':>5}"
            ),
            "-" * 95,
        ]
    )

    for mode in modes:
        b_ov = before_by_mode[mode][1]["overall"]
        a_ov = after_by_mode[mode][1]["overall"]
        for stage, ov in (("Before", b_ov), ("After", a_ov)):
            lines.append(
                f"{mode:<8} | {stage:<6} | {ov['mrr_at_k']:>7.4f} | {_fmt_pct(ov['hit_at_1']):>6} | "
                f"{_fmt_pct(ov['hit_at_k']):>6} | {_fmt_pct(ov['recall_at_k']):>8} | "
                f"{ov['mrr_at_recall_k']:>7.4f} | {_fmt_pct(ov['hit_at_recall_k']):>7} | "
                f"{ov['zero_hit_count']:>8d} | {ov['empty_count']:>5d}"
            )
        lines.append(
            f"{mode:<8} | {'Delta':<6} | "
            f"{a_ov['mrr_at_k'] - b_ov['mrr_at_k']:>+7.4f} | "
            f"{(a_ov['hit_at_1'] - b_ov['hit_at_1']) * 100:>+5.1f}% | "
            f"{(a_ov['hit_at_k'] - b_ov['hit_at_k']) * 100:>+5.1f}% | "
            f"{(a_ov['recall_at_k'] - b_ov['recall_at_k']) * 100:>+7.1f}% | "
            f"{a_ov['mrr_at_recall_k'] - b_ov['mrr_at_recall_k']:>+7.4f} | "
            f"{(a_ov['hit_at_recall_k'] - b_ov['hit_at_recall_k']) * 100:>+6.1f}% | "
            f"{a_ov['zero_hit_count'] - b_ov['zero_hit_count']:>+8d} | "
            f"{a_ov['empty_count'] - b_ov['empty_count']:>+5d}"
        )
        lines.append("-" * 95)

    compare_mode: EvalMode = "hybrid" if "hybrid" in modes else modes[-1]
    lines.append("")
    lines.append(f"## 3. 逐题排名变化（{compare_mode} 模式）")
    b_recs = before_by_mode[compare_mode][0]
    a_recs = after_by_mode[compare_mode][0]
    for b_rec, a_rec, trace in zip(b_recs, a_recs, traces):
        b_status = _fmt_rank(b_rec, top_k, recall_k)
        a_status = _fmt_rank(a_rec, top_k, recall_k)
        b_top = _fmt_targets(b_rec["retrieved"][:top_k]) or "(无候选)"
        a_top = _fmt_targets(a_rec["retrieved"][:top_k]) or "(无候选)"
        lines.append(
            f"  - [{trace['id']}] {b_status} -> {a_status} | GT: {_fmt_targets(b_rec['relevant'])}"
        )
        lines.append(f"      改造前({trace['original_query'][:28]}…): {b_top}")
        lines.append(f"      改造后({trace['selected_query']}): {a_top}")

    return "\n".join(lines)


def _write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已导出 JSON 报告：{path}")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="DataInsight 文档检索 MRR 离线评测")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="评测集 JSON 路径"
    )
    parser.add_argument(
        "--modes",
        default="bm25,vector,hybrid",
        help="逗号分隔的检索模式（bm25,vector,hybrid）",
    )
    parser.add_argument(
        "--category",
        default="",
        help="可选：逗号分隔的评测类别过滤（如 cross_lingual）",
    )
    parser.add_argument(
        "--compare-rewrite",
        action="store_true",
        help="对比原始提问与前置查询改写（query_expansion）后的检索表现",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="主截断 K（默认 3，对应 search_document 的 MAX_CANDIDATES）",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=10,
        help="诊断扩展截断 K（默认 10，观察是否在 Top-4~10 内）",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="打印全部查询明细，而不只打印未命中或非第 1 名的用例",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="可选：将完整评测结果保存为 JSON 文件",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    modes = _parse_modes(args.modes)
    all_cases = load_dataset(args.dataset)
    categories = [c.strip() for c in args.category.split(",") if c.strip()]
    cases = filter_cases_by_category(all_cases, categories)

    docs = load_vault_docs()
    if not docs:
        raise RuntimeError("data/vault 中没有可评测的文档。")
    validate_against_docs(cases, docs)

    if args.compare_rewrite:
        rewritten_cases, traces = rewrite_cases(cases, docs)
        before_by_mode: dict[EvalMode, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        after_by_mode: dict[EvalMode, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        for mode in modes:
            before_by_mode[mode] = evaluate_mode(
                cases, docs, mode=mode, top_k=args.top_k, recall_k=args.recall_k
            )
            after_by_mode[mode] = evaluate_mode(
                rewritten_cases, docs, mode=mode, top_k=args.top_k, recall_k=args.recall_k
            )
        report = format_compare_report(
            dataset_path=args.dataset,
            docs=docs,
            modes=modes,
            before_by_mode=before_by_mode,
            after_by_mode=after_by_mode,
            traces=traces,
            top_k=args.top_k,
            recall_k=args.recall_k,
        )
        print(report)
        if args.json_out is not None:
            _write_json_report(
                args.json_out,
                {
                    "dataset": str(args.dataset),
                    "categories": categories,
                    "top_k": args.top_k,
                    "recall_k": args.recall_k,
                    "traces": traces,
                    "before": {
                        mode: {"summary": summary, "records": records}
                        for mode, (records, summary) in before_by_mode.items()
                    },
                    "after": {
                        mode: {"summary": summary, "records": records}
                        for mode, (records, summary) in after_by_mode.items()
                    },
                },
            )
        return 0

    results_by_mode: dict[EvalMode, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for mode in modes:
        results_by_mode[mode] = evaluate_mode(
            cases, docs, mode=mode, top_k=args.top_k, recall_k=args.recall_k
        )

    report = format_report(
        dataset_path=args.dataset,
        docs=docs,
        modes=modes,
        results_by_mode=results_by_mode,
        top_k=args.top_k,
        recall_k=args.recall_k,
        show_all=args.show_all,
    )
    print(report)

    if args.json_out is not None:
        _write_json_report(
            args.json_out,
            {
                "dataset": str(args.dataset),
                "categories": categories,
                "top_k": args.top_k,
                "recall_k": args.recall_k,
                "modes": {
                    mode: {"summary": summary, "records": records}
                    for mode, (records, summary) in results_by_mode.items()
                },
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
