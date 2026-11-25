"""DECOMPOSITION 策略下的跨语言召回评测。

只回答一个问题：**英文检索词放在哪一层**，对英文语料的命中率差多少。

四组对照共用同一批改写结果，只换检索输入，所以差异可以全部归因到英文词的作用域：

    A baseline     只用中文子查询
    B global-pack  中文子查询 + 所有子查询英文词拼成的一份全局词包（模拟"结果级"做法）
    C per-unit     中文子查询 + 该条子查询自己的英文词（当前实现）
    D english-only 只用英文词，不搜中文（上限参考）

多路候选的合并方式与线上一致：RRF（只看名次，权重相同）。

用法：
    python eval/eval_decompose.py
    python eval/eval_decompose.py --top-k 5 --json-out eval/last_decompose.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
for _p in (ROOT, EVAL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app import config, hybrid_search, vault  # noqa: E402
from app.query_rewrite import (  # noqa: E402
    QueryRewriter,
    RewriteContext,
    RewriteStrategy,
    RetrievalUnit,
    retrieval_units,
)
from app.readers import Doc  # noqa: E402

# 指标与数据集加载直接复用检索评测那一套，避免两处各算一套、数字对不上
from eval_retrieval import (  # noqa: E402
    first_relevant_rank,
    load_dataset,
    load_vault_docs,
    recall_at_k,
    reciprocal_rank,
    validate_against_docs,
)

DEFAULT_DATASET = EVAL_DIR / "retrieval_dataset_decompose.json"

CONFIGS: tuple[tuple[str, str], ...] = (
    ("A_baseline", "只搜中文子查询"),
    ("B_global_pack", "中文 + 全局英文词包"),
    ("C_per_unit", "中文 + 各子查询英文词"),
    ("D_english_only", "只搜英文词"),
)


# ---------------------------------------------------------------- 检索与融合


def _rrf_merge(channel_hits: Sequence[Sequence[tuple[str, int]]], k: int) -> list[tuple[str, int]]:
    """把多路候选按 RRF 融成一条排名。

    与 hybrid_search 同一套公式：score = Σ 1/(k + rank)，只看名次不看分数。
    候选之间可能重复出现（多路都命中同一段），分数相加正是"互补命中优先"。
    """
    scores: dict[tuple[str, int], float] = {}
    for hits in channel_hits:
        for rank, key in enumerate(hits, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


def _channel_queries(config_name: str, unit: RetrievalUnit, global_pack: str) -> list[str]:
    """某一组对照下，这个检索单元要发哪几路查询。"""
    queries = [unit.query]
    if config_name in {"B_global_pack", "C_per_unit", "D_english_only"}:
        english = global_pack if config_name == "B_global_pack" else " ".join(unit.english_terms)
        if config_name == "D_english_only":
            return [english] if english else []
        if english:
            queries.append(english)
    return queries


def _retrieve(
    query: str,
    docs: Sequence[Doc],
    *,
    fetch_k: int,
    pool_k: int,
) -> tuple[list[tuple[str, int]], bool]:
    """跑一路混合检索，返回 (候选列表, 是否向量降级)。"""
    result = hybrid_search.hybrid_search(
        query, docs, mode="hybrid", top_k=fetch_k, recall_top_k=pool_k
    )
    hits = [(c.doc.name, c.part) for c in result.candidates]
    return hits, bool(result.vector_degraded)


# ---------------------------------------------------------------- 指标汇总


def _metrics(
    ranked: Sequence[tuple[str, int]],
    relevant: set[tuple[str, int]],
    *,
    top_k: int,
    recall_k: int,
) -> dict[str, Any]:
    rank_top = first_relevant_rank(ranked, relevant, k=top_k)
    return {
        "hit_at_k": 1.0 if rank_top is not None else 0.0,
        "mrr_at_k": reciprocal_rank(ranked, relevant, top_k),
        "recall_at_k": recall_at_k(ranked, relevant, top_k),
        "rank_in_top_k": rank_top,
        "rank_in_recall_k": first_relevant_rank(ranked, relevant, k=recall_k),
    }


def _aggregate(records: Sequence[dict[str, Any]], config_name: str) -> dict[str, Any]:
    rows = [r["configs"][config_name] for r in records if r.get("evaluated")]
    if not rows:
        return {
            "count": 0,
            "hit_at_k": 0.0,
            "mrr_at_k": 0.0,
            "recall_at_k": 0.0,
            "any_hit_at_k": 0.0,
            "zero_hit": 0,
        }
    count = len(rows)
    return {
        "count": count,
        "hit_at_k": sum(r["hit_at_k"] for r in rows) / count,
        "mrr_at_k": sum(r["mrr_at_k"] for r in rows) / count,
        "recall_at_k": sum(r["recall_at_k"] for r in rows) / count,
        "any_hit_at_k": sum(r["any_hit_at_k"] for r in rows) / count,
        "zero_hit": sum(1 for r in rows if r["hit_at_k"] == 0.0),
    }


# ---------------------------------------------------------------- 主流程


def evaluate(
    cases: Sequence[dict[str, Any]],
    docs: Sequence[Doc],
    *,
    top_k: int,
    recall_k: int,
    rewriter: QueryRewriter | None = None,
) -> list[dict[str, Any]]:
    rw = rewriter or QueryRewriter()
    ctx = RewriteContext(
        documents=tuple(
            {"name": d.name, "kind": d.kind, "chars": d.chars, "parts": d.parts} for d in docs
        )
    )
    fetch_k = max(top_k, recall_k)
    pool_k = max(config.HYBRID_RECALL_TOP_K, fetch_k)

    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        result = rw.rewrite(case["query"], ctx)
        units = retrieval_units(result)
        # 只检索文档类单元：table 类子查询不该进文档召回评测
        units = tuple(u for u in units if u.target_source != "table")
        global_pack = " ".join(
            term for unit in units for term in unit.english_terms
        )
        relevant = {(str(name), int(part)) for name, part in case["relevant"]}

        record: dict[str, Any] = {
            "id": case["id"],
            "query": case["query"],
            "strategy": result.strategy.value,
            "reason": result.reason,
            "units": [
                {"id": u.id, "query": u.query, "english_terms": list(u.english_terms)}
                for u in units
            ],
            "global_pack": global_pack,
            "relevant": [list(pair) for pair in case["relevant"]],
            "evaluated": result.strategy is RewriteStrategy.DECOMPOSITION and bool(units),
            "configs": {},
        }

        for config_name, _label in CONFIGS:
            channels: list[list[tuple[str, int]]] = []
            degraded = False
            for unit in units:
                for query in _channel_queries(config_name, unit, global_pack):
                    hits, deg = _retrieve(query, docs, fetch_k=fetch_k, pool_k=pool_k)
                    degraded = degraded or deg
                    if hits:
                        channels.append(hits)
            ranked = _rrf_merge(channels, config.RRF_K)
            # 「任一路命中」：不看融合结果，只看这几路里有没有一路把答案送进了 Top-K。
            # 它把"英文词有没有用"和"融合有没有把有用的一路压下去"分开来看，
            # 否则 RRF 的并列名次一搅，就分不清是词不好还是融合作怪。
            any_rank = min(
                (
                    rank
                    for rank in (
                        first_relevant_rank(hits, relevant, k=top_k) for hits in channels
                    )
                    if rank is not None
                ),
                default=None,
            )
            record["configs"][config_name] = {
                **_metrics(ranked, relevant, top_k=top_k, recall_k=recall_k),
                "any_hit_at_k": 1.0 if any_rank is not None else 0.0,
                "any_rank_in_top_k": any_rank,
                "degraded": degraded,
                "top": [list(pair) for pair in ranked[:top_k]],
            }

        record["latency_ms"] = (time.perf_counter() - started) * 1000.0
        records.append(record)
    return records


def format_report(
    records: Sequence[dict[str, Any]],
    *,
    dataset_path: Path,
    docs: Sequence[Doc],
    top_k: int,
    recall_k: int,
) -> str:
    skipped = [r for r in records if not r["evaluated"]]
    with_terms = sum(1 for r in records if any(u["english_terms"] for u in r["units"]))
    unit_count = [len(r["units"]) for r in records if r["evaluated"]]
    avg_units = f"{sum(unit_count) / len(unit_count):.1f}" if unit_count else "0"

    lines = [
        "=" * 88,
        "DECOMPOSITION 策略 + 英文语料召回评测（英文检索词作用域对照）",
        "=" * 88,
        f"评测集   : {dataset_path}（{len(records)} 条）",
        f"语料规模 : {len(docs)} 份文档 / {sum(d.parts for d in docs)} 段",
        f"检索参数 : top_k={top_k}, recall_k={recall_k}, "
        f"recall_pool={config.HYBRID_RECALL_TOP_K}, rrf_k={config.RRF_K}",
        f"策略校验 : {len(records) - len(skipped)}/{len(records)} 条真的路由到 decomposition；"
        f"{with_terms}/{len(records)} 条产出了英文词；平均子查询数 {avg_units}",
        "",
        f"## 1. 四组对照（Top-{top_k}）",
        f"{'Config':<16} | {'含义':<22} | {'Hit@'+str(top_k):>7} | "
        f"{'MRR@'+str(top_k):>7} | {'Recall@'+str(top_k):>9} | {'任一路Hit':>9} | {'Zero-Hit':>8}",
        "-" * 95,
    ]
    for config_name, label in CONFIGS:
        agg = _aggregate(records, config_name)
        lines.append(
            f"{config_name:<16} | {label:<22} | {agg['hit_at_k'] * 100:>6.1f}% | "
            f"{agg['mrr_at_k']:>7.4f} | {agg['recall_at_k'] * 100:>8.1f}% | "
            f"{agg['any_hit_at_k'] * 100:>8.1f}% | {agg['zero_hit']:>8d}"
        )

    if skipped:
        lines.extend(["", "## 2. 策略校验未通过（不进汇总）"])
        for rec in skipped:
            lines.append(
                f"  - [{rec['id']}] 实际策略={rec['strategy']}（{rec['reason']}）｜Q: {rec['query']}"
            )

    lines.extend(["", f"## 3. 逐题命中对比（Top-{top_k} 名次，MISS 表示未进 Top-{recall_k}）"])
    for rec in records:
        if not rec["evaluated"]:
            continue
        cells = []
        for config_name, _label in CONFIGS:
            cfg = rec["configs"][config_name]
            rank = cfg["rank_in_top_k"]
            if rank is not None:
                cells.append(f"#{rank}")
            elif cfg["any_rank_in_top_k"] is not None:
                cells.append(f"融合MISS(路内#{cfg['any_rank_in_top_k']})")
            else:
                cells.append("MISS")
        lines.append(f"  - [{rec['id']}] {' / '.join(cells)} | GT: {', '.join(f'{n}#{p}' for n, p in rec['relevant'])}")
        lines.append(f"      {rec['query']}")
        for unit in rec["units"]:
            en = " / ".join(unit["english_terms"]) or "（无英文词）"
            lines.append(f"      [{unit['id']}] {unit['query'][:38]}… → {en[:70]}")
    return "\n".join(line for line in lines if line is not None)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="DECOMPOSITION 英文召回对照评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    cases = load_dataset(args.dataset)
    docs = load_vault_docs()
    if not docs:
        raise RuntimeError("data/vault 中没有可评测的文档。")
    validate_against_docs(cases, docs)

    hybrid_search.set_hybrid_enabled(True)
    records = evaluate(cases, docs, top_k=args.top_k, recall_k=args.recall_k)
    print(
        format_report(
            records, dataset_path=args.dataset, docs=docs, top_k=args.top_k, recall_k=args.recall_k
        )
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "top_k": args.top_k,
                    "recall_k": args.recall_k,
                    "summary": {name: _aggregate(records, name) for name, _ in CONFIGS},
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n已导出 JSON 报告：{args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
