"""端到端生成质量评测 CLI：驱动真实 Agent 生成答案，再打忠实度与相关性分。

纯开发期离线脚本。`app/` 下**没有任何文件**为它改动，Streamlit 界面也不认识它。

它做的事分三段，每段都可以单独跑：

  1. 生成（默认）：为每条用例开一个隔离会话，挂载 `data/vault` 语料和/或
     `data/sales.csv`，跑 `Agent.ask()`，把答案 + 本轮工具上下文 + 改写事件
     原样存下来。用 `--save-predictions` 落盘后就再也不用重复烧 token。
  2. 评测：`AnswerEvaluator` 判忠实度与相关性。可以只读快照跑（`--predictions`），
     所以调 Judge 模型、改阈值都不用重跑 Agent。
  3. 报告：终端表格 + `--json-out` + `--md-out`。绝对不会打印任何 API Key。

策略矩阵（`--matrix`）是一次把四种配置跑在同一批用例上，用来回答
"该不该开前置改写、口语化和英文提问会不会掉分"这种问题：

    baseline        原问题，关闭改写（对照组）
    basic_rewrite   原问题，开启改写（改写器自己决定透传/基础改写/扩展）
    colloquial      口语化问法，开启改写
    cross_lingual   英文问法，开启改写

四条硬边界：
  - 每条用例一个 `eval-gen-<id>-<uuid>` 会话 key，用完删干净，不碰用户真实历史；
  - 生成阶段任何异常都记进该条记录的 `error`，不让整批中断；
  - Judge 报错只让那一条变 `fallback`；
  - 终端与导出文件里只出现模型名，不出现 Key。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, hybrid_search, vault  # noqa: E402
from app.agent import Agent, make_system_prompt  # noqa: E402
from app.query_rewrite import QueryRewriter  # noqa: E402
from app.readers import Doc  # noqa: E402
from app.session import Session  # noqa: E402
from app.tools import data_tools  # noqa: E402
from eval.answer_metrics import (  # noqa: E402
    AnswerEvaluator,
    extract_turn_contexts,
    summarize_generation_results,
)

DEFAULT_DATASET = Path(__file__).resolve().parent / "generation_dataset.json"

VARIANTS: tuple[str, ...] = ("original", "colloquial", "cross_lingual")
SOURCE_MODES: tuple[str, ...] = ("vault", "sales", "both")

# (配置名, 问法, 是否启用前置改写)
MATRIX_CONFIGS: tuple[tuple[str, str, bool], ...] = (
    ("baseline", "original", False),
    ("basic_rewrite", "original", True),
    ("colloquial", "colloquial", True),
    ("cross_lingual", "cross_lingual", True),
)


# ---------------------------------------------------------------- 评测集


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """读取并校验生成评测集。格式错误直接抛，别带着一条脏用例跑十分钟。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError(f"评测集格式错误，缺少 cases 列表：{path}")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw["cases"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条评测用例不是 JSON 对象")

        case_id = str(item.get("id") or "").strip()
        category = str(item.get("category") or "").strip()
        source_mode = str(item.get("source_mode") or "").strip()
        query = str(item.get("query") or "").strip()
        if not case_id or not category or not source_mode or not query:
            raise ValueError(f"第 {index} 条评测用例缺少 id / category / source_mode / query")
        if case_id in seen:
            raise ValueError(f"评测集存在重复 id：{case_id}")
        if source_mode not in SOURCE_MODES:
            raise ValueError(
                f"用例 {case_id} 的 source_mode `{source_mode}` 不支持，"
                f"可选：{'/'.join(SOURCE_MODES)}"
            )
        seen.add(case_id)

        hints = item.get("reference_hints") or []
        if not isinstance(hints, list):
            raise ValueError(f"用例 {case_id} 的 reference_hints 必须是列表")

        expect = item.get("expect_refusal")
        if expect is not None and not isinstance(expect, bool):
            raise ValueError(f"用例 {case_id} 的 expect_refusal 必须是布尔值")

        variants = {
            name: str(item.get(name) or "").strip()
            for name in VARIANTS
            if name != "original" and str(item.get(name) or "").strip()
        }

        cases.append(
            {
                "id": case_id,
                "category": category,
                "source_mode": source_mode,
                "query": query,
                "variants": variants,
                "expect_refusal": expect,
                "reference_hints": [str(h) for h in hints],
                "note": str(item.get("note") or "").strip(),
            }
        )

    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


def filter_cases(
    cases: Sequence[dict[str, Any]],
    *,
    categories: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """按类别 / id 过滤。指定了不存在的类别或 id 时报错，而不是静默返回空。"""
    wanted_cats = [c for c in (categories or ()) if c]
    wanted_ids = [i for i in (ids or ()) if i]
    if wanted_cats:
        available = {c["category"] for c in cases}
        unknown = [c for c in wanted_cats if c not in available]
        if unknown:
            raise ValueError(
                f"未知评测类别：{', '.join(unknown)}（可选：{', '.join(sorted(available))}）"
            )
    if wanted_ids:
        available_ids = {c["id"] for c in cases}
        unknown = [i for i in wanted_ids if i not in available_ids]
        if unknown:
            raise ValueError(f"未知用例 id：{', '.join(unknown)}")

    cat_set = set(wanted_cats)
    id_set = set(wanted_ids)
    return [
        c
        for c in cases
        if (not cat_set or c["category"] in cat_set)
        and (not id_set or c["id"] in id_set)
    ]


def _query_for(case: dict[str, Any], variant: str) -> str:
    if variant == "original":
        return case["query"]
    return case.get("variants", {}).get(variant, "")


def expand_cases(
    cases: Sequence[dict[str, Any]],
    *,
    variants: Sequence[str],
    rewrite: bool,
) -> list[dict[str, Any]]:
    """把每条用例按问法展开成多条运行任务（单配置模式）。"""
    runtime: list[dict[str, Any]] = []
    for variant in variants:
        for case in cases:
            query = _query_for(case, variant)
            if not query:
                continue
            runtime.append(
                {
                    **case,
                    "query": query,
                    "variant": variant,
                    "config": variant if not rewrite else f"{variant}+rewrite",
                    "rewrite": rewrite,
                    "run_id": f"{case['id']}@{variant}",
                }
            )
    return runtime


def expand_matrix(
    cases: Sequence[dict[str, Any]],
    configs: Sequence[tuple[str, str, bool]] = MATRIX_CONFIGS,
) -> list[dict[str, Any]]:
    """策略矩阵：每种配置 × 每条用例。"""
    runtime: list[dict[str, Any]] = []
    for name, variant, rewrite in configs:
        for case in cases:
            query = _query_for(case, variant)
            if not query:
                continue
            runtime.append(
                {
                    **case,
                    "query": query,
                    "variant": variant,
                    "config": name,
                    "rewrite": rewrite,
                    "run_id": f"{case['id']}@{name}",
                }
            )
    return runtime


# ---------------------------------------------------------------- 隔离运行 Agent


def _bind_vault_docs(key: str) -> list[Doc]:
    """把储藏区语料挂到这条用例的会话上。owned=False，清理时不会删原文件。"""
    docs: list[Doc] = []
    for entry in vault.documents():
        doc = vault.open_document(entry)
        if doc is not None:
            data_tools.register_document(doc, key=key)
            docs.append(doc)
    return docs


def prepare_sources(key: str, source_mode: str) -> tuple[list[Doc], list[str], int, list[dict[str, Any]]]:
    """按 source_mode 挂载语料。返回 (文档, 列名, 行数, 文档摘要)。"""
    docs: list[Doc] = []
    columns: list[str] = []
    rows = 0

    if source_mode in ("vault", "both"):
        docs = _bind_vault_docs(key)
    if source_mode in ("sales", "both"):
        data_tools.bind_sample(key)
        columns = data_tools.column_names()
        rows = int(data_tools.dataset_meta().get("rows") or 0)

    files = data_tools.document_digest(key)
    return docs, columns, rows, files


def cleanup_sources(key: str) -> None:
    data_tools.clear_documents(key)
    data_tools.unbind_dataset(key)
    # Agent.ask 会把活跃 key 留在 ContextVar 里（它故意不还原）。评测进程里
    # 这没影响，但跑完不归位的话，同进程后续操作会查到一个已经撤掉的会话。
    data_tools.use_dataset(data_tools.DEFAULT_KEY)


def generate_prediction(
    case: dict[str, Any],
    *,
    rewrite: bool = True,
    search_mode: str = "hybrid",
    llm: Any = None,
) -> dict[str, Any]:
    """跑一次真实问答，把评测需要的四样东西捞全：答案、上下文、改写事件、耗时。

    `llm` 只给单测用：注入脚本化模型就能离线验通整条流水线，不碰网络。
    """
    key = f"eval-gen-{case['id']}-{uuid.uuid4().hex[:6]}"
    events: list[tuple[str, dict[str, Any]]] = []
    answer = ""
    error = ""
    session: Session | None = None
    started = time.perf_counter()

    hybrid_search.set_hybrid_enabled(search_mode == "hybrid")
    try:
        _, columns, rows, files = prepare_sources(key, case["source_mode"])
        prompt = make_system_prompt(
            columns=columns,
            rows=rows or None,
            files=files,
        )
        session = Session()
        agent_kwargs: dict[str, Any] = {
            "session": session,
            "system_prompt": prompt,
            "session_key": key,
            "rewriter": QueryRewriter() if rewrite else None,
            "on_event": lambda kind, payload: events.append((kind, payload)),
        }
        if llm is not None:
            agent_kwargs["llm"] = llm
        answer = Agent(**agent_kwargs).ask(case["query"])
    except Exception as exc:  # noqa: BLE001  一条用例炸掉不该拖垮整批
        error = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        cleanup_sources(key)
        hybrid_search.set_hybrid_enabled(True)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    messages = list(session.messages) if session is not None else []
    contexts = extract_turn_contexts(messages) if messages else []
    rewrite_event = next(
        (payload for kind, payload in events if kind == "rewrite"), {}
    )

    return {
        "id": case["id"],
        "run_id": case.get("run_id") or f"{case['id']}@{case.get('variant', 'original')}",
        "category": case["category"],
        "source_mode": case["source_mode"],
        "variant": case.get("variant", "original"),
        "config": case.get("config", ""),
        "query": case["query"],
        "effective_query": str(rewrite_event.get("rewritten_query") or case["query"]),
        "rewrite": {
            "strategy": str(rewrite_event.get("strategy") or ""),
            "label": str(rewrite_event.get("label") or ""),
            "fallback": bool(rewrite_event.get("fallback")),
            "bypassed": bool(rewrite_event.get("bypassed")),
            "error": str(rewrite_event.get("error") or ""),
            "tokens": int(rewrite_event.get("tokens") or 0),
            "latency": float(rewrite_event.get("latency") or 0.0),
            "search_keywords": list(rewrite_event.get("search_keywords") or []),
            "sub_query_count": len(rewrite_event.get("sub_queries") or []),
        },
        "answer": answer,
        "contexts": [
            {
                "tool_name": item.tool_name,
                "args_summary": item.args_summary,
                "content": item.content,
                "call_id": item.call_id,
            }
            for item in contexts
        ],
        "context_count": len(contexts),
        "context_chars": sum(len(item.content) for item in contexts),
        "gen_latency_ms": elapsed_ms,
        "steps": int(getattr(session, "steps", 0)),
        "tool_calls": int(getattr(session, "tool_calls", 0)),
        "gen_tokens": int(getattr(session, "tokens", 0)),
        "expect_refusal": case.get("expect_refusal"),
        "reference_hints": list(case.get("reference_hints") or []),
        "note": case.get("note", ""),
        "error": error,
    }


# ---------------------------------------------------------------- 打分


def _refusal_mismatch(record: dict[str, Any]) -> bool:
    """该答却回避才算不符。

    反向（该拒答却答了）不在这里判：不可回答的用例上，模型只要能给出有依据的
    说明，就算答得对，硬要求它说"我不知道"是把评测标准写反了。那种情况由忠实度
    去抓——编了东西就会掉分。
    """
    if record.get("expect_refusal") is not False:
        return False
    return bool(record.get("noncommittal") or record.get("is_refusal"))


def evaluate_predictions(
    predictions: Sequence[dict[str, Any]],
    *,
    evaluator: AnswerEvaluator | None = None,
    faithfulness_threshold: float = 0.8,
    relevance_threshold: float = 0.7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对快照/实跑结果逐条打分，返回 (明细记录, 聚合汇总)。"""
    engine = evaluator or AnswerEvaluator(
        faithfulness_threshold=faithfulness_threshold,
        relevance_threshold=relevance_threshold,
    )

    records: list[dict[str, Any]] = []
    for pred in predictions:
        result = engine.evaluate(
            question=str(pred.get("query") or ""),
            answer=str(pred.get("answer") or ""),
            contexts=pred.get("contexts") or (),
            effective_query=str(pred.get("effective_query") or ""),
            expect_refusal=pred.get("expect_refusal"),
        )
        record = result.to_dict()
        record.update(
            {
                "id": pred.get("id"),
                "run_id": pred.get("run_id") or f"{pred.get('id')}@{pred.get('variant', 'original')}",
                "category": pred.get("category") or "uncategorized",
                "variant": pred.get("variant") or "original",
                "config": pred.get("config") or "",
                "source_mode": pred.get("source_mode") or "",
                "query": str(pred.get("query") or ""),
                "answer": str(pred.get("answer") or ""),
                "expect_refusal": pred.get("expect_refusal"),
                "reference_hints": list(pred.get("reference_hints") or []),
                "rewrite": pred.get("rewrite") or {},
                "gen_latency_ms": float(pred.get("gen_latency_ms") or 0.0),
                "steps": int(pred.get("steps") or 0),
                "tool_calls": int(pred.get("tool_calls") or 0),
                "gen_tokens": int(pred.get("gen_tokens") or 0),
                "gen_error": str(pred.get("error") or ""),
            }
        )
        record["refusal_mismatch"] = _refusal_mismatch(record)
        records.append(record)

    return records, summarize_generation_results(records)


# ---------------------------------------------------------------- 报告


def _short(text: Any, limit: int = 44) -> str:
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _width(text: Any) -> int:
    """终端显示宽度。中文占两列，用 len() 对齐会一行一行错开。"""
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in str(text))


def _pad(text: Any, width: int, align: str = "left") -> str:
    gap = max(0, width - _width(text))
    return (" " * gap + str(text)) if align == "right" else (str(text) + " " * gap)


def _fmt_score(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.0f}%"


def summarize_by_config(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("config") or "default"), []).append(record)
    return {
        name: summarize_generation_results(items)["overall"]
        for name, items in grouped.items()
    }


def format_generation_report(
    *,
    dataset_path: Path,
    predictions: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    show_all: bool = False,
) -> str:
    """终端报告。整体 → 分配置 → 分类别 → 逐题 → Bad Case。"""
    summaries = summarize_by_config(records)
    configs = list(summaries)
    emb_cfg = config.embedding_current()
    lines: list[str] = [
        "=" * 104,
        "DataInsight 生成质量离线评测报告（忠实度 Faithfulness / 答案相关性 Answer Relevance）",
        "=" * 104,
        f"评测集   : {dataset_path}",
        f"样本数   : {len(predictions)} 次问答 / {len({p.get('id') for p in predictions})} 条用例",
        f"Judge    : {config.current().model}（只读结果，不落 Key）",
        (
            f"相关性路 : "
            + (
                f"逆向向量 {emb_cfg.provider}:{emb_cfg.model}"
                if emb_cfg.ready
                else "llm_judge（未配置向量模型，已降级）"
            )
        ),
        "",
        "## 1. 总体指标（按配置）",
    ]
    header = " | ".join(
        [
            _pad("配置", 26),
            _pad("N", 3, "right"),
            _pad("忠实度", 7, "right"),
            _pad("相关性", 7, "right"),
            _pad("pass", 4, "right"),
            _pad("warn", 4, "right"),
            _pad("fail", 4, "right"),
            _pad("skip", 4, "right"),
            _pad("fb", 3, "right"),
            _pad("矛盾", 4, "right"),
            _pad("拒答", 4, "right"),
            _pad("生成(s)", 8, "right"),
        ]
    )
    lines += [header, "-" * _width(header)]

    def _stats_row(name: str, ov: dict[str, Any]) -> str:
        return " | ".join(
            [
                _pad(name, 26),
                _pad(ov["count"], 3, "right"),
                _pad(_fmt_score(ov["avg_faithfulness"]), 7, "right"),
                _pad(_fmt_score(ov["avg_relevance"]), 7, "right"),
                _pad(ov["pass_count"], 4, "right"),
                _pad(ov["warn_count"], 4, "right"),
                _pad(ov["fail_count"], 4, "right"),
                _pad(ov["skipped_count"], 4, "right"),
                _pad(ov["fallback_count"], 3, "right"),
                _pad(ov["contradiction_case_count"], 4, "right"),
                _pad(ov["noncommittal_count"], 4, "right"),
                _pad(f"{ov['avg_gen_latency_ms'] / 1000.0:.1f}", 8, "right"),
            ]
        )

    for name in configs:
        lines.append(_stats_row(name, summaries[name]))

    overall = summarize_generation_results(records)["overall"]
    lines.append("-" * _width(header))
    lines.append(_stats_row("合计", overall))
    lines.append(
        f"声明核验 : 共 {overall['total_claims']} 条，支持 {overall['supported_claims']} / "
        f"无依据 {overall['unsupported_claims']} / 矛盾 {overall['contradicted_claims']}；"
        f"拒答与预期不符 {overall['refusal_mismatch_count']} 条"
    )

    lines.extend(["", "## 2. 分类别（忠实度 / 相关性 / 通过率）"])
    categories = sorted({str(r.get("category") or "uncategorized") for r in records})
    cat_header = " | ".join(
        [_pad("类别", 12), _pad("N", 3, "right"), _pad("声明", 12, "right")]
        + [_pad(_short(name, 17), 18) for name in configs]
    )
    lines += [cat_header, "-" * _width(cat_header)]

    for category in categories:
        items = [r for r in records if str(r.get("category") or "uncategorized") == category]
        claims = sum(
            int(r.get("supported_count") or 0)
            + int(r.get("unsupported_count") or 0)
            + int(r.get("contradicted_count") or 0)
            for r in items
        )
        supported = sum(int(r.get("supported_count") or 0) for r in items)
        cells = [
            _pad(category, 12),
            _pad(len(items), 3, "right"),
            _pad(f"{supported}/{claims}", 12, "right"),
        ]
        for name in configs:
            subset = [r for r in items if str(r.get("config") or "default") == name]
            if not subset:
                cells.append(_pad("—", 18))
                continue
            stats = summarize_generation_results(subset)["overall"]
            cells.append(
                _pad(
                    f"{_fmt_score(stats['avg_faithfulness'])}/"
                    f"{_fmt_score(stats['avg_relevance'])} {_fmt_pct(stats['pass_rate'])}",
                    18,
                )
            )
        lines.append(" | ".join(cells))

    lines.extend(
        [
            "",
            f"## 3. 逐题评分明细（{'全部' if show_all else '仅非 pass'}）",
        ]
    )
    detail_header = " | ".join(
        [
            _pad("用例", 7),
            _pad("配置", 18),
            _pad("评级", 8),
            _pad("忠实度", 6, "right"),
            _pad("相关性", 6, "right"),
            _pad("声明s/u/c", 9, "right"),
            _pad("改写策略", 16),
            "答案摘要",
        ]
    )
    lines += [detail_header, "-" * 118]
    shown_records = (
        list(records)
        if show_all
        else [r for r in records if r.get("verdict") != "pass"]
    )
    for record in shown_records:
        claims = (
            f"{record.get('supported_count', 0)}/{record.get('unsupported_count', 0)}/"
            f"{record.get('contradicted_count', 0)}"
        )
        strategy = str((record.get("rewrite") or {}).get("strategy") or "—")
        lines.append(
            " | ".join(
                [
                    _pad(record.get("id"), 7),
                    _pad(_short(record.get("config"), 18), 18),
                    _pad(record.get("verdict"), 8),
                    _pad(_fmt_score(record.get("faithfulness")), 6, "right"),
                    _pad(_fmt_score(record.get("relevance")), 6, "right"),
                    _pad(claims, 9, "right"),
                    _pad(_short(strategy, 16), 16),
                    _short(record.get("answer"), 40),
                ]
            )
        )

    bad = [
        r
        for r in records
        if r.get("verdict") in {"fail", "warn", "fallback", "skipped"}
        or r.get("refusal_mismatch")
    ]
    lines.extend(["", f"## 4. Bad Case 诊断（{len(bad)}/{len(records)}）"])
    if not bad:
        lines.append("  全部用例通过。")
    for record in bad:
        lines.append(
            f"\n- [{record.get('id')}] {record.get('config')} · "
            f"verdict={record.get('verdict')} · fa={_fmt_score(record.get('faithfulness'))} "
            f"· rel={_fmt_score(record.get('relevance'))} "
            f"· mode={record.get('relevance_mode')}"
        )
        problem = str(record.get("error") or record.get("summary") or "").strip()
        if problem:
            lines.append(f"    问题   : {problem}")
        if record.get("refusal_mismatch"):
            lines.append(
                f"    拒答   : 与预期不符（expect_refusal={record.get('expect_refusal')}，"
                f"实际 noncommittal={record.get('noncommittal')}）"
            )
        if record.get("missing_aspects"):
            lines.append(f"    遗漏   : {'；'.join(record['missing_aspects'])}")
        for claim in record.get("claims") or []:
            if claim.get("verdict") == "supported":
                continue
            quote = str(claim.get("evidence_quote") or "").strip()
            tail = f" | 依据: {_short(quote, 40)}" if quote else ""
            lines.append(
                f"    {claim.get('verdict'):<11}: {_short(claim.get('statement'), 60)}{tail}"
            )
        lines.append(f"    答案   : {_short(record.get('answer'), 120)}")

    return "\n".join(lines)


def format_markdown_report(
    *,
    dataset_path: Path,
    predictions: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> str:
    """可提交、可贴进文档的 Markdown 版报告。"""
    summaries = summarize_by_config(records)
    overall = summarize_generation_results(records)["overall"]
    lines: list[str] = [
        "# DataInsight 生成质量评测报告",
        "",
        "指标口径：**忠实度** = 被本轮工具上下文支撑的原子声明数 / 声明总数；"
        "**相关性** = 逆向生成问题与用户问题的向量余弦均值（未配置向量时用 Judge 评分）。",
        "",
        f"- 评测集：`{dataset_path.name}`，{len(predictions)} 次问答 / {len({p.get('id') for p in predictions})} 条用例",
        f"- Judge 模型：`{config.current().model}`",
        f"- 判定阈值：忠实度 ≥ 0.8，相关性 ≥ 0.7；任一声明与上下文冲突直接判 fail",
        "",
        "## 1. 总体指标",
        "",
        "| 配置 | N | 忠实度 | 相关性 | pass | warn | fail | skip | 矛盾用例 | 拒答 | 生成(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, ov in summaries.items():
        lines.append(
            f"| `{name}` | {ov['count']} | {_fmt_score(ov['avg_faithfulness'])} | "
            f"{_fmt_score(ov['avg_relevance'])} | {ov['pass_count']} | {ov['warn_count']} | "
            f"{ov['fail_count']} | {ov['skipped_count']} | "
            f"{ov['contradiction_case_count']} | {ov['noncommittal_count']} | "
            f"{ov['avg_gen_latency_ms'] / 1000.0:.1f} |"
        )
    lines.append(
        f"| **合计** | {overall['count']} | {_fmt_score(overall['avg_faithfulness'])} | "
        f"{_fmt_score(overall['avg_relevance'])} | {overall['pass_count']} | "
        f"{overall['warn_count']} | {overall['fail_count']} | {overall['skipped_count']} | "
        f"{overall['contradiction_case_count']} | {overall['noncommittal_count']} | "
        f"{overall['avg_gen_latency_ms'] / 1000.0:.1f} |"
    )

    lines.extend(
        [
            "",
            f"声明核验合计：{overall['total_claims']} 条，"
            f"支持 {overall['supported_claims']}、无依据 {overall['unsupported_claims']}、"
            f"矛盾 {overall['contradicted_claims']}。",
            "",
            "## 2. 逐题评分",
            "",
            "| 用例 | 配置 | 问题 | 评级 | 忠实度 | 相关性 | 声明(s/u/c) | 改写策略 | 答案摘要 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for record in records:
        claims = (
            f"{record.get('supported_count', 0)}/{record.get('unsupported_count', 0)}/"
            f"{record.get('contradicted_count', 0)}"
        )
        lines.append(
            f"| {record.get('id')} | `{record.get('config')}` | "
            f"{_short(record.get('query'), 30)} | {record.get('verdict')} | "
            f"{_fmt_score(record.get('faithfulness'))} | {_fmt_score(record.get('relevance'))} | "
            f"{claims} | {(record.get('rewrite') or {}).get('strategy') or '—'} | "
            f"{_short(record.get('answer'), 40)} |"
        )

    bad = [
        r
        for r in records
        if r.get("verdict") in {"fail", "fallback", "skipped"} or r.get("refusal_mismatch")
    ]
    lines.extend(["", f"## 3. 失败与异常样本（{len(bad)} 条）", ""])
    if not bad:
        lines.append("无。")
    for record in bad:
        lines.append(f"### {record.get('id')} · `{record.get('config')}` · {record.get('verdict')}")
        lines.append("")
        lines.append(f"- 问题：{record.get('query')}")
        lines.append(f"- 答案：{_short(record.get('answer'), 400)}")
        lines.append(
            f"- 分数：忠实度 {_fmt_score(record.get('faithfulness'))}，"
            f"相关性 {_fmt_score(record.get('relevance'))}"
        )
        if record.get("error"):
            lines.append(f"- 错误：{record.get('error')}")
        for claim in record.get("claims") or []:
            if claim.get("verdict") != "supported":
                lines.append(
                    f"  - {claim.get('verdict')}：{claim.get('statement')}"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------- 快照读写


def save_predictions(path: Path, predictions: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(predictions),
        "cases": list(predictions),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """读快照。接受 `{"cases": [...]}`、`{"predictions": [...]}` 或裸列表三种写法。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("cases") or raw.get("predictions") or []
    else:
        items = []
    if not isinstance(items, list) or not items:
        raise ValueError(f"快照文件里没有可用的用例：{path}")
    return [item for item in items if isinstance(item, dict)]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- CLI


def _parse_list(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DataInsight 生成质量离线评测（忠实度 + 答案相关性）",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="评测集 JSON 路径")
    parser.add_argument("--categories", default="", help="逗号分隔的类别过滤（doc_fact,table_agg,...）")
    parser.add_argument("--ids", default="", help="逗号分隔的用例 id 过滤（g01,g05,...）")
    parser.add_argument(
        "--variants",
        default="",
        help="逗号分隔的问法（original,colloquial,cross_lingual）；默认 original",
    )
    parser.add_argument("--rewrite", dest="rewrite", action="store_true", default=None, help="启用前置查询改写")
    parser.add_argument("--no-rewrite", dest="rewrite", action="store_false", help="关闭前置查询改写")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="跑策略矩阵：baseline / basic_rewrite / colloquial / cross_lingual 四组配置",
    )
    parser.add_argument(
        "--search-mode",
        choices=("bm25", "hybrid"),
        default="hybrid",
        help="检索模式（默认 hybrid）",
    )
    parser.add_argument("--no-embedding", action="store_true", help="相关性只走 Judge 评分，不调向量接口")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.8, help="忠实度通过阈值")
    parser.add_argument("--relevance-threshold", type=float, default=0.7, help="相关性通过阈值")
    parser.add_argument("--save-predictions", type=Path, default=None, help="把生成结果存成快照")
    parser.add_argument("--predictions", type=Path, default=None, help="读快照跑评测，跳过 Agent")
    parser.add_argument(
        "--show-all", action="store_true", help="逐题明细打印全部用例（默认只打印非 pass 的）"
    )
    parser.add_argument("--json-out", type=Path, default=None, help="导出结构化报告 JSON")
    parser.add_argument("--md-out", type=Path, default=None, help="导出 Markdown 报告")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    categories = _parse_list(args.categories)
    ids = _parse_list(args.ids)

    predictions: list[dict[str, Any]]
    if args.predictions is not None:
        predictions = load_predictions(args.predictions)
        if ids:
            predictions = [p for p in predictions if p.get("id") in set(ids)]
        if categories:
            predictions = [p for p in predictions if p.get("category") in set(categories)]
        if not predictions:
            raise RuntimeError("快照里没有符合条件的用例，检查一下 --ids / --categories。")
    else:
        cases = filter_cases(load_dataset(args.dataset), categories=categories, ids=ids)
        if not cases:
            raise RuntimeError("过滤后没有任何用例。")

        cfg = config.current()
        if not cfg.ready:
            raise RuntimeError(
                "没有可用的对话模型配置（BASE_URL / MODEL / API_KEY）。"
                "先在 .env 或 data/model.json 里配好再跑。"
            )

        if args.matrix:
            variants = _parse_list(args.variants)
            configs = [
                item for item in MATRIX_CONFIGS if not variants or item[1] in set(variants)
            ]
            runtime = expand_matrix(cases, configs)
        else:
            variants = _parse_list(args.variants) or ["original"]
            unknown = [v for v in variants if v not in VARIANTS]
            if unknown:
                raise ValueError(
                    f"未知问法：{', '.join(unknown)}（可选：{', '.join(VARIANTS)}）"
                )
            rewrite = True if args.rewrite is None else bool(args.rewrite)
            runtime = expand_cases(cases, variants=variants, rewrite=rewrite)

        if not runtime:
            raise RuntimeError("展开后没有任何任务，检查一下问法是否缺失。")

        total = len(runtime)
        predictions = []
        for index, case in enumerate(runtime, start=1):
            print(
                f"[{index}/{total}] {case['id']} · {case['config']} · "
                f"{case['variant']} · rewrite={case['rewrite']}",
                file=sys.stderr,
                flush=True,
            )
            prediction = generate_prediction(
                case, rewrite=case["rewrite"], search_mode=args.search_mode
            )
            if prediction["error"]:
                print(f"    生成失败：{prediction['error']}", file=sys.stderr, flush=True)
            else:
                print(
                    f"    {prediction['steps']} 步 / {prediction['tool_calls']} 次工具 / "
                    f"{prediction['context_count']} 段上下文 / "
                    f"{prediction['gen_latency_ms'] / 1000.0:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            predictions.append(prediction)

        if args.save_predictions is not None:
            save_predictions(args.save_predictions, predictions)
            print(f"\n已保存生成快照：{args.save_predictions}", file=sys.stderr)

    evaluator = AnswerEvaluator(
        use_embedding=not args.no_embedding,
        faithfulness_threshold=args.faithfulness_threshold,
        relevance_threshold=args.relevance_threshold,
    )
    records, _ = evaluate_predictions(predictions, evaluator=evaluator)

    report = format_generation_report(
        dataset_path=args.dataset,
        predictions=predictions,
        records=records,
        show_all=args.show_all,
    )
    print(report)

    if args.json_out is not None:
        payload = {
            "dataset": str(args.dataset),
            "predictions_count": len(predictions),
            "summary": summarize_generation_results(records),
            "records": records,
        }
        _write_text(args.json_out, json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n已导出 JSON 报告：{args.json_out}")
    if args.md_out is not None:
        _write_text(
            args.md_out,
            format_markdown_report(
                dataset_path=args.dataset,
                predictions=predictions,
                records=records,
            ),
        )
        print(f"已导出 Markdown 报告：{args.md_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
