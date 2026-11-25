"""生成质量评测（eval/answer_metrics.py + eval/eval_generation.py）的离线单测。

全部用替身模型跑，不发真实网络请求，也不读写 data/vault 里的 PDF。
脚本化模型 `ScriptedLLM` 负责把 Agent 的 ReAct 循环往前推，用来验证
"隔离会话 → 跑一轮 → 提取上下文 → 打分"这条流水线确实是通的。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import EmbeddingError  # noqa: E402
from app.llm import LLMError, LLMResponse  # noqa: E402
from app.tools import data_tools  # noqa: E402
from eval import answer_metrics as am  # noqa: E402
from eval import eval_generation as eg  # noqa: E402


# ---------------------------------------------------------------- 替身模型


def _response(content: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> LLMResponse:
    return LLMResponse(
        message={"role": "assistant", "content": content},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class FakeLLM:
    """固定返回一段 JSON 的假 Judge。calls 用来断言"没该调用时确实没调"。"""

    def __init__(self, content: str = "{}") -> None:
        self.content = content
        self.calls = 0
        self.last_messages: list[dict] = []

    def __call__(self, messages, tools=None):
        self.calls += 1
        self.last_messages = messages
        return _response(self.content)


class BoomLLM:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or LLMError("boom")
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        raise self.exc


class ScriptedLLM:
    """按剧本逐步返回 assistant 消息，用来驱动真实 Agent 循环。"""

    def __init__(self, scripts: list[dict]) -> None:
        self.scripts = list(scripts)
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        index = min(self.calls - 1, len(self.scripts) - 1)
        return LLMResponse(
            message=self.scripts[index], prompt_tokens=5, completion_tokens=5
        )


def judge_payload(
    claims: list[dict] | None = None,
    *,
    noncommittal: bool = False,
    is_refusal: bool = False,
    questions: list[str] | None = None,
    relevance: float = 0.95,
    missing: list[str] | None = None,
    summary: str = "",
) -> str:
    return json.dumps(
        {
            "is_refusal": is_refusal,
            "noncommittal": noncommittal,
            "claims": claims or [],
            "generated_questions": questions
            if questions is not None
            else ["问题一", "问题二", "问题三"],
            "judge_relevance": relevance,
            "missing_aspects": missing or [],
            "summary": summary,
        },
        ensure_ascii=False,
    )


def claim(statement: str, verdict: str = "supported", quote: str = "原文依据") -> dict:
    return {
        "statement": statement,
        "verdict": verdict,
        "evidence_source": "run_sql",
        "evidence_quote": quote,
        "reason": "测试",
    }


def _ctx(content: str = "上海 241589.06", tool: str = "run_sql") -> am.EvidenceItem:
    return am.EvidenceItem(tool_name=tool, args_summary="sql=SELECT 1", content=content)


# ---------------------------------------------------------------- 上下文提取


class TestExtractTurnContexts(unittest.TestCase):
    def test_takes_only_last_user_turn(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "第一轮问题"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "run_sql", "arguments": '{"sql": "SELECT 1"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "第一轮上下文"},
            {"role": "assistant", "content": "第一轮答案"},
            {"role": "user", "content": "第二轮问题"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "b",
                        "type": "function",
                        "function": {
                            "name": "read_document",
                            "arguments": '{"name": "a.pdf", "part": 2}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "第二轮上下文"},
            {"role": "assistant", "content": "第二轮答案"},
        ]

        contexts = am.extract_turn_contexts(messages)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].tool_name, "read_document")
        self.assertIn("name=a.pdf", contexts[0].args_summary)
        self.assertIn("part=2", contexts[0].args_summary)
        self.assertEqual(contexts[0].content, "第二轮上下文")

    def test_explicit_user_idx_and_missing_pairs(self):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "tool", "tool_call_id": "nope", "content": "孤立观察"},
            {"role": "user", "content": "q2"},
        ]
        contexts = am.extract_turn_contexts(messages, user_idx=0)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].tool_name, "unknown")
        # 配不上 tool_call 的观察保留内容，但名字退化成 unknown，不假装知道是谁
        self.assertEqual(contexts[0].content, "孤立观察")

    def test_dedup_identical_observations(self):
        call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"id{i}", "function": {"name": "run_sql", "arguments": '{"sql": "SELECT 1"}'}}
                for i in range(2)
            ],
        }
        messages = [
            {"role": "user", "content": "q"},
            call,
            {"role": "tool", "tool_call_id": "id0", "content": "同样的结果"},
            {"role": "tool", "tool_call_id": "id1", "content": "同样的结果"},
        ]
        self.assertEqual(len(am.extract_turn_contexts(messages)), 1)

    def test_budget_keeps_high_value_tools(self):
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "function": {"name": "list_documents", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "run_sql", "arguments": '{"sql": "SELECT 1"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "L" * 100},
            {"role": "tool", "tool_call_id": "b", "content": "S" * 100},
        ]
        contexts = am.extract_turn_contexts(messages, max_total_chars=150)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].tool_name, "run_sql")

    def test_single_item_truncated(self):
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "function": {"name": "run_sql", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "X" * 500},
        ]
        contexts = am.extract_turn_contexts(messages, max_item_chars=50)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(len(contexts[0].content), 51)  # 50 字 + 省略号
        self.assertTrue(contexts[0].content.endswith("…"))

    def test_summarize_args_handles_bad_json(self):
        self.assertEqual(am.summarize_args('{"sql": "SELECT 1"}'), "sql=SELECT 1")
        self.assertEqual(am.summarize_args("not a json"), "not a json")
        self.assertEqual(am.summarize_args({"part": 2}), "part=2")

    def test_tool_priority_order(self):
        self.assertLess(am.tool_priority("run_sql"), am.tool_priority("search_document"))
        self.assertLess(am.tool_priority("search_document"), am.tool_priority("list_documents"))
        self.assertEqual(am.tool_priority("未知工具"), 1)


# ---------------------------------------------------------------- 忠实度


class TestFaithfulnessScoring(unittest.TestCase):
    def _evaluate(self, payload: str, *, answer="答案", contexts=(_ctx(),), **kwargs):
        llm = FakeLLM(payload)
        evaluator = am.AnswerEvaluator(llm=llm, use_embedding=False)
        result = evaluator.evaluate(
            question="哪个城市销售额最高？",
            answer=answer,
            contexts=list(contexts),
            **kwargs,
        )
        return result, llm

    def test_all_supported_is_perfect(self):
        payload = judge_payload([claim("上海最高"), claim("占比 29.7%")])
        result, _ = self._evaluate(payload)
        self.assertAlmostEqual(result.faithfulness, 1.0)
        self.assertEqual(result.verdict, "pass")
        self.assertEqual(result.supported_count, 2)
        self.assertEqual(result.contradicted_count, 0)

    def test_partial_support_ratio(self):
        payload = judge_payload(
            [
                claim("上海最高"),
                claim("深圳第二"),
                claim("杭州第三", "unsupported", ""),
            ]
        )
        result, _ = self._evaluate(payload)
        self.assertAlmostEqual(result.faithfulness, 2 / 3)
        self.assertEqual(result.unsupported_count, 1)
        self.assertEqual(result.verdict, "warn")

    def test_one_unsupported_but_high_ratio_still_passes(self):
        # 10 条里 1 条没找到依据 = 0.9，仍高于阈值；warn 留给真正掉到阈值以下的
        claims = [claim(f"事实 {i}") for i in range(9)] + [claim("多写的一句", "unsupported", "")]
        result, _ = self._evaluate(judge_payload(claims))
        self.assertAlmostEqual(result.faithfulness, 0.9)
        self.assertEqual(result.unsupported_count, 1)
        self.assertEqual(result.verdict, "pass")

    def test_contradiction_forces_fail(self):
        payload = judge_payload([claim("北京最高"), claim("占比 50%", "contradicted")])
        result, _ = self._evaluate(payload)
        self.assertEqual(result.contradicted_count, 1)
        self.assertEqual(result.verdict, "fail")

    def test_no_contexts_forces_unsupported(self):
        payload = judge_payload([claim("上海 241589 元")])
        result, _ = self._evaluate(payload, contexts=())
        self.assertAlmostEqual(result.faithfulness, 0.0)
        self.assertEqual(result.supported_count, 0)
        self.assertEqual(result.verdict, "fail")

    def test_honest_refusal_scores_full_faithfulness(self):
        payload = judge_payload(
            noncommittal=True, is_refusal=True, questions=[], relevance=0.1
        )
        result, _ = self._evaluate(payload, answer="资料里没有提到该预算。", contexts=())
        self.assertAlmostEqual(result.faithfulness, 1.0)
        self.assertEqual(result.relevance, 0.0)
        self.assertEqual(result.relevance_mode, "noncommittal")
        # 默认没声明预期时，诚实拒答算通过
        self.assertEqual(result.verdict, "pass")

    def test_refusal_against_expectation_is_fail(self):
        payload = judge_payload(
            noncommittal=True, is_refusal=True, questions=[], relevance=0.1
        )
        result, _ = self._evaluate(
            payload, answer="没找到。", contexts=(), expect_refusal=False
        )
        self.assertEqual(result.verdict, "fail")

    def test_pure_meta_answer_has_no_faithfulness(self):
        payload = judge_payload([], noncommittal=False, relevance=0.9)
        result, _ = self._evaluate(payload, answer="我可以帮你查一下。")
        self.assertIsNone(result.faithfulness)
        self.assertEqual(result.verdict, "pass")

    def test_aborted_answer_is_skipped_without_judge_call(self):
        for text in (
            "（Agent 在重复调用同一个工具，已强制中止。换个问法试试。）",
            "（这一问用掉的 token 超出预算，先停下。这通常说明问题太宽泛。",
            "（这一轮已经跑了 100 步还没收尾，先停在这里。",
            "（模型调用失败：HTTP 401）",
            "",
        ):
            result, llm = self._evaluate("{}", answer=text)
            self.assertEqual(result.verdict, "skipped")
            self.assertTrue(result.skipped)
            self.assertEqual(llm.calls, 0)


# ---------------------------------------------------------------- 相关性


def _fake_embed(texts):
    table = {
        "问题": [1.0, 0.0, 0.0],
        "q1": [1.0, 0.0, 0.0],
        "q2": [0.0, 1.0, 0.0],
        "ORIGINAL": [0.0, 1.0, 0.0],
        "EFFECTIVE": [1.0, 0.0, 0.0],
    }
    return np.asarray([table[t] for t in texts], dtype=np.float32)


class TestRelevanceScoring(unittest.TestCase):
    def test_reverse_embedding_mean_cosine(self):
        llm = FakeLLM(judge_payload([claim("x")], questions=["q1", "q2"], relevance=0.99))
        evaluator = am.AnswerEvaluator(llm=llm, embedder=_fake_embed, use_embedding=True)
        result = evaluator.evaluate(question="问题", answer="答案", contexts=[_ctx()])
        self.assertEqual(result.relevance_mode, "reverse_embedding")
        self.assertAlmostEqual(result.relevance, 0.5)  # cos=1.0 与 cos=0.0 的均值
        # 两个逆向问题里有一个完全不沾边，0.5 落在 [0.5, 0.7) 区间 -> warn
        self.assertEqual(result.verdict, "warn")

    def test_noncommittal_forces_zero_and_skips_embedding(self):
        calls: list[list[str]] = []

        def embedder(texts):
            calls.append(list(texts))
            return _fake_embed(texts)

        llm = FakeLLM(judge_payload(noncommittal=True, questions=["q1"], relevance=0.8))
        evaluator = am.AnswerEvaluator(llm=llm, embedder=embedder, use_embedding=True)
        result = evaluator.evaluate(question="问题", answer="未找到相关内容", contexts=())
        self.assertEqual(result.relevance, 0.0)
        self.assertEqual(result.relevance_mode, "noncommittal")
        self.assertEqual(calls, [])

    def test_no_embedding_falls_back_to_judge(self):
        llm = FakeLLM(judge_payload([claim("x")], relevance=0.66))
        evaluator = am.AnswerEvaluator(llm=llm, use_embedding=False)
        result = evaluator.evaluate(question="问题", answer="答案", contexts=[_ctx()])
        self.assertEqual(result.relevance_mode, "llm_judge")
        self.assertAlmostEqual(result.relevance, 0.66)

    def test_embedding_error_degrades_smoothly(self):
        llm = FakeLLM(judge_payload([claim("x")], relevance=0.72))
        evaluator = am.AnswerEvaluator(llm=llm, use_embedding=True)
        with mock.patch.object(
            am.embeddings, "embed_texts", side_effect=EmbeddingError("向量接口挂了")
        ), mock.patch.object(
            am.config, "embedding_current", return_value=SimpleNamespace(ready=True)
        ):
            result = evaluator.evaluate(question="问题", answer="答案", contexts=[_ctx()])
        self.assertEqual(result.relevance_mode, "llm_judge")
        self.assertAlmostEqual(result.relevance, 0.72)
        self.assertIn("EmbeddingError", result.error)

    def test_embedding_not_ready_skips_api(self):
        llm = FakeLLM(judge_payload([claim("x")], relevance=0.61))
        evaluator = am.AnswerEvaluator(llm=llm, use_embedding=True)
        with mock.patch.object(
            am.embeddings, "embed_texts", side_effect=AssertionError("不该被调用")
        ), mock.patch.object(
            am.config, "embedding_current", return_value=SimpleNamespace(ready=False)
        ):
            result = evaluator.evaluate(question="问题", answer="答案", contexts=[_ctx()])
        self.assertEqual(result.relevance_mode, "llm_judge")
        self.assertAlmostEqual(result.relevance, 0.61)

    def test_refusal_without_noncommittal_flag_still_zero_relevance(self):
        # Judge 有时把"资料里没有"标成 is_refusal=true、noncommittal=false。
        # 拒答没回答问题，相关性不能因为向量撞上就变成高分。
        payload = judge_payload(
            [claim("资料只写了署名 Satoshi Nakamoto")],
            noncommittal=False,
            is_refusal=True,
            relevance=0.93,
        )
        llm = FakeLLM(payload)
        evaluator = am.AnswerEvaluator(llm=llm, embedder=_fake_embed, use_embedding=True)
        result = evaluator.evaluate(
            question="问题", answer="资料里没给身份证号。", contexts=[_ctx()], expect_refusal=True
        )
        self.assertEqual(result.relevance, 0.0)
        self.assertEqual(result.relevance_mode, "noncommittal")
        self.assertAlmostEqual(result.faithfulness, 1.0)
        self.assertEqual(result.verdict, "pass")

    def test_effective_query_is_the_benchmark(self):
        llm = FakeLLM(judge_payload([claim("x")], questions=["q1"], relevance=0.0))
        evaluator = am.AnswerEvaluator(llm=llm, embedder=_fake_embed, use_embedding=True)
        result = evaluator.evaluate(
            question="ORIGINAL",
            answer="答案",
            contexts=[_ctx()],
            effective_query="EFFECTIVE",
        )
        self.assertEqual(result.relevance_mode, "reverse_embedding")
        self.assertAlmostEqual(result.relevance, 1.0)


# ---------------------------------------------------------------- 解析与降级


class TestParseAndFallback(unittest.TestCase):
    def test_strips_fences_and_surrounding_text(self):
        payload = (
            "好的，下面是结果：\n```json\n"
            + judge_payload([claim("上海最高")])
            + "\n```\n希望有帮助。"
        )
        outcome = am.parse_eval_response(payload)
        self.assertFalse(outcome.fallback)
        self.assertEqual(len(outcome.claims), 1)
        self.assertEqual(outcome.claims[0].verdict, "supported")

    def test_normalizes_illegal_fields(self):
        payload = json.dumps(
            {
                "claims": [
                    {"statement": "有效声明", "verdict": "MAYBE"},
                    {"statement": "   ", "verdict": "supported"},
                ],
                "generated_questions": ["q1", "q1", "q2", "q3", "q4"],
                "judge_relevance": 1.7,
                "missing_aspects": ["a", "a", "b", "c", "d", "e", "f"],
            },
            ensure_ascii=False,
        )
        outcome = am.parse_eval_response(payload, reverse_questions=3)
        self.assertEqual(len(outcome.claims), 1)
        self.assertEqual(outcome.claims[0].verdict, "unsupported")
        self.assertEqual(outcome.generated_questions, ("q1", "q2", "q3"))
        self.assertEqual(outcome.judge_relevance, 1.0)
        self.assertEqual(len(outcome.missing_aspects), 5)

    def test_negative_relevance_clamped(self):
        outcome = am.parse_eval_response('{"judge_relevance": -3}')
        self.assertEqual(outcome.judge_relevance, 0.0)

    def test_broken_json_is_fallback_not_exception(self):
        outcome = am.parse_eval_response("这不是 JSON")
        self.assertTrue(outcome.fallback)
        self.assertTrue(outcome.error)

    def test_judge_llm_error_becomes_fallback_result(self):
        evaluator = am.AnswerEvaluator(llm=BoomLLM(), use_embedding=False)
        result = evaluator.evaluate(question="q", answer="答案", contexts=[_ctx()])
        self.assertEqual(result.verdict, "fallback")
        self.assertTrue(result.fallback)
        self.assertIsNone(result.faithfulness)

    def test_judge_bad_payload_becomes_fallback_result(self):
        evaluator = am.AnswerEvaluator(llm=FakeLLM("我还是不返回 JSON 了"), use_embedding=False)
        result = evaluator.evaluate(question="q", answer="答案", contexts=[_ctx()])
        self.assertEqual(result.verdict, "fallback")

    def test_unexpected_judge_exception_is_contained(self):
        evaluator = am.AnswerEvaluator(llm=BoomLLM(RuntimeError("协议炸了")), use_embedding=False)
        result = evaluator.evaluate(question="q", answer="答案", contexts=[_ctx()])
        self.assertEqual(result.verdict, "fallback")


# ---------------------------------------------------------------- 聚合与评测集


def _record(
    *,
    verdict: str,
    faithfulness=None,
    relevance=None,
    supported=0,
    unsupported=0,
    contradicted=0,
    category="doc_fact",
    gen_latency=0.0,
    eval_latency=0.0,
    tokens=0,
    noncommittal=False,
    is_refusal=False,
    refusal_mismatch=False,
):
    return {
        "category": category,
        "verdict": verdict,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "supported_count": supported,
        "unsupported_count": unsupported,
        "contradicted_count": contradicted,
        "gen_latency_ms": gen_latency,
        "latency_ms": eval_latency,
        "tokens": tokens,
        "noncommittal": noncommittal,
        "is_refusal": is_refusal,
        "refusal_mismatch": refusal_mismatch,
    }


class TestSummarize(unittest.TestCase):
    def test_overall_and_by_category(self):
        records = [
            _record(verdict="pass", faithfulness=1.0, relevance=0.9, supported=3, gen_latency=100, eval_latency=50, tokens=200),
            _record(verdict="warn", faithfulness=0.5, relevance=0.6, supported=1, unsupported=1, gen_latency=200, eval_latency=70, tokens=300),
            _record(verdict="fail", faithfulness=0.0, relevance=0.4, unsupported=2, contradicted=1, category="table_agg", gen_latency=300, eval_latency=90, tokens=400),
            _record(verdict="skipped"),
            _record(verdict="fallback", gen_latency=50, eval_latency=10),
        ]
        summary = am.summarize_generation_results(records)
        ov = summary["overall"]

        self.assertEqual(ov["count"], 5)
        self.assertEqual(ov["evaluated_count"], 3)
        self.assertEqual(ov["skipped_count"], 1)
        self.assertEqual(ov["fallback_count"], 1)
        self.assertEqual((ov["pass_count"], ov["warn_count"], ov["fail_count"]), (1, 1, 1))
        self.assertAlmostEqual(ov["pass_rate"], 1 / 3)
        self.assertAlmostEqual(ov["avg_faithfulness"], 0.5)
        self.assertAlmostEqual(ov["avg_relevance"], (0.9 + 0.6 + 0.4) / 3)
        self.assertEqual(ov["total_claims"], 8)
        self.assertEqual(ov["supported_claims"], 4)
        self.assertEqual(ov["unsupported_claims"], 3)
        self.assertEqual(ov["contradicted_claims"], 1)
        self.assertEqual(ov["contradiction_case_count"], 1)
        self.assertAlmostEqual(ov["avg_gen_latency_ms"], 130.0)
        self.assertAlmostEqual(ov["avg_eval_latency_ms"], 44.0)
        self.assertAlmostEqual(ov["avg_eval_tokens"], 180.0)

        doc = summary["by_category"]["doc_fact"]
        self.assertEqual(doc["count"], 4)
        self.assertEqual(doc["evaluated_count"], 2)
        self.assertAlmostEqual(doc["pass_rate"], 0.5)
        self.assertAlmostEqual(doc["avg_faithfulness"], 0.75)
        self.assertAlmostEqual(doc["avg_relevance"], 0.75)

    def test_empty_records_do_not_crash(self):
        summary = am.summarize_generation_results([])
        self.assertEqual(summary["overall"]["count"], 0)
        self.assertIsNone(summary["overall"]["avg_faithfulness"])
        self.assertEqual(summary["by_category"], {})


class TestDataset(unittest.TestCase):
    def test_bundled_dataset_is_well_formed(self):
        cases = eg.load_dataset(ROOT / "eval" / "generation_dataset.json")
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({c["id"] for c in cases}), 16)
        self.assertEqual(
            {c["category"] for c in cases},
            {"doc_fact", "table_agg", "cross_source", "unanswerable"},
        )
        for case in cases:
            self.assertTrue(case["query"])
            self.assertIn(case["source_mode"], eg.SOURCE_MODES)
            # 三种问法齐全，策略矩阵才有意义
            self.assertTrue(case["variants"].get("colloquial"), case["id"])
            self.assertTrue(case["variants"].get("cross_lingual"), case["id"])
            if case["expect_refusal"]:
                self.assertEqual(case["category"], "unanswerable", case["id"])

    def test_dataset_rejects_bad_shape(self):
        bad = {
            "cases": [
                {"id": "a", "category": "x", "source_mode": "nope", "query": "q"},
            ]
        }
        tmp = ROOT / "eval" / "_tmp_bad_dataset.json"
        tmp.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                eg.load_dataset(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_dataset_rejects_duplicate_ids(self):
        bad = {
            "cases": [
                {"id": "a", "category": "doc_fact", "source_mode": "vault", "query": "q1"},
                {"id": "a", "category": "doc_fact", "source_mode": "vault", "query": "q2"},
            ]
        }
        tmp = ROOT / "eval" / "_tmp_dup_dataset.json"
        tmp.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                eg.load_dataset(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_filter_cases(self):
        cases = eg.load_dataset(ROOT / "eval" / "generation_dataset.json")
        only = eg.filter_cases(cases, categories=["unanswerable"])
        self.assertEqual({c["category"] for c in only}, {"unanswerable"})
        self.assertEqual(len(eg.filter_cases(cases, ids=["g01", "g07"])), 2)
        with self.assertRaises(ValueError):
            eg.filter_cases(cases, categories=["not_exists"])
        with self.assertRaises(ValueError):
            eg.filter_cases(cases, ids=["g99"])

    def test_expand_matrix_covers_all_configs(self):
        cases = eg.load_dataset(ROOT / "eval" / "generation_dataset.json")[:2]
        runtime = eg.expand_matrix(cases)
        self.assertEqual(len(runtime), 2 * len(eg.MATRIX_CONFIGS))
        self.assertEqual(
            {item["config"] for item in runtime},
            {"baseline", "basic_rewrite", "colloquial", "cross_lingual"},
        )
        baseline = next(item for item in runtime if item["config"] == "baseline")
        self.assertFalse(baseline["rewrite"])
        self.assertEqual(baseline["query"], cases[0]["query"])
        colloquial = next(
            item for item in runtime
            if item["config"] == "colloquial" and item["id"] == cases[0]["id"]
        )
        self.assertEqual(colloquial["query"], cases[0]["variants"]["colloquial"])


# ---------------------------------------------------------------- 流水线冒烟


class TestEvaluatePredictions(unittest.TestCase):
    def test_records_carry_meta_and_refusal_mismatch(self):
        predictions = [
            {
                "id": "g01",
                "category": "doc_fact",
                "run_id": "g01@baseline",
                "config": "baseline",
                "variant": "original",
                "query": "额度多少？",
                "effective_query": "额度多少？",
                "answer": "最高授信额度为 10000 元。",
                "contexts": [
                    {"tool_name": "read_document", "args_summary": "", "content": "最高授信额度为10000元"}
                ],
                "expect_refusal": False,
                "gen_latency_ms": 1200.0,
                "steps": 3,
                "tool_calls": 1,
                "gen_tokens": 900,
            },
            {
                "id": "g14",
                "category": "unanswerable",
                "run_id": "g14@baseline",
                "config": "baseline",
                "variant": "original",
                "query": "火星基地预算？",
                "answer": "资料里没有提到该预算。",
                "contexts": [],
                "expect_refusal": True,
                "gen_latency_ms": 800.0,
            },
        ]
        judged = {
            "最高授信额度为 10000 元。": judge_payload([claim("最高授信额度 10000 元")]),
            "资料里没有提到该预算。": judge_payload(
                noncommittal=True, is_refusal=True, questions=[], relevance=0.05
            ),
        }

        class ByAnswerLLM:
            def __init__(self):
                self.calls = 0

            def __call__(self, messages, tools=None):
                self.calls += 1
                text = messages[-1]["content"]
                for answer, payload in judged.items():
                    if answer in text:
                        return _response(payload)
                return _response(judge_payload([]))

        records, summary = eg.evaluate_predictions(
            predictions, evaluator=am.AnswerEvaluator(llm=ByAnswerLLM(), use_embedding=False)
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], "g01")
        self.assertEqual(records[0]["config"], "baseline")
        self.assertAlmostEqual(records[0]["faithfulness"], 1.0)
        self.assertFalse(records[0]["refusal_mismatch"])
        self.assertEqual(records[1]["verdict"], "pass")
        self.assertFalse(records[1]["refusal_mismatch"])
        self.assertEqual(summary["overall"]["count"], 2)
        self.assertIn("doc_fact", summary["by_category"])

    def test_refusal_on_answerable_case_is_flagged(self):
        # 该答却拒答：非零声明数为 0，但被判为拒答，且与预期不符
        class AlwaysRefuse:
            def __call__(self, messages, tools=None):
                return _response(
                    judge_payload(noncommittal=True, is_refusal=True, questions=[], relevance=0.0)
                )

        predictions = [
            {
                "id": "g07",
                "category": "table_agg",
                "query": "哪个城市最高？",
                "answer": "抱歉，无法确定。",
                "contexts": [],
                "expect_refusal": False,
            }
        ]
        records, _ = eg.evaluate_predictions(
            predictions, evaluator=am.AnswerEvaluator(llm=AlwaysRefuse(), use_embedding=False)
        )
        self.assertEqual(records[0]["verdict"], "fail")
        self.assertTrue(records[0]["refusal_mismatch"])

    def test_report_renders_tables_without_leaking_keys(self):
        predictions = [
            {
                "id": "g01",
                "category": "doc_fact",
                "run_id": "g01@baseline",
                "config": "baseline",
                "query": "q",
                "answer": "a",
                "contexts": [],
                "expect_refusal": False,
            }
        ]
        records, _ = eg.evaluate_predictions(
            predictions,
            evaluator=am.AnswerEvaluator(
                llm=FakeLLM(judge_payload([claim("x", "unsupported", "")])),
                use_embedding=False,
            ),
        )
        report = eg.format_generation_report(
            dataset_path=ROOT / "eval" / "generation_dataset.json",
            predictions=predictions,
            records=records,
        )
        self.assertIn("忠实度", report)
        self.assertIn("逐题评分明细", report)
        self.assertIn("Bad Case", report)
        self.assertNotIn("sk-", report)
        md = eg.format_markdown_report(
            dataset_path=ROOT / "eval" / "generation_dataset.json",
            predictions=predictions,
            records=records,
        )
        self.assertIn("| 配置 |", md)


class TestPipelineSmoke(unittest.TestCase):
    """真实 Agent + 真实 run_sql + 假模型：验证隔离挂载与上下文提取确实连通。"""

    def tearDown(self) -> None:
        data_tools.use_dataset(data_tools.DEFAULT_KEY)

    def test_generate_prediction_runs_agent_and_cleans_up(self):
        before_keys = set(data_tools._conns)
        scripts = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "run_sql",
                            "arguments": json.dumps({"sql": "SELECT COUNT(*) AS n FROM dataset"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "这张表共有 320 行订单。"},
        ]
        case = {
            "id": "smoke01",
            "category": "table_agg",
            "source_mode": "sales",
            "query": "这张表有多少行订单？",
            "variants": {},
            "expect_refusal": False,
            "reference_hints": ["320"],
            "note": "",
        }
        prediction = eg.generate_prediction(case, rewrite=False, llm=ScriptedLLM(scripts))

        self.assertEqual(prediction["answer"], "这张表共有 320 行订单。")
        self.assertEqual(prediction["tool_calls"], 1)
        self.assertEqual(prediction["context_count"], 1)
        self.assertIn("320", prediction["contexts"][0]["content"])
        self.assertEqual(prediction["error"], "")
        # 会话用完就撤：不能给别的用例留一条活连接
        self.assertEqual(set(data_tools._conns), before_keys)

        records, summary = eg.evaluate_predictions(
            [prediction],
            evaluator=am.AnswerEvaluator(
                llm=FakeLLM(judge_payload([claim("表中共 320 行")])), use_embedding=False
            ),
        )
        self.assertEqual(records[0]["verdict"], "pass")
        self.assertEqual(summary["overall"]["count"], 1)

    def test_llm_failure_becomes_abort_answer(self):
        case = {
            "id": "smoke02",
            "category": "table_agg",
            "source_mode": "sales",
            "query": "随便问问",
            "variants": {},
            "expect_refusal": False,
            "reference_hints": [],
            "note": "",
        }
        prediction = eg.generate_prediction(case, rewrite=False, llm=BoomLLM())
        # Agent 自己接住了模型错误，把它写成中止话术；评测侧应当识别并跳过
        self.assertIn("模型调用失败", prediction["answer"])
        records, _ = eg.evaluate_predictions(
            [prediction],
            evaluator=am.AnswerEvaluator(llm=FakeLLM("{}"), use_embedding=False),
        )
        self.assertEqual(records[0]["verdict"], "skipped")

    def test_setup_failure_is_recorded_not_raised(self):
        case = {
            "id": "smoke03",
            "category": "table_agg",
            "source_mode": "sales",
            "query": "随便问问",
            "variants": {},
            "expect_refusal": False,
            "reference_hints": [],
            "note": "",
        }
        with mock.patch.object(
            eg, "prepare_sources", side_effect=RuntimeError("挂载失败")
        ):
            prediction = eg.generate_prediction(case, rewrite=False, llm=FakeLLM("{}"))
        self.assertIn("挂载失败", prediction["error"])
        self.assertEqual(prediction["answer"], "")
        self.assertEqual(set(data_tools._conns), set())

        records, _ = eg.evaluate_predictions(
            [prediction],
            evaluator=am.AnswerEvaluator(llm=FakeLLM("{}"), use_embedding=False),
        )
        self.assertEqual(records[0]["verdict"], "skipped")


if __name__ == "__main__":
    unittest.main()
