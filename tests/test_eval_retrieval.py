"""离线评测脚本（eval/eval_retrieval.py）的指标计算与数据集校验单测。

纯离线运行，不发网络请求，也不依赖 data/vault 里的真实 PDF。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.query_rewrite import RewriteResult, RewriteStrategy  # noqa: E402
from app.readers import Doc  # noqa: E402
from eval import eval_retrieval as er  # noqa: E402


class TestRetrievalMetrics(unittest.TestCase):
    def test_reciprocal_rank_positions_and_cutoff(self):
        relevant = {("a.pdf", 3), ("b.pdf", 2)}
        self.assertAlmostEqual(er.reciprocal_rank([("a.pdf", 3)], relevant, k=3), 1.0)
        self.assertAlmostEqual(
            er.reciprocal_rank([("x.pdf", 1), ("b.pdf", 2)], relevant, k=3), 0.5
        )
        self.assertAlmostEqual(
            er.reciprocal_rank(
                [("x.pdf", 1), ("y.pdf", 1), ("a.pdf", 3), ("b.pdf", 2)],
                relevant,
                k=3,
            ),
            1.0 / 3.0,
        )
        # 排在第 4 名时，k=3 应判未命中，k=10 应返回 1/4
        ranked = [("x.pdf", 1), ("y.pdf", 1), ("z.pdf", 1), ("b.pdf", 2)]
        self.assertEqual(er.reciprocal_rank(ranked, relevant, k=3), 0.0)
        self.assertAlmostEqual(er.reciprocal_rank(ranked, relevant, k=10), 0.25)
        self.assertEqual(er.reciprocal_rank([], relevant, k=3), 0.0)
        self.assertEqual(er.reciprocal_rank([("a.pdf", 3)], relevant, k=0), 0.0)

    def test_reciprocal_rank_picks_earliest_when_multiple_relevant_hit(self):
        relevant = {("a.pdf", 1), ("a.pdf", 2)}
        ranked = [("x.pdf", 1), ("a.pdf", 2), ("a.pdf", 1)]
        self.assertAlmostEqual(er.reciprocal_rank(ranked, relevant, k=3), 0.5)

    def test_hit_and_recall_at_k(self):
        relevant = {("a.pdf", 1), ("a.pdf", 2)}
        ranked_partial = [("x.pdf", 9), ("a.pdf", 2), ("y.pdf", 3)]
        ranked_full = [("a.pdf", 1), ("x.pdf", 9), ("a.pdf", 2)]

        self.assertEqual(er.hit_at_k(ranked_partial, relevant, k=1), 0.0)
        self.assertEqual(er.hit_at_k(ranked_partial, relevant, k=3), 1.0)
        self.assertAlmostEqual(er.recall_at_k(ranked_partial, relevant, k=3), 0.5)
        self.assertAlmostEqual(er.recall_at_k(ranked_full, relevant, k=3), 1.0)
        self.assertEqual(er.recall_at_k([("a.pdf", 1)], set(), k=3), 0.0)

    def test_summarize_results_overall_and_by_category(self):
        records = [
            {
                "id": "q1",
                "category": "exact",
                "relevant": [("a.pdf", 1)],
                "retrieved": [("a.pdf", 1), ("b.pdf", 2)],
                "latency_ms": 4.0,
                "degraded": False,
            },
            {
                "id": "q2",
                "category": "exact",
                "relevant": [("a.pdf", 2)],
                "retrieved": [("x.pdf", 1), ("a.pdf", 2)],
                "latency_ms": 6.0,
                "degraded": False,
            },
            {
                "id": "q3",
                "category": "semantic",
                "relevant": [("b.pdf", 3)],
                "retrieved": [],
                "latency_ms": 10.0,
                "degraded": True,
            },
            {
                "id": "q4",
                "category": "semantic",
                "relevant": [("b.pdf", 4)],
                "retrieved": [("x.pdf", 1), ("y.pdf", 2), ("z.pdf", 3), ("b.pdf", 4)],
                "latency_ms": 20.0,
                "degraded": False,
            },
        ]

        summary = er.summarize_results(records, top_k=3, recall_k=10)
        ov = summary["overall"]
        self.assertEqual(ov["count"], 4)
        self.assertAlmostEqual(ov["mrr_at_k"], (1.0 + 0.5 + 0.0 + 0.0) / 4.0)
        self.assertAlmostEqual(ov["mrr_at_recall_k"], (1.0 + 0.5 + 0.0 + 0.25) / 4.0)
        self.assertAlmostEqual(ov["hit_at_1"], 0.25)
        self.assertAlmostEqual(ov["hit_at_k"], 0.5)
        self.assertAlmostEqual(ov["hit_at_recall_k"], 0.75)
        self.assertEqual(ov["zero_hit_count"], 2)
        self.assertEqual(ov["empty_count"], 1)
        self.assertEqual(ov["degraded_count"], 1)
        self.assertAlmostEqual(ov["avg_latency_ms"], 10.0)

        exact = summary["by_category"]["exact"]
        self.assertEqual(exact["count"], 2)
        self.assertAlmostEqual(exact["mrr_at_k"], 0.75)
        self.assertAlmostEqual(exact["hit_at_k"], 1.0)
        self.assertEqual(exact["zero_hit_count"], 0)

        sem = summary["by_category"]["semantic"]
        self.assertEqual(sem["count"], 2)
        self.assertAlmostEqual(sem["mrr_at_k"], 0.0)
        self.assertAlmostEqual(sem["mrr_at_recall_k"], 0.125)
        self.assertEqual(sem["empty_count"], 1)
        self.assertEqual(sem["degraded_count"], 1)


class TestDatasetAndValidation(unittest.TestCase):
    def test_bundled_dataset_loads_and_is_well_formed(self):
        cases = er.load_dataset(ROOT / "eval" / "retrieval_dataset.json")
        self.assertEqual(len(cases), 24)
        ids = [c["id"] for c in cases]
        self.assertEqual(len(set(ids)), 24)
        categories = {c["category"] for c in cases}
        self.assertEqual(
            categories, {"exact", "multi_term", "semantic", "cross_lingual"}
        )
        for c in cases:
            self.assertTrue(c["query"])
            self.assertTrue(c["relevant"])

    def test_validate_against_docs(self):
        docs = [Doc(name="a.txt", kind="txt", text="第一段内容")]
        cases = [
            {
                "id": "q1",
                "category": "exact",
                "query": "第一段",
                "relevant": [("a.txt", 1)],
            }
        ]
        er.validate_against_docs(cases, docs)

        with self.assertRaises(ValueError):
            er.validate_against_docs(
                [
                    {
                        "id": "q_bad_doc",
                        "category": "exact",
                        "query": "x",
                        "relevant": [("missing.txt", 1)],
                    }
                ],
                docs,
            )

        with self.assertRaises(ValueError):
            er.validate_against_docs(
                [
                    {
                        "id": "q_bad_part",
                        "category": "exact",
                        "query": "x",
                        "relevant": [("a.txt", 9)],
                    }
                ],
                docs,
            )

    def test_filter_cases_by_category(self):
        cases = er.load_dataset(ROOT / "eval" / "retrieval_dataset.json")
        cross = er.filter_cases_by_category(cases, ["cross_lingual"])
        self.assertEqual(len(cross), 5)
        self.assertTrue(all(c["category"] == "cross_lingual" for c in cross))
        self.assertEqual(len(er.filter_cases_by_category(cases, [])), 24)
        with self.assertRaises(ValueError):
            er.filter_cases_by_category(cases, ["not_exists"])

    def test_select_rewrite_query_priority(self):
        rw_en = RewriteResult(
            original="原问题",
            strategy=RewriteStrategy.QUERY_EXPANSION,
            rewritten_query="改写问题",
            search_keywords=("中文 关键词",),
            english_keywords=("English domain terms",),
        )
        rw_kw = RewriteResult(
            original="原问题",
            strategy=RewriteStrategy.QUERY_EXPANSION,
            rewritten_query="改写问题",
            search_keywords=("中文 关键词",),
        )
        rw_plain = RewriteResult(
            original="原问题",
            strategy=RewriteStrategy.BASIC_REWRITE,
            rewritten_query="改写问题",
        )
        self.assertEqual(er.select_rewrite_query(rw_en), "English domain terms")
        self.assertEqual(er.select_rewrite_query(rw_kw), "中文 关键词")
        self.assertEqual(er.select_rewrite_query(rw_plain), "改写问题")


if __name__ == "__main__":
    unittest.main()
