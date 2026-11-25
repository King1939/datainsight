"""DECOMPOSITION 英文召回评测的纯函数测试。

只测不花钱的部分：RRF 融合、每组对照发哪几路查询、指标计算。
测这些是因为它们是"四组对照能不能比"的前提——融合写错了，
四组之间的差异就不再是英文词带来的，而是排序 bug 带来的。

跑法：
    python -m unittest tests/test_eval_decompose.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from app.query_rewrite import RetrievalUnit  # noqa: E402
from eval_decompose import _channel_queries, _metrics, _rrf_merge  # noqa: E402


class TestRRFMerge(unittest.TestCase):
    def test_two_channels_add_up(self):
        a = [("x.pdf", 1), ("y.pdf", 2)]
        b = [("y.pdf", 2), ("x.pdf", 1)]

        merged = _rrf_merge([a, b], k=60)

        # 两路都命中的候选分数相加，两条都保留（顺序见下一个用例）
        self.assertEqual(set(merged), {("x.pdf", 1), ("y.pdf", 2)})

    def test_candidate_in_both_channels_beats_single_channel_top(self):
        both = [("both.pdf", 9), ("other.pdf", 1)]
        single = [("single.pdf", 1)]

        merged = _rrf_merge([both, single], k=60)

        self.assertEqual(merged[0], ("both.pdf", 9))

    def test_tie_keeps_first_seen_order(self):
        merged = _rrf_merge([[("a.pdf", 1)], [("b.pdf", 1)]], k=60)

        # 并列时按先出现的先排：这正是需要额外看"任一路命中"的原因
        self.assertEqual(merged, [("a.pdf", 1), ("b.pdf", 1)])

    def test_empty_input(self):
        self.assertEqual(_rrf_merge([], k=60), [])


class TestChannelQueries(unittest.TestCase):
    def _unit(self, terms=()):
        return RetrievalUnit(id="Q1", query="中文子查询", english_terms=terms)

    def test_baseline_only_chinese(self):
        self.assertEqual(_channel_queries("A_baseline", self._unit(("spv",)), ""), ["中文子查询"])

    def test_per_unit_adds_own_english_terms(self):
        queries = _channel_queries("C_per_unit", self._unit(("spv proof", "merkle")), "GLOBAL")

        self.assertEqual(queries, ["中文子查询", "spv proof merkle"])

    def test_per_unit_without_terms_stays_chinese_only(self):
        self.assertEqual(_channel_queries("C_per_unit", self._unit(), "GLOBAL"), ["中文子查询"])

    def test_global_pack_replaces_per_unit_terms(self):
        queries = _channel_queries("B_global_pack", self._unit(("spv",)), "GLOBAL PACK")

        self.assertEqual(queries, ["中文子查询", "GLOBAL PACK"])

    def test_english_only_drops_chinese(self):
        self.assertEqual(
            _channel_queries("D_english_only", self._unit(("spv proof",)), ""), ["spv proof"]
        )

    def test_english_only_without_terms_has_no_channel(self):
        self.assertEqual(_channel_queries("D_english_only", self._unit(), ""), [])


class TestMetrics(unittest.TestCase):
    def test_hit_and_mrr_and_recall(self):
        ranked = [("a.pdf", 1), ("t.pdf", 3), ("t.pdf", 4)]
        relevant = {("t.pdf", 3), ("t.pdf", 4)}

        metrics = _metrics(ranked, relevant, top_k=3, recall_k=10)

        self.assertEqual(metrics["rank_in_top_k"], 2)
        self.assertEqual(metrics["hit_at_k"], 1.0)
        self.assertAlmostEqual(metrics["mrr_at_k"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_k"], 1.0)

    def test_miss_outside_top_k_still_counted_in_recall_k(self):
        ranked = [("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 3), ("t.pdf", 5)]
        relevant = {("t.pdf", 5)}

        metrics = _metrics(ranked, relevant, top_k=3, recall_k=10)

        self.assertIsNone(metrics["rank_in_top_k"])
        self.assertEqual(metrics["hit_at_k"], 0.0)
        self.assertEqual(metrics["rank_in_recall_k"], 4)


if __name__ == "__main__":
    unittest.main()
