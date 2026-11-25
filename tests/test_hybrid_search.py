"""混合检索：BM25、向量召回、RRF 融合、整体行为。

这里用**假 Embedder**（正交语义簇）而不是真接口：单测要的是确定性，
"违约/赔偿"必须稳定落在同一维，"量子/叠加/纠缠"落在另一维。
真接口每次返回的数值会微变，断言"排第 1"就成了抽奖。

四类断言，分别盯一个设计决策：
  1. BM25 的 IDF 必须非负（只有 1~2 段的小文档集不能把命中段打成负分）；
  2. 父段取 MaxSim 而不是平均（两千字里只有三百字相关时不能被抓平均）；
  3. RRF 让"两路都进榜"的段落严格优于"只有一路第一"；
  4. 向量这一路任何异常都必须静默降级，检索整体仍然可用。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from app import bm25, embeddings, hybrid_search, vault  # noqa: E402
from app.embeddings import EmbeddingError  # noqa: E402
from app.hybrid_search import hybrid_search as run_hybrid  # noqa: E402
from app.readers import Doc  # noqa: E402
from app.tools import data_tools as dt  # noqa: E402
from app.tools import doc_tools, registry  # noqa: E402
from _isolation import isolate_embeddings, isolate_uploads  # noqa: E402

KEY = "ut_hybrid"

CONTRACT = ("违约", "赔偿", "罚金", "履约", "赔付", "损害", "条款")
CHAIN = ("区块链", "哈希", "篡改", "共识", "账本", "交易")
QUANTUM = ("量子", "叠加", "纠缠")


def fake_embed(texts):
    """四维正交语义簇。正交 -> 余弦相似度非 0 即 1，断言不用留误差。"""
    rows = []
    for text in texts:
        text = text or ""
        if any(k in text for k in CONTRACT):
            vec = [1.0, 0.0, 0.0, 0.0]
        elif any(k in text for k in CHAIN):
            vec = [0.0, 1.0, 0.0, 0.0]
        elif any(k in text for k in QUANTUM):
            vec = [0.0, 0.0, 1.0, 0.0]
        else:
            vec = [0.0, 0.0, 0.0, 1.0]
        rows.append(vec)
    return np.asarray(rows, dtype=np.float32)


def doc(name: str, text: str) -> Doc:
    return Doc(name=name, kind="txt", text=text)


class _StoreCase(unittest.TestCase):
    """每个用例一个临时储藏区：向量库落在临时目录，不碰真实 data/vault。"""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="hs-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        patcher = mock.patch.object(vault, "VAULT_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        embeddings.clear_caches()
        hybrid_search.set_hybrid_enabled(True)


def setUpModule() -> None:
    # 本机 .env 配了真实向量模型：不隔离就会在测试里发真实 HTTP 请求
    isolate_embeddings()
    isolate_uploads("uploads-hybrid-")
    if not HAS_APPTEST:
        return
    from app import chats

    # 界面冒烟用例会把 app.py 整个跑一遍，历史目录也要挪开
    root = Path(tempfile.mkdtemp(prefix="chats-hybrid-"))
    patcher = mock.patch.object(chats, "CHAT_DIR", root)
    patcher.start()
    unittest.addModuleCleanup(shutil.rmtree, root, ignore_errors=True)
    unittest.addModuleCleanup(patcher.stop)


# ---------------------------------------------------------------- BM25

class TestBM25(unittest.TestCase):
    def test_idf_never_negative_on_tiny_corpus(self):
        # 只传一份文档、两段都含该词：原始公式在这里会算出负值
        for n, df in ((1, 1), (2, 2), (3, 3)):
            self.assertGreater(bm25._idf(n, df), 0.0, (n, df))

    def test_rare_term_outranks_common_term(self):
        docs = [
            doc("a.txt", "krypton zzz"),          # 只有它含 krypton
            doc("b.txt", "report vvv"),
            doc("c.txt", "report www"),
            doc("d.txt", "report uuu"),
        ]
        result = bm25.bm25_search("krypton report", docs)
        self.assertEqual(result.hits[0].doc.name, "a.txt")
        self.assertGreater(result.hits[0].score, result.hits[1].score)

    def test_length_normalization_prefers_short_chunk(self):
        docs = [
            doc("short.txt", "krypton"),
            doc("long.txt", "krypton " + "filler " * 100),
        ]
        result = bm25.bm25_search("krypton", docs)
        scores = {h.doc.name: h.score for h in result.hits}
        self.assertGreater(scores["short.txt"], scores["long.txt"])

    def test_jieba_and_pure_python_fallback_both_work(self):
        docs = [doc("a.txt", "区块链的共识机制需要多方确认")]
        for use_jieba in (True, False):
            result = bm25.bm25_search("区块链 共识", docs, use_jieba=use_jieba)
            self.assertTrue(result.hits, f"use_jieba={use_jieba} 应该能召回")
        self.assertTrue(bm25.tokenize("区块链", use_jieba=False))

    def test_nonsense_is_rejected(self):
        docs = [doc("a.txt", "第一章 赔偿说明（资产是否存在需核实）" + "本节介绍背景。" * 200)]
        self.assertEqual(bm25.bm25_search("不存在的词甲乙丙", docs).hits, ())

    def test_scattered_bigrams_rejected(self):
        text = "共识需要多方确认。\n" + "本节介绍背景。" * 200 + "\n激励机制的设计。"
        self.assertEqual(bm25.bm25_search("共识机制", [doc("d.txt", text)]).hits, ())

    def test_rewrite_with_extra_particle_still_recalled_as_fuzzy(self):
        docs = [doc("g.txt", "区块链的创新应用\n" + "案例集。" * 100)]
        result = bm25.bm25_search("区块链创新应用", docs)
        self.assertTrue(result.hits)
        self.assertTrue(result.fuzzy, "字面降级必须标注出来")


# ---------------------------------------------------------------- 向量


class TestVectorSearch(_StoreCase):
    def test_pure_semantic_hit_without_literal_overlap(self):
        para = "未按期履约的损害赔偿责任与罚金"
        docs = [doc("other.txt", "量子计算入门"), doc("contract.txt", para)]
        # 先确认这一遍 BM25 什么都找不到：能召回就只能是语义路
        self.assertEqual(bm25.bm25_search("违约怎么赔", docs).hits, ())
        hits = hybrid_search.vector_search("违约怎么赔", docs, embedder=fake_embed)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc.name, "contract.txt")

    def test_maxsim_picks_the_relevant_tail_not_the_head(self):
        text = "背景说明" * 400 + "未按期履约需要支付赔偿金"
        docs = [doc("long.txt", text)]
        hits = hybrid_search.vector_search("违约怎么赔", docs, embedder=fake_embed)
        self.assertTrue(hits)
        hit = hits[0]
        self.assertGreater(hit.best_offset, 1000, "胜出子块应落在段尾")
        self.assertIn("赔偿", hit.best_subchunk)

    def test_min_score_filters_unrelated(self):
        docs = [doc("chain.txt", "区块链的哈希与账本")]
        hits = hybrid_search.vector_search(
            "量子叠加态", docs, embedder=fake_embed, min_score=0.35
        )
        self.assertEqual(hits, [])

    def test_dimension_mismatch_is_filtered_not_crashed(self):
        """库里混着旧模型的 8 维向量时，不能抛 shapes not aligned。"""
        docs = [doc("a.txt", "未按期履约的赔偿责任"), doc("b.txt", "另一段与内容无关的话")]
        store = embeddings.VectorStore()
        _, vec_map, _ = store.ensure_indexed(docs, embedder=fake_embed, model_key="custom")

        stale = [sc for sc in embeddings.split_subchunks(docs[1].text)]
        if stale:
            bad_hash = embeddings.chunk_hash(stale[0][1])
            store.put_many([(bad_hash, np.ones(8, dtype=np.float32))], "custom")

        hits = hybrid_search.vector_search(
            "违约怎么赔", docs, store=store, embedder=fake_embed, model_key="custom"
        )
        self.assertTrue(hits)
        self.assertEqual([h.doc.name for h in hits], ["a.txt"])


# ---------------------------------------------------------------- RRF


class TestRRFFusion(unittest.TestCase):
    def test_exact_formula(self):
        scores = hybrid_search.rrf_fuse(
            [(["a", "b"], 1.0)], key_fn=lambda x: x, k=60
        )
        self.assertAlmostEqual(scores["a"], 1 / 61)
        self.assertAlmostEqual(scores["b"], 1 / 62)

    def test_two_second_places_beat_one_first_place(self):
        scores = hybrid_search.rrf_fuse(
            [(["a", "c"], 1.0), (["b", "c"], 1.0)], key_fn=lambda x: x, k=60
        )
        self.assertAlmostEqual(scores["c"], 2 / 62)
        self.assertGreater(scores["c"], scores["a"])

    def test_weight_controls_ranking(self):
        scores = hybrid_search.rrf_fuse(
            [(["a"], 1.0), (["b"], 0.0)], key_fn=lambda x: x, k=60
        )
        self.assertNotIn("b", scores)
        self.assertAlmostEqual(scores["a"], 1 / 61)

    def test_zero_and_negative_weight_are_ignored(self):
        scores = hybrid_search.rrf_fuse(
            [(["a"], 0.0), (["b"], -1.0)], key_fn=lambda x: x, k=60
        )
        self.assertEqual(scores, {})

    def test_duplicate_key_counts_once_per_list(self):
        scores = hybrid_search.rrf_fuse(
            [(["a", "a"], 1.0)], key_fn=lambda x: x, k=60
        )
        self.assertAlmostEqual(scores["a"], 1 / 61)

    def test_three_lists_supported(self):
        scores = hybrid_search.rrf_fuse(
            [(["a"], 1.0), (["a"], 1.0), (["a"], 1.0)], key_fn=lambda x: x, k=10
        )
        self.assertAlmostEqual(scores["a"], 3 / 11)


# ---------------------------------------------------------------- 端到端


class TestHybridSearchIntegration(_StoreCase):
    def _docs(self):
        # b.txt 刻意用"罚金/履约"表达违约后果，一个字都不与查询重合：
        # 它只可能被语义路捞到，这样"两路互补"才是真的被验证了。
        return [
            doc("a.txt", "错误码 ERR-9012 出现在网关日志里"),
            doc("b.txt", "未按期履约须支付罚金并补偿对方损失"),
            doc("c.txt", "ERR-9012 对应条款为赔偿责任"),
        ]

    def test_complementary_recall_beats_either_route_alone(self):
        docs = self._docs()
        result = run_hybrid("ERR-9012 赔偿", docs, embedder=fake_embed, top_k=3)
        names = [c.doc.name for c in result.candidates]
        self.assertEqual(names[0], "c.txt", "双路都命中的段落必须排第一")
        self.assertEqual(set(names), {"a.txt", "b.txt", "c.txt"}, "两条路互补，三段都该进候选")

        only_bm25 = run_hybrid("ERR-9012 赔偿", docs, mode="bm25")
        self.assertNotIn("b.txt", [c.doc.name for c in only_bm25.candidates])

        only_vec = run_hybrid("ERR-9012 赔偿", docs, mode="vector", embedder=fake_embed)
        self.assertNotIn("a.txt", [c.doc.name for c in only_vec.candidates])

    def test_both_routes_marked_on_card(self):
        result = run_hybrid("ERR-9012 赔偿", self._docs(), embedder=fake_embed)
        top = result.candidates[0]
        self.assertEqual(top.sources, ("bm25", "vector"))
        self.assertTrue(top.has_exact_lexical)
        self.assertIn("ERR-9012", top.snippet)

    def test_embedding_error_degrades_silently(self):
        def boom(texts):
            raise EmbeddingError("模拟限流")

        result = run_hybrid("ERR-9012", self._docs(), embedder=boom)
        self.assertTrue(result.vector_degraded)
        self.assertIn("限流", result.degrade_reason)
        self.assertTrue(result.candidates, "降级后 BM25 结果必须还在")
        self.assertEqual(
            {c.doc.name for c in result.candidates}, {"a.txt", "c.txt"}
        )

    def test_timeout_degrades_too(self):
        def boom(texts):
            raise TimeoutError("timed out")

        result = run_hybrid("赔偿", self._docs(), embedder=boom)
        self.assertTrue(result.vector_degraded)
        self.assertTrue(result.candidates)

    def test_switch_off_calls_embedder_zero_times(self):
        calls: list[int] = []

        def counting(texts):
            calls.append(len(texts))
            return fake_embed(texts)

        hybrid_search.set_hybrid_enabled(False)
        try:
            result = run_hybrid("赔偿责任", self._docs(), embedder=counting)
        finally:
            hybrid_search.set_hybrid_enabled(True)
        self.assertEqual(calls, [], "关掉开关就不该再算向量")
        self.assertFalse(result.used_vector)

    def test_candidates_never_leak_across_sessions(self):
        session_a = [doc("a.txt", "区块链的哈希保证账本不可篡改")]
        session_b = [doc("b.txt", "未按期履约的赔偿责任")]
        result = run_hybrid("赔偿责任", session_a, embedder=fake_embed)
        self.assertNotIn("b.txt", [c.doc.name for c in result.candidates])


# ---------------------------------------------------------------- 工具输出


class TestSearchDocumentTool(_StoreCase):
    def setUp(self) -> None:
        super().setUp()
        dt.clear_documents(KEY)
        dt.use_dataset(KEY)
        self.addCleanup(dt.clear_documents, KEY)
        dt.add_document(
            "contract.txt", "未按期履约的损害赔偿责任与罚金。".encode("utf-8"), key=KEY
        )
        dt.add_document(
            "log.txt", "错误码 ERR-9012 出现在网关日志里。".encode("utf-8"), key=KEY
        )

    def _call_with_embedder(self, keyword: str) -> str:
        def patched(query, docs, **kwargs):
            return run_hybrid(query, docs, embedder=fake_embed, **kwargs)

        with mock.patch.object(doc_tools, "hybrid_search", patched):
            return registry.call("search_document", {"keyword": keyword})

    def test_hybrid_output_mentions_fusion(self):
        text = self._call_with_embedder("ERR-9012 赔偿")
        self.assertIn("混合检索（向量+BM25+RRF）", text)
        self.assertIn("向量+BM25双命中", text)
        self.assertIn("read_document", text)

    def test_pure_semantic_card(self):
        text = self._call_with_embedder("违约怎么赔付")
        self.assertIn("语义检索命中", text)
        self.assertIn("语义匹配（相似度", text)

    def test_bm25_only_output_is_unchanged_and_compact(self):
        # 本模块的向量配置已被清空 -> 走纯 BM25，输出格式必须和改造前一致
        text = registry.call("search_document", {"keyword": "赔偿"})
        self.assertNotIn("混合检索", text)
        self.assertIn("命中「赔偿」", text)
        self.assertIn("命中 1 个词", text)
        self.assertLess(len(text), 1200)


# ---------------------------------------------------------------- 界面


try:
    from streamlit.testing.v1 import AppTest

    HAS_APPTEST = True
except Exception:  # noqa: BLE001
    HAS_APPTEST = False


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit >= 1.28")
class TestUIHybridSmoke(unittest.TestCase):
    def test_sidebar_has_hybrid_switch(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        app.run()
        self.assertEqual(len(app.exception), 0, list(app.exception))
        labels = [c.label for c in app.sidebar.checkbox]
        self.assertIn("启用混合检索（向量+BM25）", labels)
        self.assertIn("启用查询改写", labels)


if __name__ == "__main__":
    unittest.main()
