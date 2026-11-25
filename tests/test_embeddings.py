"""向量化层：切块、归一化、HTTP 协议、缓存库。

全离线：HTTP 用替身，不联网、不花钱、不需要 Key。真机联调另有一套手动脚本，
不在回归里——单测一旦依赖"有网、有额度、服务没限流"，它就不是回归而是抽奖了。

四个必须守住的点，每条都对应一个曾经踩到或差点踩到的坑：
  1. 服务端乱序返回 `data[].index` 时要能还原顺序（OpenAI 规范允许乱序）；
  2. 空串绝不能发出去（扫描件的空块会让接口直接 400）；
  3. 向量库用完立刻关（Windows 下测试清理临时目录会报 WinError 32）；
  4. 中途失败时，已算好的批次要留在盘里（断点续传）。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from app import config, embeddings, vault  # noqa: E402
from app.embeddings import EmbeddingConfig, EmbeddingError  # noqa: E402
from app.readers import Doc  # noqa: E402
from _isolation import isolate_embeddings  # noqa: E402


def setUpModule() -> None:
    isolate_embeddings()


# ---------------------------------------------------------------- HTTP 替身


class _Resp:
    def __init__(self, payload, raw: bytes | None = None) -> None:
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FakeURL:
    """记录每次请求，按 handler(序号, 请求体) 返回响应或抛异常。"""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict] = []

    def __call__(self, request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        self.calls.append(
            {
                "url": request.full_url,
                "payload": payload,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "timeout": timeout,
            }
        )
        result = self.handler(len(self.calls), payload)
        return result if hasattr(result, "read") else _Resp(result)


def _vectors(payload: dict, *, dim: int = 4, reverse: bool = False) -> dict:
    """按 input 里的文本值造向量，方便验证"顺序有没有还原对"。"""
    inputs = payload["input"]
    items = [
        {"index": i, "embedding": [float(len(t))] * dim} for i, t in enumerate(inputs)
    ]
    if reverse:
        items.reverse()
    return {"data": items, "usage": {"total_tokens": len(inputs) * 3}}


def _cfg(**kw) -> EmbeddingConfig:
    base = dict(
        base_url="https://api.example.com/v1",
        model="test-embed",
        api_key="sk-test",
        provider="openai_compat",
        dimensions=None,
    )
    base.update(kw)
    return EmbeddingConfig(**base)


# ---------------------------------------------------------------- 配置


class TestEmbeddingConfig(unittest.TestCase):
    def test_endpoint_appends_once(self):
        self.assertEqual(_cfg().endpoint, "https://api.example.com/v1/embeddings")
        self.assertEqual(
            _cfg(base_url="https://api.example.com/v1/embeddings").endpoint,
            "https://api.example.com/v1/embeddings",
        )
        self.assertEqual(_cfg(base_url="").endpoint, "")

    def test_model_key_separates_dimensions(self):
        a = _cfg(dimensions=1024)
        b = _cfg(dimensions=512)
        self.assertNotEqual(a.model_key, b.model_key)
        self.assertNotEqual(a.model_key, _cfg(model="other").model_key)

    def test_ready_rules(self):
        self.assertFalse(_cfg(api_key="").ready, "远程没 Key 不能算就绪")
        local = _cfg(base_url="http://localhost:11434/v1", api_key="")
        self.assertTrue(local.ready, "本机 Ollama 不需要 Key")
        self.assertTrue(_cfg(provider="local", base_url="", model="bge-m3").ready)

    def test_masked_key(self):
        self.assertEqual(_cfg(api_key="").masked_key, "未填写")
        self.assertEqual(_cfg(api_key="sk-1234567890").masked_key, "sk-1…7890")
        self.assertEqual(_cfg(api_key="short").masked_key, "已填写")

    def test_save_reload_and_clear(self):
        root = Path(tempfile.mkdtemp(prefix="embcfg-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        target = root / "embedding.json"
        with mock.patch.object(config, "EMBEDDING_FILE", target):
            path = config.save_embedding(_cfg(dimensions=1024))
            self.assertEqual(path, target)
            self.assertTrue(target.exists())
            self.assertFalse(target.with_name(target.name + ".tmp").exists())
            self.assertEqual(config.reload_embedding().dimensions, 1024)

            target.write_text("{坏掉的 json", encoding="utf-8")
            self.assertEqual(config.reload_embedding(), config.EMBEDDING_ENV_DEFAULTS)

            config.save_embedding(_cfg())
            config.clear_embedding()
            self.assertFalse(target.exists())

    def test_does_not_touch_model_json(self):
        """两套配置必须互不干扰：写向量配置不能动 model.json。"""
        root = Path(tempfile.mkdtemp(prefix="embcfg2-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        with mock.patch.object(config, "EMBEDDING_FILE", root / "embedding.json"):
            with mock.patch.object(config, "MODEL_FILE", root / "model.json"):
                config.save_embedding(_cfg())
                self.assertTrue((root / "embedding.json").exists())
                self.assertFalse((root / "model.json").exists())


# ---------------------------------------------------------------- 切块与归一化


class TestSubChunksAndNormalize(unittest.TestCase):
    def test_short_text_kept_whole(self):
        self.assertEqual(embeddings.split_subchunks("一句话", size=500), [(0, "一句话")])

    def test_empty_text(self):
        self.assertEqual(embeddings.split_subchunks("   "), [])

    def test_long_text_is_windowed_with_overlap(self):
        text = "字" * 1200
        pieces = embeddings.split_subchunks(text, size=500, overlap=80)
        self.assertGreaterEqual(len(pieces), 3)
        for offset, piece in pieces:
            self.assertLessEqual(len(piece), 500)
            self.assertEqual(piece, text[offset : offset + len(piece)])
        # 相邻窗口必须有重叠，否则断句处会丢语义
        self.assertLess(pieces[1][0], pieces[0][0] + len(pieces[0][1]))

    def test_prefers_sentence_break(self):
        text = ("甲" * 400) + "。" + ("乙" * 400)
        pieces = embeddings.split_subchunks(text, size=500, overlap=50)
        self.assertTrue(pieces[0][1].endswith("。"), pieces[0][1][-5:])

    def test_dumb_arguments_do_not_loop_forever(self):
        pieces = embeddings.split_subchunks("字" * 50, size=1, overlap=10)
        self.assertGreater(len(pieces), 1)

    def test_l2_normalize_unit_norm(self):
        mat = embeddings.l2_normalize([[3.0, 4.0], [0.0, 2.0]])
        self.assertEqual(mat.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(mat, axis=1), [1.0, 1.0], rtol=1e-6)

    def test_zero_vector_stays_zero_not_nan(self):
        mat = embeddings.l2_normalize([[0.0, 0.0]])
        self.assertFalse(np.isnan(mat).any())
        self.assertEqual(float(mat.sum()), 0.0)

    def test_nan_rejected(self):
        with self.assertRaises(EmbeddingError):
            embeddings.l2_normalize([[1.0, float("nan")]])


# ---------------------------------------------------------------- HTTP


class TestEmbedTextsHTTP(unittest.TestCase):
    def test_auto_batching(self):
        fake = _FakeURL(lambda i, payload: _vectors(payload))
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            result = embeddings.embed_texts(
                ["a", "bb", "ccc", "dddd", "eeeee"], cfg=_cfg(), batch_size=2, retries=0
            )
        self.assertEqual(len(fake.calls), 3, "5 条按 batch_size=2 应拆成 3 批")
        self.assertEqual(result.vectors.shape, (5, 4))
        self.assertEqual(result.total_tokens, 15)

    def test_out_of_order_index_is_restored(self):
        def onehot(payload, reverse=False):
            n = len(payload["input"])
            items = [
                {"index": i, "embedding": [1.0 if j == i else 0.0 for j in range(n)]}
                for i in range(n)
            ]
            if reverse:
                items.reverse()
            return {"data": items}

        fake = _FakeURL(lambda i, payload: onehot(payload, reverse=True))
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            result = embeddings.embed_texts(["a", "bb", "ccc"], cfg=_cfg(), retries=0)
        # 向量是 one-hot，第 k 行非零位置必须等于 k；顺序错了会变成 [2,1,0]
        self.assertEqual([int(np.argmax(row)) for row in result.vectors], [0, 1, 2])

    def test_dimensions_only_when_configured(self):
        fake = _FakeURL(lambda i, payload: _vectors(payload))
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            embeddings.embed_texts(["x"], cfg=_cfg(dimensions=1024), retries=0)
            embeddings.embed_texts(["y"], cfg=_cfg(dimensions=None), retries=0)
        self.assertEqual(fake.calls[0]["payload"]["dimensions"], 1024)
        self.assertNotIn("dimensions", fake.calls[1]["payload"])
        # encoding_format 在部分国产网关会 400，绝不能自己加上
        self.assertNotIn("encoding_format", fake.calls[0]["payload"])

    def test_local_host_sends_no_authorization(self):
        fake = _FakeURL(lambda i, payload: _vectors(payload))
        cfg = _cfg(base_url="http://localhost:11434/v1", api_key="")
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            embeddings.embed_texts(["x"], cfg=cfg, retries=0)
        self.assertNotIn("authorization", fake.calls[0]["headers"])

    def test_retries_on_429_then_succeeds(self):
        def handler(call, payload):
            if call == 1:
                raise urllib.error.HTTPError("u", 429, "rate", None, None)
            return _vectors(payload)

        fake = _FakeURL(handler)
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            with mock.patch("app.embeddings.time.sleep", lambda *_: None):
                result = embeddings.embed_texts(["x"], cfg=_cfg(), retries=2)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(result.vectors.shape, (1, 4))

    def test_401_is_not_retried(self):
        fake = _FakeURL(
            lambda i, payload: (_ for _ in ()).throw(
                urllib.error.HTTPError("u", 401, "bad key", None, None)
            )
        )
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            with self.assertRaises(EmbeddingError) as ctx:
                embeddings.embed_texts(["x"], cfg=_cfg(), retries=3)
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(len(fake.calls), 1, "401 重试没有意义，只会浪费额度")

    def test_non_json_response(self):
        fake = _FakeURL(lambda i, payload: _Resp(None, raw=b"<html>404</html>"))
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            with self.assertRaises(EmbeddingError):
                embeddings.embed_texts(["x"], cfg=_cfg(), retries=0)

    def test_count_mismatch(self):
        fake = _FakeURL(lambda i, payload: {"data": [{"index": 0, "embedding": [1.0]}]})
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            with self.assertRaises(EmbeddingError):
                embeddings.embed_texts(["a", "b"], cfg=_cfg(), retries=0)

    def test_ragged_dimensions_rejected(self):
        fake = _FakeURL(
            lambda i, payload: {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0, 0.0, 0.0]},
                ]
            }
        )
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            with self.assertRaises((EmbeddingError, ValueError)):
                embeddings.embed_texts(["a", "b"], cfg=_cfg(), retries=0)

    def test_empty_text_rejected_before_http(self):
        fake = _FakeURL(lambda i, payload: _vectors(payload))
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            with self.assertRaises(ValueError):
                embeddings.embed_texts(["ok", "   "], cfg=_cfg(), retries=0)
        self.assertEqual(fake.calls, [], "空串不能发出去（接口会 400）")

    def test_not_ready_raises(self):
        with self.assertRaises(EmbeddingError):
            embeddings.embed_texts(["x"], cfg=_cfg(api_key=""), retries=0)

    def test_probe_ok_and_failure_never_raise(self):
        fake = _FakeURL(lambda i, payload: _vectors(payload, dim=8))
        with mock.patch("app.embeddings.urllib.request.urlopen", fake):
            ok = embeddings.probe_embedding(
                "https://api.example.com/v1", "test-embed", "sk-x"
            )
        self.assertTrue(ok.ok, ok.detail)
        self.assertEqual(ok.dim, 8)
        self.assertIn("8 维", ok.summary)

        def boom(*_a, **_kw):
            raise urllib.error.URLError("no route")

        with mock.patch("app.embeddings.urllib.request.urlopen", boom):
            bad = embeddings.probe_embedding(
                "https://api.example.com/v1", "test-embed", "sk-x"
            )
        self.assertFalse(bad.ok)
        self.assertIn("连不上", bad.detail)


# ---------------------------------------------------------------- 向量库


class TestVectorStore(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="embstore-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        patcher = mock.patch.object(vault, "VAULT_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        embeddings.clear_caches()

    def _doc(self, name: str, text: str) -> Doc:
        return Doc(name=name, kind="txt", text=text)

    def test_persists_across_restart(self):
        store = embeddings.VectorStore()
        h = embeddings.chunk_hash("一段话")
        store.put_many([(h, np.ones(4, dtype=np.float32) / 2)], "k1")
        embeddings.clear_caches()          # 模拟进程重启：内存缓存没了
        again = embeddings.VectorStore()
        got = again.get_many([h], "k1")
        self.assertIn(h, got)
        np.testing.assert_allclose(got[h], [0.5, 0.5, 0.5, 0.5])

    def test_same_text_embedded_once(self):
        calls: list[list[str]] = []

        def embedder(texts):
            calls.append(list(texts))
            return np.ones((len(texts), 4), dtype=np.float32)

        store = embeddings.VectorStore()
        docs = [self._doc("a.txt", "相同的一段内容"), self._doc("b.txt", "相同的一段内容")]
        subchunks, vecs, stats = store.ensure_indexed(docs, embedder=embedder, model_key="custom")
        self.assertEqual(stats.embedded, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(subchunks), 2)
        self.assertEqual(len({sc.chunk_hash for sc in subchunks}), 1)

    def test_model_key_isolation(self):
        store = embeddings.VectorStore()
        h = embeddings.chunk_hash("x")
        store.put_many([(h, np.ones(4, dtype=np.float32))], "model-a")
        self.assertIn(h, store.get_many([h], "model-a"))
        self.assertNotIn(h, store.get_many([h], "model-b"))

    def test_failed_batch_keeps_earlier_batches(self):
        """第 2 批失败时，第 1 批必须已经落盘——这是断点续传的全部意义。"""
        state = {"n": 0}

        def embedder(texts):
            state["n"] += 1
            if state["n"] == 2:
                raise EmbeddingError("模拟限流")
            return np.ones((len(texts), 4), dtype=np.float32)

        store = embeddings.VectorStore()
        doc = self._doc("long.txt", "甲" * 1200)   # 500 字子块 -> 3 个子块
        with self.assertRaises(EmbeddingError):
            store.ensure_indexed([doc], embedder=embedder, model_key="k", batch_size=1)

        embeddings.clear_caches()
        kept = embeddings.VectorStore().stats("k")["count"]
        self.assertEqual(kept, 1, "失败前已完成的批次不该丢")

    def test_prune_orphans(self):
        store = embeddings.VectorStore()
        hashes = [embeddings.chunk_hash(f"文本{i}") for i in range(3)]
        store.put_many([(h, np.ones(2, dtype=np.float32)) for h in hashes], "k")
        removed = store.prune_orphans({hashes[0]}, "k")
        self.assertEqual(removed, 2)
        self.assertEqual(store.stats("k")["count"], 1)
        self.assertEqual(set(store.get_many(hashes, "k")), {hashes[0]})

    def test_connections_closed_so_tempdir_can_be_removed(self):
        """Windows 上句柄没释放，rmtree 会报 WinError 32。这条专门盯它。"""
        store = embeddings.VectorStore()
        store.put_many([(embeddings.chunk_hash("x"), np.ones(4, dtype=np.float32))], "k")
        store.get_many([embeddings.chunk_hash("x")], "k")
        store.stats("k")
        shutil.rmtree(self.root)   # 不加 ignore_errors，失败就让用例红

    def test_preindex_returns_none_without_config(self):
        store = embeddings.VectorStore()
        self.assertIsNone(store.preindex_async([self._doc("a.txt", "内容")], cfg=_cfg(api_key="")))


if __name__ == "__main__":
    unittest.main()
