"""Embedding 文本向量化：切子块、调接口、算向量、落盘缓存。

为什么单独一层：
  1. 向量模型和对话模型是两套东西。对话走 `config.current()`（DeepSeek），
     而 DeepSeek 没有 /embeddings 端点；硬把同一套配置复用过来只会 404。
     所以这里只认 `config.embedding_current()`。
  2. 只依赖标准库 + numpy：`urllib.request` 发请求，`sqlite3` 存向量。
     引一个向量数据库对这个体量（几百到几千个子块，矩阵点积亚毫秒级）毫无收益。

三个设计取舍，都是踩过的坑：

  **父子切块**：`Doc.chunks`（约 2000 字）继续当阅读单位，段号不变；向量化时
  再切成 500 字、重叠 80 字的子块。2000 字直接送进 512 token 的向量模型会被截断，
  而且长段里真正的答案只占几句话，语义会被背景稀释。

  **内容寻址缓存**：库里按 `(chunk_hash, model_key)` 存。同一段文字换个文件名、
  换个会话都复用同一份向量，不重复花钱；换模型/换维度则是另一把钥匙，互不覆盖。

  **SQLite 即开即关**：Streamlit 每次交互跑在新线程，后台预索引又跑在 daemon 线程。
  长期持有一个 Connection，轻则报"SQLite objects created in a thread…"，
  重则 Windows 下测试 tearDown 的 rmtree 报 WinError 32（句柄没释放）。
  所以每次读写完立刻 close，热数据放进程内存 `_MEM_CACHE`。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import config
from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_CHARS,
    EMBEDDING_QUERY_TIMEOUT,
    EMBEDDING_SUBCHUNK_CHARS,
    EMBEDDING_SUBCHUNK_OVERLAP,
    EMBEDDING_TIMEOUT,
    EmbeddingConfig,
)
from .readers import Doc


class EmbeddingError(RuntimeError):
    """向量接口出问题。status 是 HTTP 码（0 表示没走到 HTTP，比如网络不通）。"""

    def __init__(self, message: str, status: int = 0, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class SubChunk:
    """父段里切出来的一个子块。offset 是它在**父段文本**里的起始字符位置。"""

    doc_order: int
    doc_name: str
    part: int          # 父段号，1-based，可直接喂给 read_document
    offset: int
    text: str
    chunk_hash: str


@dataclass
class EmbeddingBatchResult:
    vectors: np.ndarray     # (N, D)，float32，已 L2 归一化
    total_tokens: int = 0
    latency: float = 0.0
    model: str = ""


@dataclass
class EmbeddingProbeResult:
    ok: bool
    detail: str
    latency: float = 0.0
    model: str = ""
    dim: int = 0

    @property
    def summary(self) -> str:
        if not self.ok:
            return self.detail
        spent = f"{self.latency * 1000:.0f} ms"
        dim_part = f" · {self.dim} 维" if self.dim else ""
        name = f"，模型 `{self.model}`{dim_part}" if self.model else ""
        return f"连接正常（{spent}{name}）"


@dataclass(frozen=True)
class IndexStats:
    total_subchunks: int
    cached: int
    embedded: int
    tokens: int = 0
    latency: float = 0.0


# 供单测或自定义模型注入的统一契约：非空文本列表 -> (N, D) 向量
EmbedderCallable = Callable[[list[str]], Any]


# ---------------------------------------------------------------- 切块与归一化


def chunk_hash(text: str) -> str:
    """内容哈希。取 32 位十六进制：碰撞概率可忽略，显存/磁盘都省。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


def split_subchunks(
    text: str,
    size: int = EMBEDDING_SUBCHUNK_CHARS,
    overlap: int = EMBEDDING_SUBCHUNK_OVERLAP,
) -> list[tuple[int, str]]:
    """把父段切成滑动窗口子块，返回 [(起始偏移, 文本)]。

    overlap 被夹在 `[0, size-1]`：等于 size 的话窗口原地不动，就是死循环。
    断点优先落在换行/句号上，和 readers.split_chunks 一个思路——半句话的向量
    比整句话的向量更容易跑偏。
    """
    text = (text or "").strip()
    if not text:
        return []

    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    total = len(text)
    if total <= size:
        return [(0, text)]

    pieces: list[tuple[int, str]] = []
    start = 0
    while start < total:
        end = min(start + size, total)
        if end < total:
            window = text[start:end]
            cut = max(
                window.rfind("\n"),
                window.rfind("。"),
                window.rfind(". "),
                window.rfind("！"),
                window.rfind("？"),
            )
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append((start, piece))
        if end >= total:
            break
        start = max(start + 1, end - overlap)
    return pieces


def l2_normalize(vectors: Any) -> np.ndarray:
    """归一化成单位向量。之后点积就是余弦相似度，省掉每次除范数。"""
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim == 1:
        if mat.size == 0:
            raise EmbeddingError("向量结果是空的。")
        mat = mat.reshape(1, -1)
    if mat.ndim != 2 or mat.shape[1] == 0:
        raise EmbeddingError(f"向量维度不对：期望二维矩阵，收到 shape={mat.shape}。")
    if not np.all(np.isfinite(mat)):
        raise EmbeddingError("向量结果包含非法数值（NaN/Inf）。")

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # 零向量保持全 0：除以 0 会得到 NaN，后续点积全变 NaN，排查起来极难。
    safe = np.where(norms > 1e-12, norms, 1.0)
    return (mat / safe).astype(np.float32, copy=False)


# ---------------------------------------------------------------- HTTP 客户端

_RETRY_STATUS = {429, 500, 502, 503, 504}


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EmbeddingError(
            "向量接口返回的不是 JSON，检查一下 Base URL 是不是填成了聊天接口。",
            detail=text[:200],
        ) from exc
    if not isinstance(data, dict):
        raise EmbeddingError("向量接口返回格式不对。", detail=text[:200])
    return data


def _embed_remote(
    batch: list[str],
    cfg: EmbeddingConfig,
    *,
    timeout: int,
    retries: int,
) -> tuple[np.ndarray, int]:
    """走 OpenAI 兼容协议。返回 (向量矩阵, 消耗 token 数)。"""
    headers = {"Content-Type": "application/json"}
    # 本机 Ollama 不需要 Key；塞一个空的 Bearer 反而会被某些网关判成非法。
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    payload: dict[str, Any] = {"model": cfg.model, "input": batch}
    if cfg.dimensions is not None:
        # 刻意不发 encoding_format：部分国产网关/中转站不认这个字段，直接 400。
        payload["dimensions"] = int(cfg.dimensions)

    attempt = 0
    while True:
        try:
            data = _post_json(cfg.endpoint, payload, headers, timeout)
            break
        except urllib.error.HTTPError as exc:
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code in _RETRY_STATUS and attempt < retries:
                time.sleep(min(4, 1.5 * (attempt + 1)))
                attempt += 1
                continue
            raise EmbeddingError(
                f"向量接口返回 HTTP {exc.code}。", status=exc.code, detail=detail
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                time.sleep(min(4, 1.5 * (attempt + 1)))
                attempt += 1
                continue
            raise EmbeddingError(f"连不上向量接口：{exc}") from exc

    items = data.get("data")
    if not isinstance(items, list):
        raise EmbeddingError("向量接口没有返回 data 列表。", detail=str(data)[:200])
    if len(items) != len(batch):
        raise EmbeddingError(
            f"向量接口返回 {len(items)} 条，请求的是 {len(batch)} 条。",
            detail=str(data)[:200],
        )

    # OpenAI 规范允许乱序返回，靠 index 还原；缺 index 就按位置。
    ordered = sorted(
        enumerate(items),
        key=lambda pair: (
            pair[1].get("index", pair[0]) if isinstance(pair[1], dict) else pair[0],
            pair[0],
        ),
    )
    rows: list[list[float]] = []
    for _, item in ordered:
        if not isinstance(item, dict):
            raise EmbeddingError("向量接口的 data 项格式不对。")
        vec = item.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise EmbeddingError("向量接口返回了空向量。")
        rows.append(vec)

    dims = {len(row) for row in rows}
    if len(dims) != 1:
        raise EmbeddingError(f"同一批向量维度不一致：{sorted(dims)}")

    usage = data.get("usage") or {}
    tokens = 0
    if isinstance(usage, dict):
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
    return np.asarray(rows, dtype=np.float32), tokens


_LOCAL_MODELS: dict[str, Any] = {}
_LOCAL_LOCK = threading.Lock()


def _embed_local(batch: list[str], cfg: EmbeddingConfig, *, batch_size: int) -> np.ndarray:
    """本地 sentence-transformers。缺库时说清楚装什么，而不是抛 ImportError。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "使用本地向量模型需要安装 sentence-transformers：`pip install sentence-transformers`"
        ) from exc

    with _LOCAL_LOCK:
        model = _LOCAL_MODELS.get(cfg.model)
        if model is None:
            model = SentenceTransformer(cfg.model)
            _LOCAL_MODELS[cfg.model] = model
    encoded = model.encode(
        batch,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(encoded, dtype=np.float32)


def embed_texts(
    texts: Sequence[str],
    *,
    cfg: EmbeddingConfig | None = None,
    timeout: int = EMBEDDING_TIMEOUT,
    retries: int = 2,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> EmbeddingBatchResult:
    """把一批文本变成归一化向量。按 batch_size 自动拆批。"""
    if not texts:
        return EmbeddingBatchResult(vectors=np.zeros((0, 0), dtype=np.float32))

    cleaned = [str(t or "").strip()[:EMBEDDING_MAX_CHARS] for t in texts]
    if any(not t for t in cleaned):
        # 空串发给 /embeddings 会直接 400（扫描件抽出来的空块就是这种）。
        raise ValueError("待向量化文本不能为空")

    cfg = cfg or config.embedding_current()
    if not cfg.ready:
        raise EmbeddingError("未配置可用的向量模型（Embedding）。")

    started = time.perf_counter()
    size = max(1, int(batch_size))
    blocks: list[np.ndarray] = []
    tokens = 0
    for start in range(0, len(cleaned), size):
        batch = cleaned[start : start + size]
        if cfg.provider == "local":
            mat = _embed_local(batch, cfg, batch_size=size)
        else:
            mat, spent = _embed_remote(batch, cfg, timeout=timeout, retries=retries)
            tokens += spent
        normalized = l2_normalize(mat)
        if normalized.shape[0] != len(batch):
            raise EmbeddingError(
                f"向量条数不对：请求 {len(batch)} 条，返回 {normalized.shape[0]} 条。"
            )
        if blocks and normalized.shape[1] != blocks[0].shape[1]:
            raise EmbeddingError("分批返回的向量维度不一致，检查一下模型配置。")
        blocks.append(normalized)

    return EmbeddingBatchResult(
        vectors=np.vstack(blocks),
        total_tokens=tokens,
        latency=time.perf_counter() - started,
        model=cfg.model,
    )


_QUERY_CACHE: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
_QUERY_CACHE_LIMIT = 128
_QUERY_LOCK = threading.Lock()


def embed_query(
    query: str,
    *,
    cfg: EmbeddingConfig | None = None,
    embedder: EmbedderCallable | None = None,
    model_key: str = "",
) -> np.ndarray:
    """把查询变成向量。同一个问题反复问（改写、多轮）时走 LRU，不再发请求。"""
    query = (query or "").strip()
    if not query:
        raise ValueError("查询不能为空")

    if embedder is not None:
        # 自定义 embedder 不进全局缓存：单测之间会串味，也没必要。
        return l2_normalize(embedder([query]))[0]

    cfg = cfg or config.embedding_current()
    key = (model_key or cfg.model_key, query)
    with _QUERY_LOCK:
        hit = _QUERY_CACHE.get(key)
        if hit is not None:
            _QUERY_CACHE.move_to_end(key)
            return hit

    result = embed_texts(
        [query], cfg=cfg, timeout=EMBEDDING_QUERY_TIMEOUT, retries=1
    )
    vector = result.vectors[0]
    with _QUERY_LOCK:
        _QUERY_CACHE[key] = vector
        _QUERY_CACHE.move_to_end(key)
        while len(_QUERY_CACHE) > _QUERY_CACHE_LIMIT:
            _QUERY_CACHE.popitem(last=False)
    return vector


def probe_embedding(
    base_url: str,
    model: str,
    api_key: str = "",
    *,
    provider: str = "openai_compat",
    dimensions: int | None = None,
    timeout: int = 15,
) -> EmbeddingProbeResult:
    """试连一次。**刻意不抛异常**——它被界面按钮调用，失败要能显示原因。"""
    name = (model or "").strip()
    if not name:
        return EmbeddingProbeResult(ok=False, detail="模型名不能为空。")
    try:
        cfg = config.EmbeddingConfig(
            base_url=config.normalize_base_url(base_url) if (base_url or "").strip() else "",
            model=name,
            api_key=(api_key or "").strip(),
            provider=provider,
            dimensions=dimensions,
        )
    except ValueError as exc:
        return EmbeddingProbeResult(ok=False, detail=str(exc))
    if not cfg.ready:
        if cfg.provider != "local" and not cfg.api_key:
            return EmbeddingProbeResult(ok=False, detail="远程向量模型需要填写 API Key。")
        return EmbeddingProbeResult(ok=False, detail="Base URL 和模型名都要填。")

    started = time.perf_counter()
    try:
        result = embed_texts(["ping"], cfg=cfg, timeout=timeout, retries=0)
    except EmbeddingError as exc:
        detail = str(exc)
        if exc.detail:
            detail = f"{detail} {exc.detail[:120]}"
        return EmbeddingProbeResult(ok=False, detail=detail, model=name)
    except Exception as exc:  # noqa: BLE001  探针不能因为任何意外把界面弄崩
        return EmbeddingProbeResult(ok=False, detail=f"{type(exc).__name__}: {exc}", model=name)

    return EmbeddingProbeResult(
        ok=True,
        detail="",
        latency=time.perf_counter() - started,
        model=result.model or name,
        dim=int(result.vectors.shape[1]) if result.vectors.size else 0,
    )


# ---------------------------------------------------------------- 向量库


def default_db_path() -> Path:
    """**调用时**才解析路径。

    测试用 `mock.patch.object(vault, "VAULT_DIR", tmp)` 隔离储藏区；导入期就把
    路径写死的话，测试会往用户真实的 data/vault 里写库文件。
    """
    from . import vault

    return vault.VAULT_DIR / "embeddings.db"


# (库路径, 模型键, 内容哈希) -> 已反序列化的向量。热检索零 SQLite I/O。
_MEM_CACHE: dict[tuple[str, str, str], np.ndarray] = {}
_INFLIGHT: set[tuple[str, str]] = set()
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_hash  TEXT NOT NULL,
    model_key   TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (chunk_hash, model_key)
);
CREATE INDEX IF NOT EXISTS ix_embeddings_model ON embeddings(model_key);
"""


class VectorStore:
    """向量缓存。默认落在 `vault.VAULT_DIR / embeddings.db`，删掉可以随时重建。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None

    @property
    def db_path(self) -> Path:
        return self._db_path if self._db_path is not None else default_db_path()

    @contextlib.contextmanager
    def _connect(self):
        path = self.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=3.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 3000")
            conn.executescript(_SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()   # 用完立刻关：见模块开头第 3 条

    def get_many(self, chunk_hashes: Sequence[str], model_key: str) -> dict[str, np.ndarray]:
        """按哈希批量取向量。内存命中就不查库。"""
        path = str(self.db_path)
        found: dict[str, np.ndarray] = {}
        pending: list[str] = []
        with _LOCK:
            for h in chunk_hashes:
                cached = _MEM_CACHE.get((path, model_key, h))
                if cached is not None:
                    found[h] = cached
                else:
                    pending.append(h)

        if not pending:
            return found

        unique = list(dict.fromkeys(pending))
        with self._connect() as conn:
            for start in range(0, len(unique), 400):   # SQLite 变量上限 999
                batch = unique[start : start + 400]
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT chunk_hash, dim, vector FROM embeddings "
                    f"WHERE model_key = ? AND chunk_hash IN ({marks})",
                    [model_key, *batch],
                ).fetchall()
                for h, dim, blob in rows:
                    if not isinstance(blob, (bytes, bytearray)) or len(blob) != int(dim) * 4:
                        continue   # 坏行跳过，别让一条脏数据废掉整次检索
                    vec = np.frombuffer(blob, dtype="<f4").copy()
                    found[h] = vec
                    with _LOCK:
                        _MEM_CACHE[(path, model_key, h)] = vec
        return found

    def put_many(self, items: Sequence[tuple[str, np.ndarray]], model_key: str) -> None:
        """批量落盘。一个事务写完，要么全进要么全不进。"""
        if not items:
            return
        path = str(self.db_path)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        rows = []
        for h, vec in items:
            arr = np.asarray(vec, dtype="<f4").ravel()
            rows.append((h, model_key, int(arr.size), arr.tobytes(), stamp))

        with _LOCK:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings "
                    "(chunk_hash, model_key, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
            for h, vec in items:
                _MEM_CACHE[(path, model_key, h)] = np.asarray(vec, dtype=np.float32)

    def ensure_indexed(
        self,
        docs: Sequence[Doc],
        *,
        embedder: EmbedderCallable | None = None,
        cfg: EmbeddingConfig | None = None,
        model_key: str = "",
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> tuple[list[SubChunk], dict[str, np.ndarray], IndexStats]:
        """确保这批文档的子块都有向量，返回 (子块列表, 哈希->向量, 统计)。

        每批算完立刻落盘：限流/超时中断后，已经算好的批次下次直接复用（断点续传）。
        """
        started = time.perf_counter()
        subchunks: list[SubChunk] = []
        for order, doc in enumerate(docs):
            for part_index, chunk in enumerate(doc.chunks, 1):
                if not (chunk or "").strip():
                    continue
                for offset, piece in split_subchunks(chunk):
                    subchunks.append(
                        SubChunk(
                            doc_order=order,
                            doc_name=doc.name,
                            part=part_index,
                            offset=offset,
                            text=piece,
                            chunk_hash=chunk_hash(piece),
                        )
                    )

        if not subchunks:
            return [], {}, IndexStats(0, 0, 0)

        cfg = cfg or config.embedding_current()
        effective_key = model_key or cfg.model_key

        order_of: dict[str, str] = {}     # 哈希 -> 文本（去重且保序）
        for sc in subchunks:
            order_of.setdefault(sc.chunk_hash, sc.text)

        unique = list(order_of)
        vectors = self.get_many(unique, effective_key)
        missing = [(h, t) for h, t in order_of.items() if h not in vectors]

        if not missing:
            return subchunks, vectors, IndexStats(
                total_subchunks=len(subchunks),
                cached=len(unique),
                embedded=0,
                latency=time.perf_counter() - started,
            )

        size = max(1, int(batch_size))
        tokens = 0
        for start in range(0, len(missing), size):
            batch = missing[start : start + size]
            texts = [t for _, t in batch]
            if embedder is not None:
                mat = l2_normalize(embedder(texts))
            else:
                result = embed_texts(texts, cfg=cfg)
                mat = result.vectors
                tokens += result.total_tokens
            if mat.shape[0] != len(batch):
                raise EmbeddingError(
                    f"向量条数不对：请求 {len(batch)} 条，返回 {mat.shape[0]} 条。"
                )
            pairs = [(h, mat[i]) for i, (h, _) in enumerate(batch)]
            self.put_many(pairs, effective_key)
            for h, vec in pairs:
                vectors[h] = np.asarray(vec, dtype=np.float32)

        return subchunks, vectors, IndexStats(
            total_subchunks=len(subchunks),
            cached=len(unique) - len(missing),
            embedded=len(missing),
            tokens=tokens,
            latency=time.perf_counter() - started,
        )

    def preindex_async(
        self,
        docs: Sequence[Doc],
        *,
        cfg: EmbeddingConfig | None = None,
    ) -> threading.Thread | None:
        """上传后先在后台把向量算好，别让用户问第一句时干等。

        没配置、或者已经全在库里，就直接返回 None。线程里任何异常都吞掉：
        预索引失败最多是第一次检索慢一点，不该在界面上冒红字。
        """
        cfg = cfg or config.embedding_current()
        if not cfg.ready:
            return None

        wanted: dict[str, str] = {}
        for doc in docs:
            for chunk in doc.chunks:
                for _, piece in split_subchunks(chunk):
                    wanted.setdefault(chunk_hash(piece), piece)
        if not wanted:
            return None

        key = cfg.model_key
        known = self.get_many(list(wanted), key)
        missing = [h for h in wanted if h not in known]
        if not missing:
            return None

        # 同一份文档被重复上传/重复挂载时，别开第二个线程算同样的活。
        inflight_key = (key, chunk_hash("\n".join(sorted(missing))))
        with _LOCK:
            if inflight_key in _INFLIGHT:
                return None
            _INFLIGHT.add(inflight_key)

        def _run() -> None:
            try:
                self.ensure_indexed(docs, cfg=cfg)
            except Exception:  # noqa: BLE001  后台线程绝不能把异常抛给主流程
                pass
            finally:
                with _LOCK:
                    _INFLIGHT.discard(inflight_key)

        thread = threading.Thread(target=_run, name="embedding-preindex", daemon=True)
        thread.start()
        return thread

    def stats(self, model_key: str | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            if model_key:
                rows = conn.execute(
                    "SELECT dim, COUNT(*), COALESCE(SUM(dim * 4), 0) FROM embeddings "
                    "WHERE model_key = ? GROUP BY dim",
                    (model_key,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT dim, COUNT(*), COALESCE(SUM(dim * 4), 0) FROM embeddings GROUP BY dim"
                ).fetchall()
        return {
            "count": sum(int(r[1]) for r in rows),
            "dims": sorted({int(r[0]) for r in rows}),
            "bytes": sum(int(r[2]) for r in rows),
        }

    def prune_orphans(self, live_hashes: set[str], model_key: str | None = None) -> int:
        """删掉没有任何文档再引用的向量（文档删了/改了）。返回删除条数。"""
        path = str(self.db_path)
        with self._connect() as conn:
            if model_key:
                rows = conn.execute(
                    "SELECT chunk_hash FROM embeddings WHERE model_key = ?", (model_key,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT chunk_hash FROM embeddings").fetchall()
            dead = [r[0] for r in rows if r[0] not in live_hashes]
            if dead:
                for start in range(0, len(dead), 400):
                    batch = dead[start : start + 400]
                    marks = ",".join("?" * len(batch))
                    if model_key:
                        conn.execute(
                            f"DELETE FROM embeddings WHERE model_key = ? AND chunk_hash IN ({marks})",
                            [model_key, *batch],
                        )
                    else:
                        conn.execute(
                            f"DELETE FROM embeddings WHERE chunk_hash IN ({marks})", batch
                        )
        with _LOCK:
            dead = set(dead)
            for key in [k for k in _MEM_CACHE if k[0] == path and k[2] in dead]:
                _MEM_CACHE.pop(key, None)
        return len(dead)


def preindex_async(
    docs: Sequence[Doc],
    *,
    cfg: EmbeddingConfig | None = None,
    db_path: Path | None = None,
) -> threading.Thread | None:
    """模块级快捷入口：用默认 VectorStore 在后台预建向量索引。"""
    return VectorStore(db_path).preindex_async(docs, cfg=cfg)


def clear_caches() -> None:
    """清掉所有进程内缓存。测试隔离和"换模型后重新算"用。"""
    with _LOCK:
        _MEM_CACHE.clear()
        _INFLIGHT.clear()
        _LOCAL_MODELS.clear()
    with _QUERY_LOCK:
        _QUERY_CACHE.clear()


__all__ = [
    "EmbeddingBatchResult",
    "EmbeddingError",
    "EmbeddingProbeResult",
    "EmbedderCallable",
    "IndexStats",
    "SubChunk",
    "VectorStore",
    "chunk_hash",
    "clear_caches",
    "default_db_path",
    "embed_query",
    "embed_texts",
    "l2_normalize",
    "preindex_async",
    "probe_embedding",
    "split_subchunks",
]
