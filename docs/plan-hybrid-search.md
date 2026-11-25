# Embedding 向量化与混合检索（向量 + BM25 + RRF）架构与实施计划书

> 状态：待执行（可直接交付给执行 Agent 实施）。  
> 目标代码库：`e:\VScode\datainsight`（当前回归测试基线：513 项 `unittest`）。  
> 本机环境实测：Python 3.13.5（conda base）、SQLite 3.45.3（支持 FTS5）、`numpy 2.1.3`、`jieba 0.42.1`；未安装 `torch` / `sentence_transformers`。  
> 与 L2 关系：`docs/plan-l2.md` 尚未合入，本方案**不依赖 L2**，可在现有代码库上独立运行并平滑兼容未来 L2。  
> 本地密钥提示：用户已将智谱 `embedding-3`（1024 维）对应的 `EMBEDDING_*` 配置写入本地 `.env`（已加入 `.gitignore`），**严禁在任何代码、测试或 Git 追踪文档中硬编码明文 API Key**。



---

## 1. 目标、非目标与硬约束

### 1.1 核心目标

1. **Embedding 文本向量化层（`app/embeddings.py` + `app/config.py`）**
   - 将文档正文按适配向量模型上下文窗口的粒度切分为**子块（Sub-chunk）**，调用 Embedding 模型转为 **L2 归一化 `float32` 稠密向量**。
   - **配置彻底独立**：与对话模型 `ModelConfig`（`data/model.json`）完全解耦。当前用户主对话模型使用 DeepSeek（`https://api.deepseek.com/v1`），而 **DeepSeek 官方不提供 `/embeddings` 端点**，绝不可复用到对话模型配置。
   - **多端协议支持**：内置标准库 `urllib.request` 实现的 OpenAI 兼容协议客户端，支持智谱（`embedding-3`）、OpenAI（`text-embedding-3-small/large`）、硅基流动（`BAAI/bge-m3`）、百炼（`text-embedding-v3`）及免 Key 本机 Ollama（`http://localhost:11434/v1`）；同时预留可选本地 `sentence_transformers` 懒加载分支。
   - **内容寻址持久化缓存**：在 `vault.VAULT_DIR / "embeddings.db"`（SQLite WAL）中按 `(chunk_hash, model_key)` 持久化存储向量，实现同内容去重、多模型互不覆盖、批处理断点续传与删库可重建。
2. **BM25 精确词匹配层（`app/bm25.py`）**
   - 替换当前 `app/tools/doc_tools.py` 中的启发式打分 `_score()`，实现标准 **Okapi BM25**（采用 **Lucene 非负 IDF**，彻底解决小文档集 IDF 为负的经典缺陷）。
   - 支持中英混合分词（可选 `jieba.lcut_for_search` + 零依赖纯 Python CJK bigram 回退），并保留针对短词字面误召回的**覆盖率（0.6）与邻近窗口（48 字）防误召回闸门**。
3. **混合检索与 RRF 融合层（`app/hybrid_search.py` + `app/tools/doc_tools.py`）**
   - 并行运行 **BM25 精确词匹配路** 与 **Dense Vector 语义相似度路**（子块余弦相似度 → 父段落 MaxSim 聚合），通过**加权 RRF（Reciprocal Rank Fusion，倒数排名融合）**&#x5408;并两路排名并生成 Top-K 候选卡片。
   - **静默平滑降级**：未配置 Embedding、关闭混合检索开关、或向量接口超时/限流/断网时，自动无感退回纯 BM25 检索，绝不阻塞主问答流程，保住离线可用性。

### 1.2 非目标与硬约束

- **零外部重型编排/向量框架**：不引入 LangChain、LlamaIndex、Chroma、FAISS、Milvus、rank_bm25；仅用 Python 标准库（`sqlite3`、`urllib.request`、`hashlib`、`contextvars`、`threading`）+ `numpy`（`jieba` 与 `sentence-transformers` 均为可选懒导入）。
- **保持 `Doc.chunks` 与 `read_document(name, part)` 契约不变**：`app/readers.py` 的 2000 字父段切分与段号体系完全不动，保障 `read_document` 返回页码/段号 100% 稳定。
- **保持 `search_document(keyword: str, name: str = "")` 工具签名不变**：不新增平行检索工具让 LLM 做选择题，确保 `tests/test_docs.py` 的参数校验与 schema 契约全绿。
- **100% 向后兼容现有 513 项测试**：尤其严守 `tests/test_model_panel.py` 对 `text_input` 数量和文案、`tests/test_week4_ui.py` 对侧边栏空状态文案、`tests/test_docs.py` 对 `TestSearchScopeAndRanking` 行为的全部断言。

---

## 2. 本机真实语料实测基线

在 `e:\VScode\datainsight\data\vault\` 现有真实语料（`区块链教程-廖雪峰-2025-06-16.pdf`、`中国区块链创新应用案例集.pdf`、`bitcoin_en.pdf` 共 3 份 PDF）上的实测数据如下：

| 实测维度                                 | 实测数值                               | 对架构设计的直接约束                                                          |
| ------------------------------------ | ---------------------------------- | ------------------------------------------------------------------- |
| 文档规模（2000 字父段）                       | 3 份文档 / 263,543 字 / 134 段          | 父段偏长，直接送入 512-token 本地向量模型会被截断，且长段语义易被稀释                            |
| 父子切块规模（500 字 / 重叠 80 字）              | **665 个子块**（均长 459.0 字，最大 500 字）   | 兼容所有主流 512~8192 token 向量模型；支持精确到子句的片段定位                             |
| `numpy` 矩阵点积（665 × 1024 `float32`）   | **0.146 ms**（矩阵内存 2.60 MB）         | 纯 `numpy` 精确余弦检索已达亚毫秒级，无需引入 FAISS/Milvus                            |
| `numpy` 矩阵点积（5,000 × 1024 `float32`） | **0.178 ms**（矩阵内存 19.53 MB）        | 扩容至 ~1,000 父段（约 20 份长 PDF）仍 < 0.2 ms                                |
| `jieba` 首次加载 + 全库分词                  | 冷加载 429 ms + 分词 446 ms（且默认打印前缀树日志） | 必须 `jieba.setLogLevel(logging.WARNING)` 静音，并按 `chunk_hash` 内存缓存分词结果 |

---

## 3. 整体架构与端到端数据流

### 3.1 端到端检索数据流

```text
search_document(keyword, name="")   [app/tools/doc_tools.py]
        │
        ▼
hybrid_search(query, docs, top_k=3) [app/hybrid_search.py]
        │
        ├──────────────────────────────────────────────────────┐
        ▼                                                      ▼
[路 1：BM25 精确词匹配]  app/bm25.py              [路 2：Dense Vector 语义检索]  app/embeddings.py
  · Stage 1: 整词精确 Okapi BM25                    · 触发：mode!="bm25" 且 (注入embedder 或
    (Lucene非负IDF, k1=1.5, b=0.75)                   (is_hybrid_enabled() 且 cfg.ready))
  · 若 Stage 1 零命中 → Stage 2:                    · ensure_indexed(docs):
    - 短中文词(≤10字): bigram BM25                    - 父段(2000字) → 子块(500字, overlap=80)
      + 覆盖率≥0.6 + 邻近窗口≤48字(防假命中)           - 先查 _MEM_CACHE 与 SQLite embeddings.db
    - 长自然语言(>10字): 分词 BM25                    - 仅对缺失 chunk_hash 按 batch_size 调用 API
      + ≥2实词命中 + score≥BM25_MIN_SCORE             - L2归一化 float32，每批即刻落盘(断点可续传)
  · 输出 Top-20 有序列表 R_bm25                     · embed_query(query) (线程安全 128 条 LRU)
                                                    · numpy mat @ q_vec 余弦相似度(带维度校验)
                                                    · 父段 MaxSim 聚合 + 记录最佳命中子块偏移
                                                    · 过滤 cosine < VECTOR_MIN_SCORE (默认 0.35)
                                                    · 任何异常静默捕获 → vector_degraded=True
                                                    · 输出 Top-20 有序列表 R_vec
        │                                                      │
        └──────────────────────────┬───────────────────────────┘
                                   ▼
            [RRF 倒数排名融合]  rrf_fuse([(R_bm25, w1), (R_vec, w2)], k=60)
              · RRF(c) = I(c∈R_bm25)*w1/(60+r1) + I(c∈R_vec)*w2/(60+r2)
              · 确定性排序：(-rrf, -has_exact_lexical, -bm25, -vec, doc_order, part)
              · 片段定位：有词命中锚定关键词；纯语义命中锚定 best_subchunk
                                   │
                                   ▼
            输出 Top-3 候选卡片（标注双命中 / 精确词 / 语义相似度 + read_document 调用）
```

### 3.2 四项核心架构决策

1. **父子切块（Parent-Child Chunking）+ MaxSim 聚合**
   - `Doc.chunks`（约 2000 字）继续作为父级阅读单位，保证 `read_document(name, part)` 段号与全部旧逻辑一致。
   - 向量检索侧将每个非空父段切分为 `500` 字、重叠 `80` 字的子块；计算子块余弦相似度后按父段 `(doc_order, part)` 取最大值（MaxSim），并保存**胜出子块的文本与起始偏移**。
   - 收益：(a) 不超 512-token 模型截断；(b) 长段语义不被无关背景稀释；(c) 纯语义召回（零字面词重合）时，候选卡片摘要直接截取胜出子块而非盲截父段开头 110 字。
2. **SQLite 即开即关 + 进程内存向量缓存（防 Windows 文件锁与跨线程报错）**
   - Streamlit 每次 rerun 跑在新线程，且后台预索引运行在 daemon 线程；长期持有 `sqlite3.Connection` 极易触发 `SQLite objects created in a thread can only be used in that same thread`，Windows 下 `unittest` 的 `tearDown` 执行 `shutil.rmtree` 时也会因句柄未释放报 `WinError 32`。
   - 解法：`VectorStore` 每次 SQLite 读写（`WAL` + `busy_timeout=3000`）完毕后**立即关闭连接**；已反序列化的 `float32` 向量缓存在内存字典 `_MEM_CACHE[(db_path, model_key, chunk_hash)]` 中。热检索 0 次 SQLite I/O，冷启动与测试清理零锁冲突。
3. **动态解析 `vault.VAULT_DIR` 路径**
   - `tests/test_vault.py` 与 `tests/test_week4_ui.py` 均通过 `mock.patch.object(vault, "VAULT_DIR", tmp)` 隔离储藏区，而不是 patch `config.VAULT_DIR`。
   - 因此 `VectorStore` 的默认库路径必须在调用时动态读取 `vault.VAULT_DIR / "embeddings.db"`，不得在模块导入期写死常量路径。
4. **会话级检索隔离**
   - `embeddings.db` 按内容哈希 `chunk_hash` 共享缓存（避免不同标签页对同文档重复花钱算向量），但 `hybrid_search(query, docs)` 的候选文档集**严格限定为当前传入的 `docs`**（即当前会话 `data_tools.documents()` 或指定单文档），不同会话、临时上传与储藏区资料绝不互相泄漏。

---

## 4. 详细模块设计与接口契约

### 4.1 修改 `app/config.py`、`.gitignore`、`requirements.txt`

#### 4.1.1 `app/config.py` 新增环境变量与配置类

在 `app/config.py` 的 `QUERY_REWRITE_MAX_DECOMPOSE = 5` 下方（**务必在 `load_env()` 调用之后**，否则 `.env` 中的配置无法生效）新增：

```python
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


# 混合检索与 Embedding（默认开启混合模式；未配置向量模型时自动平滑降级为纯 BM25）
HYBRID_SEARCH_ENABLED = _env_bool("HYBRID_SEARCH_ENABLED", True)
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai_compat").strip().lower() or "openai_compat"
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "").strip()
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "").strip()
_raw_emb_dim = os.environ.get("EMBEDDING_DIMENSIONS", "").strip()
EMBEDDING_DIMENSIONS: int | None = (
    int(_raw_emb_dim) if _raw_emb_dim.isdigit() and int(_raw_emb_dim) > 0 else None
)

EMBEDDING_SUBCHUNK_CHARS = _env_int("EMBEDDING_SUBCHUNK_CHARS", 500)
EMBEDDING_SUBCHUNK_OVERLAP = _env_int("EMBEDDING_SUBCHUNK_OVERLAP", 80)
EMBEDDING_MAX_CHARS = _env_int("EMBEDDING_MAX_CHARS", 2000)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 16)
EMBEDDING_TIMEOUT = _env_int("EMBEDDING_TIMEOUT", 30)
EMBEDDING_QUERY_TIMEOUT = _env_int("EMBEDDING_QUERY_TIMEOUT", 10)

BM25_K1 = _env_float("BM25_K1", 1.5)
BM25_B = _env_float("BM25_B", 0.75)
BM25_MIN_SCORE = _env_float("BM25_MIN_SCORE", 0.5)
BM25_USE_JIEBA = _env_bool("BM25_USE_JIEBA", True)

VECTOR_MIN_SCORE = _env_float("VECTOR_MIN_SCORE", 0.35)
HYBRID_RECALL_TOP_K = _env_int("HYBRID_RECALL_TOP_K", 20)
RRF_K = _env_int("RRF_K", 60)
RRF_BM25_WEIGHT = _env_float("RRF_BM25_WEIGHT", 1.0)
RRF_VECTOR_WEIGHT = _env_float("RRF_VECTOR_WEIGHT", 1.0)
```

#### 4.1.2 独立 `EmbeddingConfig` 与 `data/embedding.json` 持久化

在 `app/config.py` 末尾新增（与 `ModelConfig` / `data/model.json` 完全平行且互不干扰）：

```python
EMBEDDING_FILE = DATA_DIR / "embedding.json"


@dataclass(frozen=True)
class EmbeddingConfig:
    """独立的向量模型配置。不复用对话 ModelConfig（如 DeepSeek 无 /embeddings 接口）。"""

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    provider: str = "openai_compat"   # "openai_compat" | "local"
    dimensions: int | None = None

    @property
    def endpoint(self) -> str:
        base = (self.base_url or "").strip().rstrip("/")
        if not base:
            return ""
        if base.endswith("/embeddings"):
            return base
        return base + "/embeddings"

    @property
    def is_local_host(self) -> bool:
        host = (urlsplit(self.base_url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    @property
    def ready(self) -> bool:
        if self.provider == "local":
            return bool(self.model)
        return bool(self.base_url and self.model and (self.api_key or self.is_local_host))

    @property
    def model_key(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions or 0}"

    @property
    def masked_key(self) -> str:
        key = self.api_key or ""
        if not key:
            return "无需填写（本机）" if self.is_local_host or self.provider == "local" else "未填写"
        if len(key) <= 8:
            return "已填写"
        return f"{key[:4]}…{key[-4:]}"
```

实现要求：


- `EMBEDDING_ENV_DEFAULTS = EmbeddingConfig(base_url=normalize_base_url(EMBEDDING_BASE_URL) if EMBEDDING_BASE_URL else "", model=EMBEDDING_MODEL, api_key=EMBEDDING_API_KEY, provider=EMBEDDING_PROVIDER, dimensions=EMBEDDING_DIMENSIONS)`（注意：**当 `EMBEDDING_BASE_URL` 为空时不要调用 `normalize_base_url("")`**，避免导入阶段抛 `ValueError`）。  
  |  |  |  
  | --- | --- |
- 配套独立内存状态 `_embedding_cfg: EmbeddingConfig | None = None` 及函数：
  - `embedding_current() -> EmbeddingConfig`
  - `set_embedding(*, base_url=None, model=None, api_key=None, provider=None, dimensions=...) -> EmbeddingConfig`（传入非空 `base_url` 时用 `normalize_base_url()` 校验协议头；`provider` 仅允许 `{"openai_compat", "local"}`）  
    |  |  |  |  
    | --- | --- | --- |
  - `save_embedding(cfg: EmbeddingConfig | None = None) -> Path | None`（使用 `.tmp` + `os.replace` 原子写入 `EMBEDDING_FILE`）
  - `clear_embedding() -> EmbeddingConfig`（删除 `EMBEDDING_FILE` 并退回 `EMBEDDING_ENV_DEFAULTS`）
  - `reload_embedding() -> EmbeddingConfig`（清内存重读磁盘；坏 JSON 时静默退回 `EMBEDDING_ENV_DEFAULTS`）

#### 4.1.3 `.gitignore` 与 `requirements.txt`

- `.gitignore`：追加一行 `data/embedding.json`。
- `requirements.txt`：
  - 必选依赖新增 `numpy>=1.24`（向量归一化、序列化与矩阵点积核心依赖）；
  - 可选依赖注释区补充 `jieba>=0.42`（中文分词增强，未安装时自动退回纯 Python bigram）与 `sentence-transformers`（可选本地离线向量模型）。

---


### 4.2 新建模块：`app/embeddings.py`

#### 4.2.1 核心数据结构

```python
class EmbeddingError(RuntimeError):
    def __init__(self, message: str, status: int = 0, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class SubChunk:
    doc_order: int     # 当前 docs 列表中的 0-based 序号
    doc_name: str
    part: int          # 对应父段号（1-based，直接用于 read_document）
    offset: int        # 本子块在父段文本中的起始字符偏移
    text: str
    chunk_hash: str    # sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass
class EmbeddingBatchResult:
    vectors: np.ndarray     # shape=(N, D), dtype=float32, 已 L2 归一化
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


# 供单测或自定义模型注入的统一契约：
# 接收非空文本列表，返回 shape=(len(texts), D) 的二维向量矩阵或嵌套列表
EmbedderCallable = Callable[[list[str]], np.ndarray | list[list[float]]]
```

#### 4.2.2 切块、哈希与向量归一化

1. **`chunk_hash(text: str) -> str`**：返回 `hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]`。
2. **`split_subchunks(text: str, size: int = EMBEDDING_SUBCHUNK_CHARS, overlap: int = EMBEDDING_SUBCHUNK_OVERLAP) -> list[tuple[int, str]]`**：
   - `text = (text or "").strip()`，为空返回 `[]`。
   - 参数防呆：`size = max(1, int(size))`，`overlap = max(0, min(int(overlap), size - 1))`（确保滑动窗口严格前进、永不死循环）。
   - 若 `len(text) <= size`，直接返回 `[(0, text)]`。
   - 否则循环切分：
     - `end = min(start + size, total)`；
     - 若 `end < total`，在 `window = text[start:end]` 中找 `cut = max(window.rfind("\n"), window.rfind("。"), window.rfind(". "), window.rfind("！"), window.rfind("？"))`，若 `cut > size * 0.5` 则 `end = start + cut + 1`；
     - 截取 `piece = text[start:end].strip()`，非空则记录 `(start, piece)`；
     - 若 `end >= total` 结束循环，否则 `start = max(start + 1, end - overlap)`。
3. **`l2_normalize(vectors: Any) -> np.ndarray`**：
   - 将输入转为 `np.asarray(vectors, dtype=np.float32)`；若为 1D（长度 `> 0`）则 `reshape(1, -1)`；非 2D 或空列数直接抛 `EmbeddingError`。
   - 校验 `np.all(np.isfinite(mat))`，若含 `NaN` / `Inf` 抛 `EmbeddingError("向量结果包含非法数值（NaN/Inf）")`。
   - 计算 `norms = np.linalg.norm(mat, axis=1, keepdims=True)`，仅对 `norms > 1e-12` 的行执行除法（零向量保持全 0，绝不产生 `NaN`）。

#### 4.2.3 Embedding 请求与本地 Provider

|  |  |
| --- | --- |
1. **`embed_texts(texts: Sequence[str], *, cfg: EmbeddingConfig | None = None, timeout: int = EMBEDDING_TIMEOUT, retries: int = 2, batch_size: int = EMBEDDING_BATCH_SIZE) -> EmbeddingBatchResult`**
   - 若 `not texts`：返回 `EmbeddingBatchResult(vectors=np.zeros((0, 0), dtype=np.float32))`。
   - 对每条文本做 `str(t or "").strip()[:EMBEDDING_MAX_CHARS]`，若有空串则立即抛 `ValueError("待向量化文本不能为空")`（防止发空串给 API 返回 400）。
   - `cfg = cfg or config.embedding_current()`；若 `not cfg.ready` 抛 `EmbeddingError("未配置可用的向量模型（Embedding）。")`。
   - **当 `cfg.provider == "openai_compat"`**：
     - 构造 URL `cfg.endpoint`（末尾为 `/embeddings`）；
     - 请求头：`{"Content-Type": "application/json"}`，**仅当 `cfg.api_key` 非空时**才设置 `"Authorization": f"Bearer {cfg.api_key}"`（兼容本机 Ollama 免 Key 访问）；
     - 请求体：`{"model": cfg.model, "input": batch}`；仅当 `cfg.dimensions is not None` 时追加 `"dimensions": int(cfg.dimensions)`；**严禁塞入 `encoding_format`**（部分国产网关或中转站不支持该参数）；
     - 重试策略：HTTP `{429, 500, 502, 503, 504}` 及 `urllib.error.URLError` / `TimeoutError` 退避重试 `retries` 次（`time.sleep(min(4, 1.5 * (attempt + 1)))`）；401/403/404/400 直接抛 `EmbeddingError(status=..., detail=...)`；
     - 响应解析：检查 `isinstance(data, dict)` 且 `isinstance(data.get("data"), list)`；**按 `(item.get("index", pos), pos)` 排序**后提取 `item["embedding"]`（OpenAI 规范允许乱序返回 `index`）；校验条数 `== len(batch)` 且每条向量非空、维度一致；累加 `usage.total_tokens`（缺省时尝试 `usage.prompt_tokens`）。
   - **当 `cfg.provider == "local"`**：
     - 延迟导入 `sentence_transformers.SentenceTransformer`，按 `cfg.model` 在进程级字典 `_LOCAL_MODELS` 缓存实例（加锁防止并发重复下载/加载）；缺库时抛 `EmbeddingError("使用本地向量模型需要安装 sentence-transformers：`pip install sentence-transformers`")`。
     - 调用 `model.encode(cleaned, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)` 并转为 `float32`。
   - 统一经 `l2_normalize` 后返回 `EmbeddingBatchResult`。

|  |  |  |
| --- | --- | --- |
2. **`embed_query(query: str, *, cfg: EmbeddingConfig | None = None, embedder: EmbedderCallable | None = None, model_key: str = "") -> np.ndarray`**
   - 检查 `query = (query or "").strip()`，空串抛 `ValueError`。
   - **线程安全 LRU 缓存**（`OrderedDict`，容量 `128`，锁保护）：
     - 若未传自定义 `embedder`：缓存键为 `(resolved_model_key, query)`；命中直接返回拷贝/只读视图；
     - 未命中时调 `embed_texts([query], cfg=cfg, timeout=EMBEDDING_QUERY_TIMEOUT, retries=1)`（查询走短超时、最多重试 1 次，避免界面卡死），取 `result.vectors[0]` 写入 LRU 并返回 1D `float32` 向量。
     - 若传入自定义 `embedder`：直接调用 `l2_normalize(embedder([query]))[0]`（不进全局缓存，防止污染真实缓存或单测间串扰）。

|  |  |
| --- | --- |
3. **`probe_embedding(base_url: str, model: str, api_key: str = "", *, provider: str = "openai_compat", dimensions: int | None = None, timeout: int = 15) -> EmbeddingProbeResult`**
   - 对标 `llm.probe()`：刻意**不抛异常**，校验参数后发一次 `"ping"` 向量化请求，返回 `EmbeddingProbeResult(ok=..., detail=..., latency=..., model=..., dim=...)`。

#### 4.2.4 持久化向量缓存 `VectorStore`

- **库路径解析**：
  ```python
  def default_db_path() -> Path:
      from . import vault
      return vault.VAULT_DIR / "embeddings.db"
  ```
|  |  |
| --- | --- |
  `VectorStore.__init__(self, db_path: Path | None = None)`：保存显式传入的 `_db_path`；为 `None` 时每次读写动态调用 `default_db_path()`。
- **SQLite 表结构**（每次 `_connect()` 确保建表存在）：
  ```sql
  PRAGMA journal_mode = WAL;
  PRAGMA busy_timeout = 3000;
  PRAGMA user_version = 1;

  CREATE TABLE IF NOT EXISTS embeddings (
      chunk_hash  TEXT NOT NULL,
      model_key   TEXT NOT NULL,
      dim         INTEGER NOT NULL,
      vector      BLOB NOT NULL,        -- little-endian float32 (<f4)，已 L2 归一化
      created_at  TEXT NOT NULL,
      PRIMARY KEY (chunk_hash, model_key)
  );
  CREATE INDEX IF NOT EXISTS ix_embeddings_model ON embeddings(model_key);
  ```
- **核心接口**：
  - `_connect(db_path)`：创建父目录、`sqlite3.connect(str(db_path), timeout=3.0)`，执行 `PRAGMA` 与 `CREATE TABLE IF NOT EXISTS`，用 `contextlib.closing` 包裹并在单次操作结束时**立刻 `close()`**。
  - `get_many(chunk_hashes: Sequence[str], model_key: str) -> dict[str, np.ndarray]`：
    - 先查内存缓存 `_MEM_CACHE[(str(db_path), model_key, h)]`；
    - 仅对未命中内存的哈希按每批 `<= 400` 个参数分批执行 `SELECT chunk_hash, dim, vector FROM embeddings WHERE model_key = ? AND chunk_hash IN (...)`；
    - 校验 `len(blob) == dim * 4`，用 `np.frombuffer(blob, dtype="<f4").copy()` 恢复（加 `.copy()` 避免只读 buffer 警告），写回 `_MEM_CACHE`。
  - `put_many(items: Sequence[tuple[str, np.ndarray]], model_key: str) -> None`：
    - 持模块级 `_LOCK` 在单个事务里 `executemany("INSERT OR REPLACE INTO embeddings ...", rows)`，blob 使用 `np.asarray(vec, dtype="<f4").tobytes()`，提交关闭后同步写入 `_MEM_CACHE`。
|  |  |  |
| --- | --- | --- |
  - `ensure_indexed(docs: Sequence[Doc], *, embedder: EmbedderCallable | None = None, cfg: EmbeddingConfig | None = None, model_key: str = "", batch_size: int = EMBEDDING_BATCH_SIZE) -> tuple[list[SubChunk], dict[str, np.ndarray], IndexStats]`：
    1. 将 `docs` 中每个非空 `chunk` 调用 `split_subchunks()` 展开为 `list[SubChunk]`；
    2. 按顺序收集唯一 `chunk_hash`；调 `get_many(unique_hashes, effective_model_key)`；
    3. 抽出缺失的 `(chunk_hash, text)`（**跨文档/同文档相同子块只算 1 次**）；
    4. 若无缺失，直接返回 `(subchunks, vectors, IndexStats(...))`；
    5. 按 `batch_size` 分批调 `embedder(batch_texts)`（若提供）或 `embed_texts(batch_texts, cfg=cfg)`，规范化为 `float32` 后**每批立即 `put_many` 落盘**（保证限流/超时中断后已完成批次永久保留，支持断点续传）；
    6. 返回 `(subchunks, vectors_by_hash, IndexStats)`。
|  |  |  |
| --- | --- | --- |
  - `preindex_async(docs: Sequence[Doc], *, cfg: EmbeddingConfig | None = None) -> threading.Thread | None`：
    - `cfg = cfg or config.embedding_current()`；若 `not cfg.ready` 或无未索引子块直接返回 `None`；
    - 使用进程级 `_INFLIGHT: set[tuple[str, str]]`（`(model_key, chunk_hash)`）去重防重入；起 `daemon=True` 后台线程执行 `ensure_indexed`，`finally` 中从 `_INFLIGHT` 清理；任何异常静默吞掉不冲击主线程。
|  |  |
| --- | --- |
  - `stats(model_key: str | None = None) -> dict[str, Any]`：返回 `{"count": int, "dims": list[int], "bytes": int}`。
  - `prune_orphans(live_hashes: set[str], model_key: str | None = None) -> int`：按需删除不再被任何文档引用的孤儿向量，并同步清理 `_MEM_CACHE`。
  - `clear_caches() -> None`（模块级函数）：清空 `_MEM_CACHE`、`_QUERY_CACHE`、`_LOCAL_MODELS` 与 `_INFLIGHT`。

---

### 4.3 新建模块：`app/bm25.py`

#### 4.3.1 数学模型：Lucene 安全非负 Okapi BM25

|  |  |  |
| --- | --- | --- |
设当前参与检索的非空文档段落数为 $N$，包含词项 $t$ 的段落数为 $df(t)$，段落长度为 $|d|$，平均段落长度为 $\text{avgdl}$：

$$\text{IDF}(t) = \ln\!\left(1 + \frac{N - df(t) + 0.5}{df(t) + 0.5}\right)$$

|  |  |  |
| --- | --- | --- |
$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t, d) \cdot (k_1 + 1)}{tf(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

- **为什么必须使用 `1 +`（Lucene 变体）**：当用户仅上传 1 份文档且只有 1~2 段（$N \le 2, df = N$）时，原始 Robertson-Sparck Jones IDF $\ln((N - df + 0.5)/(df + 0.5))$ 为**负数**，会导致命中段得分为负甚至排在后面。Lucene 变体保证 $\text{IDF}(t) > 0$ 恒成立且保留完整区分度。
- 默认参数：$k_1 = 1.5$、$b = 0.75$。

#### 4.3.2 中英混合分词器 `tokenize()`

1. **懒加载且静音的 `jieba`**：
   - 仅当 `use_jieba=True`（默认读 `BM25_USE_JIEBA`）时尝试 `import jieba`；首次导入立即调用 `jieba.setLogLevel(logging.WARNING)`，禁止其向 stderr/stdout 打印字典加载信息；
   - 若 `ImportError`，自动静默退回纯 Python 中文 2-gram 分词。
2. **分词规则**：
   - 英文 / 数字 / 编号：正则 `[a-z0-9]+(?:[._-][a-z0-9]+)*` 小写化提取整词（如 `error`、`502`、`sha-256`、`err-9012`）；若含 `.`/`_`/`-`，同时将按这些分隔符拆出的长度 $\ge 2$ 的子段加入 token 列表，兼顾精确编号与部分匹配。
   - 连续中文片段（`[\u4e00-\u9fff]+`）：
     - 若 `jieba` 可用：调用 `jieba.lcut_for_search(seg)`，去空白并过滤常见单字中文停用词（`的、了、在、是、和、与、等、中、为、对、将、被、由、及、或、之、其、于、以、从、到、把、让、给、向、按、就、都、也、还、而、但、如、若、这、那、个、一、吗、呢、啊、吧`）；如果全被过滤且原文单字有意义则保留原词。
     - 若 `jieba` 不可用或 `use_jieba=False`：对长度 $\ge 2$ 的中文片段切重叠 2-gram，单字片段保留原字（同样过滤单字停用词）。
3. **分词缓存**：模块级有序字典缓存（上限 4096 条），键为 `(use_jieba_flag, chunk_hash)`，值为 `(Counter[str], int_len)`，避免重复分词开销。

#### 4.3.3 双粒度 BM25 检索与防误召回闸门

```python
@dataclass(frozen=True)
class BM25Hit:
    doc: Doc
    doc_order: int
    part: int                    # 1-based
    score: float                 # Okapi BM25 分数
    matched_terms: tuple[str, ...]
    fuzzy: bool = False          # True = 短词整词未命中，走字面 bigram BM25 降级


@dataclass(frozen=True)
class BM25SearchResult:
    hits: tuple[BM25Hit, ...]
    fuzzy: bool = False
    query_terms: tuple[str, ...] = ()
```

实现函数 `bm25_search(query: str, docs: Sequence[Doc], *, top_k: int = HYBRID_RECALL_TOP_K, k1: float = BM25_K1, b: float = BM25_B, min_score: float = BM25_MIN_SCORE, use_jieba: bool = BM25_USE_JIEBA) -> BM25SearchResult`：

执行流程分为清晰的两阶段（兼顾 BM25 理论精度、自然语言长问句与 `tests/test_docs.py` 现有全部契约）：

1. **Stage 1：整词精确 BM25（无需等待分词冷启动，精确命中错误码、专有名词、多关键词）**
|  |  |
| --- | --- |
   - 将 `query` 按 `[\s,;，；、/|]+` 拆分为去重保序的小写检索词 `raw_terms`（保留含 1~N 字的任意精确词，如 `"税"`、`"error"`、`"违约金"`）。
   - 若 `raw_terms` 非空：对所有非空父段统计 `tf = low.count(t)` 与段字符长度 `dl = len(chunk)`，计算 Lucene IDF 与 Okapi BM25 总分（为保持"命中全词优先"，对 `matched` 覆盖全部 `raw_terms` 且 `len(raw_terms) > 1` 时给予严格区分度；实际上因每个命中词贡献独立 IDF + TF，多词命中天然远高于单词重复）。
   - 若命中至少 1 段，按 `(-score, doc_order, part)` 排序，返回前 `top_k` 条（`fuzzy=False`）。

2. **Stage 2：Stage 1 零命中时的分级 BM25 召回**
   - **分支 2A — 单一短中文词组**（`len(raw_terms) == 1`、含汉字、且 `3 <= len(raw_terms[0]) <= 10`，例如 `"区块链创新应用"`、`"共识机制"`、`"不存在的词甲乙丙"`）：
     - 拆出重叠 2-gram 列表 `grams`（去重保序）；
     - 对每段计算 2-gram 的 Okapi BM25 分数，同时强制执行两道防假命中闸门（与现有 `doc_tools.py` 逻辑等价）：
       1. 覆盖率：`len(matched_grams) >= max(2, math.ceil(len(grams) * 0.6))`；
       2. 邻近窗口：`max(first_pos) - min(first_pos) <= 48` 字符（`first_pos = [low.find(g) for g in matched_grams]`）。
     - 通过闸门的段落按 `(-score, doc_order, part)` 排序返回，设 `fuzzy=True`。
   - **分支 2B — 自然语言长问句 / 多词组合整串未命中**（例如 `"区块链如何保证历史交易不被篡改"` 或 `"违约怎么赔付"`）：
     - 调用 `tokenize(query, use_jieba=use_jieba)` 提取去重查询词项 `q_tokens`（保留词在 `query` 中至少 1 个实词）；
     - 若 `q_tokens` 非空，在所有非空父段的分词 `Counter` 上执行完整 **Okapi BM25**（使用 `dl = total_token_count`）；
     - **抗噪闸门**：至少命中 `min(2, len(q_tokens))` 个不同实词，且 `score >= min_score`（默认 `0.5`）；标记 `fuzzy=False`。

> **重要兼容性说明**：`TestSearchScopeAndRanking.test_ranking_prefers_chunk_with_more_terms` 中，原文第 1 段仅含 `"赔偿"`（出现 1 次），第 2 段含 `"违约金"`、`"赔偿"`、`"上限"`（各 1 次），BM25 下第 2 段累加 3 个词的 IDF 严格高于第 1 段的 1 个词；`test_fuzzy_does_not_invent_hits_for_nonsense`（`"不存在的词甲乙丙"`，8 字 $\le 10$）走分支 2A 被覆盖率 0.6 拦截；`test_fuzzy_rejects_scattered_bigrams`（`"共识机制"`，4 字 $\le 10$）走分支 2A 被 48 字邻近窗口拦截；`test_exact_miss_falls_back_to_fuzzy_and_says_so`（`"区块链创新应用"`，7 字 $\le 10$）走分支 2A 通过两道闸门并标注 `fuzzy=True`。

---

### 4.4 新建模块：`app/hybrid_search.py`

#### 4.4.1 运行时开关（ContextVar 会话/线程隔离）

```python
_HYBRID_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "datainsight.hybrid_enabled", default=config.HYBRID_SEARCH_ENABLED
)

def is_hybrid_enabled() -> bool:
    return bool(_HYBRID_ENABLED.get())

def set_hybrid_enabled(enabled: bool) -> None:
    _HYBRID_ENABLED.set(bool(enabled))
```

#### 4.4.2 向量召回与父段 MaxSim 聚合

```python
@dataclass(frozen=True)
class VectorHit:
    doc: Doc
    doc_order: int
    part: int
    score: float                 # 父段 MaxSim 余弦相似度
    best_offset: int             # 胜出子块在父段中的起始字符偏移
    best_subchunk: str           # 胜出子块原文


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
```

实现步骤：
1. 若 `not docs` 或 `not (query or "").strip()` 返回 `[]`。
2. 确定 `resolved_cfg = cfg or config.embedding_current()`；`resolved_key = model_key or (resolved_cfg.model_key if embedder is None else "custom")`；`store = store or VectorStore()`。
3. 调 `subchunks, vec_map, _ = store.ensure_indexed(docs, embedder=embedder, cfg=resolved_cfg, model_key=resolved_key)`；若无 `subchunks` 返回 `[]`。
4. 调 `q_vec = embed_query(query, cfg=resolved_cfg, embedder=embedder, model_key=resolved_key)`；若 `q_vec.size == 0` 或 `np.linalg.norm(q_vec) <= 1e-12` 返回 `[]`。
5. **维度对齐过滤**：仅收集 `v = vec_map.get(sc.chunk_hash)` 且 `v is not None and v.shape == q_vec.shape` 的子块，堆叠为 `(M, D)` 的 `float32` 矩阵（**绝不能因库里混有旧维度向量而触发 `ValueError: shapes not aligned`**）。
6. 矩阵点积 `sims = mat @ q_vec`（由于双方均已 L2 归一化，点积即为精确余弦相似度）。
7. **父段 MaxSim 聚合**：遍历 `(sc, float(sim))`，按 `(sc.doc_order, sc.part)` 保留最大 `sim` 及其对应的 `(sc.offset, sc.text)`。
8. 过滤 `score >= min_score`（默认 `0.35`），按 `(-score, doc_order, part)` 排序，返回前 `top_k` 条 `VectorHit`。

#### 4.4.3 通用加权 RRF 融合函数 `rrf_fuse()`

$$\text{RRF}(c) = \sum_{i=1}^{m} \mathbb{I}(c \in R_i) \cdot \frac{w_i}{k + r_i(c)}$$

```python
def rrf_fuse(
    ranked_lists: Sequence[tuple[Sequence[Any], float]],
    key_fn: Callable[[Any], Hashable],
    *,
    k: int = RRF_K,
) -> dict[Hashable, float]:
    """多路有序候选列表的加权 RRF（Reciprocal Rank Fusion）融合。

    - rank 从 1 开始；同一条路内若出现重复 key 只计首次出现的最高排名。
    - 通用 N 路设计，当前融合 BM25 + 向量 2 路，后续也可零改动复用于 multi_query 融合。
    """
    scores: dict[Hashable, float] = {}
    k = max(1, int(k))
    for items, weight in ranked_lists:
        if weight <= 0:
            continue
        seen_in_list: set[Hashable] = set()
        for rank, item in enumerate(items, start=1):
            key = key_fn(item)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            scores[key] = scores.get(key, 0.0) + (float(weight) / (k + rank))
    return scores
```

#### 4.4.4 混合检索主入口 `hybrid_search()` 与片段定位

```python
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
```

实现细节：
1. **决定执行路径**：
   - `run_bm25 = mode in {"hybrid", "bm25"}`；
   - `resolved_cfg = cfg or config.embedding_current()`；
   - `want_vector = (mode == "vector") or (mode == "hybrid" and (embedder is not None or (is_hybrid_enabled() and resolved_cfg.ready)))`。
2. **执行双路召回（带向量异常静默降级）**：
   - 若 `run_bm25`：调 `bm25_res = bm25_search(query, docs, top_k=recall_top_k)`。
   - 若 `want_vector`：在 `try ... except Exception as exc` 内调 `vec_hits = vector_search(...)`；
     - 在 `mode == "hybrid"` 下捕获任意异常（`EmbeddingError`、`TimeoutError`、`OSError`、`ValueError` 等），记 `vector_degraded = True`、`degrade_reason = str(exc)[:160]`、`vec_hits = []`，无缝保留 BM25 结果；
     - 在 `mode == "vector"`（显式测试或评测纯向量）下同样可返回降级空结果或按参数处理（建议与 hybrid 一致记录 `vector_degraded=True`，绝不让工具崩溃）。
3. **RRF 融合与确定性平手排序**：
   - 候选唯一键 `key = (doc_order, part)`（1-based `part`）。
   - 构建融合列表：`ranked_lists` 加入 `(bm25_hits, bm25_weight)`（若非空）和 `(vec_hits, vector_weight)`（若非空）。
   - 调用 `rrf_scores = rrf_fuse(ranked_lists, key_fn=lambda h: (h.doc_order, h.part), k=rrf_k)`。
   - 对每个 `key` 组装 `HybridCandidate`，排序键为：
     ```python
     key=lambda c: (
         -round(c.rrf_score, 12),
         -int(c.has_exact_lexical),
         -c.bm25_score,
         -c.vector_score,
         c.doc_order,
         c.part,
     )
     ```
4. **候选卡片片段生成 `_build_snippet(chunk, matched_terms, vec_hit)`**：
   - 若 `matched_terms` 非空 → 按首命中词位置截取 `SNIPPET_CHARS`（`110` 字，前留 `SNIPPET_LEAD = 24` 字，保留前后 `"…"`，与现有 `_snippet()` 行为一致）；
   - 否则若 `vec_hit` 非空（纯语义命中）→ 基于 `vec_hit.best_subchunk` 与 `vec_hit.best_offset` 压平空白截取前 `SNIPPET_CHARS` 字，若 `best_offset > 0` 前加 `"…"`，若后有余文加 `"…"`；
   - 兜底 → 截取 `chunk` 开头 `SNIPPET_CHARS` 字。
5. **截断并返回**：`total_candidates = len(merged)`，`candidates = tuple(merged[:max(1, int(top_k))])`。

---

### 4.5 改造 `app/tools/doc_tools.py` 与 `app/agent.py`

#### 4.5.1 `app/tools/doc_tools.py` 改造

- **保持签名完全不变**：`def search_document(keyword: str, name: str = "") -> str:`
- 将原有 `_scan()` + `_bigrams()` 替换为调用 `hybrid_search(keyword, docs, top_k=MAX_CANDIDATES)`（保留 `MAX_CANDIDATES = 3`、`SNIPPET_CHARS = 110`、`SNIPPET_LEAD = 24` 常量并供 `hybrid_search` 复用）。
- **返回文本格式化规则**（保证现有单测逐项兼容 + 混合检索信息清晰）：
  1. 空结果（`not res.candidates`）：
     - 若 `res.vector_degraded`，首行附 `（向量检索暂时不可用，已尝试 BM25 词匹配。）\n`
     - 返回 `{where}没找到「{keyword}」。\n建议：换一个更短的词、或换同义词再试；也可以 read_document 从第 1 段顺序读。`
  2. 存在候选结果：
     - 降级提示：`if res.vector_degraded: lines.append("（向量检索暂时不可用，已自动降级为 BM25 词匹配。）")`
     - 模糊提示：`elif not res.used_vector and res.fuzzy: lines.append("（整词没命中，下面是按字面模糊匹配的结果，可能是同义也可能只是字面相近，请自行判断。）")`
     - 汇总首行：
       - 当 `res.used_vector and res.used_bm25`：
         `f"{where}混合检索（向量+BM25+RRF）命中「{keyword}」，共 {res.total_candidates} 段候选，最相关的 {len(shown)} 段："`
       - 当仅 BM25（例如未配置向量或降级）：
         `f"{where}命中「{keyword}」，共 {res.total_candidates} 段候选，最相关的 {len(shown)} 段："`
       - 当仅向量：
         `f"{where}语义检索命中「{keyword}」，共 {res.total_candidates} 段候选，最相关的 {len(shown)} 段："`
     - 候选卡片行 `[rank]`：
       - 双命中（`c.bm25_rank is not None and c.vector_rank is not None`）：
         `f"\n[{rank}] `{c.doc.name}` · 第 {c.part}/{c.total_parts} 段 · 向量+BM25双命中 · 命中 {len(c.matched_terms)} 个词（{words}） · 语义 {c.vector_score:.2f}\n    {c.snippet}"`
       - 仅 BM25：
         `f"\n[{rank}] `{c.doc.name}` · 第 {c.part}/{c.total_parts} 段 · 命中 {len(c.matched_terms)} 个词（{words}）\n    {c.snippet}"`（与改造前逐字一致）
       - 仅向量（纯语义命中）：
         `f"\n[{rank}] `{c.doc.name}` · 第 {c.part}/{c.total_parts} 段 · 语义匹配（相似度 {c.vector_score:.2f}）\n    {c.snippet}"`
     - 末尾照抄指引 `read_document(name="...", part=...)`、剩余候选数提示以及 `[:MAX_OBS_CHARS]` 截断保持不变。
- 更新 `search_document` 的 docstring，向主 Agent 说明：既支持精确关键词/错误码/多词 OR，也支持自然语言描述的语义检索。

#### 4.5.2 `app/agent.py` 微调

在 `make_system_prompt()` 中将 `search_document` 策略说明（约 line 125-126）更新为：
```python
"- 已经知道要找什么（某指标、某概念、某条报错或条款）-> search_document 定位，"
"支持精确关键词与自然语言语义检索，默认搜全部资料；拿到段号再 read_document 读那一段",
```
（注意：绝不引入任何数据表相关词汇，保证 `TestPromptWithFiles.test_files_only_does_not_mention_table` 全绿。）

---

### 4.6 改造 `app/ui_streamlit.py` 与 `README.md`

1. **`_init_state()`**：新增：
   ```python
   if "hybrid_search" not in ss:
       ss.hybrid_search = config.HYBRID_SEARCH_ENABLED
   ```
2. **`_sync_hybrid_mode()`**：
   ```python
   def _sync_hybrid_mode() -> None:
       ss = st.session_state
       hybrid_search.set_hybrid_enabled(
           bool(ss.get("hybrid_search", config.HYBRID_SEARCH_ENABLED))
       )
   ```
   在 `main()` 开头（紧随 `_sync_mode()`）与 `_run_turn()` 调用 `ss.agent.ask(question)` 之前同步一次。
3. **侧边栏「状态」区（`_render_sidebar()`）**：
   - 在 `"启用查询改写"` checkbox 下方新增（必须使用 `st.checkbox`，避免 `segmented_control` 在 `AppTest` 下的中文解析 bug）：
     ```python
     st.checkbox(
         "启用混合检索（向量+BM25）",
         key="hybrid_search",
         help="结合语义向量与 BM25 精确词匹配，通过 RRF 融合排序；未配置向量模型时自动仅用 BM25。",
     )
     emb_cfg = config.embedding_current()
     if ss.get("hybrid_search"):
         if emb_cfg.ready:
             st.caption(f"向量模型 `{emb_cfg.model}`")
         else:
             st.caption("检索模式：BM25（配置向量模型后自动启用混合检索）")
     else:
         st.caption("检索模式：仅 BM25")
     ```
4. **文档上传非阻塞后台预索引**：
   - 在 `_handle_temp_upload()`（临时文档 `dt.add_document()` 成功后）和 `_handle_vault_upload()`（储藏区文档 `vault.add()` 成功且为 `KIND_DOCUMENT` 时）：
     若 `ss.get("hybrid_search", config.HYBRID_SEARCH_ENABLED)` 且 `config.embedding_current().ready`，打开对应 `Doc` 后调用 `embeddings.preindex_async([doc])`，不阻塞用户当前交互。
   - （**注意**：如需后续扩展在界面内设置向量模型，必须独立建折叠面板且默认收起，未点击展开前**绝不渲染任何 `st.text_input`**，并与 `model_panel` 互斥；本轮推荐直接通过 `.env` 或 `data/embedding.json` 配置，以免增加复杂 UI 胶水代码。）
5. **`README.md`**：在文档中补充「混合检索（Embedding + BM25 + RRF）」原理说明、`.env` 向量配置示例（智谱 / 本机 Ollama / 硅基流动）、以及 `data/vault/embeddings.db` 磁盘缓存与删库重建说明。

---

## 5. 关键边界条件与防坑清单（执行 Agent 必读）

1. **严禁向默认页面或 `_render_model_panel()` 增加 `st.text_input`**：
   - `tests/test_model_panel.py` 第 388 行断言默认收起时 `len(app.text_input) == 0`，第 396 行断言展开面板时 `labels == ["Base URL", "模型名", "API Key"]`。违背即挂。
2. **严禁碰界面文案禁区**：
   - `tests/test_week4_ui.py` 第 99–101 行禁止侧边栏/主区出现：`"会变成可查询的表"`、`"可检索的资料"`、`"还没有文件"`、`"sales.csv"`；
   - `tests/test_model_panel.py` 第 511 行禁止模型面板出现：`"改代码"`、`"重启"`、`"gitignore"`、`"data/model.json"`、`".env"`、`"兼容"`、`"400"`、`"404"`。
3. **绝不复用对话 `ModelConfig` 发 Embedding 请求**：
   - 当前主模型为 DeepSeek，无 `/embeddings` 端点；必须走独立 `EmbeddingConfig`（`.env` 中的 `EMBEDDING_*` 或 `data/embedding.json`）。
4. **绝不在 `vault.add()` 内同步发网络请求**：
   - `tests/test_vault.py` 和 `tests/test_week4_ui.py` 频繁调用 `vault.add()`；保持 `vault.add()` 为纯本地磁盘操作，异步预索引放在 UI 上传回调，兜底惰性索引放在 `vector_search` 的 `ensure_indexed()`。
5. **`VectorStore` 默认数据库路径必须动态读取 `vault.VAULT_DIR`**：
   - 测试通过 `mock.patch.object(vault, "VAULT_DIR", tmp)` 隔离目录；若模块导入期把路径写死，测试会把向量写入真实的 `data/vault/embeddings.db`。
6. **SQLite 连接绝不长期持有**：
   - 每次读写后立即关闭连接，热读走内存 `_MEM_CACHE`，避免 Windows 下 `shutil.rmtree` 报 `WinError 32` 和 Streamlit 跨线程操作 SQLite 异常。
7. **`jieba` 必须静音且允许缺包降级**：
   - 首次导入后立即 `jieba.setLogLevel(logging.WARNING)`；捕获 `ImportError` 无缝退回纯 Python CJK 2-gram。
8. **禁止发送空字符串 & 必须按 `index` 重排 API 返回**：
   - 扫描件空 PDF 的空块不得发给 `/embeddings`（否则返回 HTTP 400）；服务端响应的 `data` 列表必须按 `index` 排序再映射回对应文本。
9. **维度切换防御**：
   - 若用户将维度从 `1024` 改为 `512` 或换模型，除了 `model_key` 隔离外，`vector_search()` 在堆叠矩阵前还必须校验 `vec.shape == q_vec.shape`，杜绝 `mat @ q_vec` 抛出维度不匹配异常。

---

## 6. 测试方案与验收标准

### 6.1 测试隔离基础设施（`tests/_isolation.py` & `tests/test_docs.py`）

- 在 `tests/_isolation.py` 新增 `isolate_embeddings(prefix: str = "emb-test-") -> Path`：
  - 将 `config.EMBEDDING_FILE` patch 到临时目录；
  - 将 `config.EMBEDDING_ENV_DEFAULTS` patch 为空的 `config.EmbeddingConfig()`（`ready == False`）；
  - 调用 `config.reload_embedding()`、`embeddings.clear_caches()` 与 `hybrid_search.set_hybrid_enabled(config.HYBRID_SEARCH_ENABLED)`；
  - 用 `unittest.addModuleCleanup` 注册还原与清理。
- 在 `tests/test_docs.py` 中新增 `setUpModule()` 调用 `isolate_embeddings()`，确保即使本机 `.env` 配有真实 `EMBEDDING_API_KEY`，既有 48 项文档单测依然保持纯离线确定性。

### 6.2 新增单测文件 1：`tests/test_embeddings.py`（全离线 mock HTTP）

1. **`TestEmbeddingConfig`**：`endpoint` 拼接与防重复；`model_key` 隔离；远程无 Key 时 `ready=False`，本机 `localhost:11434` 无 Key 时 `ready=True`；`save_embedding` 原子写盘（无残留 `.tmp`）、重启 `reload_embedding` 可读、损坏 JSON 回退默认；与 `data/model.json` 完全互不干扰。
2. **`TestSubChunksAndNormalize`**：短文本不拆块；长文本断点切分及 `overlap` 校验；`size <= 0` 不死循环；空文本返回 `[]`；`l2_normalize` 后单位范数验证、零向量不生 `NaN`、含 `NaN/Inf` 抛 `EmbeddingError`。
3. **`TestEmbedTextsHTTP`**：按 `batch_size` 自动拆批；服务端乱序返回 `data[].index` 能准确还原顺序；`dimensions=None` 不发送 `dimensions` 字段；无 Key 本机地址不带空 `Authorization` 头；429/503 自动重试成功；401 / 非 JSON / 条数不符 / 维度不一致抛 `EmbeddingError`；`probe_embedding` 成功与失败均不抛异常。
4. **`TestVectorStore`**：SQLite 持久化跨重启/清内存可恢复；重复哈希跨文档复用缓存（校验 `embedder` 调用次数）；切换 `model_key` 互不串库；第 2 批模拟失败时第 1 批向量已提交落盘（断点续传）；`prune_orphans` 正确删减；`tearDown` 后 `shutil.rmtree` 在 Windows 下零文件锁异常。

### 6.3 新增单测文件 2：`tests/test_hybrid_search.py`（确定性离线可控假 Embedder）

构造正交语义簇假 Embedder（如「违约/赔/罚金/赔偿/未按期履约」→ 维度 0、「区块链/哈希/篡改/共识/账本」→ 维度 1、「量子/叠加态/纠缠」→ 维度 2、「其它噪音」→ 维度 3），覆盖 6 个测试类：

1. **`TestBM25`**：
   - 验证 Lucene 非负 IDF 在 $N=1$、$N=2$ 且 $df=N$ 时依然严格 $> 0$；
   - 验证稀有词权重高于高频词（IDF 生效）；
   - 验证长度归一化（同词频短段优于灌水长段）；
   - 验证 `jieba` 模式与 `use_jieba=False` 回退模式均正常召回；
   - 验证防假命中防线（`"不存在的词甲乙丙"` 返回空，`"共识机制"` 在 `delta.txt` 中跨 2000 字返回空，`"区块链创新应用"` 命中 `"区块链的创新应用"` 且 `fuzzy=True`）。
2. **`TestVectorSearch`**：
   - 纯语义同义查询（`"违约怎么赔"` 命中 `"未按期履约的损害赔偿责任与罚金"`，零字面重叠仍排 #1）；
   - 父段 MaxSim 验证：父段前 1500 字全是无关背景、最后 300 字包含答案 → 高分召回且 `best_subchunk` 精准落在末尾；
   - `VECTOR_MIN_SCORE` 过滤：不相关查询返回空结果；
   - 维度不匹配安全过滤：混合不同维度缓存不崩溃。
3. **`TestRRFFusion`**：
   - 精确验证 `sum(w_i / (k + r_i))` 分数计算；
   - 验证双路互补命中（BM25 #2 + Vector #2 → $2/62$）严格超越单路 #1（$1/61$）；
   - 验证权重调节与确定性平手排序；
   - 验证通用 3 路融合可用性。
4. **`TestHybridSearchIntegration`**：
   - 互补性端到端证明：构造「精确编号段 A（含 `ERR-9012`）」「纯语义段 B（同义无字面重合）」「双命中段 C」→ 验证 C 排第 1、A 与 B 互补入围，整体 Top-3 召回严格优于单独 BM25 或单独向量；
   - 静默降级：假 Embedder 抛 `EmbeddingError` / `TimeoutError` → `vector_degraded=True` 并平稳返回 BM25 结果；
   - 开关与会话隔离：`set_hybrid_enabled(False)` 时 `embedder` 零调用；会话 A 资料绝不泄露至会话 B。
5. **`TestSearchDocumentToolHybrid`**：
   - 通过 `registry.call("search_document", ...)` 验混合模式输出格式（`混合检索（向量+BM25+RRF）`、双命中标记、纯语义相似度标记、子块片段定位与 `read_document` 指引）；
   - 验证未配置 Embedding 时输出兼容且紧凑（长度 `< 1200`）。
6. **`TestUIHybridSmoke`**：
   - `AppTest` 跑通 `app.py`，断言侧边栏存在 `"启用混合检索（向量+BM25）"` checkbox，无异常，且不影响 `TestPanelUI`。

### 6.4 验收命令与通过标准

#### 离线自动化回归（必须 100% 通过）

```bash
# 1. 新增单测
python -m unittest tests/test_embeddings.py tests/test_hybrid_search.py -v

# 2. 受影响核心模块回归
python -m unittest tests/test_docs.py tests/test_vault.py tests/test_model_panel.py tests/test_week4_ui.py tests/test_query_rewrite.py -v

# 3. 全量回归（在 e:\VScode\datainsight 下使用 conda base python）
python -m unittest discover -s tests -v
```

> **已知环境提醒**：既有 `tests/test_chats.py` 中的 `TestIds.test_new_id_is_unique` 会在同一秒内生成 50 个 4 位十六进制随机 ID，存在约 1.8% 概率的生日悖论偶发撞车（非本次改动引起，单跑 `python -m unittest tests/test_chats.py -v` 即可复验）。

#### 真机联调验证（使用 `.env` 中已配置的智谱 `embedding-3`）

实施完成后，可执行一条只读联调命令（直接读取 `.env` 中的 `EMBEDDING_*` 配置，不修改代码、不把 Key 写入日志），在现有 3 份真实 PDF 上验证：
1. `embeddings.probe_embedding(...)` 连通成功且返回维度 `1024`；
2. `VectorStore.ensure_indexed(docs)` 对 3 份 PDF 的 665 个子块完成向量化并落盘 `data/vault/embeddings.db`，二次调用 `embedded == 0`（全命中缓存）；
3. 执行纯语义查询（如 `"未履行协议如何计算赔偿"`、`"如何防范交易被重复花费"` 跨语言检索 `bitcoin_en.pdf`）与精确词查询（如 `"智能合约 存证"`），确认混合检索正常输出双命中与纯语义互补结果。

---

## 7. 分步实施路线图（执行 Agent 顺序清单）

| 步骤 | 改动 / 新建文件 | 核心任务 | 单步验收 |
|---|---|---|---|
| **Step 1** | `app/config.py`<br>`.gitignore`<br>`requirements.txt` | 在 `load_env()` 后补齐 `HYBRID_SEARCH_*` / `EMBEDDING_*` / `BM25_*` / `VECTOR_*` / `RRF_*` 配置；实现 `EmbeddingConfig` 与独立 `data/embedding.json` 读写；更新 `.gitignore` 与 `requirements.txt` | `python -m unittest tests/test_model_panel.py -v` 全绿 |
| **Step 2** | `app/embeddings.py`（新建）<br>`tests/test_embeddings.py`（新建） | 实现 `SubChunk`、`split_subchunks`、`chunk_hash`、`l2_normalize`、`embed_texts`、`embed_query`（LRU）、`probe_embedding`、`VectorStore`（即开即关 SQLite WAL + `_MEM_CACHE` + 断点续传 + `preindex_async`） | `python -m unittest tests/test_embeddings.py -v` 全绿 |
| **Step 3** | `app/bm25.py`（新建） | 实现静音可选 `jieba` + CJK bigram 回退分词与缓存、Lucene 非负 IDF Okapi BM25、双粒度检索与防假命中闸门 | `tests/test_hybrid_search.py` 中 `TestBM25` 全绿 |
| **Step 4** | `app/hybrid_search.py`（新建） | 实现 `ContextVar` 开关、`vector_search`（维度校验 + `numpy` 点积 + MaxSim）、通用 `rrf_fuse`（+ 确定性排序）、`hybrid_search`（异常静默降级 + 语义片段定位） | `python -m unittest tests/test_hybrid_search.py -v` 全绿 |
| **Step 5** | `app/tools/doc_tools.py`<br>`app/agent.py`<br>`tests/_isolation.py`<br>`tests/test_docs.py` | `search_document` 接入 `hybrid_search` 与卡片格式化；更新 `agent.py` 检索策略提示；补充 `isolate_embeddings()` 并在 `test_docs.py` 启用隔离 | `python -m unittest tests/test_docs.py tests/test_hybrid_search.py tests/test_week2.py -v` 全绿 |
| **Step 6** | `app/ui_streamlit.py`<br>`README.md` | 接入 `hybrid_search` 状态与侧边栏开关、`_sync_hybrid_mode()` 同步、上传后后台 `preindex_async`；更新 `README.md`；跑通全量测试与智谱 `embedding-3` 实测联调 | `python -m unittest discover -s tests -v` 全量通过 + 实测联调通过 |

---

## 8. 异常中止与回退条件（Re-plan if）

执行 Agent 在实施过程中若遇到以下任一情况，应暂停即兴修改并重新评估：
1. 发现仓库中已合入 `app/store.py`（即 L2 已实施），需决策 `embeddings.db` 是否合并进 `index.db` 或维持独立库；
2. `tests/test_model_panel.py` 或 `tests/test_week4_ui.py` 中任意既有 UI 断言失败（重点排查是否误增默认可见的 `st.text_input` 或触碰文案禁区）；
3. `tests/test_docs.py` 中 `TestSearchScopeAndRanking` 任意一条测试失败（说明 BM25 Stage 1/2 闸门实现偏离第 4.3.3 节规格）。
