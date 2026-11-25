"""全局配置与环境变量。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
UPLOAD_DIR = DATA_DIR / "uploads"
VAULT_DIR = DATA_DIR / "vault"
CHAT_DIR = DATA_DIR / "chats"        # 对话历史（一份对话一个文件 + 一个索引）
DATA_FILE = DATA_DIR / "sales.csv"   # 测试固件，只有 tests 用，网页端不读它

# 无论上传的文件叫什么，在这个进程里都注册成同一个名字。
# 名字取中性的 dataset，不跟文件名走（换个名字就要改 prompt、改工具描述、改测试），
# 也不叫 sales —— 那是第 1 周示例订单表的遗留，别人传一份医院数据时会误导模型。
TABLE_NAME = "dataset"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VAULT_DIR.mkdir(parents=True, exist_ok=True)
CHAT_DIR.mkdir(parents=True, exist_ok=True)

# 储藏区的总容量。
VAULT_QUOTA_MB = 200

# 一次工具返回最多多少字符。不设上限的话，一次查询几万行会把上下文烧光，
# 模型反而抓不住重点——这是新手最容易忽略的成本。
MAX_OBS_CHARS = 4000

# 资料文件（pdf/docx/pptx/txt…）的三个上限。
# 为什么要在**入库**时截断而不是读取时：抽取一次就存下来，每次读都不用重新解析；
# 而一份 500 页的 PDF 全文进上下文，一轮就能把预算烧完。
MAX_DOC_CHARS = 200_000   # 单份文件最多保留多少字
DOC_CHUNK_CHARS = 2000    # read_document 一次返回多少字（约 1.5k token）
MAX_UPLOAD_MB = 20        # 单份文件大小上限

# 本地文件（files.py）的两个上限。注意这两个不是"权限限制"——
# 完全是内存保护：模型一次塞进来 50 MB 字符串，程序不该因为要写盘就 OOM。
MAX_WRITE_KB = 1024              # 单次写入上限 1 MB
MAX_READ_BYTES = 4 * 1024 * 1024  # 单次读入上限 4 MB，超出部分截断

# 单个提问的 token 预算，这一问花超了就停。防止一句话引出一轮轮试错把额度烧光。
TOKEN_BUDGET = 5000000

# 一轮里最多调几次模型。这是**兜底**，不是给正常任务设的额度。
MAX_STEPS = 100


def load_env(path: Path = ROOT / ".env") -> None:
    """读 .env（KEY=VALUE 一行一个），已有环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


# 前置查询改写（默认开启；可通过环境变量 QUERY_REWRITE_ENABLED=0 关闭）。
# 放在 load_env() 之后：.env 里写 QUERY_REWRITE_ENABLED=0 也要生效，
# 写在它前面就只认进程环境变量了。
QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", True)
QUERY_REWRITE_TIMEOUT = 15
QUERY_REWRITE_MAX_TOKENS = 600
QUERY_REWRITE_HISTORY_TURNS = 4
QUERY_REWRITE_MAX_EXPANSIONS = 6
QUERY_REWRITE_MAX_MULTI = 4
QUERY_REWRITE_MAX_DECOMPOSE = 5


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

# 多模态文档图片抽取与提交（默认开启；纯文本模型报错时会自动降级回纯文本）
MULTIMODAL_IMAGES_ENABLED = _env_bool("MULTIMODAL_IMAGES_ENABLED", True)
IMAGE_MAX_SIDE = _env_int("IMAGE_MAX_SIDE", 1568)
IMAGE_MAX_BYTES = _env_int("IMAGE_MAX_BYTES", 2 * 1024 * 1024)
IMAGE_MAX_PER_DOC = _env_int("IMAGE_MAX_PER_DOC", 24)
IMAGE_MAX_PER_TURN = _env_int("IMAGE_MAX_PER_TURN", 6)

# 联网搜索（MCP Streamable HTTP，默认走 Parallel 新版免 Key 入口；未勾选时不主动连外网）
MCP_WEB_SEARCH_ENABLED = _env_bool("MCP_WEB_SEARCH_ENABLED", False)
MCP_PARALLEL_URL = (
    os.environ.get("MCP_PARALLEL_URL", "https://mcp.parallel.ai/mcp").strip()
    or "https://mcp.parallel.ai/mcp"
)
MCP_ALLOWED_TOOLS = tuple(
    t.strip()
    for t in os.environ.get("MCP_ALLOWED_TOOLS", "web_search,web_fetch").split(",")
    if t.strip()
)
MCP_TIMEOUT = _env_int("MCP_TIMEOUT", 25)


BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("MODEL", "deepseek-chat")
API_KEY = os.environ.get("API_KEY", "")


# ---------------------------------------------------------------- 运行时模型配置
#

MODEL_FILE = DATA_DIR / "model.json"


@dataclass(frozen=True)
class ModelConfig:
    """一套完整的连接参数。冻结的，改就是换一个新对象，避免被别处偷偷改掉。

    `name` 是它在「已保存的配置」列表里的名字。空字符串表示"来自 .env 的默认那套"，
    它不在列表里，也不会被某次保存覆盖掉。
    """

    base_url: str
    model: str
    api_key: str
    name: str = ""

    @property
    def endpoint(self) -> str:
        """补出 /chat/completions。
        """
        base = (self.base_url or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    @property
    def masked_key(self) -> str:
        key = self.api_key or ""
        if not key:
            return "未填写"
        if len(key) <= 8:
            return "已填写"
        return f"{key[:4]}…{key[-4:]}"

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)


def normalize_base_url(value: str) -> str:
    """校验并规整地址。空值和漏写协议是最常见的两种填错。"""
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Base URL 不能为空。")
    if not raw.lower().startswith(("http://", "https://")):
        raise ValueError("Base URL 要以 http:// 或 https:// 开头，例如 https://api.deepseek.com/v1")
    return raw.rstrip("/")


ENV_DEFAULTS = ModelConfig(
    base_url=normalize_base_url(BASE_URL),
    model=(MODEL or "").strip(),
    api_key=(API_KEY or "").strip(),
)

# ---------------------------------------------------------------- 配置列表
#
# 换供应商是常事：今天 DeepSeek、明天智谱、后天连本地 Ollama。只存"最后一套"的话，
# 每次切回来都要把地址、模型名、Key 三样重敲一遍。所以存的是一个**列表**，
# 界面上点一行就切过去。
#
# 文件格式（data/model.json）：
#     {"active": "glm-4.7-flash",
#      "profiles": [{"name": ..., "base_url": ..., "model": ..., "api_key": ...}]}
#
# 第一版只存一套（扁平 dict），_read_store() 仍然认——升级后不用重填。

_active: ModelConfig | None = None
_profiles: list[ModelConfig] | None = None


def _parse(entry: object) -> ModelConfig | None:
    """把一个 dict 解析成配置。字段缺了、地址不合法就当这条不存在。"""
    if not isinstance(entry, dict):
        return None
    try:
        return ModelConfig(
            base_url=normalize_base_url(entry.get("base_url") or ""),
            model=str(entry.get("model") or "").strip(),
            api_key=str(entry.get("api_key") or "").strip(),
            name=str(entry.get("name") or "").strip(),
        )
    except ValueError:
        return None


def _read_store() -> tuple[str, list[ModelConfig]]:
    """读盘，返回（当前生效那条的名字, 全部配置）。

    文件坏了、少了字段都当没有——不过是回到 .env 默认，不能让页面打不开。
    """
    try:
        data = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", []
    if not isinstance(data, dict):
        return "", []

    raw = data.get("profiles")
    if raw is None:                       # 第一版的单条格式
        single = _parse(data)
        if single is None:
            return "", []
        single = replace(single, name=single.model)
        return single.name, [single]

    profiles = [c for c in (_parse(item) for item in raw) if c is not None]
    # 列表靠名字区分，没名字的一律拿模型名补上
    profiles = [p if p.name else replace(p, name=p.model) for p in profiles]

    active = str(data.get("active") or "").strip()
    return active, profiles


def _write_store(active: str, profiles: list[ModelConfig]) -> Path | None:
    """写盘。一条配置都没有就删文件，别留空壳。

    先写临时文件再整体替换：直接覆盖时若在写一半被打断（关窗口、进程被杀、断电），
    文件里是半截 JSON，下次启动读不出来就**静默**退回 .env——用户看到的是
    "我明明存过，怎么又没了"，而且没有任何报错。临时文件最多丢这一次的修改。
    """
    if not profiles:
        try:
            MODEL_FILE.unlink()
        except OSError:
            pass
        return None

    payload = json.dumps(
        {
            "active": active,
            "profiles": [
                {
                    "name": p.name,
                    "base_url": p.base_url,
                    "model": p.model,
                    "api_key": p.api_key,
                }
                for p in profiles
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_FILE.with_name(MODEL_FILE.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, MODEL_FILE)           # Windows 上也是原子的
    return MODEL_FILE


def _ensure() -> None:
    """内存里没有就从磁盘补一份。两个变量各自判空，别互相挡着。"""
    global _active, _profiles
    if _active is not None and _profiles is not None:
        return
    name, profiles = _read_store()
    if _profiles is None:
        _profiles = profiles
    if _active is None:
        _active = next((p for p in profiles if name and p.name == name), None) or ENV_DEFAULTS


def current() -> ModelConfig:
    """当前生效的那套。第一次调用时读盘，之后走内存。"""
    _ensure()
    return _active


def saved() -> list[ModelConfig]:
    """列表里的全部配置（按保存顺序）。不含 .env 默认那套。"""
    _ensure()
    return list(_profiles)


def set_current(*, base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> ModelConfig:
    """换一套参数并立即生效（只改内存，不动列表）。要留下就接着调 save()。

    校验放在这里而不是界面里：命令行、脚本、测试都走同一条路，
    免得界面上挡住了、别的入口又把 `api.deepseek.com`（没有协议头）塞进来。
    """
    base = current()
    url = normalize_base_url(base_url if base_url is not None else base.base_url)
    name = str(model if model is not None else base.model or "").strip()
    if not name:
        raise ValueError("模型名不能为空。")
    key = str(api_key if api_key is not None else base.api_key or "").strip()

    global _active
    _active = ModelConfig(base_url=url, model=name, api_key=key)
    return _active


def _label_for(cfg: ModelConfig) -> str:
    """列表里叫什么名字。默认用模型名——多数人一个模型只配一套。

    同名但地址不同（同一个模型挂在不同网关）时补上主机名，否则第二次保存会把
    第一套悄悄盖掉，而用户以为只是"又加了一条"。
    """
    label = (cfg.name or cfg.model).strip() or cfg.model
    for p in saved():
        if p.name == label and p.base_url != cfg.base_url:
            return f"{cfg.model} @ {urlsplit(cfg.base_url).netloc or cfg.base_url}"
    return label


def save(cfg: ModelConfig | None = None) -> Path | None:
    """把这一套存进列表并设为当前。同名覆盖，不新增一条。"""
    cfg = cfg or current()
    label = _label_for(cfg)
    entry = ModelConfig(cfg.base_url, cfg.model, cfg.api_key, label)

    global _active, _profiles
    merged: list[ModelConfig] = []
    replaced = False
    for p in saved():
        if p.name == label:
            merged.append(entry)          # 原位覆盖，列表顺序不跳
            replaced = True
        else:
            merged.append(p)
    if not replaced:
        merged.append(entry)

    _profiles, _active = merged, entry
    return _write_store(label, merged)


def activate(name: str) -> ModelConfig:
    """切到列表里的某一条。名字不存在就报 KeyError——界面得先刷一遍列表。"""
    match = next((p for p in saved() if p.name == name), None)
    if match is None:
        raise KeyError(name)

    global _active
    _active = match
    _write_store(match.name, _profiles or [])
    return match


def forget(name: str) -> ModelConfig:
    """从列表里删掉一条。

    删的正好是当前这条时回到 .env 默认，而不是自作主张跳到列表里的另一条——
    那等于拿别人的 Key 去发请求，还可能静默花掉额度。
    """
    global _active, _profiles
    left = [p for p in saved() if p.name != name]
    _profiles = left
    if current().name == name:
        _active = ENV_DEFAULTS
    _write_store(_active.name if _active.name else "", left)
    return _active


def clear_saved() -> ModelConfig:
    """切回 .env 那套。**列表里已保存的配置保留**——那是攒下来的，不该一起抹掉。"""
    global _active
    _active = ENV_DEFAULTS
    _write_store("", saved())
    return _active


def reload() -> ModelConfig:
    """丢掉内存里的那份，重新读盘。测试和"恢复默认"用。"""
    global _active, _profiles
    _active = None
    _profiles = None
    return current()


# ---------------------------------------------------------------- 向量模型配置
#
# 和对话模型**完全独立**：当前对话模型是 DeepSeek，它没有 /embeddings 端点，
# 把对话那套 base_url 复用过来只会得到一个 404。所以这里单独一套配置、
# 单独一个文件（data/embedding.json），互不覆盖。
#
# 配置来源的优先级：内存（本次会话里改过的）> data/embedding.json > .env。

EMBEDDING_FILE = DATA_DIR / "embedding.json"

_ALLOWED_PROVIDERS = {"openai_compat", "local"}


@dataclass(frozen=True)
class EmbeddingConfig:
    """一套向量模型连接参数。"""

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
        """能不能真的发请求。本机服务（Ollama）不需要 Key。"""
        if self.provider == "local":
            return bool(self.model)
        return bool(self.base_url and self.model and (self.api_key or self.is_local_host))

    @property
    def model_key(self) -> str:
        """缓存与落库的隔离键：换模型或换维度就是另一套向量，不能混用。"""
        return f"{self.provider}:{self.model}:{self.dimensions or 0}"

    @property
    def masked_key(self) -> str:
        key = self.api_key or ""
        if not key:
            return "无需填写（本机）" if self.is_local_host or self.provider == "local" else "未填写"
        if len(key) <= 8:
            return "已填写"
        return f"{key[:4]}…{key[-4:]}"


def _embedding_from_env() -> EmbeddingConfig:
    """从 .env 那套组装。地址为空时**不**调 normalize_base_url——那会抛 ValueError，
    导入阶段就炸，页面根本打不开。"""
    base = normalize_base_url(EMBEDDING_BASE_URL) if EMBEDDING_BASE_URL else ""
    provider = EMBEDDING_PROVIDER if EMBEDDING_PROVIDER in _ALLOWED_PROVIDERS else "openai_compat"
    return EmbeddingConfig(
        base_url=base,
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        provider=provider,
        dimensions=EMBEDDING_DIMENSIONS,
    )


EMBEDDING_ENV_DEFAULTS = _embedding_from_env()

_embedding_cfg: EmbeddingConfig | None = None


def _parse_embedding(entry: object) -> EmbeddingConfig | None:
    if not isinstance(entry, dict):
        return None
    provider = str(entry.get("provider") or "").strip().lower()
    if provider and provider not in _ALLOWED_PROVIDERS:
        return None
    dims = entry.get("dimensions")
    try:
        dimensions = int(dims) if dims not in (None, "", 0) else None
    except (TypeError, ValueError):
        dimensions = None
    base = str(entry.get("base_url") or "").strip()
    try:
        base = normalize_base_url(base) if base else ""
    except ValueError:
        return None
    return EmbeddingConfig(
        base_url=base,
        model=str(entry.get("model") or "").strip(),
        api_key=str(entry.get("api_key") or "").strip(),
        provider=provider or "openai_compat",
        dimensions=dimensions,
    )


def _read_embedding() -> EmbeddingConfig | None:
    """读盘。文件坏了就当没有，静默退回 .env。"""
    try:
        data = json.loads(EMBEDDING_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _parse_embedding(data)


def embedding_current() -> EmbeddingConfig:
    global _embedding_cfg
    if _embedding_cfg is None:
        _embedding_cfg = _read_embedding() or EMBEDDING_ENV_DEFAULTS
    return _embedding_cfg


def set_embedding(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingConfig:
    """改一套参数（只动内存）。要留下就接着调 save_embedding()。"""
    base = embedding_current()
    raw = base.base_url if base_url is None else str(base_url).strip()
    url = normalize_base_url(raw) if raw else ""
    name = str(model if model is not None else base.model or "").strip()
    key = str(api_key if api_key is not None else base.api_key or "").strip()
    want = str(provider or base.provider or "openai_compat").strip().lower()
    if want not in _ALLOWED_PROVIDERS:
        raise ValueError(f"不支持的向量模型类型：{want}")
    dims = base.dimensions if dimensions is None else dimensions

    global _embedding_cfg
    _embedding_cfg = EmbeddingConfig(
        base_url=url, model=name, api_key=key, provider=want, dimensions=dims or None
    )
    return _embedding_cfg


def save_embedding(cfg: EmbeddingConfig | None = None) -> Path | None:
    """落盘。一条内容都没有就删文件，别留空壳。写法同 model.json（临时文件 + 原子替换）。"""
    cfg = cfg or embedding_current()
    global _embedding_cfg
    _embedding_cfg = cfg

    if not (cfg.base_url or cfg.model or cfg.api_key):
        try:
            EMBEDDING_FILE.unlink()
        except OSError:
            pass
        return None

    payload = json.dumps(
        {
            "provider": cfg.provider,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "api_key": cfg.api_key,
            "dimensions": cfg.dimensions,
        },
        ensure_ascii=False,
        indent=2,
    )
    EMBEDDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = EMBEDDING_FILE.with_name(EMBEDDING_FILE.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, EMBEDDING_FILE)
    return EMBEDDING_FILE


def clear_embedding() -> EmbeddingConfig:
    """删掉存的那份，退回 .env。"""
    global _embedding_cfg
    try:
        EMBEDDING_FILE.unlink()
    except OSError:
        pass
    _embedding_cfg = EMBEDDING_ENV_DEFAULTS
    return _embedding_cfg


def reload_embedding() -> EmbeddingConfig:
    """丢掉内存那份重新读盘。测试和"恢复默认"用。"""
    global _embedding_cfg
    _embedding_cfg = None
    return embedding_current()
