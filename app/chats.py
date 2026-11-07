"""对话历史：把"问过什么、答了什么"存到磁盘，刷新和重启都还在。

和 vault 的分工很干脆：**vault 存文件，这里存对话。** 一个解决"资料不用重传"，
一个解决"问题不用重问"。

磁盘布局：

    data/chats/index.json     摘要（id / 标题 / 时间 / 轮次 / 表名），列表只读它
    data/chats/<id>.json      完整内容（turns + messages）

为什么拆成两个文件：每次重跑页面都要列一遍历史。`messages` 里带着工具返回的
大段文本，一个对话可能上百 KB；30 个对话全读一遍，光列表就要几百毫秒。
摘要是几百字节，读它足够画列表了。

**内容是真相，索引只是缓存。** index.json 丢了或写坏了，`list_chats()` 会扫一遍
内容文件重建它——宁可慢一次，也不能因为一个缓存文件让用户的历史凭空消失。

存两份东西是有原因的：
  - `turns` 是给人看的（问题、答案、思考过程、图表路径）
  - `messages` 是给模型看的（含 tool_calls 和工具返回），续聊必须靠它
少了后者，恢复出来的对话"记得自己说过什么"但"忘了查过什么"，下一句就会自相矛盾。
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CHAT_DIR

# 最多留多少个对话。到顶之后删最久没动过的那个——按 updated_at，不按创建时间：
# 一周前创建、今天还在聊的那个对话，显然比昨天创建、只问了一句的更该留。
KEEP = 50
INDEX_VERSION = 1

_LOCK = threading.Lock()


@dataclass
class Chat:
    id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    table: dict[str, Any] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.turns)

    def summary(self) -> dict[str, Any]:
        """列表页要的那几项。刻意不带 turns / messages。"""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "count": self.count,
            "table": (self.table or {}).get("name", ""),
        }


# ---------------------------------------------------------------- 路径与读写


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    """带时间的 id：翻 data/chats/ 的时候一眼能看出先后。"""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def _content_path(chat_id: str) -> Path:
    return CHAT_DIR / f"{_safe_id(chat_id)}.json"


def _index_path() -> Path:
    return CHAT_DIR / "index.json"


def _safe_id(chat_id: str) -> str:
    """id 会变成文件名，所以只允许安全字符。

    id 是我们自己生成的，但 load() 的入参来自 URL 查询参数——那是用户能改的。
    不挡的话 `?c=../../.env` 就能读写到别的目录去。
    """
    text = "".join(ch for ch in str(chat_id) if ch.isalnum() or ch in "-_")
    return text[:64] or "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """先写临时文件再换名。写到一半断电，也不会留下半个 JSON。"""
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- 索引


def _read_index() -> list[dict[str, Any]] | None:
    data = _read_json(_index_path())
    if data is None:
        return None
    items = data.get("chats")
    return items if isinstance(items, list) else None


def _write_index(items: list[dict[str, Any]]) -> None:
    _write_json(
        _index_path(),
        {"version": INDEX_VERSION, "chats": items},
    )


def _rebuild_index() -> list[dict[str, Any]]:
    """扫一遍内容文件，重新生成摘要。只在索引缺失/损坏时发生一次。"""
    items: list[dict[str, Any]] = []
    for path in CHAT_DIR.glob("*.json"):
        if path.name == "index.json":
            continue
        chat = _read_content(path)
        if chat is not None:
            items.append(chat.summary())
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    try:
        _write_index(items)
    except OSError:
        pass  # 写不了索引不影响这次读取，下次再试
    return items


def _read_content(path: Path) -> Chat | None:
    data = _read_json(path)
    if not data:
        return None
    try:
        return Chat(
            id=str(data["id"]),
            title=str(data.get("title") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            table=dict(data.get("table") or {}),
            turns=list(data.get("turns") or []),
            messages=list(data.get("messages") or []),
            stats=dict(data.get("stats") or {}),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 对外


def save(
    chat_id: str,
    *,
    title: str = "",
    table: dict[str, Any] | None = None,
    turns: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    stats: dict[str, int] | None = None,
) -> Chat:
    """写一份对话。

    不追加、不合并，整份覆盖：调用方手里已经有完整状态（turns + messages），
    增量更新的复杂度换不来什么。
    """
    chat_id = _safe_id(chat_id)
    turns = list(turns or [])
    messages = list(messages or [])

    with _LOCK:
        old = _read_content(_content_path(chat_id))
        chat = Chat(
            id=chat_id,
            title=(title or (old.title if old else "") or "未命名对话")[:40],
            created_at=(old.created_at if old else "") or _now(),
            updated_at=_now(),
            table=dict(table or {}),
            turns=turns,
            messages=messages,
            stats=dict(stats or {}),
        )
        _write_json(_content_path(chat_id), asdict(chat))

        items = [i for i in (_read_index() or []) if i.get("id") != chat_id]
        items.insert(0, chat.summary())
        _write_index(items)
        _prune(items)

    return chat


def load(chat_id: str) -> Chat | None:
    return _read_content(_content_path(chat_id))


def list_chats(limit: int | None = None) -> list[dict[str, Any]]:
    """摘要列表，最近更新的在前。索引坏了就重建一次。"""
    items = _read_index()
    if items is None:
        items = _rebuild_index()

    # 索引里可能有"内容文件已经不在了"的条目（手动删了文件、上次删到一半）。
    # 列表按索引画，但点了会打开一个空对话——所以这里顺手对一遍账。
    alive = [
        i for i in items
        if isinstance(i, dict) and i.get("id") and _content_path(str(i["id"])).exists()
    ]
    if len(alive) != len(items):
        try:
            _write_index(alive)
        except OSError:
            pass

    alive.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return alive[:limit] if limit else alive


def delete(chat_id: str) -> bool:
    chat_id = _safe_id(chat_id)
    with _LOCK:
        path = _content_path(chat_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        items = [i for i in (_read_index() or []) if i.get("id") != chat_id]
        _write_index(items)
    return existed


def _prune(items: list[dict[str, Any]]) -> None:
    """超出上限就删最久没动过的。调用时已持有 _LOCK。"""
    if len(items) <= KEEP:
        return
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    keep, drop = items[:KEEP], items[KEEP:]
    _write_index(keep)
    for item in drop:
        _content_path(str(item.get("id"))).unlink(missing_ok=True)


__all__ = [
    "Chat",
    "KEEP",
    "delete",
    "list_chats",
    "load",
    "new_id",
    "save",
]
