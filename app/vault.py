"""文件储藏区：用户长期留在这里的文件。

和 data/uploads 的分工：那里是"这次会话的工作副本"（按 sid 分目录，随时可丢），
这里是资料库——重启、换标签页、下周回来，文件还在，不用重传也能问。

磁盘布局：

    data/vault/
        index.json          元信息（名字、大小、类型、字数），唯一的真相
        files/<id><后缀>     原始文件，id 是内容的 sha1 前 16 位
        text/<id>.txt       资料抽出来的正文，抽一次存一次，别每次开页面重抽

（"../../.env" 也只会变成一个显示用的字符串），同名文件也不会互相踩。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import readers
from .config import MAX_UPLOAD_MB, VAULT_DIR, VAULT_QUOTA_MB
from .readers import Doc, DocumentError

KIND_TABLE = "table"
KIND_DOCUMENT = "document"
INDEX_VERSION = 1

_LOCK = threading.Lock()
_DOCS: dict[str, Doc] = {}


class VaultFull(RuntimeError):
    """放不下了。消息是给用户看的，要说清楚现在多大、上限多少、该怎么办。"""


@dataclass(frozen=True)
class Entry:
    id: str  # 唯一，界面拿它当 key、当"当前表"的指针
    file: str  # 内容哈希，决定磁盘文件名：同内容存两个名字也只占一份空间
    name: str
    size: int
    suffix: str
    kind: str
    added_at: str
    chars: int = 0
    parts: int = 0
    note: str = ""

    @property
    def path(self) -> Path:
        return _files_dir() / f"{self.file}{self.suffix}"

    @property
    def text_path(self) -> Path:
        return _text_dir() / f"{self.file}.txt"

    @property
    def is_table(self) -> bool:
        return self.kind == KIND_TABLE


# ---------------------------------------------------------------- 磁盘


def _files_dir() -> Path:
    path = VAULT_DIR / "files"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _text_dir() -> Path:
    path = VAULT_DIR / "text"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path() -> Path:
    return VAULT_DIR / "index.json"


def _load() -> list[Entry]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    items: list[Entry] = []
    for raw in data.get("entries") or []:
        try:
            items.append(Entry(**raw))
        except TypeError:
            continue
    return items


def _save(items: list[Entry]) -> None:
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"version": INDEX_VERSION, "entries": [asdict(e) for e in items]}
    _index_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _forget(entry: Entry, remaining: list[Entry]) -> None:
    """删文件，但只在没人共用它的时候删。

    同一份内容用两个名字存过，就是两条 entry 指一个文件；删一条不能把另一条弄坏。
    """
    _DOCS.pop(str(entry.path), None)
    if any(e.file == entry.file for e in remaining):
        return
    entry.path.unlink(missing_ok=True)
    entry.text_path.unlink(missing_ok=True)


# ---------------------------------------------------------------- 读


def entries() -> list[Entry]:
    """储藏区里的文件，新的在前。

    顺序按 index 里的追加顺序倒着排，不按时间戳：Windows 上时钟精度只有十几毫秒，
    连着存两个文件会拿到同一个时间，排序就不稳定了。
    """
    with _LOCK:
        items = _load()
    return list(reversed(items))


def tables() -> list[Entry]:
    return [e for e in entries() if e.kind == KIND_TABLE]


def documents() -> list[Entry]:
    return [e for e in entries() if e.kind == KIND_DOCUMENT]


def find(entry_id: str) -> Entry | None:
    return next((e for e in entries() if e.id == entry_id), None)


def usage() -> int:
    """实际占了多少字节。按内容去重：一份文件存两个名字只算一次。"""
    return _used_bytes(entries())


def _used_bytes(items: list[Entry]) -> int:
    sizes: dict[str, int] = {}
    for entry in items:
        sizes.setdefault(entry.file, entry.size)
    return sum(sizes.values())


def quota() -> int:
    return int(VAULT_QUOTA_MB) * 1024 * 1024


def open_document(entry: Entry) -> Doc | None:
    """把储藏区的资料变成当前会话能读的 Doc。owned=False：文件不归会话管。"""
    if entry.kind != KIND_DOCUMENT:
        return None

    key = str(entry.path)
    cached = _DOCS.get(key)
    if cached is not None:
        return cached

    try:
        text = entry.text_path.read_text(encoding="utf-8")
    except OSError:
        return None

    doc = Doc(
        name=entry.name,
        kind=entry.suffix.lstrip(".") or "text",
        text=text,
        note=entry.note,
        path=key,
        owned=False,
    )
    _DOCS[key] = doc
    return doc


def read_table(entry: Entry) -> pd.DataFrame | None:
    if entry.kind != KIND_TABLE:
        return None
    try:
        raw = entry.path.read_bytes()
    except OSError:
        return None
    return readers.read_table(raw)


# ---------------------------------------------------------------- 写


def safe_name(name: str) -> str:
    """只留最后一段文件名，去掉 Windows 不允许的字符。纯粹为了显示和工具寻址。"""
    base = Path(str(name).replace("\\", "/")).name.strip() or "未命名"
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base)
    return base[:120]


def _megabytes(num: int) -> str:
    return f"{num / (1024 * 1024):.1f} MB"


def add(name: str, raw: bytes) -> Entry:
    """收一份文件。同名视为新版本，旧的那份被替换掉。

    为什么同名要替换而不是并存：工具是按文件名找资料的，两条同名 entry
    在会话里只会留下一条，界面上却显示两行——用户会对着一个"明明存在"的文件
    让模型去读，模型说没有。 
    """
    if not raw:
        raise DocumentError("文件是空的（0 字节）。")

    name = safe_name(name)
    suffix = Path(name).suffix.lower()
    route = readers.route(name)
    if route == "unsupported":
        raise DocumentError(f"不支持 {suffix or '无后缀'} 这种格式。")

    size = len(raw)
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentError(f"文件超过 {MAX_UPLOAD_MB} MB。先压缩，或者拆成几份再传。")

    kind = KIND_TABLE if route == "table" else KIND_DOCUMENT
    digest = hashlib.sha1(raw).hexdigest()[:16]

    with _LOCK:
        items = _load()
        same = [e for e in items if e.name == name]
        if same and same[0].file == digest:
            return same[0]

        rest = [e for e in items if e.name != name]
        used = _used_bytes(rest)
        grows = 0 if any(e.file == digest for e in rest) else size
        if used + grows > quota():
            raise VaultFull(
                f"储藏区放不下了：已用 {_megabytes(used)}，这份 {_megabytes(size)}，"
                f"上限 {VAULT_QUOTA_MB} MB。先删几份不用的再传。"
            )

        entry_id, index = digest, 2
        while any(e.id == entry_id for e in rest):
            entry_id = f"{digest}-{index}"
            index += 1

        target = _files_dir() / f"{digest}{suffix}"
        target.write_bytes(raw)

        entry = Entry(
            id=entry_id,
            file=digest,
            name=name,
            size=size,
            suffix=suffix,
            kind=kind,
            added_at=datetime.now().isoformat(timespec="seconds"),
        )

        if kind == KIND_DOCUMENT:
            try:
                doc = readers.extract(target, name=name)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            entry.text_path.write_text(doc.text, encoding="utf-8")
            entry = replace(entry, chars=doc.chars, parts=doc.parts, note=doc.note)

        rest.append(entry)
        _save(rest)
        for old in same:
            _forget(old, rest)

    return entry


def drop(entry_id: str) -> bool:
    with _LOCK:
        items = _load()
        target = next((e for e in items if e.id == entry_id), None)
        if target is None:
            return False
        remaining = [e for e in items if e.id != entry_id]
        _save(remaining)
        _forget(target, remaining)
    return True


__all__ = [
    "Entry",
    "KIND_DOCUMENT",
    "KIND_TABLE",
    "VaultFull",
    "add",
    "documents",
    "drop",
    "entries",
    "find",
    "open_document",
    "quota",
    "read_table",
    "safe_name",
    "tables",
    "usage",
]
