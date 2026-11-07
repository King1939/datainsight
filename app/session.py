"""会话状态：把 messages 和统计数字打包在一起。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    steps: int = 0
    tokens: int = 0
    tool_calls: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def add_user(self, text: str) -> None:
        self.turns += 1
        self.add({"role": "user", "content": text})

    def stats(self) -> str:
        return (
            f"轮次 {self.turns}｜步数 {self.steps}｜工具调用 {self.tool_calls}｜"
            f"累计 tokens {self.tokens}"
        )

    def reset(self) -> None:
        """只清对话，保留统计（统计是给我看成本的）。"""
        self.messages.clear()
        self.turns = 0
        self.steps = 0
        self.tool_calls = 0

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "stats": {
                "turns": self.turns,
                "steps": self.steps,
                "tokens": self.tokens,
                "tool_calls": self.tool_calls,
            },
            "messages": self.messages,
        }

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = data.get("stats", {})
        return cls(
            id=data.get("id", "unknown"),
            messages=data.get("messages", []),
            turns=stats.get("turns", 0),
            steps=stats.get("steps", 0),
            tokens=stats.get("tokens", 0),
            tool_calls=stats.get("tool_calls", 0),
        )
