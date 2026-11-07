"""工具注册表：加一个工具 = 写一个函数 + 打一个装饰器。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..config import MAX_OBS_CHARS
from ..schema import build_parameters, parse_docstring


@dataclass
class Tool:
    name: str
    func: Callable[..., Any]
    description: str
    parameters: dict[str, Any]

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, fn: Callable[..., Any] | None = None):
        """装饰器。既可以 @registry.tool 也可以 @registry.tool()。"""

        def wrap(target: Callable[..., Any]) -> Callable[..., Any]:
            name = target.__name__
            if name in self._tools:
                raise ValueError(f"工具名重复：{name}")

            description, _ = parse_docstring(target.__doc__ or "")
            if not description:
                # 没有描述，模型就不知道什么时候该用它——这是硬性要求，直接拒绝注册
                raise ValueError(f"工具 {name} 缺少 docstring 描述")

            self._tools[name] = Tool(
                name=name,
                func=target,
                description=description,
                parameters=build_parameters(target),
            )
            return target

        return wrap(fn) if fn is not None else wrap

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> str:
        """给人看的概览：工具名 + 参数 + 描述，调 prompt 时对照用的。"""
        chunks = []
        for tool in self._tools.values():
            params = ", ".join(
                f"{p}: {s.get('type', '?')}"
                + ("" if p in tool.parameters.get("required", []) else " = 可选")
                for p, s in tool.parameters["properties"].items()
            )
            chunks.append(f"  {tool.name}({params})\n    {tool.description}")
        return "\n".join(chunks)

    def call(self, name: str, arguments: dict[str, Any] | str) -> str:
        """唯一的执行入口。永远返回字符串，永远不抛异常。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：不存在工具 {name}。可用工具：{', '.join(self.names())}"

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return f"错误：参数不是合法 JSON（{exc}）。请重新生成。"
        if not isinstance(arguments, dict):
            return "错误：参数必须是一个 JSON 对象。"

        allowed = set(tool.parameters["properties"])
        unknown = set(arguments) - allowed
        if unknown:
            return (
                f"错误：工具 {name} 不认识参数 {sorted(unknown)}。"
                f"它只接受这些参数：{sorted(allowed)}"
            )

        missing = [
            p for p in tool.parameters.get("required", []) if p not in arguments
        ]
        if missing:
            return f"错误：缺少必填参数 {missing}。"

        try:
            result = tool.func(**arguments)
        except TypeError as exc:
            return f"参数类型错误：{exc}"
        except Exception as exc:  # noqa: BLE001 - 故意吞掉，转成观察结果
            return f"工具执行异常（{type(exc).__name__}）：{exc}"

        text = result if isinstance(result, str) else str(result)
        if len(text) > MAX_OBS_CHARS:
            text = text[:MAX_OBS_CHARS] + f"\n……（已截断，原文 {len(text)} 字符）"
        return text
