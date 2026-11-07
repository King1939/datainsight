"""把普通 Python 函数翻译成工具 schema。

注意它只翻译**结构**：参数名、类型、必填项、枚举值。
"这个工具什么时候该用"这类语义描述，仍然只能由人在 docstring 里写，
机器推不出来——这也决定了本文件的能力边界。

翻译规则：
    str        -> string
    int        -> integer
    float      -> number
    bool       -> boolean
    list[str]  -> array / items: string
    TypedDict  -> object / properties 递归 / required 取 __required_keys__
    Literal[...] -> string + enum
    X | None   -> X（不算必填，除非没有默认值）
    有默认值   -> 不进 required
    docstring 的 Args 段 -> 每个参数的 description

TypedDict 这一条是后加的：以前 `dict` 只会翻成 `{"type": "object"}`，
字段名一个都不告诉模型。参数一旦是「对象数组」（比如 PPT 的每页），
模型就只能靠猜 `title` 还是 `heading`，猜错就是一堆"缺少必填参数"。
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_ARGS_HEADERS = {"Args:", "Arguments:", "Parameters:", "参数:"}
_STOP_HEADERS = {"Returns:", "Return:", "Raises:", "Yields:", "返回值:", "返回:"}


def _is_typed_dict(annotation: Any) -> bool:
    """判断是不是 TypedDict。3.10+ 有官方判定函数，低版本靠两个属性兜底。"""
    checker = getattr(typing, "is_typeddict", None)
    if checker is not None:
        return bool(checker(annotation))
    return hasattr(annotation, "__required_keys__") and hasattr(annotation, "__annotations__")


def _json_type(annotation: Any) -> dict[str, Any]:
    """把一个类型注解翻译成 JSON Schema 片段。"""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    # TypedDict -> 带 properties 的 object。这是「对象数组」参数能用起来的前提：
    # 光说 type: object，模型不知道里面该放什么键。
    if _is_typed_dict(annotation):
        hints = get_type_hints(annotation)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {key: _json_type(value) for key, value in hints.items()},
        }
        required = [k for k in getattr(annotation, "__required_keys__", ()) if k in hints]
        if required:
            # 按声明顺序排，别让 schema 每次生成都换个样子（缓存/对比都会抖）
            schema["required"] = [k for k in hints if k in required]
        return schema

    origin = get_origin(annotation)

    # Literal["bar", "line"] -> 枚举，模型只能在里面选，出错率大幅下降
    if origin is Literal:
        values = list(get_args(annotation))
        return {"type": "string", "enum": values}

    if origin in (list, tuple, set):
        args = get_args(annotation)
        items = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": items}

    if origin is dict:
        return {"type": "object"}

    # Optional[X] / X | None —— 去掉 None 再看内层类型
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _json_type(inner[0])
        return {"type": "string"}

    return {"type": "string"}


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """拆出「整体描述」和「每个参数的描述」。

    Returns:
        (description, {参数名: 参数说明})
    """
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    desc_lines: list[str] = []
    arg_desc: dict[str, str] = {}
    in_args = False

    for line in lines:
        stripped = line.strip()
        if stripped in _ARGS_HEADERS:
            in_args = True
            continue
        if stripped in _STOP_HEADERS:
            in_args = False
            continue

        if in_args:
            matched = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
            if matched:
                arg_desc[matched.group(1)] = matched.group(2).strip()
        else:
            desc_lines.append(stripped)

    description = " ".join(part for part in desc_lines if part)
    return description, arg_desc


def build_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
    """从函数签名生成 JSON Schema 的 parameters 部分。"""
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001 - 注解里写了无法解析的字符串时兜底
        hints = {}

    _, arg_desc = parse_docstring(fn.__doc__ or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        schema = _json_type(hints.get(name, inspect.Parameter.empty))
        if name in arg_desc:
            schema["description"] = arg_desc[name]
        if param.default is not inspect.Parameter.empty:
            # 默认值是 None 时不写 "default": null。可选参数本来就不该出现在
            # required 里，再标一个 null 只会让模型以为"要显式传 null"。
            if param.default is not None:
                schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = schema

    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return parameters


def build_tool_spec(fn: Callable[..., Any], name: str | None = None) -> dict[str, Any]:
    """生成喂给模型的那种 tools 条目。"""
    description, _ = parse_docstring(fn.__doc__ or "")
    return {
        "type": "function",
        "function": {
            "name": name or fn.__name__,
            "description": description,
            "parameters": build_parameters(fn),
        },
    }
