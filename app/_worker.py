"""子进程执行体。

它只做一件事：从 stdin 读一个 JSON 任务，往 stdout 写一个 JSON 结果。

为什么必须单独一个文件，而不是 `python -c "..."`：
  - 命令行拼字符串会被引号和换行搞疯；
  - 独立文件可以用 -I 干净启动，且能被单测直接调用；
  - 模型生成的代码永远只作为**数据**（stdin 里的字符串）传进来，不参与命令拼接。
    这是避免"参数注入"的基本功。

stdout 只允许出现那一个 JSON。用户代码里的 print 会被重定向到内存，混不进来——
否则模型 print 一行就像在协议里插了一句废话，父进程解析直接崩。

"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import traceback

MAX_RESULT = 4000


def _fail(
    error_type: str,
    error: str,
    lineno: int | None = None,
    line: str = "",
    printed: str = "",
) -> dict:
    return {
        "ok": False,
        "error_type": error_type,
        "error": error[:MAX_RESULT],
        "lineno": lineno,
        "line": line,
        "output": printed[:MAX_RESULT],
    }


def _locate(exc: BaseException, code: str) -> tuple[int | None, str]:
    """在 traceback 里找用户代码出错的那一行。

    只认 filename == "<string>" 的帧；worker 自己的帧全部跳过。
    """
    lineno = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<string>":
            lineno = tb.tb_lineno
        tb = tb.tb_next

    line = ""
    if lineno:
        lines = code.splitlines()
        if 0 < lineno <= len(lines):
            line = lines[lineno - 1]
    return lineno, line


def execute(task: dict) -> dict:
    code = task.get("code") or ""
    data_file = task.get("data_file") or ""
    table = task.get("table") or "dataset"

    import duckdb
    import pandas as pd

    # 完全工作区模式下可能压根没传数据集（只想改个文本文件）。
    # 这时 df 就是 None，让代码自己判断——比塞一份假数据诚实得多。
    frame = pd.read_csv(data_file) if data_file else None
    con = duckdb.connect()
    if frame is not None:
        con.register(table, frame)

    namespace = {"df": frame, "pd": pd, "con": con}
    buffer = io.StringIO()

    try:
        tree = ast.parse(code)
        body = tree.body
        last_expr = body[-1] if body and isinstance(body[-1], ast.Expr) else None
        head = body[:-1] if last_expr else body

        with contextlib.redirect_stdout(buffer):
            if head:
                module = ast.Module(body=head, type_ignores=[])
                exec(compile(module, "<string>", "exec"), namespace)  # noqa: S102

        values: list[str] = []
        if last_expr is not None:
            expr = ast.Expression(body=last_expr.value)
            ast.fix_missing_locations(expr)
            with contextlib.redirect_stdout(buffer):
                value = eval(compile(expr, "<string>", "eval"), namespace)  # noqa: S307
            if value is not None:
                values.append(str(value))
    except BaseException as exc:  # noqa: BLE001 - 含 MemoryError / RecursionError
        lineno, line = _locate(exc, code)
        return _fail(type(exc).__name__, str(exc) or "(无错误信息)", lineno, line, buffer.getvalue())

    printed = buffer.getvalue().strip()
    chunks = ([printed] if printed else []) + values
    return {"ok": True, "output": "\n".join(chunks)[:MAX_RESULT]}


def main() -> int:
    try:
        task = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error_type": "BadTask", "error": str(exc)}, ensure_ascii=False))
        return 0

    try:
        result = execute(task)
    except BaseException as exc:  # noqa: BLE001 - 环境准备阶段失败（比如 pandas 没装）
        detail = traceback.format_exc(limit=3)[-800:]
        result = _fail(type(exc).__name__, f"环境准备失败：{exc}\n{detail}")

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
