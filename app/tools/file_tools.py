"""本地文件工具：看目录、读文件、写文本、打开文件或程序。

四个工具的边界刻意做得"笨"：没有 move / delete / rename。
删除和移动是不可逆操作里最难解释的两种，一个"整理桌面"的请求就能让模型
把文件清掉。要这些能力时宁可让用户自己动手，或者另开一个专门工具 + 确认流程。

"打开"单独拎出来，是因为它的风险性质和前面几个都不一样：读只是把内容复制进
上下文，写会改文件，而打开 .exe / .lnk 是**让系统去运行它**。
所以它在 files.open_path 里跟写入归在同一档权限（完全工作区）。

read_file 是**统一入口**：文本、PDF、Word、PPT、Excel 走同一个函数，按后缀分派。
以前这里叫 read_text_file，只能解码文本，模型拿着一个 .pptx 去读会被判成
"二进制文件"——它于是转头去 run_python 里翻 zipfile，白撞六次墙。
一个工具能读的东西越少，模型越容易绕道去用通用工具硬扛，而通用工具那条路
既没白名单里的解析库、也没有"这是第几页第几块"的结构。

写入不做备份（见 files.write_text 的说明）。工具描述里把这件事写明，
因为模型对"能不能反悔"的判断会直接影响它的做法：它会先读一遍再改。
"""

from __future__ import annotations

from pathlib import Path

from .. import files, readers
from ..config import MAX_OBS_CHARS
from . import registry

_MAX_LIST = 80


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


@registry.tool
def list_files(path: str = "") -> str:
    """列出本机某个目录下的文件和子目录。

    不确定文件名或路径时先调它，不要凭记忆猜路径——猜错只会白烧一步。
    只读模式下也能用。

    Args:
        path: 目录路径，支持 ~、桌面、反斜杠；留空表示用户主目录
    """
    try:
        rows = files.list_dir(path, limit=_MAX_LIST)
        target = files.resolve(path or "~")
    except files.FileError as exc:
        return f"错误：{exc}"

    if not rows:
        return f"`{target}` 是空目录。"

    dirs = [r for r in rows if r["is_dir"]]
    docs = [r for r in rows if not r["is_dir"]]
    lines = [f"`{target}` 下 {len(dirs)} 个目录、{len(docs)} 个文件："]
    lines += [f"  [目录] {r['name']}/" for r in dirs]
    lines += [f"  [文件] {r['name']}  {_fmt_size(r['size'])}" for r in docs]
    if len(rows) >= _MAX_LIST:
        lines.append(f"（只列了前 {_MAX_LIST} 项，需要更多请指定子目录）")
    return "\n".join(lines)


@registry.tool
def read_file(path: str, part: int = 1) -> str:
    """读一个本地文件的正文：文本、PDF、Word、PPT、Excel 都能读。

    改文件之前必须先读一遍：本项目写入不做备份，凭猜测覆盖会把原内容丢掉。
    PPT 返回的是**结构化大纲**（第几页、什么版式、第几块文字），
    那些页码和块号就是 edit_pptx 的定位依据，不要自己数。

    Args:
        path: 文件路径，支持 ~ 和 桌面 这类写法
        part: 第几段，从 1 开始；文件不长时只有 1 段
    """
    try:
        target = files.locate(path, must_exist=True)
    except files.FileError as exc:
        return f"错误：{exc}"
    if target.is_dir():
        return f"错误：`{target}` 是文件夹，不是文件。要看里面有什么，用 list_files。"

    suffix = target.suffix.lower()
    if suffix == ".pptx":
        # 延迟导入：python-pptx 是**可选**依赖，顶层 import 会让只装了 pandas
        # 的人连程序都起不来——README 里"缺了就提示装一下"的承诺就假了。
        try:
            from .pptx_tools import describe_pptx
        except ImportError:
            return "错误：读 PPT 需要 python-pptx：`pip install python-pptx`"
        try:
            head, pieces, note = describe_pptx(target)
        except Exception as exc:  # noqa: BLE001  损坏、加密、根本不是 pptx
            return f"错误：读不了 `{target.name}`（{exc}）。"
    else:
        try:
            doc = readers.extract(target, name=target.name)
            head = f"{doc.kind} · {doc.chars:,} 字 · {doc.parts} 段"
            pieces, note = doc.chunks, doc.note
        except readers.MissingDependency as exc:
            return f"错误：{exc}"
        except readers.DocumentError as exc:
            # 不认识的扩展名（.toml / .env / .ini 之类）退回按文本解码，
            # 值得一试总比直接说"不支持"强——但二进制文件要照旧拒绝。
            if suffix in readers.KIND_BY_SUFFIX:
                return f"错误：{exc}"
            try:
                text, meta = files.read_text(path)
            except files.FileError as inner:
                return f"错误：{inner}"
            head = f"{_fmt_size(meta['size'])}，{meta['chars']:,} 字"
            pieces, note = readers.split_chunks(text) or [""], ""
            if meta["clipped"]:
                note = "文件较大，只读了前面一部分"

    if not any(piece.strip() for piece in pieces):
        return f"`{target}` 里没有可读的内容：{note or '没有抽到文字'}"

    try:
        part = int(part)
    except (TypeError, ValueError):
        return f"错误：part 必须是数字，收到的是 `{part}`。"

    if part < 1 or part > len(pieces):
        return (
            f"错误：`{target.name}` 一共 {len(pieces)} 段，part 必须在 1 到 {len(pieces)} 之间，"
            f"收到的是 {part}。"
        )

    tail = f"\n\n（第 {part}/{len(pieces)} 段）"
    if part < len(pieces):
        tail += f"，后面还有 {len(pieces) - part} 段，用 part={part + 1} 继续"
    if note and part == 1:
        tail += f"\n注意：{note}"

    body = f"`{target}`（{head}）\n\n{pieces[part - 1]}{tail}"
    return body[:MAX_OBS_CHARS]


@registry.tool
def write_text_file(path: str, content: str, mode: str = "overwrite") -> str:
    """把文本写入本地文件（新建或覆盖原内容）。

    只读模式下会被拒绝；当前模式以系统提示为准。写入**不做备份**，
    mode=overwrite 会直接清掉原内容，所以不确定时先 read_file。
    不存在的父目录会自动创建。

    Args:
        path: 目标文件路径，支持 ~ 和 桌面 这类写法
        content: 要写入的完整文本（overwrite 时就是文件的最终内容）
        mode: overwrite 覆盖（默认），或 append 追加到末尾
    """
    append = str(mode).strip().lower() == "append"
    try:
        info = files.write_text(path, content, append=append)
    except files.FileError as exc:
        return f"错误：{exc}"

    action = "追加" if append else ("覆盖" if info["existed"] else "新建")
    text = (
        f"已{action} `{info['path']}`："
        f"{info['before']} 字节 → {info['after']} 字节，共写入 {len(content)} 字。"
    )
    if not append and info["existed"]:
        text += "\n注意：原内容已被替换，且没有备份。"
    return text


@registry.tool
def find_program(name: str = "") -> str:
    """在本机开始菜单里按名字查找已安装的程序（例如 微信、计算器、Chrome）。

    "打开微信"这类请求先调它拿到路径，再交给 open_file；
    不要凭印象猜 Program Files 下面的路径——猜错只会白烧一步。
    只读模式下也能用（它只查，不启动任何东西）。

    Args:
        name: 程序名的一部分，中英文都行；留空则列出开始菜单里的前 15 项
    """
    try:
        rows = files.find_programs(name)
    except files.FileError as exc:  # 目前不会抛，留个统一出口
        return f"错误：{exc}"

    if not rows:
        return (
            f"开始菜单里没有匹配 `{name}` 的程序。"
            "可以让用户确认程序名怎么写，或者用 list_files 到常见安装目录看一下。"
        )

    lines = [f"匹配 `{name}` 的程序："]
    lines += [f"  {r['name']}  ->  {r['path']}" for r in rows]
    lines.append("要启动其中一个，用 open_file，path 传上面那条路径。")
    return "\n".join(lines)


@registry.tool
def open_file(path: str) -> str:
    """用系统默认程序打开一个文件或目录；也可以打开程序（.exe / .lnk）来启动它。

    只在「完全工作区」模式可用。**用户明确说"打开"时才调用**，不要主动打开
    任何东西。打开和读取是两件事：read_file 是把内容读进上下文，
    open_file 是把控制权交给系统——后者没有结果可检查。

    Args:
        path: 文件、目录或程序的路径，支持 ~ 和 桌面 这类写法
    """
    try:
        info = files.open_path(path)
    except files.FileError as exc:
        return f"错误：{exc}"

    what = "目录" if info["kind"] == "dir" else f"文件（{info['kind']}）"
    return (
        f"已请求系统打开{what}：`{info['path']}`。"
        "窗口由系统弹出，这里看不到它是否真的打开成功，回复时不要声称已经打开。"
    )


__all__ = [
    "list_files",
    "read_file",
    "write_text_file",
    "find_program",
    "open_file",
]
