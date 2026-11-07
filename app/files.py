"""本地文件的读写：一个模式开关 + 一套路径规则。

为什么要单独一层，而不是把逻辑塞进工具函数：

    工具层只负责"把结果翻译成模型看得懂的文本"——和 code_tools 转发 sandbox
    是同一种分工。路径归一化、只读判断、编码探测写在工具里，函数会长到读不懂，
    也没法脱离注册表单独测试。

两种模式（输入框上方切换）：

    read  —— 只读。写操作一律拒绝；沙箱也不允许 to_csv / savefig 这类落盘方法。
             注意这条必须两边都堵：只拦工具、不拦 run_python，开关就是假的。
             **也不能打开文件**——见下面 open_path 的说明。
    write —— 完全工作区。不限制路径，写入**不做备份**（用户明确要求），
             并且只有这个模式能打开文件、启动程序。

为什么"打开"也算写权限那一档：os.startfile 对 .exe / .lnk 是**直接运行**，
不是"查看"。微信那种需求必须走到运行这一步，所以没法把"打开文档"和"启动程序"
拆成两个安全等级——一旦允许任意后缀，.exe 就在里面。与其按后缀维护一张
"安全清单"（.docm 带宏、.url 会联网、.hta 直接是脚本，永远列不全），
不如复用同一道开关：切过去就是想操作本机，切回来就什么都不碰。

边界要说清楚：write 模式下这就是"把本机文件系统的写权限交给模型"。
没有备份、没有 diff 确认、没有目录白名单。它适合自己用的本机工具，
不适合对外部署——多用户场景里这个开关等于把 RCE 摆到首页。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import MAX_READ_BYTES, MAX_WRITE_KB
from .readers import decode_text

MODE_READ = "read"
MODE_WRITE = "write"
MODE_LABELS = {MODE_READ: "只读", MODE_WRITE: "完全工作区"}

# 桌面/文档这类中文目录名，模型和用户都会这么写。别名只做前缀替换，
# 不做模糊匹配——"桌面/a.txt" 能认，"我桌面那个.txt" 不能认，也不该认。
_DESKTOP_ALIASES = ("桌面", "desktop")

_mode = MODE_READ


def desktop_dir() -> Path:
    """真实的桌面目录。

    不能直接写 `Path.home() / "Desktop"`：中文版 Windows 上这个目录可能
    真的叫「桌面」，开了 OneDrive 备份额外还会重定向到
    `OneDrive\\Desktop`。两种情况都会让「桌面/test.txt」指向一个不存在的目录。
    SHGetFolderPathW 问的是系统本身，三种情况都答对。
    """
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(260)
            # CSIDL_DESKTOPDIRECTORY = 0x10，失败会返回非 0
            if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0:
                if buf.value:
                    return Path(buf.value)
        except (AttributeError, OSError):
            pass  # 拿不到就退回默认猜测，不让它影响读文件
    return Path.home() / "Desktop"


class FileError(Exception):
    """读写失败。消息是**给模型看**的，要能指导它下一步怎么改。"""


def set_mode(mode: str) -> None:
    if mode not in MODE_LABELS:
        raise ValueError(f"未知模式：{mode}")
    global _mode
    _mode = mode


def get_mode() -> str:
    return _mode


def allow_write() -> bool:
    return _mode == MODE_WRITE


def mode_label() -> str:
    return MODE_LABELS[_mode]


# ---------------------------------------------------------------- 路径


def resolve(path: str) -> Path:
    """把模型给的路径变成绝对路径。

    要吃掉这些写法，否则模型每次都会报"文件不存在"，然后开始瞎猜：
        ~/Desktop/a.txt        C:/Users/x/Desktop/a.txt
        \\Users\\x\\Desktop\\a.txt   桌面/a.txt        a.txt（相对主目录）
    """
    raw = str(path or "").strip().strip('"').strip("'")
    if not raw:
        raise FileError("路径不能为空。")

    raw = raw.replace("\\", "/")
    head, _, rest = raw.partition("/")
    if head.lower() in _DESKTOP_ALIASES:
        base = desktop_dir()
        raw = str(base / rest) if rest else str(base)

    expanded = os.path.expandvars(os.path.expanduser(raw))
    target = Path(expanded)
    if not target.is_absolute():
        target = Path.home() / target
    try:
        # resolve 会同时处理 ".." 和软链接：/tmp/link -> /etc 会被展开成 /etc
        return target.resolve()
    except OSError as exc:
        raise FileError(f"路径无法解析（{exc}）：{path}") from exc


def _bare(path: str) -> bool:
    """只有文件名，没有任何目录线索：没有分隔符、没有盘符、也没有 ~。"""
    raw = str(path or "").strip().strip('"').strip("'")
    return bool(raw) and "/" not in raw and "\\" not in raw and ":" not in raw and not raw.startswith("~")


def locate(path: str, *, must_exist: bool) -> Path:
    """resolve + 处理「只给了文件名」这个最容易错位的情况。

    这一层兜的不是语法，而是**意图**。用户说的是「我桌面上的 test.txt」，
    模型很可能只传 `test.txt`，而 resolve 会让它落到主目录——
    于是它要么在桌面找不到就说文件不存在，要么更糟：在主目录新建一个
    同名文件，然后回报「已写入」。后者不报错、不显眼，用户只看到
    「桌面上没变化」。

    规则只有三条，都不靠猜：
      1. 主目录里真有这个文件 -> 就是它（和 resolve 一致）
      2. 主目录没有、桌面有   -> 用桌面那份（唯一存在的文件，谈不上歧义）
      3. 两边都没有           -> 拒绝，并在消息里写清两种写法
    """
    target = resolve(path)
    if not _bare(path):
        # 有目录线索就直接认这个路径，不再往桌面找——但"必须存在"这一条仍然成立。
        # 少了这一句，绝对路径写错时不会报错，而是一路走到 os.startfile 才炸，
        # 报出来的是"没有关联的程序"，跟真正的原因（路径不存在）完全对不上。
        if must_exist and not target.exists():
            raise FileError(
                f"不存在：{target}。先用 list_files 看一眼上级目录里到底有什么，不要凭猜。"
            )
        return target
    if target.exists():
        return target

    desk = (desktop_dir() / Path(target.name)).resolve()
    if desk.exists():
        return desk
    if must_exist:
        raise FileError(
            f"没找到 `{Path(target.name).name}`，找过这两个位置：{target}、{desk}。"
            "如果是桌面的文件，写成 `桌面/文件名`；如果是主目录，写 `~/文件名`。"
        )
    raise FileError(
        f"只给了文件名 `{target.name}`，不确定该写到哪个目录"
        f"（默认会落在主目录 {target}）。"
        "请写清目录：桌面上的写成 `桌面/文件名`，主目录的写成 `~/文件名`。"
    )


def _check_write_size(text: str) -> None:
    size = len(text.encode("utf-8"))
    if size > MAX_WRITE_KB * 1024:
        raise FileError(
            f"内容 {size / 1024:.0f} KB，超过单次写入上限 {MAX_WRITE_KB} KB。"
            "请分批写入（先写主体，再用 mode='append' 追加）。"
        )


# ---------------------------------------------------------------- 读


def read_text(path: str, limit: int = 20_000) -> tuple[str, dict[str, Any]]:
    """读文本文件。返回 (正文, 元信息)。二进制文件直接拒绝。"""
    target = locate(path, must_exist=True)
    if not target.exists():
        raise FileError(f"文件不存在：{target}")
    if target.is_dir():
        raise FileError(f"{target} 是文件夹，不是文件。要看里面有什么，用 list_files。")

    size = target.stat().st_size
    with target.open("rb") as handle:
        raw = handle.read(MAX_READ_BYTES)

    text = decode_text(raw)
    if not text and raw:
        raise FileError(f"{target.name} 看起来是二进制文件，不能当文本读。")

    clipped_bytes = size > MAX_READ_BYTES
    clipped_chars = len(text) > limit
    meta = {
        "path": str(target),
        "size": size,
        "chars": len(text),
        "empty": not text,
        "clipped": clipped_bytes or clipped_chars,
    }
    return text[:limit], meta


def read_lines(path: str, start: int = 1, end: int = 80) -> tuple[list[str], dict[str, Any]]:
    """按行号读一段。大文件用它，比整份读便宜得多。"""
    text, meta = read_text(path, limit=200_000)
    lines = text.splitlines()
    start = max(1, int(start))
    end = max(start, int(end))
    meta["total_lines"] = len(lines)
    return lines[start - 1 : end], meta


# ---------------------------------------------------------------- 写


def write_text(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """写文件。只读模式直接拒绝。

    **不做备份**：这是明确的设计选择，不是漏了。想留退路的话，
    让模型先 read_file 把原内容读出来——读不到就没法改。
    """
    if not allow_write():
        raise FileError(
            "当前是只读模式，不能写入文件。"
            "请在输入框上方把模式切到「完全工作区」再试。"
        )

    _check_write_size(content)
    target = locate(path, must_exist=False)
    if target.is_dir():
        raise FileError(f"{target} 是文件夹，不能写入。")

    existed = target.exists()
    before = target.stat().st_size if existed else 0
    target.parent.mkdir(parents=True, exist_ok=True)

    # newline="\n"：Windows 默认会把 \n 翻成 \r\n，同一份内容在两台机器上
    # 字节数不同，回写给模型的"写了多少字节"就对不上了。
    with target.open("a" if append else "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)

    return {
        "path": str(target),
        "existed": existed,
        "before": before,
        "after": target.stat().st_size,
        "append": append,
    }


# ---------------------------------------------------------------- 列目录


def list_dir(path: str = "", limit: int = 80) -> list[dict[str, Any]]:
    target = resolve(path or str(Path.home()))
    if target.is_file():
        raise FileError(f"{target} 是文件，不是目录。要读内容用 read_file。")
    if not target.exists():
        raise FileError(f"目录不存在：{target}")

    rows: list[dict[str, Any]] = []
    try:
        children = sorted(target.iterdir(), key=lambda c: (c.is_file(), c.name.lower()))
    except OSError as exc:
        raise FileError(f"读不了这个目录（{exc}）：{target}") from exc

    for child in children:
        try:
            size = child.stat().st_size if child.is_file() else 0
        except OSError:  # 权限不足的条目跳过，不让一个坏文件拖垮整个列表
            continue
        rows.append({"name": child.name, "is_dir": child.is_dir(), "size": size})
    return rows[:limit]


# ---------------------------------------------------------------- 打开


def _launch(target: Path) -> None:
    """真正把东西拉起来。单独成一个函数，是为了**能在测试里换掉**。

    测试绝不能真的启动程序：跑一遍测试就弹出三个微信窗口、外带一个记事本，
    比起"测不过"更糟——那会让人不敢再跑测试。
    """
    if os.name == "nt":
        os.startfile(str(target))  # noqa: S606  Windows 专用，参数已校验过
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.run([opener, str(target)], check=False)


def open_path(path: str) -> dict[str, Any]:
    """用系统默认程序打开文件或目录；也能打开 .exe / .lnk 来启动程序。

    和读取的区别：读取是把内容拿进上下文，打开是把控制权交给系统。
    后者没有返回值可检查——记事本崩了、微信没登录，这边都看不见。
    所以工具层的措辞只能是"已请求系统打开"，不能说"已成功打开"。
    """
    if not allow_write():
        raise FileError(
            "当前是只读模式，不能打开文件或程序。"
            "打开 .exe / .lnk 等于运行它，所以这一项和写入归在同一档；"
            "请先在输入框上方切到「完全工作区」。"
        )

    target = locate(path, must_exist=True)
    try:
        _launch(target)
    except OSError as exc:
        raise FileError(
            f"系统没能打开 `{target.name}`（{exc}）。"
            "可能是没有关联的默认程序，或者这个文件已经损坏。"
        ) from exc

    return {
        "path": str(target),
        "kind": "dir" if target.is_dir() else (target.suffix.lower() or "无后缀"),
    }


# ---------------------------------------------------------------- 找程序


def _start_menu_dirs() -> list[Path]:
    """开始菜单里放快捷方式的两处：当前用户 + 所有用户。

    只翻 .lnk，不扫 Program Files 里的 .exe——后者要遍历几万个文件，
    而且会把一堆卸载程序、更新器也翻出来，模型选错的概率更高。
    用户在开始菜单里能看见的东西，就是这台机器"装了什么"的合理答案。
    """
    raw = [
        os.environ.get("APPDATA"),
        os.environ.get("ProgramData"),
    ]
    dirs = []
    for base in raw:
        if not base:
            continue
        path = Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        if path.is_dir():
            dirs.append(path)
    return dirs


def find_programs(name: str = "", limit: int = 15) -> list[dict[str, str]]:
    """按名字在开始菜单里找程序。返回快捷方式的路径，交给 open_path 去启动。"""
    keyword = str(name or "").strip().lower()

    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for base in _start_menu_dirs():
        try:
            shortcuts = base.rglob("*.lnk")
            for path in shortcuts:
                haystack = f"{path.stem} {path}".lower()
                if keyword and keyword not in haystack:
                    continue
                key = (path.stem.lower(), str(path))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"name": path.stem, "path": str(path)})
        except OSError:  # 某个目录读不动就跳过，不影响另一处
            continue

    # 名字短的排前面：「微信」应该排在「卸载微信」之前
    rows.sort(key=lambda r: (len(r["name"]), r["name"]))
    return rows[:limit]
