"""临时上传区（data/uploads）的回收。

这个目录里放的是**当前这次会话的工作副本**：一份 CSV 落成 `<sid>.csv`，
资料落成 `<sid>/<文件名>`。sid 只活在 `st.session_state` 里，对话 JSON 不记它
（chat 的 `table` 字段只有 vault 的 id 或表名）。所以进程一重启，里面的东西
**没有任何代码路径读得到**——它们不是旧数据，是永远走不到的副本。

也就是说这个目录不归"用户"管，归"会话"管。三个时刻各删一次：

    sweep()          进程刚起来：上一次进程留下的全是死的
    discard(key)     会话换了（切对话 / 新建 / 删除）：csv 连目录一起
    drop_table(key)  撤下临时表：只删 csv，同一会话的资料还得能读

为什么 sweep 看 mtime 而不是"整个清空"：本进程刚写进去的一定不能删——另一个
标签页可能正用着它，而界面测试也会把 app.py 整个跑一遍（那同样是一次 sweep）。
mtime 就是这条界线。

代价说清楚：两个实例同时挂同一个 data 目录时，后启动的那个仍会把先启动的
旧文件当遗留清掉。这跟 data/chats/index.json 被两个实例抢写是同一个问题，
属于"一个 data 目录只跑一个实例"的前提，本模块不解决。
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Iterable

from .config import UPLOAD_DIR

# 模块导入的时刻 ≈ 进程启动的时刻。比它更早的文件才算上一个进程的遗留。
_START = time.time()


def safe_key(key: str) -> str:
    """key 会变成文件名，必须过滤。

    sid 是我们自己生成的十六进制，但脚本和测试会传 "default"、"a:b" 这种。
    规则原本在 data_tools 里，挪过来是因为这个目录现在归本模块管。
    """
    return re.sub(r"[^0-9A-Za-z_-]", "_", str(key)) or "default"


def table_path(key: str) -> Path:
    """临时表那份 CSV 的位置。"""
    return UPLOAD_DIR / f"{safe_key(key)}.csv"


def doc_folder(key: str) -> Path:
    """临时资料的目录。"""
    return UPLOAD_DIR / safe_key(key)


def _entries() -> list[Path]:
    """目录顶层的条目。

    只认 `<key>.csv` 和 `<key>/`：其他文件是别人放进来的（脚本、手工拷贝），
    不属于本模块的管辖范围，一个都不动。
    """
    try:
        items = list(UPLOAD_DIR.iterdir())
    except OSError:
        return []
    return [p for p in items if p.is_dir() or p.suffix.lower() == ".csv"]


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _drop(path: Path) -> tuple[int, int]:
    """删一个条目，返回（条目数, 字节数）。删不掉（被占用）就当没删。"""
    if not path.exists():
        return 0, 0
    size = _size(path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        return 0, 0
    return 1, size


def discard(key: str) -> tuple[int, int]:
    """会话结束了，删掉它那一份：临时表的 csv 和临时资料的目录。

    Windows 上文件可能被沙箱子进程还占着，删不掉就留着，下次 sweep 再收。
    """
    items = freed = 0
    for path in (table_path(key), doc_folder(key)):
        n, b = _drop(path)
        items += n
        freed += b
    return items, freed


def drop_table(key: str) -> tuple[int, int]:
    """只撤下临时表，同一会话的资料不动。"""
    return _drop(table_path(key))


def sweep(keep: Iterable[str] = ()) -> tuple[int, int]:
    """删掉不属于任何活会话的遗留，返回（条目数, 字节数）。

    判定顺序是有意的：keep 里的（活会话）一定留；本进程写的一定留；
    剩下的才是上一个进程的遗留。
    """
    live = {safe_key(k) for k in keep}
    items = freed = 0
    for path in _entries():
        key = path.stem if path.is_file() else path.name
        if key in live:
            continue
        try:
            if path.stat().st_mtime >= _START:
                continue
        except OSError:
            continue
        n, b = _drop(path)
        items += n
        freed += b
    return items, freed


def usage() -> tuple[int, int]:
    """（条目数, 总字节）。给诊断和测试用，界面上没挂。"""
    items = total = 0
    for path in _entries():
        items += 1
        total += _size(path)
    return items, total


__all__ = [
    "discard",
    "doc_folder",
    "drop_table",
    "safe_key",
    "sweep",
    "table_path",
    "usage",
]
