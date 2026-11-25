"""工具包。

导入顺序很重要：先建好 registry，再导入各个工具模块（它们靠导入副作用完成注册）。
这就是"插件自动发现"最朴素的写法——加文件后在下面多一行 import；
界面进程里新建的文件则靠 `load_missing()` 在每次 rerun 时补上，
不用重启整个进程。

六个工具模块的分工：
  data_tools    —— 表：inspect_table / run_sql；以及数据集与资料的登记表
  code_tools    —— 计算：run_python（转发给 sandbox）
  chart_tools   —— 出图：make_chart（两列数）
  mindmap_tools —— 出图：make_mindmap（树形结构）
  pptx_tools    —— 幻灯片：make_pptx 从零排，edit_pptx 在已有文件上增量改
  doc_tools     —— 资料：list_documents / read_document / search_document
  file_tools    —— 本机文件：list_files / read_file / write_text_file
  time_tools    —— 时钟：get_current_time（当前时间与常用相对时间区间）

chart_tools 和 mindmap_tools 分开，是因为输入形状不同（两列数 vs 树），
不是按"图"这个字分的。make_pptx 和 edit_pptx 留在同一个模块里，
是因为它们共用一整套版式绘制和估字逻辑——分开放，改一处版式要动两个文件。
"""

import importlib
import pkgutil
import sys

from .registry import Tool, ToolRegistry

# 全局唯一的注册表
registry = ToolRegistry()
tool = registry.tool


def load_missing() -> list[str]:
    """把目录里还没导入过的工具模块补上，返回这次新加载的模块名。

    为什么需要这个：注册靠 import 副作用，而包只在**第一次**导入时执行一次。
    界面进程跑起来之后新建一个工具文件，`import app.tools` 直接命中 sys.modules
    里的缓存，那行 import 根本不会重跑——于是模型那边看到的工具列表里没有它，
    表现就是「明明写好了却调用不到」，只能整个进程重启。

    每次 rerun 调一次这个函数，扫一眼目录就知道有没有新文件。
    只解决「新增模块」，不解决「改已有模块的代码」：后者要重新执行模块，
    而重复执行会撞上注册表的重名检查。
    """
    added: list[str] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "registry":
            continue
        module = f"{__name__}.{info.name}"
        if module in sys.modules:
            continue
        try:
            importlib.import_module(module)
        except ImportError:
            # 可选依赖没装（比如没装 python-pptx 就加载不了 pptx_tools）。
            # 记下来跳过就行，不能让每次 rerun 都抛一次异常。
            continue
        added.append(info.name)
    return added

from . import (  # noqa: E402,F401  注册副作用
    data_tools,
    code_tools,
    chart_tools,
    mindmap_tools,
    doc_tools,
    file_tools,
    time_tools,
)

# pptx_tools 依赖 python-pptx，而它是**可选**依赖（requirements 里写了，但只装 pandas
# 也该能跑）。顶层硬 import 会让"少装一个库"变成"程序起不来"，那就和 readers.py 里
# "缺了就提示装一下"的承诺对不上了。缺库时这两个工具不出现在列表里，模型也就不会调。
try:  # pragma: no cover - 装了 python-pptx 就一定走这里
    from . import pptx_tools  # noqa: F401
except ImportError:  # pragma: no cover
    pptx_tools = None  # type: ignore[assignment]

__all__ = [
    "registry",
    "tool",
    "load_missing",
    "Tool",
    "ToolRegistry",
    "data_tools",
    "code_tools",
    "chart_tools",
    "mindmap_tools",
    "doc_tools",
    "file_tools",
    "time_tools",
]
