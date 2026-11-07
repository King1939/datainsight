"""计算类工具：让模型临时写 Python。

为什么需要它：SQL 擅长聚合，不擅长"标准化后再算相关性""正则清洗文本"这类活。
给模型一个 Python 解释器，它的能力上限立刻就上去了。

注意执行策略跟模式走：只读模式下 open / to_csv 这类落盘写法在 AST 那关就被拒，
完全工作区模式下才放开。工具描述里两种都写了——描述是静态的，模式是动态的，
所以**判断依据是系统提示里那句"当前模式"**，不是这里。
"""

from __future__ import annotations

from .. import files, sandbox
from . import registry
from .data_tools import active_data_file


@registry.tool
def run_python(code: str) -> str:
    """执行一段 Python 代码做复杂计算、数据清洗、文本处理，或（在完全工作区模式下）改本机文件。

    环境已预置三个变量，直接用、不用创建：
        df  —— 整张表的 pandas DataFrame（没上传数据时为 None）
        pd  —— pandas
        con —— duckdb 连接，可 con.execute("SELECT ...").df()

    适合 SQL 不擅长的活：自定义指标、标准化、正则清洗、分组后二次加工。
    只要简单聚合结果时用 run_sql 更快；要出图用 make_chart；
    单纯读写一个文本文件时用 read_file / write_text_file 更直接。

    限制（违反会直接返回错误）：
    - 只读模式：不能读写文件（open / to_csv / savefig 都会被拒），
      只能 import 计算类库（pandas/numpy/math/re/datetime 等）
    - 完全工作区：可以 open() 和 import os/pathlib/shutil 等，
      能改本机任意文件；eval / exec / subprocess 仍然被禁
    - 两种模式都不能 import matplotlib / seaborn / plotly 等绘图库：
      子进程里画出来的图回不到界面。要图请用 make_chart（没有数据表时直接把
      两列数传给它的 labels / values）
    - 超过 8 秒会被强制终止，所以不要写大循环，先在 SQL 里把数据缩小

    代码可以多行；最后一行如果是表达式，它的值会作为结果返回。

    Args:
        code: 要执行的 Python 代码
    """
    allow_write = files.allow_write()
    data_file = active_data_file()
    if data_file is None and not allow_write:
        # 不能让它继续往下跑：没有数据就没有可执行的东西，
        # 让它"往下跑"只会变成一条看不见的假数据来源。
        return "错误：还没有选择数据集，请先上传一份 CSV。"
    result = sandbox.run(code, data_file=data_file, allow_write=allow_write)
    return result.as_observation()


__all__ = ["run_python"]
