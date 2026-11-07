"""出图工具。

数据可以来自两个地方：当前数据表里跑一条 SQL，或者调用方直接给两列数字。
第二种是必须的——资料文档里读出来的数字、run_python 算出来的结果，
本来就不是表，没有理由逼着它们先变成一份 CSV 才能画图。
"""

from __future__ import annotations

import time
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # 无界面后端，必须在 pyplot 之前设置

import matplotlib.pyplot as plt  # noqa: E402

from ..config import OUTPUT_DIR  # noqa: E402
from .data_tools import NoDataset, get_conn  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

from . import registry  # noqa: E402

MAX_POINTS = 40


@registry.tool
def make_chart(
    sql: str = "",
    kind: Literal["bar", "line", "pie"] = "bar",
    title: str = "",
    labels: list[str] | None = None,
    values: list[float] | None = None,
) -> str:
    """把两列数据画成图表并保存为 PNG 文件。

    数据来源二选一：
      - 有数据表时传 sql，结果必须恰好两列（第一列维度、第二列数值）
      - 没有数据表，或者数字来自文档、上一次计算时，直接传 labels 和 values

    适合"给我看看趋势/对比"这类需求；用户只要数字、不要图时用 run_sql。
    用 sql 时先确认它能跑通、结果不超过 20 行。

    Args:
        sql: 取数用的 SQL，必须返回两列（第一列维度、第二列数值）；用 labels/values 时不传
        kind: 图表类型：bar 柱状图、line 折线图、pie 饼图
        title: 图表标题，用中文写清楚这张图在说什么
        labels: 横轴标签，如 ["1月", "2月", "3月"]，与 values 一一对应
        values: 纵轴数值，如 [120, 135, 98]，与 labels 等长
    """
    labels = [str(v) for v in (labels or [])]
    values = list(values or [])

    if labels or values:
        return _from_pairs(labels, values, sql, kind, title)

    if not sql.strip():
        return (
            "错误：没有给数据。有数据表时传 sql；"
            "没有数据表时直接传 labels 和 values 两列数字。"
        )

    try:
        frame = get_conn().execute(sql).df()
    except NoDataset:
        # 这条提示很重要：原来只说"先上传 CSV"，模型就会去催用户传文件，
        # 而它其实手里已经有数字了，直接传两列就能画。
        return (
            "错误：当前没有数据表，sql 无从执行。"
            "改用 labels 和 values 把两列数字直接传进来。"
        )
    except Exception as exc:  # noqa: BLE001
        return f"SQL 执行失败：{exc}"

    if frame.shape[1] != 2:
        return f"错误：SQL 必须返回两列，实际返回了 {frame.shape[1]} 列。"
    if frame.empty:
        return "错误：查询结果为空，画不出图。"
    if len(frame) > MAX_POINTS:
        return f"错误：结果有 {len(frame)} 行，太多了。请用 LIMIT 或先聚合到 20 行以内。"

    x_col, y_col = str(frame.columns[0]), str(frame.columns[1])
    return _draw(
        x_values=[str(v)[:16] for v in frame[x_col]],
        y_values=frame[y_col].astype(float).tolist(),
        x_col=x_col,
        y_col=y_col,
        kind=kind,
        title=title,
    )


def _from_pairs(
    labels: list[str], values: list, sql: str, kind: str, title: str
) -> str:
    """直接给两列数字的那条路。不做任何数据表查询。"""
    if sql.strip():
        return "错误：sql 和 labels/values 只能选一种，不要同时传。"
    if not labels or not values:
        return "错误：labels 和 values 要成对给，现在只给了一个。"
    if len(labels) != len(values):
        return (
            f"错误：labels 有 {len(labels)} 个、values 有 {len(values)} 个，"
            "两边长度必须一致。"
        )
    if len(labels) > MAX_POINTS:
        return f"错误：数据点有 {len(labels)} 个，太多了。请先聚合到 20 个以内。"
    try:
        y_values = [float(v) for v in values]
    except (TypeError, ValueError):
        return "错误：values 里必须全是数字，不能有文字或空值。"

    return _draw(
        x_values=[v[:16] for v in labels],
        y_values=y_values,
        x_col="类别",
        y_col="数值",
        kind=kind,
        title=title,
    )


def _draw(
    *,
    x_values: list[str],
    y_values: list[float],
    x_col: str,
    y_col: str,
    kind: str,
    title: str,
) -> str:
    """两条路都收敛到这里：拿到两列已经规整好的数，只管画。"""
    width = 6.0 if kind == "pie" else 8.0
    figure, axes = plt.subplots(figsize=(width, 4.5))

    if kind == "bar":
        axes.bar(x_values, y_values, color="#3B7DD8")
        axes.tick_params(axis="x", rotation=30)
        axes.set_ylabel(y_col, fontsize=11)
    elif kind == "line":
        axes.plot(x_values, y_values, marker="o", color="#3B7DD8")
        axes.tick_params(axis="x", rotation=30)
        axes.grid(alpha=0.25, linestyle="--")
        axes.set_ylabel(y_col, fontsize=11)
    else:
        # 饼图没有坐标轴。这里如果还调 set_ylabel，会在左侧画出一个
        # 孤零零竖着的列名（曾经踩过，看图才发现）。
        axes.pie(y_values, labels=x_values, autopct="%1.1f%%", startangle=90)

    if title:
        axes.set_title(title, fontsize=13)
    figure.tight_layout()

    path = OUTPUT_DIR / f"chart_{int(time.time() * 1000)}.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)

    return (
        f"图表已生成：{path.as_posix()}\n"
        f"类型 {kind}，{len(y_values)} 个数据点，横轴 {x_col}，纵轴 {y_col}"
    )
