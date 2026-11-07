"""思维导图工具。

它和 make_chart 分开，不是为了"多一个工具"，而是因为**输入的形状不一样**：
make_chart 吃的是两列数（维度 + 数值），思维导图吃的是树（中心 → 分支 → 叶子）。
硬塞进一个函数，签名就会变成 sql / labels / values / paths 三种互斥来源，
模型每画一次图都要先判断"这次该填哪几个"，出错率比多记一个工具名高得多。

输入用「路径列表」而不是嵌套 JSON：

    [["销售分析", "华东", "上海"],
     ["销售分析", "华东", "杭州"]]

理由有三个：
  1. JSON Schema 里就是 array 套 array，`schema.py` 的 _json_type 递归一下就有了；
  2. 模型生成嵌套 dict 时少一个括号整段作废，路径列表错一条只废一条；
  3. 层级深度不受结构限制，两层三层是同一种写法。

布局自己算，不依赖 graphviz：本机既没有 graphviz 包也没有 dot 命令，
为一个图去要求用户装系统级软件不值当。代价是 50 行坐标计算，换来的是
"pip install 完就能画"，换台机器也一样。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # 无界面后端，必须在 pyplot 之前设置

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from ..config import OUTPUT_DIR  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

from . import registry  # noqa: E402

MAX_NODES = 60
MAX_DEPTH = 5
MAX_LEAVES = 24
LABEL_LIMIT = 12  # 每行字数，最多两行
LEAF_GAP_MIN = 0.16  # 相邻两个叶子框之间的最小缝隙

X_STEP = 2.15  # 每深一层横向走多远（英寸）

ROOT_FILL = "#2F5C9E"
ROOT_TEXT = "#FFFFFF"
BRANCH_FILL = "#D6E4F7"
BRANCH_TEXT = "#1E3A5F"
LEAF_FILL = "#F2F5FA"
LEAF_TEXT = "#333333"
LINE_COLOR = "#7A93B5"


@dataclass
class _Node:
    """树节点。children 用 dict 而不是 list：插入即去重，还保序。"""

    label: str
    depth: int = 0
    direction: int = 1  # +1 向右展开，-1 向左
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    children: dict[str, "_Node"] = field(default_factory=dict)


# ---------------------------------------------------------------- 输入处理


def _wrap(label: str) -> str:
    """折成最多两行，超出部分用省略号收掉。

    不用 textwrap：它按空格分词，中文整段会被当成一个词，一个字都不折。
    已经折过的直接返回——_measure 折一次，后面几处会再遍历到同一个节点，
    不幂等的话每过一遍就再切一刀，最后变成一行两个字。
    """
    if "\n" in label:
        return label
    if len(label) <= LABEL_LIMIT:
        return label
    head = label[: LABEL_LIMIT * 2]
    if len(label) > LABEL_LIMIT * 2:
        head = head[: LABEL_LIMIT * 2 - 1] + "…"
    return head[:LABEL_LIMIT] + "\n" + head[LABEL_LIMIT:]


def _build(paths: list[list[str]]) -> _Node:
    """按公共前缀建树。重复的父节点自动合并，同一条路径写两遍也不会重复出现。"""
    root = _Node(label=paths[0][0])
    for path in paths:
        node = root
        for label in path[1:]:
            node = node.children.setdefault(str(label), _Node(label=str(label)))
    return root


def _count(node: _Node) -> int:
    return 1 + sum(_count(child) for child in node.children.values())


def _leaf_count(node: _Node) -> int:
    if not node.children:
        return 1
    return sum(_leaf_count(child) for child in node.children.values())


def _measure(node: _Node) -> None:
    """先把所有节点的文字和尺寸算好。

    必须在排 y 之前做：折行会让节点从单行变两行，高度从 0.42 涨到 0.70，
    按固定间距摆就会和上下邻居叠在一起（第一版就是这么翻车的——
    「北京」和「月度」的框边框直接压在同一条线上）。
    """
    node.label = _wrap(node.label)
    node.width, node.height = _text_size(node.label)
    for child in node.children.values():
        _measure(child)


def _assign_y(node: _Node, cursor: list[float]) -> float:
    """纵向排布：叶子按自己的实际高度往下排，父节点落在首尾子节点中间。"""
    if not node.children:
        node.y = cursor[0] + node.height / 2
        cursor[0] += node.height + LEAF_GAP_MIN
        return node.y

    ys = [_assign_y(child, cursor) for child in node.children.values()]
    node.y = (ys[0] + ys[-1]) / 2
    return node.y


def _text_size(label: str) -> tuple[float, float]:
    """节点框尺寸。中文按字宽估，英文数字窄一些——估大一点总比溢出好看。

    高度必须小于 LEAF_GAP 的下限（0.42），否则相邻两个叶子会贴在一起。
    """
    rows = label.split("\n")
    widest = max(sum(0.145 if ord(ch) > 127 else 0.082 for ch in row) for row in rows)
    width = max(0.95, widest + 0.46)
    height = 0.26 * len(rows) + 0.14
    return width, height


# ---------------------------------------------------------------- 绘制


def _link(ax, parent: _Node, child: _Node) -> None:
    """三次贝塞尔。控制点落在父子水平中线上，出来是顺滑的 S 形。"""
    sign = child.direction
    x1 = parent.x + sign * parent.width / 2
    x2 = child.x - sign * child.width / 2
    mid = (x1 + x2) / 2
    path = Path(
        [(x1, parent.y), (mid, parent.y), (mid, child.y), (x2, child.y)],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=LINE_COLOR, linewidth=1.0))


def _box(ax, node: _Node, *, fill: str, text_color: str, fontsize: float) -> None:
    """画节点框 + 文字。填充色和文字色**必须分开传**——

    第一版只传了一个 color，中心节点于是变成深蓝底 + 深蓝字，
    看上去就是一个空框，查了半天才发现字还在，只是和底色一样。
    """
    ax.add_patch(
        FancyBboxPatch(
            (node.x - node.width / 2, node.y - node.height / 2),
            node.width,
            node.height,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0.6,
            edgecolor=text_color,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(
        node.x,
        node.y,
        node.label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.35,
        zorder=3,
    )


def _draw(root: _Node, title: str = "") -> plt.Figure:
    """先算坐标，再统一画。分成两步是因为画布尺寸要由坐标范围反推。"""
    _measure(root)

    branches = list(root.children.values())
    right = [c for i, c in enumerate(branches) if i % 2 == 0]
    left = [c for i, c in enumerate(branches) if i % 2 == 1]

    cursor_r, cursor_l = [0.0], [0.0]
    for child in right:
        child.direction = 1
        _assign_y(child, cursor_r)
    for child in left:
        child.direction = -1
        _assign_y(child, cursor_l)

    span_r, span_l = cursor_r[0], cursor_l[0]
    span = max(span_r, span_l, root.height)
    # 两侧各自居中：右边叶子多、左边叶子少时，左边不会全挤在顶部
    for child in right:
        _shift(child, (span - span_r) / 2)
    for child in left:
        _shift(child, (span - span_l) / 2)

    root.x, root.y = 0.0, span / 2
    _layout_x(root, 0)

    depth_r = _max_depth(right)
    depth_l = _max_depth(left)
    # 边距 = 最外层那个框的一半 + 一点余量，不写死
    pad_l = _outer_width(left) / 2 + 0.4 if depth_l else root.width / 2 + 0.4
    pad_r = _outer_width(right) / 2 + 0.4 if depth_r else root.width / 2 + 0.4
    x_min = -depth_l * X_STEP - pad_l
    x_max = depth_r * X_STEP + pad_r
    y_min = -1.1 if title else -0.6
    y_max = span + 0.7

    figure = plt.figure(figsize=(x_max - x_min, y_max - y_min), dpi=130)
    figure.patch.set_facecolor("white")
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_max, y_min)  # 翻转 y 轴：游标从上往下长，符合阅读顺序
    axes.axis("off")

    for child in root.children.values():
        _draw_subtree(axes, root, child)
    _box(axes, root, fill=ROOT_FILL, text_color=ROOT_TEXT, fontsize=13)
    if title:
        axes.text(
            (x_min + x_max) / 2,
            y_min + 0.42,
            title,
            ha="center",
            va="center",
            fontsize=13,
            color="#222222",
        )
    return figure


def _shift(node: _Node, delta: float) -> None:
    node.y += delta
    for child in node.children.values():
        _shift(child, delta)


def _layout_x(node: _Node, depth: int) -> None:
    """横向：每一层固定步长。

    一级分支的左右方向是 _draw 里分好组的，**不能**被父节点的方向覆盖——
    第一版这里无条件写 child.direction = node.direction，根的 direction 默认是 1，
    结果左边那组全被拽回右边，整张图变成单侧树。
    """
    node.depth = depth
    for child in node.children.values():
        if depth > 0:
            child.direction = node.direction
        child.x = node.x + child.direction * X_STEP
        _layout_x(child, depth + 1)


def _max_depth(nodes: list[_Node]) -> int:
    best = 0
    for node in nodes:
        best = max(best, 1 + _max_depth(list(node.children.values())))
    return best


def _outer_width(nodes: list[_Node]) -> float:
    """最外层节点的最大宽度。

    画布左右边距只需要留"最远的那个框"的半个身位。按固定 2.3 英寸留，
    树浅的时候会空出一大片白——图上看着像画歪了。
    """
    if not nodes:
        return 0.0
    deeper = [child for node in nodes for child in node.children.values()]
    if deeper:
        return _outer_width(deeper)
    return max(node.width for node in nodes)


def _draw_subtree(ax, parent: _Node, node: _Node) -> None:
    """先画线段再画框，框会盖住线头，看起来像线插进了框里。"""
    _link(ax, parent, node)
    if not node.children:
        _box(ax, node, fill=LEAF_FILL, text_color=LEAF_TEXT, fontsize=9.5)
        return
    _box(ax, node, fill=BRANCH_FILL, text_color=BRANCH_TEXT, fontsize=10.5)
    for child in node.children.values():
        _draw_subtree(ax, node, child)


# ---------------------------------------------------------------- 工具


@registry.tool
def make_mindmap(paths: list[list[str]], title: str = "") -> str:
    """把层级结构画成思维导图并保存为 PNG 文件。

    paths 是若干条「从中心到叶子」的完整路径，每层一个字符串。相同的前缀会自动合并，
    所以只需要把想表达的每一条路径列出来，不用自己拼树。例如：

        [["销售分析", "华东", "上海"],
         ["销售分析", "华东", "杭州"],
         ["销售分析", "华北", "北京"]]

    适合画报告目录、分类体系、问题拆解、方案对比这类有层级的东西。
    数值之间的趋势和对比用 make_chart，不要用这个。

    所有路径必须从同一个中心节点开始；最多 60 个节点、5 层、30 个叶子。

    Args:
        paths: 每条路径是一个字符串列表，从中心节点写到叶子，至少两层
        title: 图表标题，用中文写清楚这张图在说什么
    """
    if not paths:
        return "错误：paths 是空的。至少给一条路径，比如 [[\"中心\", \"分支\"]]。"

    for index, path in enumerate(paths, 1):
        if isinstance(path, str):
            return (
                f"错误：第 {index} 项是一条字符串，paths 的每一项都要是路径列表。"
                '正确写法：[["销售分析", "华东", "上海"], ["销售分析", "华北", "北京"]]'
            )
        if not isinstance(path, (list, tuple)) or len(path) < 2:
            return (
                f"错误：第 {index} 条路径只有 {len(path) if isinstance(path, (list, tuple)) else 0} 层，"
                "至少要两层（中心 + 一个分支）。"
            )

    cleaned = [
        [str(label).strip() for label in path if str(label).strip()] for path in paths
    ]
    cleaned = [path for path in cleaned if len(path) >= 2]
    if not cleaned:
        return "错误：所有路径都是空的，没有可画的节点。"

    roots = {path[0] for path in cleaned}
    if len(roots) > 1:
        return (
            f"错误：所有路径必须从同一个中心节点开始，现在有 {len(roots)} 个："
            f"{'、'.join(sorted(roots))}。"
        )

    longest = max(len(path) for path in cleaned)
    if longest > MAX_DEPTH:
        return f"错误：层级有 {longest} 层，最多画 {MAX_DEPTH} 层。请先合并中间层。"

    root = _build(cleaned)
    nodes = _count(root)
    if nodes > MAX_NODES:
        return f"错误：一共 {nodes} 个节点，太多了（上限 {MAX_NODES}）。请先归类合并。"
    leaves = _leaf_count(root)
    if leaves > MAX_LEAVES:
        return f"错误：有 {leaves} 个叶子节点，太多了（上限 {MAX_LEAVES}）。请先合并同类项。"

    figure = _draw(root, title)

    path = OUTPUT_DIR / f"mindmap_{int(time.time() * 1000)}.png"
    figure.savefig(path, dpi=130, facecolor="white")
    plt.close(figure)

    return (
        f"图表已生成：{path.as_posix()}\n"
        f"思维导图：{nodes} 个节点、{longest} 层，中心「{cleaned[0][0]}」"
    )
