"""数据处理与查询工具库 (data_tools)
这个文件的主要职责有两个：
1.  表格数据管家：负责把用户上传的 CSV 表格加载到超快速的内存数据库（DuckDB）中，
   并提供工具让 AI 能够“看懂表结构”以及“执行 SQL 语句查询数据”。
2.  资料文件管家：负责管理用户上传的各种资料（如 PDF、Word、PPT、TXT 文档），
   记录它们的信息并提供查找功能，方便后续让 AI 查阅。
"""

from __future__ import annotations

import contextvars
import re
from pathlib import Path
from typing import Any

import duckdb

from .. import readers, uploads
from ..config import DATA_FILE, MAX_OBS_CHARS, MAX_UPLOAD_MB, TABLE_NAME
from ..readers import Doc, DocumentError


class NoDataset(RuntimeError):
    """自定义异常：当用户还没上传或绑定任何数据表，却尝试去查数据时抛出。"""


# ==============================================================================
# 全局存储与会话隔离机制（防止多个用户的数据互相串门）
# ==============================================================================

# 存放每个会话的 DuckDB 数据库连接对象。格式为：{"会话ID": 连接对象}
_conns: dict[str, duckdb.DuckDBPyConnection] = {}

# 存放每个会话的数据集小档案（如文件名、行数、列数等）。格式为：{"会话ID": {"label": "文件名", ...}}
_meta: dict[str, dict[str, Any]] = {}

# 默认会话标识符
DEFAULT_KEY = "default"

# 上下文变量（ContextVar）：相当于给当前执行的线程贴一张“会话便签”。
# 这样在多线程/多会话并发执行时，每个线程都能准确知道自己正在操作哪位用户的数据，绝不会发生数据串门。
_ACTIVE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "datainsight.active_key", default=DEFAULT_KEY
)


def current_key() -> str:
    """获取当前正在操作的会话标识符（Key）。工具被调用时随时取出使用。"""
    return _ACTIVE.get()


def use_dataset(key: str) -> None:
    """切换当前线程要操作的会话标识符（Key）。
    """
    _ACTIVE.set(key)


# ==============================================================================
# 安全防火墙：拦截一切危险 SQL 写操作
# ==============================================================================
  
# 正则表达式拦截器：任何包含写入、修改、删除、建表、删库、外部加载等危险关键词的 SQL，通通拒绝执行！
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|copy|export|import|install|load)\b",
    re.I,  # 忽略大小写，无论写 DROP 还是 drop 都能抓住
)


def _drop_conn(key: str) -> None:
    """关闭并清理指定会话的旧数据库连接，释放系统内存。"""
    old = _conns.pop(key, None)
    if old is not None:
        try:
            old.close()  # 关掉连接
        except Exception:  # noqa: BLE001
            pass  # 关闭失败也不影响后续绑定新连接，静默跳过即可


def _connect_csv(path: Path, key: str) -> duckdb.DuckDBPyConnection:
    """把磁盘上的 CSV 文件连接进 DuckDB 内存数据库中，并生成一个虚拟视图供查询。
    
    参数:
        path: CSV 文件的本地绝对路径
        key: 会话标识符
    返回:
        DuckDB 数据库连接对象
    """
    con = duckdb.connect()  # 创建一个全新的内存数据库
    # 使用 DuckDB 强大的 read_csv_auto 自动分析表头和数据类型，创建名为 TABLE_NAME (即 dataset) 的视图
    con.execute(
        f"CREATE VIEW {TABLE_NAME} AS SELECT * FROM read_csv_auto('{Path(path).as_posix()}')"
    )
    # 登记初始元信息（此时先不数行数，避免大文件卡顿，后面用到时再查）
    _meta[key] = {"label": Path(path).name, "kind": "csv", "rows": None, "columns": None}
    return con


def bind_dataset(
    source: Any, *, key: str | None = None, label: str | None = None
) -> str:
    """把一份数据绑定为系统全局通用的 `dataset` 表，后续所有查询工具都基于它运行。

    【支持的两种输入形式】：
      1. 文件路径（字符串或 Path 对象）：指向磁盘上的 .csv 文件。
      2. Pandas DataFrame 对象：通常是用户在网页前端上传文件后读取进来的内存数据表。

    【核心设计考量 1：为什么表名固定叫 dataset，而不随文件名变化？】
      如果表名随着文件名改变（例如变成 `sales_2024` 或 `用户清单`），那么系统给 AI 的提示词、
      函数的默认参数、测试用例里的 SQL 全部都得跟着动态变化，极其容易出现 AI 记错表名导致查询报错。
      固定叫 `dataset`，AI 每次写 SQL 就永远只查 `FROM dataset`，极其稳定！真实文件名只作为备注展示。

    【核心设计考量 2：为什么上传的 DataFrame 必须同步在硬盘存一份 CSV？】
      因为我们后续执行复杂 Python 代码分析（run_python）是在完全独立的子进程沙箱里运行的，
      子进程是通过 `pd.read_csv(...)` 读取物理文件拿到数据的。如果只存进 DuckDB 内存，
      会导致“SQL 工具能查到新数据，但 Python 代码跑起来还是读不到文件”的灾难性不一致。
      所以必须在 `data/uploads/` 落地一份物理文件。
    """
    key = key or current_key()

    if isinstance(source, (str, Path)):
        # 情况 1：传入的是文件路径
        path = Path(source)
        _drop_conn(key)  # 关掉旧连接
        _conns[key] = _connect_csv(path, key)
    else:
        # 情况 2：传入的是 Pandas DataFrame 表格对象
        path = uploads.table_path(key)  # 获取该会话专属的 CSV 存储路径
        source.to_csv(path, index=False, encoding="utf-8")  # 落盘保存为 CSV 文件
        con = duckdb.connect()
        con.register(TABLE_NAME, source)  # 直接把 DataFrame 注册成 DuckDB 中的表
        _drop_conn(key)  # 关掉旧连接
        _conns[key] = con
        _meta[key] = {
            "label": label or path.name,
            "kind": "upload",
            "rows": int(len(source)),
            "columns": int(len(source.columns)),
        }

    # 完善小档案信息
    if label:
        _meta.setdefault(key, {})["label"] = label
    _meta.setdefault(key, {})["path"] = str(path)
    
    # 刚绑好的数据集自动设为当前活跃数据集，调用方无需多写一行 use_dataset(key)
    use_dataset(key)
    return key


def bind_sample(key: str = "default") -> str:
    """绑定系统自带的测试销售数据（data/sales.csv）。
    
    【注意】：
    这个函数通常只在编写自动化测试脚本时调用。
    网页正式运行时不默认调用它——因为真实业务中必须让用户主动上传，
    否则一上来就显示一堆销售数据，用户会一头雾水：“这些数字打哪儿来的？”。
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"找不到示例数据 {DATA_FILE}，请先运行 python data/make_data.py 生成"
        )
    return bind_dataset(DATA_FILE, key=key, label=DATA_FILE.name)


def unbind_dataset(key: str | None = None) -> None:
    """解绑并移除指定会话的数据集。用户在网页点击“移除当前数据”时调用。

    【注意细节】：
    必须真正调用 `_drop_conn` 关闭连接并从 `_conns` 中 pop 掉。
    如果只是把当前指针切走，连接还在内存里，下次一旦切回这个 key，旧表又“复活”了，
    就会出现“明明删了数据，却依然能查出来”的灵异 Bug。
    """
    key = key or current_key()
    _drop_conn(key)
    _meta.pop(key, None)


def active_dataset() -> str:
    """获取当前正处于激活状态的会话标识符（Key）。"""
    return current_key()


def dataset_meta() -> dict[str, Any]:
    """获取当前数据集的基础信息（用于前端侧边栏展示）。

    包含：文件名（label）、数据类型（kind）、行数（rows）、列数（columns）。
    如果当前没有绑定数据，严禁胡编一个默认文件名，必须如实返回 None。
    """
    key = current_key()
    info = dict(_meta.get(key, {}))
    info["key"] = key
    if not info.get("label"):
        return {"key": key, "label": None, "kind": None, "rows": None, "columns": None}
    
    # 如果之前还没统计过行列数，这里临时查一次补齐
    if info.get("rows") is None:
        try:
            con = get_conn()
            info["rows"] = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
            info["columns"] = len(con.execute(f"DESCRIBE {TABLE_NAME}").fetchall())
        except Exception:  # noqa: BLE001
            pass
    return info


def active_data_file() -> Path | None:
    """获取当前数据集对应的 CSV 磁盘物理路径；如果没有数据则返回 None。

    【为什么重要？】
    独立的 Python 执行沙箱必须读取这个物理路径。
    如果这里返回的文件和 SQL 查的数据不是同一份，就会发生经典的“SQL 查出 3 行，Python 却算出 300 行”
    的致命数据不同步问题。
    """
    info = _meta.get(current_key()) or {}
    path = info.get("path")
    return Path(path) if path else None


def column_names() -> list[str]:
    """获取当前数据表所有的列名列表。
    
    主要给前端界面使用（例如在输入框下方自动生成几个带字段名的“推荐提问按钮”）。
    如果没绑定数据或出错，静默返回空列表。
    """
    try:
        return [r[0] for r in get_conn().execute(f"DESCRIBE {TABLE_NAME}").fetchall()]
    except Exception:  # noqa: BLE001
        return []


def get_conn() -> duckdb.DuckDBPyConnection:
    """获取当前会话的 DuckDB 数据库连接对象。

    如果当前会话还没有绑定数据表，绝不隐式自动绑定默认数据，而是直接抛出 NoDataset 异常，
    让前端或调用者明确知道该提示用户“请先上传 CSV 文件”。
    """
    key = current_key()
    if key not in _conns:
        raise NoDataset("还没有选择数据集（需要先上传一份 CSV）")
    return _conns[key]


def _format(columns: list[str], rows: list[tuple[Any, ...]], total: int) -> str:
    """辅助排版函数：把数据库查询出来的原始数据列表，组装成人类和 AI 都能轻松看懂的 Markdown 表格文本。
    
    参数:
        columns: 字段名列表，如 ['姓名', '年龄']
        rows: 每一行的数据元组列表，如 [('张三', 18), ('李四', 20)]
        total: 查到的总行数（用于提示是否做过截断）
    """
    # 拼接表头：列名之间用竖线分隔
    lines = [" | ".join(columns)]
    # 表头下方的虚线分割线
    lines.append("-" * max(8, len(lines[0])))
    # 逐行把数据转换成字符串，空值 None 显示为 "NULL"
    lines += [
        " | ".join("NULL" if v is None else str(v) for v in row) for row in rows
    ]
    # 如果实际结果比展示出来的行数多，底部贴心加一句说明
    if total > len(rows):
        lines.append(f"（共 {total} 行，这里只显示前 {len(rows)} 行）")
    return "\n".join(lines)


def _exec(sql: str, max_rows: int) -> tuple[list[str], list[tuple], int]:
    """底层 SQL 执行器：包含安全校验与行数截取。
    
    返回:
        (列名列表, 查询结果行列表, 实际获取到的行数)
    """
    # 步骤 1：安全检查——如果包含任何修改/删除关键字，立马报错拦截
    if _FORBIDDEN.search(sql):
        raise PermissionError(
            f"检测到写操作 {_FORBIDDEN.search(sql).group(1).upper()}，只允许只读查询"
        )
    # 步骤 2：在 DuckDB 中执行 SQL 查询
    result = get_conn().execute(sql)
    # 提取查询结果的字段名
    columns = [d[0] for d in (result.description or [])]
    # 多抓 1 行（max_rows + 1），用来精准判断后面还有没有更多数据，从而决定是否提示截断
    rows = result.fetchmany(max_rows + 1)
    return columns, rows[:max_rows], len(rows)


# ==============================================================================
# 资料文件管理（PDF / Word / PPT / TXT 等资料的登记与检索）
# ==============================================================================
#
# 【架构分工说明】：
# 本模块只负责资料在当前会话中的“登记立项”与“按名查找”。
# 文档内具体文字和图片的“内容解析与抽取”，交给专门的 `app/readers.py` 处理。
# 供 AI 调用的读文档工具（如 read_document）则位于 `doc_tools.py`。
# 这样职责分明：底层数据归本模块管，工具表现归工具模块管。

# 文档花名册字典，结构为：{"会话ID": {"文件名": Doc对象}}
_docs: dict[str, dict[str, Doc]] = {}


def _safe_name(name: str) -> str:
    """文件名安全净化小助手：去除非法字符，防止路径穿越攻击。

    比如用户或大模型偶尔会传入带有 `../../.env` 或者 Windows 不允许的特殊字符（如 : * ? 等）。
    这里只保留纯文件名，把危险字符替换为下划线，并将长度限制在 120 字符内。
    """
    base = Path(str(name).replace("\\", "/")).name.strip() or "未命名"
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base)
    return base[:120]


def add_document(name: str, raw: bytes, *, key: str | None = None) -> Doc:
    """把用户上传的资料保存到磁盘，提取其中的全文内容，并登记到当前会话中。

    【核心细节 1：文件大小限制】
      超过系统设定的最大 MB 限制直接拦截，防止把服务器内存撑爆。

    【核心细节 2：同名文件绝不覆盖，自动加 (2)(3) 后缀】
      如果用户传了两版同名的《项目汇报.docx》，直接覆盖会导致大模型读到新版，
      但界面上却只显示一个名字，排查问题会非常困惑。所以自动重命名并存。
    """
    key = key or current_key()
    # 检查文件大小
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentError(
            f"文件超过 {MAX_UPLOAD_MB} MB。先压缩，或者拆成几份再传。"
        )

    # 确定存放文档的专属文件夹
    folder = uploads.doc_folder(key)
    folder.mkdir(parents=True, exist_ok=True)

    # 生成安全的目标文件路径
    target = folder / _safe_name(name)
    stem, suffix = target.stem, target.suffix
    index = 2
    # 如果已存在同名文件，循环添加序号：报告(2).docx, 报告(3).docx ...
    while target.exists():
        target = folder / f"{stem}({index}){suffix}"
        index += 1

    # 将文件二进制字节写入硬盘
    target.write_bytes(raw)
    try:
        # 调用 readers 抽取文档文本内容
        doc = readers.extract(target, name=target.name)
    except Exception:
        target.unlink(missing_ok=True)  # 一旦解析抽取失败，立即删掉刚落盘的物理文件，不留磁盘垃圾
        raise

    # 登记到当前会话的文档花名册中
    _docs.setdefault(key, {})[doc.name] = doc
    return doc


def register_document(doc: Doc, *, key: str | None = None) -> Doc:
    """登记一份【已经在硬盘上存在、归属外部管辖】的文档。

    比如从长期知识库把已有文件“挂载”到当前会话时使用。
    因为该文件是共享的，所以其 doc.owned 为 False，会话结束清理时不会把物理文件删掉。
    """
    key = key or current_key()
    _docs.setdefault(key, {})[doc.name] = doc
    return doc


def documents(key: str | None = None) -> list[Doc]:
    """获取当前会话已经登记的所有文档列表（按登记先后顺序排列）。"""
    return list(_docs.get(key or current_key(), {}).values())


def find_document(name: str, key: str | None = None) -> Doc | None:
    """根据文件名精准查找一篇文档。

    【为什么故意不做模糊搜索？】
    因为大模型面对模糊搜索很容易张冠李戴！一旦系统“自作聪明”猜错文件，
    大模型拿到了完全不相干的内容，依然会一本正经地胡说八道。
    找不到就如实返回 None，让上层把可选的文件清单明确展示给模型看。
    """
    return _docs.get(key or current_key(), {}).get(str(name).strip())


def remove_document(name: str, key: str | None = None) -> bool:
    """从当前会话中移除指定的文档。如果是该会话独占拥有的物理文件，同时从磁盘上删除。"""
    key = key or current_key()
    doc = _docs.get(key, {}).pop(str(name).strip(), None)
    if doc is None:
        return False
    # 如果不是独占拥有的文件（比如是外部共享挂载的），内存移除后就收工，不去删物理文件
    if not doc.owned:
        return True
    try:
        # 删除物理文件
        Path(doc.path).unlink(missing_ok=True)
        # 尝试顺手把空的父文件夹也删掉；若文件夹里还有别的文件，rmdir 会静默失败，这正是期望的
        Path(doc.path).parent.rmdir()
    except OSError:  # noqa: BLE001
        pass
    return True


def clear_documents(key: str | None = None) -> None:
    """清空当前会话登记的所有资料文件，并做物理清理。"""
    key = key or current_key()
    for doc in documents(key):
        remove_document(doc.name, key)
    _docs.pop(key, None)


def has_documents(key: str | None = None) -> bool:
    """检查当前会话中是否有登记过资料文件。"""
    return bool(_docs.get(key or current_key()))


def has_dataset(key: str | None = None) -> bool:
    """检查当前会话中是否绑定了数据表格。
    
    【注】：
    和 has_documents 分开判断——一个是查“表格数据”，一个是查“文本资料”，
    二者能力和调用工具完全不同，不能混为一谈。
    """
    return (key or current_key()) in _conns


def document_digest(key: str | None = None) -> list[dict[str, Any]]:
    """生成文档的简要清单（仅包含文件名、类型、字数、分块数等元信息，不含正文）。
    
    主要给系统的 System Prompt（系统提示词）使用，让 AI 知道用户手头有哪些资料可供查阅。
    """
    return [
        {"name": d.name, "kind": d.kind, "chars": d.chars, "parts": d.parts}
        for d in documents(key)
    ]


# ==============================================================================
# 注册供 AI 智能体调用的工具函数 (Agent Tools)
# ==============================================================================

# 注意：放在文件底部导入 registry，是为了防止 Python 模块间相互引用时产生循环导入死锁
from . import registry  # noqa: E402


@registry.tool
def inspect_table() -> str:
    """查看当前数据表的字段名、数据类型、总行数以及前 3 行样例数据。

    【给 AI 的使用指导】：
    在写 SQL 查询之前，强烈建议先调用本工具确认有哪些列名以及列名叫什么。
    “盲猜列名”是模型编写 SQL 报错的最常见原因。
    因为系统中当前只维护一张主表，所以不需要传任何表名参数。
    """
    try:
        con = get_conn()
    except NoDataset as exc:
        return f"错误：{exc}"

    # 1. 查询表结构（字段名与数据类型）
    _, rows, _ = _exec(f"DESCRIBE {TABLE_NAME}", 50)
    schema_lines = "\n".join(f"  {r[0]}  {r[1]}" for r in rows)
    
    # 2. 抓取前 3 行作为样例数据
    sample = con.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 3").fetchall()
    sample_cols = [d[0] for d in con.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 0").description]
    
    # 3. 统计全表总行数
    total = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    # 4. 拼装成友好的纯文本格式返回
    text = (
        f"表 {TABLE_NAME}，共 {total} 行\n"
        f"字段：\n{schema_lines}\n"
        f"样例：\n{_format(sample_cols, sample, len(sample))}"
    )
    # 截断超长字符，防止大模型上下文超限
    return text[:MAX_OBS_CHARS]


@registry.tool
def run_sql(sql: str, max_rows: int = 30) -> str:
    """执行一条只读 SQL 查询语句（DuckDB 方言）并返回排版好的表格文本。

    【给 AI 的使用指导】：
    1. 只允许执行 SELECT / WITH / DESCRIBE / SUMMARIZE 等查询语句，任何写操作（如 UPDATE/DELETE）都会被直接拦截。
    2. 返回行数有限：如果想要统计总量、均值、排序前几名，请直接在 SQL 语句中使用 SUM/AVG/ORDER BY 等聚合函数计算好，
       不要把全表成千上万行都查出来数，否则会被系统强行截断。

    参数说明:
        sql: 需要执行的 SQL 语句，语法遵循 DuckDB 方言（基本兼容大部分标准 PostgreSQL）
        max_rows: 最多返回的结果行数，默认为 30 行，最大允许 200 行
    """
    # 限制行数范围在 1 到 200 之间，防止过大把大模型刷屏
    max_rows = max(1, min(int(max_rows), 200))
    try:
        columns, rows, total = _exec(sql, max_rows)
    except NoDataset as exc:
        return f"错误：{exc}"
    except PermissionError as exc:
        return f"错误：{exc}。请改写成 SELECT 查询。"
    except Exception as exc:  # noqa: BLE001
        return f"SQL 执行失败：{exc}"

    if not rows:
        return "查询成功，但结果为空（0 行）。可能是筛选条件写错了。"
    return _format(columns, rows, total)
