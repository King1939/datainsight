"""
布局、会话状态、把事件丢给 ui_render 渲染。

Streamlit 的执行模型和普通脚本不同，必须记住两点：
  1. 每次交互都从第一行开始重跑整个脚本。所以 Agent 和 Session 绝不能放
     模块级变量，必须放 st.session_state，否则一刷新就没了。
  2. 多个浏览器标签页共用同一个进程。data_tools 里的连接和资料都按 key 分开存，
     key 就是这里的 sid，否则两个人上传的文件会互相覆盖。
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st
from streamlit.components.v1 import html as st_html

from . import chats, config, files, readers, uploads, vault
from .agent import Agent, make_system_prompt
from .config import MAX_UPLOAD_MB, VAULT_QUOTA_MB
from .llm import chat as llm_chat
from .llm import probe as llm_probe
from .readers import UPLOAD_SUFFIXES, DocumentError
from .session import Session
from .tools import data_tools as dt
from .tools import load_missing
from .ui_render import (
    TraceLog,
    dataset_badge,
    read_any_csv,
    size_label,
    suggest_questions,
)

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


# ---------------------------------------------------------------- 浏览器标签标题
#
# set_page_config 里的 page_title 是死的；但 Streamlit 每次交互都重跑整个脚本，
# 所以每次重跑都补一小段 JS 把 document.title 改掉，标签页就能跟着会话状态走。
# 注意组件跑在 iframe 里，必须写 window.parent.document.title 才碰得到外层页面。


def _set_browser_title(text: str) -> None:
    safe = json.dumps(text, ensure_ascii=False)
    st_html(f"<script>window.parent.document.title = {safe};</script>", height=0)


def _browser_title() -> str:
    """空闲状态的标题：优先用问答生成的短标题，否则用当前表名兜底。"""
    ss = st.session_state
    name = ss.get("conv_title")
    if not name:
        name = dt.dataset_meta().get("label")
    return f"{name} · DataInsight" if name else "DataInsight"


def _page_title() -> str:
    """页面大标题：跟标签页走同一份状态

    空闲时优先用问答生成的短标题，否则用当前表名兜底，再没有就退回品牌名。
    标签页那套实时逻辑不动，这里只是让页面里的标题也跟着同步。
    """
    ss = st.session_state
    name = ss.get("conv_title")
    if not name:
        name = dt.dataset_meta().get("label")
    return name if name else "DataInsight"


def _fallback_title(question: str) -> str:
    """模型没给出标题时的兜底：截断问题原文。"""
    return question[:12] + ("…" if len(question) > 12 else "")


def _make_conv_title(question: str) -> str:
    """把问题概括成短标题。失败了就退回截断原文，别让它拖垮一轮问答。"""
    try:
        response = llm_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "把用户的问题概括成不超过12个字的对话标题。"
                        "直接输出标题本身，不要引号、句号或任何解释。"
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            timeout=15,
            retries=0,
        )
        text = (response.message.get("content") or "").strip()
        text = text.splitlines()[0].strip(" 　\"'“”‘’。．.！!？?？")
        if text:
            return text[:16]
    except Exception:  # noqa: BLE001 标题只是装饰，任何失败都不该冒到界面上
        pass
    return _fallback_title(question)


def _start_title_job(question: str) -> dict:
    """把起标题丢进后台线程，主线程接着跑回答。

    以前是回答完再同步调一次模型：整轮对话里标签页都停在"分析中…"，
    答案全部吐完才出现话题名，等于没有实时可言。改成后台跑以后，标题一回来
    就写进占位符（见 _run_turn.flush_title），回答还在流的时候标签页已经变了。

    线程里只发 HTTP 请求、不碰 st：Streamlit 的 API 只能主线程用。
    """
    job: dict = {"title": "", "done": False}

    def work() -> None:
        try:
            job["title"] = _make_conv_title(question)
        finally:
            job["done"] = True

    threading.Thread(target=work, name="conv-title", daemon=True).start()
    return job


# ---------------------------------------------------------------- 会话状态


def _init_state() -> None:
    ss = st.session_state
    if "sid" not in ss:
        ss.sid = uuid.uuid4().hex[:8]
    if "turns" not in ss:
        ss.turns = []
    if "temp_sigs" not in ss:
        # {上传签名: 文件名}。Streamlit 每次 rerun 都把上传控件里的文件再给一遍，
        # 不去重的话，每点一次按钮就多登记一份。
        ss.temp_sigs = {}
    if "vault_sigs" not in ss:
        ss.vault_sigs = {}
    if "temp_table" not in ss:
        # 临时上传的 CSV：(文件名, DataFrame)。只在内存里，进程一停就没了。
        ss.temp_table = None
    if "table_id" not in ss:
        # None = 这次还没选过表，可以自动选储藏区里最新的一张；
        # "" = 用户明确撤下了，别再自作主张绑回去。
        ss.table_id = None
    if "bound" not in ss:
        ss.bound = None
    if "doc_sig" not in ss:
        ss.doc_sig = None
    if "conv_title" not in ss:
        # 第一轮提问时后台生成的短标题；None = 还没生成，浏览器标签先显示表名
        ss.conv_title = None
    if "vault_docs" not in ss:
        # 上一轮从储藏区挂进会话的资料名，撤的时候只撤这些，不碰临时上传的
        ss.vault_docs = []
    if "session" not in ss:
        # agent 等到有内容时再建，避免用一个"没有表"的 prompt 开场
        ss.session = Session()
    if "chat_id" not in ss:
        # 当前对话的 id，和 URL 里的 ?c= 保持一致。刷新页面靠它找回历史。
        ss.chat_id = None
    if "saved_sig" not in ss:
        # 上次落盘时的状态指纹。没变就不重写文件，否则每次点按钮都在写盘。
        ss.saved_sig = None
    if "missing_table" not in ss:
        # 历史对话绑的那张表已经不在了（比如临时上传的 CSV），只能提示用户重传
        ss.missing_table = None
    if "model_panel" not in ss:
        # 右侧「模型设置」是否展开。默认收起：它不是每天都要碰的东西。
        ss.model_panel = False
    if "model_base_url" not in ss:
        # 三个输入框的初值 = 当前生效的那一套（可能是 .env，也可能是上次存的）。
        # 只在这里初始化一次，之后控件自己管值——每次重跑都覆盖回去，用户就改不动了。
        active = config.current()
        ss.model_base_url = active.base_url
        ss.model_model = active.model
        ss.model_api_key = active.api_key
    if "model_flash" not in ss:
        ss.model_flash = None

    # 每次 rerun 先把「当前上下文指向本会话的数据」这件事定下来。_apply_dataset 里
    # 还会再设一次，但那时已经晚了一步——_restore_chat、_prompt_text、侧边栏
    # 都在它之前读过数据集。ContextVar 是按上下文隔离的，这个 set 只影响本标签页
    # 这一次执行，不会碰到正在跑长循环的另一个标签页。
    dt.use_dataset(ss.sid)


MODE_READ_LABEL = files.MODE_LABELS[files.MODE_READ]
MODE_WRITE_LABEL = files.MODE_LABELS[files.MODE_WRITE]


def _sync_mode() -> bool:
    """把开关的值同步到文件层，返回"这次是否变了"。

    为什么在 main 开头就同步、而不是等控件渲染完：Streamlit 的控件值在被点击的
    那一次 rerun 里就已经写进 session_state 了，所以下一趟脚本开头读到的就是新值。
    如果等渲染完再设，这一趟的 prompt 和工具权限还是旧的，会滞后一轮——
    表现就是"点了完全工作区，但 Agent 还在说只读模式"。
    """
    ss = st.session_state
    want = files.MODE_WRITE if ss.get("write_mode") == MODE_WRITE_LABEL else files.MODE_READ
    changed = want != files.get_mode()
    files.set_mode(want)
    return changed


def _mode_control() -> None:
    """输入框上方的模式开关。

    位置必须在空状态守卫**之前**：空状态下如果看不到这个开关，
    用户就永远切不到"完全工作区"，而"什么都没传也能改本地文件"正是它的用途。
    """
    # 用 radio 而不是 segmented_control：后者在 AppTest 里解析中文选项会崩
    # （ButtonGroup.indices 把 "只读" 切成 "只"），13 项界面测试直接报错。
    # 横向 radio 长得差不多，但测试框架认得。
    st.radio(
        "本地文件权限",
        options=[MODE_READ_LABEL, MODE_WRITE_LABEL],
        index=0,
        key="write_mode",
        horizontal=True,
        help="只读：Agent 不能改动本机文件。完全工作区：可读写任意路径，写入不做备份。",
    )


def _can_chat(has_content: bool) -> bool:
    """是否显示提问框。有数据/资料可以；完全工作区下什么都没有也可以。"""
    return has_content or files.allow_write()


def _prompt_text() -> str:
    meta = dt.dataset_meta()
    return make_system_prompt(
        dt.column_names(),
        meta.get("rows"),
        meta.get("label") or "当前数据",
        dt.document_digest(),
        files.allow_write(),
    )


def _reset_conversation() -> None:
    """换数据集必须清空对话：旧对话里的列名和新表无关，留着只会误导模型。"""
    ss = st.session_state
    ss.session = Session()
    ss.agent = Agent(ss.session, system_prompt=_prompt_text(), session_key=ss.sid)
    ss.turns = []
    ss.conv_title = None


def _refresh_prompt() -> None:
    """数据或资料变了 -> 重算 prompt，但保留对话。

    上传资料走这条路而不是 _reset_conversation：用户刚问完一句，又补了一份
    附件，把对话清掉会很突兀。代价是旧回答可能没用到新文件——这是可接受的。
    """
    ss = st.session_state
    text = _prompt_text()
    agent = ss.get("agent")
    if agent is None:
        ss.agent = Agent(ss.session, system_prompt=text, session_key=ss.sid)
    else:
        agent.update_system_prompt(text)


# ---------------------------------------------------------------- 对话历史
#
# 刷新浏览器 = 一套全新的 session_state（sid、turns、Session 全没了）。所以"对话
# 还在"这件事只能靠磁盘：URL 带一个 ?c=<对话id>，开页时按它把历史和 messages 读回来。
#
# 为什么用查询参数而不是 cookie：不用第三方组件、不受浏览器隐私策略影响，而且
# 复制链接就能在另一个标签页打开同一个对话。代价是 URL 会变长——可以接受。


def _chat_id_from_url() -> str:
    try:
        value = st.query_params.get("c")
    except Exception:  # noqa: BLE001  AppTest / 老版本拿不到就当没有
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "")


def _put_chat_id(chat_id: str) -> None:
    try:
        if _chat_id_from_url() != chat_id:
            st.query_params["c"] = chat_id
    except Exception:  # noqa: BLE001  URL 写不进去不该影响对话本身
        pass


def _chat_table() -> dict[str, Any]:
    """记下这个对话是在哪张表上聊的，恢复时好把表绑回去。"""
    ss = st.session_state
    temp = ss.get("temp_table")
    if temp is not None:
        # 临时表只在内存里，记个名字就够——它注定恢复不了，界面要能说清楚
        return {"kind": "temp", "name": temp[0]}
    entry = _active_table()
    if entry is not None and dt.has_dataset():
        return {"kind": "vault", "id": entry.id, "name": entry.name}
    label = dt.dataset_meta().get("label")
    return {"kind": "table", "name": label} if label else {}


def _chat_sig() -> tuple:
    """当前对话的状态指纹。只用来判断"要不要重写文件"，不做别的事。"""
    ss = st.session_state
    table = _chat_table()
    return (
        ss.get("chat_id"),
        len(ss.get("turns") or []),
        ss.get("conv_title") or "",
        len(ss.session.messages),
        table.get("id") or table.get("name") or "",
    )


def _persist_chat(force: bool = False) -> None:
    """把当前对话写盘。状态没变就跳过，避免每次点按钮都重写一遍。"""
    ss = st.session_state
    chat_id = ss.get("chat_id")
    if not chat_id or not ss.get("turns"):
        return

    table = _chat_table()
    sig = _chat_sig()
    if not force and sig == ss.get("saved_sig"):
        return

    try:
        chats.save(
            chat_id,
            title=ss.get("conv_title") or "",
            table=table,
            turns=ss.turns,
            messages=ss.session.messages,
            stats={
                "turns": ss.session.turns,
                "steps": ss.session.steps,
                "tokens": ss.session.tokens,
                "tool_calls": ss.session.tool_calls,
            },
        )
        ss.saved_sig = sig
    except OSError:
        # 写盘失败不该毁掉这轮对话：内容还在内存里，下一轮再试
        ss.saved_sig = None


def _restore_chat() -> None:
    """按 URL 里的 id 恢复历史；没有 id 就开一个新对话。"""
    ss = st.session_state
    chat_id = _chat_id_from_url()
    if not chat_id:
        ss.chat_id = chats.new_id()
        _put_chat_id(ss.chat_id)
        return

    if ss.get("restored") == chat_id:
        return
    ss.chat_id = chat_id
    ss.restored = chat_id
    ss.saved_sig = None

    chat = chats.load(chat_id)
    if chat is None:
        return

    ss.turns = chat.turns
    ss.conv_title = chat.title or None
    ss.session = Session(
        messages=list(chat.messages),
        turns=int(chat.stats.get("turns") or chat.count),
        steps=int(chat.stats.get("steps") or 0),
        tokens=int(chat.stats.get("tokens") or 0),
        tool_calls=int(chat.stats.get("tool_calls") or 0),
    )
    ss.agent = Agent(ss.session, system_prompt=_prompt_text(), session_key=ss.sid)

    table = chat.table or {}
    if table.get("kind") == "vault" and table.get("id"):
        entry = vault.find(str(table["id"]))
        frame = vault.read_table(entry) if entry is not None else None
        if frame is not None and len(frame.columns) >= 2:
            # 提前绑好、并把 bound 指向它：_apply_dataset 看到 bound 已经对得上
            # 就不会再走 _bind_table（那里会清空对话，正好把刚恢复的历史抹掉）。
            dt.use_dataset(ss.sid)
            dt.bind_dataset(frame, key=ss.sid, label=entry.name)
            ss.table_id = entry.id
            ss.bound = entry.id
            ss.missing_table = None
            # 记住"已经存过了"：否则一打开历史就把 updated_at 刷新一遍，
            # 列表会被"只是看了一眼"的对话顶到最前面。
            ss.saved_sig = _chat_sig()
            return

    # 表回不来了（临时上传的 CSV 早没了，或者文件被删）。
    # 这里必须写成 ""，不能留 None：None 会让 _apply_dataset 自动挑一张表绑上，
    # 那条路径会清空对话——用户刚打开的历史当场蒸发。
    ss.table_id = ""
    ss.missing_table = table.get("name") or ("临时上传的表" if table else None)
    ss.saved_sig = _chat_sig()


def _forget_local_state(chat_id: str) -> None:
    """切对话前把"这一次会话"的东西清干净：连接、内存、磁盘副本、sid。

    必须换 sid：data_tools 的连接和资料都按 sid 存，沿用旧 sid 会让两个对话
    抢同一张表。换了 sid，旧 sid 的文件就成了没人认领的孤儿，所以顺手删掉。
    """
    ss = st.session_state
    _persist_chat(force=True)

    dt.unbind_dataset(ss.sid)
    dt.clear_documents(ss.sid)
    # 磁盘上那一份一起走。临时上传从来不属于"下一个对话"，界面上的文案
    # 写的就是"关掉页面就没了"，切走却把文件留在盘上只是没人管的垃圾。
    uploads.discard(ss.sid)

    ss.sid = uuid.uuid4().hex[:8]
    ss.session = Session()
    ss.agent = None
    ss.turns = []
    ss.conv_title = None
    ss.temp_table = None
    ss.table_id = None
    ss.bound = None
    ss.doc_sig = None
    ss.vault_docs = []
    ss.saved_sig = None
    ss.restored = None
    ss.missing_table = None
    ss.chat_id = chat_id
    _put_chat_id(chat_id)
    st.rerun()


def _delete_chat(chat_id: str) -> None:
    chats.delete(chat_id)
    if chat_id == st.session_state.get("chat_id"):
        _forget_local_state(chats.new_id())
    st.rerun()


def _sync_documents() -> None:
    """把储藏区里的资料挂进本会话，同时保留本次临时上传的那些。

    资料不是"这次上传的"而是"储藏区里有的"：只要文件还在，问它就能答，
    重开浏览器也一样。表不行——表要占一个连接，得显式选一张。
    """
    ss = st.session_state
    docs = vault.documents()
    sig = tuple(entry.id for entry in docs)
    if ss.get("doc_sig") == sig:
        return

    ss.doc_sig = sig

    # 只撤上一轮自己挂上的，临时上传的资料不归这里管
    for name in ss.get("vault_docs") or []:
        existing = dt.find_document(name, ss.sid)
        if existing is not None and not existing.owned:
            dt.remove_document(name, ss.sid)

    mounted: list[str] = []
    for entry in reversed(docs):
        doc = vault.open_document(entry)
        if doc is None:
            continue
        if dt.find_document(doc.name, ss.sid) is not None:
            # 本次临时传了同名文件，以临时的为准：用户刚做的动作不该被悄悄盖掉
            continue
        dt.register_document(doc, key=ss.sid)
        mounted.append(doc.name)
    ss.vault_docs = mounted


def _active_table() -> vault.Entry | None:
    entry_id = st.session_state.get("table_id")
    if not entry_id:
        return None
    return vault.find(entry_id)


def _bind_table(entry: vault.Entry) -> bool:
    frame = vault.read_table(entry)
    if frame is None or len(frame.columns) < 2:
        st.error(f"`{entry.name}` 读不出可分析的表：至少要有表头和两列。")
        return False

    dt.bind_dataset(frame, key=st.session_state.sid, label=entry.name)
    st.session_state.bound = entry.id
    _reset_conversation()
    return True


def _apply_dataset() -> bool:
    """准备本会话的数据表。返回是否绑定了表。"""
    ss = st.session_state

    # 永远指向本会话的 key。每个标签页跑在自己的线程里，这行只影响本次执行；
    # 但漏了它，本进程第一个读数据的工具就会用 ContextVar 的默认值（别人那份）。
    dt.use_dataset(ss.sid)
    _sync_documents()

    temp = ss.get("temp_table")
    if temp is not None:
        name, frame = temp
        if ss.get("bound") != f"temp:{name}" or not dt.has_dataset():
            dt.bind_dataset(frame, key=ss.sid, label=name)
            ss.bound = f"temp:{name}"
            _reset_conversation()
        _refresh_prompt()
        return dt.has_dataset()

    if ss.get("table_id") is None:
        available = vault.tables()
        ss.table_id = available[0].id if available else ""

    entry = _active_table()
    if entry is None:
        ss.bound = None
        _refresh_prompt()
        return False

    if ss.get("bound") != entry.id or not dt.has_dataset():
        if not _bind_table(entry):
            ss.table_id = ""

    _refresh_prompt()
    return dt.has_dataset()


# ---------------------------------------------------------------- 上传
#
# 两个框，按"留多久"分，不按文件类型分：
#   临时上传 —— 只进这次会话，进程一停就没了，不占储藏区额度；
#   长期上传 —— 存进储藏区，重启还在，算 200 MB 额度。
# 类型（表还是资料）都由后缀决定，不让用户选。


def _reset_uploader(which: str) -> None:
    """清空上传框。

    换一个 key 等于换一个新控件，里面已选的文件跟着丢掉。必须这么做：控件会把
    选过的文件一直交回来，光删签名的话下一轮又会被重新登记一遍。
    """
    ss = st.session_state
    ss[f"{which}_gen"] = int(ss.get(f"{which}_gen", 0)) + 1
    ss[f"{which}_sigs"] = {}


def _handle_temp_upload(files) -> None:
    """临时上传：只登记进这次会话，不写储藏区。"""
    ss = st.session_state
    if not files:
        return

    changed = False
    for file in files:
        signature = f"{file.name}:{file.size}"
        if signature in ss.temp_sigs:
            continue

        raw = file.getvalue()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"`{file.name}` 超过 {MAX_UPLOAD_MB} MB。先压缩，或者拆成几份再传。")
            continue

        kind = readers.route(file.name)
        if kind == "unsupported":
            st.error(f"`{file.name}` 这种格式还不支持。")
            continue

        if kind == "table":
            frame = read_any_csv(raw)
            if frame is None:
                st.error(f"`{file.name}` 读不出表格。检查一下是不是 CSV、有没有表头。")
                continue
            if len(frame.columns) < 2:
                st.error(
                    f"`{file.name}` 只有一列。至少需要两列（一个维度 + 一个数值）才能分析。"
                )
                continue
            ss.temp_table = (vault.safe_name(file.name), frame)
            # 临时表顶掉储藏区那张。置 "" 而不是 None，否则下一轮又会自动绑回储藏区的表。
            ss.table_id = ""
            ss.bound = None
        else:
            try:
                doc = dt.add_document(file.name, raw, key=ss.sid)
            except DocumentError as exc:
                # 刻意**不**记签名：记了就等于"这个文件永久失败"，用户改好再传也会被跳过。
                st.error(f"`{file.name}`：{exc}")
                continue
            except Exception as exc:  # noqa: BLE001  一份文件坏了不该拖垮整页
                st.error(f"`{file.name}` 读取失败（{type(exc).__name__}）：{exc}")
                continue
            if doc.note:
                st.warning(f"`{doc.name}`：{doc.note}")

        ss.temp_sigs[signature] = file.name
        changed = True

    if changed:
        st.rerun()


def _handle_vault_upload(files) -> None:
    """长期上传：先存进储藏区，再决定它是表还是资料。"""
    ss = st.session_state
    if not files:
        return

    changed = False
    for file in files:
        signature = f"{file.name}:{file.size}"
        if signature in ss.vault_sigs:
            continue

        raw = file.getvalue()
        kind = readers.route(file.name)
        if kind == "unsupported":
            st.error(f"`{file.name}` 这种格式还不支持。")
            continue
        if kind == "table":
            frame = read_any_csv(raw)
            if frame is None:
                st.error(f"`{file.name}` 读不出表格。检查一下是不是 CSV、有没有表头。")
                continue
            if len(frame.columns) < 2:
                st.error(
                    f"`{file.name}` 只有一列。至少需要两列（一个维度 + 一个数值）才能分析。"
                )
                continue

        try:
            entry = vault.add(file.name, raw)
        except vault.VaultFull as exc:
            st.error(str(exc))
            break
        except DocumentError as exc:
            st.error(f"`{file.name}`：{exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            st.error(f"`{file.name}` 保存失败（{type(exc).__name__}）：{exc}")
            continue

        ss.vault_sigs[signature] = entry.name
        if entry.note:
            st.warning(f"`{entry.name}`：{entry.note}")
        if entry.is_table:
            ss.table_id = entry.id
            ss.temp_table = None
            ss.bound = None
        changed = True

    if changed:
        st.rerun()


# ---------------------------------------------------------------- 当前表 / 储藏区


def _remove_table() -> None:
    """从这次会话撤下当前表，文件本身不动。

    例外是**临时表**：它的 CSV 是这次会话写给沙箱看的副本，撤下就没有
    第二个地方会用到它了，所以连盘上那份一起删。储藏区的表归 vault 管，
    不在这里删。
    """
    ss = st.session_state
    temp = ss.get("temp_table") is not None
    dt.unbind_dataset(ss.sid)
    if temp:
        uploads.drop_table(ss.sid)
    ss.bound = None
    if temp:
        ss.temp_table = None
        # 撤的是临时表，储藏区那张可以接着顶上；撤的是储藏区的表，就让位子空着
        ss.table_id = None
    else:
        ss.table_id = ""
    _refresh_prompt()
    st.rerun()


def _use_table(entry: vault.Entry) -> None:
    ss = st.session_state
    ss.table_id = entry.id
    ss.temp_table = None
    ss.bound = None
    st.rerun()


def _drop_entry(entry: vault.Entry) -> None:
    """从储藏区删掉，磁盘上的文件一起删。"""
    ss = st.session_state
    active = ss.get("table_id") == entry.id
    vault.drop(entry.id)
    if active:
        dt.unbind_dataset(ss.sid)
        ss.table_id = ""
        ss.bound = None
    ss.doc_sig = None
    _reset_uploader("vault")
    _refresh_prompt()
    st.rerun()


def _remove_temp_doc(name: str) -> None:
    """临时资料：从会话里撤掉，磁盘副本一起删（它本来就不该长期留着）。"""
    ss = st.session_state
    dt.remove_document(name, ss.sid)
    # 储藏区里可能有同名文件，刚才被临时这份挡住没挂上，现在得重新挂一次
    ss.doc_sig = None
    _reset_uploader("temp")
    _refresh_prompt()
    st.rerun()


def _render_temp_sources() -> None:
    """本次临时传的资料。表走 _render_active_table，这里只管资料。"""
    ss = st.session_state
    docs = [d for d in dt.documents(ss.sid) if d.owned]
    for doc in docs:
        left, right = st.columns([3, 1])
        left.markdown(f"`{doc.name}`")
        left.caption(f"临时资料 · {doc.chars:,} 字 · 关掉页面就没了")
        if right.button("移除", key=f"rm_temp_{doc.name}", use_container_width=True):
            _remove_temp_doc(doc.name)


def _render_active_table() -> None:
    if not dt.has_dataset():
        return
    left, right = st.columns([3, 1])
    left.markdown(f"当前表 {dataset_badge(dt.dataset_meta())}")
    if right.button("撤下", key="drop_table", use_container_width=True):
        _remove_table()


def _render_vault() -> None:
    ss = st.session_state
    st.subheader("文件储藏区")
    used = vault.usage()
    limit = vault.quota()
    st.progress(min(used / limit, 1.0) if limit else 0.0)
    st.caption(f"{size_label(used)} / {size_label(limit)}")
    files = st.file_uploader(
        "上传到储藏区",
        type=UPLOAD_SUFFIXES,
        accept_multiple_files=True,
        key=f"vault_uploader_{ss.get('vault_gen', 0)}",
        label_visibility="collapsed",
        help=f"长期保留：单个最大 {MAX_UPLOAD_MB} MB，储藏区一共 {VAULT_QUOTA_MB} MB。",
    )
    _handle_vault_upload(files)

    items = vault.entries()
    if not items:
        st.caption("传一个文件就存到这里。之后不用重传，也能直接问它。")
        return

    for entry in items:
        active = entry.is_table and dt.has_dataset() and ss.get("table_id") == entry.id
        left, mid, right = st.columns([3, 1, 1])
        left.markdown(f"`{entry.name}`")
        suffix = entry.suffix.lstrip(".").upper() or "文档"
        if entry.is_table:
            detail = f"数据表 · {size_label(entry.size)}"
        else:
            detail = f"{suffix} 资料 · {entry.chars:,} 字 · {size_label(entry.size)}"
        left.caption(detail + (" · 正在分析" if active else ""))

        if entry.is_table:
            if active:
                mid.caption("在用")
            elif mid.button("载入", key=f"use_{entry.id}", use_container_width=True):
                _use_table(entry)
        if right.button("删除", key=f"del_{entry.id}", use_container_width=True):
            _drop_entry(entry)


def _render_chat_history() -> None:
    """侧边栏的历史列表：最多 8 条，最近更新的在前。"""
    ss = st.session_state
    st.subheader("历史对话")
    if st.button("新对话", use_container_width=True, key="new_chat"):
        _forget_local_state(chats.new_id())

    if ss.get("missing_table"):
        st.caption(f"这个对话用的表 `{ss.missing_table}` 已经不在了。重新传一份，或者换个对话。")

    items = chats.list_chats(limit=8)
    if not items:
        st.caption("问答会自动存下来，关掉页面也还在。")
        return

    for item in items:
        chat_id = str(item.get("id"))
        title = item.get("title") or "未命名对话"
        left, right = st.columns([5, 1])
        if chat_id == ss.get("chat_id"):
            left.caption(f"▸ {title}")
        elif left.button(title, key=f"open_{chat_id}", use_container_width=True):
            _forget_local_state(chat_id)
        if right.button("×", key=f"rm_chat_{chat_id}", help="删除这个对话"):
            _delete_chat(chat_id)


# ---------------------------------------------------------------- 右侧「模型设置」
#
# 要求是"模型不要写死在代码里"。所以这一栏做的事很明确：把 base_url / 模型名 /
# api_key 三件东西交给用户填，填完立刻生效，并且能先测一下再决定用不用。
#
# 四个细节值得记：
#   1. 输入框的值放 session_state（`model_*` 三个键），点「保存并应用」才写进 config。
#      边打字边生效会让每敲一个字都重建 Agent，输入框还会被 rerun 打断。
#   2. 「测试连接」只测、不保存。用户可能只是想知道这个 Key 还有没有额度，
#      并不打算换掉正在用的配置。
#   3. 存的是一个**列表**（data/model.json，已 gitignore），不是一个"最后用的"。
#      换供应商是常事，列表里点一行就切过去，不用把三样东西重敲一遍。
#   4. 切换时改 `model_*` 是安全的：列表画在输入框之前，这一趟还没实例化它们；
#      反过来（输入框画完再改）Streamlit 会当场抛 cannot be modified。


def _toggle_model_panel() -> None:
    st.session_state.model_panel = not st.session_state.get("model_panel")
    st.rerun()


def _short_model(limit: int = 16) -> str:
    """收起时按钮上直接显示当前模型名。

    以前这里写死「模型设置」，收起之后看不出到底在用哪套配置——换过模型的人
    只能再展开一次确认，还容易怀疑"我填的是不是没存住"。名字太长会把这个窄按钮
    撑变形，截断即可，完整名字展开后看得到。
    """
    name = config.current().model or "未配置"
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _flash(kind: str, text: str) -> None:
    """跨 rerun 传一条提示。

    Streamlit 里 `st.success(...)` 紧接着 `st.rerun()` 会当场消失——所以先存起来，
    下一趟再画。
    """
    st.session_state.model_flash = (kind, text)


def _test_connection(base_url: str, model: str, api_key: str) -> None:
    """真发一条 1 token 的请求。不保存任何东西。"""
    with st.spinner("正在连接…"):
        result = llm_probe(base_url, model, api_key)
    if result.ok:
        st.success(result.summary)
    else:
        st.error(result.summary)


def _apply_model(base_url: str, model: str, api_key: str) -> None:
    """把输入框里的这套存进列表并生效。同名（默认用模型名）覆盖，不新增。"""
    try:
        cfg = config.set_current(base_url=base_url, model=model, api_key=api_key)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        config.save(cfg)
        _flash("success", f"已保存 `{config.current().name}`，正在使用。")
    except OSError:
        # 存不下不影响这次使用，但不能假装存好了
        _flash("warning", f"已切换到 `{cfg.model}`，但没能保存：下次打开页面会回到原来的设置。")
    st.rerun()


def _use_model(name: str) -> None:
    """切到列表里的某一条，顺手把三个输入框刷成那套值。"""
    try:
        cfg = config.activate(name)
    except KeyError:
        _flash("warning", "这条配置已经不在了。")
        st.rerun()
        return

    ss = st.session_state
    ss.model_base_url = cfg.base_url
    ss.model_model = cfg.model
    ss.model_api_key = cfg.api_key
    _flash("success", f"已切换到 `{cfg.name}`。")
    st.rerun()


def _delete_model(name: str) -> None:
    """从列表里删掉一条。删的若是正在用的那条，会退回默认配置。"""
    cfg = config.forget(name)
    ss = st.session_state
    ss.model_base_url = cfg.base_url
    ss.model_model = cfg.model
    ss.model_api_key = cfg.api_key
    _flash("info", f"已删除 `{name}`。")
    st.rerun()


def _reset_model() -> None:
    """恢复 .env 默认。

    不能在这里直接改 `model_base_url` 三个键——它们已经绑在上面的控件上了，
    Streamlit 会当场抛 "cannot be modified after the widget is instantiated"。
    挂一个待办标记，下一趟在控件创建**之前**再赋值。
    """
    config.clear_saved()
    st.session_state.model_reset_pending = True
    _flash("info", "已恢复默认配置。")
    st.rerun()


def _apply_pending_model_reset() -> None:
    """把「恢复默认」的赋值补上。必须在任何模型输入框实例化之前调用。"""
    ss = st.session_state
    if not ss.pop("model_reset_pending", False):
        return
    cfg = config.current()
    ss.model_base_url = cfg.base_url
    ss.model_model = cfg.model
    ss.model_api_key = cfg.api_key


def _render_flash() -> None:
    ss = st.session_state
    flash = ss.get("model_flash")
    if not flash:
        return
    ss.model_flash = None
    kind, text = flash
    {"success": st.success, "warning": st.warning, "info": st.info}.get(kind, st.info)(text)


def _render_saved_list() -> None:
    """已保存的配置列表：点一行就切过去，不用把三样东西重敲一遍。"""
    items = config.saved()
    if not items:
        return

    st.caption("已保存的配置")
    active = config.current().name
    for i, item in enumerate(items):
        left, right = st.columns([6, 1])
        if item.name == active:
            left.caption(f"▸ {item.name}")
        elif left.button(item.name, key=f"use_model_{i}", use_container_width=True):
            _use_model(item.name)
        if right.button("×", key=f"remove_model_{i}", help="删除这条配置"):
            _delete_model(item.name)
    st.divider()


def _render_model_panel() -> None:
    """右栏本体。展开时才渲染——收起状态下一行都不画。

    文案只留"填什么"，不解释实现。界面上不该出现 data/model.json、状态码、
    "不用改代码"这类话——那是写给人看代码的，不是给用户看的。
    """
    ss = st.session_state
    cfg = config.current()

    st.subheader("模型设置")
    _render_flash()
    _render_saved_list()

    st.text_input("Base URL", key="model_base_url", placeholder="https://api.deepseek.com/v1")
    st.text_input("模型名", key="model_model", placeholder="deepseek-chat")
    st.text_input("API Key", key="model_api_key", type="password", help="只保存在本机")

    left, right = st.columns(2)
    if left.button("保存并应用", key="model_apply", type="primary", use_container_width=True):
        _apply_model(ss.model_base_url, ss.model_model, ss.model_api_key)
    if right.button("测试连接", key="model_test", use_container_width=True):
        _test_connection(ss.model_base_url, ss.model_model, ss.model_api_key)

    st.caption(f"正在使用 `{cfg.model}` · Key {cfg.masked_key}")
    if st.button("恢复默认配置", key="model_reset", use_container_width=True):
        _reset_model()


# ---------------------------------------------------------------- 渲染


def _render_sidebar(has_content: bool) -> DeltaGenerator:
    """画侧边栏，返回「状态」那一行的占位符。

    统计留成 empty()：侧边栏比提问框先渲染，数字却是提问之后才变的，
    只有把槽位交出去，_run_turn 才能一边跑一边往上写。
    """
    ss = st.session_state
    with st.sidebar:
        st.subheader("临时上传")
        temp_files = st.file_uploader(
            "上传 CSV / PDF / Word / PPT / 文本 / 日志",
            type=UPLOAD_SUFFIXES,
            accept_multiple_files=True,
            key=f"temp_uploader_{ss.get('temp_gen', 0)}",
            label_visibility="collapsed",
            help=f"单个文件最大 {MAX_UPLOAD_MB} MB。",
        )
        _handle_temp_upload(temp_files)
        _render_temp_sources()
        _render_active_table()

        st.divider()
        _render_vault()

        st.divider()
        _render_chat_history()

        st.divider()
        st.subheader("状态")
        st.caption(f"模型 `{config.current().model}`")
        stats_slot = st.empty()
        stats_slot.caption(ss.session.stats())
        if has_content and st.button("清空对话", use_container_width=True):
            # 文件也一起删：只清内存的话，刷新一下历史又回来了
            chats.delete(ss.chat_id or "")
            _reset_conversation()
            st.rerun()

        questions: list[str] = []
        if dt.has_dataset():
            questions = suggest_questions(dt.column_names())
        elif dt.has_documents():
            first = dt.documents()[0].name
            questions = [f"总结一下 {first} 的主要内容", f"{first} 里有哪些关键数字？"]
        if questions:
            st.divider()
            st.subheader("试试这些")
            for index, question in enumerate(questions):
                if st.button(question, use_container_width=True, key=f"suggest_{index}"):
                    ss.pending = question
                    st.rerun()

        st.divider()

    return stats_slot


def _download_buttons(files: list[str], *, prefix: str) -> None:
    """生成的 pptx 只能给下载按钮。

    Streamlit 没有内嵌 Office 预览，pptx 也不能像 png 那样 st.image。
    文件可能已被清理，路径没了就跳过——老对话不该把整页拖成红色报错。
    """
    for index, raw in enumerate(files or []):
        path = Path(raw)
        if not path.is_file():
            continue
        st.download_button(
            f"下载 {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            key=f"dl_{prefix}_{index}_{path.stem}",
        )


def _render_turn(turn: dict, index: int = 0) -> None:
    with st.chat_message("user"):
        st.markdown(turn.get("question", ""))
    with st.chat_message("assistant"):
        if turn.get("trace"):
            with st.expander("思考过程", expanded=False):
                st.markdown(turn["trace"])
        st.markdown(turn.get("answer", ""))
        for chart in turn.get("charts") or []:
            if Path(chart).is_file():
                st.image(chart, use_container_width=True)
        _download_buttons(turn.get("files") or [], prefix=f"h{index}")


def _render_history() -> None:
    for index, turn in enumerate(st.session_state.turns):
        _render_turn(turn, index)


def _run_turn(question: str, stats_slot: DeltaGenerator | None = None, page_title_slot: DeltaGenerator | None = None) -> None:
    ss = st.session_state

    def _set_titles(browser_text: str, page_text: str) -> None:
        # 标签页逻辑原样保留；顺手把页面大标题一起换掉，两处用同一份文本源，
        # 页面那边去掉 · DataInsight 后缀（标签页要短，页面要干净）。
        _set_browser_title(browser_text)
        if page_title_slot is not None:
            page_title_slot.title(page_text)

    with st.chat_message("user"):
        st.markdown(question)

    # 标题也是进度条：先挂"分析中"。留成占位符，标题线程一回来就把它换掉，
    # 这样回答还在流的时候，标签页和页面标题都已经能显示这轮在聊什么了。
    title_slot = st.empty()
    with title_slot:
        _set_titles("⏳ 分析中… · DataInsight", "⏳ 分析中…")

    # 只有第一轮需要起标题；后面沿用，免得每提一句标签页就跳一次
    job = _start_title_job(question) if ss.conv_title is None else None

    def flush_title() -> None:
        if job is None or ss.conv_title is not None or not job["done"]:
            return
        ss.conv_title = job["title"]
        with title_slot:
            _set_titles(f"{ss.conv_title} · DataInsight", ss.conv_title)

    def show_stats() -> None:
        if stats_slot is not None:
            stats_slot.caption(ss.session.stats())

    trace = TraceLog()
    with st.chat_message("assistant"):
        trace_slot = st.empty()
        answer_slot = st.empty()

        def on_event(kind: str, payload: dict) -> None:
            # 每个事件都先看一眼标题线程：它比回答快，基本第一个 token 时就绪
            flush_title()
            if kind == "delta":
                answer_slot.markdown(payload.get("text", "") + " ▌")
                return
            if kind == "tool_call":
                answer_slot.empty()
            trace.push(kind, payload)
            trace_slot.markdown(trace.to_markdown())
            show_stats()

        ss.agent.on_event = on_event
        answer = ss.agent.ask(question)
        show_stats()
        flush_title()

        trace_slot.empty()
        if trace.items:
            with trace_slot.expander("思考过程", expanded=False):
                st.markdown(trace.to_markdown())
        answer_slot.markdown(answer)
        for chart in trace.charts:
            st.image(chart, use_container_width=True)
        _download_buttons(trace.files, prefix=f"t{len(ss.turns)}")

    ss.turns.append(
        {
            "question": question,
            "answer": answer,
            "trace": trace.to_markdown(),
            "charts": trace.charts,
            "files": trace.files,
        }
    )

    # 标题线程一般比回答快；真没赶上（模型慢/网络抖），退回收首的问题，
    # 别把标签页和页面标题永远留在"分析中"。
    if ss.conv_title is None:
        ss.conv_title = job["title"] if (job and job["done"]) else _fallback_title(question)
    with title_slot:
        _set_titles(f"{ss.conv_title} · DataInsight", ss.conv_title)

    # 落盘放在最后：标题这时已经定了，一次写入就带齐 turns / messages / 标题。
    _persist_chat(force=True)


def _render_body(has_content: bool, has_table: bool) -> None:
    """主区域：标题、左栏、对话历史、模式开关、输入框。

    抽成函数是为了右侧展开时能整个塞进左列。注意 `st.chat_input` 一旦放进列里
    就不再有"吸附在页面底部"的行为，会跟在对话流末尾——这是 Streamlit 的既定行为，
    不是配置能改的。所以只在用户主动展开设置时付这个代价。
    """
    ss = st.session_state

    head_left, head_right = st.columns([5, 1])
    label = "▸ 收起" if ss.get("model_panel") else f"⚙ {_short_model()}"
    if head_right.button(
        label, key="model_toggle", use_container_width=True, help="模型设置"
    ):
        _toggle_model_panel()

    # set_page_config 只能写死标题，动态部分靠这里每次重跑时补一刀
    _set_browser_title(_browser_title())

    # 页面大标题也跟着会话状态走：占位符留好，_run_turn 里实时刷新
    page_title_slot = st.empty()
    with page_title_slot:
        st.title(_page_title())

    stats_slot = _render_sidebar(has_content)
    _render_history()

    _mode_control()

    if not _can_chat(has_content):
        st.caption("往左边传文件：临时上传的只留到这次会话结束，传进储藏区的会长期保留。")
        return

    if has_table:
        placeholder = "问关于这张表的任何问题，例如：各城市销售额排名前五"
    elif files.allow_write():
        placeholder = "让 Agent 处理本地文件，例如：把桌面的 test.txt 写成三条本周待办"
    else:
        placeholder = "问关于这些文件的问题，例如：这份报告的主要结论是什么"
    question = st.chat_input(placeholder)
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")

    if question:
        if not config.current().api_key:
            st.error("还没有 API Key：点右上角「⚙ 模型设置」填一个。")
            return
        _run_turn(question, stats_slot, page_title_slot)


_SWEPT = False


def _sweep_uploads_once() -> None:
    """进程起来后收一次上一次留下的临时副本。

    做一次就够了，所以用模块级开关而不是 session_state：后者是每个浏览器
    标签页各一份，第二个标签页还会再扫一遍。反过来，真正的遗留由
    uploads.sweep 里的 mtime 判断兜底，本进程刚写进去的文件不会被误删
    （包括另一个标签页正在用的）。
    """
    global _SWEPT
    if _SWEPT:
        return
    _SWEPT = True
    uploads.sweep()


def main() -> None:
    st.set_page_config(page_title="DataInsight", page_icon="📊", layout="wide")

    _sweep_uploads_once()

    # 新写的工具模块要在这里补一次导入，否则进程里看不到它：
    # 注册靠 import 副作用，而 app.tools 只在第一次导入时执行一次，
    # 之后新建的文件不会自动出现在模型拿到的工具列表里。
    load_missing()

    _init_state()
    _apply_pending_model_reset()
    mode_changed = _sync_mode()
    _restore_chat()
    has_table = _apply_dataset()
    if mode_changed or st.session_state.get("agent") is None:
        # 没有数据也没有资料时，_apply_dataset 不会建 Agent；
        # 而「完全工作区」下什么都能问，所以这里必须兜一次。
        _refresh_prompt()
    _persist_chat()
    has_content = has_table or dt.has_documents()

    if st.session_state.get("model_panel"):
        body, panel = st.columns([0.72, 0.28], gap="large")
        with body:
            _render_body(has_content, has_table)
        with panel:
            _render_model_panel()
    else:
        _render_body(has_content, has_table)
