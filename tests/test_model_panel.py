"""运行时模型配置 + 右侧「模型设置」面板。

两层各测各的：
  1. 配置层（config / llm.probe）：纯逻辑，不联网、不碰真实 data/model.json；
  2. 界面层（AppTest 跑 app.py）：面板默认收起、展开后有那三个输入框、
     点「应用」之后 config.current() 真的变了。

之所以连界面一起测：这一栏的价值全在"填了之后生效"。只测 config.set_current()
能过，不代表界面上那个按钮接对了函数——中间那一层是手写的胶水。
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from _isolation import isolate_uploads
from app import config, llm

ROOT = Path(__file__).resolve().parent.parent

try:
    from streamlit.testing.v1 import AppTest

    HAS_APPTEST = True
except Exception:  # noqa: BLE001  没装 streamlit 就只跑配置层
    HAS_APPTEST = False


class _FakeResponse:
    """假的 HTTP 响应：只要能 with 和 read 就够了。"""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test/v1/chat/completions", code, "err", {}, io.BytesIO(body.encode("utf-8"))
    )


class ConfigCase(unittest.TestCase):
    """每个用例一个临时 model.json，绝不碰用户真实的那份。"""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="model-cfg-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.file = self.dir / "model.json"

        patcher = mock.patch.object(config, "MODEL_FILE", self.file)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.default = config.ModelConfig("https://default.test/v1", "default-model", "sk-default-key")
        patcher2 = mock.patch.object(config, "ENV_DEFAULTS", self.default)
        patcher2.start()
        self.addCleanup(patcher2.stop)

        config.reload()
        self.addCleanup(config.reload)

    def write_saved(self, **fields) -> None:
        self.file.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")


class TestConfigStore(ConfigCase):
    def test_default_comes_from_env_when_no_file(self):
        self.assertEqual(config.current(), self.default)

    def test_saved_file_wins_over_env(self):
        self.write_saved(base_url="https://saved.test/v1", model="saved-model", api_key="sk-saved")
        config.reload()

        active = config.current()
        self.assertEqual(active.base_url, "https://saved.test/v1")
        self.assertEqual(active.model, "saved-model")

    def test_save_is_readable_after_reload(self):
        config.set_current(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        config.save()

        config.reload()
        self.assertEqual(config.current().model, "gpt-4o-mini")
        self.assertEqual(config.current().base_url, "https://api.openai.com/v1")

    def test_save_leaves_no_temp_file(self):
        """写盘走"临时文件 + 替换"，别在半路留下 .tmp，也别写出半截 JSON。"""
        config.set_current(model="m1")
        config.save()

        self.assertEqual(list(self.file.parent.glob("model.json.tmp")), [])
        saved = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(saved["active"], "m1")
        self.assertEqual([p["model"] for p in saved["profiles"]], ["m1"])

    def test_partial_update_keeps_other_fields(self):
        config.set_current(model="only-model")
        active = config.current()

        self.assertEqual(active.model, "only-model")
        self.assertEqual(active.base_url, self.default.base_url)  # 没传的沿用

    def test_blank_base_url_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            config.set_current(base_url="   ")
        self.assertIn("Base URL", str(ctx.exception))

    def test_base_url_without_scheme_rejected(self):
        """漏写 https:// 是最常见的填错，必须在入口就拦住。"""
        with self.assertRaises(ValueError):
            config.set_current(base_url="api.deepseek.com/v1")

    def test_blank_model_rejected(self):
        with self.assertRaises(ValueError):
            config.set_current(model="  ")

    def test_trailing_slash_removed(self):
        config.set_current(base_url="https://a.test/v1/")
        self.assertEqual(config.current().base_url, "https://a.test/v1")

    def test_endpoint_accepts_full_path(self):
        """有人会把整条 /chat/completions 粘进来，不能再拼一次。"""
        cfg = config.ModelConfig("https://a.test/v1/chat/completions", "m", "k")
        self.assertEqual(cfg.endpoint, "https://a.test/v1/chat/completions")

    def test_endpoint_added_when_missing(self):
        cfg = config.ModelConfig("https://a.test/v1", "m", "k")
        self.assertEqual(cfg.endpoint, "https://a.test/v1/chat/completions")

    def test_masked_key_never_shows_whole_key(self):
        cfg = config.ModelConfig("https://a.test/v1", "m", "sk-1234567890abcd")
        self.assertNotIn("567890", cfg.masked_key)
        self.assertTrue(cfg.masked_key.startswith("sk-1"))

    def test_masked_key_short_and_empty(self):
        self.assertEqual(config.ModelConfig("https://a.test/v1", "m", "").masked_key, "未填写")
        self.assertEqual(config.ModelConfig("https://a.test/v1", "m", "short").masked_key, "已填写")

    def test_broken_file_falls_back_to_default(self):
        """文件被手动改坏不该让页面打不开——退回默认，还能用。"""
        self.file.write_text("{ 不是 JSON", encoding="utf-8")
        self.assertEqual(config.reload(), self.default)
        self.assertEqual(config.saved(), [])

    def test_old_single_config_file_migrates(self):
        """第一版只存一套（扁平 dict），升级后要能直接读成列表里的一条。"""
        self.write_saved(base_url="https://old.test/v1", model="old-model", api_key="sk-old")
        config.reload()

        self.assertEqual(config.current().model, "old-model")
        self.assertEqual(config.current().name, "old-model")   # 名字用模型名补上
        self.assertEqual([p.name for p in config.saved()], ["old-model"])

    def test_save_collects_into_a_list(self):
        """每套都留下：换模型不等于把上一套抹掉。"""
        for name in ("m-a", "m-b", "m-c"):
            config.set_current(model=name)
            config.save()

        self.assertEqual([p.name for p in config.saved()], ["m-a", "m-b", "m-c"])
        self.assertEqual(config.current().name, "m-c")

    def test_save_same_model_overwrites_in_place(self):
        config.set_current(model="m-a", api_key="k1")
        config.save()
        config.set_current(model="m-b")
        config.save()
        config.set_current(model="m-a", api_key="k2")
        config.save()

        names = [p.name for p in config.saved()]
        self.assertEqual(names, ["m-a", "m-b"])          # 不重复、顺序不跳
        self.assertEqual(next(p for p in config.saved() if p.name == "m-a").api_key, "k2")

    def test_same_model_on_another_host_gets_its_own_entry(self):
        """同一个模型名挂两个网关：不能互相覆盖，否则配第二套等于丢第一套。"""
        config.set_current(base_url="https://a.test/v1", model="m", api_key="k1")
        config.save()
        config.set_current(base_url="https://b.test/v1", model="m", api_key="k2")
        config.save()

        names = [p.name for p in config.saved()]
        self.assertEqual(len(names), 2, names)
        self.assertEqual(names[0], "m")
        self.assertIn("b.test", names[1])

    def test_activate_switches_and_persists_choice(self):
        config.set_current(model="m-a", api_key="k-a")
        config.save()
        config.set_current(model="m-b", api_key="k-b")
        config.save()

        config.activate("m-a")
        self.assertEqual(config.current().api_key, "k-a")

        config.reload()                                   # 重启后还是上次选的那条
        self.assertEqual(config.current().name, "m-a")

    def test_activate_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            config.activate("不存在")

    def test_forget_removes_only_that_one(self):
        for name in ("m-a", "m-b"):
            config.set_current(model=name)
            config.save()

        config.forget("m-a")
        self.assertEqual([p.name for p in config.saved()], ["m-b"])
        self.assertEqual(config.current().name, "m-b")

    def test_forget_active_falls_back_to_default(self):
        """删掉正在用的那条，回默认，而不是偷偷换成列表里另一条。"""
        config.set_current(model="m-a")
        config.save()
        config.set_current(model="m-b")
        config.save()

        config.forget("m-b")
        self.assertEqual(config.current(), self.default)
        self.assertEqual([p.name for p in config.saved()], ["m-a"])

    def test_clear_saved_returns_to_default_but_keeps_list(self):
        """恢复默认只切回 .env，不把攒下来的配置一起删掉。"""
        config.set_current(model="temp-model")
        config.save()
        self.assertTrue(self.file.exists())

        restored = config.clear_saved()
        self.assertEqual(restored, self.default)
        self.assertEqual([p.name for p in config.saved()], ["temp-model"])

    def test_last_config_removed_deletes_file(self):
        config.set_current(model="only-one")
        config.save()
        config.forget("only-one")
        self.assertFalse(self.file.exists())


class TestProbe(unittest.TestCase):
    """测试连接：成功要报延迟和模型名，失败要把"哪一项错了"说清楚。"""

    def probe_with(self, trigger):
        with mock.patch.object(llm.urllib.request, "urlopen", side_effect=trigger) as urlopen:
            result = llm.probe("https://example.test/v1", "m", "sk-key")
        return result, urlopen

    def test_success_reports_model_and_latency(self):
        body = {"model": "m-2026", "choices": [{"message": {"role": "assistant", "content": "p"}}]}
        result, _ = self.probe_with(lambda request, timeout=0: _FakeResponse(json.dumps(body)))

        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.model, "m-2026")
        self.assertGreaterEqual(result.latency, 0)
        self.assertIn("连接正常", result.summary)

    def test_request_is_minimal_and_not_streamed(self):
        body = {"choices": [{"message": {"content": "p"}}]}
        sent: list[dict] = []

        def fake(request, timeout=0):
            sent.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse(json.dumps(body))

        self.probe_with(fake)
        self.assertEqual(sent[0]["max_tokens"], 1)
        self.assertFalse(sent[0]["stream"])
        self.assertEqual(sent[0]["model"], "m")

    def test_401_points_at_the_key(self):
        result, _ = self.probe_with(lambda request, timeout=0: _raise(_http_error(401, '{"error":"bad key"}')))
        self.assertFalse(result.ok)
        self.assertIn("Key", result.detail)

    def test_404_points_at_url_or_model(self):
        result, _ = self.probe_with(lambda request, timeout=0: _raise(_http_error(404)))
        self.assertFalse(result.ok)
        self.assertIn("地址或模型名不存在", result.detail)

    def test_400_mentions_model_name(self):
        result, _ = self.probe_with(lambda request, timeout=0: _raise(_http_error(400)))
        self.assertIn("模型名", result.detail)

    def test_non_json_response_explained(self):
        result, _ = self.probe_with(lambda request, timeout=0: _FakeResponse("<html>404</html>"))
        self.assertFalse(result.ok)
        self.assertIn("不是 JSON", result.detail)

    def test_error_object_in_body(self):
        body = {"error": {"message": "insufficient balance"}}
        result, _ = self.probe_with(lambda request, timeout=0: _FakeResponse(json.dumps(body)))
        self.assertFalse(result.ok)
        self.assertIn("insufficient balance", result.detail)

    def test_network_failure_explained(self):
        result, _ = self.probe_with(lambda request, timeout=0: _raise(urllib.error.URLError("timed out")))
        self.assertFalse(result.ok)
        self.assertIn("连不上", result.detail)

    def test_empty_fields_fail_without_network(self):
        """空字段必须当场拒，不能真发一条请求再回来说"Key 是空的"。"""
        with mock.patch.object(llm.urllib.request, "urlopen") as urlopen:
            self.assertFalse(llm.probe("", "m", "k").ok)
            self.assertFalse(llm.probe("https://a.test/v1", "", "k").ok)
            self.assertFalse(llm.probe("https://a.test/v1", "m", "").ok)
        urlopen.assert_not_called()


def _raise(exc):
    raise exc


_CHAT_DIR = None


def setUpModule() -> None:
    """面板会写 data/model.json 和 data/chats/，整个模块都挪到临时目录。

    AppTest 会把页面完整跑一遍：页面里有 _persist_chat()，不隔开的话，
    跑一次测试就往用户真实的对话列表里塞几条空对话；入口里还有一次
    "清掉上个进程遗留"的上传区扫描，不隔开就会把用户真实的临时上传删掉。
    """
    global _CHAT_DIR
    isolate_uploads()
    if not HAS_APPTEST:
        return
    from app import chats

    _CHAT_DIR = Path(tempfile.mkdtemp(prefix="chat-model-"))
    patcher = mock.patch.object(chats, "CHAT_DIR", _CHAT_DIR)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    unittest.addModuleCleanup(shutil.rmtree, _CHAT_DIR, ignore_errors=True)


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit 的 testing 模块")
class TestPanelUI(unittest.TestCase):
    """面板本身。所有用例共用一个 AppTest 之外的干净配置目录。"""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="model-ui-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        file = self.dir / "model.json"

        for patcher in (
            mock.patch.object(config, "MODEL_FILE", file),
            mock.patch.object(
                config,
                "ENV_DEFAULTS",
                config.ModelConfig("https://default.test/v1", "default-model", "sk-default-key"),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

        config.reload()
        self.addCleanup(config.reload)
        self.file = file

    def open_panel(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        app.run()
        next(b for b in app.button if b.key == "model_toggle").click().run()
        return app

    def test_panel_hidden_by_default(self):
        """默认收起：没有输入框，页面上只有一个切换按钮。"""
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        app.run()

        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertEqual(len(app.text_input), 0)
        self.assertEqual(len([b for b in app.button if b.key == "model_toggle"]), 1)

    def test_panel_shows_three_inputs_and_buttons(self):
        app = self.open_panel()

        self.assertEqual(len(app.exception), 0, list(app.exception))
        labels = [t.label for t in app.text_input]
        self.assertEqual(labels, ["Base URL", "模型名", "API Key"])

        # Key 是密码框：屏幕上不能明文显示。
        # AppTest 的 TextInput 只暴露 value，类型得看 proto（PASSWORD 在枚举里是 1）。
        from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto

        self.assertEqual(app.text_input[2].proto.type, TextInputProto.PASSWORD)

        buttons = [b.label for b in app.button]
        self.assertIn("测试连接", buttons)
        self.assertIn("保存并应用", buttons)

    def test_inputs_prefilled_from_active_config(self):
        app = self.open_panel()
        self.assertEqual(app.text_input[0].value, "https://default.test/v1")
        self.assertEqual(app.text_input[1].value, "default-model")

    def test_collapsed_button_shows_active_model(self):
        """收起时按钮上就写着当前模型名，不用展开去确认到底生效的是哪套。"""
        config.set_current(model="gpt-4o-mini")
        config.save()

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        app.run()
        toggle = next(b for b in app.button if b.key == "model_toggle")
        self.assertIn("gpt-4o-mini", toggle.label)

    def test_applied_config_survives_new_session(self):
        """点应用 → 关掉页面重开，三个框还是那套，状态行也还是那个模型。

        这条盯的就是"填过的别丢"：只测 config.save() 存在不够，界面那一层
        可能根本没把值写进去，或者新会话又从别处盖回默认。
        """
        app = self.open_panel()
        app.text_input[0].set_value("https://saved.test/v1").run()
        app.text_input[1].set_value("saved-model").run()
        app.text_input[2].set_value("sk-saved-key").run()
        next(b for b in app.button if b.label == "保存并应用").click().run()

        config.reload()  # 相当于重启进程：内存里那份丢掉，从磁盘读
        again = self.open_panel()

        self.assertEqual(
            [t.value for t in again.text_input],
            ["https://saved.test/v1", "saved-model", "sk-saved-key"],
        )
        self.assertTrue(any("saved-model" in c.value for c in again.caption))

    def test_apply_switches_model_and_saves(self):
        app = self.open_panel()
        app.text_input[0].set_value("https://api.openai.com/v1").run()
        app.text_input[1].set_value("gpt-4o-mini").run()
        app.text_input[2].set_value("sk-new-key").run()

        next(b for b in app.button if b.label == "保存并应用").click().run()

        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertEqual(config.current().model, "gpt-4o-mini")
        self.assertEqual(config.current().base_url, "https://api.openai.com/v1")
        self.assertTrue(self.file.exists(), "应用之后要落盘，否则重启就丢")

    def test_apply_rejects_bad_url(self):
        app = self.open_panel()
        app.text_input[0].set_value("api.openai.com/v1").run()
        next(b for b in app.button if b.label == "保存并应用").click().run()

        self.assertEqual(config.current().base_url, "https://default.test/v1")  # 没被改掉
        self.assertTrue(any("http" in e.value.lower() for e in app.error), [e.value for e in app.error])

    def test_test_button_shows_success(self):
        ok = llm.ProbeResult(True, "连接正常", latency=0.12, model="default-model")
        with mock.patch.object(
            __import__("app.ui_streamlit", fromlist=["x"]), "llm_probe", return_value=ok
        ):
            app = self.open_panel()
            next(b for b in app.button if b.label == "测试连接").click().run()

        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertTrue(any("连接正常" in s.value for s in app.success), [s.value for s in app.success])

    def test_test_button_shows_failure(self):
        bad = llm.ProbeResult(False, "HTTP 401：Key 无效或没有权限。")
        with mock.patch.object(
            __import__("app.ui_streamlit", fromlist=["x"]), "llm_probe", return_value=bad
        ):
            app = self.open_panel()
            next(b for b in app.button if b.label == "测试连接").click().run()

        self.assertTrue(any("401" in e.value for e in app.error), [e.value for e in app.error])

    def test_test_button_does_not_save(self):
        """只测不用：测通了也不该悄悄换掉正在用的配置。"""
        ok = llm.ProbeResult(True, "连接正常", latency=0.1, model="m")
        with mock.patch.object(
            __import__("app.ui_streamlit", fromlist=["x"]), "llm_probe", return_value=ok
        ):
            app = self.open_panel()
            app.text_input[1].set_value("another-model").run()
            next(b for b in app.button if b.label == "测试连接").click().run()

        self.assertEqual(config.current().model, "default-model")
        self.assertFalse(self.file.exists())

    def test_panel_has_no_developer_copy(self):
        """面板上只说"填什么"，不解释怎么实现。

        这些词出现在界面上，说明文案是写给自己看的：用户不关心配置存哪个文件、
        也不关心服务端返回 401 还是 404。
        """
        app = self.open_panel()
        texts = [c.value for c in app.caption]
        texts += [m.value for m in app.markdown]
        texts += [b.label for b in app.button]
        texts += [getattr(t.proto, "help", "") for t in app.text_input]

        for word in ("改代码", "重启", "gitignore", "data/model.json", ".env", "兼容", "400", "404"):
            self.assertFalse(any(word in t for t in texts), (word, [t for t in texts if word in t]))

    def test_reset_restores_default(self):
        """恢复默认要连输入框一起退回去，但列表里存的那条得留着。"""
        config.set_current(model="temp-model")
        config.save()
        self.assertTrue(self.file.exists())

        app = self.open_panel()
        next(b for b in app.button if b.label == "恢复默认配置").click().run()

        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertEqual(config.current().model, "default-model")
        self.assertEqual([p.name for p in config.saved()], ["temp-model"])
        self.assertEqual(app.text_input[1].value, "default-model")

    def test_saved_list_shows_every_config(self):
        """列表里每条都在：当前那条是「▸ 名字」，其余是可点的按钮。"""
        for name in ("m-a", "m-b"):
            config.set_current(model=name, api_key="k-" + name)
            config.save()

        app = self.open_panel()

        self.assertIn("m-a", [b.label for b in app.button])
        self.assertTrue(any("m-b" in c.value for c in app.caption), [c.value for c in app.caption])

    def test_clicking_saved_item_switches(self):
        """点一行就切过去，三个输入框跟着变成那套——不用重填。"""
        config.set_current(base_url="https://a.test/v1", model="m-a", api_key="k-a")
        config.save()
        config.set_current(base_url="https://b.test/v1", model="m-b", api_key="k-b")
        config.save()

        app = self.open_panel()
        next(b for b in app.button if b.label == "m-a").click().run()

        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertEqual(config.current().name, "m-a")
        self.assertEqual(
            [t.value for t in app.text_input],
            ["https://a.test/v1", "m-a", "k-a"],
        )

    def test_delete_button_removes_only_that_config(self):
        for name in ("m-a", "m-b"):
            config.set_current(model=name)
            config.save()

        app = self.open_panel()
        next(b for b in app.button if b.key == "remove_model_0").click().run()

        self.assertFalse(any(b.label == "m-a" for b in app.button))
        self.assertEqual([p.name for p in config.saved()], ["m-b"])
