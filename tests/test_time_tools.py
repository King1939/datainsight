"""本地时钟工具单元测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.agent import make_system_prompt
from app.query_rewrite import RewriteContext, build_rewrite_user_prompt
from app.tools import registry
from app.tools.time_tools import _month_bounds


class TestTimeTools(unittest.TestCase):
    def test_registered_in_global_registry(self):
        self.assertIn("get_current_time", registry.names())
        tool = registry.get("get_current_time")
        self.assertIsNotNone(tool)
        self.assertIn("offset_days", tool.parameters["properties"])
        self.assertEqual(tool.parameters.get("required", []), [])

    def test_returns_current_time_and_relative_ranges(self):
        fixed = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone(timedelta(hours=8)))
        with patch("app.tools.time_tools.datetime") as mock_dt:
            mock_dt.now.return_value.astimezone.return_value = fixed
            text = registry.call("get_current_time", {})

        self.assertIn("2026-03-15 14:30:00", text)
        self.assertIn("星期日", text)
        self.assertIn("UTC+08:00", text)
        self.assertIn("今天：2026-03-15", text)
        self.assertIn("昨天：2026-03-14", text)
        self.assertIn("近 7 天：2026-03-09 ~ 2026-03-15", text)
        self.assertIn("近 30 天：2026-02-14 ~ 2026-03-15", text)
        self.assertIn("本月至今：2026-03-01 ~ 2026-03-15", text)
        self.assertIn("上个月：2026-02-01 ~ 2026-02-28", text)
        self.assertIn("今年至今：2026-01-01 ~ 2026-03-15", text)

    def test_offset_days_and_january_prev_month(self):
        fixed = datetime(2026, 1, 5, 9, 0, 0, tzinfo=timezone(timedelta(hours=8)))
        self.assertEqual(_month_bounds(fixed), ("2025-12-01", "2025-12-31"))

        with patch("app.tools.time_tools.datetime") as mock_dt:
            mock_dt.now.return_value.astimezone.return_value = fixed
            text = registry.call("get_current_time", {"offset_days": -7})

        self.assertIn("偏移 -7 天日期：2025-12-29", text)
        self.assertIn("上个月：2025-12-01 ~ 2025-12-31", text)

    def test_prompts_mention_clock_and_current_time(self):
        sys_prompt = make_system_prompt(web_search=True)
        self.assertIn("get_current_time", sys_prompt)

        rw_prompt = build_rewrite_user_prompt("最近销量如何", RewriteContext(web_search=True))
        self.assertIn("当前本地时间：", rw_prompt)


if __name__ == "__main__":
    unittest.main()
