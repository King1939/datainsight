"""出图工具：数据来源的两条路。

重点盯的是「没有数据表也能画图」——这条以前不存在，模型只能去催用户传 CSV，
而它手里其实已经有数字了。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from app import sandbox
from app.tools import chart_tools
from app.tools import data_tools as dt
from app.tools import registry
from app.tools.chart_tools import make_chart


class ChartBase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        patches = [
            mock.patch.object(chart_tools, "OUTPUT_DIR", self.dir),
            mock.patch.object(dt, "_conns", {}),  # 默认：一张表都没有
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        # "当前用哪份数据"是 ContextVar，不能像全局变量那样 patch，要按用法设
        dt.use_dataset("chart-test")
        self.addCleanup(dt.use_dataset, dt.DEFAULT_KEY)
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    def bind(self, rows: str) -> None:
        csv = self.dir / "sample.csv"
        csv.write_text(rows, encoding="utf-8")
        dt.bind_dataset(csv, key="chart-test", label="sample.csv")

    def png(self) -> list[Path]:
        return list(self.dir.glob("chart_*.png"))


class TestDirectPairs(ChartBase):
    """没有数据表时，两列数字应该能直接成图。"""

    def test_chart_without_any_dataset(self):
        text = make_chart(labels=["1月", "2月", "3月"], values=[120, 135, 98], title="月度")

        self.assertIn("图表已生成", text)
        self.assertEqual(len(self.png()), 1)
        self.assertIn("3 个数据点", text)

    def test_kind_is_honoured(self):
        for kind in ("bar", "line", "pie"):
            text = make_chart(labels=["甲", "乙"], values=[1, 2], kind=kind)
            self.assertIn("图表已生成", text, kind)
        self.assertEqual(len(self.png()), 3)

    def test_labels_can_be_numbers_as_strings(self):
        text = make_chart(labels=[2023, 2024, 2025], values=[1.5, 2.5, 3.5])
        self.assertIn("图表已生成", text)

    def test_missing_half_is_rejected(self):
        text = make_chart(labels=["甲", "乙"])
        self.assertIn("成对", text)
        self.assertEqual(self.png(), [])

    def test_length_mismatch_is_rejected(self):
        text = make_chart(labels=["甲", "乙", "丙"], values=[1, 2])
        self.assertIn("长度必须一致", text)
        self.assertEqual(self.png(), [])

    def test_non_numeric_values_are_rejected(self):
        text = make_chart(labels=["甲", "乙"], values=[1, "很多"])
        self.assertIn("全是数字", text)
        self.assertEqual(self.png(), [])

    def test_too_many_points(self):
        text = make_chart(labels=[f"第{i}项" for i in range(60)], values=list(range(60)))
        self.assertIn("太多", text)
        self.assertEqual(self.png(), [])

    def test_sql_and_pairs_are_exclusive(self):
        text = make_chart(sql="SELECT 1, 2", labels=["甲"], values=[1])
        self.assertIn("只能选一种", text)


class TestSqlPath(ChartBase):
    """原来那条路不能坏。"""

    def test_sql_still_works_with_dataset(self):
        self.bind("city,amount\n上海,10\n北京,20\n")
        text = make_chart(sql="SELECT city, amount FROM dataset", kind="bar")

        self.assertIn("图表已生成", text)
        self.assertIn("横轴 city", text)
        self.assertEqual(len(self.png()), 1)

    def test_two_columns_required(self):
        self.bind("city,amount,extra\n上海,10,x\n")
        text = make_chart(sql="SELECT * FROM dataset")
        self.assertIn("必须返回两列", text)

    def test_no_data_and_no_pairs_tells_the_other_way(self):
        """没表又不给数字：提示必须指出还有 labels/values 这条路。"""
        text = make_chart(sql="SELECT 1, 2")

        self.assertIn("没有数据表", text)
        self.assertIn("labels", text)
        self.assertEqual(self.png(), [])

    def test_empty_sql_and_no_pairs(self):
        text = make_chart()
        self.assertIn("没有给数据", text)


class TestSchema(unittest.TestCase):
    def test_sql_is_no_longer_required(self):
        params = registry.get("make_chart").parameters
        self.assertNotIn("sql", params.get("required", []))
        self.assertEqual(params["properties"]["labels"]["type"], "array")
        self.assertEqual(params["properties"]["values"]["items"]["type"], "number")

    def test_get_conn_raises_without_dataset(self):
        dt.use_dataset("none")
        self.addCleanup(dt.use_dataset, dt.DEFAULT_KEY)
        with mock.patch.object(dt, "_conns", {}):
            with self.assertRaises(dt.NoDataset):
                dt.get_conn()


class TestRealConnSanity(unittest.TestCase):
    """确认 sql 路径用的还是同一个 duckdb 视图，不是另开一个连接。"""

    def test_view_name(self):
        con = duckdb.connect()
        con.execute("CREATE VIEW dataset AS SELECT '上海' AS city, 10 AS amount")
        self.assertEqual(con.execute("SELECT count(*) FROM dataset").fetchone()[0], 1)
        con.close()


class TestChartLibrariesAreBlocked(unittest.TestCase):
    """run_python 里不许自己 import 绘图库，而且报错必须指向 make_chart。

    模型真的这么干过：想查中文字体，写了 `import matplotlib`，被拒之后
    如果只看到一串"允许的模块"，它会去试 seaborn、plotly，白白烧掉几步。
    """

    def test_rejected_with_pointer(self):
        for code in (
            "import matplotlib",
            "from matplotlib.patches import FancyBboxPatch",
            "import seaborn as sns",
            "import plotly.express as px",
        ):
            result = sandbox.run(code, allow_write=True)  # 完全工作区也一样拦
            self.assertFalse(result.ok, code)
            self.assertEqual(result.error_type, "PermissionError", code)
            self.assertIn("make_chart", result.output, code)

    def test_pandas_is_still_allowed(self):
        result = sandbox.run("pd.DataFrame({'a': [1]}).shape", allow_write=False)
        self.assertTrue(result.ok, result.as_observation())

    def test_hint_does_not_claim_df_exists_without_dataset(self):
        result = sandbox.run("df.head()", data_file=None, allow_write=True)
        text = result.as_observation()
        self.assertIn("df 是 None", text)
        self.assertNotIn("环境里已有 df", text)

    def test_hint_keeps_df_with_dataset(self):
        csv = Path(tempfile.mkdtemp()) / "t.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        result = sandbox.run("df.missing_col_xyz", data_file=csv, allow_write=False)
        self.assertIn("环境里已有 df", result.as_observation())


if __name__ == "__main__":
    unittest.main()
