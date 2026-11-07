"""入口：网页界面。

    streamlit run app.py

为什么要多这一层壳：streamlit run 会把脚本所在目录塞进 sys.path，
把入口放在项目根，`from app...` 才能稳定导入；真正的界面代码放 app/ 里，
这样它仍然是一个可以 import 的模块，而不是一个只能被 streamlit 执行的脚本。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui_streamlit import main  # noqa: E402

main()
