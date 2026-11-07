"""测试用的目录隔离：把会被写的 data/uploads 挪到临时目录。

为什么需要：文件级用例会调 `dt.bind_dataset(DataFrame, key="ut_a")`，它按 key
写一份 csv 到 `data/uploads/<key>` 供沙箱子进程读取；界面用例更狠——AppTest 会把
`app.py` 整个跑一遍，而入口里有"清掉上个进程遗留"的扫描。

不隔开的话，跑一次测试就往用户真实的上传目录里塞十几个 csv，甚至把正在用的
临时文件删掉。（2026-11-05 的真实事故：367 个 34 字节的 csv，全是测试留下的。）

用法（模块级，一次就够）：

    from _isolation import isolate_uploads

    def setUpModule() -> None:
        isolate_uploads()

没写 setup 的模块不受影响——只有真的会落盘的那几个需要。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import uploads


def isolate_uploads(prefix: str = "uploads-test-") -> Path:
    """把 uploads.UPLOAD_DIR 指到临时目录，模块跑完自动还原并清理。

    必须在 `setUpModule()` 里调：`unittest.addModuleCleanup` 只认模块级作用域。
    """
    root = Path(tempfile.mkdtemp(prefix=prefix))
    patcher = mock.patch.object(uploads, "UPLOAD_DIR", root)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    unittest.addModuleCleanup(shutil.rmtree, root, ignore_errors=True)
    return root
