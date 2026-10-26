"""让 pytest 无需安装即可从 src 导入 checker 包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
