import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
(ROOT / "results").mkdir(exist_ok=True)

runpy.run_path(str(ROOT / "src" / "diagnosis.py"), run_name="__main__")
