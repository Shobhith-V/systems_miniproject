import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))   # lets src files import each other
(ROOT / "results").mkdir(exist_ok=True)

runpy.run_path(str(ROOT / "src" / "pca.py"), run_name="__main__")
