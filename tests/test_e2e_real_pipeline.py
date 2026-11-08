"""Полная цепочка без сети: подставной Overpass + GeoTIFF -> build_demo.py -> hotels.json
-> import_hotels.py -> facts.db -> поиск по ВСЕМ шести характеристикам."""
import os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = Path(tempfile.mkdtemp())
os.environ["HOTELS_DB"] = str(WORK / "facts.db")


class _Fixed:  # подмена TemporaryDirectory: результат используется следующим этапом сквозного теста
    def __enter__(self): return str(WORK / "run")
    def __exit__(self, *a): return False


src = (ROOT / "tests" / "test_build_demo_offline.py").read_text(encoding="utf-8")
src = src.replace("tempfile.TemporaryDirectory()", "_Fixed()")
(WORK / "run").mkdir()
exec(compile(src, str(ROOT / "tests" / "test_build_demo_offline.py"), "exec"),
     {"__file__": str(ROOT / "tests" / "test_build_demo_offline.py"), "_Fixed": _Fixed, "__name__": "x"})

import json
from agents.orchestrator import Plan, Requirement
from agents.search_agent import rank_hotels
from scripts.import_hotels import import_records

recs = json.loads((WORK / "run" / "demo" / "hotels.json").read_text(encoding="utf-8"))
st = import_records(recs, "Сочи")
print("\nИмпорт:", st)
assert st["hotels"] == 1 and st["facts"] == 6 and not st["no_score"], st

plan = Plan(intent="search_hotels", location="Сочи", requirements=[
    Requirement(feature=f, detail=f, importance="nice")
    for f in ("sunset_view", "sea_view", "quiet", "walk_access", "night_amenities", "kids_infrastructure")])
r = rank_hotels(plan)["ranked"][0]
print(r["hotel"], "итог", r["score"])
for m in r["matched"] + r["misses"]:
    print(f"  {m['feature']:<20} score={m['score']:.2f} conf={m['confidence']:.2f}  {m['summary'][:70]}")
assert not r["gaps"], r["gaps"]
print("OK: все 6 характеристик реального конвейера доходят до поиска")
