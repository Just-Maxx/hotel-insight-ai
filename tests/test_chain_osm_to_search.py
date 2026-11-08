"""Сквозная проверка без сети и без Gemini:
Overpass-JSON -> parse_buildings/parse_hotels -> Scene -> visibility -> hotels.json -> БД -> поиск.

Морская характеристика здесь считается через sea_view_fraction: настоящий sea_view_feature
из конвейера характеристик в этом тесте не используется.
"""
import json, os, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HOTELS_DB"] = str(Path(tempfile.mkdtemp()) / "t.db")

from agents.orchestrator import Plan, Requirement
from agents.search_agent import rank_hotels
from scripts.import_hotels import import_records
from tools.geo.visibility import Scene, sea_view_fraction, sunset_feature
from tools.osm import (Grid, facades_from_polygon, footprint_mask, parse_buildings, parse_hotels,
                       rasterize_buildings)

grid = Grid(43.40, 39.90, 43.46, 39.98, res=10.0)
LAT = 43.43


def ring(x0, x1, y0, y1):
    out = []
    for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]:
        la, lo = grid.to_latlon(x, y)
        out.append({"lat": float(la), "lon": float(lo)})
    return out


def hotel_el(i, x, y, name):
    la, lo = grid.to_latlon(x + 15, y + 15)
    return {"type": "way", "id": i, "center": {"lat": float(la), "lon": float(lo)},
            "tags": {"tourism": "hotel", "name": name}}


# Отель A (id 1): открытый вид на запад. Отель B (id 3): тот же, но в 60 м на запад стоит 25-м дом.
els = [
    {"type": "way", "id": 1, "tags": {"building": "hotel", "building:levels": "8"}, "geometry": ring(3000, 3030, 3000, 3030)},
    {"type": "way", "id": 3, "tags": {"building": "hotel", "building:levels": "8"}, "geometry": ring(3000, 3030, 5000, 5030)},
    {"type": "way", "id": 4, "tags": {"building": "yes", "height": "25"}, "geometry": ring(2930, 2965, 4900, 5150)},
]
bdata = {"elements": els}
hdata = {"elements": [hotel_el(1, 3000, 3000, "Открытый вид"), hotel_el(3, 3000, 5000, "За домом")]}

blds = parse_buildings(bdata, grid)
hotels = parse_hotels(hdata, grid, blds)
assert len(hotels) == 2 and all(h["footprint_source"] == "osm_way" for h in hotels)

dem = np.full((grid.ny, grid.nx), 5.0)
water = np.zeros((grid.ny, grid.nx), bool)
water[:, : int(1500 / grid.res)] = True
dem[water] = 0.0
scene = Scene(dem, rasterize_buildings(grid, blds), grid.res, 0.0, 0.0, water)

records = []
for h in hotels:
    fac = facades_from_polygon(h["poly"])
    floors = list(range(1, h["levels"] + 1))
    ex = footprint_mask(grid, h["poly"])
    sun = sunset_feature(scene, fac, floors, LAT, exclude=ex)
    west = next(f for f in fac if f.name == "запад")
    sea = {west.name: {fl: round(sea_view_fraction(scene, west, fl, exclude=ex), 2) for fl in (1, 4, 8)}}
    records.append({"hotel_id": f'w{h["hotel_id"][1:]}', "name": h["name"], "lat": h["lat"], "lon": h["lon"],
                    "levels": h["levels"], "features": {
        "sunset_view": {**sun, "source": "test", "fetched_at": "2026-10-27T00:00:00+00:00", "ttl_days": 365},
        "sea_view": {"value": sea, "confidence": 0.8, "meta": {"summary": f"запад: {sea['запад']}"},
                     "source": "test", "fetched_at": "2026-10-27T00:00:00+00:00", "ttl_days": 365}}})
    print(h["name"], "|", sun["meta"]["summary"][:70], "| море", sea["запад"])

st = import_records(records, "Сочи")
assert st["hotels"] == 2 and st["facts"] == 4 and not st["no_score"], st

plan = Plan(intent="search_hotels", location="Сочи", requirements=[
    Requirement(feature="sunset_view", detail="закат", importance="must"),
    Requirement(feature="sea_view", detail="море", importance="nice")])
res = rank_hotels(plan)
names = [r["hotel"] for r in res["ranked"]]
print("ранжирование:", [(r["hotel"], r["score"]) for r in res["ranked"]])
assert names[0] == "Открытый вид", names
print("OK: цепочка OSM -> visibility -> БД -> поиск работает")
