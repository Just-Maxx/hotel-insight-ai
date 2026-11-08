"""Проверка пешей доступности, часов работы и парсеров на синтетических данных."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.geo.poi import build_target_sets, is_open_night, kids_feature, night_amenities_feature, walk_access_feature
from tools.geo.visibility import Scene
from tools.geo.walking import WalkGraph, tobler_time
from tools.osm import Grid
from tools.osm_extra import (classify_poi, parse_beach_points, parse_noise_sources, parse_pois, parse_walk)

# --- Тоблер: ровно ~5 км/ч, подъём 10% заметно медленнее, лёгкий спуск быстрее
t = lambda s: float(tobler_time(np.array([100.0]), np.array([100.0 * s]))[0])
assert 70 < t(0.0) < 73 and t(0.10) > 1.35 * t(0.0) and t(-0.05) < t(0.0)

# --- сетка улиц 5x5, шаг 100 м; ровно
def grid_net(n=5, hw="residential"):
    ids = {(i, j): 1 + i * n + j for i in range(n) for j in range(n)}
    node_xy = {k: (i * 100.0, j * 100.0) for (i, j), k in ids.items()}
    ways = [{"nodes": [ids[(i, j)] for j in range(n)], "hw": hw, "name": ""} for i in range(n)]
    ways += [{"nodes": [ids[(i, j)] for i in range(n)], "hw": hw, "name": ""} for j in range(n)]
    return node_xy, ways, ids

N, RES = 100, 10.0
flat = Scene(np.zeros((N, N)), np.zeros((N, N)), RES, 0.0, 0.0)
node_xy, ways, ids = grid_net()
wg = WalkGraph(node_xy, ways, flat)
f = wg.field("t", np.array([[400.0, 400.0]]))
d = wg.describe(f, 0.0, 0.0)
print("ровная сеть, 800 м:", d)
assert d["distance_m"] == 800 and 9.3 < d["minutes"] < 9.8 and d["busy_crossings"] == 0 and d["ascent_m"] == 0

# --- подъём: тот же маршрут в гору (10 м на 100 м по x) дольше и считает набор высоты
slope = Scene(np.tile(np.arange(N) * 1.0, (N, 1)), np.zeros((N, N)), RES, 0.0, 0.0)  # +1 м на 10 м = 10%
wg2 = WalkGraph(node_xy, ways, slope)
up = wg2.describe(wg2.field("t", np.array([[400.0, 0.0]])), 0.0, 0.0)
down = wg2.describe(wg2.field("t", np.array([[0.0, 0.0]])), 400.0, 0.0)
print("в гору %.1f мин (+%.0f м), под гору %.1f мин" % (up["minutes"], up["ascent_m"], down["minutes"]))
assert up["ascent_m"] > 35 and up["minutes"] > 1.3 * down["minutes"]

# --- пересечение крупной дороги: S(0,0) -> C(0,100) -> T(0,200); C на primary
nx = {1: (0, 0), 2: (0, 100), 3: (0, 200), 4: (-100, 100), 5: (100, 100)}
cross = [{"nodes": [1, 2, 3], "hw": "footway", "name": ""}, {"nodes": [4, 2, 5], "hw": "primary", "name": ""}]
wg3 = WalkGraph(nx, cross, flat)
c = wg3.describe(wg3.field("t", np.array([[0.0, 200.0]])), 0.0, 0.0)
assert c["busy_crossings"] == 1, c
along = wg3.describe(wg3.field("t", np.array([[100.0, 100.0]])), -100.0, 100.0)  # идём вдоль дороги — не пересечение
assert along["busy_crossings"] == 0, along

# --- недостижимо и пустая сеть
assert wg.describe(wg.field("t", np.array([[5000.0, 5000.0]])), 0.0, 0.0) is None
assert WalkGraph({}, [], flat).describe(None, 0, 0) is None

# --- часы работы
cases = {"24/7": True, "Mo-Su 00:00-24:00": True, "08:00-23:00": False, "22:00-06:00": True,
         "Mo-Su 08:00-02:30": True, "10:00-24:00": False, "Mo-Fr 09:00-18:00": None, "": None, None: None,
         "08:00-12:00, 22:00-03:00": True, "off": None, "Mo-Su 09:00-21:00; PH off": None}
for oh, exp in cases.items():
    assert is_open_night(oh) is exp, (oh, is_open_night(oh), exp)

# --- парсеры Overpass
g = Grid(43.40, 39.90, 43.46, 39.98, res=10.0)
ll = lambda x, y: dict(zip(("lat", "lon"), map(float, g.to_latlon(x, y))))
walk_el = [{"type": "node", "id": i, **ll(*xy)} for i, xy in nx.items()] + \
          [{"type": "way", "id": 9, "nodes": [1, 2, 3], "tags": {"highway": "footway"}}]
nxy, ws = parse_walk(walk_el, g)
assert len(nxy) == 5 and len(ws) == 1 and abs(nxy[2][1] - 100) < 0.5

noise_el = [
    {"type": "way", "id": 1, "tags": {"highway": "secondary", "name": "А"}, "geometry": [ll(0, 0), ll(500, 0)]},
    {"type": "way", "id": 2, "tags": {"highway": "service", "service": "parking_aisle"}, "geometry": [ll(0, 0), ll(50, 0)]},
    {"type": "way", "id": 3, "tags": {"highway": "primary", "tunnel": "yes"}, "geometry": [ll(0, 0), ll(50, 0)]},
    {"type": "node", "id": 4, "tags": {"amenity": "nightclub", "name": "К"}, **ll(100, 100)},
]
ns = parse_noise_sources(noise_el, g)
assert len(ns["lines"]) == 1 and ns["lines"][0]["cls"] == "highway:secondary" and len(ns["points"]) == 1

assert classify_poi({"amenity": "pharmacy"}) == "pharmacy" and classify_poi({"shop": "supermarket"}) == "grocery"
assert classify_poi({"amenity": "hospital", "name": "Детская больница"}) == "kids_medical"
assert classify_poi({"amenity": "hospital", "name": "Городская больница"}) is None
poi_el = [{"type": "node", "id": 1, "tags": {"amenity": "pharmacy", "opening_hours": "24/7", "name": "Ап"}, **ll(100, 0)},
          {"type": "way", "id": 2, "center": ll(200, 100), "tags": {"leisure": "playground"}},
          {"type": "node", "id": 3, "tags": {"shop": "convenience"}, **ll(0, 200)},          # без часов -> неизвестно
          {"type": "node", "id": 4, "tags": {"highway": "bus_stop"}, **ll(0, 100)}]
pois = parse_pois(poi_el, g)
assert [p["kind"] for p in pois] == ["pharmacy", "playground", "grocery", "bus_stop"]
beach = parse_beach_points([{"type": "way", "id": 5, "geometry": [ll(300, 300), ll(310, 300), ll(310, 310), ll(300, 300)]}], g)
assert beach.shape == (4, 2)

# --- сборка характеристик на сети из тестовой сетки (узлы на целых сотнях метров)
sets, stats = build_target_sets(pois, np.array([[400.0, 400.0]]))
assert len(sets["pharmacy_night"]) == 1 and len(sets["grocery_night"]) == 0 and stats["grocery"]["unknown"] == 1
fields = {k: wg.field(k, v) for k, v in sets.items()}
acc = walk_access_feature(wg, fields, 0.0, 0.0)
print(acc["meta"]["summary"], acc["confidence"])
assert acc["value"]["beach"]["distance_m"] == 800 and acc["value"]["bus_stop"]["distance_m"] == 100
night = night_amenities_feature(wg, fields, pois, stats, 0.0, 0.0)
print(night["meta"]["summary"], night["value"]["unknown_hours_share_1km"], night["confidence"])
assert night["value"]["pharmacy"]["distance_m"] == 100 and night["value"]["grocery"] is None
kids = kids_feature(wg, fields, pois, 0.0, 0.0)
print(kids["meta"]["summary"])
assert kids["value"]["playgrounds_within_300m"] == 1 and kids["value"]["kids_medical"] is None
print("OK")
