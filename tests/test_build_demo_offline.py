"""Прогон build_demo.py и analyze_point.py на подставных данных (без сети)."""
import importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np, rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
bd, ap = load("build_demo"), load("analyze_point")
import tools.osm_extra as ox
from tools.osm import Grid

BBOX = (43.40, 39.90, 43.46, 39.98)
g = Grid(*BBOX, res=10.0)
ll = lambda x, y: {"lat": float(g.to_latlon(x, y)[0]), "lon": float(g.to_latlon(x, y)[1])}
ring = lambda x0, x1, y0, y1: [ll(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]

BUILD = {"elements": [
    {"type": "way", "id": 1, "tags": {"building": "hotel", "building:levels": "8"}, "geometry": ring(3000, 3030, 3000, 3030)},
    {"type": "way", "id": 2, "tags": {"building": "yes", "height": "22"}, "geometry": ring(2800, 2840, 2900, 3100)}]}
c = ll(3015, 3015)
HOT = {"elements": [{"type": "way", "id": 1, "center": c, "tags": {"tourism": "hotel", "name": "Тестовый отель"}}]}
street = list(range(101, 117))  # улица y=3000, x = 1600..3100 через 100 м
WALK = {"elements": [{"type": "node", "id": i, **ll(1600 + 100 * k, 3000)} for k, i in enumerate(street)]
        + [{"type": "way", "id": 50, "nodes": street, "tags": {"highway": "residential"}}]}
NOISE = {"elements": [{"type": "way", "id": 60, "tags": {"highway": "secondary", "name": "ул. Курортная"},
                       "geometry": [ll(1500, 2700), ll(4000, 2700)]}]}
POI = {"elements": [
    {"type": "node", "id": 70, "tags": {"amenity": "pharmacy", "opening_hours": "24/7", "name": "Аптека"}, **ll(3100, 3000)},
    {"type": "node", "id": 71, "tags": {"highway": "bus_stop"}, **ll(2900, 3010)},
    {"type": "node", "id": 72, "tags": {"leisure": "playground"}, **ll(2800, 3010)}]}
BEACH = {"elements": [{"type": "way", "id": 80, "geometry": [ll(1550, 3000), ll(1550, 3050), ll(1500, 3050), ll(1550, 3000)]}]}

def fake_fetch(query, cache_dir):
    if 'way["building"]' in query: return BUILD
    if "tourism" in query: return HOT
    if "natural" in query and "beach" in query: return BEACH
    if "aeroway" in query: return NOISE
    if "pharmacy" in query: return POI
    if 'highway"!~' in query or "highway\"]" in query: return WALK
    raise AssertionError("неожиданный запрос: " + query[:120])
bd.fetch_overpass = fake_fetch
ox.fetch_overpass = fake_fetch

KEYS = {"sunset_view", "sea_view", "quiet", "walk_access", "night_amenities", "kids_infrastructure"}
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    step, west, north = 0.0003, 39.88, 43.48
    nxp, nyp = int(0.12 / step), int(0.10 / step)
    arr = np.full((nyp, nxp), 5.0, np.float32)
    arr[:, : int((g.to_latlon(1500, 0)[1] - west) / step)] = 0.0
    tif = td / "dem.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=nyp, width=nxp, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, step, step)) as ds:
        ds.write(arr, 1)

    # --- build_demo
    argv = ["build_demo", "--bbox", *map(str, BBOX), "--dem", str(tif), "--limit", "5", "--out", str(td / "demo")]
    sys.argv = argv; bd.main()
    recs = json.loads((td / "demo" / "hotels.json").read_text(encoding="utf-8"))
    assert len(recs) == 1 and recs[0]["levels"] == 8
    f = recs[0]["features"]
    assert set(f) == KEYS, set(f)
    for k, v in f.items():
        assert {"value", "confidence", "meta", "source", "fetched_at", "ttl_days"} <= set(v), k
        assert 0 <= v["confidence"] <= 1 and v["meta"]["summary"], k
    beach = f["walk_access"]["value"]["beach"]
    assert beach and 1300 < beach["distance_m"] < 1550 and 15 < beach["minutes"] < 21, beach
    assert f["walk_access"]["value"]["bus_stop"]["distance_m"] < 250
    assert f["night_amenities"]["value"]["pharmacy"] is not None
    assert f["kids_infrastructure"]["value"]["playgrounds_within_300m"] == 1
    q = f["quiet"]["value"]
    assert max(v["night"] for d in q.values() for v in d.values()) > 40
    print("build_demo:", {k: v["meta"]["summary"][:60] for k, v in f.items()})

    sys.argv = argv; bd.main()  # повтор: сцена из кэша
    sys.argv = ["build_demo", "--bbox", *map(str, BBOX), "--dem", str(tif), "--no-extra", "--out", str(td / "demo2")]
    bd.main()
    assert set(json.loads((td / "demo2" / "hotels.json").read_text(encoding="utf-8"))[0]["features"]) == {"sunset_view", "sea_view"}

    # --- analyze_point: тот же отель, но «по координатам», окно вокруг точки
    out = td / "point.json"
    sys.argv = ["analyze_point", "--lat", str(c["lat"]), "--lon", str(c["lon"]), "--name", "Точка", "--radius", "3000",
                "--dem", str(tif), "--cache", str(td / "cache"), "--out", str(out)]
    ap.main()
    r = json.loads(out.read_text(encoding="utf-8"))
    assert set(r["features"]) == KEYS and r["levels"] == 8
    b2 = r["features"]["walk_access"]["value"]["beach"]
    assert b2 and 1300 < b2["distance_m"] < 1600, b2
print("OK")
