"""Демо-городок «Сочи (синтетика)»: 8 отелей разного характера, прогнанных через НАСТОЯЩИЙ геоконвейер.

Зачем: Overpass и Copernicus нужны сеть и несколько минут. Этот скрипт подставляет вместо них
искусственные слои (море на западе, холм на востоке, дороги, бары, ж/д, аптеки, площадки), а всё
остальное — Scene, лучи обзора, модель шума, пешая сеть Тоблера, POI — реальный код из tools/.
Результат: data/demo/hotels.json, scene.npz и (через import_hotels) facts.db.

ВАЖНО: это НЕ сведения о реальных отелях. Названия вымышлены, география искусственная.
Реальные данные — scripts/build_demo.py (нужна сеть); формат файлов идентичен.

    python scripts/build_demo_town.py            # пересобрать data/demo целиком
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.osm_extra as ox  # noqa: E402
from tools.osm import Grid  # noqa: E402

BBOX = (43.40, 39.90, 43.46, 39.98)
SEA_X = 1500.0          # море при x < 1500 м
HILL_START = 4500.0     # к востоку от этой линии рельеф поднимается (0.05 м на метр)

# id, название, x, y, этажей, [заслон: (x0, x1, y0, y1, высота)]
HOTELS = [
    (1, "Морской бриз",   1700, 3000, 6,  []),
    (2, "Гранд Каскад",   2650, 2500, 12, [(2300, 2340, 2300, 2700, 20)]),
    (3, "Тихий дворик",   3300, 2200, 4,  [(3100, 3140, 2100, 2300, 30)]),
    (4, "Панорама",       3000, 4200, 14, [(2800, 2830, 4100, 4300, 18)]),
    (5, "Ночной город",   2500, 3600, 5,  []),
    (6, "Семейный",       3600, 3300, 6,  [(3400, 3440, 3200, 3400, 16)]),
    (7, "У вокзала",      4100, 3000, 7,  []),
    (8, "Горный приют",   5000, 3200, 3,  []),
]
# Нежилые «городские» дома: плотная застройка 5 этажей между набережной и проспектом
FILLER = [(1900 + 140 * i, 1940 + 140 * i, 3700 + 160 * j, 3800 + 160 * j, 15)
          for i in range(3) for j in range(2)]


def build_layers(g: Grid):
    ll = lambda x, y: {"lat": float(g.to_latlon(x, y)[0]), "lon": float(g.to_latlon(x, y)[1])}
    ring = lambda x0, x1, y0, y1: [ll(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]

    buildings, hotels = [], []
    bid = 1000
    for hid, name, x, y, lv, blockers in HOTELS:
        buildings.append({"type": "way", "id": hid, "tags": {"building": "hotel", "building:levels": str(lv)},
                          "geometry": ring(x, x + 30, y, y + 30)})
        hotels.append({"type": "way", "id": hid, "center": ll(x + 15, y + 15),
                       "tags": {"tourism": "hotel", "name": name}})
        for x0, x1, y0, y1, h in blockers:
            bid += 1
            buildings.append({"type": "way", "id": bid, "tags": {"building": "yes", "height": str(h)},
                              "geometry": ring(x0, x1, y0, y1)})
    for x0, x1, y0, y1, h in FILLER:
        bid += 1
        buildings.append({"type": "way", "id": bid, "tags": {"building": "yes", "height": str(h)},
                          "geometry": ring(x0, x1, y0, y1)})

    # Пешая сеть: сетка улиц 300 м, узлы через 100 м
    xs, ys = list(range(1600, 5301, 100)), list(range(1500, 5001, 100))
    nid, node_id, elements = 10_000, {}, []
    for x in xs:
        for y in ys:
            nid += 1
            node_id[(x, y)] = nid
            elements.append({"type": "node", "id": nid, **ll(x, y)})
    wid = 20_000
    for y in ys:
        if (y - 1500) % 300 == 0 or y in (2200, 3000, 3300, 3600, 4200):
            wid += 1
            elements.append({"type": "way", "id": wid, "nodes": [node_id[(x, y)] for x in xs],
                             "tags": {"highway": "residential"}})
    for x in xs:
        if (x - 1600) % 300 == 0 or x in (1700, 2500, 2600, 3300, 3600, 4100):
            wid += 1
            tags = {"highway": "pedestrian", "name": "Набережная"} if x == 1600 else {"highway": "residential"}
            elements.append({"type": "way", "id": wid, "nodes": [node_id[(x, y)] for y in ys], "tags": tags})
    walk = {"elements": elements}

    noise = {"elements": [
        {"type": "way", "id": 60, "tags": {"highway": "primary", "name": "Курортный проспект"},
         "geometry": [ll(2400, 1500), ll(2400, 5000)]},
        {"type": "way", "id": 61, "tags": {"highway": "secondary", "name": "ул. Приморская"},
         "geometry": [ll(1600, 3500), ll(4000, 3500)]},
        {"type": "way", "id": 62, "tags": {"railway": "rail", "name": "Железная дорога"},
         "geometry": [ll(4200, 1500), ll(4200, 5000)]},
        {"type": "node", "id": 63, "tags": {"amenity": "nightclub", "name": "Клуб «Волна»"}, **ll(2450, 3650)},
        {"type": "node", "id": 64, "tags": {"amenity": "bar", "name": "Бар «Причал»"}, **ll(2600, 3550)},
        {"type": "node", "id": 65, "tags": {"amenity": "bar", "name": "Бар «Якорь»"}, **ll(2350, 3560)},
    ]}

    def poi(i, tags, x, y):
        return {"type": "node", "id": 70 + i, "tags": tags, **ll(x, y)}
    pois = {"elements": [
        poi(0, {"amenity": "pharmacy", "opening_hours": "24/7", "name": "Аптека 24"}, 2500, 3500),
        poi(1, {"amenity": "pharmacy", "opening_hours": "Mo-Su 09:00-21:00", "name": "Аптека у дома"}, 3600, 3400),
        poi(2, {"shop": "convenience", "opening_hours": "24/7", "name": "Магазин 24"}, 2600, 3600),
        poi(3, {"shop": "supermarket", "opening_hours": "08:00-23:00", "name": "Супермаркет"}, 3300, 3000),
        poi(4, {"amenity": "cafe", "opening_hours": "Mo-Su 00:00-24:00", "name": "Кафе «Ночник»"}, 2400, 3700),
        poi(5, {"amenity": "restaurant", "opening_hours": "10:00-23:00", "name": "Ресторан у моря"}, 1700, 3200),
        poi(6, {"highway": "bus_stop", "name": "Остановка «Курортная»"}, 2400, 3000),
        poi(7, {"highway": "bus_stop", "name": "Остановка «Центр»"}, 3300, 2500),
        poi(8, {"highway": "bus_stop", "name": "Остановка «Горная»"}, 4800, 3300),
        poi(9, {"railway": "station", "name": "Станция «Курорт»"}, 4200, 3100),
        poi(10, {"leisure": "playground", "name": "Детская площадка"}, 3600, 3400),
        poi(11, {"leisure": "playground", "name": "Площадка во дворе"}, 3300, 2300),
        poi(12, {"amenity": "clinic", "name": "Детская поликлиника №2"}, 3700, 3500),
        poi(13, {"shop": "convenience", "name": "Продукты", "opening_hours": "07:00-22:00"}, 1800, 3100),
        poi(14, {"shop": "convenience", "name": "Мини-маркет", "opening_hours": "24/7"}, 4150, 3050),
    ]}
    # как в реальном OSM, контур пляжа плотный: цели пешего маршрута — его вершины (здесь каждые 50 м)
    shore = [ll(1550, y) for y in range(2000, 4501, 50)]
    beach = {"elements": [{"type": "way", "id": 80, "geometry": shore + [ll(1500, 4500), ll(1500, 2000), shore[0]]}]}
    return {"buildings": {"elements": buildings}, "hotels": {"elements": hotels}, "walk": walk,
            "noise": noise, "poi": pois, "beach": beach}


def write_dem(path: Path, g: Grid) -> None:
    """GeoTIFF: суша 5 м, море (0 м) западнее SEA_X, холм к востоку от HILL_START."""
    step, west, north = 0.0003, 39.88, 43.48
    nxp, nyp = int(0.12 / step), int(0.10 / step)
    lon = west + step * (np.arange(nxp) + 0.5)
    x = np.asarray(g.to_local(g.south, lon)[0])
    row = 5.0 + np.clip(x - HILL_START, 0, None) * 0.05
    row[x < SEA_X] = 0.0
    arr = np.tile(row.astype(np.float32), (nyp, 1))
    with rasterio.open(path, "w", driver="GTiff", height=nyp, width=nxp, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, step, step)) as ds:
        ds.write(arr, 1)


def build(out: Path) -> Path:
    spec = importlib.util.spec_from_file_location("build_demo", ROOT / "scripts" / "build_demo.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)

    g = Grid(*BBOX, res=10.0)
    L = build_layers(g)

    def fake_fetch(query, cache_dir):
        if 'way["building"]' in query:
            return L["buildings"]
        if "tourism" in query:
            return L["hotels"]
        if "natural" in query and "beach" in query:
            return L["beach"]
        if "aeroway" in query:
            return L["noise"]
        if "pharmacy" in query:
            return L["poi"]
        if 'highway"!~' in query or 'highway"]' in query:
            return L["walk"]
        raise AssertionError("неожиданный запрос: " + query[:100])

    bd.fetch_overpass = fake_fetch
    ox.fetch_overpass = fake_fetch
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tif = Path(td) / "dem.tif"
        write_dem(tif, g)
        sys.argv = ["build_demo", "--bbox", *map(str, BBOX), "--dem", str(tif), "--limit", "20",
                    "--out", str(out), "--rebuild"]
        bd.main()
    # синтетика не должна выдаваться за реальные отели
    path = out / "hotels.json"
    recs = json.loads(path.read_text(encoding="utf-8"))
    for r in recs:
        r["city"] = "Сочи"
        for f in r["features"].values():
            f["source"] = "[синтетическая сцена] " + f["source"]
    path.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    for junk in ("raw", "download_report.json"):
        p = out / junk
        if p.is_dir():
            for f in p.rglob("*"):
                f.unlink() if f.is_file() else None
            p.rmdir()
        elif p.exists():
            p.unlink()
    return path


if __name__ == "__main__":
    print("->", build(ROOT / "data" / "demo"))
