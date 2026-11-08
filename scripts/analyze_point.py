"""Характеристики одного отеля по координатам — для любой точки мира.

    python scripts/analyze_point.py --lat 43.4036 --lon 39.9557 --name "Мой отель" --levels 12

Вокруг точки берётся окно радиуса --radius (по умолчанию 3 км), для него скачиваются (с кэшем)
здания, дороги, пешая сеть, POI и DEM Copernicus. Первый запуск в новом городе — минуты (в основном
Overpass), повторный запуск рядом идёт из кэша. Результат печатается и пишется в JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.geo.scene_builder import DEM_URL, build_scene, dem_tile_names, load_scene, save_scene
from tools.osm import Grid, buildings_query, parse_buildings, parse_hotels
from tools.osm_extra import fetch_tiled
from tools.pipeline import compute_hotel_features, load_extra_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", default="Отель")
    ap.add_argument("--radius", type=float, default=3000.0, help="радиус окна анализа, м")
    ap.add_argument("--levels", type=int, help="число этажей, если в OSM нет (иначе оценка по высоте здания)")
    ap.add_argument("--res", type=float, default=10.0)
    ap.add_argument("--dem", nargs="*", help="свои GeoTIFF вместо Copernicus")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--out", help="куда записать JSON (по умолчанию печать краткой сводки)")
    a = ap.parse_args()

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    dlat = math.degrees(a.radius / 6_371_000.0)
    dlon = dlat / math.cos(math.radians(a.lat))
    bbox = tuple(round(v, 5) for v in (a.lat - dlat, a.lon - dlon, a.lat + dlat, a.lon + dlon))
    grid = Grid(*bbox, res=a.res)

    print(f"Окно анализа {bbox}, сетка {grid.nx}x{grid.ny}")
    bdata = {"elements": fetch_tiled(buildings_query, bbox, cache)}
    buildings = parse_buildings(bdata, grid)

    key = hashlib.sha1(f"{bbox}{a.res}{a.dem}".encode()).hexdigest()[:12]
    scene_file = cache / f"scene_{key}.npz"
    if scene_file.exists():
        scene, grid = load_scene(scene_file)
    else:
        sources = a.dem or [DEM_URL.format(name=n) for n in dem_tile_names(*bbox)]
        scene = build_scene(grid, sources, buildings)
        save_scene(scene_file, scene, grid)
    print(f"Зданий в OSM: {len(buildings)}, вода: {scene.water.mean():.0%} окна")

    fake = {"elements": [{"type": "node", "id": 0, "lat": a.lat, "lon": a.lon,
                          "tags": {"tourism": "hotel", "name": a.name}}]}
    hotel = parse_hotels(fake, grid, buildings)[0]
    if a.levels:
        hotel["levels"], hotel["height_source"] = a.levels, "user"
    print(f"Контур отеля: {hotel['footprint_source']}, этажей: {hotel['levels']} ({hotel['height_source']})")
    if hotel["footprint_source"] == "synthetic":
        print("  ! здания в точке нет в OSM: контур условный 20x20 м, окна и фасады приблизительные")

    layers = load_extra_layers(scene, grid, cache)
    print("Слои:", layers.counts)
    feats = compute_hotel_features(scene, grid, hotel, layers)

    print()
    for name, f in feats.items():
        print(f"{name} (уверенность {f['confidence']}): {f['meta']['summary']}")
    if a.out:
        rec = {"name": a.name, "lat": a.lat, "lon": a.lon, "levels": hotel["levels"], "features": feats}
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nJSON:", a.out)


if __name__ == "__main__":
    main()
