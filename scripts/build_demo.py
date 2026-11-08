"""Сборка демо-данных для Сочи (Адлер): сцена + характеристики отелей.

Запуск (нужна сеть при первом прогоне; Overpass-ответы кэшируются в data/demo/raw,
собранная сцена — в data/demo/scene.npz, повторный запуск работает без загрузки DEM):

    pip install numpy scipy pillow requests rasterio
    python scripts/build_demo.py --limit 30

Результат: data/demo/hotels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.geo.scene_builder import DEM_URL, build_scene, dem_tile_names, load_scene, save_scene
from tools.osm import (Grid, buildings_query, fetch_adaptive, fetch_overpass, hotels_query, parse_buildings,
                       parse_hotels)
from tools.pipeline import compute_hotel_features, load_extra_layers, mark_incomplete
from tools.download_report import DownloadReport

# south, west, north, east — Адлер, Олимпийский парк, Имеретинская низина; восточная граница < 40.0
ADLER_BBOX = (43.36, 39.84, 43.50, 39.995)


def split_bbox(bbox, step=0.02):
    s, w, n, e = bbox
    la = s
    while la < n - 1e-9:
        lo = w
        while lo < e - 1e-9:
            yield (round(la, 5), round(lo, 5), round(min(la + step, n), 5), round(min(lo + step, e), 5))
            lo += step
        la += step


def merged_elements(make_query, bbox, cache: Path, tiled: bool, step: float = 0.02, on_missing=None) -> dict:
    """Overpass по плиткам (на весь bbox запрос здания может не уложиться в таймаут), без дублей."""
    seen, elements = set(), []
    tiles = list(split_bbox(bbox, step)) if tiled else [bbox]
    for i, sub in enumerate(tiles, 1):
        print(f"    [плитка {i}/{len(tiles)}] {sub}", flush=True)
        for el in fetch_adaptive(make_query, sub, cache, fetch=fetch_overpass, on_missing=on_missing):
            k = (el["type"], el["id"])
            if k not in seen:
                seen.add(k)
                elements.append(el)
    return {"elements": elements}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, default=ADLER_BBOX, metavar=("S", "W", "N", "E"))
    ap.add_argument("--res", type=float, default=10.0, help="шаг сетки, м")
    ap.add_argument("--limit", type=int, default=30, help="сколько отелей считать")
    ap.add_argument("--dem", nargs="*", help="свои GeoTIFF вместо скачивания Copernicus")
    ap.add_argument("--tile", type=float, default=0.02, help="размер плитки Overpass, градусы (меньше — надёжнее)")
    ap.add_argument("--no-extra", action="store_true", help="только закат и море, без шума/пеших маршрутов/POI")
    ap.add_argument("--rebuild", action="store_true", help="пересобрать сцену, даже если есть scene.npz")
    ap.add_argument("--out", default=str(ROOT / "data" / "demo"))
    a = ap.parse_args()
    if not (0.0001 <= a.tile <= 1) or not np.isfinite(a.res) or a.res <= 0 or a.limit <= 0:
        ap.error("--tile должен быть от 0.0001 до 1, --res и --limit — положительными")
    south, west, north, east = a.bbox
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        ap.error("--bbox: нужны S < N и W < E в допустимых широтах/долготах")

    out = Path(a.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    bbox = tuple(a.bbox)

    report = DownloadReport(out / "download_report.json", bbox)

    print("1/5 здания и отели (Overpass)...")
    bdata = merged_elements(buildings_query, bbox, raw, tiled=True, step=a.tile, on_missing=report.callback(buildings_query))
    hdata = merged_elements(hotels_query, bbox, raw, tiled=False, on_missing=report.callback(hotels_query))

    scene_path = out / "scene.npz"
    cached_scene = None
    if scene_path.exists() and not a.rebuild and not report.missing("buildings_query"):
        sc, gr = load_scene(scene_path)
        if np.allclose(gr.bbox, bbox, rtol=0, atol=1e-9) and abs(gr.res - a.res) < 1e-9 and not a.dem:
            cached_scene = (sc, gr)
    if cached_scene is not None:
        print("2/5 сцена из кэша", scene_path)
        scene, grid = cached_scene
    else:
        print("2/5 DEM + растеризация...")
        grid = Grid(*bbox, res=a.res)
        sources = a.dem or [DEM_URL.format(name=n) for n in dem_tile_names(*bbox)]
        buildings = parse_buildings(bdata, grid)
        scene = build_scene(grid, sources, buildings)
        if not report.missing("buildings_query"):
            save_scene(scene_path, scene, grid)
        print(f"    зданий: {len(buildings)}, сетка {grid.nx}x{grid.ny}, вода: {scene.water.mean():.0%}")

    buildings = parse_buildings(bdata, grid)
    hotels = parse_hotels(hdata, grid, buildings)
    # самые крупные отели с названием и контуром из OSM; синтетические квадраты не берём
    hotels = [h for h in hotels if h["footprint_source"] != "synthetic"]
    area = lambda h: abs(0.5 * np.sum(h["poly"][:, 0] * np.roll(h["poly"][:, 1], -1)
                                      - np.roll(h["poly"][:, 0], -1) * h["poly"][:, 1]))
    hotels = sorted(hotels, key=area, reverse=True)[: a.limit]
    layers = None
    if not a.no_extra:
        print("3/5 слои: шум, пешая сеть, POI, пляжи (Overpass, кэш в data/demo/raw)...")
        layers = load_extra_layers(scene, grid, raw, tile_deg=a.tile * 1.25, report=report)
        print("    ", layers.counts)
    print(f"4/5 считаю характеристики для {len(hotels)} отелей...")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for i, h in enumerate(hotels, 1):
        feats = compute_hotel_features(scene, grid, h, layers, now)
        if report.missing("buildings_query"):
            for name in ("sunset_view", "sea_view", "quiet"):
                if name in feats:
                    mark_incomplete(feats[name], report.missing("buildings_query"))
        if report.missing("hotels_query"):
            for feat in feats.values():
                mark_incomplete(feat, report.missing("hotels_query"))
        records.append({"hotel_id": h["hotel_id"], "name": h["name"], "lat": h["lat"], "lon": h["lon"],
                        "levels": h["levels"], "tags": h["tags"], "features": feats})
        print(f"  [{i}/{len(hotels)}] {h['name']}: " + " | ".join(
            f"{k}: {v['meta']['summary'][:45]}" for k, v in list(feats.items())[:3]))

    report.finish()
    if not records:
        raise RuntimeError("Отелей с контурами не найдено; прежний hotels.json сохранён")
    tmp = out / "hotels.json.tmp"
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out / "hotels.json")
    print("5/5 готово:", out / "hotels.json")
    if report.data['missing']:
        print(f"ВНИМАНИЕ: неполные данные, пропусков {len(report.data['missing'])}. "
              f"Отчёт: {report.path}. Повторите команду для докачки.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nСборка не завершена: {exc}\nКэш сохранён. Повторите ту же команду; импорт пока не запускайте.", file=sys.stderr)
        sys.exit(1)
