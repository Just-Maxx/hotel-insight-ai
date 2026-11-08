"""Общий конвейер: слои данных загружаются один раз, характеристики считаются для любого отеля.

Используется и scripts/build_demo.py (много отелей региона), и scripts/analyze_point.py
(один отель по координатам в любой точке мира).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools.geo.noise import NoiseModel, noise_feature
from tools.geo.poi import (build_target_sets, kids_feature, night_amenities_feature, walk_access_feature)
from tools.geo.visibility import Scene, sea_view_feature, sunset_feature
from tools.geo.walking import Field, WalkGraph
from tools.osm import Grid, facades_from_polygon, footprint_mask
from tools import osm_extra as ox

SOURCES = {
    "sunset_view": "Copernicus GLO-30 DSM + OSM здания, расчёт по лучам обзора",
    "sea_view": "Copernicus GLO-30 DSM + OSM здания, расчёт по лучам обзора",
    "quiet": "OSM дороги/ж/д/ВПП/заведения + здания и рельеф, инженерная модель шума (не откалибрована)",
    "walk_access": "OSM пешая сеть + Copernicus DEM, функция ходьбы Тоблера",
    "night_amenities": "OSM opening_hours + пешая сеть",
    "kids_infrastructure": "OSM leisure=playground, детские медучреждения, highway=pedestrian + пешая сеть",
}
TTL_DAYS = {"sunset_view": 365, "sea_view": 365, "quiet": 180, "walk_access": 90,
            "night_amenities": 60, "kids_infrastructure": 90}


@dataclass
class ExtraLayers:
    noise: NoiseModel
    walk: WalkGraph
    fields: dict[str, Field | None]
    pois: list[dict]
    night_stats: dict
    counts: dict = field(default_factory=dict)
    incomplete: list = field(default_factory=list)


def load_extra_layers(scene: Scene, grid: Grid, cache_dir: Path, tile_deg: float = 0.025, report=None) -> ExtraLayers:
    """Скачивает (с кэшем) и подготавливает слои: источники шума, пешая сеть, POI, пляжи."""
    bbox = grid.bbox
    def fetch(query):
        return ox.fetch_tiled(query, bbox, cache_dir, tile_deg,
                              on_missing=report.callback(query) if report else None)
    noise_src = ox.parse_noise_sources(fetch(ox.noise_query), grid)
    node_xy, ways = ox.parse_walk(fetch(ox.walk_query), grid)
    pois = ox.parse_pois(fetch(ox.poi_query), grid)
    beach = ox.parse_beach_points(fetch(ox.beach_query), grid)

    walk = WalkGraph(node_xy, ways, scene)
    sets, stats = build_target_sets(pois, beach)
    fields = {k: walk.field(k, v) for k, v in sets.items()}
    counts = {"noise_lines": len(noise_src["lines"]), "noise_points": len(noise_src["points"]),
              "walk_nodes": walk.N, "pois": len(pois), "beach_points": int(len(beach))}
    return ExtraLayers(NoiseModel(noise_src), walk, fields, pois, stats, counts,
                       report.data["missing"].copy() if report else [])


def compute_hotel_features(scene: Scene, grid: Grid, hotel: dict, layers: ExtraLayers | None,
                           now: str | None = None) -> dict:
    """Все характеристики отеля в формате {value, confidence, meta, source, fetched_at, ttl_days}."""
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    facades = facades_from_polygon(hotel["poly"])
    floors = list(range(1, hotel["levels"] + 1))
    excl = footprint_mask(grid, hotel["poly"])
    common = {"own_height_source": hotel["height_source"], "levels": hotel["levels"],
              "footprint": hotel["footprint_source"]}

    feats = {
        "sunset_view": sunset_feature(scene, facades, floors, hotel["lat"], exclude=excl),
        "sea_view": sea_view_feature(scene, facades, floors, exclude=excl),
    }
    if layers is not None:
        x, y = hotel["x"], hotel["y"]
        feats["quiet"] = noise_feature(layers.noise, scene, facades, floors, exclude=excl)
        feats["walk_access"] = walk_access_feature(layers.walk, layers.fields, x, y)
        feats["night_amenities"] = night_amenities_feature(layers.walk, layers.fields, layers.pois,
                                                          layers.night_stats, x, y)
        feats["kids_infrastructure"] = kids_feature(layers.walk, layers.fields, layers.pois, x, y)

    dependencies = {"quiet": {"noise_query"}, "walk_access": {"walk_query", "poi_query", "beach_query"},
                    "night_amenities": {"walk_query", "poi_query"},
                    "kids_infrastructure": {"walk_query", "poi_query"}}
    for name, f in feats.items():
        missing = [m for m in (layers.incomplete if layers else [])
                   if m['layer'] in dependencies.get(name, set())]
        if missing:
            mark_incomplete(f, missing)
        f["meta"].update(common)
        f.update({"source": SOURCES[name], "fetched_at": now, "ttl_days": TTL_DAYS[name]})
    return feats


def mark_incomplete(feature, missing):
    """Неполное покрытие — неизвестный результат, а не плохая оценка отеля."""
    feature['value'] = None
    feature['confidence'] = 0.0
    feature['meta'].update(incomplete=True, missing_tiles=missing,
                           summary='Недостаточно геоданных: часть плиток не загружена')
