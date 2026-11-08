"""Дополнительные Overpass-запросы и разбор: источники шума, пешеходная сеть, точки интереса.

Использует Grid и fetch_overpass из tools/osm.py. Все координаты — локальные метры Grid.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tools.osm import Grid, fetch_adaptive, fetch_overpass

ROAD_CLASSES = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"
WALK_EXCLUDE = "motorway|motorway_link|construction|proposed|raceway|bus_guideway|escape|elevator|corridor|platform"
KIDS_NAME_RE = re.compile(r"детск|дет\.|педиатр|children|paediatric|pediatric", re.I)


def _b(bbox) -> str:
    s, w, n, e = bbox
    return f"({s},{w},{n},{e})"


def noise_query(bbox) -> str:
    b = _b(bbox)
    return ("[out:json][timeout:240];("
            f'way["highway"~"^({ROAD_CLASSES})$"]{b};'
            f'way["railway"~"^(rail|light_rail|tram)$"]{b};'
            f'way["aeroway"="runway"]{b};'
            f'nwr["amenity"~"^(nightclub|bar|pub)$"]{b};'
            ");out tags geom qt;")


def walk_query(bbox) -> str:
    b = _b(bbox)
    return ("[out:json][timeout:240];("
            f'way["highway"]["highway"!~"^({WALK_EXCLUDE})$"]["foot"!~"^no$"]["access"!~"^(private|no)$"]{b};'
            ");(._;>;);out body qt;")


def poi_query(bbox) -> str:
    b = _b(bbox)
    return ("[out:json][timeout:240];("
            f'nwr["amenity"~"^(pharmacy|cafe|restaurant|fast_food|hospital|clinic|doctors)$"]{b};'
            f'nwr["shop"~"^(convenience|supermarket)$"]{b};'
            f'nwr["leisure"="playground"]{b};'
            f'node["highway"="bus_stop"]{b};'
            f'nwr["railway"~"^(station|halt)$"]{b};'
            ");out center tags qt;")


def beach_query(bbox) -> str:
    return f'[out:json][timeout:240];(nwr["natural"="beach"]{_b(bbox)};);out tags geom qt;'


def split_bbox(bbox, step: float = 0.05):
    s, w, n, e = bbox
    la = s
    while la < n - 1e-9:
        lo = w
        while lo < e - 1e-9:
            yield (round(la, 5), round(lo, 5), round(min(la + step, n), 5), round(min(lo + step, e), 5))
            lo += step
        la += step


def fetch_tiled(make_query, bbox, cache_dir: Path, step: float = 0.025, on_missing=None) -> list[dict]:
    """Overpass по плиткам, элементы без дублей (type, id)."""
    seen, out = set(), []
    for sub in split_bbox(bbox, step):
        for el in fetch_adaptive(make_query, sub, cache_dir, fetch=fetch_overpass, on_missing=on_missing):
            k = (el["type"], el["id"])
            if k not in seen:
                seen.add(k)
                out.append(el)
    return out


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------
def _geom_xy(grid: Grid, geom: list[dict]) -> np.ndarray:
    a = np.array([(g["lat"], g["lon"]) for g in geom], dtype=float)
    x, y = grid.to_local(a[:, 0], a[:, 1])
    return np.column_stack([x, y])


def parse_noise_sources(elements: list[dict], grid: Grid) -> dict:
    """{'lines': [{cls,name,pts}], 'points': [{cls,name,xy}]}; тоннели и служебные проезды пропускаются."""
    lines, points = [], []
    for el in elements:
        t = el.get("tags", {})
        name = t.get("name") or t.get("ref") or ""
        if t.get("tunnel") in ("yes", "building_passage", "culvert"):
            continue
        if "amenity" in t:
            if el["type"] == "node":
                xy = np.array(grid.to_local(el["lat"], el["lon"]), dtype=float)
            elif el.get("geometry"):
                xy = _geom_xy(grid, el["geometry"]).mean(axis=0)
            else:
                continue
            points.append({"cls": "amenity:" + t["amenity"], "name": name, "xy": xy})
        elif el["type"] == "way" and el.get("geometry") and len(el["geometry"]) >= 2:
            if "highway" in t:
                if t.get("service") in ("parking_aisle", "driveway", "drive-through"):
                    continue
                cls = "highway:" + t["highway"]
            elif "railway" in t:
                cls = "railway:" + t["railway"]
            elif "aeroway" in t:
                cls = "aeroway:runway"
            else:
                continue
            lines.append({"cls": cls, "name": name, "pts": _geom_xy(grid, el["geometry"])})
    return {"lines": lines, "points": points}


def parse_walk(elements: list[dict], grid: Grid) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    """(node_id -> (x, y), ways=[{nodes, hw, name}]) из ответа `out body; >; out skel`."""
    nodes = [el for el in elements if el["type"] == "node"]
    if not nodes:
        return {}, []
    x, y = grid.to_local(np.array([n["lat"] for n in nodes]), np.array([n["lon"] for n in nodes]))
    node_xy = {n["id"]: (float(xx), float(yy)) for n, xx, yy in zip(nodes, x, y)}
    ways = [{"nodes": el["nodes"], "hw": el["tags"].get("highway", ""), "name": el["tags"].get("name", "")}
            for el in elements if el["type"] == "way" and el.get("tags", {}).get("highway")]
    return node_xy, ways


def classify_poi(tags: dict) -> str | None:
    am = tags.get("amenity")
    if am == "pharmacy":
        return "pharmacy"
    if tags.get("shop") in ("convenience", "supermarket"):
        return "grocery"
    if am in ("cafe", "restaurant", "fast_food"):
        return "food"
    if tags.get("leisure") == "playground":
        return "playground"
    if tags.get("highway") == "bus_stop":
        return "bus_stop"
    if tags.get("railway") in ("station", "halt"):
        return "station"
    if am in ("hospital", "clinic", "doctors"):
        blob = " ".join(str(tags.get(k, "")) for k in ("name", "name:ru", "name:en", "healthcare:speciality"))
        if KIDS_NAME_RE.search(blob):
            return "kids_medical"
    return None


def parse_pois(elements: list[dict], grid: Grid) -> list[dict]:
    out = []
    for el in elements:
        t = el.get("tags", {})
        kind = classify_poi(t)
        if not kind:
            continue
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None:
            continue
        x, y = grid.to_local(lat, lon)
        out.append({"kind": kind, "name": t.get("name", ""), "x": float(x), "y": float(y),
                    "opening_hours": t.get("opening_hours")})
    return out


def parse_beach_points(elements: list[dict], grid: Grid) -> np.ndarray:
    """Все вершины контуров пляжей (цели для пешего маршрута), форма (n, 2)."""
    pts = []
    for el in elements:
        geoms = [el["geometry"]] if el.get("geometry") else [m["geometry"] for m in el.get("members", []) if m.get("geometry")]
        pts += [_geom_xy(grid, g) for g in geoms if g]
    return np.vstack(pts) if pts else np.zeros((0, 2))
