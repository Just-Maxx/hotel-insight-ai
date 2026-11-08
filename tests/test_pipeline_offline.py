"""Офлайн-проверка пути: Overpass-JSON + GeoTIFF -> Scene -> характеристики.

Геометрия та же, что в синтетической демо (море на западе, дом-заслон), но
задана в lat/lon и проходит через проекцию, растеризацию и чтение GeoTIFF.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.geo.scene_builder import build_scene, dem_tile_names, load_scene, save_scene
from tools.geo.visibility import sea_view_feature, sunset_feature
from tools.osm import (Grid, building_height, facades_from_polygon, footprint_mask, hotels_query,
                       parse_buildings, parse_hotels)

LAT = 43.43
grid = Grid(43.40, 39.90, 43.46, 39.98, res=10.0)


def ring(x0, x1, y0, y1):  # локальные метры -> замкнутый контур lat/lon как в Overpass
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    out = []
    for x, y in pts:
        lat, lon = grid.to_latlon(x, y)
        out.append({"lat": float(lat), "lon": float(lon)})
    return out


# Отель 30x30 м (8 этажей) на x=3000, дом-заслон 22 м на x=2800 (запад), море при x < 1500
overpass_buildings = {"elements": [
    {"type": "way", "id": 1, "tags": {"building": "hotel", "building:levels": "8"}, "geometry": ring(3000, 3030, 3000, 3030)},
    {"type": "way", "id": 2, "tags": {"building": "yes", "height": "22 m"}, "geometry": ring(2800, 2840, 2900, 3100)},
]}
overpass_hotels = {"elements": [
    {"type": "way", "id": 1, "center": {"lat": float(grid.to_latlon(3015, 3015)[0]), "lon": float(grid.to_latlon(3015, 3015)[1])},
     "tags": {"tourism": "hotel", "name": "Тестовый отель"}},
]}

# проверки разбора
assert building_height({"height": "22 m"})[:2] == (22.0, "height")
assert building_height({"building:levels": "8"})[:2] == (24.0, "levels")
assert building_height({})[1] == "default"
assert "nwr" in hotels_query(grid.bbox)

blds = parse_buildings(overpass_buildings, grid)
hotels = parse_hotels(overpass_hotels, grid, blds)
assert len(blds) == 2 and len(hotels) == 1
h = hotels[0]
assert h["footprint_source"] == "osm_way" and h["levels"] == 8, h

fac = facades_from_polygon(h["poly"])
az = {f.name: round(f.azimuth) for f in fac}
assert az == {"север": 0, "восток": 90, "юг": 180, "запад": 270}, az  # нормали наружу, обе ориентации контура
fac_cw = facades_from_polygon(h["poly"][::-1])
assert {f.name: round(f.azimuth) for f in fac_cw} == az

# GeoTIFF вместо Copernicus: плоская суша 5 м, море (0 м) западнее x=1500; растр север-вверх
with tempfile.TemporaryDirectory() as td:
    tif = Path(td) / "dem.tif"
    step = 0.0003  # ~ 24 м по долготе
    west, north = 39.88, 43.48
    nxp, nyp = int(0.12 / step), int(0.10 / step)
    arr = np.full((nyp, nxp), 5.0, dtype=np.float32)
    lon_sea = grid.to_latlon(1500, 0)[1]
    arr[:, : int((lon_sea - west) / step)] = 0.0
    with rasterio.open(tif, "w", driver="GTiff", height=nyp, width=nxp, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, step, step)) as ds:
        ds.write(arr, 1)

    scene = build_scene(grid, [str(tif)], blds)
    save_scene(Path(td) / "s.npz", scene, grid)
    scene, grid2 = load_scene(Path(td) / "s.npz")
    assert grid2.bbox == grid.bbox

assert scene.water[:, :140].all() and not scene.water[:, 160:].any(), "море только на западе"
iy, ix = int(3015 / 10), int(3015 / 10)
assert scene.buildings[iy, ix] == 24.0 and abs(scene.dem[iy, ix] - 5.0) < 1e-3, "высота дома, земля под ним 5 м"
assert abs(scene.dem[iy, 100] - 0.0) < 1e-3

excl = footprint_mask(grid, h["poly"])
sun = sunset_feature(scene, fac, list(range(1, h["levels"] + 1)), LAT, exclude=excl)
w = sun["value"]["запад"]
print("закат:", sun["meta"]["summary"], "| уверенность", sun["confidence"])
assert w[6] is not None and w[12] is not None and w[6] < w[12]
assert all(v is None for v in sun["value"]["восток"].values())

sea = sea_view_feature(scene, fac, list(range(1, h["levels"] + 1)), exclude=excl)
print("море:", sea["meta"]["summary"], "| уверенность", sea["confidence"])
assert sea["value"]["запад"]["8"] > 0.1 and sea["value"]["восток"]["8"] == 0.0

assert dem_tile_names(43.36, 39.84, 43.50, 39.995) == ["Copernicus_DSM_COG_10_N43_00_E039_00_DEM"]
print("OK")
