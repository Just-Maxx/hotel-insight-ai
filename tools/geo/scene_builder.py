"""Сборка Scene из реальных данных: Copernicus DEM + здания OSM.

ВАЖНО: Copernicus GLO-30 — это DSM (цифровая модель поверхности): в ней уже
есть крыши зданий и кроны деревьев. Если просто сложить её с высотами зданий
из OSM, здания посчитаются дважды. Поэтому внутри контуров зданий мы заменяем
DSM на оценку «голой земли» (локальный минимум в окне), а высоту дома добавляем
из OSM. Деревья остаются в DSM как часть рельефа (для обзора это скорее плюс).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from tools.osm import Building, Grid, footprint_mask, rasterize_buildings

from .visibility import EARTH_R, Scene

# GLO-30: файлы вида Copernicus_DSM_COG_10_N43_00_E039_00_DEM (публичный бакет AWS, без ключей)
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def grid_latlon(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Широта/долгота центров пикселей сетки, формы (ny, nx)."""
    xs = (np.arange(grid.nx) + 0.0) * grid.res
    ys = (np.arange(grid.ny) + 0.0) * grid.res
    lat, _ = grid.to_latlon(np.zeros_like(ys), ys)
    _, lon = grid.to_latlon(xs, np.zeros_like(xs))
    return np.meshgrid(lat, lon, indexing="ij")


# --------------------------------------------------------------------------
# DEM
# --------------------------------------------------------------------------
def dem_tile_names(south: float, west: float, north: float, east: float) -> list[str]:
    names = []
    for la in range(math.floor(south), math.floor(north) + 1):
        for lo in range(math.floor(west), math.floor(east) + 1):
            ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            names.append(f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM")
    return names


def sample_dem(grid: Grid, sources: list[str]) -> np.ndarray:
    """Билинейная выборка DEM на сетку Grid. sources — пути/URL GeoTIFF.

    Нет пикселя (тайл отсутствует, как для открытого океана) -> 0 м.
    """
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.windows import Window, from_bounds

    lat, lon = grid_latlon(grid)
    out = np.full(lat.shape, np.nan, dtype=np.float32)
    for src in sources:
        try:
            ds = rasterio.open(src)
        except RasterioIOError:
            continue  # тайла нет: океан или закрытая страна
        with ds:
            win = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), ds.transform)
            c0, r0 = math.floor(win.col_off) - 2, math.floor(win.row_off) - 2
            c1, r1 = math.ceil(win.col_off + win.width) + 2, math.ceil(win.row_off + win.height) + 2
            c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, ds.width), min(r1, ds.height)
            if c1 <= c0 or r1 <= r0:
                continue
            win = Window(c0, r0, c1 - c0, r1 - r0)
            arr = ds.read(1, window=win).astype(np.float32)
            t = ds.window_transform(win)
            cols = (lon - t.c) / t.a - 0.5          # индексы центров пикселей
            rows = (lat - t.f) / t.e - 0.5
            inside = (cols >= -0.5) & (cols <= arr.shape[1] - 0.5) & (rows >= -0.5) & (rows <= arr.shape[0] - 0.5)
            vals = ndimage.map_coordinates(arr, [rows, cols], order=1, mode="nearest")
            out[inside] = vals[inside]
    return np.nan_to_num(out, nan=0.0)


# --------------------------------------------------------------------------
# Растеризация, вода, «голая земля»
# --------------------------------------------------------------------------
def water_mask(dem: np.ndarray, res: float, min_area_m2: float = 500_000.0, level: float = 0.5) -> np.ndarray:
    """Море/крупные водоёмы: пиксели у нуля, связные области площадью >= min_area_m2."""
    low = dem <= level
    lab, n = ndimage.label(low)
    if n == 0:
        return low
    sizes = np.bincount(lab.ravel())
    keep = sizes * res * res >= min_area_m2
    keep[0] = False
    return keep[lab]


def bare_earth(dsm: np.ndarray, footprint: np.ndarray, res: float, window_m: float = 150.0) -> np.ndarray:
    """Внутри контуров зданий заменяет DSM локальным минимумом (оценка земли под домом)."""
    k = max(3, int(window_m / res) | 1)
    return np.where(footprint, ndimage.minimum_filter(dsm, size=k, mode="nearest"), dsm)


def build_scene(grid: Grid, dem_sources: list[str], buildings: list[Building]) -> Scene:
    dsm = sample_dem(grid, dem_sources)
    if not np.any(dsm > 0.5):
        raise RuntimeError("DEM пуст: тайл недоступен или область целиком над водой. Для Армении и Азербайджана "
                           "тайлы Copernicus GLO-30 закрыты; передайте свой GeoTIFF через --dem.")
    heights = rasterize_buildings(grid, buildings).astype(np.float32)
    footprint = heights > 0
    water = water_mask(dsm, grid.res)
    ground = bare_earth(dsm, footprint & ~water, grid.res)
    heights[water] = 0.0
    return Scene(ground, heights, grid.res, 0.0, 0.0, water)  # начало координат Grid — юго-западный угол


# --------------------------------------------------------------------------
# Кэш сцены (чтобы демо запускалось без скачивания)
# --------------------------------------------------------------------------
def save_scene(path: str | Path, scene: Scene, grid: Grid) -> None:
    np.savez_compressed(path, dem=scene.dem.astype(np.float32), buildings=scene.buildings.astype(np.float32),
                        water=scene.water, grid=np.array([grid.south, grid.west, grid.north, grid.east, grid.res]))


def load_scene(path: str | Path) -> tuple[Scene, Grid]:
    z = np.load(path)
    south, west, north, east, res = (float(v) for v in z["grid"])
    return Scene(z["dem"], z["buildings"], res, 0.0, 0.0, z["water"]), Grid(south, west, north, east, res)
