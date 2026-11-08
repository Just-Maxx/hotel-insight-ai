"""Загрузка и разбор OpenStreetMap: отели, здания, фасады.

Всё приводится к локальным метрам (см. Grid). Сеть нужна только в fetch_overpass;
ответы кэшируются в data/cache/, дальше всё считается офлайн.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.geo.visibility import EARTH_R, Facade

# Публичные серверы часто перегружены: список можно переопределить, OVERPASS_URLS="url1,url2"
OVERPASS_URLS = [u.strip() for u in os.environ.get("OVERPASS_URLS", "").split(",") if u.strip()] or [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",   # бывший overpass.kumi.systems
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
DEFAULT_HEIGHT_M = 9.0   # если в OSM нет ни height, ни building:levels
LEVEL_HEIGHT_M = 3.0
SECTORS = ["север", "северо-восток", "восток", "юго-восток", "юг", "юго-запад", "запад", "северо-запад"]
HOTEL_RE = "^(hotel|guest_house|hostel|motel|apartment|resort)$"


# --------------------------------------------------------------------------
# Сетка
# --------------------------------------------------------------------------
@dataclass
class Grid:
    south: float
    west: float
    north: float
    east: float
    res: float = 10.0

    def __post_init__(self):
        self.lat0, self.lon0 = self.south, self.west
        self.width_m = self.to_local(self.south, self.east)[0]
        self.height_m = self.to_local(self.north, self.west)[1]
        self.nx = int(math.ceil(self.width_m / self.res))
        self.ny = int(math.ceil(self.height_m / self.res))

    def to_local(self, lat, lon):
        x = np.radians(np.asarray(lon) - self.lon0) * EARTH_R * math.cos(math.radians(self.lat0))
        y = np.radians(np.asarray(lat) - self.lat0) * EARTH_R
        return x, y

    def to_latlon(self, x, y):
        lat = self.lat0 + np.degrees(np.asarray(y) / EARTH_R)
        lon = self.lon0 + np.degrees(np.asarray(x) / (EARTH_R * math.cos(math.radians(self.lat0))))
        return lat, lon

    @property
    def bbox(self):  # формат Overpass: south, west, north, east
        return (self.south, self.west, self.north, self.east)


# --------------------------------------------------------------------------
# Overpass
# --------------------------------------------------------------------------
class OverpassUnavailable(RuntimeError):
    """Нет доступного сервера: дробление плитки не исправит соединение."""


class OverpassQueryError(RuntimeError):
    """Сервер прервал вычисление: можно попробовать меньшую плитку."""


_preferred_url = None
_cooldown_until: dict[str, float] = {}
_failure_counts: dict[str, int] = {}


def _validate_response(data):
    if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
        raise ValueError("ответ не содержит список elements")
    if data.get("remark"):
        raise OverpassQueryError(str(data["remark"]))
    if any(not isinstance(e, dict) or "type" not in e or "id" not in e for e in data["elements"]):
        raise ValueError("некорректные элементы OSM")
    return data


def fetch_overpass(query: str, cache_dir: Path, timeout: int = 150) -> dict:
    import requests
    global _preferred_url

    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / (hashlib.sha1(query.encode()).hexdigest()[:16] + ".json")
    if f.exists():
        try:
            data = _validate_response(json.loads(f.read_text(encoding="utf-8")))
            print(f"    [кэш] {f.name}: {len(data['elements'])} объектов", flush=True)
            return data
        except (ValueError, OverpassQueryError) as exc:
            print(f"    [кэш] {f.name} повреждён или неполон: {exc}; загружаю снова", flush=True)

    # Сохраняем старый ключ кэша; сокращаем заявленный бюджет только при отправке.
    wire_query = re.sub(r"\[timeout:\d+\]", "[timeout:45]", query)
    last = None
    query_error = None
    for attempt in range(4):
        now = time.monotonic()
        ready = [u for u in OVERPASS_URLS if _cooldown_until.get(u, 0) <= now]
        if ready:
            url = _preferred_url if _preferred_url in ready else ready[0]
        else:
            url = min(OVERPASS_URLS, key=lambda u: _cooldown_until.get(u, 0))
            pause = max(0, _cooldown_until.get(url, 0) - now)
            print(f"    [Overpass] ожидание доступного зеркала: {pause:.0f} с", flush=True)
            while pause > 0:
                chunk = min(pause, 30)
                time.sleep(chunk)
                pause -= chunk
        t0 = time.monotonic()
        print(f"    [Overpass] {f.name}: попытка {attempt + 1}/4 к {url.split('/')[2]} ...", flush=True)
        r = None
        try:
            r = requests.post(url, data={"data": wire_query}, timeout=(10, timeout),
                              headers={"User-Agent": "HotelInsightAI/1.0 (+https://github.com/Just-Maxx/hotel-insight-ai)",
                                       "Accept": "application/json"})
            # Ошибочный запрос повторять на других зеркалах бессмысленно.
            if r.status_code == 400:
                raise ValueError(f"Overpass отклонил запрос (400): {r.text[:300]}")
            r.raise_for_status()
            data = _validate_response(r.json())
        except ValueError as exc:
            if r is not None and r.status_code == 400:
                raise
            last = exc
            _cooldown_until[url] = time.monotonic() + 60
        except (requests.RequestException, OverpassQueryError) as exc:
            last = exc
            if isinstance(exc, OverpassQueryError):
                query_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (502, 504):
                query_error = OverpassQueryError(f"HTTP {status}: сервер/шлюз не обработал плитку")
            _failure_counts[url] = _failure_counts.get(url, 0) + 1
            # Разрыв соединения чаще связан с маршрутом/зеркалом: не трогаем его 15 минут.
            pause = 900 if isinstance(exc, requests.ConnectionError) else 30
            if status in (403, 404, 406):
                pause = 900
            elif status in (502, 503, 504):
                pause = min(120, 15 * _failure_counts[url])
            retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After", "")
            if str(retry_after).isdigit():
                pause = max(pause, min(int(retry_after), 300))
            _cooldown_until[url] = time.monotonic() + pause
        else:
            _preferred_url = url
            _cooldown_until.pop(url, None)
            _failure_counts.pop(url, None)
            tmp = f.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(f)
            print(f"    [Overpass] готово за {time.monotonic() - t0:.0f} с, "
                  f"{len(r.content) / 1024:.1f} КБ, {len(data['elements'])} объектов", flush=True)
            return data
        print(f"    [Overpass] ошибка за {time.monotonic() - t0:.0f} с: "
              f"{type(last).__name__}: {str(last)[:160]}; пробую другое зеркало", flush=True)
    if query_error is not None:
        raise query_error
    raise OverpassUnavailable(f"Overpass недоступен: {last}. Кэш сохранён; повторите запуск позже.")


def fetch_adaptive(make_query, bbox, cache_dir: Path, fetch=None, min_deg: float = 0.005, on_missing=None) -> list[dict]:
    """Элементы Overpass для bbox. Если плитка не скачивается (таймаут, перегрузка) —
    делим её на 4 части и пробуем каждую, пока плитка не станет совсем мелкой."""
    fetch = fetch or fetch_overpass
    try:
        return fetch(make_query(bbox), cache_dir).get("elements", [])
    except RuntimeError as exc:
        s_, w_, n_, e_ = bbox
        if isinstance(exc, OverpassUnavailable) or max(n_ - s_, e_ - w_) <= 2 * min_deg + 1e-9:
            if on_missing is None:
                raise
            on_missing(bbox, exc)
            print(f"    [пропуск] {bbox}: не загружено; продолжаю остальные плитки", flush=True)
            return []
        print(f"    [Overpass] плитка {tuple(round(v, 4) for v in bbox)} не скачалась целиком — делю на 4", flush=True)
        ms, mw = (s_ + n_) / 2, (w_ + e_) / 2
        out: list[dict] = []
        incomplete = []
        def child_missing(sub, error):
            incomplete.append(sub)
            on_missing(sub, error)
        for sub in ((s_, w_, ms, mw), (s_, mw, ms, e_), (ms, w_, n_, mw), (ms, mw, n_, e_)):
            out += fetch_adaptive(make_query, sub, cache_dir, fetch, min_deg,
                                  child_missing if on_missing is not None else None)
        # Все дочерние плитки получены: кэшируем объединение под исходным ключом.
        unique = {(el["type"], el["id"]): el for el in out}
        result = list(unique.values())
        if incomplete:
            return result  # Не выдаём неполный ответ за готовый кэш родительской плитки.
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(make_query(bbox).encode()).hexdigest()[:16]
        target = cache_dir / (key + ".json")
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"elements": result}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return result


def buildings_query(bbox) -> str:
    s, w, n, e = bbox
    return f'[out:json][timeout:240];way["building"]({s},{w},{n},{e});out geom tags;'


def hotels_query(bbox) -> str:
    s, w, n, e = bbox
    return (f'[out:json][timeout:120];nwr["tourism"~"{HOTEL_RE}"]({s},{w},{n},{e});'
            f'out center tags;')


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------
def _num(s) -> float | None:
    if s is None:
        return None
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(s))
    return float(m.group().replace(",", ".")) if m else None


def building_height(tags: dict) -> tuple[float, str, int | None]:
    """(высота_м, откуда, этажей). Откуда: height | levels | default."""
    h, lv = _num(tags.get("height")), _num(tags.get("building:levels"))
    levels = int(lv) if lv else None
    if h:
        return h, "height", levels or max(1, round(h / LEVEL_HEIGHT_M))
    if lv:
        return lv * LEVEL_HEIGHT_M, "levels", levels
    return DEFAULT_HEIGHT_M, "default", None


@dataclass
class Building:
    id: int
    poly: np.ndarray   # (N, 2) локальные метры, замкнутый контур без повтора первой точки
    height: float
    height_src: str
    levels: int | None
    tags: dict


def parse_buildings(data: dict, grid: Grid) -> list[Building]:
    out = []
    for el in data.get("elements", []):
        geom = el.get("geometry")
        if el.get("type") != "way" or not geom or len(geom) < 4:
            continue
        lat = np.array([p["lat"] for p in geom])
        lon = np.array([p["lon"] for p in geom])
        x, y = grid.to_local(lat, lon)
        pts = np.column_stack([x, y])
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            continue
        tags = el.get("tags", {})
        h, src, lv = building_height(tags)
        out.append(Building(el["id"], pts, h, src, lv, tags))
    return out


def rasterize(grid: Grid, polys: list[np.ndarray], values: list[float], dtype=float) -> np.ndarray:
    """Растр [iy, ix]; более высокие значения рисуются поверх низких."""
    img = Image.new("F", (grid.nx, grid.ny), 0.0)
    dr = ImageDraw.Draw(img)
    for i in np.argsort(values):
        pts = [(float(px) / grid.res, float(py) / grid.res) for px, py in polys[i]]
        dr.polygon(pts, fill=float(values[i]))
    return np.asarray(img, dtype=dtype)


def rasterize_buildings(grid: Grid, buildings: list[Building]) -> np.ndarray:
    return rasterize(grid, [b.poly for b in buildings], [b.height for b in buildings])


def footprint_mask(grid: Grid, poly: np.ndarray) -> np.ndarray:
    return rasterize(grid, [poly], [1.0]) > 0.5


# --------------------------------------------------------------------------
# Отели
# --------------------------------------------------------------------------
def _point_in_poly(px: float, py: float, poly: np.ndarray) -> bool:
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cond = (y > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x2 - x) * (py - y) / (y2 - y) + x
    return bool(np.count_nonzero(cond & (px < xint)) % 2)


def synthetic_square(cx: float, cy: float, side: float = 20.0) -> np.ndarray:
    h = side / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])


def parse_hotels(data: dict, grid: Grid, buildings: list[Building], match_radius: float = 30.0) -> list[dict]:
    by_id = {b.id: b for b in buildings}
    bb = np.array([[b.poly[:, 0].min(), b.poly[:, 0].max(), b.poly[:, 1].min(), b.poly[:, 1].max()]
                   for b in buildings]) if buildings else np.zeros((0, 4))
    hotels = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:ru")
        if not name:
            continue
        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        elif "lat" in el:
            lat, lon = el["lat"], el["lon"]
        else:
            continue
        x, y = (float(v) for v in grid.to_local(lat, lon))
        b, src = None, "synthetic"
        if el["type"] == "way" and el["id"] in by_id:
            b, src = by_id[el["id"]], "osm_way"
        else:
            cand = np.where((bb[:, 0] <= x) & (x <= bb[:, 1]) & (bb[:, 2] <= y) & (y <= bb[:, 3]))[0] if len(bb) else []
            for i in cand:
                if _point_in_poly(x, y, buildings[i].poly):
                    b, src = buildings[i], "osm_containing"
                    break
            if b is None and len(bb):  # ближайший дом в радиусе
                cx, cy = (bb[:, 0] + bb[:, 1]) / 2, (bb[:, 2] + bb[:, 3]) / 2
                d = np.hypot(cx - x, cy - y)
                j = int(np.argmin(d))
                if d[j] <= match_radius:
                    b, src = buildings[j], "osm_nearest"
        poly = b.poly if b is not None else synthetic_square(x, y)
        height, hsrc, levels = (b.height, b.height_src, b.levels) if b is not None else (DEFAULT_HEIGHT_M, "default", None)
        hotels.append({
            "hotel_id": f'{el["type"][0]}{el["id"]}', "name": name, "lat": lat, "lon": lon, "x": x, "y": y,
            "poly": poly, "footprint_source": src, "height": height, "height_source": hsrc,
            "levels": levels or max(1, round(height / LEVEL_HEIGHT_M)),
            "tags": {k: tags[k] for k in ("tourism", "stars", "website", "addr:street", "addr:housenumber") if k in tags},
        })
    return hotels


# --------------------------------------------------------------------------
# Фасады
# --------------------------------------------------------------------------
def facades_from_polygon(poly: np.ndarray, min_len: float = 6.0, offset: float = 1.5) -> list[Facade]:
    """Самая длинная стена в каждом из 8 секторов; окно — на 1.5 м снаружи от стены."""
    pts = np.asarray(poly, float)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area < 0:  # привести к обходу против часовой стрелки
        pts = pts[::-1]
    best: dict[int, tuple[float, Facade]] = {}
    for p, q in zip(pts, np.roll(pts, -1, axis=0)):
        e = q - p
        length = float(np.hypot(*e))
        if length < min_len:
            continue
        nx, ny = e[1] / length, -e[0] / length  # внешняя нормаль для CCW
        az = math.degrees(math.atan2(nx, ny)) % 360
        sector = int(((az + 22.5) % 360) // 45)
        mid = (p + q) / 2 + np.array([nx, ny]) * offset
        if sector not in best or length > best[sector][0]:
            best[sector] = (length, Facade(SECTORS[sector], az, float(mid[0]), float(mid[1])))
    return [f for _, (_, f) in sorted(best.items())]
