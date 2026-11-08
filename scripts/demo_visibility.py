"""Синтетическая проверка модуля видимости.

Сцена 2×2 км, шаг 5 м, море на западе (x < 300 м), берег на высоте 5 м.
Отель 30×30 м, высота 25 м (8 этажей). Между отелем и морем — длинный
дом 22 м (y от 900 до 1100): зимой закат уходит на юго-запад и упирается
в него, летом смещается на северо-запад и обходит дом сбоку.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.geo.visibility import (Facade, Scene, landmark_visible, sea_view_fraction,
                                  sunset_azimuth, sunset_feature)
from datetime import date

RES, N = 5.0, 400
dem = np.full((N, N), 5.0)
water = np.zeros((N, N), bool)
water[:, : int(300 / RES)] = True
dem[water] = 0.0

bld = np.zeros((N, N))
def box(arr, x0, x1, y0, y1, h):
    arr[int(y0 / RES): int(y1 / RES), int(x0 / RES): int(x1 / RES)] = h

box(bld, 700, 730, 1000, 1030, 25.0)   # отель
box(bld, 500, 540, 900, 1100, 22.0)    # дом-заслон
hotel_mask = np.zeros((N, N), bool)
box(hotel_mask, 700, 730, 1000, 1030, True)

scene = Scene(dem, bld, RES, 0.0, 0.0, water)
facades = [
    Facade("запад", 270, 700, 1015),
    Facade("юг", 180, 715, 1000),
    Facade("восток", 90, 730, 1015),
]
floors = list(range(1, 9))
LAT = 43.6  # широта Сочи

print("азимут заката: дек %.0f°, июн %.0f°" % (sunset_azimuth(LAT, date(2026, 12, 15)),
                                              sunset_azimuth(LAT, date(2026, 6, 15))))
feat = sunset_feature(scene, facades, floors, LAT, exclude=hotel_mask)
print("\nЗакат:", feat["meta"]["summary"])
print("Уверенность:", feat["confidence"], feat["meta"]["confidence_by_facade"])

w = feat["value"]["запад"]
assert w[6] is not None and w[12] is not None
assert w[6] < w[12], "летом закат должен быть виден с более низких этажей, чем зимой"
assert all(v is None for v in feat["value"]["восток"].values())
south = feat["value"]["юг"]  # зимой закат (237°) попадает в поле зрения южных окон
assert south[12] is not None and south[6] is None

print("\nДоля сектора с видом на море (окна на запад):")
for fl in (2, 5, 8):
    print(f"  этаж {fl}: {sea_view_fraction(scene, facades[0], fl, exclude=hotel_mask):.0%}")
assert sea_view_fraction(scene, facades[0], 8, exclude=hotel_mask) > \
       sea_view_fraction(scene, facades[0], 2, exclude=hotel_mask)

print("\nЛандшафтная цель (башня 60 м в точке 200, 1015):")
for fl in (2, 8):
    r = landmark_visible(scene, 700, 1015, fl, 200, 1015, 60 + 0.0, exclude=hotel_mask)
    print(f"  этаж {fl}: видна={r['visible']}, угол цели {r['target_angle']:.2f}°, горизонт {r['blocking_angle']:.2f}°")

print("\nOK")
