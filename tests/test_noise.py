"""Физическая проверка модели шума на сценах с известным ответом."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.geo.noise import NoiseModel, noise_class, noise_feature, L10
from tools.geo.visibility import Facade, Scene

RES, N = 5.0, 400  # 2x2 км
def flat(bld=None):
    b = np.zeros((N, N)) if bld is None else bld
    return Scene(np.zeros((N, N)), b, RES, 0.0, 0.0)

road = {"lines": [{"cls": "highway:secondary", "name": "ул. Тестовая",
                   "pts": np.array([[0.0, 1000.0], [2000.0, 1000.0]])}], "points": []}
m = NoiseModel(road)
sc = flat()

# 1) линия: на 10 м ≈ L10, на 20 м −3 дБ, на 40 м −6 дБ (затухание 1/r)
def lvl(dy, az=180.0, scene=sc, z=1.5, excl=None):
    return m.levels(scene, 1000.0, 1000.0 + dy, az, [z], excl)[0]
a, b, c = lvl(10)["day"], lvl(20)["day"], lvl(40)["day"]
print("день: 10 м %.1f, 20 м %.1f, 40 м %.1f" % (a, b, c))
assert abs(a - L10["highway:secondary"]) < 0.6
assert abs((a - b) - 3.0) < 0.4 and abs((a - c) - 6.0) < 0.5
assert abs(lvl(10)["night"] - (a - 8)) < 0.6          # ночью поток ниже

# 2) стена своего фасада: окно смотрит от дороги -> −12 дБ
back = lvl(10, az=0.0)["day"]
print("окно от дороги: %.1f" % back)
assert 11 < a - back < 13

# 3) экран: дом 20 м между дорогой и окном -> потери по Маекаве, не больше 20 дБ
b2 = np.zeros((N, N)); b2[201, :] = 20.0   # полоса дома 5 м на y≈1005, между дорогой (1000) и окном (1010)
shield = lvl(10, scene=flat(b2))["day"]
print("за домом 20 м: %.1f (было %.1f)" % (shield, a))
assert 8 < a - shield <= 20.5
# геометрия луча: дом 20 м на полпути, поэтому окно на 30 м экранировано, а на 100 м видит дорогу поверх крыши
assert a - lvl(10, scene=flat(b2), z=30.0)["day"] > 8
high = lvl(10, scene=flat(b2), z=100.0)["day"]
print("окно на 100 м: %.1f" % high)
assert abs(high - a) < 1.0

# 4) нет источников -> фон; классы
assert NoiseModel({"lines": [], "points": []}).levels(sc, 0, 0, 0, [1.5])[0]["night"] == 30.0
assert [noise_class(x) for x in (35, 45, 52, 60)] == ["тихо", "умеренно", "шумно", "очень шумно"]

# 5) ночное заведение: точка, 1/r², ночью не ослабляется
club = NoiseModel({"lines": [], "points": [{"cls": "amenity:nightclub", "name": "Клуб", "xy": np.array([1000.0, 1000.0])}]})
l20, l40 = (club.levels(sc, 1000.0, 1000.0 + d, 180.0, [1.5])[0] for d in (20, 40))
assert abs((l20["night"] - l40["night"]) - 6.0) < 0.4 and l20["day"] == l20["night"]

# 6) feature-обёртка
facs = [Facade("юг", 180, 1000, 1010), Facade("север", 0, 1000, 1030)]  # юг смотрит на дорогу
f = noise_feature(m, sc, facs, [1, 2, 3, 4])
print(f["meta"]["summary"], "| conf", f["confidence"])
assert f["value"]["юг"]["1"]["night"] - f["value"]["север"]["1"]["night"] > 10
assert f["confidence"] <= 0.7
print("OK")
