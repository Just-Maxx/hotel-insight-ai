"""Реальный пеший маршрут по сети OSM с учётом уклона (функция ходьбы Тоблера).

Для каждого набора целей (пляж, остановка, аптека...) один раз считаем поле «время до
ближайшей цели» из всех узлов графа (Дейкстра по обращённому графу из виртуального
источника), потом по каждому отелю остаётся только достать значение и восстановить путь.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .visibility import Scene

BUSY = {"trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"}
SNAP_SPEED = 1.4  # м/с для «соединителя» от точки до ближайшего узла сети


def tobler_time(length: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Время в секундах: v = 6·exp(−3.5·|s+0.05|) км/ч, s — уклон. На плоскости ≈ 5 км/ч."""
    s = np.clip(dz / np.maximum(length, 1e-6), -0.6, 0.6)
    v = 6.0 * np.exp(-3.5 * np.abs(s + 0.05)) * 1000 / 3600
    return length / np.maximum(v, 0.14)


@dataclass
class Field:
    kind: str
    time: np.ndarray   # секунды до ближайшей цели из каждого узла (inf — недостижимо)
    pred: np.ndarray   # следующий узел на пути к цели
    n_targets: int
    sink: int          # индекс виртуального источника


class WalkGraph:
    def __init__(self, node_xy: dict[int, tuple[float, float]], ways: list[dict], scene: Scene):
        used = sorted({n for w in ways for n in w["nodes"] if n in node_xy})
        self.N = len(used)
        if not self.N:
            return
        idx = {n: i for i, n in enumerate(used)}
        self.xy = np.array([node_xy[n] for n in used], dtype=float)
        dem = ndimage.median_filter(scene.dem, size=3)  # сглаживаем шум DSM (кроны, машины)
        ix = np.clip(np.rint((self.xy[:, 0] - scene.x0) / scene.res).astype(int), 0, dem.shape[1] - 1)
        iy = np.clip(np.rint((self.xy[:, 1] - scene.y0) / scene.res).astype(int), 0, dem.shape[0] - 1)
        self.z = dem[iy, ix]

        U, V, HW = [], [], []
        for w in ways:
            ns = [idx[n] for n in w["nodes"] if n in idx]
            for a, b in zip(ns, ns[1:]):
                if a != b:
                    U.append(a); V.append(b); HW.append(w["hw"])
        if not U:
            self.N = 0
            return
        u = np.array(U + V)
        v = np.array(V + U)
        hw = np.array(HW + HW)
        L = np.hypot(*(self.xy[v] - self.xy[u]).T)
        T = tobler_time(L, self.z[v] - self.z[u])
        busy = np.isin(hw, list(BUSY))
        order = np.lexsort((T, v, u))  # дубли рёбер: оставляем самое быстрое
        u, v, hw, L, T, busy = u[order], v[order], hw[order], L[order], T[order], busy[order]
        keep = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
        u, v, hw, L, T, busy = u[keep], v[keep], hw[keep], L[keep], T[keep], busy[keep]

        n = self.N
        self.fwd = sparse.csr_matrix((T, (u, v)), shape=(n, n))
        self.length = sparse.csr_matrix((L, (u, v)), shape=(n, n))
        self.busy_edge = sparse.csr_matrix((busy.astype(np.int8), (u, v)), shape=(n, n))
        self.node_busy = np.zeros(n, bool)
        self.node_busy[u[busy]] = True
        ped = hw == "pedestrian"
        self.ped_mid = (self.xy[u[ped]] + self.xy[v[ped]]) / 2
        self.ped_len = L[ped] / 2  # каждое ребро хранится в двух направлениях
        self.tree = cKDTree(self.xy)

    @property
    def empty(self) -> bool:
        return self.N == 0

    def nearest(self, x: float, y: float, max_dist: float = 150.0):
        if self.empty:
            return None, None
        d, i = self.tree.query([x, y])
        return (int(i), float(d)) if d <= max_dist else (None, float(d))

    def field(self, kind: str, targets_xy: np.ndarray, snap_max: float = 120.0) -> Field | None:
        if self.empty or len(targets_xy) == 0:
            return None
        d, nodes = self.tree.query(targets_xy)
        ok = d <= snap_max
        if not ok.any():
            return None
        conn, nodes = d[ok] / SNAP_SPEED, nodes[ok]
        order = np.argsort(conn)
        uniq, first = np.unique(nodes[order], return_index=True)  # на узел — самый близкий вход
        conn = conn[order][first]
        n = self.N
        src = sparse.csr_matrix((conn, (np.zeros(len(uniq), int), uniq)), shape=(1, n))
        G = sparse.vstack([sparse.hstack([self.fwd.T.tocsr(), sparse.csr_matrix((n, 1))]),
                           sparse.hstack([src, sparse.csr_matrix((1, 1))])]).tocsr()
        dist, pred = dijkstra(G, directed=True, indices=n, return_predecessors=True)
        return Field(kind, dist[:n], pred[:n], int(ok.sum()), n)

    def describe(self, f: Field | None, x: float, y: float, snap_max: float = 150.0) -> dict | None:
        """Минуты, метры, набор высоты и число пересечений крупных дорог от точки (x, y) до ближайшей цели."""
        if f is None or self.empty:
            return None
        node, d = self.nearest(x, y, snap_max)
        if node is None or not np.isfinite(f.time[node]):
            return None
        path = [node]
        while True:
            p = f.pred[path[-1]]
            if p < 0 or p == f.sink or len(path) > 5000:
                break
            path.append(int(p))
        length = ascent = 0.0
        crossings = 0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            length += self.length[a, b]
            ascent += max(self.z[b] - self.z[a], 0.0)
            if i > 0 and self.node_busy[a] and not self.busy_edge[path[i - 1], a] and not self.busy_edge[a, b]:
                crossings += 1  # зашли на узел крупной дороги не по ней и вышли не по ней: это пересечение
        return {"minutes": round((f.time[node] + d / SNAP_SPEED) / 60, 1), "distance_m": int(length + d),
                "ascent_m": round(float(ascent), 1), "busy_crossings": crossings, "snap_m": int(d)}

    def pedestrian_length_near(self, x: float, y: float, r: float = 300.0) -> float:
        if self.empty or not len(self.ped_mid):
            return 0.0
        m = np.hypot(*(self.ped_mid - np.array([x, y])).T) <= r
        return float(self.ped_len[m].sum())
