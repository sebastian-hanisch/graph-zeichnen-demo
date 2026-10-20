"""Szenario: Graphen, die gezeichnet werden (Karte, Karte mit fehlenden Kanten, Raster, Gitter, Zwiebel, gestapelte Triangulierung, Straßenausschnitt, Handgraphen).

Ein Graph ist eine Kantenliste über den Knoten 0 bis n − 1 (ungerichtet, einfach, jede Kante als (u, v) mit u < v). `true` ist, wo es sie gibt, die wahre Lage
(Karte, Raster, Straßenausschnitt); sie dient nur als Vorlage für die Treue einer Zeichnung und ist selbst nicht unbedingt kreuzungsfrei."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from gzn_planar import adjacency, largest_block

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Instance:
    n: int
    edges: tuple[tuple[int, int], ...]
    true: tuple[tuple[float, float], ...] | None
    name: str = ""

    @property
    def m(self) -> int:
        return len(self.edges)


def normalize_edges(edges) -> tuple[tuple[int, int], ...]:
    """Sortierte, doppelfreie Kantenliste ohne Schlingen."""
    return tuple(sorted({(min(u, v), max(u, v)) for u, v in edges if u != v}))


def relabel(edges, true=None) -> tuple[int, tuple[tuple[int, int], ...], tuple | None]:
    """Knoten, die in Kanten vorkommen, auf 0 … n − 1 umnummerieren (Reihenfolge der alten Nummern)."""
    nodes = sorted({x for e in edges for x in e})
    index = {v: i for i, v in enumerate(nodes)}
    new_edges = normalize_edges((index[u], index[v]) for u, v in edges)
    new_true = tuple(true[v] for v in nodes) if true is not None else None
    return len(nodes), new_edges, new_true


def _delaunay_edges(points: np.ndarray) -> list[tuple[int, int]]:
    """Kanten der Delaunay-Triangulierung (scipy/Qhull, nur hier gebraucht)."""
    from scipy.spatial import Delaunay

    tri = Delaunay(points)
    found = set()
    for simplex in tri.simplices:
        a, b, c = (int(x) for x in simplex)
        found.update({(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))})
    return sorted(found)


def make_delaunay(n: int, seed: int) -> Instance:
    """Zufällige Standorte im Einheitsquadrat und alle Kanten ihrer Delaunay-Triangulierung."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    pos = tuple((round(float(x), 4), round(float(y), 4)) for x, y in points)
    return Instance(n, normalize_edges(_delaunay_edges(points)), pos, "Karte")


def make_thin(n: int, seed: int, keep: float = 0.8) -> Instance:
    """Dieselbe Karte mit zufällig `keep` der Kanten; es bleibt der größte zweifach zusammenhängende Teil (umnummeriert)."""
    base = make_delaunay(n, seed)
    rng = np.random.default_rng(seed + 7919)
    count = round(keep * base.m)
    chosen = sorted(int(i) for i in rng.choice(base.m, size=count, replace=False))
    kept = [base.edges[i] for i in chosen]
    block = largest_block(n, kept)
    m, edges, true = relabel(block, base.true)
    return Instance(m, edges, true, "Karte mit fehlenden Kanten")


def make_grid(n: int, diagonals: bool) -> Instance:
    """k × k Knoten (k = Wurzel aus n, gerundet, mindestens 2); mit einer Diagonale je Zelle oder ohne."""
    k = max(2, round(math.sqrt(n)))
    edges = []
    for r in range(k):
        for c in range(k):
            i = r * k + c
            if c + 1 < k:
                edges.append((i, i + 1))
            if r + 1 < k:
                edges.append((i, i + k))
            if diagonals and r + 1 < k and c + 1 < k:
                edges.append((i, i + k + 1))
    pos = tuple((float(i % k), float(-(i // k))) for i in range(k * k))
    return Instance(k * k, normalize_edges(edges), pos, "Raster mit Diagonalen" if diagonals else "Gitter")


def make_onion(n: int) -> Instance:
    """k = n // 3 (mindestens 2) Dreiecke ineinander; die Ecken jeder Stufe sind mit den entsprechenden Ecken der nächsten verbunden. 3-fach zusammenhängend und planar."""
    k = max(2, n // 3)
    edges = []
    for i in range(k):
        a, b, c = 3 * i, 3 * i + 1, 3 * i + 2
        edges += [(a, b), (b, c), (a, c)]
        if i + 1 < k:
            edges += [(a, a + 3), (b, b + 3), (c, c + 3)]
    return Instance(3 * k, normalize_edges(edges), None, "Zwiebel")


def make_stacked(n: int, seed: int) -> Instance:
    """Von einem Dreieck aus wird n − 3 Mal eine zufällige innere Dreiecksfläche durch einen neuen Knoten geteilt (maximal planar, 3-fach zusammenhängend)."""
    n = max(n, 3)
    rng = np.random.default_rng(seed)
    edges = {(0, 1), (1, 2), (0, 2)}
    faces = [(0, 1, 2)]
    for v in range(3, n):
        a, b, c = faces.pop(int(rng.integers(len(faces))))
        edges |= {(a, v), (b, v), (c, v)}
        faces += [(a, b, v), (b, c, v), (a, c, v)]
    return Instance(n, normalize_edges(edges), None, "Gestapelte Triangulierung")


@lru_cache(maxsize=1)
def load_toronto() -> Instance:
    """Straßenkarte aus OpenStreetMap (ODbL, Auszug aus bfs-demo): Kanten ungerichtet vereinigt, die Komponente des Knotens 0; Lage in Metern."""
    payload = json.loads((DATA / "toronto_campus.json").read_text(encoding="utf-8"))
    pos = [(float(x), float(y)) for x, y in payload["nodes_xy_m"]]
    edges = normalize_edges((u, v) for u, v, _ in payload["arcs"])
    adj = adjacency(len(pos), edges)
    seen = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    keep = [(u, v) for u, v in edges if u in seen and v in seen]
    m, new_edges, new_true = relabel(keep, pos)
    return Instance(m, new_edges, new_true, "Toronto-Karte")


def make_toronto_ball(n: int) -> Instance:
    """Die n Knoten um die Mitte der Karte (Breitensuche ab dem Knoten, der dem Median der Lage am nächsten liegt; Gleichstand: kleinste Nummer) mit allen Kanten zwischen ihnen."""
    full = load_toronto()
    pos = np.array(full.true)
    centre = np.median(pos, axis=0)
    start = int(np.argmin(np.linalg.norm(pos - centre, axis=1)))
    adj = adjacency(full.n, full.edges)
    order = [start]
    seen = {start}
    queue = deque([start])
    while queue and len(order) < n:
        x = queue.popleft()
        for y in sorted(adj[x]):
            if y not in seen and len(order) < n:
                seen.add(y)
                order.append(y)
                queue.append(y)
    keep = [(u, v) for u, v in full.edges if u in seen and v in seen]
    m, edges, true = relabel(keep, full.true)
    return Instance(m, edges, true, "Straßenausschnitt")


def hand_graph(name: str) -> Instance:
    """Kleine Graphen von Hand: K4, Würfel, Rad mit sechs Speichen."""
    if name == "K4":
        return Instance(4, normalize_edges([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]), None, "K4")
    if name == "Würfel":
        edges = [(0, 1), (1, 2), (2, 3), (0, 3), (4, 5), (5, 6), (6, 7), (4, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        return Instance(8, normalize_edges(edges), None, "Würfel")
    if name == "Rad":
        edges = [(0, i) for i in range(1, 7)] + [(i, i % 6 + 1) for i in range(1, 7)]
        return Instance(7, normalize_edges(edges), None, "Rad")
    raise ValueError(name)


def make_instance(net: str, n: int, seed: int) -> Instance:
    """Deterministisch aus (net, n, seed): gleiche Eingabe, gleiche Instanz."""
    if net == "delaunay":
        return make_delaunay(n, seed)
    if net == "thin":
        return make_thin(n, seed)
    if net == "grid":
        return make_grid(n, True)
    if net == "lattice":
        return make_grid(n, False)
    if net == "onion":
        return make_onion(n)
    if net == "stacked":
        return make_stacked(n, seed)
    if net == "toronto":
        return make_toronto_ball(n)
    raise ValueError(net)
