"""Gütemaße einer Zeichnung: Kreuzungen und Berührungen (exakt, wenn die Koordinaten Fraction oder int sind), Auflösung, Fläche bei Mindestabstand 1,
Kantenverhältnis, kleinster Winkel, Treue zu einer Vorlage. Eine Zeichnung ist eine Liste von Punkten (x, y), Zahlen beliebiger Art (float, int, Fraction).

Eine Zeichnung gilt nur als *gültig*, wenn sich keine zwei Kanten kreuzen UND nichts aufeinanderliegt (Knoten auf Knoten, Knoten auf einer fremden Kante):
der einfache Kreuzungszähler allein meldet bei zusammengefallenen Knoten null (Befund der Vormessung, tests/test_claims.py)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import procrustes as _procrustes

MARGIN = 1e-12      # Rand der Float-Vorauswahl, relativ: größer als jeder Rundungsfehler beim Umwandeln einer Fraction in float


def orientation(a, b, c) -> int:
    """+1, 0, −1: c liegt links von, auf oder rechts von der Geraden a→b (exakt bei int und Fraction)."""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return int(cross > 0) - int(cross < 0)


def segments_cross(p1, p2, q1, q2) -> bool:
    """Schneiden sich die Strecken p1p2 und q1q2 in einem Punkt, der in beiden Strecken im Inneren liegt (Berührungen und Überlagerungen zählen nicht)?"""
    o1, o2 = orientation(p1, p2, q1), orientation(p1, p2, q2)
    o3, o4 = orientation(q1, q2, p1), orientation(q1, q2, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def _boxes(pos, edges):
    """Umgebende Rechtecke der Kanten in float, jeweils um einen kleinen Rand vergrößert (Vorauswahl; die Entscheidung fällt exakt)."""
    out = []
    for a, b in edges:
        xa, ya, xb, yb = float(pos[a][0]), float(pos[a][1]), float(pos[b][0]), float(pos[b][1])
        out.append((min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)))
    scale = max([1.0] + [abs(v) for box in out for v in box])
    m = MARGIN * scale
    return [(x0 - m, x1 + m, y0 - m, y1 + m) for x0, x1, y0, y1 in out]


def crossings(pos, edges) -> list[tuple[int, int]]:
    """Paare (i, j), i < j, von Kantenindizes, deren Strecken sich kreuzen (Kanten mit gemeinsamem Endpunkt zählen nie)."""
    edges = list(edges)
    boxes = _boxes(pos, edges)
    order = sorted(range(len(edges)), key=lambda i: boxes[i][0])
    found = []
    for oi, i in enumerate(order):
        a, b = edges[i]
        bi = boxes[i]
        for j in order[oi + 1:]:
            bj = boxes[j]
            if bj[0] > bi[1]:
                break
            if bj[2] > bi[3] or bj[3] < bi[2]:
                continue
            c, d = edges[j]
            if len({a, b, c, d}) == 4 and segments_cross(pos[a], pos[b], pos[c], pos[d]):
                found.append((min(i, j), max(i, j)))
    return sorted(found)


def touching(pos, edges) -> dict[str, list]:
    """Aufeinanderliegendes: `coincident` = Paare von Knoten mit genau gleicher Lage, `on_edge` = (Knoten, Kantenindex) für einen Knoten im Inneren einer fremden Kante."""
    edges = list(edges)
    seen: dict[tuple, int] = {}
    coincident = []
    for v, p in enumerate(pos):
        key = (p[0], p[1])
        if key in seen:
            coincident.append((seen[key], v))
        else:
            seen[key] = v
    boxes = _boxes(pos, edges)
    on_edge = []
    fpos = [(float(p[0]), float(p[1])) for p in pos]
    for idx, (a, b) in enumerate(edges):
        x0, x1, y0, y1 = boxes[idx]
        for v, (fx, fy) in enumerate(fpos):
            if v == a or v == b or not (x0 <= fx <= x1 and y0 <= fy <= y1):
                continue
            if tuple(pos[v]) == tuple(pos[a]) or tuple(pos[v]) == tuple(pos[b]):
                continue          # schon unter `coincident`
            if orientation(pos[a], pos[b], pos[v]) == 0:
                on_edge.append((v, idx))
    return {"coincident": coincident, "on_edge": on_edge}


def to_float(pos) -> np.ndarray:
    return np.array([[float(p[0]), float(p[1])] for p in pos], dtype=float).reshape(-1, 2)


def min_distance(P: np.ndarray) -> float:
    """Kleinster Abstand zweier Knoten."""
    if len(P) < 2:
        return 0.0
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def diagonal(P: np.ndarray) -> float:
    return float(np.hypot(*(P.max(axis=0) - P.min(axis=0)))) if len(P) else 0.0


def resolution(P: np.ndarray) -> float:
    """Kleinster Knotenabstand je Diagonale der Zeichnung (skalenfrei); 0, wenn zwei Knoten aufeinanderliegen."""
    diag = diagonal(P)
    return min_distance(P) / diag if diag > 0 else 0.0


def area_at_unit_distance(P: np.ndarray) -> float:
    """Fläche der Zeichnung (umgebendes Rechteck), wenn sie so skaliert wird, dass der kleinste Knotenabstand 1 ist: die Größe, die man bräuchte, damit nie zwei Knoten näher als 1 liegen.
    Eine Ausdehnung unter dem Mindestabstand zählt als Mindestabstand (eine Gerade hat nicht die Fläche 0); unendlich bei aufeinanderliegenden Knoten."""
    dmin = min_distance(P)
    if dmin <= 0:
        return math.inf
    ext = P.max(axis=0) - P.min(axis=0)
    return float(max(ext[0], dmin) * max(ext[1], dmin) / dmin ** 2)


def edge_ratio(P: np.ndarray, edges) -> float:
    """Längste durch kürzeste Kante; unendlich, wenn eine Kante die Länge 0 hat."""
    E = np.array(list(edges))
    if not len(E):
        return 1.0
    length = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    return float(length.max() / length.min()) if length.min() > 0 else math.inf


def min_angle(P: np.ndarray, edges) -> float:
    """Kleinster Winkel (Grad) zwischen zwei Kanten am selben Knoten; 180, wenn kein Knoten zwei Kanten hat."""
    n = len(P)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)
    best = 180.0
    for v in range(n):
        if len(nbrs[v]) < 2:
            continue
        ang = sorted(math.degrees(math.atan2(P[w][1] - P[v][1], P[w][0] - P[v][0])) % 360.0 for w in nbrs[v])
        gaps = [ang[i + 1] - ang[i] for i in range(len(ang) - 1)] + [360.0 - ang[-1] + ang[0]]
        best = min(best, min(gaps))
    return best


def fidelity(P: np.ndarray, true: np.ndarray) -> float | None:
    """Abweichung von einer Vorlage nach bestem Verschieben, Drehen und Skalieren (Procrustes, 0 = gleiche Gestalt, 1 = nichts gemeinsam); None, wenn nicht berechenbar."""
    try:
        return float(_procrustes(np.asarray(true, dtype=float), P)[2])
    except ValueError:
        return None


@dataclass(frozen=True)
class Measures:
    crossings: int
    touching: int            # Knotenpaare aufeinander plus Knoten auf fremden Kanten
    resolution: float
    area: float              # Fläche bei Mindestabstand 1
    edge_ratio: float
    min_angle: float
    stress: float | None
    fidelity: float | None

    @property
    def valid(self) -> bool:
        return self.crossings == 0 and self.touching == 0


def measure(pos, edges, stress: float | None = None, true: np.ndarray | None = None, pairs=None) -> Measures:
    """Alle Maße einer Zeichnung; `stress` und `true` kommen von außen (sie brauchen Weglängen bzw. Vorlage), `pairs` sind schon gezählte Kreuzungen."""
    edges = list(edges)
    P = to_float(pos)
    t = touching(pos, edges)
    return Measures(
        len(crossings(pos, edges) if pairs is None else pairs), len(t["coincident"]) + len(t["on_edge"]), resolution(P), area_at_unit_distance(P), edge_ratio(P, edges),
        min_angle(P, edges), stress, fidelity(P, true) if true is not None else None,
    )
