"""Kern: vier Zeichenverfahren für Graphen.

Eine Einheit ist eine Regel: Weglängen, Kräfte des Federmodells, ein Federmodell-Lauf, Stress eines Layouts, klassisches MDS, ein Schritt der Stress-Majorisierung,
SGD auf dem Stress, rationale Punkte auf dem Kreis, die Tutte-Einbettung (exakt in rationalen Zahlen), Triangulieren der Flächen, Drehsystem, kanonische Ordnung,
die Gitterzeichnung nach de Fraysseix, Pach und Pollack. Kein Zufall im Kern: Startlagen und Seeds kommen als Argument; Gleichstand entscheidet die kleinere Knotennummer.
Knoten 0 bis n − 1, Kanten als sortierte Paare."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from gzn_planar import adjacency

# --- Weglängen und Stress --------------------------------------------------------------------------------------------------------------------


def hop_distances(n: int, edges) -> np.ndarray:
    """Weglängen (Zahl der Kanten) zwischen allen Knotenpaaren; nicht verbundene Paare: unendlich."""
    adj = adjacency(n, edges)
    dist = np.full((n, n), np.inf)
    for s in range(n):
        dist[s, s] = 0.0
        queue = deque([s])
        while queue:
            x = queue.popleft()
            for y in adj[x]:
                if dist[s, y] == np.inf:
                    dist[s, y] = dist[s, x] + 1.0
                    queue.append(y)
    return dist


def clean_distances(D: np.ndarray) -> np.ndarray:
    """Nicht verbundene Paare bekommen den größten endlichen Abstand plus eins (sonst bricht jede Stress-Rechnung ab)."""
    finite = D[np.isfinite(D)]
    cap = (finite.max() if finite.size else 0.0) + 1.0
    return np.where(np.isfinite(D), D, cap)


def stress_value(pos: np.ndarray, D: np.ndarray) -> float:
    """Normierter Stress Σ w (s·x − d)² / Σ w d² über alle Paare i < j mit w = 1/d², x = Abstand in der Zeichnung, s = bester Maßstab; 0 = Weglängen genau getroffen."""
    n = len(pos)
    if n < 2:
        return 0.0
    D = clean_distances(D)
    iu = np.triu_indices(n, 1)
    d = D[iu]
    x = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)
    w = 1.0 / d ** 2
    denom = float((w * x * x).sum())
    s = float((w * d * x).sum()) / denom if denom > 0 else 0.0
    return float((w * (s * x - d) ** 2).sum() / (w * d * d).sum())


def classical_mds(D: np.ndarray) -> np.ndarray:
    """Klassisches MDS (Torgerson): Doppelzentrierung der quadrierten Abstände, die zwei größten Eigenwerte; ergibt die Startlage der Stress-Majorisierung."""
    n = len(D)
    if n == 1:
        return np.zeros((1, 2))
    D = clean_distances(D)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:2]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0.0))


def _weights(D: np.ndarray) -> np.ndarray:
    safe = np.where(D > 0, D, 1.0)
    return np.where(D > 0, 1.0 / safe ** 2, 0.0)


def majorization_step(X: np.ndarray, D: np.ndarray, W: np.ndarray, Lplus: np.ndarray) -> np.ndarray:
    """Ein Schritt der Stress-Majorisierung (Guttman-Transformation): X' = L⁺ B(X) X; der Stress wird dabei nicht größer."""
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    dist = np.where(dist > 1e-9, dist, 1e-9)
    B = -W * D / dist
    np.fill_diagonal(B, 0.0)
    np.fill_diagonal(B, -B.sum(axis=1))
    return Lplus @ (B @ X)


def _majorize(D: np.ndarray, start: np.ndarray):
    """Erzeuger: die Lage nach 1, 2, 3, … Schritten."""
    D = clean_distances(D)
    W = _weights(D)
    L = -W.copy()
    np.fill_diagonal(L, W.sum(axis=1))
    Lplus = np.linalg.pinv(L)
    X = np.array(start, dtype=float)
    while True:
        X = majorization_step(X, D, W, Lplus)
        yield X


def stress_majorization(D: np.ndarray, start: np.ndarray, iterations: int = 300) -> np.ndarray:
    """Stress-Majorisierung ab `start` (Weglängen-Maßstab): die Lage nach `iterations` Schritten."""
    X = np.array(start, dtype=float)
    if iterations <= 0 or len(X) < 2:
        return X
    gen = _majorize(D, start)
    for _ in range(iterations):
        X = next(gen)
    return X


def majorization_trace(D: np.ndarray, start: np.ndarray, steps) -> dict[int, np.ndarray]:
    """Die Lage nach den angegebenen Schrittzahlen (0 = Start); eine Rechnung für alle."""
    wanted = sorted(set(int(s) for s in steps))
    out: dict[int, np.ndarray] = {}
    X = np.array(start, dtype=float)
    if len(X) < 2:
        return {s: X for s in wanted}
    gen = _majorize(D, start)
    last = wanted[-1] if wanted else 0
    for t in range(last + 1):
        if t > 0:
            X = next(gen)
        if t in wanted:
            out[t] = X
    return out


def sgd_stress(D: np.ndarray, seed: int, iterations: int = 30, eps: float = 0.1) -> np.ndarray:
    """SGD auf dem Stress (Zheng, Pawar, Goodman): Paare in zufälliger Reihenfolge, Schrittweite fällt von dmax² auf eps."""
    D = clean_distances(D)
    n = len(D)
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2)) * math.sqrt(n)
    if n < 2:
        return X
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    eta_max, eta_min = float(D.max()) ** 2, eps
    lam = math.log(eta_max / eta_min) / max(iterations - 1, 1) if eta_max > eta_min else 0.0
    for t in range(iterations):
        eta = eta_max * math.exp(-lam * t)
        for idx in rng.permutation(len(pairs)):
            i, j = pairs[idx]
            d = D[i, j]
            mu = min(eta / d ** 2, 1.0)
            v = X[i] - X[j]
            mag = math.hypot(v[0], v[1])
            if mag < 1e-9:
                continue
            r = (mag - d) / 2 * v / mag
            X[i] -= mu * r
            X[j] += mu * r
    return X


# --- Federmodell -----------------------------------------------------------------------------------------------------------------------------


def fr_forces(pos: np.ndarray, edges, k: float) -> np.ndarray:
    """Kraft auf jeden Knoten nach Fruchterman und Reingold: Abstoßung k²/d zwischen allen Paaren, Anziehung d²/k entlang der Kanten (bei d = k heben sie sich auf)."""
    n = len(pos)
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, 1.0)
    dist = np.maximum(dist, 1e-9)
    force = (delta / dist[..., None] * (k * k / dist)[..., None]).sum(axis=1)
    if len(edges):
        E = np.array(edges)
        d = pos[E[:, 0]] - pos[E[:, 1]]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        pull = (length * length / k)[:, None] * d / length[:, None]
        np.add.at(force, E[:, 0], -pull)
        np.add.at(force, E[:, 1], pull)
    return force


def fruchterman_reingold(pos: np.ndarray, edges, iterations: int = 300) -> np.ndarray:
    """Federmodell ab `pos` (Einheitsquadrat): je Schritt höchstens die Temperatur weit entlang der Kraft, Temperatur von 0,1 linear auf 0; gleicher Start, gleicher Lauf."""
    P = np.array(pos, dtype=float)
    n = len(P)
    if n < 2:
        return P
    k = math.sqrt(1.0 / n)
    t = 0.1
    for it in range(iterations):
        force = fr_forces(P, edges, k)
        mag = np.maximum(np.linalg.norm(force, axis=1), 1e-9)
        P += force / mag[:, None] * np.minimum(mag, t)[:, None]
        t = 0.1 * (1 - (it + 1) / iterations)
    return P


# --- Tutte-Einbettung (exakt) ----------------------------------------------------------------------------------------------------------------


def canonical_face(face) -> tuple[int, ...]:
    """Fläche als Tupel, das beim kleinsten Knoten beginnt (zyklisch gedreht, Richtung bleibt)."""
    f = list(face)
    i = f.index(min(f))
    return tuple(f[i:] + f[:i])


def pick_outer_face(faces, which: str) -> tuple[int, ...]:
    """Außenfläche für die Tutte-Einbettung: 'longest' (die längste, bei Gleichstand die zuerst sortierte) oder 'shortest'."""
    ordered = sorted((canonical_face(f) for f in faces), key=lambda f: (len(f), f))
    if which == "shortest":
        return ordered[0]
    longest = max(len(f) for f in ordered)
    return next(f for f in ordered if len(f) == longest)


def rational_circle(k: int) -> list[tuple[Fraction, Fraction]]:
    """k verschiedene rationale Punkte genau auf dem Einheitskreis in zyklischer Reihenfolge (t = tan(θ/2) rational angenähert, Punkt ((1 − t²)/(1 + t²), 2t/(1 + t²))); streng konvex."""
    points = []
    for i in range(k):
        if i == 0:
            points.append((Fraction(1), Fraction(0)))
        elif 2 * i == k:
            points.append((Fraction(-1), Fraction(0)))
        else:
            t = Fraction(math.tan(math.pi * i / k)).limit_denominator(10 ** 6)
            points.append(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    return points


def tutte_embedding(n: int, edges, outer) -> list[tuple[Fraction, Fraction]]:
    """Tutte-Einbettung: die Knoten der Außenfläche auf dem Kreis, jeder andere Knoten genau im Schwerpunkt seiner Nachbarn. Das lineare System wird exakt gelöst
    (Gauß-Elimination in rationalen Zahlen, Knoten mit den wenigsten Einträgen zuerst); der Graph muss zusammenhängend sein."""
    adj = adjacency(n, edges)
    fixed = dict(zip(outer, rational_circle(len(outer))))
    inner = [v for v in range(n) if v not in fixed]
    rows: dict[int, dict[int, Fraction]] = {}
    rhs: dict[int, list[Fraction]] = {}
    for v in inner:
        row = {v: Fraction(len(adj[v]))}
        rx, ry = Fraction(0), Fraction(0)
        for w in adj[v]:
            if w in fixed:
                rx += fixed[w][0]
                ry += fixed[w][1]
            else:
                row[w] = Fraction(-1)
        rows[v] = row
        rhs[v] = [rx, ry]
    steps = []
    remaining = set(inner)
    while remaining:
        v = min(remaining, key=lambda x: (len(rows[x]), x))
        remaining.discard(v)
        pivot = rows[v][v]
        others = {u: c for u, c in rows[v].items() if u != v}
        steps.append((v, pivot, others, rhs[v]))
        for w in others:
            factor = rows[w][v] / pivot
            row_w = rows[w]
            del row_w[v]
            for u, c in others.items():
                new = row_w.get(u, Fraction(0)) - factor * c
                if new == 0:
                    row_w.pop(u, None)
                else:
                    row_w[u] = new
            rhs[w] = [rhs[w][0] - factor * rhs[v][0], rhs[w][1] - factor * rhs[v][1]]
        del rows[v]
    pos: dict[int, tuple[Fraction, Fraction]] = dict(fixed)
    for v, pivot, others, r in reversed(steps):
        x = r[0] - sum((c * pos[u][0] for u, c in others.items()), Fraction(0))
        y = r[1] - sum((c * pos[u][1] for u, c in others.items()), Fraction(0))
        pos[v] = (x / pivot, y / pivot)
    return [pos[v] for v in range(n)]


# --- Gitterzeichnung nach de Fraysseix, Pach und Pollack -------------------------------------------------------------------------------------


def triangulate_faces(faces, edges) -> tuple[list[tuple[int, ...]], list[tuple[int, int]]]:
    """Teilt jede Fläche ab vier Ecken durch Sehnen in Dreiecke: es wird je eine Sehne zwischen zwei nicht benachbarten Ecken der Fläche gesetzt, die es als Kante noch nicht gibt
    (sie existiert immer: gäbe es alle schon, kreuzten sich zwei Sehnen außerhalb der Fläche). Rückgabe: die Dreiecke (mit der Richtung der Flächen) und die neuen Kanten."""
    edge_set = {(min(u, v), max(u, v)) for u, v in edges}
    work = deque(list(f) for f in faces)
    triangles: list[tuple[int, ...]] = []
    added: list[tuple[int, int]] = []
    while work:
        f = work.popleft()
        k = len(f)
        if k == 3:
            triangles.append(tuple(f))
            continue
        chord = None
        for i in range(k):
            for j in range(i + 2, k):
                if i == 0 and j == k - 1:
                    continue
                pair = (min(f[i], f[j]), max(f[i], f[j]))
                if pair not in edge_set:
                    chord = (i, j, pair)
                    break
            if chord:
                break
        if chord is None:
            raise ValueError("keine freie Sehne in einer Fläche: der Graph ist nicht einfach planar eingebettet")
        i, j, pair = chord
        edge_set.add(pair)
        added.append(pair)
        work.appendleft(f[j:] + f[: i + 1])
        work.appendleft(f[i: j + 1])
    return triangles, added


def rotation_system(faces) -> dict[int, list[int]]:
    """Nachbarn jedes Knotens in zyklischer Reihenfolge, aus den gleich orientierten Flächen: auf u, v folgt in einer Fläche w, dann kommt w nach u."""
    succ: dict[tuple[int, int], int] = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            succ[(f[i], f[(i + 1) % k])] = f[(i + 2) % k]
    firsts: dict[int, int] = {}
    for (u, v) in sorted(succ):
        firsts.setdefault(v, u)
    rot: dict[int, list[int]] = {}
    for v, start in firsts.items():
        seq = [start]
        cur = succ[(start, v)]
        while cur != start:
            seq.append(cur)
            cur = succ[(cur, v)]
        rot[v] = seq
    return rot


@dataclass(frozen=True)
class CanonicalOrder:
    order: tuple[int, ...]                       # v1, v2, …, vn
    attach: dict[int, tuple[int, ...]]           # für v3 … vn: die Nachbarn in G_{k−1}, wie sie nacheinander auf der Außenlinie liegen (von links nach rechts)


def canonical_ordering(triangles, outer: tuple[int, int, int]) -> CanonicalOrder:
    """Kanonische Ordnung einer Triangulierung mit Außendreieck (a, b, c): v1 = a, v2 = b, vn = c. Rückwärts: von der Außenlinie (von v1 bis v2) wird wiederholt der kleinste Knoten
    ohne Sehne (Kante zwischen zwei Knoten der Außenlinie, die nicht Nachbarn auf ihr sind) entfernt; seine Nachbarn im Rest liegen dann nacheinander auf der neuen Außenlinie."""
    rot = rotation_system(triangles)
    nodes = sorted(rot)
    n = len(nodes)
    a, b, c = outer
    present = set(nodes)
    cycle = [a, c, b]
    attach: dict[int, tuple[int, ...]] = {}
    removed_order: list[int] = []
    for _ in range(n - 3):
        position = {u: i for i, u in enumerate(cycle)}
        chorded = set()
        for u in cycle:
            for w in rot[u]:
                if w in position and abs(position[u] - position[w]) != 1 and not {u, w} == {cycle[0], cycle[-1]}:
                    chorded.add(u)
        candidates = [u for u in cycle[1:-1] if u not in chorded]
        if not candidates:
            raise ValueError("keine kanonische Ordnung: die Außenlinie hat nur Sehnen")
        v = min(candidates)
        idx = cycle.index(v)
        left, right = cycle[idx - 1], cycle[idx + 1]
        rv = rot[v]
        k = len(rv)
        ia, ib = rv.index(left), rv.index(right)
        forward = [rv[(ia + t) % k] for t in range(1, (ib - ia) % k)]
        backward = [rv[(ia - t) % k] for t in range(1, (ia - ib) % k)]
        if any(x in present and x != v for x in forward) and any(x in present and x != v for x in backward):
            raise ValueError("Nachbarn von v liegen auf beiden Seiten der Außenlinie")
        inner = forward if any(x in present for x in forward) else backward if any(x in present for x in backward) else []
        inner = [x for x in inner if x in present]
        path = [left] + inner + [right]
        attach[v] = tuple(path)
        cycle = cycle[:idx] + path[1:-1] + cycle[idx + 1:]
        present.discard(v)
        removed_order.append(v)
    third = cycle[1]
    order = (a, b, third, *reversed(removed_order))
    return CanonicalOrder(order, attach)


@dataclass(frozen=True)
class GridStep:
    """Ein Einfügeschritt zum Abspielen: der neue Knoten, seine Nachbarn links/rechts auf der Außenlinie und die Lage aller bisherigen Knoten nach dem Schritt."""

    vertex: int
    left: int
    right: int
    shifted_one: tuple[int, ...]
    shifted_two: tuple[int, ...]
    positions: dict[int, tuple[int, int]]


@dataclass(frozen=True)
class GridDrawing:
    pos: tuple[tuple[int, int], ...]             # ganzzahlige Lage je Knoten
    width: int
    height: int
    order: tuple[int, ...]
    chords: tuple[tuple[int, int], ...]          # beim Triangulieren ergänzte Kanten (gehören nicht zum Graphen)
    outer: tuple[int, int, int]
    steps: tuple[GridStep, ...]


def fpp_place(canon: CanonicalOrder, record: bool = False) -> tuple[dict[int, tuple[int, int]], list[GridStep]]:
    """Schiebeverfahren: v1 (0, 0), v2 (2, 0), v3 (1, 1); jeder weitere Knoten sitzt auf der Außenlinie über seinen Nachbarn w_p … w_q (Steigungen ±1).
    Vorher rücken die Knoten (samt der von ihnen verdeckten Mengen L) zwischen w_p und w_q um 1 und alle rechts von w_q um 2 nach rechts."""
    order = canon.order
    pos: dict[int, list[int]] = {order[0]: [0, 0], order[1]: [2, 0], order[2]: [1, 1]}
    cover: dict[int, set[int]] = {order[0]: {order[0]}, order[1]: {order[1]}, order[2]: {order[2]}}
    contour = [order[0], order[2], order[1]]
    steps: list[GridStep] = []
    for v in order[3:]:
        path = canon.attach[v]
        wp, wq = path[0], path[-1]
        ip, iq = contour.index(wp), contour.index(wq)
        middle = contour[ip + 1: iq]
        right_part = contour[iq:]
        moved_one = set().union(*(cover[u] for u in middle)) if middle else set()
        moved_two = set().union(*(cover[u] for u in right_part))
        for u in moved_one:
            pos[u][0] += 1
        for u in moved_two:
            pos[u][0] += 2
        (xp, yp), (xq, yq) = pos[wp], pos[wq]
        twice_x, twice_y = xq + xp + yq - yp, xq - xp + yq + yp
        if twice_x % 2 or twice_y % 2:
            raise ValueError("ungerade Lage: Außenlinie nicht aus Steigungen ±1 aufgebaut")
        pos[v] = [twice_x // 2, twice_y // 2]
        cover[v] = {v} | moved_one
        contour = contour[: ip + 1] + [v] + contour[iq:]
        if record:
            steps.append(GridStep(v, wp, wq, tuple(sorted(moved_one)), tuple(sorted(moved_two)), {u: (p[0], p[1]) for u, p in pos.items()}))
    return {u: (p[0], p[1]) for u, p in pos.items()}, steps


def fpp_drawing(n: int, edges, faces, record: bool = False) -> GridDrawing:
    """Gitterzeichnung eines zweifach zusammenhängenden planaren Graphen (`faces` = Flächen seiner Einbettung): triangulieren, Außendreieck wählen (das zuerst sortierte),
    kanonische Ordnung, Schiebeverfahren. Ergebnis: ganzzahlige Lage im Gitter (2n − 4) × (n − 2), ohne Kreuzungen und mit Mindestabstand 1 (Satz von de Fraysseix, Pach, Pollack)."""
    if n < 3:
        raise ValueError("Gitterzeichnung braucht mindestens drei Knoten")
    triangles, chords = triangulate_faces(faces, edges)
    outer = min(canonical_face(t) for t in triangles)
    canon = canonical_ordering(triangles, outer)
    placed, steps = fpp_place(canon, record)
    pos = tuple(placed[v] for v in range(n))
    return GridDrawing(pos, max(p[0] for p in pos), max(p[1] for p in pos), canon.order, tuple(chords), outer, tuple(steps))
