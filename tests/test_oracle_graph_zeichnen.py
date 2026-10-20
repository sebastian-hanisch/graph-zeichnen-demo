"""Orakel-Tests graph-zeichnen-demo (ORACLE-PLAYBOOK): ein anderer Rechenweg als der der Demo.

- Tutte: Schwerpunkt-Eigenschaft exakt in Fraction, Lösung gegen ein float-Gleichungssystem (numpy), Kreuzungsfreiheit gegen eine einfache Doppelschleife in ganzen Zahlen (Satz von Tutte als Eigenschaftstest).
- Gitterzeichnung: ganzzahlig, Breite 2n − 4, Höhe höchstens n − 2, Mindestabstand 1, keine Kreuzung (Doppelschleife); dieselben Schranken gelten für `networkx.combinatorial_embedding_to_pos` (zweite Umsetzung).
- Kanonische Ordnung: die Bedingungen der Definition an networkx (zweifach zusammenhängend je Anfangsstück).
- Stress: gegen `networkx.kamada_kawai_layout` (gleiche Zielfunktion mit w = 1/d², scipy-Optimierer), Gradient am Ergebnis per finiter Differenz; Federmodell-Kräfte gegen den Gradienten ihrer Energie.
- Weglängen gegen networkx."""

import itertools
import math
import random
from fractions import Fraction

import numpy as np
import pytest

import gzn_algorithm as A
import gzn_geometry as G
import gzn_planar as Pl
import gzn_scenario as S

nx = pytest.importorskip("networkx")


def _naive_crossings(pos, edges):
    """Doppelschleife mit ganzzahligen/rationalen Vorzeichentests, unabhängig von gzn_geometry.crossings."""
    def side(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 0) - (v < 0)

    count = 0
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) < 4:
            continue
        if side(pos[a], pos[b], pos[c]) * side(pos[a], pos[b], pos[d]) < 0 and side(pos[c], pos[d], pos[a]) * side(pos[c], pos[d], pos[b]) < 0:
            count += 1
    return count


def _triconnected_samples():
    out = []
    for n in (8, 12, 20, 35):
        for seed in range(5):
            inst = S.make_delaunay(n, seed)
            if Pl.is_triconnected(inst.n, inst.edges):
                out.append(inst)
    out += [S.make_stacked(n, seed) for n in (6, 10, 25) for seed in range(3)]
    out += [S.make_onion(3 * k) for k in range(2, 7)]
    out += [S.hand_graph("Rad"), S.hand_graph("Würfel"), S.hand_graph("K4")]
    return out


# --- Tutte -----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["longest", "shortest"])
def test_tutte_inner_vertices_sit_exactly_at_the_mean_of_their_neighbours_and_nothing_crosses(which):
    samples = _triconnected_samples()
    assert len(samples) >= 25
    for inst in samples:
        block = Pl.embed(inst.n, inst.edges).single_block
        outer = A.pick_outer_face(block.faces, which)
        pos = A.tutte_embedding(inst.n, inst.edges, outer)
        adj = Pl.adjacency(inst.n, inst.edges)
        for v in range(inst.n):
            if v in outer:
                continue
            nb = sorted(adj[v])
            assert pos[v] == (sum(pos[w][0] for w in nb) / len(nb), sum(pos[w][1] for w in nb) / len(nb)), (inst.name, inst.n, v)
        assert _naive_crossings(pos, inst.edges) == 0 and len(set(pos)) == inst.n                         # Satz von Tutte
        t = G.touching(pos, inst.edges)
        assert not t["coincident"] and not t["on_edge"]


def test_tutte_matches_a_float_linear_solve():
    for inst in _triconnected_samples()[:12]:
        block = Pl.embed(inst.n, inst.edges).single_block
        outer = A.pick_outer_face(block.faces, "longest")
        exact = A.tutte_embedding(inst.n, inst.edges, outer)
        fixed = {v: (float(p[0]), float(p[1])) for v, p in zip(outer, A.rational_circle(len(outer)))}
        inner = [v for v in range(inst.n) if v not in fixed]
        index = {v: i for i, v in enumerate(inner)}
        adj = Pl.adjacency(inst.n, inst.edges)
        M = np.zeros((len(inner), len(inner)))
        rhs = np.zeros((len(inner), 2))
        for v in inner:
            M[index[v], index[v]] = len(adj[v])
            for w in adj[v]:
                if w in fixed:
                    rhs[index[v]] += fixed[w]
                else:
                    M[index[v], index[w]] -= 1
        sol = np.linalg.solve(M, rhs) if inner else np.zeros((0, 2))
        for v in inner:
            assert float(exact[v][0]) == pytest.approx(sol[index[v]][0], abs=1e-9) and float(exact[v][1]) == pytest.approx(sol[index[v]][1], abs=1e-9)


def test_tutte_without_triconnectivity_can_collapse_vertices():
    """Gegenstück zum Satz: eine Karte, der Kanten fehlen, ist nicht mehr 3-fach zusammenhängend; Tutte setzt dann Knoten aufeinander (der Kreuzungszähler allein sähe nichts)."""
    collapsed = 0
    for seed in range(100, 112):
        inst = S.make_thin(60, seed)
        block = Pl.embed(inst.n, inst.edges).single_block
        pos = A.tutte_embedding(inst.n, inst.edges, A.pick_outer_face(block.faces, "longest"))
        t = G.touching(pos, inst.edges)
        if t["coincident"] or t["on_edge"]:
            collapsed += 1
            assert not Pl.is_triconnected(inst.n, inst.edges)
    assert collapsed >= 1


# --- Gitterzeichnung -------------------------------------------------------------------------------------------------------------------------


def _grid_samples():
    out = [S.make_delaunay(n, seed) for n in (6, 9, 15, 30, 60) for seed in range(4)]
    out += [S.make_stacked(n, seed) for n in (5, 12, 40) for seed in range(3)]
    out += [S.make_grid(k * k, d) for k in (3, 4, 6) for d in (True, False)]
    out += [S.make_thin(n, seed) for n in (25, 40) for seed in range(4)]
    out += [S.hand_graph("Rad"), S.hand_graph("Würfel"), S.hand_graph("K4"), S.make_onion(15)]
    return out


def test_grid_drawing_is_integer_crossing_free_and_within_the_bound():
    samples = _grid_samples()
    assert len(samples) >= 40
    for inst in samples:
        block = Pl.embed(inst.n, inst.edges).single_block
        g = A.fpp_drawing(inst.n, inst.edges, block.faces)
        assert all(isinstance(c, int) for p in g.pos for c in p)
        assert g.width == 2 * inst.n - 4 and 1 <= g.height <= inst.n - 2, (inst.name, inst.n, g.width, g.height)
        assert min(min(p) for p in g.pos) >= 0
        assert _naive_crossings(g.pos, inst.edges) == 0, (inst.name, inst.n)
        assert len(set(g.pos)) == inst.n
        assert min(math.dist(p, q) for p, q in itertools.combinations(g.pos, 2)) >= 1.0
        t = G.touching(g.pos, inst.edges)
        assert not t["coincident"] and not t["on_edge"]


def test_networkx_fpp_satisfies_the_same_bounds_on_the_same_graphs():
    """Die zweite Umsetzung (networkx) hält dieselben Schranken ein: die Invarianten sind also die des Satzes, nicht Eigenheiten unserer Umsetzung."""
    for inst in _grid_samples()[:20]:
        g = nx.Graph(list(inst.edges))
        ok, emb = nx.check_planarity(g)
        pos = nx.combinatorial_embedding_to_pos(emb, fully_triangulate=True)
        xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
        assert max(xs) - min(xs) <= 2 * inst.n - 4 and max(ys) - min(ys) <= inst.n - 2
        assert _naive_crossings({v: pos[v] for v in pos}, inst.edges) == 0


def test_canonical_ordering_satisfies_the_definition():
    """Für beide Richtungen des Außendreiecks (die Bögen am Knoten liegen dann vorwärts bzw. rückwärts im Drehsystem)."""
    for inst in _grid_samples()[:30]:
        block = Pl.embed(inst.n, inst.edges).single_block
        triangles, chords = A.triangulate_faces(block.faces, inst.edges)
        edges = set(inst.edges) | {(min(a, b), max(a, b)) for a, b in chords}
        assert len(edges) == 3 * inst.n - 6
        full = nx.Graph(list(edges))
        first = min(A.canonical_face(t) for t in triangles)
        for outer in (first, (first[0], first[2], first[1])):
            canon = A.canonical_ordering(triangles, outer)
            order = canon.order
            assert sorted(order) == list(range(inst.n)) and order[:2] == outer[:2] and order[-1] == outer[2]
            assert full.has_edge(order[0], order[1])
            rank = {v: i for i, v in enumerate(order)}
            for k in range(3, inst.n + 1):
                assert nx.is_biconnected(full.subgraph(order[:k]))
            for k in range(3, inst.n):
                v = order[k]
                earlier = sorted(rank[w] for w in full[v] if rank[w] < k)
                assert len(earlier) >= 2 and set(order[i] for i in earlier) == set(canon.attach[v])
            for k in range(2, inst.n - 1):
                assert any(rank[w] > k for w in full[order[k]]), "jeder Knoten außer dem letzten hat einen späteren Nachbarn"
            g = A.fpp_place(canon)[0]
            assert _naive_crossings(g, inst.edges) == 0


# --- Stress ----------------------------------------------------------------------------------------------------------------------------------


def test_stress_majorization_is_at_least_as_good_as_kamada_kawai():
    for seed in range(4):
        inst = S.make_delaunay(14, seed)
        D = A.hop_distances(inst.n, inst.edges)
        ours = A.stress_value(A.stress_majorization(D, A.classical_mds(D), 300), D)
        kk = nx.kamada_kawai_layout(nx.Graph(list(inst.edges)))
        theirs = A.stress_value(np.array([kk[v] for v in range(inst.n)]), D)
        assert ours <= theirs * 1.05 + 1e-9, (seed, ours, theirs)


def test_raw_stress_gradient_vanishes_at_the_majorization_result():
    inst = S.make_delaunay(10, 2)
    D = A.hop_distances(inst.n, inst.edges)
    X = A.stress_majorization(D, A.classical_mds(D), 3000)
    W = 1.0 / np.where(D > 0, D, 1.0) ** 2
    np.fill_diagonal(W, 0.0)

    def raw(P):
        dist = np.linalg.norm(P[:, None] - P[None, :], axis=2)
        return float((W * (dist - D) ** 2).sum() / 2)

    h = 1e-6
    grad = np.zeros_like(X)
    for i in range(inst.n):
        for c in range(2):
            up, down = X.copy(), X.copy()
            up[i, c] += h
            down[i, c] -= h
            grad[i, c] = (raw(up) - raw(down)) / (2 * h)
    assert np.abs(grad).max() < 1e-4


def test_sgd_reaches_the_stress_of_the_majorization():
    for seed in range(3):
        inst = S.make_delaunay(30, seed)
        D = A.hop_distances(inst.n, inst.edges)
        maj = A.stress_value(A.stress_majorization(D, A.classical_mds(D), 300), D)
        sgd = A.stress_value(A.sgd_stress(D, seed), D)
        assert sgd == pytest.approx(maj, rel=0.1), (seed, sgd, maj)


def test_fr_forces_are_minus_the_gradient_of_their_energy():
    """Energie E = −k² Σ ln d_ij (Abstoßung) + Σ_Kanten d³/(3k) (Anziehung); die Kraft auf i ist −∂E/∂p_i."""
    rng = np.random.default_rng(5)
    pos = rng.random((6, 2))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
    k = 0.4

    def energy(P):
        e = 0.0
        for i, j in itertools.combinations(range(6), 2):
            e -= k * k * math.log(np.linalg.norm(P[i] - P[j]))
        for i, j in edges:
            e += np.linalg.norm(P[i] - P[j]) ** 3 / (3 * k)
        return e

    force = A.fr_forces(pos, edges, k)
    h = 1e-6
    for i in range(6):
        for c in range(2):
            up, down = pos.copy(), pos.copy()
            up[i, c] += h
            down[i, c] -= h
            assert force[i, c] == pytest.approx(-(energy(up) - energy(down)) / (2 * h), abs=1e-5)


# --- Weglängen und MDS -----------------------------------------------------------------------------------------------------------------------


def test_hop_distances_match_networkx():
    rng = random.Random(4)
    for _ in range(25):
        g = nx.gnp_random_graph(rng.randint(5, 14), 0.3, seed=rng.randrange(10 ** 6))
        n = g.number_of_nodes()
        D = A.hop_distances(n, [(min(u, v), max(u, v)) for u, v in g.edges])
        lengths = dict(nx.all_pairs_shortest_path_length(g))
        for i in range(n):
            for j in range(n):
                assert D[i, j] == (lengths[i][j] if j in lengths[i] else np.inf)


def test_classical_mds_recovers_random_planar_points():
    rng = np.random.default_rng(6)
    pts = rng.random((10, 2))
    D = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    X = A.classical_mds(D)
    assert np.linalg.norm(X[:, None] - X[None, :], axis=2) == pytest.approx(D, abs=1e-9)


def test_exact_arithmetic_is_what_decides_in_the_onion():
    """Die Zwiebel mit 12 Stufen liegt im Bereich 10⁻⁸: Kreuzungsfreiheit gilt exakt, die kleinsten Abstände sind aber nicht null."""
    inst = S.make_onion(36)
    block = Pl.embed(inst.n, inst.edges).single_block
    pos = A.tutte_embedding(inst.n, inst.edges, A.pick_outer_face(block.faces, "longest"))
    assert _naive_crossings(pos, inst.edges) == 0
    dmin = min(math.hypot(float(p[0] - q[0]), float(p[1] - q[1])) for p, q in itertools.combinations(pos, 2))
    assert 0 < dmin < 1e-6 and isinstance(pos[0][0], Fraction)
