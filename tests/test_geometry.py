"""Gütemaße einer Zeichnung an Handwerten; Kreuzungszähler gegen die einfache Doppelschleife; Exaktheit bei Fraction."""

import math
import random
from fractions import Fraction

import numpy as np
import pytest

import gzn_geometry as G

SQUARE = [(0, 0), (2, 0), (2, 2), (0, 2)]


def test_orientation_signs():
    assert G.orientation((0, 0), (1, 0), (0, 1)) == 1 and G.orientation((0, 0), (1, 0), (0, -1)) == -1 and G.orientation((0, 0), (1, 0), (2, 0)) == 0


def test_orientation_is_exact_for_fractions_where_floats_see_nothing():
    c = (Fraction(1, 3), Fraction(1, 3) + Fraction(1, 10 ** 30))
    assert G.orientation((Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)), c) == 1
    assert G.orientation((0.0, 0.0), (1.0, 1.0), (float(c[0]), float(c[1]))) == 0


def test_segments_cross_cases():
    assert G.segments_cross((0, 0), (2, 2), (0, 2), (2, 0))                   # X
    assert not G.segments_cross((0, 0), (2, 0), (1, 0), (1, 2))               # T: Berührung zählt nicht
    assert not G.segments_cross((0, 0), (1, 1), (1, 1), (2, 0))               # gemeinsamer Endpunkt
    assert not G.segments_cross((0, 0), (2, 0), (1, 0), (3, 0))               # Überlagerung
    assert not G.segments_cross((0, 0), (1, 0), (0, 1), (1, 1))               # parallel
    assert not G.segments_cross((0, 0), (1, 0), (2, -1), (2, 1))              # Geraden schneiden sich, Strecken nicht


def test_tiny_gap_is_decided_exactly():
    eps = Fraction(1, 10 ** 30)
    a, b = (Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))
    assert not G.segments_cross(a, b, (Fraction(1, 2), eps), (Fraction(1, 2), Fraction(1)))
    assert G.segments_cross(a, b, (Fraction(1, 2), -eps), (Fraction(1, 2), Fraction(1)))


def test_crossings_of_a_square_with_both_diagonals():
    edges = [(0, 2), (1, 3), (0, 1)]
    assert G.crossings(SQUARE, edges) == [(0, 1)]
    assert G.crossings(SQUARE, [(0, 1), (1, 2), (2, 3), (0, 3)]) == []
    assert G.crossings([], []) == []


def test_crossings_match_the_plain_double_loop_on_random_drawings():
    rng = random.Random(3)
    for _ in range(60):
        pos = [(rng.randint(0, 12), rng.randint(0, 12)) for _ in range(9)]
        edges = sorted({tuple(sorted(rng.sample(range(9), 2))) for _ in range(14)})
        brute = [(i, j) for i in range(len(edges)) for j in range(i + 1, len(edges))
                 if len(set(edges[i]) | set(edges[j])) == 4 and G.segments_cross(pos[edges[i][0]], pos[edges[i][1]], pos[edges[j][0]], pos[edges[j][1]])]
        assert G.crossings(pos, edges) == brute


def test_touching_finds_coincident_nodes_and_nodes_on_foreign_edges():
    assert G.touching([(0, 0), (0, 0), (1, 1)], [(0, 2)]) == {"coincident": [(0, 1)], "on_edge": []}
    assert G.touching([(0, 0), (2, 2), (1, 1)], [(0, 1)]) == {"coincident": [], "on_edge": [(2, 0)]}
    assert G.touching([(0, 0), (2, 2), (0, 0)], [(0, 1)]) == {"coincident": [(0, 2)], "on_edge": []}       # auf dem Endpunkt: nur als zusammengefallen gezählt
    assert G.touching([(0, 0), (2, 2), (3, 3)], [(0, 1)]) == {"coincident": [], "on_edge": []}             # auf der Geraden, aber außerhalb
    assert G.touching([(0, 0), (2, 2), (1, 2)], [(0, 1)]) == {"coincident": [], "on_edge": []}             # daneben
    assert G.touching([(0, 0), (2, 2), (1, 1)], [(0, 1), (0, 2)])["on_edge"] == [(2, 0)]                   # (0,2) endet in dem Knoten: keine Berührung


def test_to_float_and_distances():
    assert G.to_float([]).shape == (0, 2)
    P = G.to_float([(Fraction(1, 2), 1), (0, 0.25)])
    assert P.tolist() == [[0.5, 1.0], [0.0, 0.25]]
    S = G.to_float(SQUARE)
    assert G.min_distance(S) == 2.0 and G.diagonal(S) == pytest.approx(2 * math.sqrt(2)) and G.resolution(S) == pytest.approx(1 / math.sqrt(2))
    assert G.min_distance(S[:1]) == 0.0 and G.diagonal(np.zeros((0, 2))) == 0.0
    assert G.resolution(np.zeros((3, 2))) == 0.0


def test_area_at_unit_distance_is_scale_free_and_handles_lines_and_collisions():
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert G.area_at_unit_distance(unit) == pytest.approx(1.0) and G.area_at_unit_distance(3 * unit) == pytest.approx(1.0)
    line = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    assert G.area_at_unit_distance(line) == pytest.approx(2.0)
    assert G.area_at_unit_distance(np.array([[0, 0], [4, 0], [4, 1], [0, 1]], dtype=float)) == pytest.approx(4.0)
    assert math.isinf(G.area_at_unit_distance(np.array([[0, 0], [0, 0], [1, 1]], dtype=float)))


def test_edge_ratio():
    P = np.array([[0, 0], [1, 0], [1, 3]], dtype=float)
    assert G.edge_ratio(P, [(0, 1), (1, 2)]) == pytest.approx(3.0)
    assert math.isinf(G.edge_ratio(np.array([[0, 0], [0, 0]], dtype=float), [(0, 1)]))
    assert G.edge_ratio(P, []) == 1.0


def test_min_angle_cases():
    corner = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    assert G.min_angle(corner, [(0, 1), (0, 2)]) == pytest.approx(90.0)
    star = np.array([[0, 0], [1, 0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]])
    assert G.min_angle(star, [(0, 1), (0, 2), (0, 3)]) == pytest.approx(120.0)
    assert G.min_angle(np.array([[-1, 0], [0, 0], [1, 0]], dtype=float), [(0, 1), (1, 2)]) == pytest.approx(180.0)
    assert G.min_angle(corner, [(0, 1)]) == 180.0


def test_fidelity_is_zero_for_a_rotated_scaled_copy_and_none_for_collapsed_points():
    base = np.array([[0, 0], [2, 0], [2, 1], [0, 1.5]], dtype=float)
    theta = 0.7
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    copy = 3.0 * base @ rot.T + np.array([5.0, -2.0])
    assert G.fidelity(copy, base) == pytest.approx(0.0, abs=1e-12)
    mirrored = base * np.array([1.0, -1.0])
    assert G.fidelity(base, np.array([[0, 0], [1, 0], [1, 1], [0, 3]], dtype=float)) > 0.001
    assert G.fidelity(mirrored, base) is not None
    assert G.fidelity(np.zeros((3, 2)), base[:3]) is None


def test_measure_of_k4_in_a_square_with_both_diagonals():
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]
    m = G.measure(SQUARE, edges)
    assert m.crossings == 1 and m.touching == 0 and not m.valid
    assert m.resolution == pytest.approx(1 / math.sqrt(2)) and m.area == pytest.approx(1.0) and m.edge_ratio == pytest.approx(math.sqrt(2))
    assert m.min_angle == pytest.approx(45.0) and m.stress is None and m.fidelity is None
    planar = G.measure([(0, 0), (4, 0), (2, 3), (2, 1)], [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)], stress=0.5, true=np.array([[0, 0], [4, 0], [2, 3], [2, 1]], dtype=float))
    assert planar.valid and planar.stress == 0.5 and planar.fidelity == pytest.approx(0.0, abs=1e-12)


def test_measure_counts_a_collision_as_invalid_even_without_crossings():
    m = G.measure([(0, 0), (0, 0), (1, 1)], [(0, 2), (1, 2)])
    assert m.crossings == 0 and m.touching == 1 and not m.valid and m.resolution == 0.0


def test_measure_takes_given_crossing_pairs_instead_of_counting():
    assert G.measure(SQUARE, [(0, 1), (1, 2)], pairs=[(0, 1), (0, 1)]).crossings == 2


# --- Nachträge aus dem Mutationstest -----------------------------------------------------------------------------------------------------------


def test_axis_parallel_edges_are_found_by_the_prefilter():
    assert G.crossings([(0, 1), (2, 1), (1, 0), (1, 2)], [(0, 1), (2, 3)]) == [(0, 1)]            # waagerecht gegen senkrecht: die Rechtecke sind Linien
    assert G.touching([(0, 0), (2, 0), (1, 0)], [(0, 1)])["on_edge"] == [(2, 0)]
    assert G.touching([(0, 0), (0, 2), (0, 1)], [(0, 1)])["on_edge"] == [(2, 0)]


def test_area_is_translation_invariant_and_distances_of_two_points():
    assert G.area_at_unit_distance(np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)) == pytest.approx(1.0)
    assert G.min_distance(np.array([[0, 0], [3, 4]], dtype=float)) == pytest.approx(5.0)
    assert G.resolution(np.array([[0, 0], [0.5, 0.5]], dtype=float)) == pytest.approx(1.0)           # kleine Zeichnung: Diagonale unter 1


def test_min_angle_is_found_at_the_last_vertex():
    P = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    assert G.min_angle(P, [(2, 0), (2, 1)]) == pytest.approx(90.0)


def test_min_angle_uses_the_gap_across_zero_degrees():
    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
    P = np.array([[0, 0], [c, s], [c, -s]], dtype=float)
    assert G.min_angle(P, [(0, 1), (0, 2)]) == pytest.approx(20.0)
