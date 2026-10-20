"""Jede Einheit des Kerns an einer von Hand gerechneten Mini-Instanz (DEMO-PLAYBOOK §1b)."""

import math
from fractions import Fraction

import numpy as np
import pytest

import gzn_algorithm as A
import gzn_scenario as S

# --- Weglängen -------------------------------------------------------------------------------------------------------------------------------


def test_hop_distances_of_a_path_and_a_disconnected_graph():
    D = A.hop_distances(4, [(0, 1), (1, 2), (2, 3)])
    assert D.tolist() == [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = A.hop_distances(3, [(0, 1)])
    assert D2[0, 1] == 1 and math.isinf(D2[0, 2]) and D2[2, 2] == 0


def test_clean_distances_replaces_infinity_by_largest_plus_one():
    D = np.array([[0, 1, np.inf], [1, 0, np.inf], [np.inf, np.inf, 0]])
    assert A.clean_distances(D).tolist() == [[0, 1, 2], [1, 0, 2], [2, 2, 0]]
    assert A.clean_distances(np.zeros((1, 1))).tolist() == [[0.0]]


# --- Stress ----------------------------------------------------------------------------------------------------------------------------------


def test_stress_is_zero_for_exact_layouts_and_scale_free():
    D = A.hop_distances(3, [(0, 1), (1, 2), (0, 2)])
    equilateral = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    assert A.stress_value(equilateral, D) == pytest.approx(0, abs=1e-12)
    assert A.stress_value(7.5 * equilateral, D) == pytest.approx(0, abs=1e-12)
    path = A.hop_distances(3, [(0, 1), (1, 2)])
    assert A.stress_value(np.array([[0, 0], [2, 0], [4, 0]], dtype=float), path) == pytest.approx(0, abs=1e-12)


def test_stress_of_a_path_drawn_as_an_angle_by_hand():
    """Pfad 0–1–2 mit den Punkten (0,0), (1,0), (1,1): w = 1, 1, 1/4; bester Maßstab s = (2 + √2/2)/2,5; Stress = 1 − (2 + √2/2)²/7,5."""
    D = A.hop_distances(3, [(0, 1), (1, 2)])
    pos = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    assert A.stress_value(pos, D) == pytest.approx(1 - (2 + math.sqrt(2) / 2) ** 2 / 7.5, abs=1e-12)


def test_stress_of_a_collapsed_layout_is_one_and_one_node_is_zero():
    D = A.hop_distances(3, [(0, 1), (1, 2)])
    assert A.stress_value(np.zeros((3, 2)), D) == pytest.approx(1.0)
    assert A.stress_value(np.zeros((1, 2)), np.zeros((1, 1))) == 0.0


def test_classical_mds_reproduces_euclidean_distances_exactly():
    pts = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    D = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    X = A.classical_mds(D)
    assert np.linalg.norm(X[:, None] - X[None, :], axis=2) == pytest.approx(D, abs=1e-9)
    assert A.classical_mds(np.zeros((1, 1))).tolist() == [[0.0, 0.0]]


def test_majorization_step_of_two_nodes_lands_on_distance_one():
    """n = 2, d = 1, Start (0,0), (3,0): L⁺ = ¼·[[1,−1],[−1,1]], B = ⅓·[[1,−1],[−1,1]]  ⇒  X' = (−½, 0), (½, 0)."""
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    W = A._weights(D)
    L = -W.copy()
    np.fill_diagonal(L, W.sum(axis=1))
    X = A.majorization_step(np.array([[0.0, 0.0], [3.0, 0.0]]), D, W, np.linalg.pinv(L))
    assert X == pytest.approx(np.array([[-0.5, 0.0], [0.5, 0.0]]))


def test_majorization_never_increases_the_stress():
    inst = S.make_delaunay(20, 3)
    D = A.hop_distances(inst.n, inst.edges)
    start = A.classical_mds(D)
    trace = A.majorization_trace(D, start, range(0, 21))
    values = [A.stress_value(trace[t], D) for t in range(21)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
    assert values[-1] < values[0]


def test_majorization_trace_matches_the_plain_run_and_zero_steps_return_the_start():
    inst = S.make_delaunay(15, 4)
    D = A.hop_distances(inst.n, inst.edges)
    start = A.classical_mds(D)
    trace = A.majorization_trace(D, start, (0, 3, 10))
    assert trace[0].tolist() == start.tolist()
    assert trace[3] == pytest.approx(A.stress_majorization(D, start, 3))
    assert trace[10] == pytest.approx(A.stress_majorization(D, start, 10))
    assert A.stress_majorization(D, start, 0).tolist() == start.tolist()
    assert A.majorization_trace(D, start, ()) == {}


def test_majorization_of_one_node_is_the_start():
    X = np.array([[1.0, 2.0]])
    assert A.stress_majorization(np.zeros((1, 1)), X, 5).tolist() == [[1.0, 2.0]]
    assert A.majorization_trace(np.zeros((1, 1)), X, (0, 4))[4].tolist() == [[1.0, 2.0]]


def test_sgd_places_two_nodes_at_distance_one_and_is_repeatable():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    X = A.sgd_stress(D, seed=5)
    assert np.linalg.norm(X[0] - X[1]) == pytest.approx(1.0, abs=1e-9)
    assert A.sgd_stress(D, 5).tolist() == X.tolist() and A.sgd_stress(D, 6).tolist() != X.tolist()
    assert A.sgd_stress(np.zeros((1, 1)), 1).shape == (1, 2)


# --- Federmodell -----------------------------------------------------------------------------------------------------------------------------


def test_fr_forces_cancel_at_distance_k_and_pull_together_at_2k():
    k = 0.5
    at_k = A.fr_forces(np.array([[0.0, 0.0], [k, 0.0]]), [(0, 1)], k)
    assert at_k == pytest.approx(np.zeros((2, 2)), abs=1e-12)
    at_2k = A.fr_forces(np.array([[0.0, 0.0], [2 * k, 0.0]]), [(0, 1)], k)       # Abstoßung k/2, Anziehung (2k)²/k = 4k
    assert at_2k[0].tolist() == pytest.approx([3.5 * k, 0.0]) and at_2k[1].tolist() == pytest.approx([-3.5 * k, 0.0])


def test_fr_forces_without_edges_only_repel_and_sum_to_zero():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    f = A.fr_forces(pos, [], 0.3)
    assert f.sum(axis=0) == pytest.approx([0, 0], abs=1e-12)
    assert f[0][0] < 0 and f[1][0] > 0 and f[2][1] > 0           # jeder weg von den anderen


def test_fruchterman_reingold_settles_two_connected_nodes_at_distance_k():
    start = np.array([[0.2, 0.5], [0.8, 0.5]])
    out = A.fruchterman_reingold(start, [(0, 1)], 300)
    assert np.linalg.norm(out[0] - out[1]) == pytest.approx(math.sqrt(1 / 2), rel=0.02)
    assert A.fruchterman_reingold(start, [(0, 1)], 300).tolist() == out.tolist()            # gleicher Start, gleicher Lauf
    assert A.fruchterman_reingold(np.array([[0.3, 0.3]]), [], 10).tolist() == [[0.3, 0.3]]


# --- Tutte -----------------------------------------------------------------------------------------------------------------------------------


def test_rational_circle_is_exact_distinct_and_strictly_convex():
    assert A.rational_circle(4) == [(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)), (Fraction(-1), Fraction(0)), (Fraction(0), Fraction(-1))]
    pts = A.rational_circle(7)
    assert len(set(pts)) == 7 and all(x * x + y * y == 1 for x, y in pts)
    for i in range(7):
        a, b, c = pts[i], pts[(i + 1) % 7], pts[(i + 2) % 7]
        assert (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0         # immer links herum
    assert A.rational_circle(1) == [(Fraction(1), Fraction(0))] and A.rational_circle(2)[1] == (Fraction(-1), Fraction(0))


def test_canonical_face_rotates_to_the_smallest_vertex_and_keeps_the_direction():
    assert A.canonical_face([3, 1, 2]) == (1, 2, 3) and A.canonical_face((5, 4, 0, 9)) == (0, 9, 5, 4)


def test_pick_outer_face_longest_and_shortest_break_ties_by_order():
    faces = [(4, 2, 3), (0, 1, 2, 5), (6, 7, 8), (1, 5, 9, 2)]
    assert A.pick_outer_face(faces, "longest") == (0, 1, 2, 5)          # zwei Vierecke: das zuerst sortierte
    assert A.pick_outer_face(faces, "shortest") == (2, 3, 4)            # drei Dreiecke: (2,3,4) < (6,7,8)


def test_tutte_k4_puts_the_inner_vertex_at_the_centroid_exactly():
    pos = A.tutte_embedding(4, S.hand_graph("K4").edges, (0, 1, 2))
    circle = A.rational_circle(3)
    assert pos[:3] == circle
    assert pos[3] == (sum(p[0] for p in circle) / 3, sum(p[1] for p in circle) / 3)


def test_tutte_wheel_hub_is_the_centroid_of_the_rim():
    pos = A.tutte_embedding(7, S.hand_graph("Rad").edges, (1, 2, 3, 4, 5, 6))
    rim = A.rational_circle(6)
    assert pos[0] == (sum(p[0] for p in rim) / 6, sum(p[1] for p in rim) / 6)
    assert abs(float(pos[0][0])) < 1e-5 and abs(float(pos[0][1])) < 1e-5


def test_tutte_prism_solves_the_three_equations_by_hand():
    """Zwiebel mit k = 2 (Prisma), Außendreieck 0, 1, 2: 4·x₃ = p₀ + S mit S = p₀ + p₁ + p₂, entsprechend für 4 und 5."""
    inst = S.make_onion(6)
    pos = A.tutte_embedding(6, inst.edges, (0, 1, 2))
    circle = A.rational_circle(3)
    sx, sy = sum(p[0] for p in circle), sum(p[1] for p in circle)
    for j in range(3):
        assert pos[3 + j] == ((circle[j][0] + sx) / 4, (circle[j][1] + sy) / 4)


# --- Triangulieren, Drehsystem, kanonische Ordnung -------------------------------------------------------------------------------------------


def test_triangulate_a_square_face_with_the_first_free_chord():
    tris, added = A.triangulate_faces([(0, 1, 2, 3)], [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert tris == [(0, 1, 2), (2, 3, 0)] and added == [(0, 2)]


def test_triangulate_takes_the_other_diagonal_when_the_first_exists():
    tris, added = A.triangulate_faces([(0, 1, 2, 3)], [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)])
    assert tris == [(1, 2, 3), (3, 0, 1)] and added == [(1, 3)]


def test_triangulate_a_pentagon_gives_three_triangles_and_two_chords_with_consistent_direction():
    edges = [(i, (i + 1) % 5) if i < 4 else (0, 4) for i in range(5)]
    tris, added = A.triangulate_faces([(0, 1, 2, 3, 4)], edges)
    assert len(tris) == 3 and len(added) == 2 and all(len(t) == 3 for t in tris)
    directed = [(t[i], t[(i + 1) % 3]) for t in tris for i in range(3)]
    inner = [e for e in directed if (min(e), max(e)) in {(min(a), max(a)) for a in added}]
    assert len(inner) == 4 and len({(min(a, b), max(a, b)) for a, b in inner}) == 2 and len(set(inner)) == 4       # jede Sehne einmal in jeder Richtung


def test_triangulate_leaves_triangles_alone_and_fails_without_a_free_chord():
    assert A.triangulate_faces([(0, 1, 2)], [(0, 1), (1, 2), (0, 2)]) == ([(0, 1, 2)], [])
    k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    with pytest.raises(ValueError):
        A.triangulate_faces([(0, 1, 2, 3)], k4)


def test_rotation_system_of_k4(k4_faces):
    rot = A.rotation_system(k4_faces)
    assert rot[1] == [0, 3, 2] and rot[3] == [0, 2, 1]
    assert {v: sorted(r) for v, r in rot.items()} == {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}


def test_canonical_ordering_of_k4_by_hand(k4_faces):
    canon = A.canonical_ordering(k4_faces, (0, 2, 1))
    assert canon.order == (0, 2, 3, 1) and canon.attach == {1: (0, 3, 2)}


def test_canonical_ordering_of_a_triangle_is_the_triangle():
    canon = A.canonical_ordering([(0, 1, 2), (0, 2, 1)], (0, 1, 2))
    assert canon.order == (0, 1, 2) and canon.attach == {}


def test_fpp_place_of_k4_by_hand(k4_faces):
    canon = A.canonical_ordering(k4_faces, (0, 2, 1))
    pos, steps = A.fpp_place(canon, record=True)
    assert pos == {0: (0, 0), 2: (4, 0), 3: (2, 1), 1: (2, 2)}
    assert len(steps) == 1 and steps[0].vertex == 1 and (steps[0].left, steps[0].right) == (0, 2)
    assert steps[0].shifted_one == (3,) and steps[0].shifted_two == (2,) and steps[0].positions == pos


def test_fpp_place_without_recording_keeps_no_steps(k4_faces):
    assert A.fpp_place(A.canonical_ordering(k4_faces, (0, 2, 1)))[1] == []


def test_fpp_drawing_of_k4_uses_the_smallest_triangle_as_outside(k4_faces):
    g = A.fpp_drawing(4, S.hand_graph("K4").edges, k4_faces)
    assert g.outer == (0, 1, 3) and g.order == (0, 1, 2, 3) and g.chords == ()
    assert g.pos == ((0, 0), (4, 0), (2, 1), (2, 2)) and (g.width, g.height) == (4, 2)


def test_fpp_drawing_of_a_triangle_is_the_basic_triangle():
    g = A.fpp_drawing(3, [(0, 1), (0, 2), (1, 2)], [(0, 1, 2), (0, 2, 1)])
    assert g.pos == ((0, 0), (2, 0), (1, 1)) and (g.width, g.height) == (2, 1)


def test_fpp_drawing_needs_three_vertices():
    with pytest.raises(ValueError):
        A.fpp_drawing(2, [(0, 1)], [])


# --- Nachträge aus dem Mutationstest -----------------------------------------------------------------------------------------------------------


def test_fruchterman_reingold_one_step_moves_two_nodes_by_the_temperature():
    """Start (0,2; 0,5) und (0,8; 0,5), k = √½: Abstoßung k²/0,6 = 0,833, Anziehung 0,6²/k = 0,509, Rest 0,324 nach außen; größer als die Temperatur 0,1, also rückt jeder um 0,1."""
    out = A.fruchterman_reingold(np.array([[0.2, 0.5], [0.8, 0.5]]), [(0, 1)], 1)
    assert out == pytest.approx(np.array([[0.1, 0.5], [0.9, 0.5]]))
    assert A.fruchterman_reingold(np.array([[0.2, 0.5], [0.8, 0.5]]), [(0, 1)], 0).tolist() == [[0.2, 0.5], [0.8, 0.5]]


def test_majorization_of_two_nodes_by_hand_for_one_step_run_and_trace():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    start = np.array([[0.0, 0.0], [3.0, 0.0]])
    expected = np.array([[-0.5, 0.0], [0.5, 0.0]])
    assert A.stress_majorization(D, start, 1) == pytest.approx(expected)
    assert A.majorization_trace(D, start, (0, 1))[1] == pytest.approx(expected)
    assert A.majorization_trace(D, start, (0, 1))[0].tolist() == start.tolist()


def test_one_majorization_step_changes_the_start_and_matches_the_trace():
    inst = S.make_delaunay(15, 4)
    D = A.hop_distances(inst.n, inst.edges)
    start = A.classical_mds(D)
    one = A.stress_majorization(D, start, 1)
    assert not np.allclose(one, start) and one == pytest.approx(A.majorization_trace(D, start, (1,))[1])


def test_fpp_drawing_names_the_missing_vertices_and_the_shifted_sets_are_sorted():
    with pytest.raises(ValueError, match="mindestens drei"):
        A.fpp_drawing(2, [(0, 1)], [])
    inst = S.make_delaunay(20, 1)
    from gzn_planar import embed

    g = A.fpp_drawing(inst.n, inst.edges, embed(inst.n, inst.edges).single_block.faces, record=True)
    assert all(list(s.shifted_one) == sorted(s.shifted_one) and list(s.shifted_two) == sorted(s.shifted_two) for s in g.steps)
    assert any(len(s.shifted_one) >= 2 for s in g.steps) and any(len(s.shifted_two) >= 2 for s in g.steps)


def test_canonical_ordering_follows_the_arc_on_either_side_of_the_vertex(k4_faces):
    """Wird das Außendreieck gegen die Richtung der Flächen übergeben, liegt der innere Nachbar auf dem Rückwärtsbogen (rot[2] = [0, 1, 3]: von 0 nach 1 vorwärts nichts, rückwärts die 3)."""
    canon = A.canonical_ordering(k4_faces, (0, 1, 2))
    assert A.rotation_system(k4_faces)[2] == [0, 1, 3]
    assert canon.order == (0, 1, 3, 2) and canon.attach == {2: (0, 3, 1)}


def test_canonical_ordering_rejects_an_outer_triangle_that_is_not_a_face():
    """Stapel: K4 mit der 3 in (0, 1, 2), die 4 in (0, 1, 3). (0, 1, 3) trennt die 4 von der 2: die Nachbarn der 3 liegen auf beiden Seiten der Außenlinie."""
    from gzn_planar import embed

    edges = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (3, 4)]
    faces, _ = A.triangulate_faces(embed(5, edges).single_block.faces, edges)
    with pytest.raises(ValueError, match="beiden Seiten"):
        A.canonical_ordering(faces, (0, 3, 1))


def test_stress_of_an_exact_layout_is_zero_at_any_scale_including_tiny_ones():
    D = A.hop_distances(3, [(0, 1), (1, 2), (0, 2)])
    tiny = 0.001 * np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    assert A.stress_value(tiny, D) == pytest.approx(0, abs=1e-12)
