"""Randfälle (DEMO-PLAYBOOK §12): kleinste Größen, nicht planare und nicht zweifach zusammenhängende Graphen, getrennte Graphen, inklusive Grenzen."""

import numpy as np
import pytest

import gzn_algorithm as A
import gzn_constants as C
import gzn_evaluation as ev
import gzn_scenario as S


def _instance(n, edges, name="Hand"):
    return S.Instance(n, S.normalize_edges(edges), None, name)


K5 = _instance(5, [(i, j) for i in range(5) for j in range(i + 1, 5)], "K5")
PATH = _instance(5, [(0, 1), (1, 2), (2, 3), (3, 4)], "Pfad")
TWO_TRIANGLES = _instance(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], "Zwei Dreiecke")


def _draw(inst, method, **kw):
    return ev.draw_with(ev.prepare(inst), ev.Settings("delaunay", inst.n, 101, method, **kw))


@pytest.mark.parametrize("net", list(C.NET_TYPES))
def test_every_net_at_the_smallest_size_works_with_every_method(net):
    for method in C.METHODS:
        a = ev.analyse(ev.Settings(net, C.N_MIN, 101, method))
        assert (a.measures is None) == (a.note is not None)
        assert isinstance(ev.verdict_drawing(a)[1], str)


def test_non_planar_graph_gets_notes_for_tutte_and_grid_but_is_drawn_by_the_others():
    for method in ("tutte", "grid"):
        a = _draw(K5, method)
        assert a.measures is None and "nicht planar" in a.note
        assert ev.verdict_drawing(a) == ("info", a.note)
    fr = _draw(K5, "fr")
    assert fr.measures is not None and not fr.prep.facts.planar and fr.measures.crossings >= 1


def test_planar_graph_that_is_not_biconnected_gets_the_second_note():
    prep = ev.prepare(PATH)
    assert prep.facts.planar and not prep.facts.biconnected and not prep.facts.triconnected and prep.facts.faces is None
    for method in ("tutte", "grid"):
        a = _draw(PATH, method)
        assert a.measures is None and "nicht zweifach zusammenhängend" in a.note


def test_disconnected_graphs_are_drawn_by_the_distance_free_and_the_stress_methods():
    assert not ev.prepare(TWO_TRIANGLES).facts.biconnected
    for method in ("fr", "stress", "sgd"):
        a = _draw(TWO_TRIANGLES, method)
        assert a.measures is not None and np.isfinite(a.measures.stress)
    assert np.isinf(ev.prepare(TWO_TRIANGLES).D[0, 3])


def test_sgd_is_limited_to_the_documented_size():
    a = ev.analyse(ev.Settings("delaunay", C.SGD_MAX_N + 1, 101, "sgd"))
    assert a.measures is None and str(C.SGD_MAX_N) in a.note
    assert ev.analyse(ev.Settings("delaunay", C.SGD_MAX_N, 101, "sgd")).measures is not None


def test_triangle_is_drawn_by_tutte_as_its_outer_circle_and_by_the_grid_as_the_basic_triangle():
    tri = _instance(3, [(0, 1), (1, 2), (0, 2)], "Dreieck")
    t = _draw(tri, "tutte")
    assert t.pos == tuple(A.rational_circle(3)) and t.measures.valid
    g = _draw(tri, "grid")
    assert g.pos == ((0, 0), (2, 0), (1, 1)) and g.measures.valid


def test_outer_face_choice_changes_the_drawing_on_a_map():
    inst = S.make_delaunay(40, 101)
    longest = ev.draw_with(ev.prepare(inst), ev.Settings("delaunay", 40, 101, "tutte", outer="longest"))
    shortest = ev.draw_with(ev.prepare(inst), ev.Settings("delaunay", 40, 101, "tutte", outer="shortest"))
    assert len(longest.outer) > len(shortest.outer) == 3
    assert longest.measures.resolution > 3 * shortest.measures.resolution


def test_one_iteration_and_the_largest_size_do_not_break_fr_and_stress():
    for method in ("fr", "stress"):
        a = ev.analyse(ev.Settings("delaunay", 20, 101, method, iterations=1))
        assert a.measures is not None
    a = ev.analyse(ev.Settings("delaunay", C.N_MAX, 101, "tutte"))
    assert a.measures.valid and a.prep.facts.triconnected


def test_readable_grade_boundaries_are_inclusive():
    assert ev.readable_grade(C.READABLE_GOOD) == "gut" and ev.readable_grade(C.READABLE_GOOD - 1e-9) == "eng"
    assert ev.readable_grade(C.READABLE_TIGHT) == "eng" and ev.readable_grade(C.READABLE_TIGHT - 1e-9) == "kaum"
    assert ev.readable_grade(0.0) == "kaum" and ev.readable_grade(1.0) == "gut"


def test_reference_resolution_of_tiny_graphs():
    assert ev.reference_resolution(1) == 1.0
    assert ev.reference_resolution(4) == pytest.approx(1 / np.sqrt(2))               # 2 × 2 Raster: Abstand 1, Diagonale √2
    assert ev.reference_resolution(9) == pytest.approx(1 / (2 * np.sqrt(2)))         # 3 × 3: Abstand 1, Diagonale 2√2


def test_the_playback_steps_exist_only_up_to_the_documented_size():
    small = ev.analyse(ev.Settings("delaunay", C.PLAYBACK_MAX_N, 101, "grid"))
    assert len(small.grid.steps) == C.PLAYBACK_MAX_N - 3
    big = ev.analyse(ev.Settings("delaunay", C.PLAYBACK_MAX_N + 1, 101, "grid"))
    assert big.grid.steps == ()
