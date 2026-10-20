"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App.

Zufallsabhängige Werte (Federmodell, Stress, gestapelte Triangulierungen, Karten) als Median über feste Karten und mit Band, weil die Bibliotheksversionen der CI die Ziehungen
und das Rechnen mit Gleitkomma verschieben dürfen; Tutte, Gitterzeichnung und Handrechnungen exakt."""

import pytest

import gzn_constants as C
import gzn_evaluation as ev
import gzn_scenario as S


def _by(rows):
    return {(r["n"], r["method"]): r for r in rows}


@pytest.fixture(scope="module")
def maps():
    return _by(ev.maps_table())


@pytest.fixture(scope="module")
def stacked():
    return _by(ev.stacked_table())


@pytest.fixture(scope="module")
def onion():
    return _by(ev.onion_table())


@pytest.fixture(scope="module")
def thin():
    return _by(ev.sweep("thin", [60], C.SWEEP_SEEDS, ev.COMPARED))


@pytest.fixture(scope="module")
def drift():
    return {(r["n"], r["step"]): r for r in ev.drift_table()}


@pytest.fixture(scope="module")
def starts():
    return {r["n"]: r for r in ev.starts_table()}


@pytest.fixture(scope="module")
def lattice():
    return ev.lattice_table()


@pytest.fixture(scope="module")
def toronto():
    return {r["n"]: r for r in ev.toronto_table()}


def test_the_series_use_the_documented_fixed_maps():
    assert C.MAP_NS == (30, 60, 100) and C.SWEEP_SEEDS == tuple(range(100, 110)) and C.ONION_KS == (3, 5, 8, 12)
    assert C.STARTS_NS == (20, 30, 60) and C.STARTS_COUNT == 20 and C.DRIFT_NS == (60, 100) and C.TORONTO_NS == (100, 200, 300)
    assert (C.LATTICE_K, C.LATTICE_STARTS) == (15, 10)


def test_on_maps_the_heuristics_are_never_crossing_free_and_the_guaranteed_methods_nearly_always_are(maps):
    for n in C.MAP_NS:
        assert maps[(n, "fr")]["valid"] == 0 and maps[(n, "stress")]["valid"] == 0 and maps[(n, "grid")]["valid"] == 10
        assert maps[(n, "tutte")]["crossings_max"] == 0                                   # Tutte kreuzt nie, auch wo die Zeichnung sonst ungültig ist
    assert [maps[(n, "tutte")]["valid"] for n in C.MAP_NS] == [9, 10, 10]                  # n = 30: eine Karte hat eine Hüllenecke vom Grad 2 und ist nicht 3-fach zusammenhängend
    assert [maps[(n, "fr")]["crossings"] for n in C.MAP_NS] == pytest.approx([29, 78.5, 150.5], rel=0.3)
    assert [maps[(n, "stress")]["crossings"] for n in C.MAP_NS] == pytest.approx([28.5, 90, 185.5], rel=0.3)
    assert all(maps[(n, "stress")]["crossings"] >= 0.8 * maps[(n, "fr")]["crossings"] for n in C.MAP_NS)      # die Stress-Minimierung kreuzt nicht weniger


def test_on_maps_tutte_is_as_spread_out_as_the_spring_model_but_the_grid_is_not(maps):
    r = {m: maps[(100, m)] for m in ev.COMPARED}
    assert r["fr"]["ratio"] == pytest.approx(0.21, abs=0.03) and r["stress"]["ratio"] == pytest.approx(0.25, abs=0.03)
    assert r["tutte"]["ratio"] == pytest.approx(0.155, abs=0.03) and r["grid"]["ratio"] == pytest.approx(0.058, abs=0.01)
    assert r["tutte"]["ratio"] > 0.6 * r["fr"]["ratio"] and r["grid"]["ratio"] < 0.4 * r["fr"]["ratio"]


def test_on_maps_tutte_has_the_best_angles_and_the_grid_the_worst(maps):
    r = {m: maps[(100, m)] for m in ev.COMPARED}
    assert r["tutte"]["min_angle"] == pytest.approx(12.4, abs=3) and r["fr"]["min_angle"] < 2 and r["stress"]["min_angle"] < 2 and r["grid"]["min_angle"] < 0.1
    assert r["tutte"]["min_angle"] > 5 * r["fr"]["min_angle"]
    assert r["tutte"]["edge_ratio"] == pytest.approx(17.4, rel=0.3) and r["fr"]["edge_ratio"] == pytest.approx(10.1, rel=0.3) and r["stress"]["edge_ratio"] == pytest.approx(4.6, rel=0.3)
    assert r["grid"]["edge_ratio"] == pytest.approx(196, rel=0.02)


def test_on_maps_tutte_stays_closest_to_the_real_positions_and_the_grid_loses_them(maps):
    r = {m: maps[(100, m)]["fidelity"] for m in ev.COMPARED}
    assert r["tutte"] == pytest.approx(0.069, abs=0.03) and r["fr"] == pytest.approx(0.078, abs=0.03) and r["stress"] == pytest.approx(0.098, abs=0.03)
    assert r["grid"] > 0.5


def test_the_grid_drawing_fills_the_bound_exactly_on_maps():
    for seed in C.SWEEP_SEEDS:
        a = ev.analyse(ev.Settings("delaunay", 100, seed, "grid"))
        assert (a.grid.width, a.grid.height) == (196, 98) and a.measures.valid


def test_on_stacked_triangulations_tutte_collapses_and_the_grid_does_not(stacked):
    ns = C.MAP_NS
    assert all(stacked[(n, "tutte")]["valid"] == 10 and stacked[(n, "grid")]["valid"] == 10 for n in ns)
    assert [stacked[(n, "tutte")]["ratio"] for n in ns] == pytest.approx([0.011, 0.0043, 0.0013], rel=0.2)
    assert [stacked[(n, "grid")]["ratio"] for n in ns] == pytest.approx([0.101, 0.074, 0.058], rel=0.1)
    assert [stacked[(n, "tutte")]["edge_ratio"] for n in ns] == pytest.approx([434, 1710, 7920], rel=0.3)
    assert [stacked[(n, "fr")]["crossings"] for n in ns] == pytest.approx([90.5, 224, 448], rel=0.3)
    assert all(stacked[(n, "fr")]["valid"] == 0 and stacked[(n, "stress")]["valid"] == 0 for n in ns)


def test_on_the_onion_tutte_loses_about_a_factor_of_five_per_level_exactly_counted(onion):
    ks = C.ONION_KS
    assert all(onion[(3 * k, "tutte")]["valid"] == 1 and onion[(3 * k, "grid")]["valid"] == 1 for k in ks)
    res = [onion[(3 * k, "tutte")]["resolution"] for k in ks]
    assert res == pytest.approx([5.3e-2, 2.3e-3, 2.1e-5, 4.0e-8], rel=0.05)
    assert 4 < (res[2] / res[3]) ** 0.25 < 6                              # Faktor je Stufe zwischen k = 8 und k = 12
    assert [onion[(3 * k, "grid")]["resolution"] for k in ks] == pytest.approx([0.064, 0.034, 0.020, 0.013], rel=0.05)
    assert [onion[(3 * k, "fr")]["crossings"] for k in ks] == pytest.approx([2, 4, 7, 13], rel=0.5)
    assert [onion[(3 * k, "stress")]["crossings"] for k in ks] == pytest.approx([3, 5, 8, 12], rel=0.5)


def test_maps_with_missing_edges_make_tutte_collapse_vertices_while_the_crossing_counter_sees_nothing(thin):
    rows = {method: next(v for k, v in thin.items() if k[1] == method) for method in ev.COMPARED}
    assert rows["tutte"]["maps"] == 10 and rows["tutte"]["valid"] == 2 and rows["tutte"]["crossings_max"] == 0
    assert rows["tutte"]["touching"] >= 1                                              # im Median fallen Knoten aufeinander
    assert rows["grid"]["valid"] == 10 and rows["fr"]["valid"] == 0
    assert 58 <= rows["tutte"]["n"] <= 60


def test_spring_model_with_many_starts_never_reaches_zero_crossings(starts):
    assert [starts[n]["zero_maps"] for n in C.STARTS_NS] == [0, 0, 0]
    assert [(starts[n]["best_min"], starts[n]["best_max"]) for n in C.STARTS_NS] == pytest.approx([(5, 19), (11, 32), (31, 103)], rel=0.5)
    assert all(starts[n]["maps"] == 10 and starts[n]["starts"] == 20 for n in C.STARTS_NS)
    assert [starts[n]["median"] for n in C.STARTS_NS] == pytest.approx([16, 24, 81.5], rel=0.3)


def test_stress_from_the_tutte_start_loses_crossing_freedom_within_ten_steps(drift):
    for n in C.DRIFT_NS:
        assert drift[(n, 0)]["crossings"] == 0 and drift[(n, 0)]["crossings_max"] == 0
        assert drift[(n, 1)]["crossings"] > 0
    assert [drift[(60, s)]["crossings"] for s in (1, 10, 300)] == pytest.approx([8.5, 78, 90], rel=0.3)
    assert [drift[(100, s)]["crossings"] for s in (1, 10, 300)] == pytest.approx([17.5, 179.5, 188], rel=0.3)
    assert drift[(100, 10)]["crossings"] > 0.8 * drift[(100, 300)]["crossings"]
    assert all(drift[(n, 300)]["crossings_min"] > 0 for n in C.DRIFT_NS)


def test_on_the_lattice_the_spring_model_folds_in_about_half_of_the_runs_and_stress_never(lattice):
    assert lattice["n"] == 225 and lattice["stress"] == 0
    assert 3 <= lattice["starts"] - lattice["fr_zero"] <= 7 and max(lattice["fr"]) >= 100


def test_sgd_reaches_the_stress_of_the_majorization():
    for n in (30, 60, 100):
        maj = ev.analyse(ev.Settings("delaunay", n, 101, "stress")).measures.stress
        sgd = ev.analyse(ev.Settings("delaunay", n, 101, "sgd")).measures.stress
        assert sgd == pytest.approx(maj, rel=0.02)


def test_street_excerpts(toronto):
    assert [toronto[n]["geography"] for n in C.TORONTO_NS] == [7, 13, 13]
    assert [toronto[n]["planar"] for n in C.TORONTO_NS] == [True, False, False]
    assert not any(toronto[n]["biconnected"] for n in C.TORONTO_NS)
    assert [toronto[n]["stress"] for n in C.TORONTO_NS] == pytest.approx([3, 35, 48], rel=0.3)
    assert [toronto[n]["fr"] for n in C.TORONTO_NS] == pytest.approx([12, 36, 90], rel=0.6)
    for n in C.TORONTO_NS:
        for method in ("tutte", "grid"):
            assert ev.analyse(ev.Settings("toronto", n, 101, method)).measures is None


def test_the_reference_resolution_and_the_thresholds_are_the_documented_ones():
    assert ev.reference_resolution(100) == pytest.approx(1 / (2 ** 0.5 * 9))
    assert (C.READABLE_GOOD, C.READABLE_TIGHT) == (0.15, 0.02)
    assert S.make_onion(36).n == 36


def test_the_one_invalid_map_at_n30_has_a_hull_vertex_of_degree_two():
    import numpy as np
    from scipy.spatial import ConvexHull

    import gzn_planar as Pl

    invalid = [seed for seed in C.SWEEP_SEEDS if not ev.analyse(ev.Settings("delaunay", 30, seed, "tutte")).measures.valid]
    assert invalid == [102]
    inst = S.make_delaunay(30, 102)
    adj = Pl.adjacency(inst.n, inst.edges)
    assert min(len(adj[v]) for v in ConvexHull(np.array(inst.true)).vertices) == 2


def test_street_excerpts_stop_being_planar_between_160_and_170_nodes():
    assert ev.prepare(S.make_toronto_ball(160)).facts.planar and not ev.prepare(S.make_toronto_ball(170)).facts.planar
