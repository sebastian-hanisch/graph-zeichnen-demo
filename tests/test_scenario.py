"""Szenario: Generatoren sind deterministisch, haben die versprochene Gestalt und die Eigenschaften, auf die sich die Demo stützt."""

import pytest

import gzn_planar as P
import gzn_scenario as S

nx = pytest.importorskip("networkx")


def test_normalize_edges_sorts_dedups_and_drops_loops():
    assert S.normalize_edges([(3, 1), (1, 3), (2, 2), (0, 1)]) == ((0, 1), (1, 3))


def test_relabel_renumbers_in_the_order_of_the_old_numbers():
    n, edges, true = S.relabel([(5, 9), (9, 2)], {2: (0, 0), 5: (1, 1), 9: (2, 2)})
    assert n == 3 and edges == ((0, 2), (1, 2)) and true == ((0, 0), (1, 1), (2, 2))          # 2 → 0, 5 → 1, 9 → 2: (5, 9) → (1, 2), (9, 2) → (0, 2)
    assert S.relabel([(1, 2)])[2] is None


def test_delaunay_is_deterministic_planar_and_within_the_euler_bound():
    a, b = S.make_delaunay(40, 7), S.make_delaunay(40, 7)
    assert a == b and S.make_delaunay(40, 8) != a
    assert a.n == 40 and len(a.true) == 40 and a.m <= 3 * 40 - 6
    assert nx.check_planarity(nx.Graph(list(a.edges)))[0]
    assert all(u < v for u, v in a.edges)


def test_thin_keeps_the_largest_biconnected_block_of_eighty_percent_of_the_edges():
    inst = S.make_thin(60, 5)
    full = S.make_delaunay(60, 5)
    assert inst.n <= 60 and inst.m < full.m and len(inst.true) == inst.n
    assert P.is_biconnected(inst.n, P.adjacency(inst.n, inst.edges))
    assert S.make_thin(60, 5) == inst
    whole = S.make_thin(60, 5, keep=1.0)
    assert (whole.n, whole.m) == (full.n, full.m)                                       # eine Delaunay-Triangulierung ist zweifach zusammenhängend


def test_grid_and_lattice_edge_counts():
    lattice = S.make_grid(25, False)
    assert lattice.n == 25 and lattice.m == 2 * 5 * 4 and lattice.name == "Gitter"
    grid = S.make_grid(25, True)
    assert grid.n == 25 and grid.m == 2 * 5 * 4 + 16 and grid.name == "Raster mit Diagonalen"
    assert S.make_grid(26, False).n == 25 and S.make_grid(2, False).n == 4              # gerundet, mindestens 2 × 2
    assert lattice.true[0] == (0.0, 0.0) and lattice.true[5] == (0.0, -1.0)


def test_onion_has_3k_nodes_6k_minus_3_edges_and_is_triconnected():
    inst = S.make_onion(12)
    assert inst.n == 12 and inst.m == 6 * 4 - 3 and inst.true is None
    assert P.is_triconnected(inst.n, inst.edges)
    assert S.make_onion(13).n == 12 and S.make_onion(1).n == 6                          # abgerundet, mindestens zwei Stufen


def test_stacked_is_maximal_planar_and_triconnected():
    inst = S.make_stacked(30, 4)
    assert inst.n == 30 and inst.m == 3 * 30 - 6
    assert nx.check_planarity(nx.Graph(list(inst.edges)))[0] and P.is_triconnected(inst.n, inst.edges)
    assert S.make_stacked(30, 4) == inst and S.make_stacked(30, 5) != inst
    assert S.make_stacked(1, 0).n == 3


def test_toronto_ball_is_connected_has_at_most_n_nodes_and_keeps_the_coordinates():
    full = S.load_toronto()
    assert full.n == 5072 and len(full.true) == 5072
    ball = S.make_toronto_ball(80)
    assert 2 <= ball.n <= 80 and len(ball.true) == ball.n
    assert nx.is_connected(nx.Graph(list(ball.edges)))
    assert S.make_toronto_ball(80) == ball
    assert S.make_toronto_ball(1).n <= 1


def test_hand_graphs():
    assert S.hand_graph("K4").m == 6 and S.hand_graph("Würfel").m == 12 and S.hand_graph("Rad").m == 12
    assert S.hand_graph("Rad").n == 7
    with pytest.raises(ValueError):
        S.hand_graph("gibt es nicht")


def test_make_instance_dispatches_every_net_and_rejects_unknown_ones():
    import gzn_constants as C

    for net in C.NET_TYPES:
        inst = S.make_instance(net, 30, 101)
        assert inst.n >= 2 and inst.m >= 1
    assert S.make_instance("lattice", 36, 0).name == "Gitter" and S.make_instance("grid", 36, 0).name == "Raster mit Diagonalen"
    with pytest.raises(ValueError):
        S.make_instance("unbekannt", 10, 0)


def test_results_do_not_depend_on_the_hash_seed():
    """Mengen und Wörterbücher dürfen die Ergebnisse nicht beeinflussen: zwei Läufe mit verschiedenem PYTHONHASHSEED geben dieselbe Ausgabe."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; sys.path.insert(0, %r); import gzn_evaluation as ev; out = []\n"
        "for net, method in (('thin', 'tutte'), ('delaunay', 'grid'), ('stacked', 'tutte'), ('delaunay', 'fr'), ('onion', 'grid'), ('toronto', 'stress')):\n"
        "    a = ev.analyse(ev.Settings(net, 40, 101, method, 50)); out.append((net, method, a.pos, a.crossing_pairs, a.note))\n"
        "print(out)" % str(root)
    )
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
        outputs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert outputs[0] == outputs[1] and outputs[0].strip()


def test_toronto_data_and_license():
    inst = S.load_toronto()
    assert (inst.n, inst.m) == (5072, 7621)
    assert nx.is_connected(nx.Graph(list(inst.edges)))
    assert (S.DATA / "LICENSE-ODbL.md").exists()
