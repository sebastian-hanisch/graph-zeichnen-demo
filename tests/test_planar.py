"""Einbettung und Zusammenhang (kopiert aus planaritaet-demo, hier gekürzt): Handwerte und Orakel (networkx)."""

import random

import pytest

import gzn_planar as P
import gzn_scenario as S

nx = pytest.importorskip("networkx")


def _edges(g):
    return sorted((min(u, v), max(u, v)) for u, v in g.edges)


def test_adjacency_and_components():
    adj = P.adjacency(5, [(0, 1), (1, 2), (3, 4)])
    assert adj == [{1}, {0, 2}, {1}, {4}, {3}]
    assert P.components(5, adj) == 2 and P.components(3, P.adjacency(3, [])) == 3


def test_blocks_of_two_triangles_sharing_a_vertex():
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    blocks = P.biconnected_blocks(5, P.adjacency(5, edges))
    assert sorted(blocks) == [[(0, 1), (0, 2), (1, 2)], [(2, 3), (2, 4), (3, 4)]]
    assert P.largest_block(5, edges) == [(2, 3), (2, 4), (3, 4)]          # Gleichstand: der zuerst abgeschlossene Block (die Tiefensuche schließt den inneren zuerst)


def test_largest_block_picks_the_block_with_more_nodes():
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5), (4, 6), (5, 6)]
    assert P.largest_block(7, edges) == [(3, 4), (3, 5), (4, 5), (4, 6), (5, 6)]
    assert P.largest_block(3, []) == []


def test_find_cycle_in_a_cycle_and_none_in_a_path():
    adj = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}
    cycle = P.find_cycle(adj, {0, 1, 2, 3})
    assert sorted(cycle) == [0, 1, 2, 3]
    assert P.find_cycle({0: {1}, 1: {0, 2}, 2: {1}}, {0, 1, 2}) is None


def test_split_face_hand_values():
    face = [0, 1, 2, 3]
    first, second = P.split_face(face, [0, 9, 2])        # Weg 0 – 9 – 2 durch die Fläche
    assert first == [0, 1, 2, 9] and second == [2, 3, 0, 9]


def test_embed_block_of_k4_has_four_triangles_and_euler_holds():
    block = P.embed_block(S.hand_graph("K4").edges)
    assert block.planar and len(block.faces) == 4 and all(len(f) == 3 for f in block.faces)
    assert 4 - 6 + len(block.faces) == 2


def test_embed_block_of_k5_and_k33_fails():
    k5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    k33 = [(i, j) for i in range(3) for j in range(3, 6)]
    assert not P.embed_block(k5).planar and not P.embed_block(k33).planar


def test_tiny_blocks_have_no_faces():
    assert P.embed_block([(0, 1)]) == P.BlockResult((0, 1), ((0, 1),), True, ())


def test_single_block_only_for_biconnected_graphs():
    assert P.embed(4, S.hand_graph("K4").edges).single_block is not None
    path = P.embed(4, [(0, 1), (1, 2), (2, 3)])
    assert path.single_block is None and path.planar
    assert P.embed(2, [(0, 1)]).single_block is None            # weniger als drei Knoten


def test_triconnected_hand_cases():
    assert P.is_triconnected(4, S.hand_graph("K4").edges)
    assert P.is_triconnected(7, S.hand_graph("Rad").edges)
    assert P.is_triconnected(8, S.hand_graph("Würfel").edges)
    assert not P.is_triconnected(5, [(i, (i + 1) % 5) for i in range(5)])                 # Kreis: nur zweifach
    assert not P.is_triconnected(4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])             # K4 ohne eine Kante: zwei Knoten vom Grad 2
    assert not P.is_triconnected(5, [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)])     # K2,3: zweifach, nicht 3-fach
    assert not P.is_triconnected(3, [(0, 1), (1, 2), (0, 2)])                             # zu klein
    assert not P.is_triconnected(4, [(0, 1), (1, 2), (2, 3)])                             # nicht einmal zweifach


def test_is_biconnected():
    assert P.is_biconnected(3, P.adjacency(3, [(0, 1), (1, 2), (0, 2)]))
    assert not P.is_biconnected(3, P.adjacency(3, [(0, 1), (1, 2)]))
    assert not P.is_biconnected(2, P.adjacency(2, [(0, 1)]))


def _random_graph(rng, n):
    p = rng.choice([0.3, 0.45, 0.6])
    return nx.gnp_random_graph(n, p, seed=rng.randrange(10 ** 6))


def test_oracle_planarity_matches_networkx_on_random_graphs():
    rng = random.Random(7)
    for _ in range(120):
        g = _random_graph(rng, rng.randint(5, 9))
        expected = nx.check_planarity(g)[0]
        assert P.embed(g.number_of_nodes(), _edges(g)).planar == expected


def test_oracle_blocks_match_networkx():
    rng = random.Random(8)
    for _ in range(80):
        g = _random_graph(rng, rng.randint(5, 10))
        ours = sorted(sorted(b) for b in P.biconnected_blocks(g.number_of_nodes(), P.adjacency(g.number_of_nodes(), _edges(g))))
        theirs = sorted(sorted((min(u, v), max(u, v)) for u, v in comp) for comp in nx.biconnected_component_edges(g))
        assert ours == theirs


def test_oracle_triconnected_matches_node_connectivity():
    rng = random.Random(9)
    seen = {True: 0, False: 0}
    for _ in range(150):
        n = rng.randint(4, 9)
        g = nx.gnp_random_graph(n, rng.choice([0.5, 0.65, 0.8]), seed=rng.randrange(10 ** 6))
        expected = nx.is_connected(g) and nx.node_connectivity(g) >= 3
        seen[expected] += 1
        assert P.is_triconnected(n, _edges(g)) == expected
    assert seen[True] >= 10 and seen[False] >= 10        # beide Fälle kommen vor


def test_every_embedding_is_consistently_oriented_and_satisfies_euler():
    """Jede gerichtete Kante steht in genau einer Fläche (Voraussetzung für das Drehsystem) und n − m + f = 2."""
    graphs = [S.make_delaunay(n, seed) for n in (8, 20, 40) for seed in range(3)] + [S.make_stacked(n, seed) for n in (6, 15) for seed in range(3)] + [S.make_grid(16, True), S.make_grid(25, False)]
    for inst in graphs:
        block = P.embed(inst.n, inst.edges).single_block
        directed = [(f[i], f[(i + 1) % len(f)]) for f in block.faces for i in range(len(f))]
        assert len(directed) == len(set(directed)) == 2 * inst.m
        assert inst.n - inst.m + len(block.faces) == 2


def test_is_biconnected_needs_the_block_to_cover_every_vertex():
    triangle_and_isolated = P.adjacency(4, [(0, 1), (1, 2), (0, 2)])
    assert len(P.biconnected_blocks(4, triangle_and_isolated)) == 1 and not P.is_biconnected(4, triangle_and_isolated)
