"""Figuren: Aufbau, Farben, gesperrte Achsen, Filter für das Abspielen."""

import numpy as np

import gzn_constants as C
import gzn_visualization as V

P = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
EDGES = [(0, 2), (1, 3), (0, 1)]


def _traces(fig):
    return {t.line.color: t for t in fig.data if t.mode == "lines"}


def test_drawing_splits_crossing_edges_from_the_rest_and_locks_the_axes():
    fig = V.build_drawing(P, EDGES, crossing_pairs=[(0, 1)], title="Titel")
    lines = _traces(fig)
    assert set(lines) == {C.COLOR_EDGE, C.COLOR_EDGE_BAD}
    assert list(lines[C.COLOR_EDGE_BAD].x) == [0.0, 2.0, None, 2.0, 0.0, None]
    assert list(lines[C.COLOR_EDGE].x) == [0.0, 2.0, None]
    assert fig.layout.title.text == "Titel" and fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange
    assert fig.layout.yaxis.scaleanchor == "x" and not fig.layout.showlegend
    markers = [t for t in fig.data if t.mode == "markers"][0]
    assert list(markers.x) == [0.0, 2.0, 2.0, 0.0] and markers.hovertext[2] == "Knoten 2"


def test_drawing_without_crossings_has_no_red_trace_and_no_title():
    fig = V.build_drawing(P, EDGES)
    assert set(_traces(fig)) == {C.COLOR_EDGE} and fig.layout.title.text is None


def test_drawing_draws_the_outer_face_extra_edges_and_highlights():
    fig = V.build_drawing(P, EDGES, outer=(0, 1, 2), extra_edges=[(2, 3)], highlight={3})
    lines = _traces(fig)
    assert C.COLOR_OUTER in lines and list(lines[C.COLOR_OUTER].x) == [0.0, 2.0, None, 2.0, 2.0, None, 2.0, 0.0, None]
    dotted = [t for t in fig.data if t.mode == "lines" and t.line.dash == "dot"]
    assert len(dotted) == 1 and list(dotted[0].x) == [2.0, 0.0, None]
    markers = [t for t in fig.data if t.mode == "markers"][0]
    assert markers.marker.color[3] == C.COLOR_OUTER and markers.marker.color[0] == C.COLOR_NODE and markers.marker.size[3] == 12


def test_drawing_restricted_to_placed_nodes_keeps_only_their_edges_and_fits_the_view_to_them():
    fig = V.build_drawing(P, EDGES, placed={0, 1}, extra_edges=[(2, 3)])
    lines = _traces(fig)
    assert list(lines[C.COLOR_EDGE].x) == [0.0, 2.0, None]                  # nur Kante (0, 1)
    assert not [t for t in fig.data if t.mode == "lines" and t.line.dash == "dot"][0].x      # die Sehne (2, 3) fällt weg
    markers = [t for t in fig.data if t.mode == "markers"][0]
    assert list(markers.x) == [0.0, 2.0]
    assert fig.layout.yaxis.range[0] < 0 < fig.layout.yaxis.range[1] < 1          # Höhe nur von den platzierten Knoten, mit Rand
    assert fig.layout.xaxis.range[0] < 0 and fig.layout.xaxis.range[1] > 2


def test_big_drawings_use_thin_lines():
    big = np.random.default_rng(1).random((130, 2))
    edges = [(i, i + 1) for i in range(129)]
    fig = V.build_drawing(big, edges)
    assert _traces(fig)[C.COLOR_EDGE].line.width == 1
    assert [t for t in fig.data if t.mode == "markers"][0].marker.size[0] == 5
    assert _traces(V.build_drawing(P, EDGES))[C.COLOR_EDGE].line.width == 2


def _ratio_rows():
    return [
        {"method": "fr", "n": 30, "ratio": 0.3}, {"method": "fr", "n": 60, "ratio": 0.2},
        {"method": "tutte", "n": 30, "ratio": 0.0}, {"method": "tutte", "n": 60, "ratio": 1e-5},
        {"method": "stress", "n": 30}, {"method": "grid", "n": 30, "ratio": 0.1},
    ]


def test_ratio_chart_has_one_line_per_method_with_values_clipped_to_the_log_axis():
    fig = V.build_ratio(_ratio_rows(), "Knoten n", "Titel")
    names = [t.name for t in fig.data]
    assert names == ["Federmodell", "Tutte", "Gitterzeichnung"]                   # Stress hat keine Werte
    tutte = fig.data[1]
    assert list(tutte.y) == [1e-12, 1e-5] and list(tutte.x) == [30, 60]
    assert fig.layout.yaxis.type == "log" and fig.layout.xaxis.type == "log" and fig.layout.xaxis.fixedrange
    assert list(fig.layout.xaxis.tickvals) == [30, 60] and fig.layout.title.text == "Titel"
    only = V.build_ratio(_ratio_rows(), "x", "t", methods=("fr",))
    assert [t.name for t in only.data] == ["Federmodell"]


def test_drift_chart_has_one_line_per_size_over_categorical_steps():
    rows = [{"n": 60, "step": s, "crossings": c} for s, c in ((0, 0.0), (1, 8.5))] + [{"n": 100, "step": s, "crossings": c} for s, c in ((0, 0.0), (1, 17.5))]
    fig = V.build_drift(rows)
    assert [t.name for t in fig.data] == ["n = 60", "n = 100"]
    assert list(fig.data[1].x) == ["0", "1"] and list(fig.data[1].y) == [0.0, 17.5]
    assert fig.layout.xaxis.type == "category" and fig.layout.yaxis.fixedrange
