"""Verdikte und Zahlenformate an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import dataclasses

import pytest

import gzn_algorithm as A
import gzn_constants as C
import gzn_evaluation as ev
import gzn_geometry as G
import gzn_scenario as S

K4 = S.hand_graph("K4")
K4_FACES = [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)]


def _analysis(method, crossings=0, touching=0, ratio=0.5, planar=True, tricon=True, grid=None, note=None):
    """Eine Analyse von K4 (n = 4, Bezugsauflösung 1/√2) mit gesetzten Maßen: Auflösung = ratio · Bezug."""
    prep = ev.prepare(K4)
    prep = dataclasses.replace(prep, facts=ev.Facts(planar, True, tricon, 4))
    measures = None if note else G.Measures(crossings, touching, ratio * ev.reference_resolution(4), 5.0, 2.0, 30.0, 0.1, None)
    return ev.Analysis(ev.Settings("delaunay", 4, 0, method), prep, None, None, measures, (), None, grid, note)


def test_number_formats():
    assert ev.num(1234567.891, 1) == "1.234.567,9" and ev.num(0.5, 2) == "0,50" and ev.num(1000) == "1.000"
    assert ev.plural(1, "Kreuzung", "Kreuzungen") == "1 Kreuzung" and ev.plural(0, "Kreuzung", "Kreuzungen") == "0 Kreuzungen"
    assert ev.sci(7.5e-11) == "7,5·10⁻¹¹" and ev.sci(0) == "0" and ev.sci(1234.0) == "1,2·10³" and ev.sci(0.1) == "1,0·10⁻¹"
    assert ev.sci(9.96) == "1,0·10¹"                                         # Rundung auf 10,0 springt in die nächste Zehnerpotenz
    assert ev.percent(0.5) == "50,0 %" and ev.percent(0.1234) == "12,3 %" and ev.percent(0.001) == "0,1 %" and ev.percent(0.0005) == "5,0·10⁻² %"


def test_stress_text():
    assert ev.stress_text(None) == "-" and ev.stress_text(0.0427) == "0,043"


def test_verdict_for_a_note_is_the_note():
    assert ev.verdict_drawing(_analysis("tutte", note="Hinweistext")) == ("info", "Hinweistext")


def test_verdict_heuristic_with_crossings_on_a_planar_graph():
    kind, text = ev.verdict_drawing(_analysis("fr", crossings=3))
    assert kind == "warning"
    assert text == ("**3 Kreuzungen.** Das Verfahren kennt die Planarität nicht und garantiert keine kreuzungsfreie Zeichnung. "
                    "Der Graph ist planar, eine kreuzungsfreie Zeichnung gibt es.")


def test_verdict_heuristic_with_one_crossing_on_a_non_planar_graph():
    kind, text = ev.verdict_drawing(_analysis("stress", crossings=1, planar=False))
    assert kind == "warning" and text.startswith("**1 Kreuzung.** ") and text.endswith("Der Graph ist nicht planar, ganz ohne Kreuzungen geht es nicht.")


def test_verdict_heuristic_without_crossings_names_the_missing_guarantee_and_the_tightness():
    kind, text = ev.verdict_drawing(_analysis("sgd", ratio=0.5))
    assert kind == "success"
    assert text == ("Diesmal ohne Kreuzung, aber das Verfahren garantiert das nicht. Die Knoten sind 50,0 % so weit auseinander wie bei einem gleichmäßigen Raster aus 4 Knoten "
                    "(kleinster Abstand je Diagonale: 0,3536).")
    assert ev.verdict_drawing(_analysis("fr", ratio=0.1))[0] == "info" and ev.verdict_drawing(_analysis("fr", ratio=0.01))[0] == "info"


def test_verdict_tightness_switches_to_powers_of_ten_for_tiny_distances():
    _, text = ev.verdict_drawing(_analysis("fr", ratio=1e-6))
    assert "1,0·10⁻⁴ %" in text and "(kleinster Abstand je Diagonale: 7,1·10⁻⁷)" in text


def test_verdict_invalid_tutte_names_collisions_and_the_blind_crossing_counter():
    kind, text = ev.verdict_drawing(_analysis("tutte", touching=2, tricon=False))
    assert kind == "warning"
    assert text == ("**Ungültige Zeichnung:** 2 Knoten liegen auf einem anderen Knoten oder einer fremden Kante. "
                    "Die Garantie gilt nicht, weil der Graph nicht 3-fach zusammenhängend ist. Der einfache Kreuzungszähler allein würde hier null melden.")
    _, one = ev.verdict_drawing(_analysis("tutte", touching=1, tricon=False))
    assert "1 Knoten liegt auf einem anderen Knoten" in one


def test_verdict_invalid_with_crossings_only_and_with_both():
    _, only = ev.verdict_drawing(_analysis("tutte", crossings=2))
    assert only == "**Ungültige Zeichnung:** 2 Kreuzungen. Die Garantie gilt nicht, weil der Graph nicht 3-fach zusammenhängend ist."
    _, both = ev.verdict_drawing(_analysis("tutte", crossings=1, touching=1))
    assert both.startswith("**Ungültige Zeichnung:** 1 Knoten liegt auf einem anderen Knoten oder einer fremden Kante und 1 Kreuzung.")
    assert "null melden" not in both


def test_verdict_valid_tutte_by_grade():
    head = "**Kreuzungsfrei**, wie der Satz von Tutte es für 3-fach zusammenhängende planare Graphen garantiert."
    kind, text = ev.verdict_drawing(_analysis("tutte", ratio=0.5))
    assert kind == "success" and text.startswith(head + " Die Knoten sind 50,0 % so weit") and text.endswith(" Gut lesbar.")
    kind, text = ev.verdict_drawing(_analysis("tutte", ratio=0.1))
    assert kind == "info" and text.endswith(" Eng.")
    kind, text = ev.verdict_drawing(_analysis("tutte", ratio=0.01))
    assert kind == "warning" and text.endswith(" Kaum lesbar: Knoten und Kanten fallen in der Anzeige zusammen.")


def test_verdict_valid_tutte_without_triconnectivity_has_no_guarantee():
    _, text = ev.verdict_drawing(_analysis("tutte", ratio=0.5, tricon=False))
    assert text.startswith("Diesmal gültig, aber ohne Garantie: der Graph ist nicht 3-fach zusammenhängend. Die Knoten sind")


def test_verdict_valid_grid_names_the_grid_and_the_bound():
    grid = A.fpp_drawing(4, K4.edges, K4_FACES)
    kind, text = ev.verdict_drawing(_analysis("grid", ratio=0.1, grid=grid))
    assert kind == "info"
    assert text.startswith("**Kreuzungsfrei im Gitter 4 × 2** (Schranke (2n − 4) × (n − 2) = 4 × 2), garantiert für jeden zweifach zusammenhängenden planaren Graphen. Die Knoten sind 10,0 %")


def test_analysis_ratio_is_the_resolution_over_the_reference():
    assert _analysis("fr", ratio=0.25).ratio == pytest.approx(0.25)
    assert _analysis("fr", note="x").ratio is None


# --- Zeilen der Reihen -------------------------------------------------------------------------------------------------------------------------


def _row(method, n=60, maps=10, valid=0, crossings=5.0, ratio=0.2, resolution=0.01, min_angle=1.0, edge_ratio=2.0, **extra):
    return {"net": "delaunay", "method": method, "maps": maps, "available": maps, "n": n, "valid": valid, "crossings": crossings, "crossings_min": 0, "crossings_max": 9,
            "touching": 0.0, "resolution": resolution, "ratio": ratio, "min_angle": min_angle, "edge_ratio": edge_ratio, "area": 10.0, "stress": 0.1, "fidelity": None} | extra


def _rows(n=100):
    return [
        _row("fr", n, crossings=150.5, valid=0, ratio=0.2066, resolution=0.0162, min_angle=0.57),
        _row("stress", n, crossings=185.5, valid=0, ratio=0.25),
        _row("tutte", n, crossings=0.0, valid=10, ratio=0.1546, resolution=0.0121, min_angle=12.36, edge_ratio=12.4),
        _row("grid", n, crossings=0.0, valid=10, ratio=0.0581, resolution=0.00456, min_angle=0.01, edge_ratio=196.0),
        _row("fr", 30, crossings=1.0), _row("stress", 30), _row("tutte", 30), _row("grid", 30),
    ]


def test_verdict_maps_takes_the_largest_size_and_names_every_number():
    kind, text = ev.verdict_maps(_rows())
    assert kind == "info"
    assert text == ("Bei n = 100 (Median über 10 Karten): Federmodell 150 Kreuzungen (kreuzungsfrei in 0 von 10 Karten), Stress-Minimierung 186 (0 von 10); "
                    "Tutte-Einbettung und Gitterzeichnung kreuzungsfrei in 10 bzw. 10 von 10 Karten. "
                    "Die Tutte-Zeichnung liegt mit 15,5 % der Auflösung eines Quadratrasters nahe am Federmodell (20,7 %), die Gitterzeichnung mit 5,8 % deutlich darunter; "
                    "der kleinste Winkel an einem Knoten ist bei Tutte 12,4°, beim Federmodell 0,6°, bei der Gitterzeichnung 0,01°.")


def test_verdict_hard_compares_tutte_and_grid_at_the_largest_size():
    rows = [
        _row("fr", 48, crossings=16.0, resolution=0.0172), _row("stress", 48),
        _row("tutte", 48, maps=1, valid=1, crossings=0.0, ratio=6.3e-10, resolution=7.5e-11, edge_ratio=6.66e9),
        _row("grid", 48, maps=1, valid=1, crossings=0.0, ratio=0.082, resolution=0.0097),
        _row("tutte", 9, maps=1, valid=1),
    ]
    kind, text = ev.verdict_hard(rows, "Zwiebel")
    assert kind == "warning"
    assert text == ("Zwiebel, n = 48: Die Tutte-Zeichnung ist in 1 von 1 Fällen kreuzungsfrei, aber der kleinste Knotenabstand je Diagonale ist nur 7,5·10⁻¹¹ "
                    "(6,3·10⁻⁸ % eines Quadratrasters) und das Verhältnis längste zu kürzester Kante 6,7·10⁹. Die Gitterzeichnung hat 0,0097 (8,2 %), "
                    "das Federmodell 0,0172, hat aber 16 Kreuzungen.")


def test_verdict_drift_names_start_after_ten_steps_and_end():
    rows = [{"n": 60, "step": s, "crossings": c, "crossings_min": c - 1, "crossings_max": c + 1} for s, c in ((0, 0.0), (1, 8.5), (10, 78.0), (300, 90.0))]
    rows += [{"n": 100, "step": s, "crossings": c, "crossings_min": c - 2, "crossings_max": c + 2} for s, c in ((0, 0.0), (3, 104.5), (10, 179.5), (30, 197.0), (300, 188.0))]
    kind, text = ev.verdict_drift(rows)
    assert kind == "warning"
    assert text == ("Bei n = 100: Start 0 Kreuzungen (Tutte), nach 10 Schritten der Stress-Majorisierung 180, nach 300 Schritten 188 "
                    "(Median über die Karten, kleinster Wert 186). Weglängen im Graphen kennen die Planarität nicht.")
    short = [r for r in rows if r["n"] == 60 and r["step"] in (0, 1)]
    assert ev.verdict_drift(short)[1] == ("Bei n = 60: Start 0 Kreuzungen (Tutte), nach 1 Schritt der Stress-Majorisierung 8, nach 1 Schritt 8 "
                                          "(Median über die Karten, kleinster Wert 8). Weglängen im Graphen kennen die Planarität nicht.")       # ohne Schritt ab 10: der letzte; 8,5 und 7,5 runden auf gerade


def test_verdict_starts_is_a_warning_without_zero_maps_and_a_success_with_one():
    rows = [
        {"n": 20, "maps": 10, "starts": 20, "best": [5], "best_min": 5, "best_max": 19, "zero_maps": 0, "median": 16.0},
        {"n": 30, "maps": 10, "starts": 20, "best": [11], "best_min": 11, "best_max": 32, "zero_maps": 0, "median": 24.0},
    ]
    kind, text = ev.verdict_starts(rows)
    assert kind == "warning"
    assert text == "Mit 20 Startlagen je Karte erreicht das Federmodell in 0 von 20 Karten null Kreuzungen (n = 20: kleinste Zahl je Karte 5 bis 19; n = 30: kleinste Zahl je Karte 11 bis 32)."
    rows[0] = rows[0] | {"zero_maps": 2}
    kind, text = ev.verdict_starts(rows)
    assert kind == "success" and "in 2 von 20 Karten" in text


def test_verdict_toronto_names_planarity_and_the_three_counts():
    rows = [{"n": 100, "m": 120, "planar": True, "biconnected": False, "geography": 7, "fr": 7, "stress": 3},
            {"n": 200, "m": 255, "planar": False, "biconnected": False, "geography": 13, "fr": 48, "stress": 35}]
    kind, text = ev.verdict_toronto(rows)
    assert kind == "info"
    assert text.startswith("n = 100 (planar): geografische Lage 7 Kreuzungen, Federmodell 7, Stress 3; n = 200 (nicht planar): geografische Lage 13 Kreuzungen, Federmodell 48, Stress 35. ")


# --- summarize ------------------------------------------------------------------------------------------------------------------------------


def test_summarize_takes_medians_counts_valid_maps_and_skips_missing_drawings():
    analyses = [_analysis("fr", crossings=c, ratio=r) for c, r in ((4, 0.2), (0, 0.4), (10, 0.3))] + [_analysis("fr", note="keine Zeichnung")]
    row = ev.summarize("delaunay", "fr", analyses)
    assert row["maps"] == 4 and row["available"] == 3 and row["valid"] == 1 and row["n"] == 4
    assert row["crossings"] == 4 and row["crossings_min"] == 0 and row["crossings_max"] == 10
    assert row["ratio"] == pytest.approx(0.3) and row["touching"] == 0 and row["stress"] == 0.1 and row["fidelity"] is None
    assert row["min_angle"] == 30.0 and row["edge_ratio"] == 2.0 and row["area"] == 5.0 and row["resolution"] == pytest.approx(0.3 * ev.reference_resolution(4))


def test_summarize_without_any_drawing_has_only_counts():
    row = ev.summarize("delaunay", "tutte", [_analysis("tutte", note="x"), _analysis("tutte", note="y")])
    assert row == {"net": "delaunay", "method": "tutte", "maps": 2, "available": 0, "n": 4, "valid": 0}


def test_row_picks_the_largest_size_of_the_method():
    rows = [_row("fr", 30), _row("fr", 100), _row("grid", 100)]
    assert ev._row(rows, "fr")["n"] == 100 and ev._row(rows, "fr", 30)["n"] == 30 and ev._row(rows, "stress") is None


def test_compared_methods_exclude_sgd():
    assert ev.COMPARED == ("fr", "stress", "tutte", "grid") and set(ev.COMPARED) < set(C.METHODS)


def test_verdict_lattice_counts_the_folded_runs():
    row = {"k": 15, "n": 225, "starts": 10, "fr": [201, 0, 229, 0, 0, 144, 108, 192, 0, 0], "fr_zero": 5, "stress": 0}
    kind, text = ev.verdict_lattice(row)
    assert kind == "warning"
    assert text == ("Auf dem 15 × 15-Gitter (n = 225) endet das Federmodell in 5 von 10 Läufen mit Kreuzungen (je Lauf: 201, 0, 229, 0, 0, 144, 108, 192, 0, 0); "
                    "die Stress-Minimierung endet mit 0 Kreuzungen: Im Gitter passen Weglängen und ebene Lage gut zusammen.")
    clean = ev.verdict_lattice(row | {"fr": [0] * 10, "fr_zero": 10, "stress": 1})
    assert clean[0] == "success" and "in 0 von 10 Läufen" in clean[1] and "mit 1 Kreuzung:" in clean[1]


def test_percent_short_hides_tiny_values():
    assert ev.percent_short(0.5) == "50,0 %" and ev.percent_short(0.001) == "0,1 %" and ev.percent_short(0.0009) == "< 0,1 %" and ev.percent_short(0.0) == "< 0,1 %"
