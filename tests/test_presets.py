"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und was die Hilfetexte über das Ergebnis behaupten."""

import gzn_constants as C
import gzn_evaluation as ev
import gzn_presets as P
import gzn_scenario as S

FIRST = "Lehrbeispiel: Tutte auf einer Karte"


def _analysis(name, **override):
    values = dict(C.PRESETS[name]) | override
    return ev.analyse(ev.Settings(**values))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for preset in C.PRESETS.values():
        assert preset["net"] in C.NET_TYPES and preset["method"] in C.METHODS and preset["outer"] in C.OUTER
        assert C.N_MIN <= preset["n"] <= C.N_MAX and 0 <= preset["seed"] <= C.SEED_MAX and C.ITER_MIN <= preset["iterations"] <= C.ITER_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[FIRST]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_textbook_map_is_valid_triconnected_and_as_spread_out_as_the_spring_model():
    a = _analysis(FIRST)
    fr = _analysis(FIRST, method="fr")
    assert a.measures.valid and a.prep.facts.triconnected and ev.readable_grade(a.ratio) == "gut"
    assert a.ratio >= 0.7 * fr.ratio                        # "so weit auseinander wie beim Federmodell"
    assert "60 Standorte" in C.PRESET_HELP[FIRST] and a.prep.inst.n == 60


def test_spring_model_preset_has_crossings_on_a_planar_map():
    a = _analysis("Federmodell: nie kreuzungsfrei")
    assert a.settings.method == "fr" and a.measures.crossings > 0 and a.prep.facts.planar


def test_onion_preset_is_valid_but_barely_readable_and_loses_a_factor_of_about_five_per_step():
    a = _analysis("Zwiebel: Tutte ballt die Knoten")
    assert a.prep.inst.n == 36 and a.measures.valid and ev.readable_grade(a.ratio) == "kaum"
    one_step_less = ev.analyse(ev.Settings("onion", 33, 101, "tutte"))
    assert 3 < one_step_less.measures.resolution / a.measures.resolution < 8


def test_grid_preset_is_valid_in_the_bound_grid_with_long_edges_and_tiny_angles():
    a = _analysis("Gitterzeichnung: garantiert, aber eng")
    assert a.prep.inst.n == 40 and a.measures.valid and (a.grid.width, a.grid.height) == (76, 38)
    assert a.measures.edge_ratio > 20 and a.measures.min_angle < 1.0


def test_thin_map_preset_collapses_vertices_under_tutte():
    a = _analysis("Karte mit fehlenden Kanten: Tutte entartet")
    assert not a.prep.facts.triconnected and a.measures.touching > 0 and not a.measures.valid
    assert a.measures.crossings == 0                                         # der einfache Zähler sähe nichts


def test_randomize_seed_stays_inside_the_bounds():
    import streamlit as st

    st.session_state.clear()
    P.randomize_seed()
    assert 0 <= st.session_state["seed_input"] <= C.SEED_MAX


def test_named_presets_build_instances():
    for preset in C.PRESETS.values():
        assert S.make_instance(preset["net"], preset["n"], preset["seed"]).n >= 3
