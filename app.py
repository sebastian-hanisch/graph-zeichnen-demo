"""Graphen zeichnen – Garantie oder Lesbarkeit? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gzn_constants as C
import gzn_evaluation as ev
from gzn_layout import dunkles_design, kopf_css
from gzn_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from gzn_visualization import build_drawing, build_drift, build_ratio

st.set_page_config(page_title="Graphen zeichnen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _compare(settings):
    return ev.compare_all(settings)


@st.cache_data(show_spinner=False)
def _maps():
    return ev.maps_table()


@st.cache_data(show_spinner=False)
def _stacked():
    return ev.stacked_table()


@st.cache_data(show_spinner=False)
def _onion():
    return ev.onion_table()


@st.cache_data(show_spinner=False)
def _drift():
    return ev.drift_table()


@st.cache_data(show_spinner=False)
def _starts():
    return ev.starts_table()


@st.cache_data(show_spinner=False)
def _lattice():
    return ev.lattice_table()


@st.cache_data(show_spinner=False)
def _toronto():
    return ev.toronto_table()


def _row_table(rows, with_n=True):
    """Zeilen einer Reihe als Tabelle (alle Zellen Text, damit Arrow keine gemischten Spalten sieht)."""
    names = {"fr": "Federmodell", "stress": "Stress", "tutte": "Tutte", "grid": "Gitterzeichnung"}
    out = []
    for r in rows:
        if not r.get("available"):
            continue
        out.append({
            "n": str(r["n"]), "Verfahren": names[r["method"]], "Kreuzungsfrei": f"{r['valid']} von {r['maps']}", "Kreuzungen (Median)": ev.num(r["crossings"]),
            "Auflösung (Quadratraster = 100 %)": ev.percent(r["ratio"]), "Kleinster Winkel": f"{ev.num(r['min_angle'], 2)}°", "Kantenverhältnis": ev.sci(r["edge_ratio"]) if r["edge_ratio"] >= 1000 else ev.num(r["edge_ratio"], 1),
        })
    return out


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in gzn_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🗺️ Graphen zeichnen – Garantie oder Lesbarkeit?")
st.markdown(
    """
Jeder **planare** Graph lässt sich ohne Kreuzungen zeichnen. Aber welches Verfahren tut es wirklich, und ist die Zeichnung dann **lesbar**?
Die Demo zeichnet Karten, Gitter, „Zwiebeln“ und Straßenausschnitte mit vier Verfahren, **Federmodell**, **Stress-Minimierung**, **Tutte-Einbettung** und **Gitterzeichnung** nach de Fraysseix, Pach und Pollack,
und misst jede Zeichnung: Kreuzungen (exakt gezählt), Auflösung, Winkel, Kantenverhältnis und Stress.
Viertes Stück der Linie *Planarität und Zeichnen*; es setzt auf die Einbettung des ersten Stücks (`planaritaet-demo`) und die Karten des dritten (`planar-separator-demo`) auf.

Weiter unten: **🔬 Vier Verfahren, ein Graph**, **🔬 Wann hilft die Garantie?**, **🔬 Warum Heuristiken Kreuzungen behalten**, **🔬 Die Gitterzeichnung Schritt für Schritt**, **🔬 Straßenausschnitt**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktionieren die Verfahren", expanded=True):
    st.markdown(
        """
1. **Federmodell (Fruchterman und Reingold):** Alle Knoten stoßen sich ab, jede Kante zieht ihre Enden zusammen; in kleinen Schritten, die mit der Zeit kürzer werden, rücken die Knoten dahin, wo die Kräfte sich aufheben. Das Verfahren kennt die Planarität nicht.
2. **Stress-Minimierung:** Der Abstand zweier Knoten in der Zeichnung soll der Zahl der Kanten auf dem kürzesten Weg entsprechen. Gelöst wird das hier mit der **Majorisierung** (ab klassischem MDS) und zum Vergleich mit dem **stochastischen Gradientenverfahren (SGD)**; beide enden im selben Wert.
3. **Tutte-Einbettung:** Die Knoten einer Außenfläche liegen auf einem Kreis, jeder andere Knoten genau im Schwerpunkt seiner Nachbarn. Ist der Graph planar und **3-fach zusammenhängend**, ist die Zeichnung kreuzungsfrei (Satz von Tutte). Die Rechnung läuft hier **exakt in rationalen Zahlen**, damit auch winzige Abstände richtig gezählt werden.
4. **Gitterzeichnung:** Der Graph wird durch zusätzliche Kanten in eine Triangulierung verwandelt, die Knoten in einer **kanonischen Ordnung** nacheinander oben aufgesetzt (Steigungen ±1) und früher gesetzte Teile nach rechts geschoben. Ergebnis: ganzzahlige Lage im Gitter (2n − 4) × (n − 2), kreuzungsfrei für **jeden** zweifach zusammenhängenden planaren Graphen.

Eine Zeichnung gilt hier nur als **gültig**, wenn sich keine zwei Kanten kreuzen **und** nichts aufeinanderliegt (Knoten auf Knoten, Knoten auf einer fremden Kante). **Lesbar** heißt: Der kleinste Knotenabstand je Diagonale ist nicht viel kleiner als bei einem gleichmäßigen Quadratraster mit gleich vielen Knoten (Schwellen 15 % und 2 %, festgelegt, nicht gemessen).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Graph**")
    net = st.selectbox("Graphtyp", list(C.NET_TYPES), key="net_select", format_func=C.NET_TYPES.get, help=C.NET_HELP)
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", help="Zahl der Knoten (Raster und Gitter werden auf ein Quadrat gerundet, die Zwiebel auf ein Vielfaches von 3, bei der Karte mit fehlenden Kanten bleibt der größte zweifach zusammenhängende Teil).")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Verfahren**")
    method = st.selectbox("Verfahren", list(C.METHODS), key="method_select", format_func=C.METHODS.get, help=C.METHOD_HELP)
    iterations = st.slider("Schritte", *bounds("iter_slider"), key="iter_slider", help="Nur Federmodell und Stress-Majorisierung: Zahl der Schritte (beim Federmodell kühlt die Temperatur in dieser Zahl von Schritten ab).")
    outer = st.selectbox("Außenfläche", list(C.OUTER), key="outer_select", format_func=C.OUTER.get, help=C.OUTER_HELP)

sync_query_params({"net_select": net, "n_slider": int(n), "seed_input": int(seed), "method_select": method, "iter_slider": int(iterations), "outer_select": outer})

settings = ev.Settings(net, int(n), int(seed), method, int(iterations), outer)
a = _analysis(settings)
inst = a.prep.inst
facts = a.prep.facts

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.measures is not None:
    mm = a.measures
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Knoten", f"{inst.n}", delta=f"{inst.m} Kanten", delta_color="off")
    m2.metric("Kreuzungen", ev.num(mm.crossings), delta="gültig" if mm.valid else "ungültig", delta_color="off",
              help=f"Paare von Kanten, die sich im Inneren schneiden (exakt gezählt). Gültig heißt: keine Kreuzung und keine Berührung; Berührungen (Knoten auf Knoten oder auf einer fremden Kante): {mm.touching}.")
    m3.metric("Auflösung", ev.percent_short(a.ratio), delta=ev.readable_grade(a.ratio), delta_color="off", help="Kleinster Knotenabstand je Diagonale, im Verhältnis zu einem gleichmäßigen Quadratraster mit gleich vielen Knoten (100 %).")
    m4.metric("Kleinster Winkel", f"{ev.num(mm.min_angle, 2)}°", delta=f"Kantenverh. {ev.sci(mm.edge_ratio) if mm.edge_ratio >= 1000 else ev.num(mm.edge_ratio, 1)}", delta_color="off",
              help="Kleinster Winkel zwischen zwei Kanten am selben Knoten; darunter das Verhältnis der längsten zur kürzesten Kante.")
kind, text = ev.verdict_drawing(a)
getattr(st, kind)(text)
if a.measures is not None:
    st.plotly_chart(
        build_drawing(a.P, inst.edges, a.crossing_pairs, a.outer, title=f"{inst.name}, {C.METHODS[method]}: {ev.plural(a.measures.crossings, 'Kreuzung', 'Kreuzungen')}"),
        width="stretch", key="main_chart",
    )
    st.caption(
        "Rote Kanten kreuzen sich, die gelbe Linie ist die Außenfläche der Tutte-Einbettung. "
        f"Der Graph ist {'planar' if facts.planar else 'nicht planar'}, {'zweifach' if facts.biconnected else 'nicht zweifach'} zusammenhängend"
        + (f" und {'3-fach' if facts.triconnected else 'nicht 3-fach'} zusammenhängend." if facts.planar and facts.biconnected else ".")
        + (f" Stress (0 = Weglängen genau getroffen): {ev.stress_text(a.measures.stress)}." if a.measures.stress is not None else "")
    )

st.markdown("---")

# --- Vier Verfahren, ein Graph -------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Vier Verfahren, ein Graph")
st.caption("Derselbe Graph (Einstellungen links, Schritte und Außenfläche wie dort) mit allen vier Verfahren.")
with st.spinner("Zeichne mit allen vier Verfahren..."):
    cmp_all = _compare(settings)
names = {"fr": "Federmodell", "stress": "Stress (Majorisierung)", "tutte": "Tutte-Einbettung", "grid": "Gitterzeichnung"}
table = []
for key in ev.COMPARED:
    r = cmp_all[key]
    if r.measures is None:
        table.append({"Verfahren": names[key], "Kreuzungen": "-", "Berührungen": "-", "Gültig": "nicht möglich", "Auflösung": "-", "Kleinster Winkel": "-", "Kantenverhältnis": "-", "Stress": "-"})
        continue
    mm = r.measures
    table.append({
        "Verfahren": names[key], "Kreuzungen": ev.num(mm.crossings), "Berührungen": ev.num(mm.touching), "Gültig": "ja" if mm.valid else "nein", "Auflösung": ev.percent(r.ratio),
        "Kleinster Winkel": f"{ev.num(mm.min_angle, 2)}°", "Kantenverhältnis": ev.sci(mm.edge_ratio) if mm.edge_ratio >= 1000 else ev.num(mm.edge_ratio, 1), "Stress": ev.stress_text(mm.stress),
    })
st.dataframe(table, hide_index=True, width="stretch")
left, right = st.columns(2)
for slot, keys in ((left, ("fr", "tutte")), (right, ("stress", "grid"))):
    with slot:
        for key in keys:
            r = cmp_all[key]
            if r.measures is None:
                st.info(f"{names[key]}: {r.note}")
            else:
                st.plotly_chart(build_drawing(r.P, inst.edges, r.crossing_pairs, r.outer, title=f"{names[key]}: {ev.plural(r.measures.crossings, 'Kreuzung', 'Kreuzungen')}", height=340), width="stretch", key=f"cmp_{key}")

st.markdown("---")

# --- Wann hilft die Garantie? ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Wann hilft die Garantie?")
st.markdown(
    "Tutte und Gitterzeichnung sind garantiert kreuzungsfrei, die zwei anderen nicht. Dafür hat jede Garantie einen Graphtyp, auf dem die Zeichnung kaum lesbar ist. "
    f"Jede Reihe läuft über {len(C.SWEEP_SEEDS)} feste Karten und gibt Mediane an."
)
tab_maps, tab_stacked, tab_onion = st.tabs(["Karten", "Gestapelte Triangulierungen", "Zwiebel"])
with tab_maps:
    st.caption(f"Delaunay-Karten mit n = {', '.join(str(x) for x in C.MAP_NS)}.")
    if st.button("Karten messen", key="maps_start"):
        st.session_state["maps_done"] = True
    if st.session_state.get("maps_done"):
        with st.spinner("Zeichne und messe die Karten..."):
            rows = _maps()
        getattr(st, ev.verdict_maps(rows)[0])(ev.verdict_maps(rows)[1])
        st.dataframe(_row_table(rows), hide_index=True, width="stretch")
        st.plotly_chart(build_ratio(rows, "Knoten n", "Auflösung auf Karten"), width="stretch", key="chart_maps")
with tab_stacked:
    st.caption(f"Gestapelte Triangulierungen mit n = {', '.join(str(x) for x in C.MAP_NS)}: von einem Dreieck aus wird immer wieder eine Fläche durch einen neuen Knoten geteilt.")
    if st.button("Gestapelte Triangulierungen messen", key="stacked_start"):
        st.session_state["stacked_done"] = True
    if st.session_state.get("stacked_done"):
        with st.spinner("Zeichne und messe..."):
            rows = _stacked()
        getattr(st, ev.verdict_hard(rows, "Gestapelte Triangulierung")[0])(ev.verdict_hard(rows, "Gestapelte Triangulierung")[1])
        st.dataframe(_row_table(rows), hide_index=True, width="stretch")
        st.plotly_chart(build_ratio(rows, "Knoten n", "Auflösung auf gestapelten Triangulierungen"), width="stretch", key="chart_stacked")
with tab_onion:
    st.caption(f"Zwiebel mit k = {', '.join(str(k) for k in C.ONION_KS)} Dreiecken ineinander (n = 3k). Bis k = {C.ONION_MAX_K} sind die Abstände noch auflösbar.")
    if st.button("Zwiebel messen", key="onion_start"):
        st.session_state["onion_done"] = True
    if st.session_state.get("onion_done"):
        with st.spinner("Zeichne und messe..."):
            rows = _onion()
        getattr(st, ev.verdict_hard(rows, "Zwiebel")[0])(ev.verdict_hard(rows, "Zwiebel")[1])
        st.dataframe(_row_table(rows), hide_index=True, width="stretch")
        st.plotly_chart(build_ratio(rows, "Knoten n (= 3k)", "Auflösung auf der Zwiebel"), width="stretch", key="chart_onion")

st.markdown("---")

# --- Warum Heuristiken Kreuzungen behalten ---------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Warum Heuristiken Kreuzungen behalten")
st.markdown(
    "Das Federmodell sucht ein Kräftegleichgewicht, die Stress-Minimierung die Weglängen: **keines von beiden bekommt die Planarität mit**. "
    "Zwei Messungen: Wie viele Kreuzungen bleiben beim Federmodell nach vielen Startlagen, und was passiert mit einer kreuzungsfreien Tutte-Zeichnung, wenn man sie nach Stress optimiert?"
)
tab_starts, tab_drift, tab_lattice = st.tabs(["Federmodell, viele Starts", "Stress ab Tutte-Start", "Gitter: das Federmodell faltet sich"])
with tab_starts:
    st.caption(f"Delaunay-Karten mit n = {', '.join(str(x) for x in C.STARTS_NS)}; je Karte {C.STARTS_COUNT} zufällige Startlagen, gezählt wird die beste.")
    if st.button("Starts messen (etwa eine halbe Minute)", key="starts_start"):
        st.session_state["starts_done"] = True
    if st.session_state.get("starts_done"):
        with st.spinner("Federmodell mit vielen Startlagen..."):
            rows = _starts()
        getattr(st, ev.verdict_starts(rows)[0])(ev.verdict_starts(rows)[1])
        st.dataframe(
            [{"n": str(r["n"]), "Karten": str(r["maps"]), "Starts je Karte": str(r["starts"]), "Median aller Starts": ev.num(r["median"]), "Beste Karte (wenigste Kreuzungen)": ev.num(r["best_min"]),
              "Schlechteste Karte (je beste von 20)": ev.num(r["best_max"]), "Karten mit 0": str(r["zero_maps"])} for r in rows],
            hide_index=True, width="stretch",
        )
with tab_drift:
    st.caption(f"Delaunay-Karten mit n = {', '.join(str(x) for x in C.DRIFT_NS)}: Start ist die Tutte-Zeichnung (0 Kreuzungen), dann Schritte der Stress-Majorisierung.")
    if st.button("Stress ab Tutte messen", key="drift_start"):
        st.session_state["drift_done"] = True
    if st.session_state.get("drift_done"):
        with st.spinner("Majorisierung ab der Tutte-Zeichnung..."):
            rows = _drift()
        getattr(st, ev.verdict_drift(rows)[0])(ev.verdict_drift(rows)[1])
        st.plotly_chart(build_drift(rows), width="stretch", key="chart_drift")
with tab_lattice:
    st.caption(f"Gitter {C.LATTICE_K} × {C.LATTICE_K} (n = {C.LATTICE_K ** 2}): {C.LATTICE_STARTS} Läufe des Federmodells mit verschiedenen Startlagen gegen einen Lauf der Stress-Minimierung.")
    if st.button("Gitter messen (etwa zehn Sekunden)", key="lattice_start"):
        st.session_state["lattice_done"] = True
    if st.session_state.get("lattice_done"):
        with st.spinner("Federmodell auf dem Gitter..."):
            row = _lattice()
        getattr(st, ev.verdict_lattice(row)[0])(ev.verdict_lattice(row)[1])

st.markdown("---")

# --- Die Gitterzeichnung Schritt für Schritt -------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Die Gitterzeichnung Schritt für Schritt")
grid_a = _analysis(ev.Settings(net, int(n), int(seed), "grid", int(iterations), outer))
if grid_a.measures is None:
    st.info(grid_a.note)
elif grid_a.grid is None or not grid_a.grid.steps and inst.n > 3:
    st.info(f"Zum Abspielen ist die Größe auf n ≤ {C.PLAYBACK_MAX_N} begrenzt (hier n = {inst.n}); die fertige Zeichnung steht oben unter „Vier Verfahren, ein Graph“.")
else:
    gd = grid_a.grid
    total = len(gd.steps)
    st.markdown(
        f"Der Graph wird mit {len(gd.chords)} zusätzlichen Kanten (gestrichelt) zur Triangulierung ergänzt. Die ersten drei Knoten {gd.order[0]}, {gd.order[1]}, {gd.order[2]} stehen bei (0, 0), (2, 0), (1, 1); "
        "jeder weitere Knoten wird oben auf die Nachbarn seiner Außenlinie gesetzt, vorher rücken die Teile dazwischen um 1 und alles rechts davon um 2 nach rechts."
    )
    step = st.slider("Schritt", 0, total, total, key="grid_step", help="0 = nur die ersten drei Knoten; Maximum = fertige Zeichnung.") if total else 0
    if step == 0:
        placed_pos = {gd.order[0]: (0, 0), gd.order[1]: (2, 0), gd.order[2]: (1, 1)}
        highlight = set(gd.order[:3])
        info = "Die ersten drei Knoten bilden das Außendreieck."
    else:
        s = gd.steps[step - 1]
        placed_pos = s.positions
        highlight = {s.vertex, s.left, s.right}
        info = (
            f"Schritt {step}: Knoten {s.vertex} wird über den Nachbarn {s.left} (links) und {s.right} (rechts) eingefügt; "
            f"{ev.plural(len(s.shifted_one), 'Knoten rückt', 'Knoten rücken')} um 1, {ev.plural(len(s.shifted_two), 'Knoten rückt', 'Knoten rücken')} um 2 nach rechts."
        )
    Pgrid = [placed_pos.get(v, (0, 0)) for v in range(inst.n)]
    st.plotly_chart(
        build_drawing(Pgrid, inst.edges, (), None, title=f"Gitterzeichnung nach {step} von {total} Einfügeschritten", height=420, placed=set(placed_pos), extra_edges=gd.chords, highlight=highlight),
        width="stretch", key="chart_grid_steps",
    )
    st.caption(info + f" Gitter am Ende: {gd.width} × {gd.height}, Schranke (2n − 4) × (n − 2) = {2 * inst.n - 4} × {inst.n - 2}.")

st.markdown("---")

# --- Straßenausschnitt -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Straßenausschnitt")
st.markdown(
    "Die n Knoten um die Mitte der Toronto-Karte (OpenStreetMap, ODbL). Die geografische Lage ist selbst nicht kreuzungsfrei, weil Straßen als gerade Strecken gezeichnet werden; "
    "zwischen 160 und 170 Knoten ist der Ausschnitt auch als Graph nicht mehr planar, und kein Verfahren kann ihn kreuzungsfrei zeichnen. Auch der Ausschnitt mit 100 Knoten ist nicht zweifach zusammenhängend, Tutte und Gitterzeichnung gibt es für keinen der drei also nicht."
)
if st.button("Straßenausschnitte messen", key="toronto_start"):
    st.session_state["toronto_done"] = True
if st.session_state.get("toronto_done"):
    with st.spinner("Zeichne die Ausschnitte..."):
        rows = _toronto()
    getattr(st, ev.verdict_toronto(rows)[0])(ev.verdict_toronto(rows)[1])
    st.dataframe(
        [{"n": str(r["n"]), "Kanten": str(r["m"]), "Planar": "ja" if r["planar"] else "nein", "Zweifach zusammenhängend": "ja" if r["biconnected"] else "nein",
          "Kreuzungen der geografischen Lage": str(r["geography"]), "Federmodell": str(r["fr"]), "Stress": str(r["stress"])} for r in rows],
        hide_index=True, width="stretch",
    )

st.markdown("---")

# --- Wo die Annahmen enden -------------------------------------------------------------------------------------------------------------------

st.markdown("## 🚧 Wo die Annahmen enden")
st.markdown(
    f"""
- **Lesbarkeit ist hier die Auflösung.** Sie misst nur den kleinsten Knotenabstand. Ob eine Zeichnung gefällt, hängt auch von Symmetrie, Kantenlängen und Platz für Beschriftungen ab; die Schwellen {int(C.READABLE_GOOD * 100)} % und {int(C.READABLE_TIGHT * 100)} % sind festgelegt, nicht aus Messungen abgeleitet.
- **Tutte braucht 3-fachen Zusammenhang.** Fehlen Kanten, fallen Knoten aufeinander; der einfache Kreuzungszähler meldet dann null. Darum zählt die Demo auch Berührungen.
- **Die Gitterzeichnung** wird nur für zweifach zusammenhängende planare Graphen gerechnet (sonst bräuchte man eine globale Einbettung über die Blöcke hinweg). Die Schranke (2n − 4) × (n − 2) wird hier in jedem Lauf genau erreicht; Schnyders Verfahren mit einem kleineren Gitter ist nicht nachgebaut.
- **Federmodell und Stress hängen von Start und Schritten ab.** Die Zahlen sind Mediane über feste Karten und Bänder, keine festen Werte. Stress-Majorisierung und SGD enden im selben Wert; SGD ist hier in reinem Python bis n = {C.SGD_MAX_N} vorgesehen.
- **Die Zwiebel ist der ungünstigste Fall für Tutte, nicht der typische.** Auf Karten mit großer Außenfläche ist Tutte so lesbar wie das Federmodell. Die Zehnerpotenzen der Zwiebel sind hier selbst gemessen, nicht aus der Literatur.
- **Rechengrenzen:** n ≤ {C.N_MAX}; bei der Zwiebel liegen die Knoten ab k = {C.ONION_MAX_K} unter 10⁻⁸ der Zeichnung auseinander, die Anzeige kann sie nicht mehr trennen.
        """
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        """
| Symbol | Bedeutung |
|---|---|
| $G=(V,E)$, $n=\\lvert V\\rvert$ | Graph und Knotenzahl |
| $d_{ij}$ | Weglänge (Zahl der Kanten) zwischen $i$ und $j$ |
| $p_i\\in\\mathbb{R}^2$ | Lage des Knotens $i$ in der Zeichnung |
| $k$ | Wunschlänge $\\sqrt{1/n}$ im Federmodell |

**Federmodell (Fruchterman und Reingold 1991).** Auf Knoten $i$ wirkt die Kraft

$$F_i=\\sum_{j\\ne i}\\frac{k^2}{\\lVert p_i-p_j\\rVert}\\,\\frac{p_i-p_j}{\\lVert p_i-p_j\\rVert}-\\sum_{j:\\,ij\\in E}\\frac{\\lVert p_i-p_j\\rVert^2}{k}\\,\\frac{p_i-p_j}{\\lVert p_i-p_j\\rVert}.$$

Zwei durch eine Kante verbundene Knoten sind im Gleichgewicht bei $\\lVert p_i-p_j\\rVert=k$.

**Stress.**

$$\\mathrm{stress}(p)=\\sum_{i<j}w_{ij}\\,\\bigl(\\lVert p_i-p_j\\rVert-d_{ij}\\bigr)^2,\\qquad w_{ij}=d_{ij}^{-2}.$$

Die Demo gibt ihn normiert aus: nach bestem Maßstab $s$ und geteilt durch $\\sum w_{ij}d_{ij}^2$. Ein Majorisierungsschritt ist $p\\leftarrow L_w^{+}\\,B(p)\\,p$ mit dem gewichteten Laplace-Operator $L_w$; der Stress wird dabei nicht größer.

**Tutte-Einbettung (1963).** Die Knoten der Außenfläche liegen auf einem Kreis, für jeden anderen Knoten gilt

$$p_i=\\frac{1}{\\deg(i)}\\sum_{j:\\,ij\\in E}p_j.$$

Dieses lineare System hat eine eindeutige Lösung; ist $G$ planar und 3-fach zusammenhängend, ist die Zeichnung kreuzungsfrei und jede Fläche ein konvexes Vieleck.

**Gitterzeichnung (de Fraysseix, Pach und Pollack 1990).** Ist $v_1,\\dots,v_n$ eine kanonische Ordnung der Triangulierung, so wird $v_k$ über den Nachbarn $w_p,\\dots,w_q$ der Außenlinie $C_{k-1}$ mit Steigungen $\\pm1$ gesetzt:

$$x_{v_k}=\\tfrac12\\,(x_{w_q}+x_{w_p}+y_{w_q}-y_{w_p}),\\qquad y_{v_k}=\\tfrac12\\,(x_{w_q}-x_{w_p}+y_{w_q}+y_{w_p}).$$

Vorher rücken die Knoten zwischen $w_p$ und $w_q$ um 1 und alle rechts davon um 2 nach rechts; die Breite wächst je Knoten um 2, das Ergebnis liegt im Gitter $(2n-4)\\times(n-2)$.

**Literatur.** W. T. Tutte, *How to draw a graph*, Proceedings of the London Mathematical Society 13 (1963), 743–768. T. M. J. Fruchterman, E. M. Reingold, *Graph drawing by force-directed placement*, Software: Practice and Experience 21(11) (1991), 1129–1164. H. de Fraysseix, J. Pach, R. Pollack, *How to draw a planar graph on a grid*, Combinatorica 10 (1990), 41–51 (Gittergröße (2n − 4) × (n − 2) laut Übersichtsarbeit arXiv 1410.1006). Zheng, Pawar, Goodman, *Graph drawing by stochastic gradient descent* (arXiv 1710.04626); Börsig, Brandes, Pasztor (arXiv 2008.10376) widersprechen dem behaupteten Qualitätsvorteil gegenüber der Majorisierung. Die Angaben wurden per Suche gegen Bibliografie-Einträge und Zusammenfassungen geprüft, die Originalarbeiten nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
