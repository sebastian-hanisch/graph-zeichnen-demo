"""Plotly-Figuren: Zeichnung eines Graphen (mit Kreuzungen und Außenfläche), Auflösung über die Größe, Kreuzungen über die Schritte. Alle Achsen fest (fixedrange)."""

from __future__ import annotations

import plotly.graph_objects as go

import gzn_constants as C


def _edge_trace(P, edges, color, width, dash=None, name=None):
    xs, ys = [], []
    for u, v in edges:
        xs += [P[u][0], P[v][0], None]
        ys += [P[u][1], P[v][1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", showlegend=False, name=name)


def _drawing_layout(fig, title, height, P, placed=None):
    pts = [P[v] for v in (placed if placed is not None else range(len(P)))]
    xs = [p[0] for p in pts] or [0.0]
    ys = [p[1] for p in pts] or [0.0]
    pad = 0.04 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=14)) if title else None,
        height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10), showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(visible=False, fixedrange=True, constrain="domain", range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, fixedrange=True, constrain="domain", scaleanchor="x", scaleratio=1, range=[min(ys) - pad, max(ys) + pad])
    return fig


def build_drawing(P, edges, crossing_pairs=(), outer=None, title=None, height=460, placed=None, extra_edges=(), highlight=None):
    """Die Zeichnung: Kanten grau, sich kreuzende Kanten rot, die Außenfläche (Tutte) gelb, Knoten als Punkte.

    `placed` beschränkt die Zeichnung auf diese Knoten (zum Abspielen), `extra_edges` sind gestrichelt (ergänzte Sehnen), `highlight` sind Knoten in der Außenfarbe."""
    shown = set(range(len(P))) if placed is None else set(placed)
    bad = {k for pair in crossing_pairs for k in pair}
    good_edges, bad_edges = [], []
    for k, (u, v) in enumerate(edges):
        if u in shown and v in shown:
            (bad_edges if k in bad else good_edges).append((u, v))
    fig = go.Figure()
    thin = len(shown) > 120
    if extra_edges:
        fig.add_trace(_edge_trace(P, [(u, v) for u, v in extra_edges if u in shown and v in shown], "rgba(150,150,150,0.35)", 1, dash="dot"))
    fig.add_trace(_edge_trace(P, good_edges, C.COLOR_EDGE, 1 if thin else 2))
    if bad_edges:
        fig.add_trace(_edge_trace(P, bad_edges, C.COLOR_EDGE_BAD, 2 if thin else 3))
    if outer:
        ring = list(outer) + [outer[0]]
        fig.add_trace(_edge_trace(P, [(a, b) for a, b in zip(ring, ring[1:])], C.COLOR_OUTER, 4))
    marked = set(highlight or ())
    nodes = sorted(shown)
    fig.add_trace(
        go.Scatter(
            x=[P[v][0] for v in nodes], y=[P[v][1] for v in nodes], mode="markers",
            marker=dict(color=[C.COLOR_OUTER if v in marked else C.COLOR_NODE for v in nodes], size=[12 if v in marked else (5 if thin else 8) for v in nodes], line=dict(color="white", width=0.5)),
            hovertext=[f"Knoten {v}" for v in nodes], hoverinfo="text",
        )
    )
    return _drawing_layout(fig, title, height, P, placed=sorted(shown))


def build_ratio(rows, x_label, title, methods=("fr", "stress", "tutte", "grid")):
    """Auflösung im Verhältnis zu einem gleichmäßigen Quadratraster über die Größe (logarithmische Achse), je Verfahren eine Linie."""
    names = {"fr": "Federmodell", "stress": "Stress", "tutte": "Tutte", "grid": "Gitterzeichnung"}
    colors = dict(zip(("fr", "stress", "tutte", "grid"), C.COLOR_PALETTE))
    fig = go.Figure()
    for m in methods:
        mine = sorted((r for r in rows if r["method"] == m and r.get("ratio") is not None), key=lambda r: r["n"])
        if not mine:
            continue
        fig.add_trace(go.Scatter(x=[r["n"] for r in mine], y=[max(r["ratio"], 1e-12) for r in mine], mode="lines+markers", name=names[m], line=dict(color=colors[m], width=3),
                                 hovertemplate="n = %{x}: %{y:.2e}<extra>" + names[m] + "</extra>"))
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)), height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25, x=0))
    fig.update_xaxes(title=x_label, fixedrange=True, type="log", tickmode="array", tickvals=sorted({r["n"] for r in rows}), ticktext=[str(x) for x in sorted({r["n"] for r in rows})])
    fig.update_yaxes(title="Auflösung / Quadratraster", fixedrange=True, type="log")
    return fig


def build_drift(rows):
    """Kreuzungen der Stress-Majorisierung ab der Tutte-Zeichnung über die Schritte (Median, Band Minimum bis Maximum), je Kartengröße eine Linie."""
    fig = go.Figure()
    labels = {s: str(s) for s in sorted({r["step"] for r in rows})}
    for i, n in enumerate(sorted({r["n"] for r in rows})):
        mine = sorted((r for r in rows if r["n"] == n), key=lambda r: r["step"])
        xs = [labels[r["step"]] for r in mine]
        color = C.COLOR_PALETTE[i % len(C.COLOR_PALETTE)]
        fig.add_trace(go.Scatter(x=xs, y=[r["crossings"] for r in mine], mode="lines+markers", name=f"n = {n}", line=dict(color=color, width=3),
                                 hovertemplate="nach %{x} Schritten: %{y:.0f}<extra>n = " + str(n) + "</extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25, x=0))
    fig.update_xaxes(title="Schritte der Stress-Majorisierung (Start = Tutte-Zeichnung)", fixedrange=True, type="category")
    fig.update_yaxes(title="Kreuzungen (Median)", fixedrange=True)
    return fig
