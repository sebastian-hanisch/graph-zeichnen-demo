"""Auswertung: eine Zeichnung samt Kennzahlen, Urteile (reine Funktionen) und die Messreihen der App.

`prepare` bestimmt einmal je Graph Einbettung, Zusammenhang und Weglängen; `draw_with` zeichnet mit einem Verfahren und misst; die Reihen (`sweep`, `drift_table`, `starts_table`,
`toronto_table`) laufen über feste Karten (`C.SWEEP_SEEDS`), nie über einen einzelnen Seed."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

import gzn_algorithm as A
import gzn_constants as C
import gzn_geometry as G
import gzn_planar as Pl
import gzn_scenario as S

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def num(x: float, digits: int = 0) -> str:
    """Zahl im deutschen Format: Tausenderpunkt, Dezimalkomma."""
    return f"{x:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def plural(count: int, singular: str, plural_form: str) -> str:
    return f"{count} {singular if count == 1 else plural_form}"


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenzschreibweise, z. B. 7,5·10⁻¹¹; 0 bleibt 0."""
    if x == 0:
        return "0"
    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / 10 ** exponent
    if round(mantissa, digits) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{num(mantissa, digits)}·10{str(exponent).translate(_SUPERSCRIPT)}"


def percent(ratio: float) -> str:
    """Anteil als Prozentangabe; unter 0,1 % in Zehnerpotenzen."""
    p = 100.0 * ratio
    return f"{num(p, 1)} %" if p >= 0.1 else f"{sci(p)} %"


def percent_short(ratio: float) -> str:
    """Wie `percent`, aber unter 0,1 % nur „< 0,1 %“ (für schmale Kennzahlfelder)."""
    p = 100.0 * ratio
    return f"{num(p, 1)} %" if p >= 0.1 else "< 0,1 %"


def reference_resolution(n: int) -> float:
    """Auflösung (kleinster Knotenabstand je Diagonale) eines gleichmäßigen Quadratrasters mit n Knoten: 1/(√2 (√n − 1)); der Maßstab für „lesbar“."""
    return 1.0 / (math.sqrt(2.0) * (math.sqrt(n) - 1.0)) if n > 1 else 1.0


def readable_grade(ratio: float) -> str:
    """'gut' ab READABLE_GOOD, 'eng' ab READABLE_TIGHT, sonst 'kaum' (Verhältnis der Auflösung zu der eines Quadratrasters)."""
    if ratio >= C.READABLE_GOOD:
        return "gut"
    if ratio >= C.READABLE_TIGHT:
        return "eng"
    return "kaum"


@dataclass(frozen=True)
class Settings:
    net: str = C.DEFAULT_NET
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    method: str = C.DEFAULT_METHOD
    iterations: int = C.DEFAULT_ITER
    outer: str = C.DEFAULT_OUTER


@dataclass(frozen=True)
class Facts:
    planar: bool
    biconnected: bool          # ein einziger Block deckt alle Knoten (n ≥ 3)
    triconnected: bool         # nur bestimmt, wenn planar und zweifach zusammenhängend
    faces: int | None


@dataclass(frozen=True)
class Prepared:
    inst: S.Instance
    emb: Pl.Embedding
    facts: Facts
    D: np.ndarray


def prepare(inst: S.Instance) -> Prepared:
    emb = Pl.embed(inst.n, inst.edges)
    block = emb.single_block
    tri = emb.planar and block is not None and Pl.is_triconnected(inst.n, inst.edges)
    facts = Facts(emb.planar, block is not None, tri, len(block.faces) if emb.planar and block is not None else None)
    return Prepared(inst, emb, facts, A.hop_distances(inst.n, inst.edges))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    prep: Prepared
    pos: tuple[tuple, ...] | None                 # exakt (Fraction) bei Tutte, ganzzahlig bei der Gitterzeichnung, sonst float
    P: np.ndarray | None                          # dieselbe Lage in float, für die Anzeige
    measures: G.Measures | None
    crossing_pairs: tuple[tuple[int, int], ...]
    outer: tuple[int, ...] | None
    grid: A.GridDrawing | None
    note: str | None                              # warum es keine Zeichnung gibt

    @property
    def ratio(self) -> float | None:
        """Auflösung im Verhältnis zu der eines gleichmäßigen Quadratrasters."""
        return None if self.measures is None else self.measures.resolution / reference_resolution(self.prep.inst.n)


def _unavailable(prep: Prepared, settings: Settings, note: str) -> Analysis:
    return Analysis(settings, prep, None, None, None, (), None, None, note)


def draw_with(prep: Prepared, settings: Settings) -> Analysis:
    """Zeichnet den Graphen mit dem Verfahren der Einstellungen und misst die Zeichnung."""
    inst, facts, D = prep.inst, prep.facts, prep.D
    n, edges = inst.n, inst.edges
    method = settings.method
    pos: tuple[tuple, ...]
    outer = None
    grid = None
    if method == "fr":
        start = np.random.default_rng(settings.seed + 1).random((n, 2))
        pos = _as_floats(A.fruchterman_reingold(start, edges, settings.iterations))
    elif method == "stress":
        pos = _as_floats(A.stress_majorization(D, A.classical_mds(D), settings.iterations))
    elif method == "sgd":
        if n > C.SGD_MAX_N:
            return _unavailable(prep, settings, f"SGD ist in dieser Demo bis n = {C.SGD_MAX_N} Knoten vorgesehen (reines Python, bei n = {n} zu langsam); die Majorisierung liefert denselben Stress.")
        pos = _as_floats(A.sgd_stress(D, settings.seed, C.SGD_ITERATIONS))
    elif method in ("tutte", "grid"):
        if not facts.planar:
            return _unavailable(prep, settings, "Der Graph ist nicht planar (das Fragmentverfahren findet keine Einbettung): Tutte-Einbettung und Gitterzeichnung setzen Planarität voraus.")
        block = prep.emb.single_block
        if block is None:
            return _unavailable(prep, settings, "Der Graph ist planar, aber nicht zweifach zusammenhängend (Trennknoten oder mehrere Teile): Tutte-Einbettung und Gitterzeichnung brauchen eine Fläche als Rand.")
        if method == "tutte":
            outer = A.pick_outer_face(block.faces, settings.outer)
            pos = tuple(A.tutte_embedding(n, edges, outer))
        else:
            grid = A.fpp_drawing(n, edges, block.faces, record=n <= C.PLAYBACK_MAX_N)
            pos = tuple(grid.pos)
    else:
        raise ValueError(method)
    P = G.to_float(pos)
    pairs = tuple(G.crossings(pos, edges))
    true = np.array(inst.true) if inst.true is not None else None
    measures = G.measure(pos, edges, A.stress_value(P, D), true, pairs)
    return Analysis(settings, prep, pos, P, measures, pairs, outer, grid, None)


def _as_floats(P: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in P)


def analyse(settings: Settings) -> Analysis:
    return draw_with(prepare(S.make_instance(settings.net, settings.n, settings.seed)), settings)


COMPARED = ("fr", "stress", "tutte", "grid")


def compare_all(settings: Settings) -> dict[str, Analysis]:
    """Dieselbe Instanz mit den vier Verfahren (ohne SGD); der Graph wird nur einmal vorbereitet."""
    prep = prepare(S.make_instance(settings.net, settings.n, settings.seed))
    return {m: draw_with(prep, Settings(settings.net, settings.n, settings.seed, m, settings.iterations, settings.outer)) for m in COMPARED}


# --- Urteile (rein: die App ruft st.<Art>(Text), der Test prüft jede Zahl) ---------------------------------------------------------------------


def verdict_drawing(a: Analysis) -> tuple[str, str]:
    """Urteil zur einen Zeichnung: ob das Verfahren hier garantiert kreuzungsfrei ist, ob die Zeichnung gültig ist und wie eng die Knoten liegen."""
    if a.note is not None:
        return "info", a.note
    m, f, method = a.measures, a.prep.facts, a.settings.method
    ratio = a.ratio
    grade = readable_grade(ratio)
    tight = (f"Die Knoten sind {percent(ratio)} so weit auseinander wie bei einem gleichmäßigen Raster aus {a.prep.inst.n} Knoten (kleinster Abstand je Diagonale: {sci(m.resolution) if m.resolution < 0.001 else num(m.resolution, 4)}).")
    if method in ("fr", "stress", "sgd"):
        if m.crossings > 0:
            planar = "Der Graph ist planar, eine kreuzungsfreie Zeichnung gibt es." if f.planar else "Der Graph ist nicht planar, ganz ohne Kreuzungen geht es nicht."
            return "warning", f"**{plural(m.crossings, 'Kreuzung', 'Kreuzungen')}.** Das Verfahren kennt die Planarität nicht und garantiert keine kreuzungsfreie Zeichnung. {planar}"
        kind = "success" if grade == "gut" else "info"
        return kind, f"Diesmal ohne Kreuzung, aber das Verfahren garantiert das nicht. {tight}"
    if not m.valid:
        what = []
        if m.touching:
            what.append(f"{plural(m.touching, 'Knoten liegt', 'Knoten liegen')} auf einem anderen Knoten oder einer fremden Kante")
        if m.crossings:
            what.append(plural(m.crossings, "Kreuzung", "Kreuzungen"))
        zero = " Der einfache Kreuzungszähler allein würde hier null melden." if m.crossings == 0 else ""
        return "warning", f"**Ungültige Zeichnung:** {' und '.join(what)}. Die Garantie gilt nicht, weil der Graph nicht 3-fach zusammenhängend ist.{zero}"
    if method == "grid":
        inst_n = a.prep.inst.n
        width, height = a.grid.width, a.grid.height
        head = f"**Kreuzungsfrei im Gitter {width} × {height}** (Schranke (2n − 4) × (n − 2) = {2 * inst_n - 4} × {inst_n - 2}), garantiert für jeden zweifach zusammenhängenden planaren Graphen."
    elif f.triconnected:
        head = "**Kreuzungsfrei**, wie der Satz von Tutte es für 3-fach zusammenhängende planare Graphen garantiert."
    else:
        head = "Diesmal gültig, aber ohne Garantie: der Graph ist nicht 3-fach zusammenhängend."
    kind = {"gut": "success", "eng": "info", "kaum": "warning"}[grade]
    tail = {"gut": "Gut lesbar.", "eng": "Eng.", "kaum": "Kaum lesbar: Knoten und Kanten fallen in der Anzeige zusammen."}[grade]
    return kind, f"{head} {tight} {tail}"


def stress_text(value: float | None) -> str:
    return "-" if value is None else num(value, 3)


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------------


def _median(values):
    values = list(values)
    return statistics.median(values) if values else None


def summarize(net: str, method: str, analyses: list[Analysis]) -> dict:
    """Eine Zeile der Reihe: Mediane über die Karten, in denen das Verfahren eine Zeichnung liefert."""
    ok = [a for a in analyses if a.measures is not None]
    row = {"net": net, "method": method, "maps": len(analyses), "available": len(ok), "n": int(statistics.median([a.prep.inst.n for a in analyses])) if analyses else 0}
    if not ok:
        return row | {"valid": 0}
    ms = [a.measures for a in ok]
    cr = [m.crossings for m in ms]
    row |= {
        "valid": sum(1 for m in ms if m.valid),
        "crossings": _median(cr), "crossings_min": min(cr), "crossings_max": max(cr),
        "touching": _median(m.touching for m in ms),
        "resolution": _median(m.resolution for m in ms),
        "ratio": _median(a.ratio for a in ok),
        "min_angle": _median(m.min_angle for m in ms),
        "edge_ratio": _median(m.edge_ratio for m in ms),
        "area": _median(m.area for m in ms),
        "stress": _median(m.stress for m in ms),
        "fidelity": _median(m.fidelity for m in ms if m.fidelity is not None),
    }
    return row


def sweep(net: str, ns, seeds, methods, iterations: int = C.DEFAULT_ITER, outer: str = C.DEFAULT_OUTER) -> list[dict]:
    """Für jede Größe und jedes Verfahren eine Zeile (Medianwerte über die Karten `seeds`; Zwiebel und Gitter haben nur eine Instanz je Größe)."""
    rows = []
    for n in ns:
        results: dict[str, list[Analysis]] = {m: [] for m in methods}
        for seed in seeds:
            prep = prepare(S.make_instance(net, n, seed))
            for m in methods:
                results[m].append(draw_with(prep, Settings(net, n, seed, m, iterations, outer)))
        rows += [summarize(net, m, results[m]) for m in methods]
    return rows


def maps_table(ns=C.MAP_NS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    return sweep("delaunay", ns, seeds, COMPARED)


def stacked_table(ns=C.MAP_NS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    return sweep("stacked", ns, seeds, COMPARED)


def onion_table(ks=C.ONION_KS) -> list[dict]:
    return sweep("onion", [3 * k for k in ks], (C.SWEEP_SEEDS[0],), COMPARED)


def drift_table(ns=C.DRIFT_NS, steps=C.DRIFT_STEPS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Kreuzungen der Stress-Majorisierung ab der Tutte-Zeichnung nach den angegebenen Schritten (Median, kleinster und größter Wert über die Karten)."""
    rows = []
    for n in ns:
        per_step: dict[int, list[int]] = {s: [] for s in steps}
        for seed in seeds:
            prep = prepare(S.make_delaunay(n, seed))
            a = draw_with(prep, Settings("delaunay", n, seed, "tutte"))
            edges = prep.inst.edges
            E = np.array(edges)
            scale = 1.0 / float(np.linalg.norm(a.P[E[:, 0]] - a.P[E[:, 1]], axis=1).mean())
            trace = A.majorization_trace(prep.D, a.P * scale, steps)
            for s in steps:
                per_step[s].append(len(a.crossing_pairs) if s == 0 else len(G.crossings([tuple(p) for p in trace[s]], edges)))
        rows += [{"n": n, "step": s, "crossings": _median(v), "crossings_min": min(v), "crossings_max": max(v)} for s, v in per_step.items()]
    return rows


def starts_table(ns=C.STARTS_NS, count: int = C.STARTS_COUNT, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Federmodell mit `count` verschiedenen Startlagen je Karte: die kleinste Kreuzungszahl je Karte und der Median aller Starts."""
    rows = []
    for n in ns:
        best, every = [], []
        for seed in seeds:
            inst = S.make_delaunay(n, seed)
            cr = []
            for s in range(count):
                start = np.random.default_rng(1000 * seed + s).random((n, 2))
                pos = _as_floats(A.fruchterman_reingold(start, inst.edges))
                cr.append(len(G.crossings(pos, inst.edges)))
            best.append(min(cr))
            every += cr
        rows.append({"n": n, "maps": len(seeds), "starts": count, "best": best, "best_min": min(best), "best_max": max(best), "zero_maps": sum(1 for b in best if b == 0), "median": _median(every)})
    return rows


def lattice_table(k: int = C.LATTICE_K, starts: int = C.LATTICE_STARTS) -> dict:
    """Federmodell mit `starts` Startlagen auf dem Gitter k × k (Kreuzungen je Lauf) gegen die Stress-Minimierung (ein Lauf, deterministisch)."""
    inst = S.make_grid(k * k, False)
    fr = []
    for s in range(starts):
        start = np.random.default_rng(1000 + s).random((inst.n, 2))
        fr.append(len(G.crossings(_as_floats(A.fruchterman_reingold(start, inst.edges)), inst.edges)))
    stress = draw_with(prepare(inst), Settings("lattice", inst.n, 0, "stress")).measures.crossings
    return {"k": k, "n": inst.n, "starts": starts, "fr": fr, "fr_zero": sum(1 for c in fr if c == 0), "stress": stress}


def toronto_table(ns=C.TORONTO_NS) -> list[dict]:
    """Straßenausschnitte: Kreuzungen der geografischen Lage, des Federmodells und der Stress-Majorisierung; ob Tutte und Gitterzeichnung möglich sind."""
    rows = []
    for n in ns:
        inst = S.make_toronto_ball(n)
        prep = prepare(inst)
        edges = inst.edges
        geo = len(G.crossings(inst.true, edges))
        fr = draw_with(prep, Settings("toronto", n, C.DEFAULT_SEED, "fr"))
        st = draw_with(prep, Settings("toronto", n, C.DEFAULT_SEED, "stress"))
        rows.append({
            "n": inst.n, "m": inst.m, "planar": prep.facts.planar, "biconnected": prep.facts.biconnected, "geography": geo,
            "fr": fr.measures.crossings, "stress": st.measures.crossings,
        })
    return rows


# --- Urteile zu den Reihen ----------------------------------------------------------------------------------------------------------------------


def _row(rows, method, n=None):
    candidates = [r for r in rows if r["method"] == method and (n is None or r["n"] == n)]
    return max(candidates, key=lambda r: r["n"]) if candidates else None


def verdict_maps(rows: list[dict]) -> tuple[str, str]:
    """Karten: wer ist wie oft kreuzungsfrei, und wie eng liegen die Knoten (größte Größe der Reihe)."""
    n = max(r["n"] for r in rows)
    fr, st, tu, gr = (_row(rows, m, n) for m in COMPARED)
    return "info", (
        f"Bei n = {n} (Median über {fr['maps']} Karten): Federmodell {num(fr['crossings'])} Kreuzungen (kreuzungsfrei in {fr['valid']} von {fr['maps']} Karten), "
        f"Stress-Minimierung {num(st['crossings'])} ({st['valid']} von {st['maps']}); Tutte-Einbettung und Gitterzeichnung kreuzungsfrei in {tu['valid']} bzw. {gr['valid']} von {tu['maps']} Karten. "
        f"Die Tutte-Zeichnung liegt mit {percent(tu['ratio'])} der Auflösung eines Quadratrasters nahe am Federmodell ({percent(fr['ratio'])}), die Gitterzeichnung mit {percent(gr['ratio'])} deutlich darunter; "
        f"der kleinste Winkel an einem Knoten ist bei Tutte {num(tu['min_angle'], 1)}°, beim Federmodell {num(fr['min_angle'], 1)}°, bei der Gitterzeichnung {num(gr['min_angle'], 2)}°."
    )


def verdict_hard(rows: list[dict], label: str) -> tuple[str, str]:
    """Gestapelte Triangulierungen bzw. Zwiebel: Tutte gegen Gitterzeichnung bei der größten Größe der Reihe."""
    n = max(r["n"] for r in rows)
    fr, st, tu, gr = (_row(rows, m, n) for m in COMPARED)
    return "warning", (
        f"{label}, n = {n}: Die Tutte-Zeichnung ist in {tu['valid']} von {tu['maps']} Fällen kreuzungsfrei, aber der kleinste Knotenabstand je Diagonale ist nur {sci(tu['resolution'])} "
        f"({percent(tu['ratio'])} eines Quadratrasters) und das Verhältnis längste zu kürzester Kante {sci(tu['edge_ratio'])}. Die Gitterzeichnung hat {num(gr['resolution'], 4)} ({percent(gr['ratio'])}), "
        f"das Federmodell {num(fr['resolution'], 4)}, hat aber {num(fr['crossings'])} Kreuzungen."
    )


def verdict_drift(rows: list[dict]) -> tuple[str, str]:
    """Stress-Majorisierung ab Tutte: wie schnell die Kreuzungsfreiheit verloren geht (größte Größe)."""
    n = max(r["n"] for r in rows)
    mine = sorted((r for r in rows if r["n"] == n), key=lambda r: r["step"])
    first, last = mine[0], mine[-1]
    after = next((r for r in mine if r["step"] >= 10), last)
    return "warning", (
        f"Bei n = {n}: Start {num(first['crossings'])} Kreuzungen (Tutte), nach {plural(after['step'], 'Schritt', 'Schritten')} der Stress-Majorisierung {num(after['crossings'])}, "
        f"nach {plural(last['step'], 'Schritt', 'Schritten')} {num(last['crossings'])} (Median über die Karten, kleinster Wert {num(last['crossings_min'])}). Weglängen im Graphen kennen die Planarität nicht."
    )


def verdict_starts(rows: list[dict]) -> tuple[str, str]:
    """Federmodell mit vielen Starts: erreicht es je null Kreuzungen?"""
    total = sum(r["zero_maps"] for r in rows)
    maps = sum(r["maps"] for r in rows)
    parts = "; ".join(f"n = {r['n']}: kleinste Zahl je Karte {num(r['best_min'])} bis {num(r['best_max'])}" for r in rows)
    starts = rows[0]["starts"]
    return ("success" if total else "warning"), (
        f"Mit {starts} Startlagen je Karte erreicht das Federmodell in {total} von {maps} Karten null Kreuzungen ({parts})."
    )


def verdict_lattice(row: dict) -> tuple[str, str]:
    """Gitter: in wie vielen Läufen das Federmodell sich faltet (Kreuzungen), gegen die Stress-Minimierung."""
    folded = row["starts"] - row["fr_zero"]
    runs = ", ".join(str(c) for c in row["fr"])
    return ("warning" if folded else "success"), (
        f"Auf dem {row['k']} × {row['k']}-Gitter (n = {row['n']}) endet das Federmodell in {folded} von {row['starts']} Läufen mit Kreuzungen (je Lauf: {runs}); "
        f"die Stress-Minimierung endet mit {plural(row['stress'], 'Kreuzung', 'Kreuzungen')}: Im Gitter passen Weglängen und ebene Lage gut zusammen."
    )


def verdict_toronto(rows: list[dict]) -> tuple[str, str]:
    """Straßenausschnitte: die wahre Lage ist selbst nicht kreuzungsfrei; Tutte und Gitter gibt es nur für planare, zweifach zusammenhängende Ausschnitte."""
    parts = []
    for r in rows:
        planar = "planar" if r["planar"] else "nicht planar"
        parts.append(f"n = {r['n']} ({planar}): geografische Lage {num(r['geography'])} Kreuzungen, Federmodell {num(r['fr'])}, Stress {num(r['stress'])}")
    return "info", "; ".join(parts) + ". Die Karte der Straßen ist als Strecken gezeichnet; Kurven und Brücken erzeugen Kreuzungen, die der Graph selbst nicht hat."
