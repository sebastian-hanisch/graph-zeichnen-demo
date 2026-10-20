"""Konstanten: Regler-Grenzen, Rechengrenzen, feste Annahmen, Schwellen der Urteile, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 6, 200, 60                     # Knoten; Federmodell und exakte Tutte-Rechnung brauchen bei n = 200 etwa eine halbe Sekunde
SEED_MAX, DEFAULT_SEED = 999999, 101
SWEEP_SEEDS = tuple(range(100, 110))                     # feste Karten für alle Messreihen (10): nie ein einzelner Seed als Befund
ITER_MIN, ITER_MAX, DEFAULT_ITER = 1, 300, 300           # Schritte des Federmodells bzw. der Stress-Majorisierung
SGD_MAX_N = 100                                          # SGD in reinem Python: etwa 1 s bei n = 100, darüber zu langsam
SGD_ITERATIONS = 30
ONION_MAX_K = 12                                         # Zwiebel: ab etwa k = 12 liegen die Knoten unter 10⁻⁸ der Zeichnung auseinander; darüber nur mit Hinweis
PLAYBACK_MAX_N = 60                                      # Gitterzeichnung zum Abspielen: Lage nach jedem Schritt wird gespeichert

NET_TYPES = {
    "delaunay": "Karte (Delaunay-Triangulierung)",
    "thin": "Karte, ein Fünftel der Kanten fehlt",
    "grid": "Raster mit Diagonalen",
    "lattice": "Gitter (ohne Diagonalen)",
    "onion": "Zwiebel (Dreiecke ineinander)",
    "stacked": "Gestapelte Triangulierung",
    "toronto": "Straßenausschnitt (Toronto)",
}
DEFAULT_NET = "delaunay"
NET_HELP = (
    "Karte: zufällige Standorte mit allen Kanten ihrer Delaunay-Triangulierung (planar, voll trianguliert). "
    "Karte mit fehlenden Kanten: dieselbe Karte mit 80 % der Kanten, der größte zweifach zusammenhängende Teil; sie ist nicht mehr 3-fach zusammenhängend. "
    "Raster mit Diagonalen und Gitter ohne: k × k Knoten (n wird auf ein Quadrat gerundet). "
    "Zwiebel: n/3 Dreiecke ineinander, die Ecken jeder Stufe sind mit der nächsten verbunden (3-fach zusammenhängend, planar). "
    "Gestapelte Triangulierung: von einem Dreieck aus wird immer wieder eine Dreiecksfläche durch einen neuen Knoten geteilt. "
    "Straßenausschnitt: die n Knoten um die Mitte der Toronto-Karte (Breitensuche); zwischen 160 und 170 Knoten nicht mehr planar."
)

METHODS = {
    "fr": "Federmodell (Fruchterman–Reingold)",
    "stress": "Stress-Minimierung (Majorisierung)",
    "sgd": "Stress-Minimierung (SGD)",
    "tutte": "Tutte-Einbettung",
    "grid": "Gitterzeichnung (de Fraysseix–Pach–Pollack)",
}
DEFAULT_METHOD = "tutte"
METHOD_HELP = (
    "Federmodell: alle Knoten stoßen sich ab, Kanten ziehen ihre Enden zusammen. Stress-Minimierung: Abstände in der Zeichnung sollen den Weglängen im Graphen entsprechen "
    "(Majorisierung oder stochastisches Gradientenverfahren). Tutte-Einbettung: die Außenfläche auf einem Kreis, jeder andere Knoten im Schwerpunkt seiner Nachbarn; "
    "kreuzungsfrei garantiert, wenn der Graph 3-fach zusammenhängend ist. Gitterzeichnung: ganzzahlige Lage im Gitter (2n − 4) × (n − 2), für jeden zweifach zusammenhängenden planaren Graphen kreuzungsfrei."
)
OUTER = {"longest": "Längste Fläche", "shortest": "Kürzeste Fläche"}
DEFAULT_OUTER = "longest"
OUTER_HELP = "Nur für die Tutte-Einbettung: welche Fläche der Einbettung als Außenfläche auf den Kreis gelegt wird."

# Lesbarkeit: Auflösung (kleinster Knotenabstand je Diagonale) im Verhältnis zu der eines gleichmäßigen Quadratrasters mit n Knoten, 1/(√2 (√n − 1)).
# Die Schwellen sind Festlegungen, keine Messwerte: gut lesbar ab 15 %, eng ab 2 %, darunter fallen Knoten in der Anzeige zusammen.
READABLE_GOOD, READABLE_TIGHT = 0.15, 0.02

# Messreihen
MAP_NS = (30, 60, 100)                                   # Karten und gestapelte Triangulierungen der Größenreihe
ONION_KS = (3, 5, 8, 12)                                 # Stufen der Zwiebel-Reihe (n = 3k)
DRIFT_STEPS = (0, 1, 3, 10, 30, 100, 300)                # Schritte der Majorisierung ab dem Tutte-Start
DRIFT_NS = (60, 100)
STARTS_NS = (20, 30, 60)                                 # Größen der Reihe „Federmodell, beste von 20 Starts“
STARTS_COUNT = 20
TORONTO_NS = (100, 200, 300)
LATTICE_K, LATTICE_STARTS = 15, 10                       # Gitter k × k (n = 225): Federmodell mit 10 Startlagen gegen die Stress-Minimierung

COLOR_EDGE = "rgba(120,120,120,0.55)"
COLOR_EDGE_BAD = "#c0504d"
COLOR_NODE = "#2F6B65"
COLOR_OUTER = "#e0a030"
COLOR_LINE = "#4c78a8"
COLOR_PALETTE = ("#4c78a8", "#e0a030", "#2F6B65", "#c0504d", "#8c6bb1")

PRESETS = {
    "Lehrbeispiel: Tutte auf einer Karte": {"net": "delaunay", "n": 60, "seed": 101, "method": "tutte", "iterations": 300, "outer": "longest"},
    "Federmodell: nie kreuzungsfrei": {"net": "delaunay", "n": 60, "seed": 101, "method": "fr", "iterations": 300, "outer": "longest"},
    "Zwiebel: Tutte ballt die Knoten": {"net": "onion", "n": 36, "seed": 101, "method": "tutte", "iterations": 300, "outer": "longest"},
    "Gitterzeichnung: garantiert, aber eng": {"net": "delaunay", "n": 40, "seed": 101, "method": "grid", "iterations": 300, "outer": "longest"},
    "Karte mit fehlenden Kanten: Tutte entartet": {"net": "thin", "n": 60, "seed": 101, "method": "tutte", "iterations": 300, "outer": "longest"},
}
PRESET_HELP = {
    "Lehrbeispiel: Tutte auf einer Karte": "60 Standorte mit Delaunay-Kanten, die Hülle als Außenfläche auf dem Kreis: keine Kreuzung, und die Knoten liegen so weit auseinander wie beim Federmodell.",
    "Federmodell: nie kreuzungsfrei": "Dieselbe Karte mit dem Federmodell: die Zeichnung sieht ordentlich aus, hat aber Kreuzungen; das Verfahren kennt die Planarität nicht.",
    "Zwiebel: Tutte ballt die Knoten": "12 Dreiecke ineinander: Tutte ist kreuzungsfrei, aber jede Stufe rückt die Knoten um etwa den Faktor fünf zusammen; die inneren Stufen sind nicht mehr zu erkennen.",
    "Gitterzeichnung: garantiert, aber eng": "Eine Karte mit 40 Standorten als Gitterzeichnung: garantiert kreuzungsfrei im Gitter 76 × 38, aber lange Kanten und winzige Winkel.",
    "Karte mit fehlenden Kanten: Tutte entartet": "Fehlen Kanten, ist die Karte nicht mehr 3-fach zusammenhängend: die Garantie gilt nicht, und Knoten können aufeinanderfallen.",
}
