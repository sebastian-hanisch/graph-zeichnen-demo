# Graphen zeichnen – Garantie oder Lesbarkeit? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-graph-zeichnen-demo.streamlit.app/)** -->

Viertes Stück der **Planarität-und-Zeichnen-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es setzt auf die Einbettung des ersten Stücks [planaritaet-demo](https://github.com/sebastian-hanisch/planaritaet-demo) (Fragmentverfahren, Flächenliste) und die Karten und Toronto-Daten des dritten [planar-separator-demo](https://github.com/sebastian-hanisch/planar-separator-demo) auf. Verwandt: [isomap-demo](https://github.com/sebastian-hanisch/isomap-demo) nutzt klassisches MDS, das hier die Startlage der Stress-Minimierung liefert; dort geht es um Punktwolken, hier um Graphen.

Jeder planare Graph lässt sich ohne Kreuzungen zeichnen. Die Demo zeichnet Karten, Gitter, „Zwiebeln“ und Straßenausschnitte mit vier Verfahren, **Federmodell** (Fruchterman und Reingold), **Stress-Minimierung** (Majorisierung und SGD), **Tutte-Einbettung** und **Gitterzeichnung** nach de Fraysseix, Pach und Pollack, und misst jede Zeichnung: Kreuzungen (exakt gezählt), Auflösung, Winkel, Kantenverhältnis, Stress. Eigene Löser in Python; `networkx` nur als Gegenprobe in den Tests.

## Warum dieses Problem

Wer einen Netzplan, eine Karte oder ein Abhängigkeitsdiagramm zeichnet, will keine Kreuzungen und trotzdem etwas erkennen. Zwei Verfahren garantieren die Kreuzungsfreiheit, zwei sind die gebräuchlichen Heuristiken. Kernfrage: **was sagt die Garantie über die Lesbarkeit, und was kosten die Heuristiken?**

## Modell

Ein einfacher ungerichteter Graph; eine Zeichnung ordnet jedem Knoten einen Punkt zu, Kanten sind Strecken. Eine Zeichnung gilt als **gültig**, wenn sich keine zwei Kanten kreuzen **und** nichts aufeinanderliegt (Knoten auf Knoten, Knoten auf einer fremden Kante): ein reiner Kreuzungszähler meldet bei zusammengefallenen Knoten null. Gezählt wird mit exakten Vorzeichentests; die Tutte-Einbettung wird in rationalen Zahlen (`Fraction`) gelöst, damit auch winzige Abstände richtig entschieden werden. **Auflösung** = kleinster Knotenabstand je Zeichnungsdiagonale, angegeben im Verhältnis zu einem gleichmäßigen Quadratraster mit gleich vielen Knoten (100 %); die Schwellen „gut lesbar“ ab 15 % und „eng“ ab 2 % sind **Festlegungen**, keine Messwerte. Formeln und Literatur im Expander „📐 Mathematische Formulierung“.

Graphtypen: **Karte** (Delaunay-Triangulierung zufälliger Standorte), **Karte mit fehlenden Kanten** (80 % der Kanten, größter zweifach zusammenhängender Teil), **Raster mit Diagonalen**, **Gitter**, **Zwiebel** (n/3 Dreiecke ineinander, Ecken stufenweise verbunden), **gestapelte Triangulierung** (von einem Dreieck aus wird immer wieder eine Fläche durch einen neuen Knoten geteilt) und **Straßenausschnitt** (die n Knoten um die Mitte der Toronto-Karte, OpenStreetMap, ODbL, aus `bfs-demo`).

## Methodik

- **Eine Zeichnung:** Graphtyp, n, Zufalls-Seed, Verfahren, Schritte (Federmodell, Stress), Außenfläche (Tutte); Kennzahlen, Urteil und Zeichnung (rote Kanten kreuzen sich).
- **Vier Verfahren, ein Graph:** dieselbe Instanz mit allen vier Verfahren, Tabelle und vier Zeichnungen.
- **Messreihen auf Klick** (je 10 feste Karten, Mediane): Karten, gestapelte Triangulierungen und Zwiebel (Garantie gegen Lesbarkeit); Federmodell mit vielen Startlagen, Stress ab Tutte-Start und das Gitter (warum die Heuristiken Kreuzungen behalten); Straßenausschnitte.
- **Gitterzeichnung Schritt für Schritt** bis n = 60: kanonische Ordnung, Einfügen, Verschieben.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen unten: Karten mit n = 30 / 60 / 100 Knoten, je 10 feste Karten (Zufalls-Seeds 100 bis 109), Median, wenn nichts anderes steht; Federmodell 300 Schritte, Stress-Majorisierung 300 Schritte ab klassischem MDS.

| Frage | Befund | Test |
|---|---|---|
| Sind Federmodell und Stress-Minimierung kreuzungsfrei? | **Nein, in keiner der 30 Karten.** Kreuzungen im Median: Federmodell 29 / 78,5 / 150,5, Stress-Minimierung 28,5 / 90 / 185,5. Die Stress-Minimierung kreuzt **nicht weniger** als das Federmodell. | `test_on_maps_the_heuristics_are_never_crossing_free_and_the_guaranteed_methods_nearly_always_are` |
| Und die zwei Verfahren mit Garantie? | Die Gitterzeichnung ist in allen 30 Karten gültig, Tutte nie mit einer Kreuzung, aber bei n = 30 in einer von 10 Karten ungültig (n = 30: 9, n = 60 und 100: 10 von 10): diese Karte hat eine Hüllenecke vom Grad 2, ist also nicht 3-fach zusammenhängend, und Knoten fallen aufeinander. | dieselbe |
| Wie weit liegen die Knoten auseinander? (n = 100, Quadratraster = 100 %) | Federmodell **21 %**, Stress **25 %**, Tutte **15 %**, Gitterzeichnung **6 %**. Auf Karten ist Tutte fast so locker wie das Federmodell, die Gitterzeichnung deutlich enger. | `test_on_maps_tutte_is_as_spread_out_as_the_spring_model_but_the_grid_is_not` |
| Winkel und Kantenlängen? (n = 100) | Kleinster Winkel an einem Knoten: Tutte **12°**, Federmodell 0,6°, Stress 0,8°, Gitterzeichnung 0,01°. Kantenverhältnis längste/kürzeste: Tutte 17, Federmodell 10, Stress 4,6, Gitterzeichnung 196. | `test_on_maps_tutte_has_the_best_angles_and_the_grid_the_worst` |
| Wie treu ist die Zeichnung der echten Lage? (Procrustes, 0 = gleiche Gestalt) | Tutte 0,07, Federmodell 0,08, Stress 0,10, Gitterzeichnung 0,77. | `test_on_maps_tutte_stays_closest_to_the_real_positions_and_the_grid_loses_them` |
| Wie groß wird das Gitter? | Genau die Schranke: **(2n − 4) × (n − 2) = 196 × 98** bei n = 100, in allen 10 Karten. | `test_the_grid_drawing_fills_the_bound_exactly_on_maps` |
| Wo versagt Tutte? Gestapelte Triangulierungen | Tutte bleibt in 10 von 10 kreuzungsfrei, aber die Auflösung fällt auf **1,1 % / 0,43 % / 0,13 %** (n = 30 / 60 / 100) und das Kantenverhältnis steigt auf 434 / 1.710 / 7.920; die Gitterzeichnung hält 10,1 % / 7,4 % / 5,8 %. Das Federmodell hat dort 91 / 224 / 448 Kreuzungen. | `test_on_stacked_triangulations_tutte_collapses_and_the_grid_does_not` |
| Wo versagt Tutte? Zwiebel | k = 3 / 5 / 8 / 12 Dreiecke ineinander (n = 9 / 15 / 24 / 36): kleinster Abstand je Diagonale **5,3·10⁻² / 2,3·10⁻³ / 2,1·10⁻⁵ / 4,0·10⁻⁸**, etwa **Faktor 5 je Stufe**, dabei exakt kreuzungsfrei; Gitterzeichnung 0,064 / 0,034 / 0,020 / 0,013. | `test_on_the_onion_tutte_loses_about_a_factor_of_five_per_level_exactly_counted`, `test_exact_arithmetic_is_what_decides_in_the_onion` |
| Und ohne 3-fachen Zusammenhang? | Karten mit n = 60, denen 20 % der Kanten fehlen (58 bis 60 Knoten im größten zweifach zusammenhängenden Teil): Tutte ist nur in **2 von 10** Karten gültig, in den übrigen fallen Knoten aufeinander, **obwohl der Kreuzungszähler in allen 10 null meldet**; die Gitterzeichnung ist in 10 von 10 gültig. | `test_maps_with_missing_edges_make_tutte_collapse_vertices_while_the_crossing_counter_sees_nothing` |
| Erreicht das Federmodell mit vielen Startlagen null Kreuzungen? | **Nein:** 20 Startlagen je Karte, 10 Karten je Größe: in 0 von 30 Karten. Kleinste erreichte Zahl je Karte: n = 20: 5 bis 19, n = 30: 11 bis 32, n = 60: 31 bis 103. | `test_spring_model_with_many_starts_never_reaches_zero_crossings` |
| Was macht die Stress-Minimierung mit einer kreuzungsfreien Zeichnung? | Ab der Tutte-Zeichnung (0 Kreuzungen) bei n = 60: nach 1 / 10 / 300 Schritten 8,5 / 78 / 90 Kreuzungen; bei n = 100: 17,5 / 179,5 / 188. | `test_stress_from_the_tutte_start_loses_crossing_freedom_within_ten_steps` |
| Ist das Federmodell auf dem Gitter besser? | Gitter 15 × 15 (n = 225), 10 Startlagen: in 5 von 10 Läufen bleiben Kreuzungen (108 bis 229), die Stress-Minimierung endet mit 0. Auf Gittern passen Weglängen und Ebene zusammen. | `test_on_the_lattice_the_spring_model_folds_in_about_half_of_the_runs_and_stress_never` |
| Erreicht SGD denselben Stress? | Ja (Abweichung unter 2 %): n = 30 / 60 / 100 etwa 0,055 / 0,029 / 0,038 bei beiden. | `test_sgd_reaches_the_stress_of_the_majorization` |
| Straßenausschnitte? | n = 100 / 200 / 300 um die Mitte der Toronto-Karte: planar nur der kleinste; die **geografische Lage hat 7 / 13 / 13 Kreuzungen**, das Federmodell 12 / 36 / 90, die Stress-Minimierung 3 / 35 / 48. Tutte und Gitterzeichnung gibt es für keinen der drei (nicht zweifach zusammenhängend). Die Ausschnitte sind bis n = 160 planar und ab n = 170 nicht mehr (gemessen in Zehnerschritten). | `test_street_excerpts`, `test_street_excerpts_stop_being_planar_between_160_and_170_nodes` |

## Ehrliche Grenzen

- **Lesbarkeit ist hier die Auflösung.** Sie misst nur den kleinsten Knotenabstand (zusätzlich Winkel, Kantenverhältnis, Stress und Treue zur Karte). Ob eine Zeichnung gefällt, hängt von Symmetrie, Kantenlängen und Platz für Beschriftungen ab; die Schwellen 15 % und 2 % sind festgelegt.
- **Es gibt kein bestes Verfahren.** Jedes der zwei Verfahren mit Garantie hat einen Graphtyp, auf dem es kaum brauchbar ist (Tutte: Zwiebel und gestapelte Triangulierungen, Gitterzeichnung: Karten), und die zwei lesbaren haben keine Garantie.
- **Die Zwiebel ist der ungünstigste Fall für Tutte, nicht der typische.** Auf Karten mit großer Außenfläche ist Tutte so lesbar wie das Federmodell. Die Zehnerpotenzen sind hier gemessen; eine Quelle für die exponentiell kleine Auflösung Tutte-artiger Zeichnungen fand die Suche nur in einer Arbeit über Verformungen von Zeichnungen (arXiv 2108.09483, nicht gelesen), sie wird hier nicht zitiert.
- **Die Gitterzeichnung** gibt es nur für zweifach zusammenhängende planare Graphen (sonst bräuchte man eine globale Einbettung über die Blöcke hinweg). Sie wird durch Sehnen zu einer Triangulierung ergänzt; die Sehnen gehören nicht zum Graphen. Schnyders Verfahren mit kleinerem Gitter ist nicht nachgebaut.
- **Federmodell und Stress hängen von Start und Schritten ab.** Die Zahlen sind Mediane über feste Karten, in den Tests mit Bändern; SGD läuft in reinem Python und ist auf n ≤ 100 begrenzt.
- **Die Tutte-Rechnung ist exakt, die Anzeige nicht:** Bei der Zwiebel mit k ≥ 12 liegen Knoten unter 10⁻⁸ der Zeichnung auseinander, die Anzeige kann sie nicht mehr trennen.
- **Kleine Stichprobe:** 10 feste Karten je Größe; die Aussagen „nie“ gelten für diese Karten und Startlagen, nicht für alle.

## Befunde und Korrekturen gegenüber dem Plan

- **Seeds und Karten:** die Vormessung im Einzelplan lief auf den Karten 0 bis 9; die Demo misst auf 100 bis 109. Die Zahlen unterscheiden sich im Einzelnen (Federmodell bei n = 100: 150 statt 122 Kreuzungen, Tutte-Auflösung 15 % statt 22 %), die Befunde bleiben: Federmodell und Stress nie kreuzungsfrei, Tutte auf Karten fast so locker wie das Federmodell, Gitterzeichnung eng.
- **Karte mit fehlenden Kanten:** die Vormessung nahm die größte Komponente (Knoten vom Grad 1 fallen immer auf ihren Nachbarn); die Demo nimmt den größten zweifach zusammenhängenden Teil, die schärfere Prüfung. Ergebnis: ungültig in 8 statt 3 von 10 Karten bei 80 % der Kanten.
- **Gitter:** die Vormessung nannte 20 × 20 (n = 400, das Federmodell braucht dort etwa vier Sekunden je Lauf); die Demo nimmt 15 × 15 (n = 225), dort bleiben in 5 von 10 Läufen Kreuzungen.
- **Straßenausschnitt:** der Federmodell-Lauf der Demo hat einen anderen Seed als die Vormessung (12 / 36 / 90 statt 7 / 48 / 47 Kreuzungen); die geografische Lage (7 / 13 / 13) ist gleich.
- **Zwiebel:** die Vormessung zählte in Gleitkomma und nannte „0 Kreuzungen“ auch bei k = 16; die Demo rechnet exakt und begrenzt die Zwiebel-Reihe auf k ≤ 12.
- **Gitterzeichnung:** die Vormessung (mit `networkx`) ergab genau die Schranke (2n − 4) × (n − 2); die eigene Umsetzung trifft Breite und Höhe in allen 10 Karten mit n = 100 ebenfalls.

## Verifikation

- **Kern gegen Handrechnung:** Weglängen, Stress (Pfad als Winkel gezeichnet), klassisches MDS, ein Majorisierungsschritt, Kräfte des Federmodells (Gleichgewicht bei d = k), exakte Kreispunkte, Tutte an K4, Rad und Prisma, Triangulieren, Drehsystem, kanonische Ordnung und Schiebeverfahren an K4 (`tests/test_algorithm.py`); Gütemaße an Quadraten, Sternen und Winkeln (`tests/test_geometry.py`).
- **Orakel:** `tests/test_oracle_graph_zeichnen.py`: Tutte-Schwerpunkteigenschaft exakt in `Fraction`, Lösung gegen ein float-Gleichungssystem, keine Kreuzung gegen eine einfache Doppelschleife (Satz von Tutte als Eigenschaftstest); Gitterzeichnung ganzzahlig, Breite 2n − 4, Mindestabstand 1, keine Kreuzung, und `networkx.combinatorial_embedding_to_pos` hält dieselben Schranken; kanonische Ordnung erfüllt die Definition (zweifach zusammenhängend je Anfangsstück, `networkx`); Stress gegen `networkx.kamada_kawai_layout`, Gradient am Ergebnis per finiter Differenz, SGD gleich Majorisierung; Federmodell-Kräfte gleich dem Gradienten ihrer Energie; Weglängen gegen `networkx`; `tests/test_planar.py`: Planarität, Blöcke und 3-facher Zusammenhang gegen `networkx` auf Zufallsgraphen.
- **Mutationsprüfung** (`tools/mutate_demo.py`, Stand vor dem ersten Commit): 406 von 418 Stellen der Module gesichert (gefangen oder als äquivalent festgehalten, `tests/mutation_state.json`), keine offene Lücke; 12 Mutanten ergaben Endlosschleifen (Zeitüberschreitung, zählt nicht). Die äquivalenten sind Sortierreihenfolgen, Schleifengrenzen ohne Wirkung, unerreichbare Schutzabfragen und Zahlen im Zeitplan von SGD und bei den rationalen Kreispunkten, die das Ergebnis nur im Rahmen der Testbänder ändern.
- **Randfälle:** nicht planare, nicht zweifach zusammenhängende und getrennte Graphen, Dreieck, kleinste und größte Größe, inklusive Schwellen (`tests/test_edge_cases.py`).
- **Verdikte** als reine Funktionen, jede Zahl im Text (`tests/test_verdicts.py`); **Claims:** jede Zahl dieser README (`tests/test_claims.py`).
- **Szenario:** Generatoren, Hash-Seed-Unabhängigkeit, Toronto-Daten (`tests/test_scenario.py`); Darstellung (`tests/test_visualization.py`); Presets halten, was ihre Hilfetexte sagen (`tests/test_presets.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an beiden Enden), Layout, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `gzn_algorithm.py` | Kern: Weglängen, Federmodell, Stress (Majorisierung, SGD), Tutte-Einbettung (exakt), Triangulieren, kanonische Ordnung, Gitterzeichnung |
| `gzn_planar.py` | Einbettung (Fragmentverfahren), Blöcke, 3-facher Zusammenhang (aus `planaritaet-demo` gekürzt) |
| `gzn_geometry.py` | Gütemaße: exakte Kreuzungen und Berührungen, Auflösung, Fläche, Kantenverhältnis, Winkel, Treue |
| `gzn_constants.py` | Regler-Grenzen, Rechengrenzen, Schwellen, Farben, Presets |
| `gzn_evaluation.py` | Zeichnen und Messen, Urteile (reine Funktionen), Messreihen |
| `gzn_scenario.py` | Graphen: Karte, Karte mit fehlenden Kanten, Raster, Gitter, Zwiebel, gestapelte Triangulierung, Straßenausschnitt, Handgraphen |
| `gzn_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `gzn_layout.py` | Fester Kopf (Stylesheet) |
| `gzn_presets.py` | Permalink (Begrenzen), Presets (Schnellstart oben, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `data/toronto_campus.json`, `data/LICENSE-ODbL.md` | Straßenkarte (OpenStreetMap, ODbL) |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnung, Orakel, Randfälle, Verdikte, Claims, Szenario, Darstellung, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Schnyders Gitterzeichnung** mit kleinerem Gitter ((n − 2) × (n − 2) laut Übersichtsarbeit): Einordnung, kein Nachbau.
- **Kreuzungsminimierung und geschichtetes Zeichnen** (Sugiyama): eigenes Stück der Linie.
- **Orthogonales Zeichnen, Mehrebenen-Verfahren für große Graphen, 3D:** andere Aufgabe.
- **Planarisieren nichtplanarer Graphen** (Kreuzungen als Knoten einfügen, Kanten entfernen): eigenes Stück der Linie; hier nur die Feststellung, dass der Straßenausschnitt zwischen 160 und 170 Knoten nicht mehr planar ist.
- **Laufzeitvergleiche:** verglichen wird die Gestalt der Zeichnung, nicht die Rechenzeit.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
