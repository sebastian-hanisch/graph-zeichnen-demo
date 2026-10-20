"""Planarität und Einbettung: Fragmentverfahren (Demoucron–Malgrange–Pertuiset), Flächenliste je Block, 3-facher Zusammenhang.

Kopiert und gekürzt aus `planaritaet-demo` (plt_algorithm.py, Commit bf55b55; dort mit Ereignissen zum Abspielen, Zeugen und Schrittzählern, die hier nicht gebraucht werden).
Gebraucht wird die Einbettung, um (1) für die Tutte-Zeichnung eine Außenfläche zu wählen und (2) für die Gitterzeichnung eine Triangulierung samt Drehsystem zu bilden.
Graphen sind einfach und ungerichtet; Knoten 0 bis n − 1, Kanten als sortierte Paare. Kein Zufall: Gleichstand entscheidet die kleinere Knotennummer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


def adjacency(n: int, edges) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def components(n: int, adj) -> int:
    """Zahl der Zusammenhangskomponenten (isolierte Knoten zählen mit)."""
    seen = [False] * n
    count = 0
    for s in range(n):
        if seen[s]:
            continue
        count += 1
        seen[s] = True
        queue = deque([s])
        while queue:
            x = queue.popleft()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    queue.append(y)
    return count


def biconnected_blocks(n: int, adj) -> list[list[tuple[int, int]]]:
    """Blöcke (zweifach zusammenhängende Teile) als Kantenlisten; eine Brücke ist ein Block mit einer Kante. Ein Graph ist genau dann planar, wenn jeder Block es ist."""
    disc = [-1] * n
    low = [0] * n
    clock = 0
    edge_stack: list[tuple[int, int]] = []
    blocks: list[list[tuple[int, int]]] = []
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = clock
        clock += 1
        stack = [(root, -1, iter(sorted(adj[root])))]
        while stack:
            v, parent, it = stack[-1]
            advanced = False
            for w in it:
                if w == parent:
                    continue
                if disc[w] == -1:
                    edge_stack.append((v, w))
                    disc[w] = low[w] = clock
                    clock += 1
                    stack.append((w, v, iter(sorted(adj[w]))))
                    advanced = True
                    break
                if disc[w] < disc[v]:
                    edge_stack.append((v, w))
                    low[v] = min(low[v], disc[w])
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[v])
                if low[v] >= disc[p]:
                    block = []
                    while True:
                        e = edge_stack.pop()
                        block.append((min(e), max(e)))
                        if e == (p, v):
                            break
                    blocks.append(sorted(block))
    return blocks


def find_cycle(adj, nodes) -> list[int] | None:
    """Ein Kreis in einem zweifach zusammenhängenden Teil (Tiefensuche, kleinster Nachbar zuerst); None, wenn es keinen gibt."""
    start = min(nodes)
    parent = {start: None}
    depth = {start: 0}
    stack = [(start, iter(sorted(w for w in adj[start] if w in nodes)))]
    while stack:
        v, it = stack[-1]
        advanced = False
        for w in it:
            if w == parent[v]:
                continue
            if w in parent:
                if depth[w] < depth[v]:
                    cycle = [v]
                    x = v
                    while x != w:
                        x = parent[x]
                        cycle.append(x)
                    return cycle
                continue
            parent[w] = v
            depth[w] = depth[v] + 1
            stack.append((w, iter(sorted(y for y in adj[w] if y in nodes))))
            advanced = True
            break
        if not advanced:
            stack.pop()
    return None


@dataclass(frozen=True)
class Fragment:
    """Eine Brücke relativ zum bisher Eingebetteten: eine Kante zwischen zwei eingebetteten Knoten oder eine Komponente der übrigen Knoten samt Anschlusskanten."""

    attach: tuple[int, ...]       # eingebettete Knoten, mit denen das Fragment verbunden ist (sortiert)
    interior: tuple[int, ...]     # noch nicht eingebettete Knoten des Fragments (sortiert, leer bei einer einzelnen Kante)


def fragments(adj, nodes, embedded: set[int], embedded_edges: set[tuple[int, int]]) -> list[Fragment]:
    """Alle Fragmente. Einzelkanten zuerst, dann Komponenten nach kleinster Knotennummer."""
    found: list[Fragment] = []
    for u in sorted(embedded):
        for v in sorted(adj[u]):
            if v > u and v in embedded and (u, v) not in embedded_edges:
                found.append(Fragment((u, v), ()))
    seen: set[int] = set()
    for s in sorted(nodes):
        if s in embedded or s in seen:
            continue
        comp = {s}
        attach: set[int] = set()
        queue = deque([s])
        while queue:
            x = queue.popleft()
            for y in sorted(adj[x]):
                if y not in nodes:
                    continue
                if y in embedded:
                    attach.add(y)
                elif y not in comp:
                    comp.add(y)
                    queue.append(y)
        seen |= comp
        found.append(Fragment(tuple(sorted(attach)), tuple(sorted(comp))))
    return found


def admissible_faces(fragment: Fragment, face_sets: list[set[int]]) -> list[int]:
    """Nummern der Flächen, die alle Anschlussknoten des Fragments enthalten (nur dort kann es eingezeichnet werden)."""
    need = set(fragment.attach)
    return [i for i, members in enumerate(face_sets) if need <= members]


def choose_fragment(options: list[tuple[Fragment, list[int]]]) -> int:
    """Index des Fragments mit den wenigsten zulässigen Flächen (das erste bei Gleichstand): hat eines nur eine, muss es dorthin."""
    best = 0
    for i, (_, adm) in enumerate(options):
        if len(adm) < len(options[best][1]):
            best = i
    return best


def path_in_fragment(adj, nodes, fragment: Fragment) -> list[int]:
    """Ein kürzester Weg zwischen zwei verschiedenen Anschlussknoten durch das Fragment (bei einer Einzelkante: die Kante)."""
    a = fragment.attach[0]
    if not fragment.interior:
        return [a, fragment.attach[1]]
    inside = set(fragment.interior)
    attach = set(fragment.attach)
    prev: dict[int, int | None] = {a: None}
    queue = deque([a])
    end = None
    while queue and end is None:
        x = queue.popleft()
        for y in sorted(adj[x]):
            if y not in nodes or y in prev:
                continue
            if y in inside:
                prev[y] = x
                queue.append(y)
            elif y in attach and y != a and x in inside:
                prev[y] = x
                end = y
                break
    path = [end]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def split_face(face: list[int], path: list[int]) -> tuple[list[int], list[int]]:
    """Teilt die Fläche (zyklische Knotenfolge) durch den Weg zwischen zwei ihrer Knoten in zwei Flächen."""
    a, b = path[0], path[-1]
    interior = path[1:-1]
    k = len(face)
    i, j = face.index(a), face.index(b)
    first = [face[(i + t) % k] for t in range(((j - i) % k) + 1)]    # von a bis b
    second = [face[(j + t) % k] for t in range(((i - j) % k) + 1)]   # von b bis a
    return first + interior[::-1], second + interior


@dataclass(frozen=True)
class BlockResult:
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    planar: bool
    faces: tuple[tuple[int, ...], ...]     # nur bei planar und mindestens drei Knoten; jede gerichtete Kante steht in genau einer Fläche


def embed_block(edges) -> BlockResult:
    """Fragmenttest für einen zweifach zusammenhängenden Block: Einbettung (Flächenliste) oder Scheitern, wenn ein Fragment keine zulässige Fläche hat."""
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in edges))
    nodes = sorted({x for e in edges for x in e})
    if len(nodes) <= 2:
        return BlockResult(tuple(nodes), edges, True, ())
    adj: dict[int, set[int]] = {x: set() for x in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    node_set = set(nodes)
    cycle = find_cycle(adj, node_set)
    faces = [list(cycle), list(cycle[::-1])]
    face_sets = [set(cycle), set(cycle)]
    embedded = set(cycle)
    embedded_edges = {(min(cycle[i], cycle[(i + 1) % len(cycle)]), max(cycle[i], cycle[(i + 1) % len(cycle)])) for i in range(len(cycle))}
    while len(embedded_edges) < len(edges):
        options = []
        for frag in fragments(adj, node_set, embedded, embedded_edges):
            adm = admissible_faces(frag, face_sets)
            if not adm:
                return BlockResult(tuple(nodes), edges, False, ())
            options.append((frag, adm))
        frag, adm = options[choose_fragment(options)]
        path = path_in_fragment(adj, node_set, frag)
        fi = adm[0]
        f1, f2 = split_face(faces[fi], path)
        faces[fi], face_sets[fi] = f1, set(f1)
        faces.append(f2)
        face_sets.append(set(f2))
        embedded |= set(path)
        embedded_edges |= {(min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1)}
    return BlockResult(tuple(nodes), edges, True, tuple(tuple(f) for f in faces))


@dataclass(frozen=True)
class Embedding:
    """Ergebnis für den ganzen Graphen: ein Block-Ergebnis je Block."""

    n: int
    components: int
    blocks: tuple[BlockResult, ...]

    @property
    def planar(self) -> bool:
        return all(b.planar for b in self.blocks)

    @property
    def single_block(self) -> BlockResult | None:
        """Der Block, wenn der ganze Graph (n ≥ 3) zweifach zusammenhängend ist, sonst None."""
        if len(self.blocks) == 1 and len(self.blocks[0].nodes) == self.n and self.n >= 3:
            return self.blocks[0]
        return None


def embed(n: int, edges) -> Embedding:
    """Fragmenttest für den ganzen Graphen: alle Blöcke, in der Reihenfolge der Tiefensuche."""
    adj = adjacency(n, edges)
    return Embedding(n, components(n, adj), tuple(embed_block(b) for b in biconnected_blocks(n, adj)))


def largest_block(n: int, edges) -> list[tuple[int, int]]:
    """Kanten des größten Blocks (nach Knotenzahl, bei Gleichstand der erste in Tiefensuche-Reihenfolge)."""
    blocks = biconnected_blocks(n, adjacency(n, edges))
    if not blocks:
        return []
    return max(blocks, key=lambda b: len({x for e in b for x in e}))


def is_biconnected(n: int, adj) -> bool:
    """Zusammenhängend, mindestens drei Knoten und ohne Trennknoten (ein einziger Block deckt alle Knoten)."""
    if n < 3:
        return False
    blocks = biconnected_blocks(n, adj)
    return len(blocks) == 1 and len({x for e in blocks[0] for x in e}) == n


def is_triconnected(n: int, edges) -> bool:
    """3-fach zusammenhängend: mindestens vier Knoten, zweifach zusammenhängend, und nach Entfernen jedes einzelnen Knotens bleibt der Rest zweifach zusammenhängend."""
    if n < 4:
        return False
    adj = adjacency(n, edges)
    if not is_biconnected(n, adj):
        return False
    for v in range(n):
        rest = [{w for w in adj[u] if w != v} if u != v else set() for u in range(n)]
        blocks = biconnected_blocks(n, rest)
        if len(blocks) != 1 or len({x for e in blocks[0] for x in e}) != n - 1:
            return False
    return True
