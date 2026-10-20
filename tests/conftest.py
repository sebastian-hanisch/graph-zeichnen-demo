"""Gemeinsame Fixtures: Handgraphen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gzn_scenario as S  # noqa: E402

# K4 mit Knoten 3 in der Mitte des Dreiecks 0, 1, 2; alle Flächen gleich orientiert (jede gerichtete Kante steht genau einmal): drei innere Dreiecke gegen den Uhrzeigersinn, die Außenfläche (0, 2, 1)
K4_FACES = [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)]


@pytest.fixture
def k4():
    return S.hand_graph("K4")


@pytest.fixture
def k4_faces():
    return list(K4_FACES)
