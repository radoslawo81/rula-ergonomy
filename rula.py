"""
Implementacja metody RULA (Rapid Upper Limb Assessment, McAtamney & Corlett, 1993).

Zawiera oficjalne tabele przeliczeniowe (Tabela A, B, C) oraz funkcje
mapujące zmierzone kąty ciała na punktację RULA dla poszczególnych
segmentów. Moduł jest niezależny od źródła kątów (może być zasilany
danymi z estymacji pozy, czujników IMU itp.).

Ograniczenia względem oryginalnej metody RULA (patrz README.md):
 - Ocena nadgarstka (wrist) i skrętu nadgarstka (wrist twist) wymaga
   w oryginale obserwacji dłoni, której model pozy (17 punktów COCO)
   nie dostarcza -> przyjmowana jest wartość neutralna (1), chyba że
   wywołujący kod poda inną.
 - Ocena "muscle use" (praca statyczna/powtarzalna) jest przybliżana
   na podstawie czasu utrzymywania podobnej postawy (patrz app.py).
 - Ocena "force/load" wymaga ręcznego wskazania przez użytkownika
   (obciążenie nie jest wykrywalne z samego obrazu kamery).
"""

from dataclasses import dataclass, field


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# TABELA A: Ramię, przedramię, nadgarstek, skręt nadgarstka -> Score A
# TABLE_A[upper_arm][lower_arm] = [w1t1, w1t2, w2t1, w2t2, w3t1, w3t2, w4t1, w4t2]
# ---------------------------------------------------------------------------
TABLE_A = {
    1: {1: [1, 2, 2, 2, 2, 3, 3, 3],
        2: [2, 2, 2, 2, 3, 3, 3, 3],
        3: [2, 3, 3, 3, 3, 3, 4, 4]},
    2: {1: [2, 3, 3, 3, 3, 4, 4, 4],
        2: [3, 3, 3, 3, 3, 4, 4, 4],
        3: [3, 4, 4, 4, 4, 4, 5, 5]},
    3: {1: [3, 3, 4, 4, 4, 4, 5, 5],
        2: [3, 4, 4, 4, 4, 4, 5, 5],
        3: [4, 4, 4, 4, 4, 5, 5, 5]},
    4: {1: [4, 4, 4, 4, 4, 5, 5, 5],
        2: [4, 4, 4, 4, 4, 5, 5, 5],
        3: [4, 4, 4, 5, 5, 5, 6, 6]},
    5: {1: [5, 5, 5, 5, 5, 6, 6, 7],
        2: [5, 6, 6, 6, 6, 7, 7, 7],
        3: [6, 6, 6, 7, 7, 7, 7, 8]},
    6: {1: [7, 7, 7, 7, 7, 8, 8, 9],
        2: [8, 8, 8, 8, 8, 9, 9, 9],
        3: [9, 9, 9, 9, 9, 9, 9, 9]},
}

# ---------------------------------------------------------------------------
# TABELA B: Szyja, tułów, nogi -> Score B
# TABLE_B[neck][trunk] = [legs1, legs2]
# ---------------------------------------------------------------------------
TABLE_B = {
    1: {1: [1, 3], 2: [2, 3], 3: [3, 4], 4: [5, 5], 5: [6, 6], 6: [7, 7]},
    2: {1: [2, 3], 2: [2, 3], 3: [4, 5], 4: [5, 5], 5: [6, 7], 6: [7, 7]},
    3: {1: [3, 3], 2: [3, 4], 3: [4, 5], 4: [5, 6], 5: [6, 7], 6: [7, 7]},
    4: {1: [5, 5], 2: [5, 6], 3: [6, 7], 4: [7, 7], 5: [7, 7], 6: [8, 8]},
    5: {1: [7, 7], 2: [7, 7], 3: [7, 8], 4: [8, 8], 5: [8, 8], 6: [8, 8]},
    6: {1: [8, 8], 2: [8, 8], 3: [8, 8], 4: [8, 9], 5: [9, 9], 6: [9, 9]},
}

# ---------------------------------------------------------------------------
# TABELA C: Score A (wiersz, 1-8+) x Score B (kolumna, 1-7+) -> wynik koncowy
# ---------------------------------------------------------------------------
TABLE_C = [
    # B:   1  2  3  4  5  6  7+
    [1, 2, 3, 3, 4, 5, 5],   # A=1
    [2, 2, 3, 4, 4, 5, 5],   # A=2
    [3, 3, 3, 4, 4, 5, 6],   # A=3
    [3, 3, 3, 4, 5, 6, 6],   # A=4
    [4, 4, 4, 5, 6, 7, 7],   # A=5
    [4, 4, 5, 6, 6, 7, 7],   # A=6
    [5, 5, 6, 6, 7, 7, 7],   # A=7
    [5, 5, 6, 7, 7, 7, 7],   # A=8+
]

RISK_LEVELS = [
    (1, 2, "Akceptowalne", "#2e7d32"),
    (3, 4, "Do dalszej obserwacji, mogą być potrzebne zmiany", "#f9a825"),
    (5, 6, "Wskazana dalsza analiza i zmiany wkrótce", "#ef6c00"),
    (7, 7, "Wymagana natychmiastowa analiza i zmiany", "#c62828"),
]


def risk_level(final_score: int):
    for low, high, label, color in RISK_LEVELS:
        if low <= final_score <= high:
            return label, color
    return "Nieznany", "#616161"


# ---------------------------------------------------------------------------
# Segmentowe funkcje kąt -> ocena
# ---------------------------------------------------------------------------

def upper_arm_score(angle_deg: float, raised: bool = False,
                     abducted: bool = False, supported: bool = False) -> int:
    """Kąt mierzony od pionu tułowia; dodatni = zgięcie do przodu."""
    if angle_deg < -20:
        base = 2
    elif -20 <= angle_deg <= 20:
        base = 1
    elif 20 < angle_deg <= 45:
        base = 2
    elif 45 < angle_deg <= 90:
        base = 3
    else:
        base = 4
    base += 1 if raised else 0
    base += 1 if abducted else 0
    base -= 1 if supported else 0
    return _clamp(base, 1, 6)


def lower_arm_score(angle_deg: float, across_midline: bool = False) -> int:
    """Kąt zgięcia w łokciu (180 = ręka wyprostowana)."""
    if 60 <= angle_deg <= 100:
        base = 1
    else:
        base = 2
    base += 1 if across_midline else 0
    return _clamp(base, 1, 3)


def wrist_score(angle_deg: float = 0.0, bent_from_midline: bool = False) -> int:
    if abs(angle_deg) < 2:
        base = 1
    elif abs(angle_deg) <= 15:
        base = 2
    else:
        base = 3
    base += 1 if bent_from_midline else 0
    return _clamp(base, 1, 4)


def wrist_twist_score(mid_range: bool = True) -> int:
    return 1 if mid_range else 2


def neck_score(angle_deg: float, twisted: bool = False, side_bending: bool = False) -> int:
    if angle_deg < 0:
        base = 4  # wyprost do tyłu
    elif angle_deg <= 10:
        base = 1
    elif angle_deg <= 20:
        base = 2
    else:
        base = 3
    base += 1 if twisted else 0
    base += 1 if side_bending else 0
    return _clamp(base, 1, 6)


def trunk_score(angle_deg: float, twisted: bool = False,
                 side_bending: bool = False, well_supported_upright: bool = False) -> int:
    if well_supported_upright and angle_deg <= 0:
        base = 1
    elif angle_deg <= 20:
        base = 2
    elif angle_deg <= 60:
        base = 3
    else:
        base = 4
    base += 1 if twisted else 0
    base += 1 if side_bending else 0
    return _clamp(base, 1, 6)


def legs_score(supported_balanced: bool = True) -> int:
    return 1 if supported_balanced else 2


@dataclass
class RulaInputs:
    upper_arm: int
    lower_arm: int
    wrist: int
    wrist_twist: int
    neck: int
    trunk: int
    legs: int
    muscle_arm: int = 0     # 0 lub 1
    force_arm: int = 0      # 0-3
    muscle_body: int = 0    # 0 lub 1
    force_body: int = 0     # 0-3


@dataclass
class RulaResult:
    score_a_posture: int
    score_a: int
    score_b_posture: int
    score_b: int
    final_score: int
    risk_label: str
    risk_color: str
    inputs: RulaInputs = field(default_factory=lambda: None)


def compute_rula(inputs: RulaInputs) -> RulaResult:
    ua = _clamp(inputs.upper_arm, 1, 6)
    la = _clamp(inputs.lower_arm, 1, 3)
    wr = _clamp(inputs.wrist, 1, 4)
    wt = _clamp(inputs.wrist_twist, 1, 2)
    col_index = (wr - 1) * 2 + (wt - 1)
    score_a_posture = TABLE_A[ua][la][col_index]
    score_a = _clamp(score_a_posture + inputs.muscle_arm + inputs.force_arm, 1, 9)

    nk = _clamp(inputs.neck, 1, 6)
    tr = _clamp(inputs.trunk, 1, 6)
    lg = _clamp(inputs.legs, 1, 2)
    score_b_posture = TABLE_B[nk][tr][lg - 1]
    score_b = _clamp(score_b_posture + inputs.muscle_body + inputs.force_body, 1, 9)

    row = _clamp(score_a, 1, 8) - 1
    col = _clamp(score_b, 1, 7) - 1
    final = TABLE_C[row][col]
    label, color = risk_level(final)

    return RulaResult(
        score_a_posture=score_a_posture,
        score_a=score_a,
        score_b_posture=score_b_posture,
        score_b=score_b,
        final_score=final,
        risk_label=label,
        risk_color=color,
        inputs=inputs,
    )
