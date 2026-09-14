"""
Obliczanie kątów ciała potrzebnych do oceny RULA na podstawie punktów
charakterystycznych (keypoints) zwracanych przez model YOLOv8-Pose
(format COCO-17):

 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

WAŻNE OGRANICZENIE: model dostarcza tylko 2D punkty pojedynczej kamery
RGB. Kąty zgięcia tułowia/szyi liczone są jako odchylenie wektora
bark-biodro / ucho-bark od pionu obrazu, co dobrze odwzorowuje
rzeczywiste zgięcie tylko wtedy, gdy kamera patrzy na osobę Z BOKU
(profil). Przy kamerze ustawionej z przodu wynik odzwierciedla raczej
przechylenie boczne niż zgięcie do przodu — patrz README.md.
Nadgarstek i skręt nadgarstka nie są mierzalne (brak punktów dłoni)
i przyjmowane są jako neutralne w app.py.
"""

import math
from dataclasses import dataclass
from typing import Optional

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

CONF_THRESHOLD = 0.3


def _pt(keypoints, idx) -> Optional[tuple]:
    x, y, conf = keypoints[idx]
    if conf is not None and conf < CONF_THRESHOLD:
        return None
    return (float(x), float(y))


def _midpoint(a, b):
    if a is None or b is None:
        return a or b
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_from_vertical(p_from, p_to) -> Optional[float]:
    """Kąt wektora p_from->p_to względem pionu (oś Y obrazu), w stopniach.
    Zwraca wartość >=0. Dodatnia wartość = odchylenie od pionu w dowolną stronę."""
    if p_from is None or p_to is None:
        return None
    dx = p_to[0] - p_from[0]
    dy = p_from[1] - p_to[1]  # w obrazie Y rośnie w dół, odwracamy
    if dx == 0 and dy == 0:
        return 0.0
    angle = math.degrees(math.atan2(abs(dx), dy))
    return angle


def _angle_between(v1, v2) -> Optional[float]:
    if v1 is None or v2 is None:
        return None
    (x1, y1), (x2, y2) = v1, v2
    dot = x1 * x2 + y1 * y2
    n1 = math.hypot(x1, y1)
    n2 = math.hypot(x2, y2)
    if n1 == 0 or n2 == 0:
        return None
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


@dataclass
class BodyAngles:
    side: str
    trunk_angle: Optional[float] = None
    neck_angle: Optional[float] = None
    upper_arm_angle: Optional[float] = None
    lower_arm_angle: Optional[float] = None
    legs_balanced: Optional[bool] = None
    shoulder_raised: bool = False
    arm_abducted: bool = False


def choose_side(keypoints) -> str:
    """Wybiera stronę (lewa/prawa) o wyższej średniej pewności punktów."""
    left_idx = [L_SHOULDER, L_ELBOW, L_WRIST, L_HIP]
    right_idx = [R_SHOULDER, R_ELBOW, R_WRIST, R_HIP]
    left_conf = sum(keypoints[i][2] for i in left_idx) / len(left_idx)
    right_conf = sum(keypoints[i][2] for i in right_idx) / len(right_idx)
    return "left" if left_conf >= right_conf else "right"


def compute_body_angles(keypoints, side: str = "auto") -> BodyAngles:
    """keypoints: sekwencja 17 krotek (x, y, conf) w pikselach obrazu."""
    if side == "auto":
        side = choose_side(keypoints)

    if side == "left":
        shoulder_i, elbow_i, wrist_i, hip_i, ear_i, knee_i, ankle_i = (
            L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_EAR, L_KNEE, L_ANKLE)
        other_shoulder_i = R_SHOULDER
    else:
        shoulder_i, elbow_i, wrist_i, hip_i, ear_i, knee_i, ankle_i = (
            R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_EAR, R_KNEE, R_ANKLE)
        other_shoulder_i = L_SHOULDER

    shoulder = _pt(keypoints, shoulder_i)
    elbow = _pt(keypoints, elbow_i)
    wrist = _pt(keypoints, wrist_i)
    hip = _pt(keypoints, hip_i)
    ear = _pt(keypoints, ear_i)
    other_shoulder = _pt(keypoints, other_shoulder_i)

    l_hip = _pt(keypoints, L_HIP)
    r_hip = _pt(keypoints, R_HIP)
    hip_mid = _midpoint(l_hip, r_hip)
    l_shoulder = _pt(keypoints, L_SHOULDER)
    r_shoulder = _pt(keypoints, R_SHOULDER)
    shoulder_mid = _midpoint(l_shoulder, r_shoulder)

    angles = BodyAngles(side=side)

    # Tułów: odchylenie wektora biodro->bark od pionu
    angles.trunk_angle = _angle_from_vertical(hip_mid, shoulder_mid)

    # Szyja: odchylenie wektora bark->ucho od kierunku tułowia (bark->biodro odwrócone)
    if ear is not None and shoulder is not None and hip_mid is not None and shoulder_mid is not None:
        neck_vec = (ear[0] - shoulder[0], ear[1] - shoulder[1])
        trunk_vec_up = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        a = _angle_between(neck_vec, trunk_vec_up)
        angles.neck_angle = a
    else:
        angles.neck_angle = None

    # Ramię (upper arm): odchylenie wektora bark->łokieć od pionu tułowia
    if shoulder is not None and elbow is not None and hip_mid is not None and shoulder_mid is not None:
        upper_arm_vec = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        trunk_vec_down = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
        angles.upper_arm_angle = _angle_between(upper_arm_vec, trunk_vec_down)
    else:
        angles.upper_arm_angle = None

    # Przedramię (lower arm): kąt w łokciu między bark->łokieć i łokieć->nadgarstek
    if shoulder is not None and elbow is not None and wrist is not None:
        v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
        v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        angles.lower_arm_angle = _angle_between(v1, v2)
    else:
        angles.lower_arm_angle = None

    # Uniesienie barku: porównanie wysokości (y) barków, znormalizowane szerokością barków
    if l_shoulder is not None and r_shoulder is not None:
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0]) or 1.0
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
        angles.shoulder_raised = (shoulder_diff / shoulder_width) > 0.15

    # Odwiedzenie ramienia (przybliżenie): łokieć znacznie szerzej niż bark względem tułowia
    if shoulder is not None and elbow is not None and hip_mid is not None and shoulder_mid is not None:
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0]) if (l_shoulder and r_shoulder) else 1.0
        lateral_offset = abs(elbow[0] - shoulder[0])
        angles.arm_abducted = shoulder_width > 0 and (lateral_offset / shoulder_width) > 0.6

    # Nogi (praca stojąca): sprawdzenie, czy ciężar ciała jest rozłożony symetrycznie
    l_ankle = _pt(keypoints, L_ANKLE)
    r_ankle = _pt(keypoints, R_ANKLE)
    if l_ankle is not None and r_ankle is not None and l_hip is not None and r_hip is not None:
        hip_center_x = (l_hip[0] + r_hip[0]) / 2.0
        ankle_center_x = (l_ankle[0] + r_ankle[0]) / 2.0
        hip_width = abs(l_hip[0] - r_hip[0]) or 1.0
        offset = abs(hip_center_x - ankle_center_x) / hip_width
        angles.legs_balanced = offset < 0.35
    else:
        angles.legs_balanced = None  # brak danych - domyślnie przyjmowane jako zbalansowane

    return angles
