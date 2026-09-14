"""
Aplikacja do oceny ergonomii stanowiska pracy (RULA) w czasie
rzeczywistym na podstawie obrazu z kamery (np. Aurora 930 lub kamera
wbudowana MacBooka) i estymacji pozy YOLOv8-Pose.

Uruchomienie:
    python app.py
a nastepnie otworzenie w przegladarce: http://localhost:8000

Uwaga macOS: przy pierwszym uruchomieniu aplikacja proboje na glownym
watku "rozgrzac" kamere, zeby system pokazal okienko z prosba o zgode
na dostep do kamery (AVFoundation potrafi to okienko pokazac tylko z
glownego watku procesu) - zaakceptuj je, gdy sie pojawi.

Zobacz README.md po pelne instrukcje instalacji i uwagi dotyczace
ograniczen metody (nadgarstek, sila/obciazenie, kat tulowia).
"""

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import rula
from pose_utils import compute_body_angles

MODEL_PATH = os.environ.get("RULA_YOLO_MODEL", "yolov8n-pose.pt")
DEFAULT_CAMERA_INDEX = int(os.environ.get("RULA_CAMERA_INDEX", "0"))
STATIC_POSTURE_SECONDS = 60.0   # próg "postawa statyczna" -> +1 muscle score
POSTURE_BUCKET_TOLERANCE = 1    # tolerancja zmiany wyniku Score A/B uznawana za "tę samą" postawę
HISTORY_MAXLEN = 600            # ~10 minut przy probkowaniu co 1s
LOG_PATH = os.environ.get("RULA_LOG_PATH", "rula_log.csv")

app = Flask(__name__)

_lock = threading.Lock()
_state = {
    "camera_index": DEFAULT_CAMERA_INDEX,
    "side": "auto",
    "force_score": 0,          # 0-3, ustawiane recznie przez uzytkownika
    "assume_legs_ok": True,    # nadpisanie recznie oceny nog, gdy detekcja niepewna
    "running": True,
    "latest_jpeg": None,
    "latest_result": None,     # dict do JSON
    "latest_angles": None,
    "error": None,
    "posture_since": None,
    "posture_bucket": None,
}

_history = deque(maxlen=HISTORY_MAXLEN)

SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # ramiona / przedramiona
    (5, 6), (5, 11), (6, 12), (11, 12),       # tulow
    (11, 13), (13, 15), (12, 14), (14, 16),   # nogi
    (0, 5), (0, 6),                            # glowa -> barki
]


def _ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "final_score", "risk", "score_a", "score_b",
                              "upper_arm", "lower_arm", "wrist", "neck", "trunk", "legs"])


def _log_result(result: rula.RulaResult):
    try:
        _ensure_log_header()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            inp = result.inputs
            writer.writerow([datetime.now().isoformat(timespec="seconds"),
                              result.final_score, result.risk_label,
                              result.score_a, result.score_b,
                              inp.upper_arm, inp.lower_arm, inp.wrist,
                              inp.neck, inp.trunk, inp.legs])
    except OSError:
        pass


def _pick_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _to_rula_inputs(angles, force_score: int, assume_legs_ok: bool, muscle_flag: bool):
    ua = angles.upper_arm_angle if angles.upper_arm_angle is not None else 20.0
    la = angles.lower_arm_angle if angles.lower_arm_angle is not None else 90.0
    trunk = angles.trunk_angle if angles.trunk_angle is not None else 0.0
    neck = angles.neck_angle if angles.neck_angle is not None else 0.0

    legs_ok = angles.legs_balanced if angles.legs_balanced is not None else assume_legs_ok

    upper_arm = rula.upper_arm_score(ua, raised=angles.shoulder_raised,
                                      abducted=angles.arm_abducted)
    lower_arm = rula.lower_arm_score(la)
    wrist = rula.wrist_score(0.0)          # niemierzalne z pozy -> neutralne
    wrist_twist = rula.wrist_twist_score(True)
    neck_s = rula.neck_score(neck)
    trunk_s = rula.trunk_score(trunk)
    legs_s = rula.legs_score(supported_balanced=legs_ok)

    muscle = 1 if muscle_flag else 0

    return rula.RulaInputs(
        upper_arm=upper_arm, lower_arm=lower_arm, wrist=wrist, wrist_twist=wrist_twist,
        neck=neck_s, trunk=trunk_s, legs=legs_s,
        muscle_arm=muscle, force_arm=force_score,
        muscle_body=muscle, force_body=force_score,
    )


def _posture_bucket(inputs: rula.RulaInputs):
    return (inputs.upper_arm, inputs.lower_arm, inputs.neck, inputs.trunk, inputs.legs)


def _bucket_matches(a, b, tol=POSTURE_BUCKET_TOLERANCE):
    if a is None or b is None:
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _draw_overlay(frame, keypoints, angles, result: rula.RulaResult):
    h, w = frame.shape[:2]
    for i, j in SKELETON_EDGES:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci > 0.3 and cj > 0.3:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 200, 0), 2)
    for x, y, c in keypoints:
        if c > 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 140, 255), -1)

    label = f"RULA: {result.final_score}  ({result.risk_label})"
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _capture_loop():
    from ultralytics import YOLO

    device = _pick_device()
    model = YOLO(MODEL_PATH)

    cap = None
    current_index = None

    while True:
        with _lock:
            running = _state["running"]
            wanted_index = _state["camera_index"]
            side = _state["side"]
            force_score = _state["force_score"]
            assume_legs_ok = _state["assume_legs_ok"]

        if not running:
            time.sleep(0.2)
            continue

        if cap is None or wanted_index != current_index:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(wanted_index)
            current_index = wanted_index
            if not cap.isOpened():
                with _lock:
                    _state["error"] = f"Nie mozna otworzyc kamery o indeksie {wanted_index}"
                time.sleep(1.0)
                continue
            else:
                with _lock:
                    _state["error"] = None

        ok, frame = cap.read()
        if not ok or frame is None:
            with _lock:
                _state["error"] = "Brak klatki z kamery"
            time.sleep(0.2)
            continue

        results = model.predict(frame, device=device, verbose=False)
        person_found = False

        if results and len(results) > 0 and results[0].keypoints is not None \
                and results[0].keypoints.data is not None and len(results[0].keypoints.data) > 0:
            kp_tensor = results[0].keypoints.data[0]  # (17, 3) dla pierwszej wykrytej osoby
            keypoints = [(float(x), float(y), float(c)) for x, y, c in kp_tensor.cpu().numpy()]
            person_found = True

            angles = compute_body_angles(keypoints, side=side)

            with _lock:
                bucket = _state["posture_bucket"]
                since = _state["posture_since"]
            candidate = None
            muscle_flag = False
            now = time.time()

            # tymczasowe obliczenie inputow (bez flagi miesniowej) zeby uzyskac "kubelek" postawy
            provisional = _to_rula_inputs(angles, force_score, assume_legs_ok, muscle_flag=False)
            candidate = _posture_bucket(provisional)

            if _bucket_matches(bucket, candidate):
                if since is not None and (now - since) >= STATIC_POSTURE_SECONDS:
                    muscle_flag = True
            else:
                since = now
                bucket = candidate

            rula_inputs = _to_rula_inputs(angles, force_score, assume_legs_ok, muscle_flag)
            result = rula.compute_rula(rula_inputs)

            frame = _draw_overlay(frame, keypoints, angles, result)

            with _lock:
                _state["posture_bucket"] = bucket
                _state["posture_since"] = since
                _state["latest_angles"] = {
                    "side": angles.side,
                    "trunk_angle": angles.trunk_angle,
                    "neck_angle": angles.neck_angle,
                    "upper_arm_angle": angles.upper_arm_angle,
                    "lower_arm_angle": angles.lower_arm_angle,
                    "legs_balanced": angles.legs_balanced,
                }
                _state["latest_result"] = {
                    "final_score": result.final_score,
                    "risk_label": result.risk_label,
                    "risk_color": result.risk_color,
                    "score_a": result.score_a,
                    "score_b": result.score_b,
                    "inputs": {
                        "upper_arm": rula_inputs.upper_arm,
                        "lower_arm": rula_inputs.lower_arm,
                        "wrist": rula_inputs.wrist,
                        "neck": rula_inputs.neck,
                        "trunk": rula_inputs.trunk,
                        "legs": rula_inputs.legs,
                        "muscle": muscle_flag,
                        "force": force_score,
                    },
                    "static_seconds": round(now - since, 1) if since else 0.0,
                }
                _history.append({"t": now, "score": result.final_score})
            if int(now) % 5 == 0:
                _log_result(result)

        if not person_found:
            cv2.putText(frame, "Nie wykryto osoby", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with _lock:
                _state["latest_jpeg"] = buf.tobytes()

        time.sleep(0.03)  # ~30 fps max przetwarzania


def _mjpeg_generator():
    while True:
        with _lock:
            jpeg = _state["latest_jpeg"]
        if jpeg is None:
            time.sleep(0.1)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "result": _state["latest_result"],
            "angles": _state["latest_angles"],
            "error": _state["error"],
            "settings": {
                "camera_index": _state["camera_index"],
                "side": _state["side"],
                "force_score": _state["force_score"],
                "assume_legs_ok": _state["assume_legs_ok"],
            },
            "history": list(_history)[-120:],
        })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True, silent=True) or {}
    with _lock:
        if "camera_index" in data:
            _state["camera_index"] = int(data["camera_index"])
        if "side" in data and data["side"] in ("auto", "left", "right"):
            _state["side"] = data["side"]
        if "force_score" in data:
            _state["force_score"] = max(0, min(3, int(data["force_score"])))
        if "assume_legs_ok" in data:
            _state["assume_legs_ok"] = bool(data["assume_legs_ok"])
    return jsonify({"ok": True})


@app.route("/api/cameras")
def api_cameras():
    available = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            available.append(idx)
        cap.release()
    return jsonify({"cameras": available})


def _warm_up_camera_permission(index: int):
    """Na macOS system moze pokazac okienko z prosba o dostep do kamery
    (AVFoundation) tylko wtedy, gdy pierwsze otwarcie kamery nastapi na
    GLOWNYM watku procesu. Jesli otwieramy kamere dopiero w watku w tle
    (_capture_loop), macOS cicho odmawia dostepu (blad w konsoli:
    "not authorized to capture video ... can not spin main run loop
    from other thread"). Dlatego robimy to raz, tutaj, przed startem
    watku przechwytywania."""
    try:
        cap = cv2.VideoCapture(index)
        cap.read()
        cap.release()
    except Exception:
        pass


if __name__ == "__main__":
    _warm_up_camera_permission(DEFAULT_CAMERA_INDEX)
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()
    # UWAGA (macOS): port 5000 jest domyslnie zajety przez usluge
    # AirPlay Receiver (Ustawienia systemowe -> Ogolne -> AirDrop i Uchwyt),
    # ktora odpowiada strona "HTTP ERROR 403" zamiast tej aplikacji.
    # Dlatego domyslnie uzywany jest port 8000 - mozna to zmienic zmienna PORT.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False, threaded=True)
