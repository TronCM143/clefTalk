#!/usr/bin/env python3
"""ClefTalk web UI: press-and-hold Raspberry Pi microphone recording + local speech model.

Run from this directory: uvicorn main:app --host 0.0.0.0 --port 8000

The browser is only a wired display/controller; sound is captured by Raspberry Pi
via ALSA (arecord). No browser microphone permission or Internet is required.
"""
from __future__ import annotations

import json
import os
import secrets
import signal
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from cleftalk import audio_from_file, features

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CLEF_DATA_DIR", str(BASE_DIR / "data"))).resolve()
RECORDINGS_DIR = DATA_DIR / "recordings"
DB_FILE = DATA_DIR / "history.sqlite3"
MODEL_FILE = Path(os.environ.get("CLEF_MODEL_PATH", str(BASE_DIR / "output" / "cleftalk_model.joblib"))).resolve()
AUDIO_DEVICE = os.environ.get("CLEF_AUDIO_DEVICE", "default")
AUDIO_FORMAT = os.environ.get("CLEF_AUDIO_FORMAT", "S16_LE")
AUDIO_CHANNELS = int(os.environ.get("CLEF_AUDIO_CHANNELS", "1"))
AUDIO_SAMPLE_RATE = int(os.environ.get("CLEF_AUDIO_SAMPLE_RATE", "16000"))
MAX_RECORD_SECONDS = 12  # cleftalk.py accepts up to 12 seconds after trimming.
TOP_K = 5

app = FastAPI(title="ClefTalk", docs_url=None, redoc_url=None)


@dataclass
class ActiveRecording:
    process: subprocess.Popen
    path: Path
    started: float


_active: ActiveRecording | None = None
_recording_lock = threading.Lock()
_processing = False


def init_storage() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE, timeout=10) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                filename TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                text TEXT NOT NULL,
                candidates_json TEXT NOT NULL
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS history_text_idx ON recordings (category, text)")
        connection.commit()


init_storage()


@lru_cache(maxsize=2)
def load_model(model_path: str, modification_time_ns: int):
    # Only load a model YOU trained or trust: joblib files execute Python during loading.
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Invalid model file: expected a trained ClefTalk payload")
    model = payload["model"]
    if not hasattr(model, "predict") or not hasattr(model, "classes_"):
        raise ValueError("Model must support predict() and classes_")
    return model


def get_model():
    if not MODEL_FILE.is_file():
        raise RuntimeError(f"Model not found: {MODEL_FILE}")
    return load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)


def parts(class_key: str) -> dict:
    category, _, spoken_text = class_key.partition("/")
    return {"category": category, "text": spoken_text or class_key}


def relative_scores(model, feature_vector: np.ndarray) -> tuple[str, list[dict]]:
    """Uncalibrated relative ranking scores, NOT acoustic similarity or certainty.

    Existing ClefTalk SVC model was trained with probability=False. Its decision
    scores are converted with softmax strictly for displaying comparable ranks.
    These percentages should NEVER be called predictive confidence/accuracy.
    """
    labels = np.asarray([str(name) for name in model.classes_])
    predicted = str(model.predict(feature_vector)[0])
    raw = np.asarray(model.decision_function(feature_vector), dtype=float)
    if len(labels) == 2:
        # Binary SVC margin corresponds to classes_[1] when positive.
        margin = float(raw.ravel()[0])
        decision = np.array([-margin, margin], dtype=float)
    elif raw.ndim == 2 and raw.shape[1] == len(labels):
        decision = raw[0]
    else:
        raise RuntimeError("Unexpected classifier scoring output")
    decision = decision - np.max(decision)
    weights = np.exp(np.clip(decision, -700, 0))
    weights /= np.sum(weights)
    order = np.argsort(weights)[::-1]
    # Prediction is the trained model's actual selected class, including ties.
    predicted_idx = np.flatnonzero(labels == predicted)
    if predicted_idx.size:
        winner = int(predicted_idx[0])
        order = np.r_[winner, order[order != winner]]
    matches = []
    for idx in order[:TOP_K]:
        match = parts(labels[idx])
        match["score_percent"] = round(float(weights[idx]) * 100, 1)
        matches.append(match)
    return predicted, matches


def analyze_audio(audio_path: Path) -> dict:
    model = get_model()
    audio = audio_from_file(audio_path)  # FFmpeg decodes WAV; trims and normalizes
    feature_vector = features(audio)[None, :]
    prediction, matches = relative_scores(model, feature_vector)
    return {**parts(prediction), "candidates": matches}


def save_prediction(audio_path: Path, duration: float, prediction: dict) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_FILE, timeout=10) as connection:
        cursor = connection.execute("""
            INSERT INTO recordings
            (created_at, duration_seconds, filename, category, text, candidates_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp, round(duration, 2), audio_path.name,
            prediction["category"], prediction["text"],
            json.dumps(prediction["candidates"], ensure_ascii=False),
        ))
        recording_id = cursor.lastrowid
        connection.commit()
    return {
        "id": recording_id,
        "created_at": timestamp,
        "duration_seconds": round(duration, 2),
        "category": prediction["category"],
        "text": prediction["text"],
        "candidates": prediction["candidates"],
        "audio_url": f"/api/history/{recording_id}/audio",
    }


def as_history_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "created_at": row["created_at"],
        "duration_seconds": row["duration_seconds"],
        "category": row["category"], "text": row["text"],
        "candidates": json.loads(row["candidates_json"]),
        "audio_url": f"/api/history/{row['id']}/audio",
    }


@app.middleware("http")
async def restrict_cross_origin_writes(request: Request, call_next):
    # Browser page must be served from the same Raspberry Pi hostname/port.
    # This is NOT a substitute for a firewall: use on an isolated wired network.
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlsplit
            parsed = urlsplit(origin)
            if parsed.netloc != request.headers.get("host") or parsed.scheme not in ("http", "https"):
                return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "static" / "index.html", media_type="text/html")


@app.get("/api/status")
def status():
    try:
        get_model()
        model_ready = True
        model_error = None
    except Exception as exc:
        model_ready = False
        model_error = str(exc)
    with _recording_lock:
        is_recording = _active is not None
        is_processing = _processing
    return {
        "model_ready": model_ready,
        "model_error": model_error,
        "microphone": f"Raspberry Pi ALSA: {AUDIO_DEVICE}",
        "recording": is_recording,
        "processing": is_processing,
        "max_seconds": MAX_RECORD_SECONDS,
    }


@app.post("/api/record/start")
def start_recording():
    global _active
    with _recording_lock:
        if _active is not None or _processing:
            raise HTTPException(409, "Another recording is already active or being processed")
        try:
            get_model()
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
        file_path = RECORDINGS_DIR / f"{secrets.token_hex(16)}.wav"
        command = [
            "arecord", "-q", "-D", AUDIO_DEVICE, "-f", AUDIO_FORMAT,
            "-c", str(AUDIO_CHANNELS), "-r", str(AUDIO_SAMPLE_RATE),
            "-t", "wav", "-d", str(MAX_RECORD_SECONDS), str(file_path),
        ]
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise HTTPException(503, "arecord is missing; install alsa-utils") from exc
        except OSError as exc:
            raise HTTPException(503, f"Cannot start Raspberry Pi microphone: {exc}") from exc
        # Let the recorder initialize; distinguish a failed ALSA configuration early.
        time.sleep(0.12)
        if process.poll() is not None:
            error = (process.stderr.read().decode(errors="replace")[-350:]
                     if process.stderr else "Unknown ALSA error")
            file_path.unlink(missing_ok=True)
            raise HTTPException(503, f"Microphone capture failed: {error}")
        _active = ActiveRecording(process=process, path=file_path, started=time.monotonic())
    return {"recording": True, "max_seconds": MAX_RECORD_SECONDS}


@app.post("/api/record/stop")
def stop_recording():
    global _active, _processing
    with _recording_lock:
        if _active is None:
            raise HTTPException(409, "No active recording. Press and hold the microphone button.")
        active = _active
        _active = None
        _processing = True
    duration = min(MAX_RECORD_SECONDS, time.monotonic() - active.started)
    try:
        if active.process.poll() is None:
            active.process.send_signal(signal.SIGINT)
        try:
            active.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            active.process.kill()
            active.process.wait(timeout=2)
            raise RuntimeError("Microphone could not stop cleanly; recording discarded")
        if duration < 0.17:
            raise ValueError("Recording too short. Press and hold while speaking.")
        if not active.path.is_file() or active.path.stat().st_size < 150:
            diagnostic = (active.process.stderr.read().decode(errors="replace")[-250:]
                          if active.process.stderr else "")
            raise RuntimeError(f"No usable audio was captured. {diagnostic}")
        prediction = analyze_audio(active.path)
        return save_prediction(active.path, duration, prediction)
    except (RuntimeError, ValueError, OSError) as exc:
        active.path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except Exception:
        active.path.unlink(missing_ok=True)
        raise
    finally:
        if active.process.stderr:
            active.process.stderr.close()
        with _recording_lock:
            _processing = False


@app.get("/api/history")
def history(limit: int = Query(default=300, ge=1, le=1000)):
    """Most recent N recordings, grouped by recognized text."""
    with sqlite3.connect(DB_FILE, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT * FROM recordings ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        total = connection.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    recent = [as_history_row(row) for row in rows]
    by_text: dict[tuple[str, str], dict] = {}
    for entry in recent:
        key = (entry["category"], entry["text"])
        if key not in by_text:
            by_text[key] = {
                "category": entry["category"], "text": entry["text"],
                "count": 0, "recordings": [],
            }
        by_text[key]["recordings"].append(entry)
        by_text[key]["count"] += 1
    return {"total": total, "showing": len(recent),
            "groups": list(by_text.values()), "recent": recent[:5]}


@app.get("/api/history/{recording_id}/audio")
def history_audio(recording_id: int):
    with sqlite3.connect(DB_FILE, timeout=10) as connection:
        row = connection.execute("SELECT filename FROM recordings WHERE id = ?", (recording_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Recording not found")
    # The filename was generated by the server; prevent accidental path traversal.
    filename = row[0]
    if Path(filename).name != filename or not filename.endswith(".wav"):
        raise HTTPException(404, "Invalid recording file")
    file_path = RECORDINGS_DIR / filename
    if not file_path.is_file():
        raise HTTPException(404, "Recorded audio file is missing")
    return FileResponse(file_path, media_type="audio/wav", filename=filename,
                        content_disposition_type="inline")
