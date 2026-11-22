#!/usr/bin/env python3
"""ClefTalk: small-vocabulary cleft-palate speech classifier.

Training labels come from folder names. This is a closed-vocabulary acoustic
classifier, NOT an open-ended speech-to-text model.

Examples:
    python cleftalk.py check --data "Voice Recordings"
    python cleftalk.py train --data "Voice Recordings" --out output
    python cleftalk.py predict --model output/cleftalk_model.joblib --audio test.mp3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly, stft
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SR = 16_000
N_MFCC = 20
N_MELS = 40
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".webm", ""}
GROUPS = {"word": "words", "words": "words", "phrase": "phrases", "phrases": "phrases"}


def scan_dataset(data_dir: Path) -> list[tuple[str, str, Path]]:
    """(class_key, spoken_text, file_path) from category/label/audio hierarchy."""
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    # Allow either VoiceRecordings/labeled or its parent VoiceRecordings.
    has_groups = any(p.is_dir() and p.name.casefold() in GROUPS for p in data_dir.iterdir())
    if not has_groups and (data_dir / "labeled").is_dir():
        data_dir = data_dir / "labeled"
    items = []
    used = set()
    for group_dir in sorted(data_dir.iterdir()):
        if not group_dir.is_dir() or group_dir.name.casefold() not in GROUPS:
            continue
        category = GROUPS[group_dir.name.casefold()]
        for label_dir in sorted(group_dir.iterdir()):
            if not label_dir.is_dir() or label_dir.name.startswith("."):
                continue
            phrase = label_dir.name.strip()
            key = f"{category}/{phrase}"
            for filepath in sorted(label_dir.rglob("*")):
                if filepath.is_file() and filepath.suffix.lower() in AUDIO_EXTS and not filepath.name.startswith("."):
                    identifier = (key.casefold(), str(filepath.resolve()).casefold())
                    if identifier not in used:
                        items.append((key, phrase, filepath))
                        used.add(identifier)
    if not items:
        raise ValueError("No audio found. Expected DATA/labeled/words/hi/audio1.mp3 or DATA/labeled/phrases/i love you/audio1.mp3")
    return items


def audio_from_file(path: Path) -> np.ndarray:
    """Use ffmpeg so MP3, WAV, M4A and extensionless MP3/ID3 work the same."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found. Install FFmpeg and restart your terminal.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Audio conversion timed out: {path}") from exc
    if proc.returncode != 0:
        raise ValueError(f"Cannot decode {path}: {proc.stderr.decode(errors='replace')[-350:]}")
    x = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)
    if x.size < int(SR * 0.15) or not np.all(np.isfinite(x)):
        raise ValueError(f"Too short, empty or invalid audio: {path}")
    x = trim_silence(x)
    if x.size > 12 * SR:
        raise ValueError(f"Recording exceeds 12 seconds after silence trimming (one utterance per file): {path}")
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-6:
        raise ValueError(f"Silent audio: {path}")
    x = x * (0.08 / rms)
    peak = np.max(np.abs(x))
    if peak > 0.99:
        x = x / peak * 0.99
    return x.astype(np.float32)


def trim_silence(x: np.ndarray) -> np.ndarray:
    hop, frame = 160, 320
    if len(x) < frame * 2:
        return x
    rms = np.array([
        np.sqrt(np.mean(x[i:i + frame] ** 2))
        for i in range(0, len(x) - frame + 1, hop)
    ])
    if not rms.size or float(np.max(rms)) < 1e-7:
        return x
    active = np.flatnonzero(rms >= max(1e-5, float(np.max(rms)) * 0.055))
    if active.size == 0:
        return x
    lead = int(0.17 * SR)
    first = max(0, int(active[0]) * hop - lead)
    last = min(len(x), int(active[-1]) * hop + frame + lead)
    return x[first:last]


def mel_bank(sr: int = SR, n_fft: int = 512, n_mels: int = N_MELS) -> np.ndarray:
    m2h = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    h2m = lambda h: 2595.0 * np.log10(1.0 + h / 700.0)
    hz = m2h(np.linspace(h2m(50.0), h2m(sr / 2), n_mels + 2))
    frequencies = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    bank = np.zeros((n_mels, len(frequencies)), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = hz[i:i + 3]
        rise = (frequencies - left) / max(center - left, 1e-9)
        fall = (right - frequencies) / max(right - center, 1e-9)
        bank[i] = np.maximum(0.0, np.minimum(rise, fall))
    return bank


BANK = mel_bank()


def features(audio: np.ndarray) -> np.ndarray:
    """MFCC + deltas, global statistics, and ordered 5-part temporal pooling."""
    if audio.size < 2400:
        raise ValueError("Audio must contain at least 0.15 seconds of speech")
    x = audio.astype(np.float64)
    emphasized = np.concatenate(([x[0]], x[1:] - 0.97 * x[:-1]))
    _, _, z = stft(
        emphasized, fs=SR, window="hann", nperseg=400, noverlap=240,
        nfft=512, boundary="zeros", padded=True,
    )
    power = np.abs(z) ** 2
    log_mel = np.log(np.maximum(1e-12, BANK @ power))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]
    delta = np.gradient(mfcc, axis=1)
    delta2 = np.gradient(delta, axis=1)
    matrix = np.concatenate((mfcc, delta, delta2), axis=0)
    parts = [matrix.mean(axis=1), matrix.std(axis=1)]
    for chunk in np.array_split(np.arange(matrix.shape[1]), 5):
        parts.append(matrix[:, chunk].mean(axis=1))
    parts.append(np.array([len(audio) / SR]))
    return np.concatenate(parts).astype(np.float32)


def augmentation(audio: np.ndarray, iteration: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic speech variants are derived ONLY from training clips."""
    rate = [0.94, 1.06, 0.98, 1.02][iteration % 4]
    result = resample_poly(audio, 100, round(100 * rate)).astype(np.float32)
    noise_strength = [0.002, 0.004, 0.001, 0.003][iteration % 4]
    result += rng.normal(0, noise_strength, len(result)).astype(np.float32)
    return result


def feature_rows(items, *, verbose=True):
    xs, ys, good, failed = [], [], [], []
    for index, (label, text, path) in enumerate(items, start=1):
        try:
            audio = audio_from_file(path)
            xs.append(features(audio))
            ys.append(label)
            good.append((label, text, path))
        except (ValueError, RuntimeError) as exc:
            failed.append((str(path), str(exc)))
        if verbose and (index % 50 == 0 or index == len(items)):
            print(f"Processed {index}/{len(items)} files", flush=True)
    if not xs:
        raise ValueError("None of the audio recordings could be processed")
    return np.vstack(xs), np.array(ys), good, failed


def build_training_X(base_x, base_y, items, indices, n_aug, seed):
    matrices = [base_x[indices]]
    labels = [base_y[indices]]
    if n_aug:
        rng = np.random.default_rng(seed)
        additions, classes = [], []
        for index in indices:
            audio = audio_from_file(items[index][2])
            for i in range(n_aug):
                additions.append(features(augmentation(audio, i, rng)))
                classes.append(base_y[index])
        if additions:
            matrices.append(np.vstack(additions))
            labels.append(np.array(classes))
    return np.vstack(matrices), np.concatenate(labels)


def build_model(C):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced"))


def do_check(args):
    items = scan_dataset(Path(args.data))
    counts = Counter(label for label, _, _ in items)
    print(f"Folders/classes: {len(counts)} | Audio files: {len(items)}")
    for label, n in sorted(counts.items()):
        warning = "  <-- FEW EXAMPLES" if n < 7 else ""
        print(f"  {label}: {n}{warning}")
    if args.decode:
        _, _, _, failed = feature_rows(items)
        print(f"Unreadable audio files: {len(failed)}")
        for path, err in failed:
            print(f"  {path}: {err}")


def do_train(args):
    if args.augment < 0 or args.augment > 4:
        raise ValueError("--augment must be between 0 and 4")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = scan_dataset(Path(args.data))
    X, y, good, failed = feature_rows(items)
    if failed:
        with (out / "skipped_files.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "error"])
            writer.writerows(failed)
        print(f"WARNING: {len(failed)} unreadable file(s). Details: {out / 'skipped_files.csv'}")
    counts = Counter(y)
    if len(counts) < 2:
        raise ValueError("At least 2 different word/phrase classes are required")
    too_small = {label: n for label, n in counts.items() if n < 3}
    if too_small:
        raise ValueError(f"Each class needs 3+ recordings to run evaluation. Too few: {too_small}")
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=args.test_size, stratify=y, random_state=args.seed)
    # Augment only AFTER the split, otherwise test audio leaks into training.
    train_x, train_y = build_training_X(X, y, good, train_idx, args.augment, args.seed)
    model = build_model(args.C)
    print(f"Training on {len(train_idx)} original + {len(train_x) - len(train_idx)} augmented clips; evaluating on {len(val_idx)} held-out originals.")
    model.fit(train_x, train_y)
    actual = y[val_idx]
    predicted = model.predict(X[val_idx])
    accuracy = float(accuracy_score(actual, predicted))
    balanced = float(balanced_accuracy_score(actual, predicted))
    print(f"Held-out accuracy: {accuracy:.3f} | Balanced accuracy: {balanced:.3f}")
    labels = list(model.classes_)
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    meta = {
        "model_type": "MFCC + delta + temporal pooled features, RBF SVM",
        "task": "closed-vocabulary speech classification",
        "sample_rate_hz": SR,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_recordings": len(good),
        "trained_classes": len(counts),
        "augmentation_copies_per_file": args.augment,
        "validation_fraction": args.test_size,
        "seed": args.seed,
        "held_out_accuracy": accuracy,
        "held_out_balanced_accuracy": balanced,
        "refit_all": not args.no_refit_all,
    }
    with (out / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": meta, "per_class": report}, f, indent=2, ensure_ascii=False)
    with (out / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "actual", "predicted", "correct"])
        for idx, true, pred in zip(val_idx, actual, predicted):
            writer.writerow([str(good[idx][2]), true, pred, bool(true == pred)])
    with (out / "dataset_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "file_count"])
        writer.writerows(sorted(counts.items()))
    if not args.no_refit_all:
        print("Refitting deployment model on ALL valid recordings (held-out report remains from evaluation model)...", flush=True)
        full_x, full_y = build_training_X(X, y, good, indices, args.augment, args.seed)
        model = build_model(args.C)
        model.fit(full_x, full_y)
    payload = {"model": model, "metadata": meta, "classes": labels}
    model_path = out / "cleftalk_model.joblib"
    joblib.dump(payload, model_path, compress=3)
    print(f"Saved: {model_path}")
    print(f"Saved evaluation: {out / 'evaluation.json'}")
    print("Important: these metrics are NOT evidence of generalization to new speakers.")


def do_predict(args):
    # IMPORTANT: Only open model files created by you; joblib/pickle isn't safe for untrusted files.
    payload = joblib.load(args.model)
    model = payload["model"]
    audio = audio_from_file(Path(args.audio))
    feat = features(audio)[None, :]
    predicted = str(model.predict(feat)[0])
    scores = model.decision_function(feat)
    ranked = []
    if scores.ndim == 2 and scores.shape[1] == len(model.classes_):
        order = np.argsort(scores[0])[::-1]
        ranked = [str(model.classes_[i]) for i in order]
    candidates = [predicted] + [label for label in ranked if label != predicted]
    candidates = candidates[:max(1, args.top_k)]
    print("Predicted phrase:", predicted.split("/", 1)[1])
    print("Category:", predicted.split("/", 1)[0])
    print("Closest candidates (ranking, not confidence %):")
    for rank, candidate in enumerate(candidates, 1):
        print(f"  {rank}. {candidate}")
    if args.json:
        print(json.dumps({"text": predicted.split("/", 1)[1], "category": predicted.split("/", 1)[0], "candidates": candidates}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check", help="Count labels and optionally validate all audio")
    check.add_argument("--data", required=True, help="Path containing words/ and phrases/, or its parent with labeled/")
    check.add_argument("--decode", action="store_true", help="Try decoding every recording")
    check.set_defaults(fn=do_check)
    train = commands.add_parser("train", help="Train and evaluate a custom closed-vocabulary model")
    train.add_argument("--data", required=True)
    train.add_argument("--out", default="output")
    train.add_argument("--test-size", type=float, default=0.20)
    train.add_argument("--augment", type=int, default=1, help="Synthetic variants per TRAINING file, 0-4 (default: 1)")
    train.add_argument("--C", type=float, default=5.0, help="SVM regularization hyperparameter")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-refit-all", action="store_true", help="Keep the held-out split model rather than retraining on all recordings")
    train.set_defaults(fn=do_train)
    pred = commands.add_parser("predict", help="Recognize an existing recorded audio file")
    pred.add_argument("--model", required=True)
    pred.add_argument("--audio", required=True)
    pred.add_argument("--top-k", type=int, default=3)
    pred.add_argument("--json", action="store_true", help="Also print a JSON result (last line)")
    pred.set_defaults(fn=do_predict)
    args = parser.parse_args()
    try:
        args.fn(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
