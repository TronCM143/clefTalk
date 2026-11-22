#!/usr/bin/env python3
"""Record one utterance from Raspberry Pi ALSA input, then classify it.

Example: python record_and_predict.py --device default --seconds 4
The INMP441 must already appear as an ALSA capture device (check: arecord -l).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="default", help="ALSA capture device e.g. default or plughw:1,0")
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--model", default="output/cleftalk_model.joblib")
    args = parser.parse_args()
    if not (1 <= args.seconds <= 12):
        parser.error("--seconds must be 1–12")
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "sample.wav"
        command = ["arecord", "-D", args.device, "-f", "S16_LE", "-c", "1", "-r", "16000", "-d", str(args.seconds), str(path)]
        print("Recording now. Speak one word or phrase...", flush=True)
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError:
            parser.error("arecord is not installed; run: sudo apt install alsa-utils")
        except subprocess.CalledProcessError as exc:
            parser.error(f"ALSA recording failed: {exc}. Run 'arecord -l' to find your capture device.")
        subprocess.run([sys.executable, str(Path(__file__).with_name("cleftalk.py")), "predict", "--model", args.model, "--audio", str(path)], check=True)


if __name__ == "__main__":
    main()
