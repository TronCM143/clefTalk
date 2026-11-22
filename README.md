# ClefTalk — Small-vocabulary cleft-palate speech recognition starter

**This is a runnable TRAINING PIPELINE, not a model already trained on your voice.**

It learns acoustic patterns for the exact words/phrases present in your folders. Inference can run offline on a Raspberry Pi 4 or 5 after training on Windows/Linux. It **cannot recognize unseen words/sentences**, and accuracy on new speakers is not guaranteed.

## 1. Folder structure

Put your voice dataset in a folder (e.g., `C:\Users\tron\OneDrive\Desktop\VoiceRecordings\labeled`). You may pass the `labeled` folder itself or its parent `VoiceRecordings` as the `--data` argument. Supported category folder names are **words**, **word**, **phrase**, or **phrases** (case insensitive). Every subfolder becomes one class. Its name is the expected text:

```text
VoiceRecordings/
  labeled/
    words/
      hi/
        audio1.mp3
        audio2.mp3
        ...
      hello/
        audio1.mp3
        ...
    phrases/
      i love you/
        audio1.mp3
        ...
      i have you/
        audio1.mp3
        ...
```

Audio may be MP3, WAV, M4A, FLAC, OGG, OPUS, AAC, WMA, WEBM **or extensionless MP3 files** (FFmpeg detects the format). One audio file should contain **one spoken instance** of its folder label, not 10 repetitions combined. Don't add unrelated files in label folders.

For research-grade validation, retain speaker and recording-session identifiers outside this directory too (the starter only uses a clip-level split; it cannot detect overlap from the same speaker/session).

## 2. Setup on your Windows 11 training PC

Install Python 3.10+ and FFmpeg. In PowerShell in the extracted `ClefTalk_Starter` directory:

```powershell
# Install FFmpeg using Windows Package Manager (or install it manually)
winget install --id Gyan.FFmpeg --exact
# Close PowerShell and reopen so ffmpeg is on your PATH.
ffmpeg -version

py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` in each command; no activation is required.

## 3. Count/verify your labels and audio

```powershell
python cleftalk.py check --data "C:\Users\tron\OneDrive\Desktop\VoiceRecordings\labeled" --decode
```

You should see nearly 100 categories and roughly 9–10 clips per category. Bad clips will be printed.

## 4. Train and evaluate

```powershell
python cleftalk.py train --data "C:\Users\tron\OneDrive\Desktop\VoiceRecordings\labeled" --out output --augment 1
```

The training process:

1. Reads every MP3/WAV recording (16 kHz, mono) and extracts MFCC + delta + temporal features.
2. Holds out 20% of the **original files** per class for an initial test. (The classifier doesn't have access to the held-out clips when predicting them.)
3. Adds synthetic small speed/noise changes **only to the training split**.
4. Trains a lightweight RBF Support Vector Machine.
5. Saves held-out evaluation (`output/evaluation.json`, `output/validation_predictions.csv`), class counts, and any skipped files.
6. Retrains a **deployment version on all valid original recordings plus augmentations** and saves `output/cleftalk_model.joblib`.

The evaluation measures clip-level recognition, not unseen-speaker performance. If all audio was captured by one speaker in one session, test with new speech captured on another day. For multiple speakers, a speaker-independent test should hold out entire speakers (requires speaker metadata and a more advanced split). With ~9–10 clips/class, accuracy may be limited; never claim clinical performance without real-world validation.

To make a model that retains the original split rather than refitting on all recordings, add `--no-refit-all`. To disable augmentation, use `--augment 0`.

## 5. Predict new audio

```powershell
python cleftalk.py predict --model output/cleftalk_model.joblib --audio "C:\path\to\new_recording.mp3"
```

Sample output:

```text
Predicted phrase: i love you
Category: phrases
Closest candidates (ranking, not confidence %):
  1. phrases/i love you
  2. phrases/i have you
  3. words/hello
```

These are example labels, **not measured results**. Prediction always chooses a known class and should not be trusted as an indication that the person said exactly that. For live use, consider a confirmation button or a calibrated reject/unknown class based on additional data.

## 6. Deploy on Raspberry Pi OS (64-bit)

Copy **the extracted Python scripts, requirements, and `output/cleftalk_model.joblib`** to the Raspberry Pi. Keep the same Python/scikit-learn versions on the two devices where possible; `joblib` is not a cross-version standardized model format. You don't need the entire training dataset on the Pi for inference.

```bash
sudo apt update
sudo apt install -y python3-venv ffmpeg alsa-utils
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cleftalk.py predict --model output/cleftalk_model.joblib --audio sample.wav
```

Your INMP441 microphone needs to be configured as a functioning ALSA capture source before using:

```bash
arecord -l
python record_and_predict.py --device default --seconds 4 --model output/cleftalk_model.joblib
```

If `arecord` reports the wrong device, use its actual ID, e.g. `--device plughw:1,0` **only if `arecord -l` shows it**. First check microphone and amp hardware separately.

## 7. How to increase accuracy

- Record **more than 9–10 samples per label** if possible, with variations in normal speaking speed, loudness, environment, and pronunciation.
- Include **easily confused pairs** like `I love you` vs. `I have you` and inspect `validation_predictions.csv`.
- Avoid nearly identical duplicates. Use actual new speech with the target speaker on different days for a stronger test.
- Ensure each clip is correctly labeled, only contains one utterance, and has minimal background TV/music.
- Collect **unlisted speech, silence, coughing, and background noises** to test (and later train) an explicit rejection or unknown mechanism.
- For new-speaker and free-form transcription, plan a second-stage pretrained speech model adaptation (Whisper/wav2vec2 family) using a larger, diverse dataset; this starter is not that solution.

## Ethical research note

Cleft-palate speech varies significantly across speakers and contexts. Don't assume a vocabulary classifier trained on one person's speech works for other people with cleft palate. Collect recordings with consent; safeguard identifying voice recordings; evaluate on representative new utterances before assistive use.
