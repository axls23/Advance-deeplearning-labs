# Experiment 3 - Speech recognition with Whisper-tiny (English + Punjabi)

Fine-tunes `openai/whisper-tiny` and transcribes audio. Run everything from this folder with the project `.venv`.

## Layout
| Path | What |
|---|---|
| `speech_recognition.py` | Train (English, Punjabi or both), evaluate (`--eval-only`) and transcribe. One script. |
| `infer.py` | Inference: long-audio chunking, language auto-detect, repetition guard, optional vocal separation (Demucs). |
| `prepare_fleurs_pa.py`, `prepare_librispeech_en.py` | One-off: build local FLAC + TSV data in `data/`. |
| `data/en`, `data/pa` | Local training/validation audio (`train/`, `validation/`, `*.tsv`). |
| `models/` | `whisper_tiny_mix` (use this), `whisper_tiny_ft` (English), `whisper_tiny_pa` (Punjabi-only, see notes). |
| `results/` | Per-epoch `history_<ft|pa|mix>.json`. |
| `samples/` | Example audio. |

## Usage
```
python infer.py "samples/audio_sample.mp3"                  # mix model, language auto-detected
python infer.py clip.wav --lang pa                           # force Punjabi
python infer.py --live                                      # microphone: each phrase is printed when you pause (Ctrl+C stops)
python infer.py song.mp3 --separate                         # music: isolate vocals (Demucs) first
python infer.py clip.wav --model openai/whisper-small        # any Hugging Face model or local folder
```
Quote paths that contain spaces.

## Rebuild data / retrain
```
python prepare_fleurs_pa.py            # downloads ~1.6 GB parquet (HF cache), writes ~0.3 GB to data/pa
python prepare_librispeech_en.py       # streams 1,000 English clips into data/en
python speech_recognition.py           # Punjabi + English "mix" model from base whisper-tiny (4 epochs)
python speech_recognition.py --langs pa            # Punjabi only
python speech_recognition.py --eval-only --init mix   # re-score a saved model
```

## Results (100 validation clips per language, text normalised: lowercase, no punctuation, vowel signs kept)
| Model | English WER | Punjabi WER | Punjabi CER |
|---|---|---|---|
| stock whisper-tiny | **0.061** | 1.509 | 1.206 |
| `whisper_tiny_mix` (epoch 3) | 0.149 | **0.675** | **0.375** |

Fine-tuning on Punjabi cost some English accuracy (0.061 -> 0.149 WER). Punjabi is still weak (WER 0.68): 39M parameters, ~1.9k clips.

## Notes
- Data: Google FLEURS (https://huggingface.co/datasets/google/fleurs, CC-BY-4.0) and LibriSpeech.
- `models/whisper_tiny_pa` and `models/whisper_tiny_ft` were trained before a label-masking fix (pad id == end-of-text id,
  so the model never learned to stop and could loop). Retrain or prefer `whisper_tiny_mix`.
- An earlier version of the scoring code stripped Gurmukhi vowel signs and did not lowercase output, so older numbers
  (e.g. Punjabi CER 0.33, stock English WER 0.149) were wrong. The table above uses the fixed `normalize`.
- Trained on clean read speech only: music, noise and other languages give poor, hallucinated text.
- Keep >= 5 GB of disk free; training needs ~6 GB GPU memory and several GB RAM (close the browser first).
