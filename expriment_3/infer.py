"""Transcribe audio with a Whisper model (default: the fine-tuned Punjabi + English one).

    python infer.py clip.mp3                      # language auto-detected (Punjabi vs English)
    python infer.py clip.wav --lang pa            # force a language
    python infer.py --live                        # microphone: prints each phrase when you pause (Ctrl+C to stop)
    python infer.py song.mp3 --separate           # music: isolate the vocals (Demucs) first
    python infer.py clip.wav --model openai/whisper-small

--model: alias (mix, base, english, punjabi), a local folder, or a Hugging Face id. Quote paths that contain spaces.
Long audio is split into <=30 s chunks, cut at quiet moments, and the pieces are joined.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

HERE = Path(__file__).resolve().parent
SR = 16_000
CHUNK = 30 * SR                                   # Whisper's window
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {
    "mix": HERE / "models" / "whisper_tiny_mix",       # Punjabi + English fine-tune (default)
    "english": HERE / "models" / "whisper_tiny_ft",
    "punjabi": HERE / "models" / "whisper_tiny_pa",    # trained before the stop-token fix; prefer mix
    "base": "openai/whisper-tiny",
}


def load_audio(path, sr=SR, mono=True):
    """-> float32 waveform resampled to `sr`: shape (n,) if mono, else stereo (2, n)."""
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.T                                                    # (channels, samples)
    if file_sr != sr:
        g = math.gcd(file_sr, sr)
        wav = resample_poly(wav, sr // g, file_sr // g, axis=1).astype("float32")
    if mono:
        return wav.mean(0)
    return np.repeat(wav, 2, axis=0) if wav.shape[0] == 1 else wav[:2]


def separate_vocals(path):
    """Isolate singing/speech from music with Demucs (htdemucs; ~80 MB download on first use). -> mono, 16 kHz."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dm = get_model("htdemucs").to(DEVICE).eval()
    dm_sr, vocal_idx = dm.samplerate, dm.sources.index("vocals")
    x = torch.from_numpy(load_audio(path, dm_sr, mono=False))
    mean, std = x.mean(0).mean(), x.mean(0).std() + 1e-8             # Demucs' own input normalisation
    with torch.no_grad():
        out = apply_model(dm, ((x - mean) / std)[None].to(DEVICE), split=True, overlap=0.25,
                          progress=False, device=DEVICE)[0]
    vocals = (out[vocal_idx].cpu() * std + mean).mean(0).numpy()
    del dm, out
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    g = math.gcd(dm_sr, SR)
    return resample_poly(vocals, SR // g, dm_sr // g).astype("float32")


def split_chunks(wav):
    """Split into <=30 s pieces, cutting at the quietest 50 ms in the last 4 s of each window (avoids cutting words)."""
    chunks, start, hop = [], 0, SR // 20
    while len(wav) - start > CHUNK:
        lo = start + CHUNK - 4 * SR
        seg = np.abs(wav[lo:start + CHUNK])
        energy = seg[:len(seg) // hop * hop].reshape(-1, hop).mean(1)
        cut = lo + int(energy.argmin()) * hop + hop // 2
        chunks.append(wav[start:cut])
        start = cut
    return chunks + [wav[start:]]


@torch.no_grad()
def detect_language(model, processor, feats):
    """Punjabi or English? Compare the decoder's first-step scores for the two language tokens."""
    tok = processor.tokenizer
    sot = torch.tensor([[tok.convert_tokens_to_ids("<|startoftranscript|>")]], device=DEVICE)
    logits = model(input_features=feats[:1].to(DEVICE), decoder_input_ids=sot).logits[0, -1]
    return max(("pa", "en"), key=lambda l: logits[tok.convert_tokens_to_ids(f"<|{l}|>")].item())


@torch.no_grad()
def transcribe(model, processor, wav, lang="auto", guard=False):
    """-> (text, language used). Handles audio of any length.
    guard=True blocks repeated phrases (helps if the model loops on music) but HURTS Punjabi accuracy
    (CER 0.33 -> 0.52 on validation), so it is off by default."""
    extra = dict(no_repeat_ngram_size=4, repetition_penalty=1.1) if guard else {}
    feats = processor.feature_extractor(split_chunks(wav), sampling_rate=SR, return_tensors="pt").input_features
    if lang == "auto":
        lang = detect_language(model, processor, feats)
    texts = []
    for i in range(0, len(feats), 4):
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            ids = model.generate(feats[i:i + 4].to(DEVICE), language=lang, task="transcribe", **extra)
        texts += [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]
    return " ".join(t for t in texts if t), lang


def mic_blocks():
    """Yield 100 ms blocks of 16 kHz mono microphone audio until interrupted (needs `sounddevice`)."""
    import queue

    import sounddevice as sd
    q = queue.Queue()
    with sd.InputStream(samplerate=SR, channels=1, blocksize=SR // 10, dtype="float32",
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        while True:
            yield q.get()


def phrases(blocks, pause=0.8, max_seconds=25):
    """Group 100 ms blocks into phrases. The first second sets the room-noise level; a phrase starts when sound rises
    well above it and ends after `pause` seconds of quiet (or `max_seconds`). Yields one waveform per phrase."""
    from collections import deque
    rms = lambda b: float(np.sqrt((b ** 2).mean()))
    blocks = iter(blocks)
    threshold = max(0.01, 3 * np.mean([rms(next(blocks)) for _ in range(10)]))
    preroll, seg, loud, quiet = deque(maxlen=3), [], 0, 0          # preroll keeps the soft start of a word
    for b in blocks:
        if not seg:
            if rms(b) > threshold:
                seg, loud, quiet = [*preroll, b], 1, 0
            else:
                preroll.append(b)
            continue
        seg.append(b)
        loud, quiet = (loud + 1, 0) if rms(b) > threshold else (loud, quiet + 1)
        if quiet >= pause * 10 or len(seg) >= max_seconds * 10:
            if loud >= 4:                                          # ignore clicks and bumps
                yield np.concatenate(seg)
            seg, loud, quiet = [], 0, 0
            preroll.clear()


def live(model, processor, lang, guard):
    """Listen to the microphone and print each phrase as soon as you pause. Ctrl+C to stop."""
    print("Stay quiet for 1 second, then speak. Pause between phrases to see the text. Ctrl+C to stop.", flush=True)
    try:
        for wav in phrases(mic_blocks()):
            text, used = transcribe(model, processor, wav, lang, guard)
            print(f"[{used}] {text}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped")


def fix_paths(args):
    """Re-join an unquoted path that the shell split at spaces ('/a/My' 'Folder/x.wav' -> '/a/My Folder/x.wav')."""
    out, i = [], 0
    while i < len(args):
        for j in range(len(args), i, -1):                  # longest run of args that forms an existing file
            if Path(" ".join(args[i:j])).is_file():
                out.append(" ".join(args[i:j]))
                i = j
                break
        else:
            raise SystemExit(f"file not found: {args[i]!r}  (quote paths that contain spaces)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="*", help="audio files (wav, flac, mp3, ogg ...)")
    ap.add_argument("--live", action="store_true", help="transcribe the microphone live, phrase by phrase")
    ap.add_argument("--model", default="mix", metavar="NAME_OR_PATH")
    ap.add_argument("--lang", default="auto", help="pa, en, any Whisper language code, or 'auto' (pa vs en)")
    ap.add_argument("--separate", action="store_true", help="isolate the vocals with Demucs first (music, noisy audio)")
    ap.add_argument("--guard", action="store_true", help="block repeated phrases (use only if output loops, e.g. on music)")
    a = ap.parse_args()
    if not a.audio and not a.live:
        ap.error("give audio files, or use --live for the microphone")

    src = str(MODELS.get(a.model, a.model))
    processor = WhisperProcessor.from_pretrained(src)
    model = WhisperForConditionalGeneration.from_pretrained(src).to(DEVICE).eval()
    if a.live:
        return live(model, processor, a.lang, a.guard)
    for path in fix_paths(a.audio):
        wav = separate_vocals(path) if a.separate else load_audio(path)
        text, lang = transcribe(model, processor, wav, a.lang, a.guard)
        print(f"{path} ({len(wav) / SR:.0f}s, {lang}): {text}")


if __name__ == "__main__":
    main()
