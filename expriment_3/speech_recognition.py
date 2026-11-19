"""Experiment 3: fine-tune Whisper-tiny for speech-to-text in English, Punjabi, or both (PyTorch + Hugging Face).

Model:  openai/whisper-tiny (39M params): log-Mel spectrogram -> audio encoder -> text decoder (token by token).
Data:   local FLAC + TSV built by prepare_librispeech_en.py (data/en) and prepare_fleurs_pa.py (data/pa).
Metric: WER (word error rate) and CER (character error rate); lower is better. The best epoch is saved, ranked by
        the sum of CER for Punjabi and WER for English.
Each clip is tokenized with its own language token (<|en|> / <|pa|>), so ONE model can transcribe both languages.

Train (default: Punjabi + English "mix" model -> models/whisper_tiny_mix):
    python speech_recognition.py
    python speech_recognition.py --langs en            # English only  -> models/whisper_tiny_ft
    python speech_recognition.py --langs pa            # Punjabi only  -> models/whisper_tiny_pa
    python speech_recognition.py --init models/whisper_tiny_mix --epochs 2 --lr 5e-5    # continue from a checkpoint
Evaluate a saved model:   python speech_recognition.py --eval-only --init mix
Transcribe (long audio OK): python speech_recognition.py --transcribe clip.wav --model mix --lang auto
(infer.py has the same transcription plus vocal separation for music.)
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from infer import MODELS, fix_paths, load_audio, transcribe     # inference helpers shared with infer.py

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE = "openai/whisper-tiny"
SR = 16_000                               # Whisper expects 16 kHz mono audio
MAX_SECONDS = 30                          # Whisper's window
MAX_TOKENS = 440                          # decoder limit is 448; Gurmukhi is byte-level BPE so sequences are long
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGS = {"en": ("english", DATA / "en"), "pa": ("punjabi", DATA / "pa")}
NAMES = {("en",): "ft", ("pa",): "pa", ("en", "pa"): "mix"}     # which languages -> output folder name
PRIMARY = {"pa": "cer"}                   # metric used to rank checkpoints per language (default: wer)


# ----------------------------------------------------------------------------- data
def manifest(lang, split, n=None):
    """[(audio path, text, lang)] from data/<lang>/<split>.tsv (first n)."""
    base = LANGS[lang][1]
    lines = [l for l in (base / f"{split}.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(base / split / l.split("\t", 1)[0], l.split("\t", 1)[1], lang) for l in lines[:n]]


class ClipDataset(Dataset):
    """Reads one FLAC at a time (low RAM) and tokenizes with that clip's own language prefix."""

    def __init__(self, items, processor):
        self.items, self.proc = items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        for k in range(len(self.items)):                       # skip clips that are too long
            path, text, lang = self.items[(i + k) % len(self.items)]
            self.proc.tokenizer.set_prefix_tokens(language=LANGS[lang][0], task="transcribe")
            wav, _ = sf.read(path, dtype="float32")             # data is already 16 kHz mono
            ids = self.proc.tokenizer(text).input_ids
            if len(wav) <= MAX_SECONDS * SR and len(ids) <= MAX_TOKENS:
                feats = self.proc.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features[0]
                return feats, torch.tensor(ids), text
        raise RuntimeError("no usable clips")


def collate(batch, pad_id):
    """Decoder input = ids[:-1] (teacher forcing), target = ids[1:]. Mask ONLY real padding with -100: in Whisper
    pad_id == eos_id (50257), so masking by value would also hide the true <|endoftext|> label and the model would
    never learn to stop (it then loops until the token limit)."""
    feats = torch.stack([b[0] for b in batch])
    ids = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([b[1][1:] for b in batch], batch_first=True, padding_value=-100)
    return feats, ids[:, :-1], labels


# ----------------------------------------------------------------------------- evaluation
def normalize(text):
    """Lowercase, drop punctuation, squeeze spaces - so errors measure words, not formatting.
    Keeps letters, combining marks (Gurmukhi vowel signs!), digits and apostrophes."""
    text = unicodedata.normalize("NFC", text.lower())
    text = "".join(c if unicodedata.category(c)[0] in "LMN" or c.isspace() or c == "'" else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def decode(model, processor, feats, lang, bs=8):
    model.eval()
    out = []
    for i in range(0, len(feats), bs):
        x = torch.stack(feats[i:i + bs]).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            ids = model.generate(x, language=lang, task="transcribe", max_new_tokens=200)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return out


def evaluate(model, processor, ev):
    """ev: {lang: [(feats, ids, text)]} -> {lang: {"wer": .., "cer": ..}}"""
    res = {}
    for lang, data in ev.items():
        hyp = [normalize(h) for h in decode(model, processor, [d[0] for d in data], lang)]
        ref = [normalize(d[2]) for d in data]
        res[lang] = {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}
    return res


def fmt(res):
    return " | ".join(f"{l} WER {r['wer']:.3f} CER {r['cer']:.3f}" for l, r in res.items())


def rank(res):
    """Lower is better: sum of each language's primary metric."""
    return sum(r[PRIMARY.get(l, "wer")] for l, r in res.items())


# ----------------------------------------------------------------------------- training
def load_model(src):
    src = str(MODELS.get(src, src)) if src else BASE
    return src, WhisperProcessor.from_pretrained(src), WhisperForConditionalGeneration.from_pretrained(src).to(DEVICE)


def train(args):
    random.seed(0), torch.manual_seed(0)
    langs = tuple(sorted(args.langs))
    name = NAMES[langs]
    ckpt = Path(args.out) if args.out else HERE / "models" / f"whisper_tiny_{name}"
    history_path = HERE / "results" / f"history_{name}.json"
    history_path.parent.mkdir(exist_ok=True)

    src, processor, model = load_model(args.init)
    print(f"initialising from: {src} | languages: {', '.join(langs)} | output: {ckpt}", flush=True)
    ev = {}
    for l in langs:
        items = manifest(l, "validation", args.eval_n)
        ds = ClipDataset(items, processor)
        ev[l] = [ds[i] for i in range(len(ds))]
    base = evaluate(model, processor, ev)
    print(f"Baseline  {fmt(base)}", flush=True)
    if args.eval_only:
        return

    items = []
    for l in langs:
        items += manifest(l, "train", args.pa_n if l == "pa" else args.en_n)
    print(f"train: {len(items)} clips", flush=True)

    if args.freeze_encoder:                                    # train only the decoder: faster, less over-fitting
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(items) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    pad_id = processor.tokenizer.pad_token_id
    loader = DataLoader(ClipDataset(items, processor), batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate(b, pad_id))

    flat = lambda r: {f"{l}_{m}": v for l, d in r.items() for m, v in d.items()}
    history = [{"epoch": 0, "train_loss": None, **flat(base)}]
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for feats, dec_in, labels in loader:
            feats, dec_in, labels = feats.to(DEVICE), dec_in.to(DEVICE), labels.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(input_features=feats, decoder_input_ids=dec_in).logits
            loss = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            total += loss.item()
            nb += 1
            if nb % 20 == 0:
                print(f"  epoch {epoch} step {nb} loss {total / nb:.4f}", flush=True)
        res = evaluate(model, processor, ev)
        history.append({"epoch": epoch, "train_loss": total / max(nb, 1), **flat(res)})
        history_path.write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch}/{args.epochs}  loss {total / max(nb, 1):.4f}  {fmt(res)}", flush=True)
        if rank(res) < best:                                   # keep only the best epoch; a crash loses little
            best = rank(res)
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
    print(f"saved best epoch to {ckpt}")


# ----------------------------------------------------------------------------- inference
def predict(paths, model_name, lang):
    src, processor, model = load_model(model_name or "mix")
    model.eval()
    print(f"model: {src} | language: {lang}")
    for p in fix_paths(paths):
        wav = load_audio(p)
        text, used = transcribe(model, processor, wav, lang)   # handles long audio by chunking
        print(f"{p} ({len(wav) / SR:.0f}s, {used}): {text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcribe", nargs="+", metavar="AUDIO", help="transcribe these audio files instead of training")
    ap.add_argument("--model", default=None, metavar="NAME_OR_PATH",
                    help="model for --transcribe: alias (%s), a folder or a Hugging Face id (default: mix)" % ", ".join(MODELS))
    ap.add_argument("--lang", default="auto", help="--transcribe language: pa, en, any Whisper code, or auto (pa vs en)")
    ap.add_argument("--langs", nargs="+", choices=list(LANGS), default=["en", "pa"], help="languages to train on")
    ap.add_argument("--init", default=None, metavar="NAME_OR_PATH", help="start from this checkpoint (default: base whisper-tiny)")
    ap.add_argument("--out", default=None, help="where to save the model (default: models/whisper_tiny_<ft|pa|mix>)")
    ap.add_argument("--eval-only", action="store_true", help="just score --init (default mix) on the validation clips")
    ap.add_argument("--pa-n", type=int, default=1923, help="Punjabi training clips")
    ap.add_argument("--en-n", type=int, default=1000, help="English training clips")
    ap.add_argument("--eval-n", type=int, default=100, help="validation clips per language")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    a = ap.parse_args()
    if a.eval_only and not a.init:
        a.init = "mix"
    predict(a.transcribe, a.model, a.lang) if a.transcribe else train(a)
