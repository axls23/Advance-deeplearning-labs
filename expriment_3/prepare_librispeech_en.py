"""One-off: stream a small LibriSpeech slice and store it as local FLAC + TSV (same layout as data_pa/)."""
import io
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

OUT = Path(__file__).resolve().parent / "data" / "en"
for split, hub_split, n in [("train", "train.100", 1000), ("validation", "validation", 100)]:
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openslr/librispeech_asr", "clean", split=hub_split, streaming=True).cast_column("audio", Audio(decode=False))
    rows = []
    for r in ds:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if sr != 16000 or len(wav) > 15 * 16000:
            continue
        name = f"{r['id']}.flac"
        sf.write(d / name, wav, sr, format="FLAC")
        rows.append(f"{name}\t{r['text'].lower()}")
        if len(rows) == n:
            break
    (OUT / f"{split}.tsv").write_text("\n".join(rows), encoding="utf-8")
    print(split, len(rows), flush=True)
print("finished")
