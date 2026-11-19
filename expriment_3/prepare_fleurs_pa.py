"""One-off: convert FLEURS pa_in parquet (huge row groups) into small local FLAC files + a TSV manifest.
Reads ONE row group at a time (<=~800 MB RAM), writes ~0.5 GB total to ./data/pa/."""
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent / "data" / "pa"
for split in ["validation", "train"]:
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    local = hf_hub_download("google/fleurs", f"parquet-data/pa_in/{split}-00000-of-00001.parquet", repo_type="dataset")
    pf = pq.ParquetFile(local)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["id", "audio", "transcription"]).to_pylist()
        for r in t:
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            name = f"{r['id']}.flac"
            sf.write(d / name, wav, sr, format="FLAC")
            rows.append(f"{name}\t{r['transcription'].strip()}")
        del t
        print(split, "row group", g, "done", len(rows), flush=True)
    (OUT / f"{split}.tsv").write_text("\n".join(rows), encoding="utf-8")
print("finished")
