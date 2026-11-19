"""Experiment 2: classify images into animals, vehicles and plants with a CNN (PyTorch).

Data: 30 CIFAR-100 classes (10 per category), downloaded automatically.
The network learns the 30 fine classes; each is rolled up to its category for the
final animals / vehicles / plants accuracy.

Train:   python image_classification.py            (add --resume to continue from classifier.pt)
Predict: python image_classification.py --predict photo.jpg folder/ https://example.com/cat.png
Demo:    python image_classification.py --demo 10
"""
import argparse
import io
import json
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import pyarrow.parquet as pq
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageStat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CATEGORIES = {
    "animal": ["bear", "camel", "cattle", "dolphin", "elephant", "fox", "leopard", "lion", "rabbit", "wolf"],
    "vehicle": ["bicycle", "bus", "motorcycle", "pickup_truck", "streetcar", "tank", "tractor", "train", "lawn_mower", "rocket"],
    "plant": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree", "orchid", "poppy", "rose", "sunflower", "tulip"],
}
CAT_NAMES = list(CATEGORIES)
FINE_NAMES = [n for c in CAT_NAMES for n in CATEGORIES[c]]          # 30 labels
FINE_TO_CAT = torch.tensor([i // 10 for i in range(len(FINE_NAMES))])  # fine idx -> category idx
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CKPT = HERE / "classifier.pt"
HISTORY = HERE / "history.json"


MIRROR = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{}-00000-of-00001.parquet"


class Subset100(Dataset):
    """CIFAR-100 (Hugging Face parquet mirror) restricted to FINE_NAMES, relabelled 0..29.

    Images stay as compressed bytes and are decoded one at a time when the DataLoader requests them.
    """

    def __init__(self, root, train, transform):
        split = "train" if train else "test"
        path = Path(root) / f"{split}.parquet"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading CIFAR-100 {split} split...")
            urllib.request.urlretrieve(MIRROR.format(split), path)
        table = pq.read_table(path)
        names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["fine_label"]["names"]
        remap = {names.index(n): i for i, n in enumerate(FINE_NAMES)}
        labels = table.column("fine_label").to_pylist()
        images = table.column("img").to_pylist()
        keep = [i for i, y in enumerate(labels) if y in remap]
        self.x = [images[i]["bytes"] for i in keep]  # compressed PNG bytes; decoded lazily in image()
        self.y = [remap[labels[i]] for i in keep]
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def image(self, i):
        return Image.open(io.BytesIO(self.x[i])).convert("RGB")

    def __getitem__(self, i):
        return self.transform(self.image(i)), self.y[i]


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def make_model():
    return nn.Sequential(
        block(3, 64), block(64, 128), block(128, 256),   # 32 -> 4 spatial
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, len(FINE_NAMES)),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean loss, 30x30 confusion matrix; rows = true, cols = predicted)."""
    model.eval()
    cm = torch.zeros(30, 30, dtype=torch.long)
    total = n = 0
    for x, y in loader:
        out = model(x.to(device)).cpu()
        total += F.cross_entropy(out, y, reduction="sum").item()
        n += len(y)
        cm.index_put_((y, out.argmax(1)), torch.ones_like(y), accumulate=True)
    return total / n, cm


def metrics(cm):
    """Accuracy and F1 for the 30 classes and for the 3 categories (cm is 30x30)."""
    cm3 = cm.view(3, 10, 3, 10).sum((1, 3))
    def f1(m):
        tp = m.diag().float()
        return 2 * tp / (m.sum(0) + m.sum(1)).clamp(min=1)  # = 2PR/(P+R) per class
    f30, f3 = f1(cm), f1(cm3)
    return dict(acc30=cm.diag().sum().item() / cm.sum().item(), acc3=cm3.diag().sum().item() / cm3.sum().item(),
                f1_30=f30.mean().item(), f1_3=f3.mean().item()), f30, cm3


def report(model, test_dl, device):
    """Evaluate the best checkpoint and write report.html (loss, accuracy, F1, confusion matrices)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("plotly is required for the report: pip install plotly")
    model.load_state_dict(torch.load(CKPT, map_location=device))
    _, cm = evaluate(model, test_dl, device)
    m, f30, cm3 = metrics(cm)
    print(f"\nBest checkpoint {CKPT.name}: acc {m['acc30']:.1%} / F1 {m['f1_30']:.3f} (30 classes), "
          f"acc {m['acc3']:.1%} / F1 {m['f1_3']:.3f} (3 categories)")
    h = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    ep = [r["epoch"] for r in h]
    fig = make_subplots(
        rows=3, cols=3, row_heights=[0.25, 0.3, 0.45], vertical_spacing=0.1,
        subplot_titles=("Loss", "Accuracy", "Macro F1", "Confusion matrix: 3 categories", "F1 per class (best model)", "",
                        "Confusion matrix: 30 classes"),
        specs=[[{}, {}, {}], [{"colspan": 1}, {"colspan": 2}, None], [{"colspan": 3}, None, None]])
    for col, series in enumerate([[("train_loss", "train"), ("test_loss", "test")],
                                  [("acc30", "30 classes"), ("acc3", "3 categories")],
                                  [("f1_30", "30 classes"), ("f1_3", "3 categories")]], 1):
        for key, label in series:
            fig.add_trace(go.Scatter(x=ep, y=[r[key] for r in h], name=label, mode="lines+markers",
                                     legendgroup=label, showlegend=col != 3), 1, col)
    fig.add_trace(go.Heatmap(z=cm3.numpy(), x=CAT_NAMES, y=CAT_NAMES, text=cm3.numpy(), texttemplate="%{text}",
                             colorscale="Blues", showscale=False), 2, 1)
    fig.add_trace(go.Bar(x=[n.replace("_", " ") for n in FINE_NAMES], y=f30.numpy(), name="F1",
                         marker_color=["#1f77b4"] * 10 + ["#ff7f0e"] * 10 + ["#2ca02c"] * 10, showlegend=False), 2, 2)
    fig.add_trace(go.Heatmap(z=cm.numpy(), x=FINE_NAMES, y=FINE_NAMES, colorscale="Blues", showscale=True,
                             hovertemplate="true %{y}<br>predicted %{x}<br>count %{z}<extra></extra>"), 3, 1)
    fig.update_xaxes(title_text="predicted", row=2, col=1)
    fig.update_yaxes(title_text="true", autorange="reversed", row=2, col=1)
    fig.update_xaxes(title_text="predicted", row=3, col=1)
    fig.update_yaxes(title_text="true", autorange="reversed", row=3, col=1)
    fig.update_xaxes(title_text="epoch", row=1)
    fig.update_layout(height=1500, title=f"Animals / vehicles / plants classifier - test acc {m['acc3']:.1%} (3 cat), "
                      f"{m['acc30']:.1%} (30 cls), macro F1 {m['f1_3']:.3f} / {m['f1_30']:.3f}", template="plotly_white")
    fig.write_html(HERE / "report.html", include_plotlyjs=True)
    print("Wrote", HERE / "report.html")


def train(args, device):
    norm = T.Normalize(MEAN, STD)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    test_tf = T.Compose([T.ToTensor(), norm])
    train_dl = DataLoader(Subset100(args.data, True, train_tf), args.batch, shuffle=True, num_workers=4, drop_last=True)
    test_dl = DataLoader(Subset100(args.data, False, test_tf), 256, num_workers=4)

    model = make_model().to(device)
    best, history = 0.0, []
    if args.resume:
        model.load_state_dict(torch.load(CKPT, map_location=device))
        m, _, _ = metrics(evaluate(model, test_dl, device)[1])
        best = m["acc30"]
        if HISTORY.exists():
            history = json.loads(HISTORY.read_text())
        print(f"Resumed from {CKPT.name}: test acc (30 classes) {m['acc30']:.1%}  (3 categories) {m['acc3']:.1%}")
    lr = args.lr or (5e-4 if args.resume else 2e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    steps = args.epochs * len(train_dl)
    if args.resume:  # already converged: no warm-up ramp (it pushes the loss back up), just decay from a low rate
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        test_loss, cm = evaluate(model, test_dl, device)
        m = metrics(cm)[0]
        history.append(dict(epoch=len(history) + 1, train_loss=total / len(train_dl), test_loss=test_loss, **m))
        saved = ""
        if m["acc30"] > best:  # keep only the best checkpoint
            best, saved = m["acc30"], "  * saved"
            torch.save(model.state_dict(), CKPT)
        print(f"epoch {epoch:2d}/{args.epochs}  loss {total / len(train_dl):.3f}  test acc (30 classes) {m['acc30']:.1%}  "
              f"(3 categories) {m['acc3']:.1%}  F1 {m['f1_3']:.3f}{saved}")

    HISTORY.write_text(json.dumps(history))
    report(model, test_dl, device)


def fit_to_input(img, size=32):
    """Fit any image to the model's size x size input without distorting it.

    Fixes EXIF rotation, flattens transparency onto white, scales the whole image to fit
    (keeping its aspect ratio, high-quality downscale) and pads the rest with the mean colour.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(flat, img)
    img = img.convert("RGB")
    scale = size / max(img.size)
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new, Image.LANCZOS, reducing_gap=3.0)
    canvas = Image.new("RGB", (size, size), tuple(int(v) for v in ImageStat.Stat(img).mean))
    canvas.paste(img, ((size - new[0]) // 2, (size - new[1]) // 2))
    return canvas


def load_image(src):
    """Open a local file or an http(s) URL as an RGB image."""
    if str(src).startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=20).read())).convert("RGB")
    return Image.open(src).convert("RGB")


@torch.no_grad()
def predict(args, device):
    model = make_model().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    tf = T.Compose([T.Lambda(fit_to_input), T.ToTensor(), T.Normalize(MEAN, STD)])  # any size -> 32x32

    jobs = []  # (label shown, PIL image, true fine index or None)
    if args.demo:
        test = Subset100(args.data, False, tf)
        for i in torch.randperm(len(test))[: args.demo].tolist():
            jobs.append((f"test image #{i}", test.image(i), test.y[i]))
    for src in args.predict or []:
        files = sorted(f for f in Path(src).iterdir() if f.suffix.lower() in IMG_EXT) if Path(src).is_dir() else [src]
        for f in files:
            try:
                jobs.append((Path(str(f)).name if not str(f).startswith("http") else str(f), load_image(f), None))
            except Exception as e:
                print(f"{f}: could not load ({e})")

    for name, img, truth in jobs:
        probs = F.softmax(model(tf(img).unsqueeze(0).to(device)), 1)[0].cpu()
        cat_probs = probs.view(3, 10).sum(1)  # fine classes are grouped 10 per category
        top = probs.topk(3)
        guesses = ", ".join(f"{FINE_NAMES[i].replace('_', ' ')} {p:.0%}" for p, i in zip(top.values, top.indices))
        line = f"{name}: {CAT_NAMES[cat_probs.argmax()].upper()} ({cat_probs.max():.0%})  top-3: {guesses}"
        if truth is not None:
            line += f"  | truth: {CAT_NAMES[truth // 10]} / {FINE_NAMES[truth].replace('_', ' ')}"
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Animals / vehicles / plants image classifier")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=None, help="Peak learning rate (default 2e-3, or 5e-4 with --resume)")
    p.add_argument("--report", action="store_true", help="Rebuild report.html from classifier.pt and history.json")
    p.add_argument("--resume", action="store_true", help="Continue training from classifier.pt")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--data", default=str(HERE / "data"), help="Dataset download directory")
    p.add_argument("--predict", nargs="+", help="Image files, folders or URLs to classify with the saved model")
    p.add_argument("--demo", type=int, nargs="?", const=8, help="Predict N random test images (default 8)")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.report:
        test_dl = DataLoader(Subset100(args.data, False, T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])), 256, num_workers=4)
        report(make_model().to(dev), test_dl, dev)
    else:
        predict(args, dev) if (args.predict or args.demo) else train(args, dev)
