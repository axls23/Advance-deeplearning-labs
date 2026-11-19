"""Ex5: LSTM time-series prediction of stock closing prices.

python stock_lstm.py                      # AAPL, 5y, default settings
python stock_lstm.py --ticker MSFT --epochs 50 --future 30
python stock_lstm.py --csv my.csv         # CSV with a 'Close' column
If no data can be downloaded, a synthetic price series is used (and stated).
"""
import argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

ap = argparse.ArgumentParser()
ap.add_argument("--ticker", default="AAPL"); ap.add_argument("--period", default="5y")
ap.add_argument("--csv", default="stock_prices.csv" if __import__("os").path.exists("stock_prices.csv") else None, help="CSV with a Close column (default: stock_prices.csv if present)"); ap.add_argument("--window", type=int, default=60)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--future", type=int, default=30, help="number of steps to forecast ahead")
ap.add_argument("--unit", default="day", help="label for one time step: day, minute, second ...")
ap.add_argument("--synthetic", action="store_true", help="allow synthetic data if the download fails")
ap.add_argument("--split", type=float, default=0.2, help="fraction of the series (the LAST part) held out as the unseen test chunk")
ap.add_argument("--returns", action="store_true", help="model log-returns (percent change) instead of raw prices; prices are rebuilt from them")
ap.add_argument("--dropout", type=float, default=0.2, help="dropout between LSTM layers")
ap.add_argument("--weight-decay", type=float, default=0.0, help="L2 penalty (AdamW)")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if a.csv and (a.ticker != "AAPL" or a.period != "5y"):
    print("Warning: --csv given, so --ticker/--period are ignored.")
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. Data collection ----
def load():
    if a.csv:
        return pd.read_csv(a.csv)["Close"].dropna().values.astype("float32"), a.csv
    try:
        import yfinance as yf
        df = yf.download(a.ticker, period=a.period, auto_adjust=True, progress=False)
        if len(df) > a.window * 3:
            return df["Close"].values.reshape(-1).astype("float32"), a.ticker
    except Exception as e:
        print("Download failed:", e)
    if not a.synthetic:
        raise SystemExit("No data could be loaded. Fix --ticker/--csv, or pass --synthetic to use fake data.")
    t = np.arange(1500)
    y = 100 + 0.05 * t + 8 * np.sin(t / 40) + np.cumsum(np.random.randn(len(t)) * 0.8)
    return y.astype("float32"), "SYNTHETIC (no real data available)"
prices, source = load()
print(f"Data: {source}, {len(prices)} points")

# ---- 2. Preprocessing: chronological split, scale on train only, windowing ----
n_train = int(len(prices) * (1 - a.split))
print(f"Train: points 0-{n_train-1} ({n_train}) | Unseen test chunk: points {n_train}-{len(prices)-1} ({len(prices)-n_train})")
if a.returns:      # series[i] = log(price[i+1] / price[i]); scale with train-only statistics
    lr = np.diff(np.log(prices.astype("float64"))); off = 1
    mu, sd = lr[:n_train - off].mean(), lr[:n_train - off].std()
    scaled = ((lr - mu) / sd).astype("float32")[:, None]
    inv = lambda v: np.asarray(v).reshape(-1) * sd + mu
else:
    scaler = MinMaxScaler().fit(prices[:n_train, None]); off = 0
    scaled = scaler.transform(prices[:, None]).astype("float32")
    inv = lambda v: scaler.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
n_tr = n_train - off                           # index in `scaled` where the test chunk begins

def windows(x, w):
    X = np.stack([x[i:i + w] for i in range(len(x) - w)]); y = x[w:]
    return torch.tensor(X), torch.tensor(y)
Xtr, ytr = windows(scaled[:n_tr], a.window)
Xte, yte = windows(scaled[n_tr - a.window:], a.window)   # test windows may look back into train

# ---- 3. Model ----
class LSTM(nn.Module):
    def __init__(s, h, layers=2, drop=0.2):
        super().__init__()
        s.lstm = nn.LSTM(1, h, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
        s.drop = nn.Dropout(drop); s.fc = nn.Linear(h, 1)
    def forward(s, x):
        o, _ = s.lstm(x); return s.fc(s.drop(o[:, -1]))
model = LSTM(a.hidden, a.layers, a.dropout).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=a.weight_decay); loss_fn = nn.MSELoss()

# ---- 4. Training ----
nv = int(len(Xtr) * 0.1)                      # last 10% of training windows = validation (as in Keras version)
Xva, yva, Xfit, yfit = Xtr[-nv:], ytr[-nv:], Xtr[:-nv], ytr[:-nv]
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xfit, yfit), batch_size=32, shuffle=True)
tr_hist, va_hist = [], []
for ep in range(1, a.epochs + 1):
    model.train(); tot = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); l = loss_fn(model(xb), yb); l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += l.item() * len(xb)
    model.eval()
    with torch.no_grad(): vl = loss_fn(model(Xva.to(dev)), yva.to(dev)).item()
    tr_hist.append(tot / len(Xfit)); va_hist.append(vl)
    if ep % 5 == 0 or ep == 1: print(f"epoch {ep:3d}  train MSE {tr_hist[-1]:.5f}  val MSE {vl:.5f}  (scaled)")

# ---- 5. Evaluation (one-step-ahead on unseen test data) ----
model.eval()
with torch.no_grad(): pred = model(Xte.to(dev)).cpu().numpy()
if a.returns:      # one-step price = REAL previous price * exp(predicted return)
    p, t = prices[n_train - 1:-1] * np.exp(inv(pred)), prices[n_train:].astype("float64")
else:
    p, t = inv(pred), inv(yte.numpy())
rmse = float(np.sqrt(np.mean((p - t) ** 2))); mae = float(np.mean(np.abs(p - t)))
mape = float(np.mean(np.abs((p - t) / t)) * 100)
n_rmse = float(np.sqrt(np.mean((t[1:] - t[:-1]) ** 2))); n_mae = float(np.mean(np.abs(t[1:] - t[:-1])))
print("\nTest metrics (price units)")
print(f"{'':16}{'RMSE':>10}{'MAE':>10}")
print(f"{'LSTM':16}{rmse:10.3f}{mae:10.3f}   (MAPE {mape:.2f}%)")
print(f"{'Naive baseline':16}{n_rmse:10.3f}{n_mae:10.3f}   (next value = previous value)")
plt.figure(figsize=(8, 4)); plt.plot(tr_hist, label="Training loss"); plt.plot(va_hist, label="Validation loss")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE (standardized returns, log axis)" if a.returns else "MSE (scaled 0-1, log axis)")
plt.title("Training vs validation loss"); plt.legend(); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=120); plt.close(); print("Saved loss_curve.png"); print(f"Val loss: best {min(va_hist):.4f} at epoch {int(np.argmin(va_hist))+1}, final {va_hist[-1]:.4f}")

# ---- 5b. Unseen-chunk forecast: from the last training window only, predict the WHOLE test chunk
# recursively (it never sees a real test price), so this is true out-of-sample forecasting.
n_test = len(prices) - n_train
win0 = torch.tensor(scaled[n_tr - a.window:n_tr]).unsqueeze(0).to(dev); rec = []
with torch.no_grad():
    for _ in range(n_test):
        nxt = model(win0); rec.append(nxt.item()); win0 = torch.cat([win0[:, 1:], nxt.view(1, 1, 1)], 1)
rec = inv(np.array(rec))
if a.returns: rec = prices[n_train - 1] * np.exp(np.cumsum(rec))   # chain predicted returns from last train price
actual_chunk = prices[n_train:]
r_rmse = float(np.sqrt(np.mean((rec - actual_chunk) ** 2))); r_mae = float(np.mean(np.abs(rec - actual_chunk)))
flat = np.full(n_test, prices[n_train - 1]); f_rmse = float(np.sqrt(np.mean((flat - actual_chunk) ** 2))); f_mae = float(np.mean(np.abs(flat - actual_chunk)))
print(f"\nUnseen {n_test}-step chunk forecast (no test prices used)")
print(f"{'':22}{'RMSE':>10}{'MAE':>10}")
print(f"{'LSTM recursive':22}{r_rmse:10.3f}{r_mae:10.3f}")
print(f"{'Flat (last train val)':22}{f_rmse:10.3f}{f_mae:10.3f}")

# ---- 6. Future forecast (recursive; uncertainty grows with horizon) ----
win = torch.tensor(scaled[-a.window:]).unsqueeze(0).to(dev); fut = []
with torch.no_grad():
    for _ in range(a.future):
        nxt = model(win); fut.append(nxt.item())
        win = torch.cat([win[:, 1:], nxt.view(1, 1, 1)], 1)
fut = inv(np.array(fut)) if a.future > 0 else None
if a.returns and fut is not None: fut = prices[-1] * np.exp(np.cumsum(fut))
if a.future > 0: print(f"Next {a.future} {a.unit}(s) forecast: first {fut[0]:.2f} ... last {fut[-1]:.2f}")

# ---- 7. Plot ----
fig, ax = plt.subplots(2 if fut is not None else 1, 1, figsize=(11, 8 if fut is not None else 4.5), squeeze=False)
ax = ax[:, 0]
ax[0].plot(range(n_train), prices[:n_train], label="Train")
ax[0].plot(range(n_train, len(prices)), actual_chunk, label="Test (actual, unseen)")
ax[0].plot(range(n_train, len(prices)), p, label="One-step prediction")
ax[0].plot(range(n_train, len(prices)), rec, "--", label="Recursive forecast of whole chunk")
ax[0].axvline(n_train, color="k", ls=":"); ax[0].set_title(f"Train / unseen test split ({source})")
ax[0].set_xlabel(f"{a.unit}s"); ax[0].legend()
if fut is not None:
    hist = prices[-120:]
    ax[1].plot(range(len(hist)), hist, label="History")
    ax[1].plot(range(len(hist), len(hist) + a.future), fut, "--", label="Forecast")
    ax[1].set_title(f"{a.future}-{a.unit} forecast"); ax[1].set_xlabel(f"{a.unit}s"); ax[1].legend()
plt.tight_layout(); plt.savefig("prediction.png", dpi=120); print("Saved prediction.png")
