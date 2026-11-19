"""Ex5: LSTM stock price prediction (TensorFlow/Keras). Needs stock_prices.csv with a 'Close' column."""
# 1. Import libraries
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="stock_prices.csv", help="CSV with a 'Close' column")
ap.add_argument("--ticker", help="download this ticker from Yahoo instead of using --csv (e.g. AAPL)")
ap.add_argument("--period", default="5y")
ap.add_argument("--split", type=float, default=0.2, help="fraction (the LAST part) held out as the unseen test chunk")
ap.add_argument("--window", type=int, default=60); ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--future", type=int, default=0, help="steps to forecast ahead (0 = skip)")
ap.add_argument("--unit", default="day", help="label for one time step: day, second ...")
a = ap.parse_args()
WINDOW, EPOCHS = a.window, a.epochs

# 2. Load data
if a.ticker:
    import yfinance as yf
    df = yf.download(a.ticker, period=a.period, auto_adjust=True, progress=False)
    if df.empty: raise SystemExit(f"No data for ticker {a.ticker!r}")
    prices = df["Close"].values.reshape(-1, 1).astype("float32")
else:
    prices = pd.read_csv(a.csv)["Close"].dropna().values.reshape(-1, 1)
print(f"Data: {a.ticker or a.csv}, {len(prices)} points")

# 3. Normalize (fit on the training part only to avoid leakage)
split = int(len(prices) * (1 - a.split))                          # 6. chronological split (default 80/20)
print(f"Train: points 0-{split-1} ({split}) | Unseen test chunk: points {split}-{len(prices)-1} ({len(prices)-split})")
scaler = MinMaxScaler().fit(prices[:split])
scaled = scaler.transform(prices)

# 4. Sliding-window sequences
def make_seq(data):
    X = np.array([data[i - WINDOW:i, 0] for i in range(WINDOW, len(data))])
    return X, data[WINDOW:, 0]
X_train, y_train = make_seq(scaled[:split])
X_test, y_test = make_seq(scaled[split - WINDOW:])               # test windows may look back into train

# 5. Reshape to [samples, timesteps, features]
X_train = X_train.reshape(-1, WINDOW, 1); X_test = X_test.reshape(-1, WINDOW, 1)

# 7. Build and compile
model = Sequential([Input((WINDOW, 1)), LSTM(50), Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.summary()

# 8. Train
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.1, verbose=2)

# 9. Predict and inverse-transform
pred = scaler.inverse_transform(model.predict(X_test))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 11. Metrics (+ naive "yesterday" baseline for context)
from sklearn.metrics import mean_absolute_error
rmse = np.sqrt(mean_squared_error(actual, pred)); mae = mean_absolute_error(actual, pred)
n_rmse = np.sqrt(mean_squared_error(actual[1:], actual[:-1])); n_mae = mean_absolute_error(actual[1:], actual[:-1])
print("\nTest metrics (price units)")
print(f"{'':16}{'RMSE':>10}{'MAE':>10}")
print(f"{'LSTM':16}{rmse:10.3f}{mae:10.3f}")
print(f"{'Naive baseline':16}{n_rmse:10.3f}{n_mae:10.3f}   (next value = previous value)")

# Loss curve: training vs validation loss per epoch
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE (scaled 0-1, log axis)")
plt.title("Training vs validation loss"); plt.legend(); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=120); plt.close(); print("Saved loss_curve.png")

# 11b. Unseen-chunk forecast from the last training window only (no real test prices used)
n_test = len(prices) - split
w0 = scaled[split - WINDOW:split, 0].reshape(1, WINDOW, 1); rec = []
for _ in range(n_test):
    nx = model.predict(w0, verbose=0)[0, 0]; rec.append(nx); w0 = np.concatenate([w0[:, 1:, :], [[[nx]]]], axis=1)
rec = scaler.inverse_transform(np.array(rec).reshape(-1, 1)).ravel(); chunk = prices[split:, 0]
flat = np.full(n_test, prices[split - 1, 0])
print(f"\nUnseen {n_test}-step chunk forecast (no test prices used)")
print(f"{'':22}{'RMSE':>10}{'MAE':>10}")
print(f"{'LSTM recursive':22}{np.sqrt(np.mean((rec-chunk)**2)):10.3f}{np.mean(np.abs(rec-chunk)):10.3f}")
print(f"{'Flat (last train val)':22}{np.sqrt(np.mean((flat-chunk)**2)):10.3f}{np.mean(np.abs(flat-chunk)):10.3f}")

# 12. Optional recursive forecast
fut = None
if a.future > 0:
    win = scaled[-WINDOW:, 0].reshape(1, WINDOW, 1); out = []
    for _ in range(a.future):
        nxt = model.predict(win, verbose=0)[0, 0]; out.append(nxt)
        win = np.concatenate([win[:, 1:, :], [[[nxt]]]], axis=1)
    fut = scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()
    print(f"Next {a.future} {a.unit}(s) forecast: first {fut[0]:.2f} ... last {fut[-1]:.2f}")

# 10. Plot
fig, ax = plt.subplots(2 if fut is not None else 1, 1, figsize=(11, 8 if fut is not None else 5), squeeze=False)
ax = ax[:, 0]
ax[0].plot(range(split), prices[:split, 0], label="Train")
ax[0].plot(range(split, len(prices)), chunk, label="Test (actual, unseen)")
ax[0].plot(range(split, len(prices)), pred.ravel(), label="One-step prediction")
ax[0].plot(range(split, len(prices)), rec, "--", label="Recursive forecast of whole chunk")
ax[0].axvline(split, color="k", ls=":"); ax[0].set_title("Train / unseen test split (LSTM)")
ax[0].set_xlabel(f"{a.unit}s"); ax[0].set_ylabel("Price"); ax[0].legend()
if fut is not None:
    h = prices[-120:, 0]
    ax[1].plot(range(len(h)), h, label="History"); ax[1].plot(range(len(h), len(h) + a.future), fut, "--", label="Forecast")
    ax[1].set_title(f"{a.future}-{a.unit} forecast"); ax[1].set_xlabel(f"{a.unit}s"); ax[1].legend()
plt.tight_layout(); plt.savefig("prediction.png", dpi=120); print("Saved prediction.png")
