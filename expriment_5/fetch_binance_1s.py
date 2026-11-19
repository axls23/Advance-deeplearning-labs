"""Download 1-second candles from Binance public market data -> prices_1s.csv (Date, Close).
python fetch_binance_1s.py [SYMBOL] [N_SECONDS]   e.g. BTCUSDT 10000"""
import sys, time, requests, pandas as pd
sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
hosts = ["https://data-api.binance.vision", "https://api.binance.com"]
rows, end = [], None
for host in hosts:
    try:
        rows, end = [], None
        while len(rows) < n:
            p = {"symbol": sym, "interval": "1s", "limit": 1000}
            if end: p["endTime"] = end
            r = requests.get(f"{host}/api/v3/klines", params=p, timeout=15); r.raise_for_status()
            k = r.json()
            if not k: break
            rows = k + rows; end = k[0][0] - 1; time.sleep(0.1)
        break
    except Exception as e:
        print(host, "failed:", e)
if not rows: sys.exit("No data downloaded")
df = pd.DataFrame({"Date": pd.to_datetime([r[0] for r in rows], unit="ms"),
                   "Close": [float(r[4]) for r in rows]}).tail(n)
df.to_csv("prices_1s.csv", index=False)
print(f"Saved prices_1s.csv: {len(df)} rows, {df.Date.iloc[0]} -> {df.Date.iloc[-1]} UTC")
