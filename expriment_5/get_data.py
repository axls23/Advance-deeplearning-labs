"""Create stock_prices.csv (Date, Close) from Yahoo Finance.  python get_data.py [TICKER]"""
import sys, yfinance as yf
t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
df = yf.download(t, period="5y", auto_adjust=True, progress=False)
df[["Close"]].reset_index().to_csv("stock_prices.csv", index=False)
print("Saved stock_prices.csv", len(df), "rows")
