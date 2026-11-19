# Ex 5: Time Series Prediction with LSTM

**Aim:** Design and implement an LSTM network in Python to predict future stock prices from historical data.

## Procedure
1. Collect daily closing prices (yfinance, or a CSV with a `Close` column).
2. Split chronologically 80/20 (no shuffling, to avoid look-ahead leakage).
3. Scale with MinMaxScaler fitted on the training part only.
4. Build sliding windows (60 past days -> next day).
5. Define a 2-layer LSTM + linear output; train with Adam, MSE loss.
6. Evaluate on the test set with RMSE, MAE, MAPE and compare to a naive "yesterday" baseline.
7. Forecast the next N days recursively; plot results to `prediction.png`.

## Run
    pip install -r requirements.txt
    python get_data.py AAPL      # creates stock_prices.csv
    python stock_lstm.py         # Keras version (matches the lab procedure)
    python stock_lstm_torch.py   # PyTorch alternative if TensorFlow won't install

## Notes
Stock prices are close to a random walk; the LSTM often only matches the naive baseline.
Multi-day forecasts compound error, so treat them as illustrative, not financial advice.
