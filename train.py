# train.py

import os
import pandas as pd
import numpy as np
import pickle
from build import build_model

# -------------------------------------------------------------------
# STEP 1: LOAD YOUR DATA
# -------------------------------------------------------------------

def load_historical_data(csv_path="historical_data.csv"):
    """
    Loads historical price and sentiment data from CSV (example).
    In production, you'd fetch from an API (Alpha Vantage, etc.) 
    and from sentiment APIs, then merge the results.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")

    df = pd.read_csv(csv_path)
    # Example expected columns:
    # date, symbol, close_price, sentiment_score, volume, ...
    # This is your mock data file. 
    return df

def engineer_features(df):
    """
    Create features (X) and targets (y) from the raw data.
    Suppose we predict next-day close price. We'll shift 
    close_price by -1 day to align with today's sentiment.
    """
    df = df.sort_values(['symbol','date']).reset_index(drop=True)

    # SHIFT the close_price to get next-day close as target
    df['target'] = df.groupby('symbol')['close_price'].shift(-1)

    # Drop the last row per symbol which won't have a next-day close
    df.dropna(subset=['target'], inplace=True)

    # Example features: [today_close, sentiment_score, volume]
    # You can add more sophisticated features (moving averages, etc.)
    X = df[['close_price', 'sentiment_score', 'volume']].values
    y = df['target'].values

    return X, y, df

# -------------------------------------------------------------------
# STEP 2: TRAIN THE MODEL
# -------------------------------------------------------------------

def train_model(csv_path="historical_data.csv", model_out="model.pkl"):
    # Load data
    df = load_historical_data(csv_path)
    # Create features/targets
    X, y, df_processed = engineer_features(df)

    # Split data into train/test for a simple hold-out approach
    # (Time-series splits are recommended in real solutions)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Build model pipeline
    model = build_model(n_estimators=100, max_depth=5)

    # Fit on training data
    model.fit(X_train, y_train)

    # Evaluate on test data (quick check)
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test)**2)
    print(f"Test MSE: {mse:.4f}")

    # Save the trained model to disk
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to: {model_out}")

if __name__ == "__main__":
    train_model(
        csv_path="historical_data.csv",
        model_out="model.pkl"
    )
