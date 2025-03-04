from flask import Flask, jsonify, render_template
import os
import pickle
import datetime
import requests
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html

# -------------------- Flask API --------------------

app = Flask(__name__)

# Load trained ML model
MODEL_PATH = "model.pkl"

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model file {MODEL_PATH} not found. Please train the model first.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# Load symbols list from CSV (for UI dropdown selection)
SYMBOLS_CSV = "stocks.csv"

def load_symbol_list():
    if not os.path.exists(SYMBOLS_CSV):
        raise FileNotFoundError(f"{SYMBOLS_CSV} not found.")
    return pd.read_csv(SYMBOLS_CSV)

# Mock functions for retrieving real-time sentiment & market data
def get_today_sentiment(symbol):
    """
    Mock function to return a simulated sentiment score.
    Replace with API calls to Google NLP, IBM Watson, etc.
    """
    return np.random.uniform(-1, 1)  # Random sentiment score

def get_latest_closing_price(symbol):
    """
    Mock function to return a simulated latest closing price.
    Replace with API calls to Alpha Vantage, Finnhub, etc.
    """
    return np.random.uniform(50, 500)  # Random price

# Prediction logic
def predict_next_day_price(symbol, model):
    """
    1) Retrieve today's closing price (or last close),
    2) Retrieve today's sentiment score,
    3) Construct feature vector,
    4) Predict tomorrow's closing price.
    """
    close_price_today = get_latest_closing_price(symbol)
    sentiment_score_today = get_today_sentiment(symbol)
    volume_today = np.random.randint(100000, 5000000)  # Simulated volume

    # Prepare feature vector
    X_input = np.array([[close_price_today, sentiment_score_today, volume_today]])
    predicted_price = model.predict(X_input)[0]

    return {
        "symbol": symbol,
        "today_date": datetime.date.today().strftime('%Y-%m-%d'),
        "today_close": close_price_today,
        "today_sentiment": sentiment_score_today,
        "today_volume": volume_today,
        "predicted_tomorrow_close": predicted_price
    }

@app.route("/predict", methods=["GET"])
def predict():
    """
    API endpoint that returns next-day stock price predictions.
    """
    model = load_trained_model()
    symbol_df = load_symbol_list()
    
    results = []
    for _, row in symbol_df.iterrows():
        symbol = row['stock_symbol']
        result = predict_next_day_price(symbol, model)
        results.append(result)
    
    return jsonify(results)

# -------------------- Dash Dashboard --------------------

# Initialize Dash app for visualization
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Layout for the dashboard
symbol_options = [{"label": symbol, "value": symbol} for symbol in load_symbol_list()['stock_symbol']]

dash_app.layout = html.Div([
    html.H1("Stock & Crypto Sentiment Prediction Dashboard", style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id="symbol-dropdown",
        options=symbol_options,
        value=symbol_options[0]['value'],  # Default selection
        placeholder="Select a stock or cryptocurrency"
    ),

    html.Div(id="prediction-output"),
    
    dcc.Graph(id="price-chart"),
])

# Callback function to update predictions & chart
@dash_app.callback(
    [dash.dependencies.Output("prediction-output", "children"),
     dash.dependencies.Output("price-chart", "figure")],
    [dash.dependencies.Input("symbol-dropdown", "value")]
)
def update_prediction(symbol):
    """
    Updates prediction result and visualization for selected symbol.
    """
    model = load_trained_model()
    result = predict_next_day_price(symbol, model)

    # Format output text
    output_text = [
        html.H3(f"Predicted Closing Price for {symbol}: ${result['predicted_tomorrow_close']:.2f}"),
        html.P(f"Today's Closing Price: ${result['today_close']:.2f}"),
        html.P(f"Sentiment Score: {result['today_sentiment']:.2f}"),
        html.P(f"Trading Volume: {result['today_volume']:,}")
    ]

    # Create a mock historical data chart (Replace with real data)
    days = [f"Day {-i}" for i in range(5, 0, -1)] + ["Today", "Predicted Next Day"]
    prices = list(np.random.uniform(50, 500, size=5)) + [result["today_close"], result["predicted_tomorrow_close"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=prices, mode="lines+markers", name=symbol))
    fig.update_layout(title=f"{symbol} Price Trend", xaxis_title="Days", yaxis_title="Closing Price")

    return output_text, fig

# -------------------- Run Server --------------------

if __name__ == "__main__":
    app.run(port=5000, debug=True)
