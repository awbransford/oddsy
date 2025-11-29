# Oddsy — Prediction Market Terminal (MVP)

Oddsy is an early-stage prediction market terminal that aggregates markets
from platforms like Kalshi and Polymarket. This MVP focuses on:

- Fetching live Kalshi markets
- Cleaning and normalizing the data
- Converting price → probability
- Filtering by league (NFL, NBA)
- Status filtering
- Min-volume filtering
- Sort by volume, last traded %, or close time
- Card-based Streamlit UI
- Platform tagging for future multi-exchange comparison

## Running the app

streamlit run app.py

## Roadmap

- Add authenticated Kalshi API endpoints
- Integrate Polymarket API
- Create unified market schema
- Compare odds across exchanges
- Add arbitrage surface detection
- Search, alerts, and watchlists
