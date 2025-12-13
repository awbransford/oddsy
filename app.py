import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import datetime
import base64

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
from oddsy_services.stats_service import get_top_level_stats
from ui.components.stats_bar import render_stats_bar

load_dotenv()

API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("KALSHI_API_PRIVATE_KEY")
PRIVATE_KEY_PEM = os.getenv("KALSHI_API_PRIVATE_KEY_PEM")
BASE_URL = "https://api.elections.kalshi.com"

def load_private_key_from_path(key_path: str):
    """Load the Kalshi private key from a file path (local dev)."""
    if not key_path:
        raise RuntimeError("KALSHI_API_PRIVATE_KEY (path) is not set in .env")

    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend(),
        )

def load_private_key_from_pem(pem_data: str):
    """Load the Kalshi private key directly from PEM text (for cloud)."""
    if not pem_data:
        raise RuntimeError("KALSHI_API_PRIVATE_KEY_PEM is not set")
    return serialization.load_pem_private_key(
        pem_data.encode("utf-8"),
        password=None,
        backend=default_backend(),
    )
    
# Decide which source to use: PEM string (cloud) or file path (local)
if PRIVATE_KEY_PEM:
    PRIVATE_KEY = load_private_key_from_pem(PRIVATE_KEY_PEM)
else:
    PRIVATE_KEY = load_private_key_from_path(PRIVATE_KEY_PATH)

def create_signature(private_key, timestamp: str, method: str, path: str) -> str:
    """Create the request signature according to Kalshi docs."""
    # Strip query parameters before signing
    path_without_query = path.split("?")[0]

    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")

    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

    return base64.b64encode(signature).decode("utf-8")

def kalshi_get(path: str):
    """
    Make an authenticated GET request to Kalshi.
    `path` must start with /trade-api/v2/...
    """
    if not API_KEY_ID:
        raise RuntimeError("KALSHI_API_KEY_ID is not set in .env")

    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))  # ms
    signature = create_signature(PRIVATE_KEY, timestamp, "GET", path)

    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

    url = BASE_URL + path
    response = requests.get(url, headers=headers)

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        # Try to get error body text for debugging
        body = ""
        try:
            body = response.text
        except Exception:
            pass

        msg = f"Kalshi GET error for {url}: {e} | Body: {body}"
        print(msg)
        # Optional: surface in the UI too
        st.error(msg)
        # Re-raise so we see the traceback while debugging
        raise

    return response.json()


def fetch_kalshi_markets(status: str = "open", max_pages: int = 5, page_limit: int = 500):
    # Fetch markets from Kalshi with pagination and safety guards.

    # - status: "open", "closed", etc. (depends on what you want)
    # - max_pages: safety cap so we do not loop forever
    # - page_limit: per-page limit

    # Returns a normalized DataFrame of all markets fetched.
    
    all_markets = []
    cursor = None
    pages_fetched = 0
    
    while True:
        pages_fetched += 1
        if pages_fetched > max_pages:
            break
        query = f"?limit={page_limit}"
        if status:
            query += f"&status={status}"
        if cursor:
            query += f"&cursor={cursor}"
    
        path = f"/trade-api/v2/markets{query}"
        
        try:
            data = kalshi_get(path)
        except Exception as e:
            print(f"Error fetching markets page {pages_fetched}: {e}")
            break
        
        markets = data.get("markets", data)
        if not markets:
            break
        
        all_markets.extend(markets)
        
        cursor = data.get("cursor")
        if not cursor:
            break
        
    if not all_markets:
        return pd.DataFrame()
        
    return pd.json_normalize(markets)

def compute_probability(row):
    # 1) Prefer last traded
    lt = row.get("last_traded_pct")
    if lt is not None and not np.isnan(lt) and lt > 0:
        return lt

    # 2) Then mid between bid/ask if both present
    bid = row.get("yes_bid_pct")
    ask = row.get("yes_ask_pct")
    if bid is not None and ask is not None:
        if not np.isnan(bid) and not np.isnan(ask) and (bid > 0 or ask > 0):
            return round((bid + ask) / 2.0, 1)

    # 3) Then whichever side is non-zero
    for v in [bid, ask]:
        if v is not None and not np.isnan(v) and v > 0:
            return v

    # 4) Fallback: unknown
    return None

def inspect_kalshi_categories(df):
    if "category" not in df.columns:
        st.warning("No 'category' column found in markets_df")
        return

    cats = sorted(df["category"].fillna("unknown").unique().tolist())
    st.write("Detected categories:", cats)

def fetch_kalshi_trades_last_week(max_pages: int = 5):
    
    # Fetch all trades from the last 7 days w/ safety guards:
    # - Limit the number of pages (max_pages)
    # - Handles errors so Streamlit doesn't run infinitely
    
    # Kalshi expects Unix timestamps (seconds, UTC)
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now = int(now_dt.timestamp())
    week_ago = now - 7 * 24 * 60 * 60

    all_trades = []
    cursor = None
    limit = 500
    
    pages_fetched = 0

    while True:
        pages_fetched += 1
        if pages_fetched > max_pages:
                break
            
        query = f"?limit={limit}&min_ts={week_ago}&max_ts={now}"
        if cursor:
            query += f"&cursor={cursor}"

        path = f"/trade-api/v2/markets/trades{query}"
        
        try:
            data = kalshi_get(path)
        except Exception as e:
            print(f"Error fetching trades page {pages_fetched}: {e}")
            break
        
        trades = data.get("trades", [])
        all_trades.extend(trades)

        cursor = data.get("cursor")
        if not cursor:  # when cursor is empty / null, no more pages
            break

    if not all_trades:
        return pd.DataFrame()

    return pd.json_normalize(all_trades)

st.set_page_config(page_title="Prediction Markets MVP", layout="wide")
st.title("Prediction Market Terminal (Kalshi - Public Endpoint MVP)")
st.write("Data from Kalshi elections API.")

if st.button("Refresh Data"):
    # You can choose status="open" or "" to get all
    markets_df = fetch_kalshi_markets(status="open", max_pages=5, page_limit=500)
    trades_df = fetch_kalshi_trades_last_week(max_pages=5)
    
    st.session_state["markets_df"] = markets_df
    st.session_state["trades_df"] = trades_df
    
    st.success("Fetched latest markets and trades!")

markets_df = st.session_state.get("markets_df")
trades_df = st.session_state.get("trades_df")

if markets_df is None:
    st.info("Click 'Refresh Data' to load markets.")
else:
    # ---- Debug counts (keep while iterating) ----
    st.write(
        "DEBUG: markets_df rows =", len(markets_df),
        "| trades_df rows =", 0 if trades_df is None else len(trades_df)
    )

    # ---- Top-level stats bar (already working) ----
    top_level_stats = get_top_level_stats(markets_df, trades_df)
    render_stats_bar(top_level_stats)
    st.markdown("---")

    # ---- Build df_display for cards ----
    df = markets_df.copy()

    # Only keep useful columns for now
    cols = [
        "title",
        "subtitle",
        "ticker",
        "event_ticker",
        "category",
        "market_type",
        "status",
        "close_time",
        "yes_bid_dollars",
        "yes_ask_dollars",
        "no_bid_dollars",
        "no_ask_dollars",
        "last_price_dollars",
        "volume",
        "volume_24h",
        "open_interest",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    
    def compute_implied_yes_prob(row):
        """
        Return implied YES probability in percent for binary markets.
        - Uses last_price_dollars when available
        - Falls back to midpoint of yes_bid_dollars / yes_ask_dollars
        - Returns None for non-binary markets or missing data
        """
        mtype = row.get("market_type")
        if mtype and str(mtype).lower() != "binary":
            return None

        def to_float(x):
            if x is None:
                return None
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        last = to_float(row.get("last_price_dollars"))
        if last is not None:
            # dollars → percent
            return round(last * 100.0, 1)

        yes_bid = to_float(row.get("yes_bid_dollars"))
        yes_ask = to_float(row.get("yes_ask_dollars"))

        candidates = [v for v in (yes_bid, yes_ask) if v is not None]
        if candidates:
            mid = sum(candidates) / len(candidates)
            return round(mid * 100.0, 1)

        return None

    # Add implied YES probability (in percent) on the *raw* dollar data
    df["implied_yes_prob"] = df.apply(compute_implied_yes_prob, axis=1)

    # Tag everything as Kalshi for now (Polymarket later)
    df["platform"] = "Kalshi"

    # Convert dollar odds → percentages
    df_display = df.copy()
    for col in [
        "yes_bid_dollars",
        "yes_ask_dollars",
        "no_bid_dollars",
        "no_ask_dollars",
        "last_price_dollars",
    ]:
        if col in df_display.columns:
            df_display[col] = (df_display[col].astype(float) * 100).round(1)

    df_display = df_display.rename(
        columns={
            "yes_bid_dollars": "yes_bid_pct",
            "yes_ask_dollars": "yes_ask_pct",
            "no_bid_dollars": "no_bid_pct",
            "no_ask_dollars": "no_ask_pct",
            "last_price_dollars": "last_traded_pct",
        }
    )
    df_display["implied_prob_pct"] = df_display.apply(compute_probability, axis=1)

    # ---- Sort by 24h volume (Top Markets by Volume) ----
    if "volume_24h" in df_display.columns:
        df_display = df_display.sort_values("volume_24h", ascending=False)
    elif "volume" in df_display.columns:
        df_display = df_display.sort_values("volume", ascending=False)

    # ---- Build event-level groups (group by event_ticker) ----
    if "event_ticker" not in df_display.columns:
        # Fallback: treat each market as its own event
        df_display["event_ticker"] = df_display.get("ticker")

    events = []
    for event_id, group in df_display.groupby("event_ticker"):
        # Sort markets in this event by implied YES probability (desc)
        group_sorted = group.sort_values("implied_yes_prob", ascending=False)

        first = group_sorted.iloc[0]

        event_title = first.get("title") or str(event_id)
        category = first.get("category")
        status = first.get("status")

        total_vol_24h = (
            group_sorted["volume_24h"].fillna(0).sum()
            if "volume_24h" in group_sorted.columns
            else 0
        )
        total_oi = (
            group_sorted["open_interest"].fillna(0).sum()
            if "open_interest" in group_sorted.columns
            else None
        )
        event_close = first.get("close_time")

        events.append(
            {
                "event_ticker": event_id,
                "title": event_title,
                "category": category,
                "status": status,
                "volume_24h": total_vol_24h,
                "open_interest": total_oi,
                "close_time": event_close,
                "markets": group_sorted,
            }
        )

    # Sort events by total 24h volume (Top Events by Volume)
    events_sorted = sorted(
        events,
        key=lambda e: e["volume_24h"] if e["volume_24h"] is not None else 0,
        reverse=True,
    )

    st.subheader("Top Events by Volume")
    st.caption(
        "Grouped by event; top 2 outcomes shown inline, remaining options in an expander."
    )
    st.write(f"Showing {len(events_sorted)} events")

    # Optional: raw markets table for debugging
    show_table = st.checkbox("Show raw markets table", value=False)

    if show_table:
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True)
    else:
        # ---- Event card grid layout ----
        n_cols = 2  # 2 event cards per row
        for row_start in range(0, len(events_sorted), n_cols):
            row_events = events_sorted[row_start : row_start + n_cols]
            cols_streamlit = st.columns(len(row_events))

            for offset, (col, event) in enumerate(zip(cols_streamlit, row_events)):
                rank = row_start + offset + 1
                markets = event["markets"]
                markets_sorted = markets.sort_values(
                    "implied_yes_prob", ascending=False
                )

                top_two = markets_sorted.head(2)
                rest = markets_sorted.iloc[2:]
                total_markets = len(markets_sorted)

                with col:
                    with st.container(border=True):
                        # Rank badge
                        st.markdown(
                            f"<div style='font-size: 0.85rem; "
                            f"background-color: #f5a623; color: white; "
                            f"display: inline-block; padding: 0.2rem 0.6rem; "
                            f"border-radius: 999px; font-weight: 600;'>"
                            f"{rank}</div>",
                            unsafe_allow_html=True,
                        )

                        # Header row: event title + platform pill
                        title = event.get("title", f"Event {event['event_ticker']}")
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; "
                            f"align-items:center; margin-top:0.5rem;'>"
                            f"<div style='font-weight:600; font-size:1rem;'>{title}</div>"
                            f"<div style='background-color:#1a73e8; color:white; "
                            f"padding:0.15rem 0.6rem; border-radius:999px; "
                            f"font-size:0.75rem; font-weight:500;'>Kalshi</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # Chips: category + status
                        category = event.get("category")
                        status = event.get("status")

                        chips_html = "<div style='margin-top:0.4rem; display:flex; gap:0.4rem;'>"
                        if category:
                            chips_html += (
                                "<div style='font-size:0.7rem; padding:0.1rem 0.4rem; "
                                "border-radius:999px; background-color:#f2f2f2;'>"
                                f"{category}</div>"
                            )
                        if status:
                            color = "#e6f4ea" if str(status).lower() == "open" else "#fce8e6"
                            text_color = "#137333" if str(status).lower() == "open" else "#c5221f"
                            chips_html += (
                                f"<div style='font-size:0.7rem; padding:0.1rem 0.4rem; "
                                f"border-radius:999px; background-color:{color}; "
                                f"color:{text_color};'>"
                                f"{status}</div>"
                            )
                        chips_html += "</div>"

                        st.markdown(chips_html, unsafe_allow_html=True)

                        st.markdown("---")

                        # Helper to choose a label per outcome row
                        def outcome_label(row):
                            """
                            Choose a short, outcome-specific label.
                            Priority:
                            1) yes_sub_title  (often candidate/party name)
                            2) subtitle
                            3) custom_strike.* fields
                            4) ticker
                            5) title
                            """
                            val = row.get("yes_sub_title")
                            if isinstance(val, str) and val.strip():
                                return val.strip()

                            val = row.get("subtitle")
                            if isinstance(val, str) and val.strip():
                                return val.strip()

                            # Look through any custom_strike.* keys
                            for key in row.index:
                                if str(key).startswith("custom_strike"):
                                    v = row.get(key)
                                    if isinstance(v, str) and v.strip():
                                        return v.strip()

                            val = row.get("ticker")
                            if isinstance(val, str) and val.strip():
                                return val.strip()

                            return row.get("title") or "Unknown"

                        # ---- Top 2 outcomes (by YES prob) ----
                        st.markdown(
                            "<div style='font-size:0.8rem; font-weight:600; "
                            "margin-bottom:0.3rem;'>Top outcomes (YES probability)"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        for _, m in top_two.iterrows():
                            label = outcome_label(m)
                            prob = m.get("implied_yes_prob", None)
                            prob_text = "n/a" if prob is None else f"{prob:.1f}%"

                            st.markdown(
                                "<div style='display:flex; justify-content:space-between; "
                                "align-items:center; padding:0.35rem 0.6rem; "
                                "border-radius:0.6rem; background-color:#f8fafc; "
                                "margin-bottom:0.3rem;'>"
                                f"<span style='font-size:0.85rem;'>{label}</span>"
                                f"<span style='font-weight:600;'>{prob_text}</span>"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                        # ---- Remaining outcomes in expander ----
                        if total_markets > 2:
                            with st.expander(f"View all {total_markets} options"):
                                pills_cols = st.columns(3)
                                for idx2, (_, m2) in enumerate(markets_sorted.iterrows()):
                                    label2 = outcome_label(m2)
                                    prob2 = m2.get("implied_yes_prob", None)
                                    prob_text2 = (
                                        "n/a" if prob2 is None else f"{prob2:.1f}%"
                                    )

                                    pill_col = pills_cols[idx2 % 3]
                                    with pill_col:
                                        st.markdown(
                                            "<div style='padding:0.25rem 0.5rem; "
                                            "border-radius:999px; background-color:#f1f5f9; "
                                            "margin-bottom:0.25rem; font-size:0.75rem; "
                                            "display:flex; justify-content:space-between;'>"
                                            f"<span>{label2}</span>"
                                            f"<span style='font-weight:600;'>{prob_text2}</span>"
                                            "</div>",
                                            unsafe_allow_html=True,
                                        )

                        # ---- Event-level stats row ----
                        vol_24h = event.get("volume_24h")
                        oi = event.get("open_interest")
                        close_time = event.get("close_time", "N/A")

                        bottom_html = (
                            "<div style='display:flex; justify-content:space-between; "
                            "margin-top:0.6rem; font-size:0.8rem;'>"
                        )

                        bottom_html += (
                            "<div>"
                            "<div style='color:#6b6b6b; text-transform:uppercase; "
                            "font-size:0.7rem;'>24h Volume</div>"
                            f"<div style='font-weight:600;'>{vol_24h}</div>"
                            "</div>"
                        )

                        bottom_html += (
                            "<div>"
                            "<div style='color:#6b6b6b; text-transform:uppercase; "
                            "font-size:0.7rem;'>Open Int.</div>"
                            f"<div style='font-weight:600;'>{oi}</div>"
                            "</div>"
                        )

                        bottom_html += (
                            "<div>"
                            "<div style='color:#6b6b6b; text-transform:uppercase; "
                            "font-size:0.7rem;'>Ends</div>"
                            f"<div style='font-weight:600;'>{close_time}</div>"
                            "</div>"
                        )

                        bottom_html += "</div>"

                        st.markdown(bottom_html, unsafe_allow_html=True)