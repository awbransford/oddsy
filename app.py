import streamlit as st
import pandas as pd
import requests
import os
import datetime
import base64

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
from services.stats_service import get_top_level_stats
from ui.components.stats_bar import render_stats_bar

load_dotenv()

API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("KALSHI_API_PRIVATE_KEY")
BASE_URL = "https://demo-api.kalshi.co"

def load_private_key(key_path: str):
    """Load the Kalshi private key from file."""
    if not key_path:
        raise RuntimeError("KALSHI_PRIVATE_KEY_PATH is not set in .env")

    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend(),
        )


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


PRIVATE_KEY = load_private_key(PRIVATE_KEY_PATH)


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

    response = requests.get(BASE_URL + path, headers=headers)
    response.raise_for_status()
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
    st.info("Click 'Refresh data' to load markets.")
else:
    # ---- Top-level stats bar ----
    top_level_stats = get_top_level_stats(markets_df, trades_df)
    render_stats_bar(top_level_stats)
    st.markdown("---")

    # ---- Base dataframe for filters / display ----
    df = markets_df.copy()
    cols = [
        "title",
        "subtitle",
        "ticker",
        "event_ticker",
        "category",
        "close_time",
        "status",
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

    # ---- Platform tagging (Kalshi only for now) ----
    df["platform"] = "kalshi"

    # Filter out MULTIGAME noise
    if "ticker" in df.columns:
        df = df[~df["ticker"].str.contains("SPORTSMULTIGAME", case=False, na=False)]

    # ---- Category normalization / filters (demo-safe) ----
    # Only show category filter if there is at least one non-null category
    st.write(
        "DEBUG category check:",
        "has_category_col =", "category" in df.columns,
        "| any_non_null =", df["category"].notna().any(),
    )
    
    if (
        "category" in df.columns
        and df["category"].notna().any()
        and not BASE_URL.startswith("https://demo-api")
    ):

        def normalize_category(raw):
            if not isinstance(raw, str):
                return "other"

            value = raw.strip().lower()

            # Direct matches to DeFiRate-style buckets
            if value in {
                "crypto",
                "culture",
                "economics",
                "finance",
                "mentions",
                "politics",
                "sports",
                "tech",
            }:
                return value

            # Fuzzy matching for prod
            if "crypto" in value:
                return "crypto"
            if any(x in value for x in ["econ", "inflation", "gdp"]):
                return "economics"
            if any(x in value for x in ["finance", "rates", "interest"]):
                return "finance"
            if "politic" in value or "election" in value:
                return "politics"
            if "sport" in value or "nfl" in value or "nba" in value:
                return "sports"
            if "tech" in value or "ai" in value:
                return "tech"
            if "culture" in value or "entertainment" in value:
                return "culture"
            if "mentions" in value:
                return "mentions"

            return "other"

        df["category_normalized"] = df["category"].apply(normalize_category)

        CATEGORY_OPTIONS = [
            "All categories",
            "crypto",
            "culture",
            "economics",
            "finance",
            "mentions",
            "politics",
            "sports",
            "tech",
        ]

        selected_category = st.selectbox("Category", CATEGORY_OPTIONS, index=0)

        if selected_category != "All categories":
            df = df[df["category_normalized"] == selected_category]
    else:
        # Demo environment: all category values are null
        st.info("Category data not available in this environment. Category filter disabled.")

    # ---- Platform toggle (future-proof) ----
    platform_values = sorted(df["platform"].unique().tolist())

    ui_options = []
    if "kalshi" in platform_values:
        ui_options.append("Kalshi")
    if "polymarket" in platform_values:
        ui_options.append("Polymarket")

    if len(ui_options) > 1:
        ui_options.insert(0, "Both")

    selected_platform = st.radio("Platform", ui_options, horizontal=True)

    if selected_platform == "Kalshi":
        df = df[df["platform"] == "kalshi"]
    elif selected_platform == "Polymarket":
        df = df[df["platform"] == "polymarket"]
    # If "Both": no filtering

    # ---- Status filter ----
    if "status" in df.columns:
        statuses = sorted(df["status"].dropna().unique().tolist())
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
        df = df[df["status"].isin(selected_statuses)]

    # ---- Volume filter ----
    if "volume" in df.columns:
        max_vol = df["volume"].fillna(0).max()
        if pd.isna(max_vol):
            max_vol = 0
        max_vol = int(max_vol)

        if max_vol > 0:
            min_volume = st.slider("Minimum total volume", 0, max_vol, 0)
            df = df[df["volume"].fillna(0) >= min_volume]

    # ---- Search ----
    search = st.text_input("Search markets")

    if search:
        mask = df.apply(
            lambda row: row.astype(str).str.contains(search, case=False).any(),
            axis=1,
        )
        df_filtered = df[mask]
    else:
        df_filtered = df

    df_filtered = df_filtered.copy()

    # ---- Convert dollar odds to % ----
    for col in [
        "yes_bid_dollars",
        "yes_ask_dollars",
        "no_bid_dollars",
        "no_ask_dollars",
        "last_price_dollars",
    ]:
        if col in df_filtered.columns:
            df_filtered[col] = (df_filtered[col].astype(float) * 100).round(1)

    # Rename for nicer labels
    df_filtered = df_filtered.rename(
        columns={
            "yes_bid_dollars": "yes_bid_pct",
            "yes_ask_dollars": "yes_ask_pct",
            "no_bid_dollars": "no_bid_pct",
            "no_ask_dollars": "no_ask_pct",
            "last_price_dollars": "last_traded_pct",
        }
    )

    # ---- Sort options ----
    sort_options = []
    if "volume_24h" in df_filtered.columns:
        sort_options.append("24h volume")
    if "volume" in df_filtered.columns:
        sort_options.append("total volume")
    if "last_traded_pct" in df_filtered.columns:
        sort_options.append("last traded %")
    if "close_time" in df_filtered.columns:
        sort_options.append("close time")

    if sort_options:
        if "24h volume" in sort_options:
            default_index = sort_options.index("24h volume")
        elif "total volume" in sort_options:
            default_index = sort_options.index("total volume")
        else:
            default_index = 0
    else:
        sort_options = ["none"]
        default_index = 0

    sort_by = st.selectbox("Sort by", sort_options, index=default_index)

    if sort_by == "24h volume":
        df_filtered = df_filtered.sort_values("volume_24h", ascending=False)
    elif sort_by == "total volume":
        df_filtered = df_filtered.sort_values("volume", ascending=False)
    elif sort_by == "last traded %":
        df_filtered = df_filtered.sort_values("last_traded_pct", ascending=False)
    elif sort_by == "close time":
        df_filtered = df_filtered.sort_values("close_time", ascending=True)

    st.write(f"Showing {len(df_filtered)} markets")

    # ---- Table or cards ----
    show_table = st.checkbox("Show raw table view", value=False)

    if show_table:
        st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)
    else:
        n_cols = 3
        df_display = df_filtered.reset_index(drop=True)

        for i in range(0, len(df_display), n_cols):
            row = df_display.iloc[i : i + n_cols]
            cols = st.columns(len(row))

            for col, (_, m) in zip(cols, row.iterrows()):
                with col:
                    with st.container(border=True):
                        # header
                        category_label = m.get("category_normalized", m.get("category", "Uncategorized"))
                        platform_label = m.get("platform", "Kalshi")
                        st.caption(f"{category_label.capitalize()} · {platform_label.capitalize()}")

                        st.markdown(f"**{m.get('title', 'Untitled market')}**")

                        close_time = m.get("close_time", "N/A")
                        status = m.get("status", "N/A")
                        st.write(f"⌛ Closes: {close_time}")
                        st.write(f"📌 Status: `{status}`")

                        yes_bid = m.get("yes_bid_pct", None)
                        yes_ask = m.get("yes_ask_pct", None)
                        last_traded = m.get("last_traded_pct", None)

                        st.write("**Yes side**")
                        st.write(f"- Bid: {yes_bid}%  | Ask: {yes_ask}%")
                        st.write(f"- Last traded: {last_traded}%")

                        vol_24h = m.get("volume_24h", None)
                        vol_total = m.get("volume", None)
                        st.write("**Activity**")
                        st.write(f"- 24h volume: {vol_24h}")
                        st.write(f"- Total volume: {vol_total}")

                        with st.expander("Raw details"):
                            st.json(dict(m))
