import streamlit as st
import pandas as pd
import requests

API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"


def fetch_kalshi_markets():
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()
    markets = data.get("markets", data)
    return pd.json_normalize(markets)


st.set_page_config(page_title="Prediction Markets MVP", layout="wide")
st.title("Prediction Market Terminal (Kalshi - Public Endpoint MVP)")
st.write("Data from Kalshi elections API.")

if st.button("Refresh Data"):
    st.session_state["df"] = fetch_kalshi_markets()
    st.success("Fetched latest markets!")

df = st.session_state.get("df")

if df is None:
    st.info("Click 'Refresh data' to load markets.")
else:
    cols = [
        "title",
        "subtitle",
        "ticker",
        "event_ticker",
        "category",
        # "market_type",
        # "created_time",
        # "open_time",
        "close_time",
        # "expected_expiration_time",
        # "expiration_time",
        # "latest_expiration_time",
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
    # "liquidity_dollars",
    # "notional_value_dollars"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    # filter out MULTIGAME from ticker
    if "ticker" in df.columns:
        df = df[~df["ticker"].str.contains("SPORTSMULTIGAME", case=False, na=False)]

    # grab league from the ticker column
    def extract_league(ticker):
        if not isinstance(ticker, str):
            return "OTHER"
        t = ticker.upper()
        if "NFL" in t:
            return "NFL"
        if "NBA" in t:
            return "NBA"
        return "OTHER"

    if "ticker" in df.columns:
        df["league"] = df["ticker"].apply(extract_league)
    else:
        df["league"] = "UNKNOWN"

    leagues = sorted(df["league"].unique().tolist())
    selected_leagues = st.multiselect("League", leagues, default=leagues)

    # apply the league filter to df
    df = df[df["league"].isin(selected_leagues)]

    if "status" in df.columns:
        statuses = sorted(df["status"].dropna().unique().tolist())
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
        df = df[df["status"].isin(selected_statuses)]
        
    if "volume" in df.columns:
        max_vol = df["volume"].fillna(0).max()
        if pd.isna(max_vol):
            max_vol = 0
        max_vol = int(max_vol)
        
        # only show slider if there's a variation in volume. min cannot == max
        if max_vol > 0:
            min_volume = st.slider("Minimum total volume", 0, max_vol, 0)
            df = df[df["volume"].fillna(0) >= min_volume]

    search = st.text_input("Search markets")

    if search:
        mask = df.apply(
            lambda row: row.astype(str).str.contains(search, case=False).any(),
            axis=1,
        )
        df_filtered = df[mask]
    else:
        df_filtered = df
    
    df_filtered["platform"] = "kalshi"
    
    # convert dollar odds to % where present
    for col in [
        "yes_bid_dollars",
        "yes_ask_dollars",
        "no_bid_dollars",
        "no_ask_dollars",
        "last_price_dollars",
    ]:
        if col in df_filtered.columns:
            df_filtered[col] = (df_filtered[col].astype(float) * 100).round(1)

    # rename for nicer labels
    df_filtered = df_filtered.rename(
        columns={
            "yes_bid_dollars": "yes_bid_pct",
            "yes_ask_dollars": "yes_ask_pct",
            "no_bid_dollars": "no_bid_pct",
            "no_ask_dollars": "no_ask_pct",
            "last_price_dollars": "last_traded_pct",
        }
    )

    # probabilities instead of raw dollar prices
    # for col in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"]:
    #     if col in df_filtered.columns:
    #         df_filtered[col] = (df_filtered[col].astype(float) * 100).round(1)

    # Sort options
    sort_options = []
    if "volume_24h" in df_filtered.columns:
        sort_options.append("24h volume")
    if "volume" in df_filtered.columns:
        sort_options.append("total volume")
    if "last_traded_pct" in df_filtered.columns:
        sort_options.append("last traded %")
    if "close_time" in df_filtered.columns:
        sort_options.append("close time")

    sort_by = st.selectbox("Sort by", sort_options or ["none"])

    if sort_by == "24h volume":
        df_filtered = df_filtered.sort_values("volume_24h", ascending=False)
    elif sort_by == "total volume":
        df_filtered = df_filtered.sort_values("volume", ascending=False)
    elif sort_by == "last traded %":
        df_filtered = df_filtered.sort_values("last_traded_pct", ascending=False)
    elif sort_by == "close time":
        df_filtered = df_filtered.sort_values("close_time", ascending=True)

    st.write(f"Showing {len(df_filtered)} markets")

    # toggle to still see the raw table if you want
    show_table = st.checkbox("Show raw table view", value=False)

    if show_table:
        st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)
    else:
        # card grid: 3 cards per row
        n_cols = 3
        df_display = df_filtered.reset_index(drop=True)

        for i in range(0, len(df_display), n_cols):
            row = df_display.iloc[i : i + n_cols]
            cols = st.columns(len(row))

            for col, (_, m) in zip(cols, row.iterrows()):
                with col:
                    with st.container(border=True):
                        # header
                        league = m.get("league", "Unknown")
                        st.caption(f"{league} · Kalshi")

                        st.markdown(f"**{m.get('title', 'Untitled market')}**")

                        # core metrics
                        close_time = m.get("close_time", "N/A")
                        status = m.get("status", "N/A")
                        st.write(f"⌛ Closes: {close_time}")
                        st.write(f"📌 Status: `{status}`")

                        # odds section
                        yes_bid = m.get("yes_bid_pct", None)
                        yes_ask = m.get("yes_ask_pct", None)
                        last_traded = m.get("last_traded_pct", None)

                        st.write("**Yes side**")
                        st.write(f"- Bid: {yes_bid}%  | Ask: {yes_ask}%")
                        st.write(f"- Last traded: {last_traded}%")

                        # activity
                        vol_24h = m.get("volume_24h", None)
                        vol_total = m.get("volume", None)
                        st.write("**Activity**")
                        st.write(f"- 24h volume: {vol_24h}")
                        st.write(f"- Total volume: {vol_total}")

                        # optional: raw details expander
                        with st.expander("Raw details"):
                            st.json(dict(m))
