# services/stats_service.py

from dataclasses import dataclass
from typing import Optional
import pandas as pd


# ---------------------------
# Data models
# ---------------------------

@dataclass
class ExchangeStats:
    name: str
    weekly_notional_volume: float
    active_markets: int
    weekly_transactions: int
    open_interest: float


@dataclass
class TopLevelStats:
    kalshi: ExchangeStats
    polymarket: Optional[ExchangeStats] = None  # for later


# ---------------------------
# Stats computation
# ---------------------------

def compute_kalshi_stats(
    markets_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
) -> ExchangeStats:
    """
    Compute REAL weekly stats when trades_df is provided.
    Fallback to approximate stats if trades_df is empty.

    markets_df → open_interest, active_markets
    trades_df  → weekly_notional_volume, weekly_transactions
    """

    # No markets → zero everything
    if markets_df is None or markets_df.empty:
        return ExchangeStats(
            name="Kalshi",
            weekly_notional_volume=0.0,
            active_markets=0,
            weekly_transactions=0,
            open_interest=0.0,
        )

    # ----- Active markets -----
    active_markets = len(markets_df)

    # ----- Open interest -----
    if "open_interest" in markets_df.columns:
        open_interest = float(markets_df["open_interest"].fillna(0).sum())
    else:
        open_interest = 0.0

    # Default values until trades override them
    weekly_transactions = 0
    weekly_notional = 0.0

    # ================================================================
    #  CASE 1: REAL weekly stats using trades_df (THE GOOD CASE)
    # ================================================================
    if trades_df is not None and not trades_df.empty:

        # Weekly transactions = total contract count
        if "count" in trades_df.columns:
            weekly_transactions = int(trades_df["count"].fillna(0).sum())
        else:
            # Fallback: treat each row as one trade
            weekly_transactions = len(trades_df)

        # Weekly notional = SUM(price * count)
        # price is in cents → convert to dollars
        if "price" in trades_df.columns and "count" in trades_df.columns:
            price_cents = trades_df["price"].fillna(0).astype(float)
            size = trades_df["count"].fillna(0).astype(float)
            weekly_notional = float((price_cents / 100.0 * size).sum())

        else:
            # Fallback estimate: use any *_price_dollars column
            price_cols = [c for c in trades_df.columns if c.endswith("_price_dollars")]
            if price_cols and "count" in trades_df.columns:
                price_dollars = trades_df[price_cols[0]].fillna(0).astype(float)
                size = trades_df["count"].fillna(0).astype(float)
                weekly_notional = float((price_dollars * size).sum())

        return ExchangeStats(
            name="Kalshi",
            weekly_notional_volume=weekly_notional,
            active_markets=active_markets,
            weekly_transactions=weekly_transactions,
            open_interest=open_interest,
        )

    # ================================================================
    #  CASE 2: No trades_df → fallback approximation using markets_df
    # ================================================================
    if "volume_24h" in markets_df.columns:
        v24 = markets_df["volume_24h"].fillna(0).astype(float)
        weekly_transactions = int(v24.sum() * 7)

        if "last_price_dollars" in markets_df.columns:
            price = markets_df["last_price_dollars"].fillna(0).astype(float)
            weekly_notional = float((v24 * price * 7).sum())
        else:
            weekly_notional = float(v24.sum() * 7)

    elif "volume" in markets_df.columns:
        vol = markets_df["volume"].fillna(0).astype(float)
        weekly_transactions = int(vol.sum())

        if "last_price_dollars" in markets_df.columns:
            price = markets_df["last_price_dollars"].fillna(0).astype(float)
            weekly_notional = float((vol * price).sum())
        else:
            weekly_notional = float(vol.sum())

    return ExchangeStats(
        name="Kalshi",
        weekly_notional_volume=weekly_notional,
        active_markets=active_markets,
        weekly_transactions=weekly_transactions,
        open_interest=open_interest,
    )


# ---------------------------
# Public API
# ---------------------------

def get_top_level_stats(
    kalshi_markets_df: pd.DataFrame,
    kalshi_trades_df: Optional[pd.DataFrame] = None,
) -> TopLevelStats:

    kalshi_stats = compute_kalshi_stats(
        markets_df=kalshi_markets_df,
        trades_df=kalshi_trades_df,
    )

    return TopLevelStats(
        kalshi=kalshi_stats,
        polymarket=None,  # added later
    )