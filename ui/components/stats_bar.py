# ui/components/stats_bar.py
import streamlit as st
from typing import Optional
from services.stats_service import TopLevelStats, ExchangeStats


def format_dollar(value: float) -> str:
    # Simple formatting, tweak later to use humanize/abbreviations if you want
    return f"${value:,.0f}"


def format_int(value: int) -> str:
    return f"{value:,}"


def _metric_row(label: str,
                kalshi_value: str,
                polymarket_value: Optional[str] = None,
                show_polymarket: bool = False):
    """
    One row inside a card: label + (Kalshi vs Polymarket).
    For now we really only use one value, but the layout is future-proof.
    """
    st.markdown(f"**{label}**")
    cols = st.columns(2) if show_polymarket else [st.container()]

    with cols[0]:
        st.caption("Kalshi")
        st.markdown(f"### {kalshi_value}")

    if show_polymarket:
        with cols[1]:
            st.caption("Polymarket")
            st.markdown(f"### {polymarket_value or '—'}")


def render_stats_bar(stats: TopLevelStats):
    # Decide if we should show a Polymarket column at all
    show_polymarket = stats.polymarket is not None

    kalshi = stats.kalshi
    polymarket = stats.polymarket

    # Prepare values
    weekly_notional_k = format_dollar(kalshi.weekly_notional_volume)
    active_markets_k = format_int(kalshi.active_markets)
    weekly_tx_k      = format_int(kalshi.weekly_transactions)
    open_interest_k  = format_dollar(kalshi.open_interest)

    weekly_notional_p = format_dollar(polymarket.weekly_notional_volume) if polymarket else None
    active_markets_p  = format_int(polymarket.active_markets) if polymarket else None
    weekly_tx_p       = format_int(polymarket.weekly_transactions) if polymarket else None
    open_interest_p   = format_dollar(polymarket.open_interest) if polymarket else None

    # Top-level container styling
    st.markdown("### Overview")
    st.markdown(
        """
        <style>
        .oddsy-card {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #333333;
            background-color: #0e1117;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container():
            st.markdown('<div class="oddsy-card">', unsafe_allow_html=True)
            _metric_row(
                label="Weekly Notional Volume",
                kalshi_value=weekly_notional_k,
                polymarket_value=weekly_notional_p,
                show_polymarket=show_polymarket,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        with st.container():
            st.markdown('<div class="oddsy-card">', unsafe_allow_html=True)
            _metric_row(
                label="Active Markets",
                kalshi_value=active_markets_k,
                polymarket_value=active_markets_p,
                show_polymarket=show_polymarket,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        with st.container():
            st.markdown('<div class="oddsy-card">', unsafe_allow_html=True)
            _metric_row(
                label="Weekly Transactions",
                kalshi_value=weekly_tx_k,
                polymarket_value=weekly_tx_p,
                show_polymarket=show_polymarket,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        with st.container():
            st.markdown('<div class="oddsy-card">', unsafe_allow_html=True)
            _metric_row(
                label="Open Interest",
                kalshi_value=open_interest_k,
                polymarket_value=open_interest_p,
                show_polymarket=show_polymarket,
            )
            st.markdown('</div>', unsafe_allow_html=True)
