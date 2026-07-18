"""
app.py  –  ArthaPulse: Indian Portfolio Backtester
Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta

from backtest import (
    download_stock_data,
    download_nifty50,
    run_backtest,
    run_sip_backtest,
    calculate_benchmark_equity,
    compute_comparison_metrics,
    rolling_sharpe,
    rolling_volatility,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArthaPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
#   Carrot  : #E8622A  (warm, energetic orange-red)
#   Teal    : #0D9488  (deep confident teal)
#   Dark bg : #0F1A1C  (very dark teal-black)
#   Card bg : #152426
#   Border  : #1E3538
# Light mode uses inverted surface colours but keeps the same accent pair.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root, [data-theme="light"] {
    --accent: #0F766E;
    --accent-2: #EA580C;
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --text: #0F172A;
    --muted: #64748B;
    --border: #E2E8F0;
    --sidebar-bg: #0F172A;
    --sidebar-text: #F8FAFC;
}

[data-theme="dark"] {
    --accent: #2DD4BF;
    --accent-2: #FB923C;
    --bg: #020617;
    --card: #111827;
    --text: #F8FAFC;
    --muted: #94A3B8;
    --border: #1F2937;
    --sidebar-bg: #020617;
    --sidebar-text: #F8FAFC;
}

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.main .block-container { padding-top: 0.2rem; max-width: 1440px; background: var(--bg); }
h1, h2, h3 { font-family: 'Fraunces', serif; letter-spacing: -0.02em; }

.ap-hero { display: flex; align-items: center; gap: 12px; padding: 0.4rem 0 0.7rem; }
.ap-wordmark h1 { margin: 0; font-size: 1.7rem; font-weight: 700; color: var(--accent); }
.ap-tagline { font-size: 0.88rem; color: var(--muted); }
.ap-divider { height: 2px; background: linear-gradient(90deg, var(--accent-2), var(--accent)); margin: 0.15rem 0 0.9rem; border-radius: 999px; }
.ap-section-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: var(--accent); margin: 0.2rem 0 0.6rem; }

.ap-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 0.95rem; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05); height: 100%; }
.ap-card-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.25rem; }
.ap-card-value { font-size: 1.3rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.ap-card-caption { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }
.ap-card-value.positive { color: #059669; }
.ap-card-value.negative { color: #DC2626; }

.sip-banner { background: rgba(15, 118, 110, 0.08); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 0.55rem 0.75rem; color: var(--accent); font-weight: 600; margin: 0.55rem 0; }

section[data-testid="stSidebar"] { background: var(--sidebar-bg) !important; border-right: 1px solid #1F2937 !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] label { color: var(--sidebar-text) !important; }
.sb-section { font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #5EEAD4 !important; padding: 0.35rem 0 0.25rem; border-bottom: 1px solid #1F2937; margin-bottom: 0.4rem; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] div[data-baseweb="select"], section[data-testid="stSidebar"] div[data-baseweb="input"] input { background: #111827 !important; color: var(--sidebar-text) !important; border-color: #334155 !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] span[data-baseweb="tag"] { background: #134E4A !important; color: #CCFBF1 !important; border: 1px solid #2DD4BF !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] .stButton > button { width: 100% !important; background: linear-gradient(90deg, var(--accent-2), var(--accent)) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.7rem 1rem !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] .stButton > button:hover { filter: brightness(1.04); }

.alloc-warn { background: #FFF7ED; color: #C2410C; border: 1px solid #FDBA74; border-radius: 8px; padding: 0.5rem 0.65rem; font-weight: 600; margin-top: 0.25rem; }
.stDownloadButton > button { background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; font-weight: 600 !important; }
.stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
div[data-testid="stAlert"] { border-radius: 8px !important; }
div[data-testid="stExpander"] summary { font-weight: 600; color: var(--text); }
div[data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); background: var(--card); margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SVG Logo  (carrot→teal gradient, no external deps)
# ─────────────────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg width="50" height="50" viewBox="0 0 50 50" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="50" height="50" rx="13" fill="url(#lg)"/>
  <polyline points="7,38 17,25 24,31 34,17 43,23"
            stroke="white" stroke-width="3" stroke-linecap="round"
            stroke-linejoin="round" fill="none"/>
  <circle cx="43" cy="23" r="3" fill="white"/>
  <circle cx="7"  cy="38" r="2" fill="rgba(255,255,255,0.5)"/>
  <defs>
    <linearGradient id="lg" x1="0" y1="0" x2="50" y2="50" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#E8622A"/>
      <stop offset="100%" stop-color="#0D9488"/>
    </linearGradient>
  </defs>
</svg>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ap-hero">
  {LOGO_SVG}
  <div class="ap-wordmark">
    <h1>ArthaPulse</h1>
    <span class="ap-tagline">
      Backtest any NSE portfolio &nbsp;·&nbsp; Benchmark against Nifty 50 &nbsp;·&nbsp; Measure real risk
    </span>
  </div>
</div>
<div class="ap-divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "BHARTIARTL",
    "ITC", "SBIN", "KOTAKBANK", "AXISBANK", "LT", "MARUTI", "TITAN", "SUNPHARMA",
    "ASIANPAINT", "M&M", "ULTRACEMCO", "NTPC", "POWERGRID", "ONGC", "BAJAJFINSV",
    "HEROMOTOCO", "CIPLA", "DRREDDY", "WIPRO", "TECHM", "TATAMOTORS", "TATASTEEL",
    "COALINDIA", "BRITANNIA", "INDUSINDBK", "ADANIENT", "JSWSTEEL", "GRASIM",
    "EICHERMOT", "NESTLEIND", "BPCL", "HCLTECH", "PNB", "BANKBARODA", "IDFCFIRSTB",
    "ZOMATO", "DMART", "SHRIRAMFIN", "LICI", "IRCTC", "YESBANK", "PFC"
]

@st.cache_data(ttl=86_400)
def load_nse_symbols() -> list[str]:
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = pd.read_csv(url, on_bad_lines="skip")
        syms = df[df[" SERIES"] == "EQ"]["SYMBOL"].str.strip().tolist()
        cleaned = sorted(set(s for s in syms if s))
        if cleaned:
            return cleaned
    except Exception:
        pass

    return FALLBACK_SYMBOLS

@st.cache_data(ttl=300)
def cached_stock_data(tickers_tuple, start, end):
    return download_stock_data(list(tickers_tuple), start, end)

@st.cache_data(ttl=300)
def cached_nifty(start, end):
    return download_nifty50(start, end)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-section">Universe</div>', unsafe_allow_html=True)
    all_symbols = load_nse_symbols()
    selected_tickers: list[str] = st.multiselect(
        "NSE stocks",
        options=all_symbols,
        default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        help="Type to search any NSE EQ-series stock",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start", date.today() - timedelta(days=365 * 3))
    with c2:
        end_date = st.date_input("End", date.today())

    if selected_tickers and (end_date - start_date).days < 180:
        st.warning("< 6 months — metrics may be unreliable.")

    st.markdown('<div class="sb-section">Investment</div>', unsafe_allow_html=True)
    invest_mode = st.radio("Mode", ("Lump Sum", "SIP (monthly)"), horizontal=True)
    if invest_mode == "Lump Sum":
        initial_capital = st.number_input("Total amount (₹)", min_value=1_000, value=1_00_000, step=5_000)
        monthly_sip = None
    else:
        monthly_sip = st.number_input("Monthly SIP (₹)", min_value=500, value=10_000, step=500)
        initial_capital = None

    st.markdown('<div class="sb-section">Weights</div>', unsafe_allow_html=True)
    alloc_method = st.radio("Method", ("Equal weight", "Custom %"), horizontal=True)
    allocations: dict[str, float] = {}
    total_alloc = 0.0

    if alloc_method == "Equal weight" and selected_tickers:
        w = 1.0 / len(selected_tickers)
        allocations = {t: w for t in selected_tickers}
        total_alloc = 1.0
    elif alloc_method == "Custom %" and selected_tickers:
        cols = st.columns(2)
        rem = 100
        for i, ticker in enumerate(selected_tickers):
            with cols[i % 2]:
                if i < len(selected_tickers) - 1:
                    default = rem // max(len(selected_tickers) - i, 1)
                    alloc   = st.slider(ticker, 0, max(rem, 0), value=min(default, max(rem, 0)), key=f"a_{ticker}")
                    allocations[ticker] = alloc / 100
                    rem -= alloc
                else:
                    rem = max(rem, 0)
                    allocations[ticker] = rem / 100
                    st.markdown(f"**{ticker}:** {rem}%")
        total_alloc = sum(allocations.values())
        if abs(total_alloc - 1.0) > 0.005:
            st.markdown(
                f'<div class="alloc-warn">Weights sum to {total_alloc:.0%} — must be 100%</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sb-section">Parameters</div>', unsafe_allow_html=True)
    risk_free_rate = st.slider("Risk-free rate (% p.a.)", 0.0, 12.0, 6.5, 0.25,
        help="Indian 10-yr G-sec yield approx. Used for Sharpe, Sortino, Alpha.") / 100
    compare_benchmark = st.checkbox("Compare with Nifty 50", value=True)
    show_rolling      = st.checkbox("Show 90-day rolling metrics", value=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
# Carrot+teal chart palette
C_PORT  = "#14B8A6"   # teal  — portfolio line
C_BENCH = "#F0733A"   # carrot — benchmark
C_DD    = "#F87171"   # soft red for drawdown
C_GREEN = "#34D399"
C_BAR_POS = "#0D9488"
C_BAR_NEG = "#E8622A"

def fmt_pct(v: float) -> str: return f"{v:.2%}"
def fmt_inr(v: float) -> str: return f"₹{v:,.2f}"

def card(label: str, value: str, caption: str = "", vcls: str = "") -> None:
    st.markdown(
        f'<div class="ap-card">'
        f'<div class="ap-card-label">{label}</div>'
        f'<div class="ap-card-value {vcls}">{value}</div>'
        f'<div class="ap-card-caption">{caption}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def section(label: str) -> None:
    st.markdown(f'<div class="ap-section-label">{label}</div>', unsafe_allow_html=True)

def make_fig(title: str, xlab: str = "Date", ylab: str = "") -> go.Figure:
    """Return a blank figure with consistent layout — legend BELOW chart, no collision."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Fraunces, serif", size=16, color="#0C1F1E"),
            x=0,
            xanchor="left",
            pad=dict(l=4, t=4),
        ),
        xaxis_title=xlab,
        yaxis_title=ylab,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=52, b=52, l=10, r=10),
        # Legend anchored BELOW the plot area — no overlap with chart lines
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Plus Jakarta Sans, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F4F4", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F4F4", zeroline=False)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if run_button:
    if not selected_tickers:
        st.warning("Select at least one stock.")
        st.stop()
    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        st.stop()
    if abs(total_alloc - 1.0) > 0.005 and alloc_method == "Custom %":
        st.error(f"Weights sum to {total_alloc:.0%}. Fix to 100% first.")
        st.stop()

    with st.spinner("Downloading market data…"):
        prices, failed = cached_stock_data(tuple(selected_tickers), start_date, end_date)

    if failed:
        st.toast(f"Skipped: {', '.join(failed)}", icon="⚠️")

    valid_tickers = [t for t in selected_tickers if t in prices.columns]
    if prices.empty or not valid_tickers:
        st.error("No price data found. Check tickers / date range.")
        st.stop()

    valid_allocs = {t: allocations[t] for t in valid_tickers}
    total_w      = sum(valid_allocs.values())
    valid_allocs = {t: v / total_w for t, v in valid_allocs.items()}
    prices       = prices[valid_tickers]

    with st.spinner("Running backtest…"):
        if invest_mode == "Lump Sum":
            port_val, dd, metrics, shares = run_backtest(prices, initial_capital, valid_allocs, risk_free_rate)
            total_invested = initial_capital
        else:
            port_val, dd, metrics = run_sip_backtest(prices, monthly_sip, valid_allocs, risk_free_rate)
            shares         = {}
            total_invested = metrics["Total Invested"]

        bench_equity  = None
        bench_metrics = None
        if compare_benchmark:
            try:
                bench_prices = cached_nifty(start_date, end_date)
                if not bench_prices.empty:
                    bench_equity = calculate_benchmark_equity(bench_prices, total_invested)
                    common = port_val.index.intersection(bench_equity.index)
                    if len(common) > 1:
                        pr = port_val.pct_change().dropna().squeeze()
                        br = bench_equity.pct_change().dropna().squeeze()
                        bench_metrics = compute_comparison_metrics(pr, br, risk_free_rate)
            except Exception as be:
                st.warning(f"Benchmark error: {be}")

    years        = (prices.index[-1] - prices.index[0]).days / 365.25
    first_prices = prices.iloc[0]
    last_prices  = prices.iloc[-1]

    # ── Snapshot ─────────────────────────────────────────────────────────────
    st.markdown('<div class="ap-divider"></div>', unsafe_allow_html=True)
    section("Portfolio snapshot")

    r1 = st.columns(4)
    with r1[0]: card("Final Value", fmt_inr(metrics["Final Value"]), f"vs {fmt_inr(total_invested)} invested")
    with r1[1]:
        ret = metrics["Total Return"]
        card("Total Return", fmt_pct(ret), f"Over {years:.1f} yrs", "positive" if ret >= 0 else "negative")
    with r1[2]:
        card("CAGR", fmt_pct(metrics["CAGR"]), "Annualised growth",
             "positive" if metrics["CAGR"] >= 0 else "negative")
    with r1[3]:
        card("Max Drawdown", fmt_pct(metrics["Max Drawdown"]),
             metrics["Max DD Date"].strftime("on %d %b %Y"), "negative")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    r2 = st.columns(4)
    with r2[0]: card("Sharpe Ratio",  f"{metrics['Sharpe Ratio']:.2f}",  f"rf = {risk_free_rate:.1%} p.a.")
    with r2[1]: card("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}",  "Downside vol only")
    with r2[2]: card("Calmar Ratio",  f"{metrics['Calmar Ratio']:.2f}",   "CAGR ÷ |Max DD|")
    with r2[3]: card("Volatility",    fmt_pct(metrics["Annualised Volatility"]), "Annualised daily σ")

    if bench_metrics:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        section("vs Nifty 50")
        r3 = st.columns(4)
        bm_meta = {
            "Beta":              ("Beta",          "Market sensitivity",            ""),
            "Alpha (ann.)":      ("Alpha",          "Jensen's alpha",               "positive" if bench_metrics.get("Alpha (ann.)", 0) >= 0 else "negative"),
            "Tracking Error":    ("Tracking Error", "Active risk",                  ""),
            "Information Ratio": ("Info Ratio",     "Active return / tracking error",""),
        }
        for i, (key, (lbl, cap, vcls)) in enumerate(bm_meta.items()):
            if key in bench_metrics:
                v = bench_metrics[key]
                with r3[i]:
                    card(lbl, fmt_pct(v) if key != "Beta" else f"{v:.2f}", cap, vcls)

    if invest_mode == "SIP (monthly)":
        st.markdown(
            f'<div class="sip-banner">'
            f'SIP — {metrics["SIP Instalments"]} monthly instalments of '
            f'₹{monthly_sip:,} = ₹{total_invested:,.0f} total deployed'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Charts ───────────────────────────────────────────────────────────────
    st.markdown('<div class="ap-divider"></div>', unsafe_allow_html=True)
    section("Performance charts")

    # 1. Equity curve  — legend placed BELOW, title has room at top
    fig1 = make_fig("Equity Curve", ylab="₹")
    fig1.add_trace(go.Scatter(
        x=port_val.index, y=port_val,
        mode="lines", name="Your Portfolio",
        line=dict(color=C_PORT, width=2.5),
        hovertemplate="₹%{y:,.0f}<extra>Portfolio</extra>",
    ))
    if bench_equity is not None and not bench_equity.empty:
        fig1.add_trace(go.Scatter(
            x=bench_equity.index, y=bench_equity,
            mode="lines", name="Nifty 50 (rebased)",
            line=dict(color=C_BENCH, width=1.8, dash="dot"),
            hovertemplate="₹%{y:,.0f}<extra>Nifty 50</extra>",
        ))
    # Extra bottom margin so legend doesn't sit over the x-axis
    fig1.update_layout(margin=dict(t=52, b=80, l=10, r=10), height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Drawdown
    fig2 = make_fig("Portfolio Drawdown", ylab="%")
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd * 100,
        fill="tozeroy", mode="lines", name="Drawdown",
        line=dict(color=C_DD, width=1.2),
        fillcolor="rgba(248,113,113,0.15)",
        hovertemplate="%{y:.2f}%<extra>Drawdown</extra>",
    ))
    fig2.update_layout(yaxis_ticksuffix="%", showlegend=False, height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Rolling metrics
    if show_rolling:
        daily_ret = port_val.pct_change().dropna()
        roll_sh   = rolling_sharpe(daily_ret, 90, risk_free_rate)
        roll_vol  = rolling_volatility(daily_ret, 90)

        fig3 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("90-Day Rolling Sharpe Ratio", "90-Day Rolling Volatility (ann.)"),
            vertical_spacing=0.16,
        )
        fig3.add_trace(go.Scatter(
            x=roll_sh.index, y=roll_sh, mode="lines",
            line=dict(color=C_PORT, width=1.8), name="Rolling Sharpe",
        ), row=1, col=1)
        fig3.add_hline(y=1.0, line_dash="dot", line_color=C_GREEN,
                       annotation_text="Sharpe = 1", row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol * 100, mode="lines",
            line=dict(color=C_BENCH, width=1.8), name="Rolling Vol",
        ), row=2, col=1)
        fig3.update_layout(
            template="plotly_white", showlegend=False, height=430,
            margin=dict(t=56, b=40, l=10, r=10),
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        )
        fig3.update_xaxes(showgrid=True, gridcolor="#F0F4F4", zeroline=False)
        fig3.update_yaxes(showgrid=True, gridcolor="#F0F4F4", zeroline=False,
                          title_text="Sharpe", row=1, col=1)
        fig3.update_yaxes(title_text="Vol (%)", ticksuffix="%", row=2, col=1)
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Contribution bar
    if "Contributions" in metrics and metrics["Contributions"]:
        contribs = metrics["Contributions"]
        sorted_c = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        tickers_c, values_c = zip(*sorted_c)
        colors_c = [C_BAR_POS if v >= 0 else C_BAR_NEG for v in values_c]

        fig4 = make_fig("Stock Contribution to Portfolio Return", xlab="Stock", ylab="%")
        fig4.add_trace(go.Bar(
            x=list(tickers_c),
            y=[v * 100 for v in values_c],
            marker_color=colors_c,
            marker_line_width=0,
            text=[f"{v:.1%}" for v in values_c],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig4.update_layout(
            yaxis_ticksuffix="%", showlegend=False,
            margin=dict(t=52, b=40, l=10, r=10), height=340,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Holdings table ────────────────────────────────────────────────────────
    st.markdown('<div class="ap-divider"></div>', unsafe_allow_html=True)
    section("Holdings breakdown")

    rows = []
    for t in valid_tickers:
        stock_ret = (last_prices[t] / first_prices[t]) - 1
        contrib   = metrics.get("Contributions", {}).get(t, 0.0)
        rows.append({
            "Ticker":        t,
            "Allocation":    fmt_pct(valid_allocs[t]),
            "Shares":        round(shares.get(t, 0), 4) if shares else "—",
            "Buy Price":     first_prices[t],
            "Current Price": last_prices[t],
            "Return":        stock_ret,
            "Contribution":  contrib,
        })

    holdings_df = pd.DataFrame(rows)
    st.dataframe(
        holdings_df.style
        .format({
            "Buy Price":     "₹{:,.2f}",
            "Current Price": "₹{:,.2f}",
            "Return":        "{:.2%}",
            "Contribution":  "{:.2%}",
        })
        .map(
            lambda v: "color:#059669;font-weight:700" if isinstance(v, float) and v >= 0
                      else ("color:#DC2626;font-weight:700" if isinstance(v, float) and v < 0 else ""),
            subset=["Return", "Contribution"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown('<div class="ap-divider"></div>', unsafe_allow_html=True)
    section("Export data")
    dl1, dl2 = st.columns(2)
    with dl1:
        csv_port = port_val.reset_index()
        csv_port.columns = ["Date", "Portfolio Value (INR)"]
        st.download_button(
            "Download equity curve (CSV)",
            data=csv_port.to_csv(index=False).encode(),
            file_name=f"ArthaPulse_equity_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download holdings (CSV)",
            data=holdings_df.to_csv(index=False).encode(),
            file_name=f"ArthaPulse_holdings_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Landing state ─────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-top: 3px solid var(--teal);
        border-radius: 14px;
        padding: 2rem 2.2rem;
        margin-top: 0.4rem;
        box-shadow: var(--shadow-sm);
    ">
      <div style="font-family:'Fraunces',serif;font-size:1.2rem;
                  color:var(--text-primary);margin-bottom:1rem;font-weight:600">
        Getting started
      </div>
      <ol style="color:var(--text-secondary);line-height:2.1;font-size:0.88rem;
                 margin:0;padding-left:1.2rem">
        <li>Search and pick <strong style="color:var(--carrot)">NSE stocks</strong> in the sidebar</li>
        <li>Choose <strong style="color:var(--carrot)">Lump Sum</strong> or <strong style="color:var(--carrot)">SIP</strong> investment mode</li>
        <li>Set <strong style="color:var(--teal)">allocation weights</strong> — equal or custom</li>
        <li>Optionally benchmark against <strong style="color:var(--teal)">Nifty 50</strong></li>
        <li>Hit <strong style="color:var(--carrot)">Run Backtest</strong> and explore the results</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    with st.expander("Metric glossary"):
        st.markdown("""
| Metric | Definition |
|---|---|
| **CAGR** | Compound Annual Growth Rate — smooth annualised return |
| **Sharpe Ratio** | Excess return over risk-free rate per unit of total risk |
| **Sortino Ratio** | Like Sharpe, but penalises downside volatility only |
| **Calmar Ratio** | CAGR divided by absolute Max Drawdown |
| **Max Drawdown** | Largest peak-to-trough decline in portfolio value |
| **Beta** | Sensitivity of portfolio returns to Nifty 50 moves |
| **Alpha** | Jensen's Alpha — return beyond what Beta explains |
| **Tracking Error** | Annualised std dev of active returns vs benchmark |
| **Information Ratio** | Active return per unit of tracking error |
| **Contribution** | Each stock's weighted impact on total portfolio return |
        """)