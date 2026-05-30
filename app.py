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

# ──────────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ArthaPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main .block-container { padding-top: 1.2rem; max-width: 1400px; }

    h1, h2, h3 { font-family: 'DM Serif Display', serif; }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e8edf2;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .metric-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.09); }

    /* Primary button */
    .stButton>button {
        border-radius: 10px;
        padding: 0.55rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        background: linear-gradient(135deg, #1a6fc4 0%, #0e4f91 100%);
        color: white;
        border: none;
        transition: opacity 0.15s, transform 0.1s;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }

    /* Pill badge */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pill-green { background: #e6f4ea; color: #1e7e34; }
    .pill-red   { background: #fce8e6; color: #c62828; }

    .quote {
        font-style: italic;
        color: #6b7a8d;
        font-size: 1rem;
        border-left: 3px solid #1a6fc4;
        padding-left: 0.8rem;
        margin-bottom: 0.5rem;
    }

    /* Allocation warning */
    .alloc-warn { color: #e65100; font-size: 0.82rem; font-weight: 600; }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] { background: #f5f8fc; }
    section[data-testid="stSidebar"] h2 { font-size: 1.1rem; }

    /* Table hover */
    .dataframe tbody tr:hover td { background: #f0f6ff !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.image("https://img.icons8.com/fluency/64/combo-chart.png", width=52)
with col_title:
    st.title("ArthaPulse — Indian Portfolio Backtester")

st.markdown(
    '<p class="quote">"Know your past portfolio to craft your future wealth."</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────
@st.cache_data(ttl=86_400)          # NSE list changes at most once/day
def load_nse_symbols() -> list[str]:
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df  = pd.read_csv(url)
    syms = df[df[" SERIES"] == "EQ"]["SYMBOL"].str.strip().tolist()
    return sorted(set(s for s in syms if s))


@st.cache_data(ttl=300)
def cached_stock_data(tickers_tuple, start, end):
    """Wrapper so we can hash a tuple."""
    df, failed = download_stock_data(list(tickers_tuple), start, end)
    return df, failed


@st.cache_data(ttl=300)
def cached_nifty(start, end):
    return download_nifty50(start, end)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    all_symbols = load_nse_symbols()

    selected_tickers: list[str] = st.multiselect(
        "🔍 Select NSE stocks",
        options=all_symbols,
        default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        help="Type to search any NSE EQ-series stock",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", date.today() - timedelta(days=365 * 3))
    with col2:
        end_date = st.date_input("End date", date.today())

    # Date range warning
    if selected_tickers and (end_date - start_date).days < 180:
        st.warning("⚠️ < 6 months selected — metrics may be unreliable.")

    st.markdown("---")

    # Investment mode
    st.subheader("💰 Investment Mode")
    invest_mode = st.radio("Mode", ("Lump Sum", "SIP (monthly)"), horizontal=True)

    if invest_mode == "Lump Sum":
        initial_capital = st.number_input(
            "Total investment (₹)", min_value=1_000, value=1_00_000, step=5_000,
        )
    else:
        monthly_sip = st.number_input(
            "Monthly SIP amount (₹)", min_value=500, value=10_000, step=500,
        )
        initial_capital = None   # not used in SIP mode

    st.markdown("---")
    st.subheader("📐 Allocation Method")
    alloc_method = st.radio("Weights", ("Equal weight", "Custom %"), horizontal=True)

    allocations: dict[str, float] = {}
    total_alloc = 0.0

    if alloc_method == "Equal weight" and selected_tickers:
        w = 1.0 / len(selected_tickers)
        allocations = {t: w for t in selected_tickers}
        total_alloc = 1.0

    elif alloc_method == "Custom %" and selected_tickers:
        st.markdown("##### Set allocation (%)")
        cols = st.columns(2)
        rem  = 100
        for i, ticker in enumerate(selected_tickers):
            with cols[i % 2]:
                if i < len(selected_tickers) - 1:
                    default = rem // max(len(selected_tickers) - i, 1)
                    alloc   = st.slider(
                        ticker, 0, max(rem, 0),
                        value=min(default, max(rem, 0)),
                        key=f"a_{ticker}",
                    )
                    allocations[ticker] = alloc / 100
                    rem -= alloc
                else:
                    rem = max(rem, 0)
                    allocations[ticker] = rem / 100
                    st.markdown(f"**{ticker}:** {rem}%")

        total_alloc = sum(allocations.values())
        if abs(total_alloc - 1.0) > 0.005:
            st.markdown(
                f'<p class="alloc-warn">⚠ Allocations sum to {total_alloc:.0%} (must be 100%)</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("🔧 Advanced Settings")
    risk_free_rate = st.slider(
        "Risk-free rate (% p.a.)", 0.0, 12.0, 6.5, 0.25,
        help="Used for Sharpe, Sortino, Alpha. Approx. Indian 10-yr G-sec yield.",
    ) / 100
    compare_benchmark = st.checkbox("📈 Compare with Nifty 50", value=True)
    show_rolling      = st.checkbox("📉 Show rolling metrics (90-day)", value=True)

    st.markdown("---")
    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
COLOR_PORT  = "#1a6fc4"
COLOR_BENCH = "#f5a623"
COLOR_DD    = "#d32f2f"

def fmt_pct(v: float) -> str:
    return f"{v:.2%}"

def fmt_inr(v: float) -> str:
    return f"₹{v:,.2f}"

def delta_color(v: float) -> str:
    return "normal" if v >= 0 else "inverse"

def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:0.78rem;color:#6b7a8d;font-weight:600;text-transform:uppercase;letter-spacing:.04em">{label}</div>'
        f'<div style="font-size:1.55rem;font-weight:700;color:#0d1b2a;margin:4px 0">{value}</div>'
        f'<div style="font-size:0.78rem;color:#8e99aa">{caption}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if run_button:
    # Validation
    if not selected_tickers:
        st.warning("⚠️ Select at least one stock.")
        st.stop()
    if start_date >= end_date:
        st.warning("⚠️ Start date must be before end date.")
        st.stop()
    if abs(total_alloc - 1.0) > 0.005 and alloc_method == "Custom %":
        st.error(f"Allocations sum to {total_alloc:.0%}. Please fix to 100% before running.")
        st.stop()

    with st.spinner("⏳ Downloading market data…"):
        prices, failed = cached_stock_data(
            tuple(selected_tickers), start_date, end_date,
        )

    if failed:
        st.toast(f"⚠️ Could not download: {', '.join(failed)}", icon="⚠️")

    valid_tickers = [t for t in selected_tickers if t in prices.columns]
    if prices.empty or not valid_tickers:
        st.error("No price data found. Check tickers / date range.")
        st.stop()

    # Re-normalise allocations if some tickers failed
    valid_allocs = {t: allocations[t] for t in valid_tickers}
    total_w      = sum(valid_allocs.values())
    valid_allocs = {t: v / total_w for t, v in valid_allocs.items()}
    prices       = prices[valid_tickers]

    with st.spinner("⏳ Running backtest…"):
        if invest_mode == "Lump Sum":
            port_val, dd, metrics, shares = run_backtest(
                prices, initial_capital, valid_allocs, risk_free_rate,
            )
            total_invested = initial_capital
        else:
            port_val, dd, metrics = run_sip_backtest(
                prices, monthly_sip, valid_allocs, risk_free_rate,
            )
            shares         = {}
            total_invested = metrics["Total Invested"]
            first_prices   = prices.iloc[0]
            last_prices    = prices.iloc[-1]

        bench_equity  = None
        bench_metrics = None
        if compare_benchmark:
            try:
                bench_prices = cached_nifty(start_date, end_date)
                if not bench_prices.empty:
                    bench_equity = calculate_benchmark_equity(bench_prices, total_invested)
                    common       = port_val.index.intersection(bench_equity.index)
                    if len(common) > 1:
                        pr = port_val.pct_change().dropna().squeeze()
                        br = bench_equity.pct_change().dropna().squeeze()
                        bench_metrics = compute_comparison_metrics(pr, br, risk_free_rate)
                    else:
                        st.warning("Not enough overlapping dates for Nifty 50 comparison.")
            except Exception as be:
                st.warning(f"Benchmark error: {be}")

    # ── Snapshot cards ──────────────────────────────────
    st.subheader("📌 Portfolio Snapshot")
    years = (prices.index[-1] - prices.index[0]).days / 365.25

    r1 = st.columns(4)
    with r1[0]:
        metric_card("Final Value", fmt_inr(metrics["Final Value"]),
                    f"vs {fmt_inr(total_invested)} invested")
    with r1[1]:
        metric_card("Total Return", fmt_pct(metrics["Total Return"]),
                    f"Over {years:.1f} years")
    with r1[2]:
        metric_card("CAGR", fmt_pct(metrics["CAGR"]), "Annualised growth")
    with r1[3]:
        metric_card("Max Drawdown", fmt_pct(metrics["Max Drawdown"]),
                    f"on {metrics['Max DD Date'].strftime('%d-%b-%Y')}")

    st.markdown("<br>", unsafe_allow_html=True)
    r2 = st.columns(4)
    with r2[0]:
        metric_card("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}",
                    f"rf = {risk_free_rate:.1%} p.a.")
    with r2[1]:
        metric_card("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}",
                    "Downside risk-adjusted")
    with r2[2]:
        metric_card("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}",
                    "CAGR / |Max Drawdown|")
    with r2[3]:
        metric_card("Volatility (ann.)", fmt_pct(metrics["Annualised Volatility"]),
                    "Daily σ × √252")

    if bench_metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        r3 = st.columns(4)
        labels = {
            "Beta":              ("Beta", "Market sensitivity"),
            "Alpha (ann.)":      ("Alpha (ann.)", "Jensen's alpha vs Nifty 50"),
            "Tracking Error":    ("Tracking Error", "Active risk"),
            "Information Ratio": ("Information Ratio", "Active return / active risk"),
        }
        for i, (key, (label, caption)) in enumerate(labels.items()):
            if key in bench_metrics:
                v = bench_metrics[key]
                with r3[i]:
                    metric_card(label,
                                fmt_pct(v) if key != "Beta" else f"{v:.2f}",
                                caption)

    # ── SIP-specific info ───────────────────────────────
    if invest_mode == "SIP (monthly)":
        st.info(
            f"💡 SIP summary: {metrics['SIP Instalments']} monthly instalments "
            f"of ₹{monthly_sip:,} = ₹{total_invested:,.0f} total invested."
        )

    # ── Charts ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Performance Charts")

    # 1. Equity curve
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=port_val.index, y=port_val,
        mode="lines", name="Your Portfolio",
        line=dict(color=COLOR_PORT, width=2.5),
        hovertemplate="₹%{y:,.0f}<extra>Portfolio</extra>",
    ))
    if bench_equity is not None and not bench_equity.empty:
        fig1.add_trace(go.Scatter(
            x=bench_equity.index, y=bench_equity,
            mode="lines", name="Nifty 50 (rebased)",
            line=dict(color=COLOR_BENCH, width=2, dash="dash"),
            hovertemplate="₹%{y:,.0f}<extra>Nifty 50</extra>",
        ))
    fig1.update_layout(
        title="Equity Curve",
        xaxis_title="Date", yaxis_title="₹",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Drawdown
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd * 100,
        fill="tozeroy", mode="lines", name="Drawdown",
        line=dict(color=COLOR_DD, width=1),
        fillcolor="rgba(211,47,47,0.15)",
        hovertemplate="%{y:.2f}%<extra>Drawdown</extra>",
    ))
    fig2.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date", yaxis_title="%",
        yaxis_ticksuffix="%",
        template="plotly_white",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Rolling metrics
    if show_rolling:
        daily_ret = port_val.pct_change().dropna()
        roll_sh   = rolling_sharpe(daily_ret, 90, risk_free_rate)
        roll_vol  = rolling_volatility(daily_ret, 90)

        fig3 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("90-Day Rolling Sharpe", "90-Day Rolling Volatility (ann.)"),
            vertical_spacing=0.12,
        )
        fig3.add_trace(
            go.Scatter(x=roll_sh.index, y=roll_sh, mode="lines",
                       line=dict(color="#2196f3", width=1.8), name="Rolling Sharpe"),
            row=1, col=1,
        )
        fig3.add_hline(y=1.0, line_dash="dot", line_color="green",
                       annotation_text="Sharpe = 1", row=1, col=1)
        fig3.add_trace(
            go.Scatter(x=roll_vol.index, y=roll_vol * 100, mode="lines",
                       line=dict(color="#ff9800", width=1.8), name="Rolling Vol %"),
            row=2, col=1,
        )
        fig3.update_yaxes(title_text="Sharpe", row=1, col=1)
        fig3.update_yaxes(title_text="Vol (%)", ticksuffix="%", row=2, col=1)
        fig3.update_layout(
            template="plotly_white", showlegend=False,
            height=480, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Stock contribution bar
    if "Contributions" in metrics and metrics["Contributions"]:
        contribs = metrics["Contributions"]
        sorted_c = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        tickers_c, values_c = zip(*sorted_c)
        colors_c = [COLOR_PORT if v >= 0 else COLOR_DD for v in values_c]

        fig4 = go.Figure(go.Bar(
            x=list(tickers_c),
            y=[v * 100 for v in values_c],
            marker_color=colors_c,
            text=[f"{v:.1%}" for v in values_c],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig4.update_layout(
            title="Stock Contribution to Portfolio Return (%)",
            xaxis_title="Stock", yaxis_title="Contribution (%)",
            yaxis_ticksuffix="%",
            template="plotly_white",
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Holdings table ───────────────────────────────────
    st.markdown("---")
    st.subheader("🧾 Portfolio Holdings")

    first_prices = prices.iloc[0]
    last_prices  = prices.iloc[-1]

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
            "Stock Return":  stock_ret,
            "Contribution":  contrib,
        })

    holdings_df = pd.DataFrame(rows)
    st.dataframe(
        holdings_df.style
        .format({
            "Buy Price":     "₹{:,.2f}",
            "Current Price": "₹{:,.2f}",
            "Stock Return":  "{:.2%}",
            "Contribution":  "{:.2%}",
        })
        .map(
            lambda v: "color: #1e7e34; font-weight:600" if isinstance(v, float) and v >= 0
                      else ("color: #c62828; font-weight:600" if isinstance(v, float) and v < 0 else ""),
            subset=["Stock Return", "Contribution"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Downloads ────────────────────────────────────────
    st.markdown("---")
    dl1, dl2 = st.columns(2)
    with dl1:
        csv_port = port_val.reset_index()
        csv_port.columns = ["Date", "Portfolio Value (₹)"]
        st.download_button(
            "📥 Download Portfolio Values (CSV)",
            data=csv_port.to_csv(index=False).encode(),
            file_name=f"ArthaPulse_portfolio_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
    with dl2:
        st.download_button(
            "📥 Download Holdings (CSV)",
            data=holdings_df.to_csv(index=False).encode(),
            file_name=f"ArthaPulse_holdings_{start_date}_{end_date}.csv",
            mime="text/csv",
        )

else:
    # Landing state
    st.info("👈 Configure your portfolio in the sidebar and click **Run Backtest** to see results.")

    with st.expander("💡 Quick Start Guide"):
        st.markdown("""
        1. **Select stocks** – type to search any NSE EQ-series stock.
        2. **Choose investment mode** – Lump Sum (one-time) or SIP (monthly instalments).
        3. **Set allocation** – equal weight or custom percentages (must sum to 100 %).
        4. **Adjust risk-free rate** – used for Sharpe, Sortino, and Jensen's Alpha.
        5. **Compare with Nifty 50** – optional benchmark overlay.
        6. **Run Backtest** → see equity curve, drawdown, rolling metrics, contribution breakdown, and holdings table.
        """)

    with st.expander("📖 Metric Glossary"):
        st.markdown("""
        | Metric | What it means |
        |---|---|
        | **CAGR** | Compound Annual Growth Rate – annualised return |
        | **Sharpe Ratio** | Excess return over risk-free rate per unit of total risk |
        | **Sortino Ratio** | Like Sharpe, but only penalises *downside* volatility |
        | **Calmar Ratio** | CAGR divided by absolute Max Drawdown |
        | **Max Drawdown** | Largest peak-to-trough decline |
        | **Beta** | Portfolio sensitivity to Nifty 50 moves |
        | **Alpha (ann.)** | Jensen's Alpha – excess return beyond what Beta predicts |
        | **Tracking Error** | Std dev of portfolio return minus benchmark return (annualised) |
        | **Information Ratio** | Active return per unit of tracking error |
        | **Contribution** | Each stock's weighted impact on total portfolio return |
        """)