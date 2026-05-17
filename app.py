import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# ------------------------------
# Page config - Name & Quote
# ------------------------------
st.set_page_config(
    page_title="ArthaPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        background: #1f77b4;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background: #1565a9;
    }
    .quote {
        font-style: italic;
        color: #555;
        font-size: 1rem;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#e6f0fa, #ffffff);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header with quote
# ------------------------------
st.title("📊 ArthaPulse — Indian Portfolio Backtester")
st.markdown('<p class="quote">"Know your past portfolio to craft your future wealth." — Ancient Wisdom</p>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------
# Cached data loading functions
# ------------------------------
@st.cache_data(ttl=3600)
def load_nse_symbols():
    """Download the official NSE equity list."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    symbols = df[df[' SERIES'] == 'EQ']['SYMBOL'].str.strip().tolist()
    symbols = sorted([s.strip() for s in symbols if s.strip()])
    return symbols

@st.cache_data(ttl=300)
def download_stock_data(tickers, start_date, end_date):
    """Download adjusted close prices for given tickers (NSE)."""
    yf_tickers = [f"{t}.NS" for t in tickers]
    data = yf.download(yf_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if data.empty:
        return pd.DataFrame()
    prices = data["Close"].copy() if "Close" in data.columns else pd.DataFrame()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices.ffill(inplace=True)
    prices.dropna(how="all", inplace=True)
    prices.columns = tickers  # Remove .NS suffix
    return prices

@st.cache_data(ttl=300)
def download_nifty50(start_date, end_date):
    """Download Nifty 50 index data."""
    data = yf.download("^NSEI", start=start_date, end=end_date, progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float)
    return data["Close"]

# ------------------------------
# Backtest calculation
# ------------------------------
def run_backtest(prices, initial_capital, allocations):
    """Return portfolio value series and metrics."""
    first_prices = prices.iloc[0]
    shares = {}
    for ticker, alloc in allocations.items():
        capital = initial_capital * alloc
        shares[ticker] = capital / first_prices[ticker]
    
    portfolio_value = prices.multiply(pd.Series(shares)).sum(axis=1)
    
    # Metrics
    final_value = portfolio_value.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25
    cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    daily_returns = portfolio_value.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    metrics = {
        "Final Value": final_value,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Max DD Date": max_dd_date,
    }
    return portfolio_value, drawdown, metrics, shares

# ------------------------------
# Benchmark comparison functions
# ------------------------------
def calculate_benchmark_equity(benchmark_prices, initial_capital):
    """Calculate benchmark equity curve if invested on day 1."""
    if benchmark_prices.empty:
        return pd.Series(dtype=float)
    start_price = benchmark_prices.iloc[0]
    units = initial_capital / start_price
    return benchmark_prices * units

def compute_comparison_metrics(port_returns, bench_returns):
    """Compute alpha, beta, tracking error, information ratio."""
    common_idx = port_returns.index.intersection(bench_returns.index)
    port = port_returns[common_idx]
    bench = bench_returns[common_idx]
    if len(port) < 2:
        return {}
    
    cov = np.cov(port, bench)[0, 1]
    var = np.var(bench)
    beta = cov / var if var != 0 else 0
    alpha = (port.mean() - beta * bench.mean()) * 252  # annualised
    tracking_err = (port - bench).std() * np.sqrt(252)
    info_ratio = (port.mean() - bench.mean()) / (port - bench).std() * np.sqrt(252) if tracking_err != 0 else 0
    
    return {
        "Beta": beta,
        "Alpha (ann.)": alpha,
        "Tracking Error": tracking_err,
        "Information Ratio": info_ratio,
    }

# ------------------------------
# Sidebar - Configuration
# ------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/stock-share.png", width=48)  # Optional: replace with a local icon
    st.header("⚙️ Configuration")
    
    # Load symbols (cached hourly)
    all_symbols = load_nse_symbols()
    selected_tickers = st.multiselect(
        "🔍 Select stocks",
        options=all_symbols,
        default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        help="Start typing to search for any NSE stock"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", date.today() - timedelta(days=365*3))
    with col2:
        end_date = st.date_input("End date", date.today())
    
    initial_capital = st.number_input(
        "💰 Total investment (₹)",
        min_value=1000, value=100000, step=5000,
        help="Initial capital deployed equally or as per allocation"
    )
    
    # Benchmark toggle
    compare_benchmark = st.checkbox("📈 Compare with Nifty 50", value=True,
                                    help="Overlay Nifty 50 performance on charts")
    
    # Allocation method
    st.subheader("🎯 Allocation Method")
    alloc_method = st.radio("Choose allocation", ("Equal weight", "Custom %"), horizontal=True)
    
    allocations = {}
    if alloc_method == "Equal weight" and selected_tickers:
        allocations = {t: 1/len(selected_tickers) for t in selected_tickers}
    elif alloc_method == "Custom %" and selected_tickers:
        st.markdown("##### Set target allocation (%)")
        cols = st.columns(2)
        rem = 100
        tickers_with_errors = []
        for i, ticker in enumerate(selected_tickers):
            with cols[i % 2]:
                if i < len(selected_tickers) - 1:
                    default_val = rem // (len(selected_tickers) - i)
                    alloc = st.slider(
                        ticker, 0, rem,
                        value=default_val,
                        key=f"alloc_{ticker}"
                    )
                    allocations[ticker] = alloc / 100
                    rem -= alloc
                else:
                    allocations[ticker] = rem / 100
                    st.markdown(f"**{ticker}:** {rem}%")
    
    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ------------------------------
# Main area - Display results
# ------------------------------
if run_button and selected_tickers and start_date < end_date:
    with st.spinner("⏳ Downloading market data & computing..."):
        try:
            # Download portfolio data
            prices = download_stock_data(selected_tickers, start_date, end_date)
            if prices.empty:
                st.error("No data found for selected tickers or date range. Try different symbols.")
                st.stop()
            
            # Run backtest
            port_val, dd, metrics, shares = run_backtest(prices, initial_capital, allocations)
            
            # Download benchmark if requested
            bench_equity = None
            bench_metrics = None
            if compare_benchmark:
                bench_prices = download_nifty50(start_date, end_date)
                if not bench_prices.empty:
                    bench_equity = calculate_benchmark_equity(bench_prices, initial_capital)
                    # Align dates
                    common_dates = port_val.index.intersection(bench_equity.index)
                    if len(common_dates) > 1:
                        port_returns = port_val.pct_change().dropna()
                        bench_returns = bench_equity.pct_change().dropna()
                        bench_metrics = compute_comparison_metrics(port_returns, bench_returns)
                    else:
                        st.warning("Not enough overlapping data with Nifty 50 for comparison.")
            
            # -------- Snapshot cards --------
            st.subheader("📌 Portfolio Snapshot")
            cols = st.columns(4)
            with cols[0]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Final Value", f"₹{metrics['Final Value']:,.2f}")
                st.caption("vs ₹{:,} invested".format(initial_capital))
                st.markdown('</div>', unsafe_allow_html=True)
            with cols[1]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Return", f"{metrics['Total Return']:.2%}")
                st.caption(f"Over {((end_date - start_date).days / 365.25):.1f} years")
                st.markdown('</div>', unsafe_allow_html=True)
            with cols[2]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("CAGR", f"{metrics['CAGR']:.2%}")
                st.caption("Annualised growth rate")
                st.markdown('</div>', unsafe_allow_html=True)
            with cols[3]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                st.caption(f"On {metrics['Max DD Date'].strftime('%d-%b-%Y')}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            cols = st.columns(4)
            with cols[0]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                st.caption("Risk-adjusted return")
                st.markdown('</div>', unsafe_allow_html=True)
            with cols[1]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Volatility (ann.)", f"{metrics['Annualised Volatility']:.2%}")
                st.caption("Standard deviation of returns")
                st.markdown('</div>', unsafe_allow_html=True)
            with cols[2]:
                if bench_metrics and "Alpha (ann.)" in bench_metrics:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Alpha (ann.)", f"{bench_metrics['Alpha (ann.)']:.2%}")
                    st.caption("Excess return over benchmark")
                    st.markdown('</div>', unsafe_allow_html=True)
            with cols[3]:
                if bench_metrics and "Beta" in bench_metrics:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Beta", f"{bench_metrics['Beta']:.2f}")
                    st.caption("Sensitivity to market")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # -------- Charts --------
            st.subheader("📈 Performance Charts")
            
            # Combined equity curve with benchmark
            fig1 = make_subplots(specs=[[{"secondary_y": False}]])
            fig1.add_trace(
                go.Scatter(x=port_val.index, y=port_val, mode='lines',
                           name='Your Portfolio', line=dict(color='#1f77b4', width=2))
            )
            if bench_equity is not None and not bench_equity.empty:
                fig1.add_trace(
                    go.Scatter(x=bench_equity.index, y=bench_equity, mode='lines',
                               name='Nifty 50', line=dict(color='#ff7f0e', width=2, dash='dash'))
                )
            fig1.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Value (₹)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40),
                template="plotly_white"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Drawdown chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=dd.index, y=dd*100,
                fill='tozeroy', mode='lines',
                name='Drawdown',
                line=dict(color='#d62728', width=1),
                fillcolor='rgba(214,39,40,0.2)'
            ))
            fig2.update_layout(
                title="Drawdown (%)",
                xaxis_title="Date",
                yaxis_title="Drawdown %",
                yaxis=dict(ticksuffix="%"),
                template="plotly_white"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # -------- Benchmark stats --------
            if bench_metrics:
                st.subheader("📉 Benchmark Comparison (vs Nifty 50)")
                bcols = st.columns(4)
                for idx, (key, val) in enumerate(bench_metrics.items()):
                    with bcols[idx]:
                        st.metric(key, f"{val:.2%}" if isinstance(val, float) else val)
            
            # -------- Holdings breakdown --------
            st.subheader("🧾 Portfolio Holdings")
            shares_df = pd.DataFrame({
                "Ticker": list(shares.keys()),
                "Shares": [round(shares[t], 4) for t in shares],
                "Allocation %": [f"{allocations[t]:.1%}" for t in shares],
                "Initial Price": [prices.iloc[0][t] for t in shares],
                "Current Price": [prices.iloc[-1][t] for t in shares],
                "Stock Return": [f"{(prices.iloc[-1][t] / prices.iloc[0][t] - 1):.2%}" for t in shares]
            })
            st.dataframe(
                shares_df.style.format({"Initial Price": "₹{:,.2f}", "Current Price": "₹{:,.2f}"}),
                use_container_width=True, hide_index=True
            )
            
            # Download option
            csv = port_val.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Portfolio Values (CSV)",
                data=csv,
                file_name=f"ArthaPulse_{start_date}_{end_date}.csv",
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
elif run_button:
    st.warning("⚠️ Please select at least one stock and ensure start date is before end date.")
else:
    # Welcome screen
    st.info("👈 Configure your portfolio in the sidebar and click **Run Backtest** to see results.")
    with st.expander("💡 Quick Start Guide"):
        st.markdown("""
        1. **Select stocks** – search and pick NSE equities.
        2. **Set investment amount** and choose equal or custom weights.
        3. **Compare with Nifty 50** (optional).
        4. **Run backtest** to see performance, risk metrics, and charts.
        """)