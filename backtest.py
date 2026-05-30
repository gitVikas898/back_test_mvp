"""
backtest.py  –  Pure business logic for ArthaPulse.
No Streamlit imports here; fully testable in isolation.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
import yfinance as yf
 
 
# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
 
def download_stock_data(
    tickers: list[str],
    start_date,
    end_date,
    warn_callback=None,          # optional callable(str) for UI warnings
) -> tuple[pd.DataFrame, list[str]]:
    """
    Download each ticker individually.
 
    Returns
    -------
    prices : pd.DataFrame  (Date index, ticker columns, float values)
    failed : list[str]     tickers that could not be downloaded
    """
    prices: dict[str, pd.Series] = {}
    failed: list[str] = []
 
    for t in tickers:
        try:
            symbol = f"{t}.NS"
            data = yf.download(
                symbol, start=start_date, end=end_date,
                progress=False, auto_adjust=True,
            )
            if data.empty:
                failed.append(t)
                continue
 
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze()
            close.name = t
            prices[t] = close
 
        except Exception as exc:
            failed.append(t)
            if warn_callback:
                warn_callback(f"Skipped {t}: {exc}")
 
    if not prices:
        return pd.DataFrame(columns=tickers), failed
 
    df = pd.DataFrame(prices)
    df.index = pd.to_datetime(df.index)
    # bfill first so stocks listed mid-range don't pull everything to NaN
    df = df.sort_index().bfill().ffill().dropna(how="all")
    df = df[[t for t in tickers if t in df.columns]]
    return df, failed
 
 
def download_nifty50(start_date, end_date) -> pd.Series:
    """Download Nifty 50, return a clean float Series."""
    data = yf.download(
        "^NSEI", start=start_date, end=end_date,
        progress=False, auto_adjust=True,
    )
    if data.empty:
        return pd.Series(dtype=float)
 
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()
    close.name = "NIFTY50"
    return close
 
 
# ──────────────────────────────────────────────
# Core backtest
# ──────────────────────────────────────────────
 
def run_backtest(
    prices: pd.DataFrame,
    initial_capital: float,
    allocations: dict[str, float],
    risk_free_rate: float = 0.065,   # annual, e.g. 6.5 %
) -> tuple[pd.Series, pd.Series, dict, dict]:
    """
    Run a buy-and-hold backtest.
 
    Returns
    -------
    portfolio_value : pd.Series
    drawdown        : pd.Series  (negative fractions)
    metrics         : dict
    shares          : dict  {ticker: float}
    """
    first_prices = prices.iloc[0]
    last_prices  = prices.iloc[-1]
 
    shares: dict[str, float] = {
        t: (initial_capital * alloc) / first_prices[t]
        for t, alloc in allocations.items()
    }
 
    portfolio_value: pd.Series = prices.multiply(pd.Series(shares)).sum(axis=1)
 
    final_value  = portfolio_value.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    years        = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr         = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
 
    daily_returns   = portfolio_value.pct_change().dropna()
    ann_vol         = daily_returns.std() * np.sqrt(252)
 
    rf_daily = risk_free_rate / 252
    excess   = daily_returns - rf_daily
    sharpe   = (excess.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
 
    # Sortino  – downside deviation only
    downside    = daily_returns[daily_returns < rf_daily]
    down_std    = downside.std() if len(downside) > 1 else np.nan
    sortino     = (excess.mean() / down_std) * np.sqrt(252) if (down_std and down_std != 0) else 0.0
 
    cummax   = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_dd   = drawdown.min()
    max_dd_date = drawdown.idxmin()
 
    # Calmar  = CAGR / |Max Drawdown|
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
 
    # Per-stock contribution  = weight × stock return
    contributions = {
        t: allocations[t] * ((last_prices[t] / first_prices[t]) - 1)
        for t in allocations
    }
 
    metrics = {
        "Final Value":          final_value,
        "Total Return":         total_return,
        "CAGR":                 cagr,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio":         sharpe,
        "Sortino Ratio":        sortino,
        "Calmar Ratio":         calmar,
        "Max Drawdown":         max_dd,
        "Max DD Date":          max_dd_date,
        "Contributions":        contributions,
    }
    return portfolio_value, drawdown, metrics, shares
 
 
# ──────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────
 
def calculate_benchmark_equity(
    benchmark_prices: pd.Series,
    initial_capital: float,
) -> pd.Series:
    if benchmark_prices.empty:
        return pd.Series(dtype=float)
    units = initial_capital / benchmark_prices.iloc[0]
    return benchmark_prices * units
 
 
def compute_comparison_metrics(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    risk_free_rate: float = 0.065,
) -> dict:
    """Compute Alpha, Beta, Tracking Error, Information Ratio."""
    port_returns  = port_returns.squeeze()
    bench_returns = bench_returns.squeeze()
 
    common = port_returns.index.intersection(bench_returns.index)
    p = port_returns[common]
    b = bench_returns[common]
 
    if len(p) < 2:
        return {}
 
    rf_daily = risk_free_rate / 252
    cov_matrix = np.cov(p, b)
    cov  = cov_matrix[0, 1]
    var  = np.var(b, ddof=1)
    beta = cov / var if var != 0 else 0.0
 
    # Jensen's Alpha (annualised)
    alpha = ((p.mean() - rf_daily) - beta * (b.mean() - rf_daily)) * 252
 
    diff         = p - b
    tracking_err = diff.std() * np.sqrt(252)
    info_ratio   = ((p.mean() - b.mean()) / diff.std()) * np.sqrt(252) if diff.std() != 0 else 0.0
 
    return {
        "Beta":              beta,
        "Alpha (ann.)":      alpha,
        "Tracking Error":    tracking_err,
        "Information Ratio": info_ratio,
    }
 
 
# ──────────────────────────────────────────────
# Rolling metrics
# ──────────────────────────────────────────────
 
def rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 90,
    risk_free_rate: float = 0.065,
) -> pd.Series:
    rf_daily = risk_free_rate / 252
    excess   = daily_returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std  = daily_returns.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(252)
 
 
def rolling_volatility(
    daily_returns: pd.Series,
    window: int = 90,
) -> pd.Series:
    return daily_returns.rolling(window).std() * np.sqrt(252)
 
 
# ──────────────────────────────────────────────
# SIP simulation
# ──────────────────────────────────────────────
 
def run_sip_backtest(
    prices: pd.DataFrame,
    monthly_investment: float,
    allocations: dict[str, float],
    risk_free_rate: float = 0.065,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Simulate a monthly SIP (Systematic Investment Plan).
 
    On the first trading day of each month, invest `monthly_investment`
    split according to `allocations`.
 
    Returns
    -------
    portfolio_value : pd.Series
    drawdown        : pd.Series
    metrics         : dict
    """
    # Identify first trading day of each month
    monthly_idx = prices.resample("MS").first().index
    monthly_dates = [
        prices.index[prices.index >= m][0]
        for m in monthly_idx
        if len(prices.index[prices.index >= m]) > 0
    ]
 
    shares: dict[str, float] = {t: 0.0 for t in allocations}
    total_invested = 0.0
 
    for invest_date in monthly_dates:
        row = prices.loc[invest_date]
        for t, alloc in allocations.items():
            capital = monthly_investment * alloc
            shares[t] += capital / row[t]
        total_invested += monthly_investment
 
    portfolio_value: pd.Series = prices.multiply(pd.Series(shares)).sum(axis=1)
 
    final_value  = portfolio_value.iloc[-1]
    total_return = (final_value / total_invested) - 1
    years        = (prices.index[-1] - prices.index[0]).days / 365.25
    # XIRR approximation via CAGR on invested capital
    cagr         = (final_value / total_invested) ** (1 / years) - 1 if years > 0 else 0.0
 
    daily_returns   = portfolio_value.pct_change().dropna()
    ann_vol         = daily_returns.std() * np.sqrt(252)
 
    rf_daily = risk_free_rate / 252
    excess   = daily_returns - rf_daily
    sharpe   = (excess.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
 
    downside = daily_returns[daily_returns < rf_daily]
    down_std = downside.std() if len(downside) > 1 else np.nan
    sortino  = (excess.mean() / down_std) * np.sqrt(252) if (down_std and down_std != 0) else 0.0
 
    cummax   = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_dd   = drawdown.min()
    max_dd_date = drawdown.idxmin()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0.0
 
    metrics = {
        "Total Invested":       total_invested,
        "Final Value":          final_value,
        "Total Return":         total_return,
        "CAGR":                 cagr,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio":         sharpe,
        "Sortino Ratio":        sortino,
        "Calmar Ratio":         calmar,
        "Max Drawdown":         max_dd,
        "Max DD Date":          max_dd_date,
        "SIP Instalments":      len(monthly_dates),
    }
    return portfolio_value, drawdown, metrics