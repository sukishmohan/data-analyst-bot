"""
forecaster.py — Sales Forecasting Module

Provides simple time-series forecasting using:
  • Linear trend extrapolation (always available)
  • Holt-Winters exponential smoothing (if statsmodels is installed)

Designed to be called by the main orchestrator when the planner
detects a forecast-type query.
"""

import pandas as pd
import numpy as np
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_logger
from visualization import apply_style, COLORS, save_figure

log = get_logger("forecaster")


def forecast_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 6,
    freq: str = "MS",
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Forecast a time series and return both the forecast DataFrame
    and a chart figure.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing date and value columns.
    date_col : str
        Name of the date column.
    value_col : str
        Name of the numeric column to forecast.
    periods : int
        Number of future periods to predict.
    freq : str
        Pandas frequency alias (MS=month start, QS=quarter start, etc.).

    Returns
    -------
    tuple of (forecast_df, fig)
        forecast_df has columns: Date, Actual, Forecast
    """
    # Prepare monthly aggregation
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    ts = ts.set_index(date_col).resample(freq)[value_col].sum().reset_index()
    ts.columns = ["Date", "Actual"]

    # Try Holt-Winters first, fall back to linear trend
    forecast_values = _holt_winters_forecast(ts, periods, freq)
    method = "Holt-Winters"

    if forecast_values is None:
        forecast_values = _linear_forecast(ts, periods)
        method = "Linear Trend"

    # Build future dates
    last_date = ts["Date"].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1,
                                 freq=freq)[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_values,
    })

    # Combine for plotting
    combined = pd.concat([
        ts.assign(Forecast=np.nan),
        forecast_df.assign(Actual=np.nan),
    ], ignore_index=True)

    # Plot
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts["Date"], ts["Actual"], marker="o", markersize=3,
            color=COLORS[0], linewidth=2, label="Actual")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], marker="s",
            markersize=4, color=COLORS[1], linewidth=2, linestyle="--",
            label=f"Forecast ({method})")
    ax.fill_between(forecast_df["Date"],
                     forecast_df["Forecast"] * 0.9,
                     forecast_df["Forecast"] * 1.1,
                     alpha=0.15, color=COLORS[1], label="±10% Confidence")
    ax.set_title(f"{value_col} Forecast — Next {periods} Periods", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.legend()
    plt.tight_layout()

    log.info("Forecast generated using %s for %d periods.", method, periods)
    return forecast_df, fig


def _holt_winters_forecast(
    ts: pd.DataFrame, periods: int, freq: str
) -> Optional[np.ndarray]:
    """Attempt Holt-Winters exponential smoothing."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            ts["Actual"].values,
            trend="add",
            seasonal="add",
            seasonal_periods=min(12, len(ts) // 2),
        ).fit(optimized=True)

        return model.forecast(periods)
    except Exception as exc:
        log.info("Holt-Winters unavailable (%s), using linear fallback.", exc)
        return None


def _linear_forecast(ts: pd.DataFrame, periods: int) -> np.ndarray:
    """Simple linear trend extrapolation."""
    y = ts["Actual"].values
    x = np.arange(len(y))

    # Fit linear regression
    coeffs = np.polyfit(x, y, deg=1)
    future_x = np.arange(len(y), len(y) + periods)
    return np.polyval(coeffs, future_x)
