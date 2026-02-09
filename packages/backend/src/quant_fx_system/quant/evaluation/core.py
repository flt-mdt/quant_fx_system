from __future__ import annotations

from dataclasses import asdict
import math

import pandas as pd

from .drawdowns import compute_drawdowns, compute_equity
from .distribution import compute_distribution_metrics
from .performance import compute_performance_metrics
from .regression import compute_benchmark_metrics
from .rolling import compute_rolling_metrics
from .robustness import (
    deflated_sharpe_ratio,
    hac_adjusted_sharpe,
    probabilistic_sharpe_ratio,
    stability_from_rolling,
)
from .types import EvaluationConfig, EvaluationResult
from .validation import align_series, infer_periods_per_year, validate_utc_series


def evaluate_strategy(
    *,
    returns: pd.Series,
    equity: pd.Series | None = None,
    pnl: pd.Series | None = None,
    position: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    cfg: EvaluationConfig | None = None,
) -> EvaluationResult:
    config = cfg or EvaluationConfig()

    returns = validate_utc_series(returns, "returns", allow_nans=True)
    if equity is not None:
        equity = validate_utc_series(equity, "equity")
    if pnl is not None:
        pnl = validate_utc_series(pnl, "pnl")
    if position is not None:
        position = validate_utc_series(position, "position")
    if turnover is not None:
        turnover = validate_utc_series(turnover, "turnover")
    if costs is not None:
        costs = validate_utc_series(costs, "costs")
    if benchmark_returns is not None:
        benchmark_returns = validate_utc_series(benchmark_returns, "benchmark_returns")

    aligned = align_series(
        returns,
        equity,
        pnl,
        position,
        turnover,
        costs,
        benchmark_returns,
    )
    returns, equity, pnl, position, turnover, costs, benchmark_returns = aligned

    returns = returns.dropna()
    if equity is not None:
        equity = equity.reindex(returns.index)
        if (equity <= 0).any():
            raise ValueError("equity must be strictly positive.")
    if pnl is not None:
        pnl = pnl.reindex(returns.index)
    if position is not None:
        position = position.reindex(returns.index)
    if turnover is not None:
        turnover = turnover.reindex(returns.index)
    if costs is not None:
        costs = costs.reindex(returns.index)
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.reindex(returns.index)

    periods_per_year: int | None = None
    if config.annualization == "auto":
        periods_per_year = config.periods_per_year_override or infer_periods_per_year(returns.index)

    summary: dict[str, float] = {
        "n_obs": float(len(returns)),
    }
    tables: dict[str, pd.DataFrame] = {}
    series: dict[str, pd.Series] = {}

    perf_metrics = compute_performance_metrics(
        returns,
        equity=equity,
        position=position,
        periods_per_year=periods_per_year,
        risk_free_rate=config.risk_free_rate,
    )
    summary.update(perf_metrics)

    if config.drawdown_enabled:
        equity_curve = compute_equity(returns, equity)
        underwater, max_drawdown, max_tuw, ulcer = compute_drawdowns(equity_curve)
        series["equity"] = equity_curve
        series["underwater"] = underwater
        summary["max_drawdown"] = max_drawdown
        summary["tuw_max"] = float(max_tuw)
        summary["ulcer_index"] = ulcer
        cagr = summary.get("cagr")
        if cagr is not None and not math.isnan(cagr) and max_drawdown > 0:
            summary["calmar"] = cagr / max_drawdown

    if config.distribution_enabled:
        dist_summary, dist_tables = compute_distribution_metrics(returns)
        summary.update(dist_summary)
        tables.update(dist_tables)

    if config.regression_enabled and benchmark_returns is not None:
        summary.update(
            compute_benchmark_metrics(
                returns,
                benchmark_returns,
                periods_per_year=periods_per_year,
            )
        )

    rolling_df = compute_rolling_metrics(
        returns,
        windows=config.rolling_windows,
        periods_per_year=periods_per_year,
        risk_free_rate=config.risk_free_rate,
    )
    if not rolling_df.empty:
        tables["rolling_metrics"] = rolling_df

    if config.robustness_enabled:
        sharpe_value = summary.get("sharpe", float("nan"))
        skew = summary.get("skew", float("nan"))
        kurt = summary.get("kurtosis", float("nan"))
        n_obs = len(returns)
        if config.psr_enabled:
            summary["psr_0"] = probabilistic_sharpe_ratio(sharpe_value, n_obs, skew, kurt, 0.0)
        if config.dsr_enabled:
            summary["dsr_0"] = deflated_sharpe_ratio(
                sharpe_value, n_obs, skew, kurt, config.n_trials
            )
        summary["sharpe_hac"] = hac_adjusted_sharpe(returns, sharpe_value)

        if not rolling_df.empty:
            longest_window = max(config.rolling_windows)
            col = f"sr_roll_{longest_window}"
            if col in rolling_df.columns:
                summary.update(stability_from_rolling(rolling_df[col]))

    summary["start"] = returns.index.min().isoformat() if len(returns) else ""
    summary["end"] = returns.index.max().isoformat() if len(returns) else ""

    metadata = {
        "config": asdict(config),
        "periods_per_year": periods_per_year,
        "index_start": summary["start"],
        "index_end": summary["end"],
    }
    if config.annualization == "none":
        metadata["warnings"] = ["annualization_disabled", "vol_annual_not_annualized"]

    return EvaluationResult(summary=summary, tables=tables, series=series, metadata=metadata)
