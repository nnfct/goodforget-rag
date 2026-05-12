"""Statistical helpers for benchmark summaries."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


BOOTSTRAP_METRICS = [
    "leakage_rate",
    "utility_recall",
    "over_filter_rate",
    "leakage_utility_tradeoff_score",
]


def bootstrap_confidence_intervals(
    rows: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    samples: int,
    seed: int = 7,
) -> pd.DataFrame:
    """Bootstrap 95% CIs over query-level rows."""

    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for group_key, group in rows.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = {
            "leakage_rate": group["leakage"].astype(float).to_numpy(),
            "utility_recall": group["utility_recall"].astype(float).to_numpy(),
            "over_filter_rate": group["over_filter_rate"].astype(float).to_numpy(),
            "leakage_utility_tradeoff_score": group["tradeoff_score"].astype(float).to_numpy(),
        }
        n = len(group)
        for metric, metric_values in values.items():
            draws = []
            for _ in range(samples):
                indices = rng.integers(0, n, size=n)
                draws.append(float(np.mean(metric_values[indices])))
            row = dict(zip(group_cols, group_key))
            row.update(
                {
                    "metric": metric,
                    "mean": float(np.mean(metric_values)),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "num_queries": n,
                }
            )
            output.append(row)
    return pd.DataFrame(output)


def seed_stability(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize method stability across seeds."""

    seed_summary = (
        rows.groupby(["seed", "method"], as_index=False)
        .agg(
            leakage_rate=("leakage", "mean"),
            utility_recall=("utility_recall", "mean"),
            over_filter_rate=("over_filter_rate", "mean"),
            tradeoff_score=("tradeoff_score", "mean"),
        )
    )
    return (
        seed_summary.groupby("method", as_index=False)
        .agg(
            leakage_rate_mean=("leakage_rate", "mean"),
            leakage_rate_std=("leakage_rate", "std"),
            leakage_rate_min=("leakage_rate", "min"),
            leakage_rate_max=("leakage_rate", "max"),
            utility_recall_mean=("utility_recall", "mean"),
            utility_recall_std=("utility_recall", "std"),
            tradeoff_score_mean=("tradeoff_score", "mean"),
            tradeoff_score_std=("tradeoff_score", "std"),
            num_seeds=("seed", "count"),
        )
        .fillna(0.0)
    )
