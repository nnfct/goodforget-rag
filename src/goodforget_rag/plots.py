"""Matplotlib chart generation for validation reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_plots(results_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results_dir / "benchmark_summary.csv")
    by_split = pd.read_csv(results_dir / "benchmark_by_split.csv")
    by_attack = pd.read_csv(results_dir / "benchmark_by_attack.csv")
    label_noise = pd.read_csv(results_dir / "label_noise_summary.csv")
    ablation = pd.read_csv(results_dir / "ablation_summary.csv")
    representation = summary.groupby("representation", as_index=False).agg(
        leakage_rate=("leakage_rate", "mean"),
        utility_recall=("utility_recall", "mean"),
    )

    _bar(summary, "method", "leakage_rate", figures_dir / "leakage_rate_by_method.png")
    _bar(summary, "method", "utility_recall", figures_dir / "utility_recall_by_method.png")
    _scatter(summary, figures_dir / "leakage_vs_utility.png")
    _bar(summary, "method", "over_filter_rate", figures_dir / "over_filter_rate_by_method.png")
    _bar(summary, "method", "leakage_utility_tradeoff_score", figures_dir / "tradeoff_score_by_method.png")
    _bar(by_split, "split", "leakage_rate", figures_dir / "leakage_rate_by_split.png")
    _bar(by_attack, "attack_type", "leakage_rate", figures_dir / "leakage_rate_by_attack_type.png")
    _line(label_noise, "noise_setting", "leakage_rate", figures_dir / "label_noise_impact.png")
    _line(ablation[ablation["parameter"] == "gamma"], "value", "leakage_rate", figures_dir / "gamma_sensitivity_curve.png")
    _bar(representation, "representation", "leakage_rate", figures_dir / "representation_comparison.png")


def _bar(frame: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    plot_frame = frame.groupby(x_col, as_index=False)[y_col].mean().sort_values(y_col)
    plt.figure(figsize=(9, 4.8))
    plt.bar(plot_frame[x_col].astype(str), plot_frame[y_col])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _line(frame: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    plot_frame = frame.groupby(x_col, as_index=False)[y_col].mean()
    plt.figure(figsize=(8, 4.5))
    plt.plot(plot_frame[x_col].astype(str), plot_frame[y_col], marker="o")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _scatter(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(frame["leakage_rate"], frame["utility_recall"])
    for _, row in frame.iterrows():
        plt.annotate(str(row["method"])[:18], (row["leakage_rate"], row["utility_recall"]), fontsize=7)
    plt.xlabel("leakage_rate")
    plt.ylabel("utility_recall")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
