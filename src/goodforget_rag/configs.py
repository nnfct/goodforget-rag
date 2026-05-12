"""Configuration loading for benchmark scripts."""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    num_domains: int
    scenarios_per_domain: int
    documents_per_scenario: int
    query_variants_per_scenario: int
    seeds: list[int]
    top_k: int
    candidate_k: int
    representations: list[str]
    alpha: float
    beta: float
    gamma: float
    forget_threshold: float
    bootstrap_samples: int


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a simple YAML subset without adding a PyYAML dependency."""

    values: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, raw_value = line.split(":", 1)
        values[key.strip()] = _parse_value(raw_value.strip())
    return BenchmarkConfig(
        mode=str(values["mode"]),
        num_domains=int(values["num_domains"]),
        scenarios_per_domain=int(values["scenarios_per_domain"]),
        documents_per_scenario=int(values["documents_per_scenario"]),
        query_variants_per_scenario=int(values["query_variants_per_scenario"]),
        seeds=[int(seed) for seed in values["seeds"]],
        top_k=int(values["top_k"]),
        candidate_k=int(values["candidate_k"]),
        representations=[str(item) for item in values["representations"]],
        alpha=float(values["alpha"]),
        beta=float(values["beta"]),
        gamma=float(values["gamma"]),
        forget_threshold=float(values["forget_threshold"]),
        bootstrap_samples=int(values["bootstrap_samples"]),
    )


def _parse_value(value: str) -> object:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")
