"""HTML and Markdown reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def html_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted.to_html(index=False, escape=True, classes="data-table")


def write_validation_report(results_dir: Path, docs_dir: Path) -> Path:
    summary = pd.read_csv(results_dir / "benchmark_summary.csv")
    by_split = pd.read_csv(results_dir / "benchmark_by_split.csv")
    by_attack = pd.read_csv(results_dir / "benchmark_by_attack.csv")
    by_domain = pd.read_csv(results_dir / "benchmark_by_domain.csv")
    ablation = pd.read_csv(results_dir / "ablation_summary.csv")
    label_noise = pd.read_csv(results_dir / "label_noise_summary.csv")
    negative = pd.read_csv(results_dir / "negative_vector_baseline.csv")
    ci = pd.read_csv(results_dir / "bootstrap_confidence_intervals.csv")
    failures = pd.read_csv(results_dir / "failure_cases.csv")
    seed = pd.read_csv(results_dir / "seed_stability_summary.csv")
    docs_dir.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GoodForget-RAG Validation Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f7f8fa; color: #18212f; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 56px; }}
    h1 {{ margin: 0 0 8px; }}
    h2 {{ margin-top: 30px; }}
    p, li {{ color: #52606d; line-height: 1.55; }}
    .notice {{ background: #fff7ed; border-left: 4px solid #c2410c; padding: 12px; border-radius: 0 8px 8px 0; }}
    .ok {{ background: #ecfdf5; border-left: 4px solid #047857; padding: 12px; border-radius: 0 8px 8px 0; }}
    .table-wrap {{ overflow-x: auto; background: white; border: 1px solid #d8dee8; border-radius: 8px; }}
    table.data-table {{ border-collapse: collapse; min-width: 900px; width: 100%; font-size: 13px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #d8dee8; padding: 8px 10px; text-align: left; }}
    .data-table th {{ background: #eef2f7; }}
  </style>
</head>
<body>
<main>
  <h1>GoodForget-RAG Validation Benchmark Report</h1>
  <div class="notice">
    This is synthetic. It is not model unlearning, not a safety guarantee, and not proof of semantic forgetting.
    GoodForget-RAG may tie or lose to simpler baselines. Metadata filtering is better when reliable labels exist.
  </div>
  <h2>Executive Summary</h2>
  <p>This benchmark studies retrieval-time control tradeoffs across synthetic domains, attacks, representations, label noise, ablations, and failure cases. The convenience tradeoff score is utility_recall - leakage_rate - 0.5 * over_filter_rate, not a universal metric.</p>
  <h2>Key Takeaways</h2>
  <ul>
    <li>GoodForget-RAG is useful mainly when labels are incomplete, dynamic, or too coarse.</li>
    <li>Keyword and metadata baselines can win on literal or well-labeled cases.</li>
    <li>Codeword, incomplete forget set, and mixed-evidence cases remain difficult.</li>
  </ul>
  <h2>Main Results</h2><div class="table-wrap">{html_table(summary)}</div>
  <h2>Results by Split</h2><div class="table-wrap">{html_table(by_split.head(80))}</div>
  <h2>Results by Attack Type</h2><div class="table-wrap">{html_table(by_attack.head(80))}</div>
  <h2>Results by Domain</h2><div class="table-wrap">{html_table(by_domain.head(80))}</div>
  <h2>Ablation Summary</h2><div class="table-wrap">{html_table(ablation.head(80))}</div>
  <h2>Label Noise Study</h2><div class="table-wrap">{html_table(label_noise)}</div>
  <h2>Forget-Set Noise Study</h2><p>Forget-set noise is represented through noisy_forget_set and incomplete_forget_set splits and the attack-type tables above.</p>
  <h2>Representation Sensitivity</h2><p>See the representation column in the main results table. Optional dense models are not required and are skipped unless already available.</p>
  <p>Optional dense representation status: not run in this report. The suite avoids runtime downloads and keeps dense models optional.</p>
  <h2>Span-Level Experiment</h2><p>Span-level GoodForget-RAG is reported in the main table when span-derived documents are evaluated.</p>
  <h2>Negative Vector Baseline</h2><div class="table-wrap">{html_table(negative)}</div>
  <h2>Bootstrap Confidence Intervals</h2><div class="table-wrap">{html_table(ci.head(120))}</div>
  <h2>Failure Cases</h2><div class="table-wrap">{html_table(failures.head(80))}</div>
  <h2>Seed Stability</h2><div class="table-wrap">{html_table(seed)}</div>
  <h2>Conservative Interpretation</h2>
  <div class="ok">Good forgetting is the ability to suppress forbidden evidence while preserving utility on adjacent, non-forbidden knowledge.</div>
  <p>The benchmark is exploratory. It should be used to find tradeoffs and failures, not to claim solved forgetting.</p>
  <h2>Recommended LinkedIn Framing</h2>
  <p>Frame this as a synthetic validation suite for retrieval-time forgetting controls, with explicit baselines and failure cases. Do not frame it as LLM unlearning or a safety guarantee.</p>
</main>
</body>
</html>
"""
    output = docs_dir / "validation_report.html"
    output.write_text(html, encoding="utf-8")
    (results_dir / "validation_report.html").write_text(html, encoding="utf-8")
    return output


def write_failure_markdown(results_dir: Path, docs_dir: Path) -> Path:
    failures = pd.read_csv(results_dir / "failure_cases.csv").head(30)
    lines = [
        "# Failure Cases",
        "",
        "These examples are mined automatically from synthetic benchmark runs. They are intended for debugging and red-team review, not as proof of real-world behavior.",
        "",
    ]
    for _, row in failures.iterrows():
        lines.append(f"## {row['query_id']} / {row['method']} / {row['representation']}")
        lines.append("")
        lines.append(f"- Domain: {row['domain']}")
        lines.append(f"- Split: {row['split']}")
        lines.append(f"- Attack type: {row['attack_type']}")
        lines.append(f"- Selected forbidden docs: {row['selected_forbidden_doc_ids']}")
        lines.append(f"- Missed expected docs: {row['missed_expected_doc_ids']}")
        lines.append(f"- Likely reason: {row['likely_failure_reason']}")
        lines.append("")
    output = docs_dir / "failure_cases.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
