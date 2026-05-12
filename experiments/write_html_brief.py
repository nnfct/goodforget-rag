"""Generate a standalone human-readable HTML experiment brief."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def _fmt(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted


def _table(frame: pd.DataFrame) -> str:
    return _fmt(frame).to_html(index=False, escape=True, classes="data-table")


def main() -> None:
    summary = _load("summary.csv")
    split_summary = _load("split_summary.csv")
    sensitivity = _load("sensitivity.csv")
    auto_intent = _load("auto_intent_summary.csv")
    representation = _load("representation_summary.csv")
    span = _load("span_summary.csv")

    best_sensitivity = sensitivity.sort_values(
        ["leakage_rate", "utility_recall", "safe_context_recall"],
        ascending=[True, False, False],
    ).head(5)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GoodForget-RAG Experiment Brief</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5f6c7b;
      --line: #d7dde5;
      --accent: #155e75;
      --risk: #9f1239;
      --ok: #047857;
      --warn: #b45309;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 30px 18px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 30px 0 12px;
      font-size: 21px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.55;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .card strong {{
      display: block;
      font-size: 24px;
      margin-top: 6px;
    }}
    .tag {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      color: #fff;
      background: var(--accent);
      margin-right: 6px;
    }}
    .tag.warn {{ background: var(--warn); }}
    .tag.ok {{ background: var(--ok); }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 760px;
    }}
    .data-table th, .data-table td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: #eef2f7;
      font-weight: 700;
      white-space: nowrap;
    }}
    .note {{
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
      background: #eef7fa;
      color: var(--ink);
      border-radius: 0 8px 8px 0;
    }}
    .risk {{
      border-left-color: var(--risk);
      background: #fff1f2;
    }}
    ul {{
      line-height: 1.6;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .summary-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 560px) {{
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>GoodForget-RAG Experiment Brief</h1>
    <p>Generated {generated_at}. This brief summarizes intermediate mitigation work for red-team limitations. It is a toy retrieval-control evaluation, not model unlearning or a safety guarantee.</p>

    <div>
      <span class="tag ok">local deterministic</span>
      <span class="tag">TF-IDF</span>
      <span class="tag">Local LSA sensitivity</span>
      <span class="tag">span-level check</span>
      <span class="tag warn">synthetic data</span>
    </div>

    <section class="summary-grid">
      <div class="card">Default GoodForget leakage<strong>{summary.loc[summary['method'] == 'GoodForget-RAG', 'leakage_rate'].iloc[0]:.3f}</strong></div>
      <div class="card">Default GoodForget utility<strong>{summary.loc[summary['method'] == 'GoodForget-RAG', 'utility_recall'].iloc[0]:.3f}</strong></div>
      <div class="card">Best sensitivity leakage<strong>{best_sensitivity['leakage_rate'].iloc[0]:.3f}</strong></div>
      <div class="card">Best sensitivity utility<strong>{best_sensitivity['utility_recall'].iloc[0]:.3f}</strong></div>
    </section>

    <h2>Goal Status</h2>
    <div class="note">
      Added three mitigation checks: heuristic intent extraction to reduce oracle dependence, span-level retrieval labels to probe document-level limitations, and Local LSA representation sensitivity to avoid relying on one raw TF-IDF result. These do not eliminate the limitations, but they make them measurable.
    </div>

    <h2>Main Retrieval Results</h2>
    <div class="table-wrap">{_table(summary)}</div>

    <h2>Split-Level Results</h2>
    <div class="table-wrap">{_table(split_summary)}</div>

    <h2>Intent Assumption Check</h2>
    <p>Compares oracle positive/forget intents with a simple deterministic heuristic extracted from the query.</p>
    <div class="table-wrap">{_table(auto_intent)}</div>

    <h2>Representation Sensitivity</h2>
    <p>Compares raw TF-IDF with a local TF-IDF + truncated SVD LSA proxy. This is still local and deterministic; it is not a downloaded dense embedding model.</p>
    <div class="table-wrap">{_table(representation)}</div>

    <h2>Span-Level Limitation Check</h2>
    <p>Splits documents into sentence-like spans and maps mixed-evidence labels to spans. This tests whether span-level retrieval can preserve safe text inside otherwise forbidden parent documents.</p>
    <div class="table-wrap">{_table(span)}</div>

    <h2>Best Sensitivity Settings</h2>
    <div class="table-wrap">{_table(best_sensitivity)}</div>

    <h2>Residual Limitations</h2>
    <div class="note risk">
      The project remains synthetic and small. Heuristic intents are weak, span labels are derived heuristically, and Local LSA is only a local representation sensitivity check. Full LLM answer leakage, real corpora, human labels, policy enforcement, and true model unlearning remain outside this implementation.
    </div>
  </main>
</body>
</html>
"""
    DOCS_DIR.mkdir(exist_ok=True)
    output_path = DOCS_DIR / "experiment_brief.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
