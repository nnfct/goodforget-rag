"""Create plots, HTML report, and failure-case Markdown."""

from __future__ import annotations

from pathlib import Path

from goodforget_rag.plots import write_plots
from goodforget_rag.report import write_failure_markdown, write_validation_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    docs_dir = PROJECT_ROOT / "docs"
    write_plots(results_dir, docs_dir / "figures")
    report_path = write_validation_report(results_dir, docs_dir)
    failure_path = write_failure_markdown(results_dir, docs_dir)
    print(f"Wrote: {report_path}")
    print(f"Wrote: {failure_path}")


if __name__ == "__main__":
    main()
