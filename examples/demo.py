"""Demo: train the churn model, export the dashboard data, write a report.

Run from the project root:  python examples/demo.py
Writes web/data/*.json and outputs/report.md + outputs/summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dashboard.pipeline import build, export, summary  # noqa: E402


def main() -> None:
    result = build()
    paths = export(result, ROOT)
    s = summary(result)
    print("AI Dashboard — churn model (2,000 synthetic customers)")
    print(f"  train: {s['n_train']} rows, test: {s['n_test']} rows, "
          f"churn rate: {s['churn_rate']:.1%}")
    print(f"  train loss {s['final_train_loss']:.4f} "
          f"(epoch 1: {result['history'][0]['loss']:.4f}) -> "
          f"train accuracy {s['final_train_accuracy']:.1%}")
    print(f"  test accuracy {s['test_accuracy']:.1%}  precision "
          f"{s['test_precision']:.1%}  recall {s['test_recall']:.1%}  "
          f"F1 {s['test_f1']:.1%}")
    print(f"  ROC AUC {s['roc_auc']:.3f}  confusion {s['confusion']}")
    print(f"  most important feature: {s['top_feature']}")
    print("  exported:")
    for name, path in paths.items():
        print(f"    {name}: {path}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    cm = s["confusion"]
    lines = [
        "# AI Dashboard — demo report",
        "",
        f"- Train rows: {s['n_train']}, test rows: {s['n_test']}, "
        f"churn rate: {s['churn_rate']:.1%}",
        f"- Final train loss: {s['final_train_loss']:.4f}, "
        f"train accuracy: {s['final_train_accuracy']:.1%} "
        f"after {s['epochs']} epochs",
        f"- Test accuracy: {s['test_accuracy']:.1%}, precision: "
        f"{s['test_precision']:.1%}, recall: {s['test_recall']:.1%}, "
        f"F1: {s['test_f1']:.1%}",
        f"- ROC AUC: {s['roc_auc']:.3f}",
        f"- Confusion: TN {cm['tn']}, FP {cm['fp']}, "
        f"FN {cm['fn']}, TP {cm['tp']}",
        f"- Most important feature: {s['top_feature']}",
        "",
        "Open web/index.html in a browser — no server, no network needed.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("  wrote outputs/summary.json and outputs/report.md")


if __name__ == "__main__":
    main()
