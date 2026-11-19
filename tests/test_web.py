"""Tests for the pipeline export and the web dashboard files."""

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def test_pipeline_build_and_summary():
    from dashboard.pipeline import build, summary

    result = build(epochs=60)
    s = summary(result)
    assert s["n_train"] == 1600 and s["n_test"] == 400
    assert s["test_accuracy"] >= 0.80
    assert s["roc_auc"] >= 0.80
    assert s["epochs"] == 60
    assert s["final_train_loss"] < 0.6
    assert result["history"][0]["loss"] > result["history"][-1]["loss"]
    assert len(result["live_rows"]) == 12
    for row in result["live_rows"]:
        assert 0.0 <= row["churn_probability"] <= 1.0
        assert len(row["features"]) == 8


def test_export_writes_valid_json(tmp_path):
    from dashboard.pipeline import build, export

    result = build(epochs=10)
    paths = export(result, tmp_path)
    for key in ("weights", "history", "metrics"):
        data = json.loads(Path(paths[key]).read_text())
        assert isinstance(data, dict)
    weights = json.loads(Path(paths["weights"]).read_text())
    assert weights["architecture"] == [8, 16, 8, 1]
    history = json.loads(Path(paths["history"]).read_text())
    assert len(history["history"]) == 10
    metrics = json.loads(Path(paths["metrics"]).read_text())
    assert metrics["report"]["confusion"]["tp"] + metrics["report"]["confusion"]["fn"] > 0
    assert 0.0 <= metrics["roc"]["auc"] <= 1.0


def test_web_files_exist_and_reference_each_other():
    assert (WEB / "index.html").exists()
    assert (WEB / "css" / "style.css").exists()
    assert (WEB / "js" / "model.js").exists()
    assert (WEB / "js" / "charts.js").exists()
    assert (WEB / "js" / "app.js").exists()
    html = (WEB / "index.html").read_text()
    assert 'href="css/style.css"' in html
    assert 'src="js/model.js"' in html
    assert 'src="js/charts.js"' in html
    assert 'src="js/app.js"' in html
    assert 'id="theme-toggle"' in html
    assert 'id="chart-training"' in html
    assert 'id="chart-roc"' in html
    assert 'id="live-feed"' in html
    assert 'id="sliders"' in html


def test_web_is_fully_offline():
    """No external URLs, CDNs, fonts or tracking in any web file."""
    for path in WEB.rglob("*"):
        if path.is_file() and path.suffix in {".html", ".css", ".js"}:
            text = path.read_text()
            assert "https://" not in text, f"{path} contains an external URL"
            assert "http://" not in text, f"{path} contains an external URL"
            assert "googleapis" not in text and "cdn." not in text


def test_dark_mode_css_present():
    css = (WEB / "css" / "style.css").read_text()
    assert '[data-theme="light"]' in css
    assert "--bg" in css


def test_js_syntax_valid_with_node():
    for name in ("model.js", "charts.js", "app.js"):
        proc = subprocess.run(
            ["node", "--check", str(WEB / "js" / name)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr}"


def test_js_inference_matches_python(tmp_path):
    """The browser model must agree with the Python model to 1e-6."""
    from dashboard.data import generate
    from dashboard.pipeline import build

    result = build(epochs=30)
    weights_path = tmp_path / "weights.json"
    result["model"].save_weights(weights_path, result["mean"], result["std"])
    rows_path = tmp_path / "rows.json"
    X, _ = generate(n=10, seed=1234)
    rows_path.write_text(json.dumps(X.tolist()))
    mean = result["mean"]
    std = result["std"]
    py_probs = result["model"].predict_proba((X - mean) / std)
    js_code = (
        "const fs = require('fs');\n"
        "eval(fs.readFileSync(" + json.dumps(str(WEB / "js" / "model.js")) + ", 'utf8'));\n"
        "const weights = JSON.parse(fs.readFileSync(" + json.dumps(str(weights_path)) + ", 'utf8'));\n"
        "const rows = JSON.parse(fs.readFileSync(" + json.dumps(str(rows_path)) + ", 'utf8'));\n"
        "console.log(JSON.stringify(rows.map(r => DashboardModel.forward(weights, r))));\n"
    )
    proc = subprocess.run(
        ["node", "-e", js_code], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stderr
    js_probs = np.array(json.loads(proc.stdout))
    assert np.allclose(py_probs, js_probs, atol=1e-6), (
        f"max diff {np.max(np.abs(py_probs - js_probs))}"
    )
