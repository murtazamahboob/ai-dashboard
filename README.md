# AI Dashboard — Live Churn Model, Animated Charts & Dark Mode

An **AI-powered interactive dashboard** that runs a neural network live in
your browser: move the sliders, and the model scores the customer
instantly — no server, no API key, no network at all.

The model is a churn predictor trained from scratch in pure NumPy
(8→16→8→1 MLP, Adam, 60 epochs) on 2,000 synthetic customers. Its
weights are exported to `web/data/weights.json`, and the dashboard runs
the *exact same model* in JavaScript — the Python and JS predictions
agree to 1e-6, verified by the test suite.

## What you see

- **Live KPI cards** — test accuracy, ROC AUC, precision/recall, churn rate, and per-prediction latency measured in your browser.
- **Animated training chart** — the 60-epoch training run replays on canvas: loss falls, accuracy climbs. Hit replay any time.
- **ROC curve** — drawn from scratch on canvas, with the AUC.
- **Confusion matrix** — test-set outcomes at threshold 0.5.
- **Feature importance** — permutation importance: how much accuracy is lost when each feature is shuffled.
- **Live inference playground** — 8 sliders (tenure, charges, contract, support calls, usage, age, services) plus three presets (loyal / at-risk / new & expensive). The churn probability, risk verdict and meter update as you drag.
- **Live prediction feed** — a simulated stream of customers scored once a second, with a latency sparkline.
- **Dark mode** — dark by default, one click to light, remembered in `localStorage`. Every chart follows the theme.

## Results (measured, not claimed)

Trained on 1,600 customers, tested on 400 (churn rate 30.0%):

| Metric | Value |
| --- | --- |
| Test accuracy | 84.5% |
| Precision / Recall | 82.2% / 67.2% |
| F1 | 74.0% |
| ROC AUC | 0.887 |
| Confusion (TN / FP / FN / TP) | 250 / 19 / 43 / 88 |
| Train loss, epoch 1 → 60 | 0.4297 → 0.2628 |
| Train accuracy | 89.8% |
| Most important feature | support_calls |

(Reproduced by `python examples/demo.py`, twice, including in a fresh venv — identical numbers.)

## Quickstart

Just open the dashboard — no build step, no server:

```bash
open web/index.html        # or double-click it
# or serve the web/ folder:
python -m http.server --directory web 8000
```

Retrain the model and regenerate all dashboard data:

```bash
pip install -r requirements.txt
python examples/demo.py
# writes web/data/weights.json, training_history.json, metrics.json
# and outputs/report.md + outputs/summary.json
```

## How it works

1. **Synthetic data, real signal.** `src/dashboard/data.py` generates 2,000 customers from a known churn rule — long tenure, long contracts and high usage keep customers; high charges, many support calls and few services push them away, with an interaction (expensive + new churns fastest). No real customer data.
2. **Model trained from scratch.** `src/dashboard/model.py` is an 8→16→8→1 MLP in pure NumPy — He initialisation, ReLU hidden layers, sigmoid output, minibatch Adam. No scikit-learn, no PyTorch.
3. **Same model, two languages.** `src/dashboard/pipeline.py` exports the weights to `web/data/weights.json`; `web/js/model.js` runs the identical forward pass in the browser. The test suite runs both and compares to 1e-6.
4. **Charts drawn from scratch.** `web/js/charts.js` draws every chart on `<canvas>` — no chart library, no CDN, no external requests anywhere in the site (tested).

## Project layout

```
web/
  index.html            dashboard page (KPIs, charts, playground, live feed)
  css/style.css         dark/light themes, cards, animations
  js/model.js           in-browser MLP inference (loads weights.json)
  js/charts.js          canvas charts: training, ROC, importance, latency
  js/app.js             theme, KPIs, sliders, presets, live feed
  data/weights.json         exported MLP weights + standardisation
  data/training_history.json  loss/accuracy per epoch
  data/metrics.json         report, ROC, importance, live-feed rows
src/dashboard/
  data.py       synthetic churn dataset (seeded, reproducible)
  model.py      NumPy MLP + Adam trainer + weight export
  metrics.py    confusion matrix, ROC/AUC, permutation importance
  pipeline.py   train → evaluate → export everything the web app needs
examples/demo.py  train, export, write outputs/report.md
tests/          24 tests — data, model, metrics, export, web, JS parity
```

## Tests

```bash
python -m pytest -q   # 24 tests, all offline (incl. a Node check of the JS)
```

## Keywords

AI dashboard, machine learning dashboard, churn prediction, neural network
in JavaScript, live model inference, animated charts, dark mode dashboard,
canvas charts, offline AI, NumPy MLP, ROC curve, confusion matrix, Python.
