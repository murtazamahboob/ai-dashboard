/* Dashboard wiring: theme, KPIs, charts, playground, live feed.
 * All data is local JSON exported by the Python trainer. */
(function () {
  "use strict";

  var FEATURE_RANGES = {
    tenure_months: [0, 72, 12], monthly_charges: [20, 120, 70],
    total_charges: [0, 8640, 840], contract_months: [1, 24, 12],
    support_calls: [0, 10, 2], data_usage_gb: [0, 500, 120],
    age: [18, 80, 42], num_services: [1, 6, 3],
  };
  var PRESETS = {
    loyal: [60, 45, 2700, 24, 0, 320, 45, 5],
    atrisk: [3, 110, 330, 1, 8, 40, 29, 1],
    new: [1, 115, 115, 1, 1, 200, 24, 2],
  };
  var FEATURE_LABELS = {
    tenure_months: "Tenure (months)", monthly_charges: "Monthly charge ($)",
    total_charges: "Total charges ($)", contract_months: "Contract (months)",
    support_calls: "Support calls (6 mo)", data_usage_gb: "Data usage (GB)",
    age: "Age", num_services: "Services",
  };

  // Seeded PRNG so the live feed is the same on every load.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function $(id) { return document.getElementById(id); }
  function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

  // --- theme ------------------------------------------------------------
  var root = document.documentElement;
  var savedTheme = null;
  try { savedTheme = localStorage.getItem("ai-dashboard-theme"); } catch (e) { /* private mode */ }
  if (savedTheme) root.setAttribute("data-theme", savedTheme);
  function syncThemeButton() {
    var dark = root.getAttribute("data-theme") !== "light";
    $("theme-toggle").textContent = dark ? "☀ Light" : "☾ Dark";
  }
  $("theme-toggle").addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("ai-dashboard-theme", next); } catch (e) { /* ignore */ }
    syncThemeButton();
    redrawAll();
  });
  syncThemeButton();

  // --- state ------------------------------------------------------------
  var model = null, metrics = null, history = null;
  var trainingTimer = null, liveTimer = null, liveOn = true;
  var latencies = [];
  var feedRand = mulberry32(20261112);

  function loadJson(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error("could not load " + url);
      return r.json();
    });
  }

  function redrawAll() {
    if (!metrics || !history) return;
    DashboardCharts.drawRoc($("chart-roc"), metrics.roc);
    DashboardCharts.drawImportance($("chart-importance"), metrics.importance);
    DashboardCharts.drawLatency($("chart-latency"), latencies);
    animateTraining();
  }

  function animateTraining() {
    var canvas = $("chart-training");
    if (trainingTimer) clearInterval(trainingTimer);
    var upto = 2;
    DashboardCharts.drawTraining(canvas, history.history, upto);
    trainingTimer = setInterval(function () {
      upto += 2;
      if (upto > history.history.length) { clearInterval(trainingTimer); upto = history.history.length; }
      DashboardCharts.drawTraining(canvas, history.history, upto);
    }, 60);
  }
  $("replay-training").addEventListener("click", animateTraining);

  // --- playground -------------------------------------------------------
  var sliderValues = {};
  function buildSliders() {
    var container = $("sliders");
    container.innerHTML = "";
    model.features.forEach(function (name, i) {
      var range = FEATURE_RANGES[name];
      var wrap = document.createElement("div");
      wrap.className = "slider";
      var label = document.createElement("label");
      label.textContent = FEATURE_LABELS[name];
      var out = document.createElement("output");
      out.id = "out-" + name;
      label.appendChild(out);
      var input = document.createElement("input");
      input.type = "range"; input.min = range[0]; input.max = range[1];
      input.step = name === "total_charges" ? 10 : 1; input.value = range[2];
      input.setAttribute("aria-label", FEATURE_LABELS[name]);
      input.addEventListener("input", function () {
        sliderValues[name] = Number(input.value);
        out.textContent = input.value;
        scorePlayground();
      });
      sliderValues[name] = range[2];
      out.textContent = range[2];
      wrap.appendChild(label); wrap.appendChild(input);
      container.appendChild(wrap);
    });
  }
  function currentFeatures() {
    return model.features.map(function (name) { return sliderValues[name]; });
  }
  function scorePlayground() {
    var t0 = performance.now();
    var p = model.predict(currentFeatures());
    var ms = performance.now() - t0;
    $("verdict-value").textContent = fmtPct(p);
    $("meter-fill").style.width = (p * 100).toFixed(1) + "%";
    $("verdict-text").textContent = p >= 0.7 ? "High risk — offer a retention deal now."
      : p >= 0.4 ? "Medium risk — worth a check-in."
      : "Low risk — this customer is likely staying.";
    $("kpi-latency").textContent = ms.toFixed(2) + " ms";
    return p;
  }
  document.querySelectorAll("[data-preset]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var values = PRESETS[btn.getAttribute("data-preset")];
      model.features.forEach(function (name, i) {
        sliderValues[name] = values[i];
        var input = document.querySelector("#sliders input[aria-label='" + FEATURE_LABELS[name] + "']");
        var out = $("out-" + name);
        if (input) input.value = values[i];
        if (out) out.textContent = values[i];
      });
      scorePlayground();
    });
  });

  // --- live feed --------------------------------------------------------
  function pushFeedRow() {
    if (!liveOn || !model || !metrics) return;
    var rows = metrics.live_rows;
    var row = rows[Math.floor(feedRand() * rows.length)];
    var t0 = performance.now();
    var p = model.predict(Object.values(row.features));
    var ms = performance.now() - t0;
    latencies.push(ms);
    if (latencies.length > 40) latencies.shift();
    var li = document.createElement("li");
    var cls = p >= 0.5 ? "risk-high" : "risk-low";
    li.innerHTML = "<span>" + row.customer + "</span><span class='" + cls + "'>" +
      fmtPct(p) + (p >= 0.5 ? " · churn" : " · stay") + "</span><span>" + ms.toFixed(2) + " ms</span>";
    var feed = $("live-feed");
    feed.prepend(li);
    while (feed.children.length > 12) feed.removeChild(feed.lastChild);
    DashboardCharts.drawLatency($("chart-latency"), latencies);
  }
  $("live-toggle").addEventListener("click", function () {
    liveOn = !liveOn;
    this.textContent = liveOn ? "⏸ Pause live feed" : "▶ Resume live feed";
    this.setAttribute("aria-pressed", String(liveOn));
  });

  // --- boot -------------------------------------------------------------
  Promise.all([
    DashboardModel.load("data/weights.json"),
    loadJson("data/metrics.json"),
    loadJson("data/training_history.json"),
  ]).then(function (parts) {
    model = parts[0]; metrics = parts[1]; history = parts[2];
    $("model-badge").textContent = "model: MLP " + model.weights.architecture.join(" → ") + " · live in-browser";
    var r = metrics.report;
    $("kpi-accuracy").textContent = fmtPct(r.accuracy);
    $("kpi-accuracy-sub").textContent = "on " + metrics.n_test + " test customers";
    $("kpi-auc").textContent = metrics.roc.auc.toFixed(3);
    $("kpi-pr").textContent = fmtPct(r.precision) + " / " + fmtPct(r.recall);
    $("kpi-f1").textContent = "F1 " + fmtPct(r.f1);
    $("kpi-churn").textContent = fmtPct(metrics.churn_rate);
    $("kpi-rows").textContent = metrics.n_train + " train · " + metrics.n_test + " test";
    $("roc-note").textContent = "AUC " + metrics.roc.auc.toFixed(3) +
      " — the model ranks a random churner above a random stayer " +
      (metrics.roc.auc * 100).toFixed(1) + "% of the time.";
    var cm = r.confusion;
    $("confusion").innerHTML =
      "<div class='cell'><strong>" + cm.tn + "</strong><span>True negatives — stayed, predicted stay</span></div>" +
      "<div class='cell'><strong>" + cm.fp + "</strong><span>False positives — stayed, flagged churn</span></div>" +
      "<div class='cell'><strong>" + cm.fn + "</strong><span>False negatives — churned, missed</span></div>" +
      "<div class='cell'><strong>" + cm.tp + "</strong><span>True positives — churned, caught</span></div>";
    $("confusion-note").textContent = "Caught " + cm.tp + " of " + (cm.tp + cm.fn) +
      " churners (recall " + fmtPct(r.recall) + "); " + cm.fp + " false alarms.";
    buildSliders();
    scorePlayground();
    redrawAll();
    liveTimer = setInterval(pushFeedRow, 1000);
    pushFeedRow();
  }).catch(function (err) {
    $("model-badge").textContent = "could not load model data — serve the web/ folder or open index.html from disk";
    console.error(err);
  });
})();
