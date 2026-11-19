/* Canvas charts drawn from scratch — no chart library, no CDN.
 * Each helper reads colours from CSS variables so charts follow the theme. */
(function (global) {
  "use strict";

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function setup(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = rect.width || 640, h = rect.height || 260;
    canvas.width = w * dpr; canvas.height = h * dpr;
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }
  function axes(ctx, w, h, pad) {
    ctx.strokeStyle = cssVar("--card-border") || "#22304f";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b);
    ctx.stroke();
  }
  function line(ctx, pts, w, h, pad, ymin, ymax, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach(function (p, i) {
      var x = pad.l + (i / (pts.length - 1)) * (w - pad.l - pad.r);
      var y = h - pad.b - ((p - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function drawTraining(canvas, history, upto) {
    var s = setup(canvas), ctx = s.ctx, w = s.w, h = s.h;
    var pad = { l: 40, r: 12, t: 12, b: 26 };
    ctx.clearRect(0, 0, w, h);
    axes(ctx, w, h, pad);
    var shown = history.slice(0, upto);
    if (shown.length < 2) return;
    var losses = shown.map(function (d) { return d.loss; });
    var accs = shown.map(function (d) { return d.accuracy; });
    line(ctx, losses, w, h, pad, 0, Math.max.apply(null, history.map(function (d) { return d.loss; })) * 1.05, cssVar("--accent") || "#38bdf8");
    line(ctx, accs, w, h, pad, 0, 1, cssVar("--good") || "#34d399");
    ctx.fillStyle = cssVar("--muted") || "#93a3c4";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("epoch " + shown.length + " / " + history.length, pad.l, h - 8);
    ctx.fillText("loss " + shown[shown.length - 1].loss.toFixed(3) +
      " · acc " + (shown[shown.length - 1].accuracy * 100).toFixed(1) + "%",
      w - 190, h - 8);
  }

  function drawRoc(canvas, roc) {
    var s = setup(canvas), ctx = s.ctx, w = s.w, h = s.h;
    var pad = { l: 40, r: 12, t: 12, b: 26 };
    ctx.clearRect(0, 0, w, h);
    axes(ctx, w, h, pad);
    // Diagonal (random classifier).
    ctx.strokeStyle = cssVar("--muted") || "#93a3c4";
    ctx.setLineDash([4, 4]); ctx.beginPath();
    ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, pad.t); ctx.stroke();
    ctx.setLineDash([]);
    // x = FPR, y = TPR.
    ctx.strokeStyle = cssVar("--accent2") || "#a78bfa"; ctx.lineWidth = 2; ctx.beginPath();
    roc.fpr.forEach(function (f, i) {
      var x = pad.l + f * (w - pad.l - pad.r);
      var y = h - pad.b - roc.tpr[i] * (h - pad.t - pad.b);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = cssVar("--muted") || "#93a3c4";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("AUC " + roc.auc.toFixed(3), pad.l, h - 8);
  }

  function drawImportance(canvas, importance) {
    var s = setup(canvas), ctx = s.ctx, w = s.w, h = s.h;
    var pad = { l: 150, r: 50, t: 10, b: 10 };
    var maxImp = Math.max.apply(null, importance.map(function (d) { return d.importance; })) || 1;
    ctx.clearRect(0, 0, w, h);
    var rowH = (h - pad.t - pad.b) / importance.length;
    importance.forEach(function (d, i) {
      var y = pad.t + i * rowH + 4;
      var barW = (d.importance / maxImp) * (w - pad.l - pad.r);
      ctx.fillStyle = cssVar("--accent") || "#38bdf8";
      ctx.fillRect(pad.l, y, Math.max(barW, 2), rowH - 8);
      ctx.fillStyle = cssVar("--text") || "#e8eefc";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(d.feature, 4, y + rowH / 2);
      ctx.fillStyle = cssVar("--muted") || "#93a3c4";
      ctx.fillText(d.importance.toFixed(3), pad.l + barW + 6, y + rowH / 2);
    });
  }

  function drawLatency(canvas, samples) {
    var s = setup(canvas), ctx = s.ctx, w = s.w, h = s.h;
    var pad = { l: 40, r: 12, t: 10, b: 20 };
    ctx.clearRect(0, 0, w, h);
    axes(ctx, w, h, pad);
    if (samples.length < 2) return;
    var maxV = Math.max.apply(null, samples) * 1.2 || 1;
    line(ctx, samples, w, h, pad, 0, maxV, cssVar("--good") || "#34d399");
    ctx.fillStyle = cssVar("--muted") || "#93a3c4";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("last: " + samples[samples.length - 1].toFixed(2) + " ms", pad.l, h - 6);
  }

  global.DashboardCharts = {
    drawTraining: drawTraining, drawRoc: drawRoc,
    drawImportance: drawImportance, drawLatency: drawLatency,
  };
})(typeof window !== "undefined" ? window : globalThis);
