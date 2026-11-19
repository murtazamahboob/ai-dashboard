/* Live model inference in the browser.
 * Loads web/data/weights.json (exported by the Python trainer) and runs
 * the exact same MLP forward pass: standardise -> ReLU(8->16) ->
 * ReLU(16->8) -> sigmoid(8->1). No network calls besides the local JSON. */
(function (global) {
  "use strict";

  function relu(z) { return z > 0 ? z : 0; }
  function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z)))); }

  function matVec(vec, matrix) {
    // matrix is rows x cols (input dim x output dim); returns vec @ matrix.
    var out = new Array(matrix[0].length).fill(0);
    for (var i = 0; i < vec.length; i++) {
      for (var j = 0; j < out.length; j++) out[j] += vec[i] * matrix[i][j];
    }
    return out;
  }

  function forward(weights, rawFeatures) {
    var x = rawFeatures.map(function (v, i) {
      return (v - weights.feature_mean[i]) / weights.feature_std[i];
    });
    var h1 = matVec(x, weights.weights.W1).map(function (v, j) {
      return relu(v + weights.weights.b1[j]);
    });
    var h2 = matVec(h1, weights.weights.W2).map(function (v, j) {
      return relu(v + weights.weights.b2[j]);
    });
    var out = matVec(h2, weights.weights.W3)[0] + weights.weights.b3[0];
    return sigmoid(out);
  }

  function load(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error("could not load " + url);
      return r.json();
    }).then(function (weights) {
      return {
        weights: weights,
        features: weights.features,
        predict: function (rawFeatures) { return forward(weights, rawFeatures); },
      };
    });
  }

  global.DashboardModel = { load: load, forward: forward, relu: relu, sigmoid: sigmoid };
})(typeof window !== "undefined" ? window : globalThis);
