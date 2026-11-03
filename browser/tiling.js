"use strict";
/* Seamless tiling, the same maths as the desktop app's tiling.py. Shared by the page (self-test) and the worker.
   Every tile has the same shape (WebGPU compiles per shape), tiles are spread evenly, and each axis has 1D
   cross-fade weights that sum to exactly 1 at every pixel, so a single output buffer is enough. */
var Tiling = (function () {
  function tileStarts(length, tile, overlap) {
    if (length <= tile) return [0];
    var step = Math.max(1, tile - 2 * overlap);
    var n = Math.ceil((length - tile) / step) + 1;
    var seen = {}, out = [];
    for (var i = 0; i < n; i++) {
      var s = Math.round((i * (length - tile)) / (n - 1));
      if (!seen[s]) { seen[s] = 1; out.push(s); }
    }
    return out.sort(function (a, b) { return a - b; });
  }

  function axisWeights(starts, tile, length, ramp) {
    var raw = starts.map(function (s) {
      var w = new Float64Array(tile).fill(1);
      for (var i = 0; i < ramp; i++) {
        var r = (i + 0.5) / ramp;
        if (s > 0) w[i] = Math.min(w[i], r);
        if (s + tile < length) w[tile - 1 - i] = Math.min(w[tile - 1 - i], r);
      }
      for (var j = 0; j < tile; j++) w[j] = Math.max(w[j], 1e-4);
      return w;
    });
    var total = new Float64Array(length);
    starts.forEach(function (s, k) { for (var i = 0; i < tile; i++) total[s + i] += raw[k][i]; });
    return starts.map(function (s, k) {
      var out = new Float32Array(tile);
      for (var i = 0; i < tile; i++) out[i] = raw[k][i] / total[s + i];
      return out;
    });
  }

  function ceilTo(n, m) { return Math.ceil(n / m) * m; }

  /* Plan the tiles for an h x w image. Returns tile size, padded size, starts and per-axis weights (in output px). */
  function plan(h, w, scale, tile, overlap, multiple) {
    tile = Math.max(multiple, Math.floor(tile / multiple) * multiple);
    var th = Math.min(tile, ceilTo(h, multiple)), tw = Math.min(tile, ceilTo(w, multiple));
    var ph = Math.max(h, th), pw = Math.max(w, tw);
    var o = Math.min(overlap, Math.floor(th / 4), Math.floor(tw / 4));
    var ys = tileStarts(ph, th, o), xs = tileStarts(pw, tw, o);
    return {
      th: th, tw: tw, ph: ph, pw: pw, ys: ys, xs: xs,
      wy: axisWeights(ys.map(function (y) { return y * scale; }), th * scale, ph * scale, o * scale),
      wx: axisWeights(xs.map(function (x) { return x * scale; }), tw * scale, pw * scale, o * scale),
    };
  }

  /* Reflect-pad (or edge-pad when the image is smaller than the pad) a planar CHW float image. */
  function pad(img, h, w, ph, pw) {
    if (ph === h && pw === w) return img;
    var out = new Float32Array(3 * ph * pw);
    var reflect = ph - h < h && pw - w < w;
    function idx(i, n) {
      if (i < n) return i;
      return reflect ? 2 * n - 2 - i : n - 1;
    }
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < ph; y++) {
        var sy = idx(y, h), src = c * h * w + sy * w, dst = c * ph * pw + y * pw;
        out.set(img.subarray(src, src + w), dst);
        for (var x = w; x < pw; x++) out[dst + x] = img[src + idx(x, w)];
      }
    }
    return out;
  }

  return { tileStarts: tileStarts, axisWeights: axisWeights, plan: plan, pad: pad };
})();
if (typeof self !== "undefined") self.Tiling = Tiling;
