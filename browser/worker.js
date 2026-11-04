"use strict";
/* AI Photo Enhancer worker: runs the ONNX models with onnxruntime-web, on the GPU (WebGPU) or the CPU (WASM).
   Pipeline per photo: remove noise -> sharpen -> upscale, each tiled with seamless cross-fades. */
importScripts("lib/ort.webgpu.min.js", "tiling.js");

var MODEL_BASE = "models/"; // relative to this file; everything is same-origin
ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated, so no wasm threads
ort.env.wasm.wasmPaths = new URL("lib/", self.location.href).href;
ort.env.logLevel = "error";

var CACHE = "s3-enhancer-models-v1";
var UPSCALE = { standard: "realesrgan-x4", art: "realesrgan-anime-x4", fast: "realesr-general-x4" };
var TILE = {
  webgpu: { "realesrgan-x4": 256, "realesrgan-anime-x4": 256, "realesr-general-x4": 256, "nafnet-denoise-32": 256, "nafnet-deblur-32": 256 },
  wasm: { "realesrgan-x4": 128, "realesrgan-anime-x4": 128, "realesr-general-x4": 192, "nafnet-denoise-32": 192, "nafnet-deblur-32": 192 },
};
var OVERLAP = { upscale: 12, denoise: 24, sharpen: 24 };
var COST = { denoise: 1, sharpen: 1, standard: 6, art: 2, fast: 0.5 };

var models = null, device = null, cancelled = false, notice = "";
var sessions = []; // [{id, dev, sess}] most recent last

function post(m, transfer) { self.postMessage(m, transfer || []); }

async function loadModels() {
  if (!models) {
    var r = await fetch(new URL(MODEL_BASE + "models.json", self.location.href));
    models = (await r.json()).models;
  }
  return models;
}

async function pickDevice(want) {
  if (want === "wasm") return "wasm";
  try {
    if (self.navigator.gpu && (await self.navigator.gpu.requestAdapter())) return "webgpu";
  } catch (e) { /* no WebGPU */ }
  return "wasm";
}

function hex(buf) {
  return Array.prototype.map.call(new Uint8Array(buf), function (b) { return b.toString(16).padStart(2, "0"); }).join("");
}

/* Download a model once (with progress), check its SHA-256, keep it in Cache Storage for next time. */
async function modelBytes(id, onBytes) {
  var m = (await loadModels())[id];
  var url = new URL(MODEL_BASE + m.file, self.location.href).href;
  var cache = null;
  try { cache = await caches.open(CACHE); } catch (e) { cache = null; }
  if (cache) {
    var hit = await cache.match(url);
    if (hit) {
      var buf = new Uint8Array(await hit.arrayBuffer());
      if (buf.length === m.bytes) return buf;
    }
  }
  var res = await fetch(url);
  if (!res.ok) throw new Error("Could not load the " + id + " model (" + res.status + ")");
  var out = new Uint8Array(m.bytes), got = 0, reader = res.body.getReader();
  for (;;) {
    var part = await reader.read();
    if (part.done) break;
    if (got + part.value.length > out.length) { var bigger = new Uint8Array(got + part.value.length); bigger.set(out); out = bigger; }
    out.set(part.value, got);
    got += part.value.length;
    onBytes(got, m.bytes);
  }
  out = out.subarray(0, got);
  if (hex(await crypto.subtle.digest("SHA-256", out)) !== m.sha256) throw new Error("The " + id + " model download is damaged. Try again.");
  if (cache) { try { await cache.put(url, new Response(out)); } catch (e) { /* storage full: still works this time */ } }
  return out;
}

async function session(id, onBytes) {
  for (var i = 0; i < sessions.length; i++) {
    if (sessions[i].id === id && sessions[i].dev === device) {
      var hit = sessions.splice(i, 1)[0]; sessions.push(hit); return hit.sess;
    }
  }
  var bytes = await modelBytes(id, onBytes);
  var sess;
  try {
    sess = await ort.InferenceSession.create(bytes, { executionProviders: [device], graphOptimizationLevel: "all", logSeverityLevel: 3 });
  } catch (e) {
    if (device !== "webgpu") throw e;
    device = "wasm";
    notice = "Your GPU could not run this model, so it runs on the CPU (slower).";
    sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 });
  }
  sessions.push({ id: id, dev: device, sess: sess });
  while (sessions.length > 2) { var old = sessions.shift(); try { old.sess.release(); } catch (e) { /* already gone */ } }
  return sess;
}

/* Run one model over a planar float image, tile by tile. Finished output rows go to put(band, y0, rows, bandH, OW). */
async function runTiled(id, kind, img, h, w, onTile, put) {
  var m = models[id], scale = m.scale;
  var p = Tiling.plan(h, w, scale, (TILE[device] || TILE.wasm)[id] || 128, OVERLAP[kind], m.multiple);
  var padded = Tiling.pad(img, h, w, p.ph, p.pw);
  var OW = w * scale, OH = h * scale, bandH = p.th * scale, tsz = p.th * p.tw;
  var acc = new Float32Array(3 * bandH * OW), accY = 0;
  var sess = await session(id, function () {});
  var patch = new Float32Array(3 * tsz);
  var total = p.ys.length * p.xs.length, done = 0;

  function flush(upto) {
    upto = Math.min(upto, OH);
    var n = upto - accY;
    if (n <= 0) return;
    put(acc, accY, n, bandH, OW);
    for (var c = 0; c < 3; c++) {
      var base = c * bandH * OW;
      acc.copyWithin(base, base + n * OW, base + bandH * OW);
      acc.fill(0, base + (bandH - n) * OW, base + bandH * OW);
    }
    accY = upto;
  }

  for (var iy = 0; iy < p.ys.length; iy++) {
    var y = p.ys[iy];
    flush(y * scale);
    for (var ix = 0; ix < p.xs.length; ix++) {
      if (cancelled) throw new Error("cancelled");
      var x = p.xs[ix];
      for (var c = 0; c < 3; c++) {
        for (var r = 0; r < p.th; r++) {
          var src = c * p.ph * p.pw + (y + r) * p.pw + x;
          patch.set(padded.subarray(src, src + p.tw), c * tsz + r * p.tw);
        }
      }
      var feeds = { input: new ort.Tensor("float32", patch, [1, 3, p.th, p.tw]) };
      var res;
      try {
        res = (await sess.run(feeds)).output;
      } catch (e) {
        if (device !== "webgpu") throw e;
        device = "wasm";
        notice = "The GPU ran out of memory or failed, so this continues on the CPU (slower).";
        sess = await session(id, function () {});
        res = (await sess.run(feeds)).output;
      }
      var out = res.data, oth = p.th * scale, otw = p.tw * scale;
      var oy = y * scale, ox = x * scale;
      var hh = Math.min(oth, OH - oy), ww = Math.min(otw, OW - ox);
      var wy = p.wy[iy], wx = p.wx[ix];
      for (var c2 = 0; c2 < 3; c2++) {
        for (var r2 = 0; r2 < hh; r2++) {
          var wv = wy[r2], ri = c2 * oth * otw + r2 * otw, ro = c2 * bandH * OW + (oy - accY + r2) * OW + ox;
          for (var k = 0; k < ww; k++) acc[ro + k] += out[ri + k] * wv * wx[k];
        }
      }
      if (res.dispose) res.dispose();
      onTile(++done, total);
    }
  }
  flush(OH);
}

function toPlanar(rgba, n) {
  var img = new Float32Array(3 * n);
  for (var i = 0; i < n; i++) {
    img[i] = rgba[4 * i] / 255; img[n + i] = rgba[4 * i + 1] / 255; img[2 * n + i] = rgba[4 * i + 2] / 255;
  }
  return img;
}

function blendInto(base, result, strength) {
  if (strength >= 1) return result;
  for (var i = 0; i < base.length; i++) result[i] = base[i] + strength * (result[i] - base[i]);
  return result;
}

function clamp8(v) { return v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0; }

/* A sink that writes finished rows into an RGBA image, averaging f x f blocks (f=2 turns a 4x model into 2x). */
function rgbaSink(out, OW, f) {
  var W = OW / f;
  return function (band, y0, n, bandH) {
    for (var r = 0; r < n; r += f) {
      var Y = (y0 + r) / f;
      for (var X = 0; X < W; X++) {
        var o = 4 * (Y * W + X);
        for (var c = 0; c < 3; c++) {
          var s = 0, base = c * bandH * OW + r * OW + X * f;
          for (var dy = 0; dy < f; dy++) for (var dx = 0; dx < f; dx++) s += band[base + dy * OW + dx];
          out[o + c] = clamp8(s / (f * f));
        }
        out[o + 3] = 255;
      }
    }
  };
}

function planarSink(dst, OW, OH) {
  return function (band, y0, n, bandH) {
    for (var c = 0; c < 3; c++) dst.set(band.subarray(c * bandH * OW, c * bandH * OW + n * OW), c * OH * OW + y0 * OW);
  };
}

async function run(msg) {
  var s = msg.s, w = msg.w, h = msg.h, n = w * h, t0 = Date.now();
  await loadModels();
  var steps = [];
  if (s.denoise > 0) steps.push(["denoise", "nafnet-denoise-32", COST.denoise, "Removing noise"]);
  if (s.sharpen > 0) steps.push(["sharpen", "nafnet-deblur-32", COST.sharpen, "Sharpening"]);
  if (s.upscale > 1) steps.push(["upscale", UPSCALE[s.model] || UPSCALE.fast, COST[s.model] || 1, "Upscaling"]);
  var totalCost = steps.reduce(function (a, st) { return a + st[2]; }, 0) || 1, doneCost = 0, lastPost = 0;
  var img = toPlanar(msg.rgba, n);
  var outRGBA = null, OW = w, OH = h;

  for (var i = 0; i < steps.length; i++) {
    var st = steps[i];
    var report = function (frac, label) {
      var now = Date.now();
      if (now - lastPost < 120 && frac < 1) return;
      lastPost = now;
      post({ type: "progress", id: msg.id, frac: Math.min(0.999, (doneCost + st[2] * frac) / totalCost), stage: label });
    };
    report(0, st[3]);
    await session(st[1], function (got, all) { report(0, "Downloading AI model " + Math.round((100 * got) / all) + "%"); });
    var onTile = function (d, t) { report(d / t, st[3]); };
    if (st[0] === "upscale") {
      var f = 4 / s.upscale;
      OW = w * s.upscale; OH = h * s.upscale;
      outRGBA = new Uint8ClampedArray(4 * OW * OH);
      await runTiled(st[1], "upscale", img, h, w, onTile, rgbaSink(outRGBA, w * 4, f));
    } else {
      var res = new Float32Array(3 * n);
      await runTiled(st[1], st[0], img, h, w, onTile, planarSink(res, w, h));
      img = blendInto(img, res, st[0] === "denoise" ? s.denoise : s.sharpen);
    }
    doneCost += st[2];
  }
  if (!outRGBA) {
    outRGBA = new Uint8ClampedArray(4 * n);
    for (var j = 0; j < n; j++) {
      outRGBA[4 * j] = clamp8(img[j]); outRGBA[4 * j + 1] = clamp8(img[n + j]); outRGBA[4 * j + 2] = clamp8(img[2 * n + j]);
      outRGBA[4 * j + 3] = 255;
    }
  }
  post({ type: "done", id: msg.id, rgba: outRGBA, w: OW, h: OH, seconds: (Date.now() - t0) / 1000, device: device, notice: notice },
    [outRGBA.buffer]);
}

self.onmessage = async function (e) {
  var msg = e.data;
  if (msg.type === "cancel") { cancelled = true; return; }
  if (msg.type === "init") {
    device = await pickDevice(msg.want);
    try { await loadModels(); } catch (err) { post({ type: "fatal", message: "Could not read the model list: " + err.message }); return; }
    post({ type: "ready", device: device, models: models });
    return;
  }
  if (msg.type === "run") {
    cancelled = false;
    try {
      await run(msg);
    } catch (err) {
      if (cancelled) post({ type: "cancelled", id: msg.id });
      else post({ type: "error", id: msg.id, message: (err && err.message) || String(err) });
    }
  }
};
