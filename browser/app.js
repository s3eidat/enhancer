"use strict";
/* AI Photo Enhancer, page side: files, settings, the before/after view and saving.
   The AI models run in worker.js (WebGPU, or WASM on the CPU), so the page stays responsive. */
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var MAX_OUT_MP = 64, MAX_SIDE = 16384, PREVIEW_OUT = 640;
  var NOTES = {
    standard: "Real-ESRGAN: the best for photos. Cleans JPEG blocks and blur while it upscales. Heavier: about 8x slower than Fast.",
    art: "Real-ESRGAN anime: flat colours and clean lines for art, anime, logos and screenshots.",
    fast: "Real-ESRGAN general: small and quick, a lighter touch. Good for big batches.",
  };
  var EXT = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };
  var DEFAULTS = { upscale: 2, model: "fast", detail: 1, denoise: 0, sharpen: 0, format: "image/png", engine: "auto" };
  var s = Object.assign({}, DEFAULTS, S3.store("enh.settings") || {});
  var items = [], sel = null, running = false, worker = null, ready = null, device = "", nextId = 1, jobs = {};
  var V = { mode: "empty", w: 0, h: 0, zoom: 1, x: 0, y: 0, split: 0.5, fit: true, hold: null };

  /* ---------- worker ---------- */
  function startWorker() {
    if (worker) worker.terminate();
    Object.keys(jobs).forEach(function (k) { jobs[k].reject(new Error("restarted")); delete jobs[k]; });
    $("device").textContent = "Checking GPU…";
    worker = new Worker("worker.js");
    ready = new Promise(function (resolve, reject) {
      worker.onmessage = function (e) {
        var m = e.data;
        if (m.type === "ready") { device = m.device; showDevice(); resolve(m); return; }
        if (m.type === "fatal") { reject(new Error(m.message)); return; }
        var j = jobs[m.id];
        if (!j) return;
        if (m.type === "progress") { j.onProgress(m.frac, m.stage); return; }
        delete jobs[m.id];
        if (m.type === "done") { device = m.device; showDevice(); j.resolve(m); }
        else if (m.type === "cancelled") { var c = new Error("Cancelled"); c.cancelled = true; j.reject(c); }
        else j.reject(new Error(m.message));
      };
      worker.onerror = function (e) { reject(new Error(e.message || "The AI worker failed to start")); };
    });
    ready.catch(function (e) { $("device").textContent = "AI unavailable"; S3.toast(e.message, "error"); });
    worker.postMessage({ type: "init", want: s.engine });
  }

  function showDevice() {
    var gpu = device === "webgpu";
    $("device").textContent = gpu ? "GPU · WebGPU" : "CPU · WASM";
    $("device").className = "chip " + (gpu ? "accent" : "warn");
    $("device").title = gpu ? "The AI runs on your graphics card" : "No usable WebGPU here, so the AI runs on the CPU (slower)";
  }

  function enhanceRGBA(rgba, w, h, settings, onProgress) {
    return ready.then(function () {
      return new Promise(function (resolve, reject) {
        var id = nextId++;
        jobs[id] = { resolve: resolve, reject: reject, onProgress: onProgress || function () {} };
        worker.postMessage({ type: "run", id: id, rgba: rgba, w: w, h: h, s: settings }, [rgba.buffer]);
      });
    });
  }

  /* ---------- helpers ---------- */
  function canvas(w, h) { var c = document.createElement("canvas"); c.width = w; c.height = h; return c; }
  function fmtSize(w, h) { return w + "×" + h; }
  function mp(w, h) { return (w * h / 1e6).toFixed(1) + " MP"; }
  function fmtTime(sec) { sec = Math.max(0, Math.round(sec)); var m = Math.floor(sec / 60); return m + ":" + String(sec % 60).padStart(2, "0"); }
  function pctText(v) { return Math.round(v * 100) + "%"; }
  function toBlob(c, type) { return new Promise(function (res) { c.toBlob(res, type, 0.95); }); }
  function settingsForRun() { return { upscale: s.upscale, model: s.model, denoise: s.denoise, sharpen: s.sharpen }; }
  function nothing() { return s.upscale === 1 && !s.denoise && !s.sharpen; }
  function outSize(it) { return [it.w * s.upscale, it.h * s.upscale]; }
  function tooBig(it) { var o = outSize(it); return o[0] > MAX_SIDE || o[1] > MAX_SIDE || o[0] * o[1] > MAX_OUT_MP * 1e6; }
  function selItem() { return items.filter(function (i) { return i.id === sel; })[0]; }
  function outName(it) { return it.name.replace(/\.[^.]+$/, "") + "_enhanced." + EXT[s.format]; }

  /* ---------- settings ---------- */
  function renderSettings() {
    [["upscale", true], ["model", false]].forEach(function (p) {
      S3.$$("#" + p[0] + " button").forEach(function (b) { b.setAttribute("aria-pressed", String(String(s[p[0]]) === b.dataset.v)); });
    });
    $("modelField").classList.toggle("off", s.upscale === 1);
    $("detailRow").classList.toggle("off", s.upscale === 1);
    $("modelNote").textContent = NOTES[s.model];
    ["detail", "denoise", "sharpen"].forEach(function (k) {
      $(k).value = Math.round(s[k] * 100);
      var off = s[k] === 0 && k !== "detail";
      $(k + "Out").textContent = off ? "Off" : pctText(s[k]);
      $(k + "Out").classList.toggle("on", !off);
    });
    $("format").value = s.format;
    $("engine").value = s.engine;
    S3.store("enh.settings", s);
    renderButtons();
  }

  function bindSettings() {
    ["upscale", "model"].forEach(function (k) {
      $(k).addEventListener("click", function (e) {
        var b = e.target.closest("button"); if (!b) return;
        s[k] = k === "upscale" ? Number(b.dataset.v) : b.dataset.v;
        changed();
      });
    });
    ["detail", "denoise", "sharpen"].forEach(function (k) {
      $(k).addEventListener("input", function (e) { s[k] = Number(e.target.value) / 100; changed(); });
    });
    $("format").addEventListener("change", function (e) { s.format = e.target.value; renderSettings(); });
    $("engine").addEventListener("change", function (e) {
      if (running) { e.target.value = s.engine; S3.toast("Wait for the current photos to finish", "error"); return; }
      s.engine = e.target.value; renderSettings(); startWorker();
    });
  }

  var previewTimer = 0;
  function changed() {
    renderSettings(); renderList();
    if (V.mode === "live" && V.live) { clearTimeout(previewTimer); previewTimer = setTimeout(function () { spotPreview(V.live.cx, V.live.cy); }, 500); }
  }

  function renderButtons() {
    var todo = items.filter(function (i) { return !i.out && !i.error && !tooBig(i); });
    var it = selItem();
    var redo = it && it.out && !running;
    $("go").disabled = running || nothing() || (!todo.length && !redo);
    $("goLbl").textContent = nothing() ? "Pick at least one step" : todo.length ? "Enhance " + todo.length + " photo" + (todo.length > 1 ? "s" : "")
      : redo ? "Run again with these settings" : "Enhance";
    $("cancel").hidden = !running;
    $("save").disabled = !(it && it.out);
    $("saveLbl").textContent = it && it.out ? "Download " + EXT[it.out.type].toUpperCase() : "Download";
    var outs = items.filter(function (i) { return i.out; });
    $("saveAll").hidden = outs.length < 2;
    var big = items.filter(tooBig);
    $("warn").hidden = !big.length;
    if (big.length) {
      var o = outSize(big[0]);
      $("warn").className = "en-warn bad";
      $("warn").textContent = big[0].name + " would come out " + fmtSize(o[0], o[1]) + " (" + mp(o[0], o[1]) + "), too big for a browser tab. Pick 2× or Off, or use a smaller photo.";
    }
  }

  /* ---------- files ---------- */
  async function addFiles(files) {
    var added = 0;
    for (var k = 0; k < files.length; k++) {
      var f = files[k];
      if (!/^image\//.test(f.type) && !/\.(jpe?g|png|webp|bmp|gif|avif)$/i.test(f.name)) continue;
      var it = { id: nextId++, file: f, name: f.name || "pasted.png", url: URL.createObjectURL(f), w: 0, h: 0, status: "ready", progress: 0, out: null, error: "" };
      try {
        it.bmp = await createImageBitmap(f, { imageOrientation: "from-image" });
        it.w = it.bmp.width; it.h = it.bmp.height;
      } catch (e) { it.error = "Cannot read this image"; }
      items.push(it); added++;
    }
    if (!added) { S3.toast("Those files are not photos", "error"); return; }
    if (!selItem()) select(items[items.length - added].id); else { renderList(); renderButtons(); }
  }

  function statusOf(it) {
    if (it.error) return ["bad", it.error];
    if (it.status === "running") return ["", it.stage + " " + pctText(it.progress) + (it.eta != null ? " · " + fmtTime(it.eta) + " left" : "")];
    if (it.status === "queued") return ["", "Waiting…"];
    if (it.out) return ["ok", fmtSize(it.out.w, it.out.h) + " · " + it.out.seconds.toFixed(1) + "s on " + (it.out.device === "webgpu" ? "GPU" : "CPU")];
    if (tooBig(it)) return ["bad", "Too big for " + s.upscale + "× here"];
    return ["", fmtSize(it.w, it.h) + " → " + fmtSize(outSize(it)[0], outSize(it)[1])];
  }

  function renderList() {
    var ul = $("list"); ul.textContent = "";
    items.forEach(function (it) {
      var st = statusOf(it);
      var bar = S3.h("i", {}, [S3.h("b", { style: "width:" + pctText(it.progress) })]);
      bar.hidden = it.status !== "running";
      var x = S3.h("button", { class: "en-x", title: it.status === "running" ? "Cancel" : "Remove", "aria-label": "Remove " + it.name, text: "×" });
      x.addEventListener("click", function (e) { e.stopPropagation(); remove(it); });
      var li = S3.h("li", { class: "en-item" + (it.id === sel ? " sel" : ""), tabindex: "0" }, [
        S3.h("img", { src: it.url, alt: "" }),
        S3.h("div", { class: "en-meta" }, [S3.h("b", { text: it.name }), S3.h("span", { class: st[0], text: st[1] }), bar]),
        x,
      ]);
      li.addEventListener("click", function () { select(it.id); });
      li.addEventListener("keydown", function (e) { if (e.key === "Enter") select(it.id); });
      ul.appendChild(li);
    });
  }

  function remove(it) {
    if (it.status === "running") { cancel(); return; }
    items = items.filter(function (i) { return i !== it; });
    URL.revokeObjectURL(it.url);
    if (it.out) URL.revokeObjectURL(it.out.url);
    if (sel === it.id) { sel = null; if (items[0]) select(items[0].id); else showEmpty(); }
    renderList(); renderButtons();
  }

  function select(id) {
    sel = id;
    var it = selItem();
    renderList(); renderButtons();
    if (!it) return;
    if (it.out) showCompare(it); else showOriginal(it);
  }

  /* ---------- running ---------- */
  async function rgbaOf(source, sx, sy, sw, sh) {
    var c = canvas(sw, sh), ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
    var d = ctx.getImageData(0, 0, sw, sh).data;
    var alpha = false;
    for (var i = 3; i < d.length; i += 4) if (d[i] < 255) { alpha = true; break; }
    return { rgba: new Uint8ClampedArray(d.buffer), alpha: alpha };
  }

  /* The AI result, plus the "AI detail" mix with a smooth resize and the original transparency. */
  function finish(res, source, sx, sy, sw, sh, alpha) {
    var out = canvas(res.w, res.h), ctx = out.getContext("2d");
    ctx.putImageData(new ImageData(res.rgba, res.w, res.h), 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    if (s.upscale > 1 && s.detail < 1) {
      ctx.globalAlpha = 1 - s.detail;
      ctx.drawImage(source, sx, sy, sw, sh, 0, 0, res.w, res.h);
      ctx.globalAlpha = 1;
    }
    if (alpha) {
      var a = canvas(res.w, res.h), actx = a.getContext("2d", { willReadFrequently: true });
      actx.imageSmoothingQuality = "high";
      actx.drawImage(source, sx, sy, sw, sh, 0, 0, res.w, res.h);
      var ad = actx.getImageData(0, 0, res.w, res.h).data, od = ctx.getImageData(0, 0, res.w, res.h);
      for (var i = 3; i < ad.length; i += 4) od.data[i] = ad[i];
      ctx.putImageData(od, 0, 0);
    }
    return out;
  }

  async function runItem(it) {
    var t0 = Date.now();
    it.status = "running"; it.progress = 0; it.stage = "Starting"; it.eta = null; it.error = "";
    renderList();
    var src = await rgbaOf(it.bmp, 0, 0, it.w, it.h);
    var res = await enhanceRGBA(src.rgba, it.w, it.h, settingsForRun(), function (frac, stage) {
      it.progress = frac; it.stage = stage;
      var el = (Date.now() - t0) / 1000;
      it.eta = frac > 0.03 ? el * (1 - frac) / frac : null;
      $("bar").value = frac; $("pText").textContent = it.name + " · " + stage + " " + pctText(frac);
      $("pEta").textContent = it.eta != null ? fmtTime(it.eta) + " left" : "";
      renderList();
    });
    var out = finish(res, it.bmp, 0, 0, it.w, it.h, src.alpha);
    if (s.format === "image/jpeg" && src.alpha) { // JPG has no transparency: put it on white
      var flat = canvas(out.width, out.height), fctx = flat.getContext("2d");
      fctx.fillStyle = "#fff"; fctx.fillRect(0, 0, flat.width, flat.height); fctx.drawImage(out, 0, 0); out = flat;
    }
    var blob = await toBlob(out, s.format);
    if (it.out) URL.revokeObjectURL(it.out.url);
    it.out = { blob: blob, url: URL.createObjectURL(blob), w: res.w, h: res.h, seconds: (Date.now() - t0) / 1000, device: res.device, type: s.format };
    it.status = "done";
    if (res.notice) S3.toast(res.notice);
    S3.done();
  }

  var cancelled = false;
  async function enhanceAll() {
    if (running) return;
    var it = selItem();
    var todo = items.filter(function (i) { return !i.out && !i.error && !tooBig(i); });
    if (!todo.length && it && it.out && !tooBig(it)) todo = [it];
    if (!todo.length) return;
    running = true; cancelled = false;
    todo.forEach(function (i) { i.status = "queued"; });
    $("prog").hidden = false; $("bar").value = 0;
    renderButtons(); renderList();
    for (var k = 0; k < todo.length && !cancelled; k++) {
      var cur = todo[k];
      try {
        await runItem(cur);
        if (cur.id === sel) showCompare(cur);
      } catch (e) {
        cur.status = "ready";
        if (e.cancelled) { cancelled = true; break; }
        cur.error = e.message || "Failed"; console.error(e);
      }
      renderList();
    }
    todo.forEach(function (i) { if (i.status === "queued") i.status = "ready"; });
    running = false;
    $("prog").hidden = true;
    renderButtons(); renderList();
  }

  function cancel() { cancelled = true; if (worker) worker.postMessage({ type: "cancel" }); }

  /* ---------- the viewer ---------- */
  function showEmpty() {
    V.mode = "empty";
    $("empty").hidden = false; $("divider").hidden = true; $("tagB").hidden = $("tagA").hidden = true;
    $("imgB").removeAttribute("src"); $("imgA").removeAttribute("src");
    $("live").hidden = true; $("tip").hidden = true; $("info").textContent = "";
  }

  function load(bUrl, aUrl, w, h) {
    return new Promise(function (resolve) {
      var n = aUrl ? 2 : 1, done = function () { if (--n === 0) resolve(); };
      $("imgB").onload = $("imgB").onerror = done; $("imgB").src = bUrl;
      if (aUrl) { $("imgA").onload = $("imgA").onerror = done; $("imgA").src = aUrl; } else $("imgA").removeAttribute("src");
      V.w = w; V.h = h;
    });
  }

  async function showOriginal(it) {
    V.mode = "original"; V.live = null;
    $("empty").hidden = true; $("live").hidden = true; $("divider").hidden = true; $("tagB").hidden = $("tagA").hidden = true;
    $("info").textContent = it.error ? it.error : fmtSize(it.w, it.h) + " · " + mp(it.w, it.h);
    await load(it.url, null, it.w, it.h);
    $("tip").hidden = !!it.error; $("view").classList.add("picking");
    fit();
  }

  async function showCompare(it) {
    V.mode = "compare"; V.live = null;
    $("empty").hidden = true; $("live").hidden = true; $("tip").hidden = true; $("view").classList.remove("picking");
    await load(it.url, it.out.url, it.out.w, it.out.h);
    $("divider").hidden = false; $("tagB").hidden = $("tagA").hidden = false;
    $("info").textContent = fmtSize(it.w, it.h) + " → " + fmtSize(it.out.w, it.out.h) + " · " + it.out.seconds.toFixed(1) + "s";
    fit();
  }

  var spotSeq = 0;
  async function spotPreview(cx, cy) {
    var it = selItem();
    if (!it || it.error || nothing()) return;
    if (running) { S3.toast("Previews wait until the photos finish"); return; }
    var seq = ++spotSeq, side = Math.max(32, Math.round(PREVIEW_OUT / s.upscale));
    var cw = Math.min(side, it.w), ch = Math.min(side, it.h);
    var x0 = Math.round(Math.min(Math.max(0, cx * it.w - cw / 2), it.w - cw));
    var y0 = Math.round(Math.min(Math.max(0, cy * it.h - ch / 2), it.h - ch));
    $("live").hidden = false; $("liveText").textContent = "Previewing this spot…"; $("tip").hidden = true;
    try {
      var src = await rgbaOf(it.bmp, x0, y0, cw, ch), t0 = Date.now();
      var res = await enhanceRGBA(src.rgba, cw, ch, settingsForRun(), function (f, stage) { if (seq === spotSeq) $("liveText").textContent = stage + " " + pctText(f); });
      if (seq !== spotSeq) return;
      var after = finish(res, it.bmp, x0, y0, cw, ch, src.alpha);
      var before = canvas(res.w, res.h), bctx = before.getContext("2d");
      bctx.imageSmoothingQuality = "high"; bctx.drawImage(it.bmp, x0, y0, cw, ch, 0, 0, res.w, res.h);
      var bu = URL.createObjectURL(await toBlob(before, "image/png")), au = URL.createObjectURL(await toBlob(after, "image/png"));
      V.mode = "live"; V.live = { cx: cx, cy: cy };
      $("view").classList.remove("picking");
      await load(bu, au, res.w, res.h);
      $("divider").hidden = false; $("tagB").hidden = $("tagA").hidden = false;
      $("liveText").textContent = "Preview of one spot · " + ((Date.now() - t0) / 1000).toFixed(1) + "s on " + (res.device === "webgpu" ? "GPU" : "CPU");
      V.fit = true; fit();
    } catch (e) {
      if (!e.cancelled) S3.toast(e.message, "error");
      $("live").hidden = true;
    }
  }

  function apply() {
    var vw = $("view").clientWidth;
    [$("imgB"), $("imgA")].forEach(function (img) {
      img.style.width = V.w * V.zoom + "px"; img.style.height = V.h * V.zoom + "px";
      img.style.transform = "translate(" + V.x + "px," + V.y + "px)";
      img.style.imageRendering = V.zoom >= 2 ? "pixelated" : "auto";
    });
    var split = V.hold === "b" ? 1 : V.hold === "a" ? 0 : V.split;
    $("layerA").style.clipPath = V.mode === "original" ? "inset(0 0 0 100%)" : "inset(0 0 0 " + split * vw + "px)";
    $("divider").style.left = split * vw + "px";
    $("zoomV").textContent = Math.round(V.zoom * 100) + "%";
  }

  function fit() {
    if (!V.w) return;
    var vw = $("view").clientWidth, vh = $("view").clientHeight;
    // small pictures and spot previews fill the view; big ones stop at 100%
    V.zoom = Math.min(vw / V.w, vh / V.h, V.mode === "live" ? 4 : Math.max(1, 480 / Math.max(V.w, V.h)));
    V.x = (vw - V.w * V.zoom) / 2; V.y = (vh - V.h * V.zoom) / 2; V.fit = true;
    apply();
  }

  function zoomTo(z, px, py) {
    var vw = $("view").clientWidth, vh = $("view").clientHeight;
    if (px == null) { px = vw / 2; py = vh / 2; }
    var ix = (px - V.x) / V.zoom, iy = (py - V.y) / V.zoom;
    V.zoom = Math.min(16, Math.max(0.02, z));
    V.x = px - ix * V.zoom; V.y = py - iy * V.zoom; V.fit = false;
    apply();
  }

  function bindViewer() {
    var vr = $("view"), drag = null;
    vr.addEventListener("wheel", function (e) {
      if (V.mode === "empty") return;
      e.preventDefault();
      var r = vr.getBoundingClientRect();
      zoomTo(V.zoom * Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    vr.addEventListener("pointerdown", function (e) {
      if (V.mode === "empty" || e.button !== 0 || e.target.closest("button")) return;
      var onDiv = !!e.target.closest("#divider");
      drag = { split: onDiv, sx: e.clientX, sy: e.clientY, x: V.x, y: V.y, moved: false };
      vr.setPointerCapture(e.pointerId);
    });
    vr.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.split) { var r = vr.getBoundingClientRect(); V.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); }
      else { V.x = drag.x + dx; V.y = drag.y + dy; V.fit = false; }
      apply();
    });
    vr.addEventListener("pointerup", function (e) {
      var d = drag; drag = null;
      if (!d || d.moved || d.split || V.mode !== "original") return;
      var r = vr.getBoundingClientRect();
      var cx = (e.clientX - r.left - V.x) / (V.w * V.zoom), cy = (e.clientY - r.top - V.y) / (V.h * V.zoom);
      if (cx >= 0 && cx <= 1 && cy >= 0 && cy <= 1) spotPreview(cx, cy);
    });
    vr.addEventListener("keydown", function (e) {
      if (V.mode === "empty") return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") { V.split = Math.min(1, Math.max(0, V.split + (e.key === "ArrowLeft" ? -0.05 : 0.05))); apply(); e.preventDefault(); }
      if (e.key === "+" || e.key === "=") zoomTo(V.zoom * 1.25);
      if (e.key === "-") zoomTo(V.zoom / 1.25);
      if (e.key === "0") fit();
      if (e.key === "1") zoomTo(1);
    });
    window.addEventListener("keydown", function (e) {
      if (e.repeat || !(V.mode === "compare" || V.mode === "live") || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      if (e.key === "b" || e.key === "a") { V.hold = e.key; apply(); }
    });
    window.addEventListener("keyup", function (e) { if (V.hold && (e.key === "b" || e.key === "a")) { V.hold = null; apply(); } });
    $("zoomIn").addEventListener("click", function () { zoomTo(V.zoom * 1.25); });
    $("zoomOut").addEventListener("click", function () { zoomTo(V.zoom / 1.25); });
    $("fitBtn").addEventListener("click", fit);
    $("oneBtn").addEventListener("click", function () { zoomTo(1); });
    $("liveBack").addEventListener("click", function () { spotSeq++; var it = selItem(); if (it) showOriginal(it); });
    window.addEventListener("resize", function () { if (V.fit) fit(); else apply(); });
  }

  /* ---------- sample + saving ---------- */
  async function sampleFile() {
    var c = canvas(480, 320), x = c.getContext("2d");
    var g = x.createLinearGradient(0, 0, 0, 320); g.addColorStop(0, "#1d4e89"); g.addColorStop(0.6, "#f2a65a"); g.addColorStop(1, "#5b3a29");
    x.fillStyle = g; x.fillRect(0, 0, 480, 320);
    x.fillStyle = "#ffe7a3"; x.beginPath(); x.arc(360, 120, 42, 0, 7); x.fill();
    x.fillStyle = "#2b2d42"; x.beginPath(); x.moveTo(0, 250); x.lineTo(120, 150); x.lineTo(210, 230); x.lineTo(320, 140); x.lineTo(480, 260); x.lineTo(480, 320); x.lineTo(0, 320); x.fill();
    x.strokeStyle = "#fff"; x.lineWidth = 2;
    for (var i = 0; i < 9; i++) { x.beginPath(); x.moveTo(30 + i * 12, 40); x.lineTo(30 + i * 12, 110); x.stroke(); }
    x.fillStyle = "#fff"; x.font = "bold 44px sans-serif"; x.fillText("S3eidat", 150, 80);
    x.font = "16px sans-serif"; x.fillText("small, noisy, blocky", 160, 104);
    var small = canvas(160, 107), sx = small.getContext("2d");
    sx.imageSmoothingQuality = "high"; sx.drawImage(c, 0, 0, 160, 107);
    var d = sx.getImageData(0, 0, 160, 107), seed = 7;
    for (var k = 0; k < d.data.length; k++) { if (k % 4 === 3) continue; seed = (seed * 16807) % 2147483647; d.data[k] += (seed / 2147483647 - 0.5) * 26; }
    sx.putImageData(d, 0, 0);
    var blob = await new Promise(function (r) { small.toBlob(r, "image/jpeg", 0.45); });
    return new File([blob], "sample.jpg", { type: "image/jpeg" });
  }

  function save() {
    var it = selItem(); if (!it || !it.out) return;
    S3.download(it.out.blob, outName(it)); S3.done();
  }

  async function saveAll() {
    var outs = items.filter(function (i) { return i.out; });
    var files = [], used = {};
    for (var k = 0; k < outs.length; k++) {
      var n = outName(outs[k]), base = n, j = 2;
      while (used[n]) n = base.replace(/(\.[^.]+)$/, "-" + j++ + "$1");
      used[n] = 1;
      files.push({ name: n, data: new Uint8Array(await outs[k].out.blob.arrayBuffer()) });
    }
    S3.download(S3.zip(files), "enhanced-photos.zip"); S3.done();
  }

  /* ---------- self-test and demo ---------- */
  window.S3_SELFTEST = async function () {
    var cases = [[100, 32, 4], [517, 128, 12], [64, 64, 8], [1000, 192, 24], [33, 64, 8]];
    for (var c = 0; c < cases.length; c++) {
      var L = cases[c][0], T = Math.min(cases[c][1], L), O = cases[c][2];
      var st = Tiling.tileStarts(L, T, O);
      if (st[0] !== 0 || st[st.length - 1] + T !== L) return "tiles do not cover " + L;
      var ws = Tiling.axisWeights(st, T, L, O), sum = new Float64Array(L);
      st.forEach(function (s0, k) { for (var i = 0; i < T; i++) sum[s0 + i] += ws[k][i]; });
      for (var i = 0; i < L; i++) if (Math.abs(sum[i] - 1) > 1e-5) return "weights do not sum to 1 at " + i;
    }
    var p = Tiling.pad(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), 2, 2, 3, 3);
    if (p[2] !== 1 || p[5] !== 3 || p[6] !== 1 || p[8] !== 1 || p[9] !== 5) return "reflect pad is wrong";
    // A real model run: a small gradient through the Fast 4x model should come out 4x bigger and close to a smooth resize.
    var w = 24, h = 16, rgba = new Uint8ClampedArray(4 * w * h);
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) { var o = 4 * (y * w + x); rgba[o] = x * 10; rgba[o + 1] = y * 15; rgba[o + 2] = 128; rgba[o + 3] = 255; }
    var res = await enhanceRGBA(rgba, w, h, { upscale: 4, model: "fast", denoise: 0, sharpen: 0 });
    if (res.w !== 96 || res.h !== 64) return "wrong output size " + res.w + "x" + res.h;
    var diff = 0;
    for (var yy = 0; yy < 64; yy++) for (var xx = 0; xx < 96; xx++) {
      var q = 4 * (yy * 96 + xx);
      diff += Math.abs(res.rgba[q] - Math.min(255, (xx / 4) * 10)) + Math.abs(res.rgba[q + 1] - Math.min(255, (yy / 4) * 15));
    }
    diff /= 96 * 64 * 2;
    return diff < 12 ? true : "model output is off by " + diff.toFixed(1);
  };
  window.S3_DEMO = async function () {
    if (items.length) return;
    await addFiles([await sampleFile()]);
    await ready;
    await enhanceAll();
  };

  /* ---------- start ---------- */
  bindSettings(); bindViewer();
  S3.drop($("drop"), addFiles, { accept: "image/*,.jpg,.jpeg,.png,.webp,.bmp,.avif", multiple: true });
  document.addEventListener("paste", function (e) {
    var files = Array.prototype.slice.call((e.clipboardData && e.clipboardData.files) || []);
    if (files.length) addFiles(files);
  });
  $("go").addEventListener("click", enhanceAll);
  $("cancel").addEventListener("click", cancel);
  $("save").addEventListener("click", save);
  $("saveAll").addEventListener("click", saveAll);
  $("sample").addEventListener("click", async function () { addFiles([await sampleFile()]); });
  renderSettings(); showEmpty(); startWorker();
  window.__enhancer = { items: function () { return items; }, run: enhanceRGBA, addFiles: addFiles, enhanceAll: enhanceAll, spotPreview: spotPreview, sampleFile: sampleFile, select: select };
})();
