"use strict";
/* S3eidat Enhancer UI. Talks to Python through window.pywebview.api (desktop) or, in --browser dev mode,
   through POST /<token>/api/<method>. */

const $ = (id) => document.getElementById(id);
const BASE = location.pathname.split("/ui/")[0];
const BROWSER_MODE = new URLSearchParams(location.search).has("browser");

function call(name, ...args) {
  if (!BROWSER_MODE && window.pywebview && window.pywebview.api) return window.pywebview.api[name](...args);
  return fetch(`${BASE}/api/${name}`, { method: "POST", body: JSON.stringify(args) }).then((r) => r.json());
}

const MODEL_TEXT = {
  standard: "Real-ESRGAN: best for photos. Cleans JPEG blocks and blur while it upscales.",
  art: "Real-ESRGAN anime: flat colours and clean lines for art, anime, logos, screenshots.",
  fast: "Real-ESRGAN general: about 5x faster, lighter touch. Good for big batches.",
};

const state = {
  info: null,
  settings: null,
  outDir: "",
  items: [],          // {key, path, name, thumb, size, jobId, job, error}
  sel: null,          // selected key
  view: { mode: "empty", w: 0, h: 0, zoom: 1, x: 0, y: 0, split: 0.5, fit: true, live: null },
  previewBusy: false,
};

/* ---------- helpers ---------- */
const pct = (v) => `${Math.round(v * 100)}%`;
const fmtSize = (s) => (s ? `${s[0]}×${s[1]}` : "");
const selItem = () => state.items.find((i) => i.key === state.sel);
function mp(s) { return s ? ((s[0] * s[1]) / 1e6).toFixed(1) + " MP" : ""; }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

/* ---------- settings panel ---------- */
const SLIDERS = ["detail", "denoise", "sharpen", "faces"];

function renderSettings() {
  const s = state.settings;
  for (const [segId, key] of [["upscaleSeg", "upscale"], ["modelSeg", "model"], ["qualitySeg", "quality"]]) {
    for (const b of $(segId).children) b.classList.toggle("on", String(s[key]) === b.dataset.v);
  }
  $("modelGroup").style.opacity = s.upscale > 1 ? 1 : 0.4;
  $("modelText").textContent = MODEL_TEXT[s.model];
  for (const k of SLIDERS) {
    const input = $(k), out = $(k + "Out");
    input.value = Math.round(s[k] * 100);
    input.style.setProperty("--p", `${input.value}%`);
    const off = s[k] === 0 && k !== "detail";
    out.textContent = off ? "Off" : pct(s[k]);
    out.classList.toggle("on", !off);
  }
  $("format").value = s.format;
  $("outDir").textContent = state.outDir;
  $("outDir").title = state.outDir;
  updateSizeText();
  updateButtons();
}

function updateSizeText() {
  const it = selItem();
  const s = state.settings;
  if (!it || !it.size) { $("sizeText").textContent = s.upscale > 1 ? `Output is ${s.upscale}× wider and taller` : "Keeps the original size"; return; }
  const out = [it.size[0] * s.upscale, it.size[1] * s.upscale];
  $("sizeText").textContent = `${fmtSize(it.size)} → ${fmtSize(out)} (${mp(out)})`;
}

function updateButtons() {
  const ready = state.items.filter((i) => !i.jobId || ["done", "error", "cancelled"].includes(i.job?.status));
  const nothing = state.settings.upscale === 1 && !state.settings.denoise && !state.settings.sharpen && !state.settings.faces;
  const go = $("goBtn");
  const todo = state.items.filter((i) => !i.jobId && !i.error).length;
  const redo = selItem() && ["done", "error", "cancelled"].includes(selItem().job?.status);
  go.disabled = nothing || (!todo && !redo);
  go.textContent = nothing ? "Pick at least one step" : todo ? `Enhance ${todo} photo${todo > 1 ? "s" : ""}` : redo ? "Run again with these settings" : "Enhance";
  const it = selItem();
  $("autoBtn").disabled = !it || !!it.error;
  $("previewBtn").disabled = !it || !!it.error || nothing || state.previewBusy;
  $("clearBtn").disabled = !state.items.some((i) => ["done", "error", "cancelled"].includes(i.job?.status));
  $("count").textContent = state.items.length ? state.items.length : "";
  void ready;
}

let saveTimer = 0;
function changed(livePreview = true) {
  renderSettings();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => call("save_prefs", state.settings, state.outDir), 500);
  if (livePreview && state.view.mode === "live") schedulePreview();
}

function bindSettings() {
  for (const [segId, key, num] of [["upscaleSeg", "upscale", true], ["modelSeg", "model"], ["qualitySeg", "quality"]]) {
    $(segId).addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      state.settings[key] = num ? Number(b.dataset.v) : b.dataset.v;
      $("reasons").hidden = true;
      changed();
    });
  }
  for (const k of SLIDERS) {
    $(k).addEventListener("input", (e) => { state.settings[k] = Number(e.target.value) / 100; $("reasons").hidden = true; changed(); });
  }
  $("format").addEventListener("change", (e) => { state.settings.format = e.target.value; changed(false); });
  $("outBtn").addEventListener("click", async () => { const d = await call("pick_folder"); if (d) { state.outDir = d; changed(false); } });
  $("outOpen").addEventListener("click", () => call("open_folder", state.outDir));
  $("autoBtn").addEventListener("click", runAutopilot);
  $("previewBtn").addEventListener("click", () => livePreview(0.5, 0.5));
  $("goBtn").addEventListener("click", enhance);
}

async function runAutopilot() {
  const it = selItem(); if (!it) return;
  setBusy("Autopilot is looking at the photo…");
  const res = await call("autopilot", it.path);
  setBusy(null);
  if (res.error) { toast(res.error); return; }
  Object.assign(state.settings, res.settings);
  if (!state.info.faces_available) state.settings.faces = 0;
  const ul = $("reasons"); ul.innerHTML = "";
  for (const r of res.reasons) ul.append(el("li", null, r));
  ul.hidden = false;
  changed();
}

/* ---------- queue ---------- */
async function addFiles(paths) {
  if (!paths || !paths.length) return;
  const known = new Set(state.items.map((i) => i.path));
  const fresh = paths.filter((p) => !known.has(p));
  if (!fresh.length) return;
  setBusy("Reading photos…");
  const staged = await call("stage", fresh);
  setBusy(null);
  for (const s of staged) state.items.push({ key: s.path, ...s, jobId: null, job: null });
  if (!state.sel && state.items.length) select(state.items[0].key);
  renderList();
  updateButtons();
}

function statusText(it) {
  if (it.error) return ["error", it.error];
  const j = it.job;
  if (!j) return ["", `${fmtSize(it.size)} · ready`];
  if (j.status === "queued") return ["", "Waiting…"];
  if (j.status === "running") return ["", `${j.stage || "Starting"} ${pct(j.progress)}`];
  if (j.status === "done") {
    const faces = j.faces != null ? ` · ${j.faces} face${j.faces === 1 ? "" : "s"}` : "";
    return ["done", `${fmtSize(j.out_size)} · ${j.seconds}s${faces}`];
  }
  if (j.status === "cancelled") return ["", "Cancelled"];
  return ["error", j.error || "Failed"];
}

function renderList() {
  const ul = $("list");
  $("emptyList").hidden = state.items.length > 0;
  const existing = new Map([...ul.children].map((li) => [li.dataset.key, li]));
  for (const it of state.items) {
    let li = existing.get(it.key);
    if (!li) {
      li = el("li", "item"); li.dataset.key = it.key;
      const img = el("img"); img.src = it.thumb || ""; img.alt = "";
      const meta = el("div", "meta");
      meta.append(el("div", "nm", it.name), el("div", "st"), el("div", "bar"));
      meta.lastChild.append(el("i"));
      const x = el("button", "x", "×"); x.title = "Remove";
      x.addEventListener("click", (e) => { e.stopPropagation(); removeItem(it.key); });
      li.append(img, meta, x);
      li.addEventListener("click", () => select(it.key));
      ul.append(li);
    }
    existing.delete(it.key);
    li.classList.toggle("sel", it.key === state.sel);
    const [cls, txt] = statusText(it);
    const st = li.querySelector(".st"); st.className = "st " + cls; st.textContent = txt; st.title = txt;
    const j = it.job;
    li.querySelector(".bar").style.visibility = j && ["running", "queued"].includes(j.status) ? "visible" : "hidden";
    li.querySelector(".bar i").style.width = pct(j ? j.progress : 0);
    li.querySelector(".x").title = j && ["running", "queued"].includes(j.status) ? "Cancel" : "Remove";
  }
  for (const li of existing.values()) li.remove();
}

async function removeItem(key) {
  const it = state.items.find((i) => i.key === key); if (!it) return;
  if (it.jobId && ["running", "queued"].includes(it.job?.status)) { await call("cancel", it.jobId); return; }
  if (it.jobId) await call("remove", it.jobId);
  state.items = state.items.filter((i) => i.key !== key);
  if (state.sel === key) { state.sel = null; if (state.items[0]) select(state.items[0].key); else showEmpty(); }
  renderList(); updateButtons();
}

async function enhance() {
  const todo = state.items.filter((i) => !i.jobId && !i.error);
  const it = selItem();
  if (!todo.length && it && it.jobId) {
    await call("retry", it.jobId, state.settings);
  } else {
    const jobs = await call("add", todo.map((i) => i.path), state.settings, state.outDir);
    for (const j of jobs) { const i = state.items.find((x) => x.path === j.src && !x.jobId); if (i) { i.jobId = j.id; i.job = j; } }
  }
  renderList(); updateButtons(); poll();
}

let pollTimer = 0;
async function poll() {
  clearTimeout(pollTimer);
  const jobs = await call("jobs");
  const byId = new Map(jobs.map((j) => [j.id, j]));
  let active = false;
  for (const it of state.items) {
    if (!it.jobId) continue;
    const j = byId.get(it.jobId); if (!j) continue;
    const wasDone = it.job?.status === "done" && it.job?.out === j.out;
    it.job = j;
    if (["queued", "running"].includes(j.status)) active = true;
    if (j.status === "done" && !wasDone && it.key === state.sel) showCompare(it);
    if (j.notice) showNotice(j.notice);
  }
  renderList(); updateButtons();
  const s = selItem();
  if (s && s.job && s.job.status === "running") $("stageInfo").textContent = `${s.job.stage} ${pct(s.job.progress)}`;
  if (active) pollTimer = setTimeout(poll, 250);
}

function select(key) {
  state.sel = key;
  renderList(); updateSizeText(); updateButtons();
  $("reasons").hidden = true;
  const it = selItem(); if (!it) return;
  $("stageTitle").textContent = it.name;
  if (it.job && it.job.status === "done") showCompare(it); else showOriginal(it);
}

/* ---------- viewer ---------- */
const V = state.view;

function setBusy(text) { $("busy").hidden = !text; if (text) $("busyText").textContent = text; }
let toastTimer = 0;
function toast(msg) { showNotice(msg); clearTimeout(toastTimer); toastTimer = setTimeout(() => showNotice(state.info?.notice || ""), 6000); }
function showNotice(msg) { $("notice").hidden = !msg; $("notice").textContent = msg; }

function showEmpty() {
  V.mode = "empty";
  $("viewerHint").hidden = false; $("divider").hidden = true; $("tagBefore").hidden = $("tagAfter").hidden = true;
  $("imgBefore").removeAttribute("src"); $("imgAfter").removeAttribute("src");
  $("liveBar").hidden = true; $("clickTip").hidden = true;
  $("stageTitle").textContent = "No photo selected"; $("stageInfo").textContent = "";
}

function loadImages(beforeUrl, afterUrl, w, h) {
  return new Promise((resolve) => {
    let n = afterUrl ? 2 : 1;
    const done = () => { if (--n === 0) resolve(); };
    const b = $("imgBefore"), a = $("imgAfter");
    b.onload = b.onerror = done; b.src = beforeUrl;
    if (afterUrl) { a.onload = a.onerror = done; a.src = afterUrl; } else a.removeAttribute("src");
    V.w = w; V.h = h;
  });
}

async function showOriginal(it) {
  V.mode = "original"; V.live = null;
  $("viewerHint").hidden = true; $("liveBar").hidden = true;
  $("divider").hidden = true; $("tagBefore").hidden = $("tagAfter").hidden = true;
  $("stageInfo").textContent = it.size ? `${fmtSize(it.size)} · ${mp(it.size)}` : "";
  const res = await call("original", it.path);
  if (state.sel !== it.key) return;
  await loadImages(res.url, null, res.size[0], res.size[1]);
  $("layerAfter").style.clipPath = "inset(0 0 0 100%)";
  $("clickTip").hidden = false;
  $("viewer").classList.add("clickable");
  fit();
}

async function showCompare(it) {
  const res = await call("compare", it.jobId);
  if (!res.before || state.sel !== it.key) return;
  V.mode = "compare"; V.live = null;
  $("viewerHint").hidden = true; $("liveBar").hidden = true; $("clickTip").hidden = true;
  $("viewer").classList.remove("clickable");
  await loadImages(res.before, res.after, res.out_size[0], res.out_size[1]);
  $("divider").hidden = false; $("tagBefore").hidden = $("tagAfter").hidden = false;
  $("stageInfo").textContent = `${fmtSize(res.in_size)} → ${fmtSize(res.out_size)} · ${it.job.seconds}s`;
  fit();
}

let previewTimer = 0, previewSeq = 0;
function schedulePreview() { clearTimeout(previewTimer); previewTimer = setTimeout(() => V.live && livePreview(V.live.cx, V.live.cy), 450); }

async function livePreview(cx, cy) {
  const it = selItem(); if (!it) return;
  const seq = ++previewSeq;
  state.previewBusy = true; updateButtons();
  setBusy("Previewing this spot…");
  const res = await call("preview", it.path, state.settings, cx, cy);
  if (seq !== previewSeq) return;
  state.previewBusy = false; setBusy(null); updateButtons();
  if (res.error) { toast(res.error); return; }
  V.mode = "live"; V.live = { cx, cy };
  $("viewerHint").hidden = true; $("clickTip").hidden = true;
  $("viewer").classList.remove("clickable");
  await loadImages(res.before, res.after, res.size[0], res.size[1]);
  $("divider").hidden = false; $("tagBefore").hidden = $("tagAfter").hidden = false;
  $("liveBar").hidden = false;
  $("liveText").textContent = `Preview of one spot · ${res.seconds}s on ${res.device}`;
  if (res.notice) showNotice(res.notice);
  V.fit = true; fit();
}

function apply() {
  const vw = $("viewer").clientWidth;
  for (const img of [$("imgBefore"), $("imgAfter")]) {
    img.style.width = `${V.w * V.zoom}px`;
    img.style.height = `${V.h * V.zoom}px`;
    img.style.transform = `translate(${V.x}px, ${V.y}px)`;
  }
  const splitX = V.split * vw;
  if (V.mode !== "original") $("layerAfter").style.clipPath = `inset(0 0 0 ${splitX}px)`;
  $("divider").style.left = `${splitX}px`;
  for (const b of $("zoomSeg").children) {
    b.classList.toggle("on", b.dataset.zoom === "fit" ? V.fit : !V.fit && Math.abs(V.zoom - Number(b.dataset.zoom)) < 1e-3);
  }
  // 1:1 or bigger shows real pixels; smaller uses smooth scaling
  $("imgBefore").style.imageRendering = V.zoom >= 2 ? "pixelated" : "auto";
  $("imgAfter").style.imageRendering = V.zoom >= 2 ? "pixelated" : "auto";
}

function fit() {
  if (!V.w) return;
  const vw = $("viewer").clientWidth, vh = $("viewer").clientHeight;
  V.zoom = Math.min(vw / V.w, vh / V.h, V.mode === "live" ? 4 : 1);
  V.x = (vw - V.w * V.zoom) / 2; V.y = (vh - V.h * V.zoom) / 2; V.fit = true;
  apply();
}

function zoomTo(z, px, py) {
  const vw = $("viewer").clientWidth, vh = $("viewer").clientHeight;
  px ??= vw / 2; py ??= vh / 2;
  const ix = (px - V.x) / V.zoom, iy = (py - V.y) / V.zoom;
  V.zoom = Math.min(16, Math.max(0.02, z));
  V.x = px - ix * V.zoom; V.y = py - iy * V.zoom; V.fit = false;
  apply();
}

function bindViewer() {
  const vr = $("viewer");
  let drag = null;
  vr.addEventListener("wheel", (e) => {
    if (V.mode === "empty") return;
    e.preventDefault();
    const r = vr.getBoundingClientRect();
    zoomTo(V.zoom * Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  vr.addEventListener("pointerdown", (e) => {
    if (V.mode === "empty" || e.button !== 0) return;
    const onDivider = e.target.closest("#divider");
    drag = { kind: onDivider ? "split" : "pan", sx: e.clientX, sy: e.clientY, x: V.x, y: V.y, moved: false };
    vr.setPointerCapture(e.pointerId);
    if (!onDivider) vr.classList.add("panning");
  });
  vr.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.kind === "split") {
      const r = vr.getBoundingClientRect();
      V.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    } else { V.x = drag.x + dx; V.y = drag.y + dy; V.fit = false; }
    apply();
  });
  vr.addEventListener("pointerup", (e) => {
    vr.classList.remove("panning");
    const d = drag; drag = null;
    if (!d || d.moved || d.kind !== "pan" || V.mode !== "original") return;
    const r = vr.getBoundingClientRect();
    const cx = (e.clientX - r.left - V.x) / (V.w * V.zoom), cy = (e.clientY - r.top - V.y) / (V.h * V.zoom);
    if (cx >= 0 && cx <= 1 && cy >= 0 && cy <= 1) livePreview(cx, cy);
  });
  vr.addEventListener("dblclick", (e) => {
    if (V.mode === "empty" || V.mode === "original") return;
    const r = vr.getBoundingClientRect();
    if (V.fit) zoomTo(1, e.clientX - r.left, e.clientY - r.top); else fit();
  });
  $("zoomSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.zoom === "fit") fit(); else zoomTo(Number(b.dataset.zoom));
  });
  $("liveBack").addEventListener("click", () => { previewSeq++; const it = selItem(); if (it) showOriginal(it); });
  window.addEventListener("resize", () => (V.fit ? fit() : apply()));
  // hold B to see the whole "before" image, hold A for the whole "after" image
  window.addEventListener("keydown", (e) => {
    if (e.repeat || !["compare", "live"].includes(V.mode) || e.target.tagName === "INPUT") return;
    if (e.key === "b") { V._split = V.split; V.split = 1; apply(); }
    if (e.key === "a") { V._split = V.split; V.split = 0; apply(); }
  });
  window.addEventListener("keyup", (e) => { if ((e.key === "b" || e.key === "a") && V._split != null) { V.split = V._split; V._split = null; apply(); } });
}

/* ---------- drag & drop ---------- */
function bindDrop() {
  let depth = 0;
  document.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; $("dropOverlay").hidden = false; });
  document.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; $("dropOverlay").hidden = true; } });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => { e.preventDefault(); depth = 0; $("dropOverlay").hidden = true; });
  window.onFilesDropped = (paths) => addFiles(paths);
}

/* ---------- start ---------- */
async function start() {
  const info = await call("ready");
  state.info = info; state.settings = info.settings; state.outDir = info.out_dir;
  $("ver").textContent = `v${info.version}`;
  $("device").textContent = info.gpu ? "GPU · DirectML" : "CPU";
  $("device").classList.toggle("gpu", info.gpu);
  showNotice(info.notice);
  if (!info.faces_available) { $("facesRow").classList.add("disabled"); $("faces").disabled = true; state.settings.faces = 0; }
  const ul = $("creditList");
  for (const c of info.credits) {
    const li = el("li"); li.append(el("b", null, c.credit), el("span", "lic", `${c.license} · ${c.project}`));
    if (c.note) li.append(el("div", "note", c.note));
    ul.append(li);
  }
  $("creditsBtn").addEventListener("click", () => $("credits").showModal());
  const pick = async () => addFiles(await call("pick_files"));
  $("addBtn").addEventListener("click", pick); $("addBtn2").addEventListener("click", pick);
  $("clearBtn").addEventListener("click", async () => {
    const gone = new Set(await call("clear_finished"));
    state.items = state.items.filter((i) => !gone.has(i.jobId));
    if (!selItem()) { state.sel = null; if (state.items[0]) select(state.items[0].key); else showEmpty(); }
    renderList(); updateButtons();
  });
  bindSettings(); bindViewer(); bindDrop();
  renderSettings(); renderList(); showEmpty();
  window.__enhancer = { state, addFiles, select, livePreview, poll }; // for automated checks
}

if (BROWSER_MODE) start();
else if (window.pywebview && window.pywebview.api) start();
else window.addEventListener("pywebviewready", start, { once: true });
