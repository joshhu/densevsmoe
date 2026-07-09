// app.js — 應用狀態、API 呼叫與播放引擎
import { buildDensePanel, buildHeatmap, buildMoePanel } from "./viz.js";

const $ = (id) => document.getElementById(id);
const state = {
  data: null, dense: null, moe: null, heat: null,
  playing: false, progress: 0, speed: 1, lastTs: 0, rafId: 0,
};

const STATE_TEXT = { idle: "未載入", loading: "載入中…", ready: "✓ 已載入" };

async function init() {
  const models = await (await fetch("/api/models")).json();
  for (const m of models) {
    const sel = $(m.kind === "dense" ? "dense-select" : "moe-select");
    sel.add(new Option(`${m.name}（${m.size_gb}GB）`, m.id));
  }
  // 預設用總參數同級距的配對：7B dense vs 7B MoE
  $("dense-select").value = "Qwen/Qwen2.5-7B-Instruct";
  $("moe-select").value = "allenai/OLMoE-1B-7B-0924-Instruct";
  $("btn-load-dense").onclick = () => requestLoad("dense");
  $("btn-load-moe").onclick = () => requestLoad("moe");
  $("btn-run").onclick = run;
  $("btn-play").onclick = togglePlay;
  $("speed").onchange = (e) => { state.speed = parseFloat(e.target.value); };
  $("scrub").oninput = (e) => seek(e.target.value / 1000);
  setInterval(pollStatus, 1000);
  pollStatus();
}

async function requestLoad(kind) {
  const model_id = $(`${kind}-select`).value;
  const r = await fetch("/api/load", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id }),
  });
  if (!r.ok) alert((await r.json()).detail);
  pollStatus();
}

async function pollStatus() {
  const status = await (await fetch("/api/status")).json();
  for (const kind of ["dense", "moe"]) {
    const st = status[$(`${kind}-select`).value] || { state: "idle", detail: "" };
    const badge = $(`${kind}-status`);
    badge.textContent = st.state === "error" ? `✗ ${st.detail}` : STATE_TEXT[st.state];
    badge.className = `status ${st.state}`;
  }
  const ready = (kind) =>
    (status[$(`${kind}-select`).value] || {}).state === "ready";
  $("btn-run").disabled = !(ready("dense") && ready("moe"));
}

async function run() {
  const btn = $("btn-run");
  btn.disabled = true;
  btn.textContent = "推論中…";
  try {
    const r = await fetch("/api/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dense_model: $("dense-select").value,
        moe_model: $("moe-select").value,
        sentence: $("sentence").value,
      }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    setupPlayback(await r.json());
  } catch (e) {
    alert(`推論失敗：${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "執行推論";
  }
}

function setupPlayback(data) {
  state.playing = false;
  cancelAnimationFrame(state.rafId);
  state.data = data;
  state.dense = buildDensePanel($("dense-panel"), data.dense);
  state.moe = buildMoePanel($("moe-panel"), data.moe);
  state.heat = buildHeatmap($("heatmap"), data.moe.n_experts);
  $("dense-title").textContent = `Dense — ${data.dense.model_id}`;
  $("moe-title").textContent = `MoE — ${data.moe.model_id}`;
  const d = data.dense, m = data.moe;
  $("dense-info").textContent =
    `⬛ 總參數 ${fmt(d.params_per_token)} — 每個 token 全部動用（100%）`;
  const pct = (100 * m.active_params_per_token / m.total_params).toFixed(1);
  $("moe-info").textContent =
    `🔀 總參數 ${fmt(m.total_params)}・${m.n_experts} 個 experts × ${m.n_layers} 層 ` +
    `— 每個 token 只啟用 ${m.top_k} 個 experts（≈${pct}% 參數）`;
  buildTokenStrip("dense-tokens", data.dense.tokens);
  buildTokenStrip("moe-tokens", data.moe.tokens);
  $("btn-play").disabled = false;
  state.progress = 0;
  play();
}

function buildTokenStrip(id, tokens) {
  const div = $(id);
  div.innerHTML = "";
  tokens.forEach((t, i) => {
    const span = document.createElement("span");
    span.textContent = t.trim() || "␣";
    span.onclick = () => seek(i / tokens.length);
    div.appendChild(span);
  });
}

// 主時間軸 progress ∈ [0,1]，兩側各自換算成 (token, 已亮層數, 層內進度)
function sidePos(p, T, L) {
  const pos = Math.min(p * T, T - 1e-6);
  const token = Math.floor(pos);
  const frac = pos - token;
  const litLayers = Math.floor(frac * L);
  return { token, litLayers, layerFrac: frac * L - litLayers };
}

function fmt(n) {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} 兆`;
  if (n >= 1e8) return `${(n / 1e8).toFixed(1)} 億`;
  if (n >= 1e4) return `${(n / 1e4).toFixed(0)} 萬`;
  return `${Math.round(n)}`;
}

function computeCounts(moe, token, litLayers) {
  const counts = new Array(moe.n_experts).fill(0);
  for (let t = 0; t <= token; t++) {
    const nL = t < token ? moe.n_layers : litLayers;
    for (let l = 0; l < nL; l++) {
      for (const { expert } of moe.routing[l][t]) counts[expert] += 1;
    }
  }
  return counts;
}

function highlightToken(id, current) {
  [...$(id).children].forEach((span, i) => {
    span.className = i < current ? "done" : i === current ? "current" : "";
  });
}

function render() {
  const d = state.data;
  if (!d) return;
  const dp = sidePos(state.progress, d.dense.tokens.length, d.dense.n_layers);
  const mp = sidePos(state.progress, d.moe.tokens.length, d.moe.n_layers);
  state.dense.update(dp.token, dp.litLayers, dp.layerFrac);
  state.moe.update(mp.token, mp.litLayers, mp.layerFrac);
  highlightToken("dense-tokens", dp.token);
  highlightToken("moe-tokens", mp.token);
  $("dense-counter").textContent =
    fmt(d.dense.params_per_token * (dp.token + dp.litLayers / d.dense.n_layers));
  $("moe-counter").textContent =
    fmt(d.moe.active_params_per_token * (mp.token + mp.litLayers / d.moe.n_layers));
  const pct = (100 * d.moe.active_params_per_token / d.moe.total_params).toFixed(1);
  $("moe-ratio").textContent =
    `（總參數 ${fmt(d.moe.total_params)}，每 token 僅動用 ${pct}%）`;
  state.heat.update(computeCounts(d.moe, mp.token, mp.litLayers));
  $("scrub").value = Math.round(state.progress * 1000);
}

function tick(ts) {
  if (!state.playing) return;
  const dt = ts - (state.lastTs || ts);
  state.lastTs = ts;
  const durMs = state.data.dense.tokens.length * 1400; // 每 token 約 1.4 秒（1x）
  state.progress = Math.min(1, state.progress + (dt * state.speed) / durMs);
  render();
  if (state.progress >= 1) {
    state.playing = false;
    $("btn-play").textContent = "↺ 重播";
    return;
  }
  state.rafId = requestAnimationFrame(tick);
}

function play() {
  cancelAnimationFrame(state.rafId);
  if (state.progress >= 1) state.progress = 0;
  state.playing = true;
  state.lastTs = 0;
  $("btn-play").textContent = "⏸ 暫停";
  state.rafId = requestAnimationFrame(tick);
}

function togglePlay() {
  if (state.playing) {
    state.playing = false;
    $("btn-play").textContent = "▶ 播放";
  } else {
    play();
  }
}

function seek(p) {
  state.progress = Math.min(Math.max(p, 0), 0.9999);
  if (!state.playing) render();
}

init();
