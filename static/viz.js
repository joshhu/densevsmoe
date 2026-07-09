// viz.js — SVG 點亮動畫繪製
const NS = "http://www.w3.org/2000/svg";
const DIM = "#232838";

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function layout(svg, nLayers) {
  const W = 640;
  const rowH = Math.max(8, Math.min(22, 500 / nLayers));
  const H = nLayers * (rowH + 3) + 8;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  // layer 0 在最底層（資料由下往上流）
  return { W, rowH, left: 40, yTop: (i) => H - 4 - (i + 1) * (rowH + 3) + 3 };
}

function buildGrid(svg, nLayers, nCols) {
  svg.innerHTML = "";
  const { W, rowH, left, yTop } = layout(svg, nLayers);
  const cw = (W - left - 8) / nCols;
  const cells = [];
  for (let l = 0; l < nLayers; l++) {
    const label = el("text", { x: 4, y: yTop(l) + rowH * 0.78, class: "layer-label" }, svg);
    label.textContent = `L${l + 1}`;
    const row = [];
    for (let c = 0; c < nCols; c++) {
      row.push(el("rect", {
        x: left + c * cw, y: yTop(l),
        width: Math.max(cw - 1, 1), height: rowH, rx: 2, fill: DIM,
      }, svg));
    }
    cells.push(row);
  }
  return cells;
}

function denseColor(v) {
  if (v <= 0.02) return DIM;
  const a = (0.25 + 0.75 * v).toFixed(3);
  return `rgba(255, ${Math.round(130 + 70 * v)}, 40, ${a})`;
}

const DIM_OFF = "#1b1f2b"; // MoE 未選中格：比 dense 底色更暗，拉大稀疏對比

// expert 依編號均分色相環：同一 expert 跨層／跨 token／熱度圖永遠同色
export function expertHue(e, nExperts) {
  return Math.round((e / nExperts) * 360);
}

// 色相 = expert 身分；亮度 = router 權重 × 點亮進度
function expertColor(e, nExperts, weight, lit) {
  const s = Math.min(1, weight * 2.5) * lit;
  if (s <= 0.02) return DIM_OFF;
  return `hsl(${expertHue(e, nExperts)}, 78%, ${Math.round(30 + 38 * s)}%)`;
}

function colLabelStep(nExperts) {
  return nExperts > 32 ? 8 : 4;
}

// litLayers = 已完全點亮層數；frac = 下一層的點亮進度 0..1
export function buildDensePanel(svg, data) {
  const L = data.n_layers;
  const B = data.activations[0][0].length;
  const cells = buildGrid(svg, L, B);
  return {
    update(token, litLayers, frac) {
      for (let l = 0; l < L; l++) {
        const lit = l < litLayers ? 1 : (l === litLayers ? frac : 0);
        for (let b = 0; b < B; b++) {
          cells[l][b].setAttribute(
            "fill", denseColor(data.activations[l][token][b] * lit));
        }
      }
    },
  };
}

export function buildMoePanel(svg, data) {
  const L = data.n_layers;
  const E = data.n_experts;
  const K = data.top_k;
  svg.innerHTML = "";
  const W = 640;
  const rowH = Math.max(8, Math.min(22, 480 / L));
  const top = 16, left = 40, right = 44;
  const H = top + L * (rowH + 3) + 8;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const cw = (W - left - right) / E;
  const yTop = (i) => H - 4 - (i + 1) * (rowH + 3) + 3;

  const step = colLabelStep(E);
  for (let e = 0; e < E; e += step) {
    const t = el("text", { x: left + e * cw + cw / 2, y: top - 4,
      class: "col-label", "text-anchor": "middle" }, svg);
    t.textContent = `E${e + 1}`;
  }

  const cells = [];
  for (let l = 0; l < L; l++) {
    const label = el("text", { x: 4, y: yTop(l) + rowH * 0.78, class: "layer-label" }, svg);
    label.textContent = `L${l + 1}`;
    const kLabel = el("text", { x: W - right + 6, y: yTop(l) + rowH * 0.78,
      class: "layer-label" }, svg);
    kLabel.textContent = `${K}/${E}`;
    const row = [];
    for (let e = 0; e < E; e++) {
      row.push(el("rect", { x: left + e * cw, y: yTop(l),
        width: Math.max(cw - 1, 1), height: rowH, rx: 2, fill: DIM_OFF }, svg));
    }
    cells.push(row);
  }
  return {
    update(token, litLayers, frac) {
      for (let l = 0; l < L; l++) {
        for (let e = 0; e < E; e++) cells[l][e].setAttribute("fill", DIM_OFF);
        const lit = l < litLayers ? 1 : (l === litLayers ? frac : 0);
        if (lit <= 0) continue;
        for (const { expert, weight } of data.routing[l][token]) {
          cells[l][expert].setAttribute("fill", expertColor(expert, E, weight, lit));
        }
      }
    },
  };
}

// 熱度圖 = 身分色長條圖：高度 = 累計被選次數／最大值，顏色 = expert 身分色
export function buildHeatmap(svg, nExperts) {
  const W = 640, H = 46, base = H - 12;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";
  const cw = (W - 8) / nExperts;
  const step = colLabelStep(nExperts);
  const bars = [];
  for (let e = 0; e < nExperts; e++) {
    bars.push(el("rect", { x: 4 + e * cw, y: base - 2,
      width: Math.max(cw - 1, 1), height: 2, rx: 1, fill: DIM_OFF }, svg));
    if (e % step === 0) {
      const t = el("text", { x: 4 + e * cw + cw / 2, y: H - 2,
        class: "col-label", "text-anchor": "middle" }, svg);
      t.textContent = `E${e + 1}`;
    }
  }
  return {
    update(counts) {
      const max = Math.max(1, ...counts);
      counts.forEach((c, e) => {
        const h = c === 0 ? 2 : 2 + (base - 8) * (c / max);
        bars[e].setAttribute("height", h);
        bars[e].setAttribute("y", base - h);
        bars[e].setAttribute("fill",
          c === 0 ? DIM_OFF : `hsl(${expertHue(e, counts.length)}, 78%, 55%)`);
      });
    },
  };
}
