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

function moeColor(w) {
  const a = (0.35 + 0.65 * Math.min(1, w * 3)).toFixed(3);
  return `rgba(60, 220, 255, ${a})`;
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
  const cells = buildGrid(svg, L, E);
  return {
    update(token, litLayers, frac) {
      for (let l = 0; l < L; l++) {
        for (let e = 0; e < E; e++) cells[l][e].setAttribute("fill", DIM);
        const lit = l < litLayers ? 1 : (l === litLayers ? frac : 0);
        if (lit <= 0) continue;
        for (const { expert, weight } of data.routing[l][token]) {
          cells[l][expert].setAttribute("fill", moeColor(weight * lit));
        }
      }
    },
  };
}

export function buildHeatmap(svg, nExperts) {
  const cols = nExperts;
  const W = 640, cw = (W - 8) / cols, H = 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";
  const cells = [];
  for (let e = 0; e < cols; e++) {
    cells.push(el("rect", {
      x: 4 + e * cw, y: 2, width: Math.max(cw - 1, 1), height: H - 4,
      rx: 2, fill: DIM,
    }, svg));
  }
  return {
    update(counts) {
      const max = Math.max(1, ...counts);
      counts.forEach((c, e) => {
        const v = c / max;
        cells[e].setAttribute(
          "fill", c === 0 ? DIM : `hsl(190, 90%, ${Math.round(12 + 43 * v)}%)`);
      });
    },
  };
}
