# MoE Expert 身分色視覺化重寫 — 設計文件

日期：2026-07-09
狀態：已核准
背景：使用者反映原 MoE 面板「看不出是 MoE」——顏色單一、未標 experts 數量、與 dense 側視覺結構無差異。

## 目標

讓 MoE 面板一眼可辨：每個 expert 有固定身分色，稀疏彩色點陣對比 dense 的單色橘牆；
experts 數量、top-k、啟用比例以大字標註；熱度圖改為身分色長條圖。

## 範圍

只動前端：`static/viz.js`、`static/app.js`、`static/index.html`、`static/style.css`。
後端 API、dense 面板繪製、播放引擎、token 條、參數計數器不變。

## 設計

### 色彩系統（viz.js）

- `expertHue(e, E) = round(e / E * 360)`：expert 依編號均分色相環，跨層／跨 token／熱度圖永遠同色。
- 亮格顏色 `hsl(hue, 78%, L)`，L = 30 + 38 × min(1, weight×2.5) × lit——色相=身分、亮度=router 權重×點亮進度。
- 未選中格用更暗底色 `#1b1f2b`（比 dense 側的 `#232838` 暗），拉大稀疏對比。
- Dense 側維持單色橘，不動。

### MoE 面板（viz.js `buildMoePanel` 重寫，不再共用 `buildGrid`）

- 上方 expert 欄標 `E1、E5、E9…`（E≤32 每 4 個標一個；>32 每 8 個標一個），top 邊距 16px。
- 每層列尾右側標 `8/32`（`top_k/n_experts`，動態）。
- 其餘幾何同原版（層由下往上、viewBox 寬 640）。

### 熱度圖（viz.js `buildHeatmap` 重寫）

- 由等高色塊改為**長條圖**：高度 = 該 expert 累計被選次數／最大值，顏色 = 該 expert 身分色。
- 下方同樣標 E1、E5… 欄標；未被選過的 expert 顯示 2px 底線。

### 標註（index.html + app.js + style.css）

- MoE 面板說明列改為 id=`moe-info`，`setupPlayback` 時動態填入：
  `🔀 {E} 個 experts × {L} 層 — 每個 token 每層只啟用 {k} 個（{k/E %}）`，
  以 MoE 主題色、粗體、0.92rem 顯示（換 OLMoE 自動變 64/8/12.5%）。
- 熱度圖下方加圖例 `#moe-legend`：
  「每格一個 expert・顏色 = expert 身分（跨層同色可追蹤）・亮起 = 該層 router 選中・下方長條 = 各 expert 累計被選次數」。
- style.css 新增 `.col-label`（SVG 欄標）、`.legend`、`#moe-info` 樣式。

## 驗證

- 既有 pytest 套件不受影響（純前端）；跑 e2e 確認斷言（`#moe-panel rect`、`#heatmap rect`、計數器）仍成立。
- 以 Chrome 實開頁面執行推論，逐項核對：彩色稀疏點陣、E 欄標、8/32 列尾、資訊列數字正確、熱度長條圖身分色。
