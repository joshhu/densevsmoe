# Dense vs MoE 推論視覺化 — 設計文件

日期：2026-07-09
狀態：已核准

## 目標

一個網頁工具：從策展的 HuggingFace 模型清單各選一個 dense 與一個 MoE 模型，輸入一個句子，
以左右同步對比的動畫視覺化句子通過兩個模型時各層「點亮」的過程，
示範 dense（每 token 動用全部參數）與 MoE（每 token 只動用少數 experts）的架構差異。

點亮資料來自**真實模型推論**（PyTorch + transformers），非前端模擬。

## 使用情境

- 教學／簡報示範 dense vs MoE 差異（深色主題，投影友善）。
- 執行環境：DGX Spark（GB10 Blackwell，128GB UMA，arm64，CUDA 13）。
- 介面語言：繁體中文（台灣用語）。

## 架構

```
densevsmoe/
├── server/            # Python（uv venv）
│   ├── main.py        # FastAPI：serve 前端靜態檔 + API
│   ├── models.py      # 模型載入管理（策展清單、記憶體控管、狀態）
│   └── extract.py     # 推論 + activation / routing 擷取（核心邏輯）
├── static/            # 前端（原生 JS + SVG/Canvas，無 build 工具）
│   ├── index.html
│   ├── app.js         # 播放控制、API 呼叫、應用狀態
│   └── viz.js         # 點亮動畫繪製
├── tests/             # pytest 單元 + API 測試、Playwright e2e
└── docs/superpowers/specs/
```

資料流：

1. 前端 `GET /api/models` 取得策展清單與各模型載入狀態。
2. 使用者選一個 dense + 一個 MoE，`POST /api/load` 觸發載入（非同步），前端輪詢 `GET /api/status` 顯示進度。
3. 使用者輸入句子，`POST /api/infer`：後端對兩個模型各跑一次前向傳播，擷取每層 × 每 token 資料，回傳單一 JSON。
4. 前端以該 JSON 播放動畫；暫停、倒帶、調速、跳轉皆為前端重播，不重新推論。

## API

- `GET /api/models` → `[{id, name, kind: "dense"|"moe", size_gb, n_layers, n_experts?, top_k?, loaded}]`
- `POST /api/load` `{model_id}` → `202 {status}`；載入於背景執行緒進行。
- `GET /api/status` → `{model_id: {state: "idle"|"downloading"|"loading"|"ready"|"error", detail}}`
- `POST /api/infer` `{dense_model, moe_model, sentence}` →

```jsonc
{
  "dense": {
    "model_id": "...", "tokens": ["今", "天", ...],   // 兩模型 tokenizer 不同，各自回傳
    "n_layers": 24,
    // activations[layer][token] = 長度 32 的 neuron-bin 強度陣列（0..1 正規化）
    "activations": [[[...32 floats...], ...], ...],
    "params_per_token": 494000000                      // 每 token 動用參數量
  },
  "moe": {
    "model_id": "...", "tokens": [...], "n_layers": 16,
    "n_experts": 64, "top_k": 8,
    // routing[layer][token] = [{expert: int, weight: float}, ...top_k 個]
    "routing": [[[{"expert": 3, "weight": 0.31}, ...], ...], ...],
    "active_params_per_token": 1300000000,
    "total_params": 6900000000
  }
}
```

## 策展模型清單

| 類型 | 模型 | 約略大小 (bf16) | 備註 |
|------|------|------|------|
| Dense | `openai-community/gpt2` | 0.25GB | 12 層，最快 |
| Dense | `Qwen/Qwen2.5-0.5B-Instruct` | 1GB | 24 層，中文友善（預設） |
| MoE | `ibm-granite/granite-3.1-1b-a400m-instruct` | 2.6GB | 32 experts，輕量（預設） |
| MoE | `allenai/OLMoE-1B-7B-0924-Instruct` | 14GB | 64 experts 選 8，對比最震撼 |

任意 HF model ID 不在範圍內（MoE routing 擷取需逐架構適配）。

## 資料擷取（extract.py）

- **Dense**：forward hook 掛在每層 MLP 的中間激活（act_fn 輸出），取絕對值後沿 hidden 維度分成 32 個
  neuron-bins 取平均，跨全句正規化到 0..1。
- **MoE**：`output_router_logits=True`（OLMoE、Granite MoE 皆支援）；每層每 token 取 softmax 後
  top-k 的 expert 編號與權重。若該架構旗標不可用，退回 hook 在 gate/router module 上。
- 推論以 `torch.no_grad()`、bf16、CUDA 執行；單一句子長度上限 64 tokens（超過回 400 與中文錯誤訊息）。

## 記憶體控管（models.py）

- 同時最多常駐一個 dense + 一個 MoE；切換模型時先 `del` 舊模型並 `torch.cuda.empty_cache()`。
- 載入前檢查可用記憶體（UMA：讀 `/proc/meminfo` 的 MemAvailable），不足則回明確中文錯誤，
  不嘗試載入。預設組合（Qwen 0.5B + Granite MoE）約 4GB 可跑。

## 前端視覺化（特色）

- **左右同步對比**：左 dense、右 MoE，同一句子，由下往上逐層點亮，逐 token 播放。
- **Dense 側**：每層一條 32 格 neuron bar，全部點亮，亮度 = 激活強度 → 「全部參數都參與」。
- **MoE 側**：每層 expert 網格（64 → 8×8；32 → 8×4），僅 top-k 格亮起，亮度 = router 權重，
  其餘保持暗色 → 稀疏性一目了然。
- **即時參數計數器**：播放時兩側各自累計「本次已動用參數量」，dense 遠大於 MoE active params。
- **互動**：播放／暫停／速度（0.5x/1x/2x）、點下方 token 條跳轉、播放完顯示 expert 使用熱度圖
  （各 expert 被選次數，看分工）。
- 深色主題、繁體中文介面；動畫用 SVG（層/格子數量級 ~10³，效能足夠）。

## 錯誤處理

- 模型載入失敗／OOM／下載中斷：`/api/status` 帶 `error` 狀態與中文訊息，前端顯示並允許重試。
- 推論時模型未載入：400 + 提示先載入。
- 句子過長、空白句子：400 + 中文提示。

## 測試

- **單元（pytest）**：extract.py 對 gpt2 與 granite MoE 實跑短句，驗證 JSON 結構、
  維度（層數 × token 數 × bins/top_k）、權重總和≈1、正規化範圍。
- **API 功能**：FastAPI TestClient 走 models → load → status → infer 全流程與錯誤路徑。
- **e2e（Playwright）**：開頁 → 選預設模型 → 輸入句子 → 播放 → 驗證兩側動畫元素與參數計數器出現。

## 交付

- 公開 GitHub repo `densevsmoe`（repo 名 = 資料夾名），標準 README（安裝、啟動、截圖、原理說明）。
- 啟動方式：`uv run uvicorn server.main:app`。
