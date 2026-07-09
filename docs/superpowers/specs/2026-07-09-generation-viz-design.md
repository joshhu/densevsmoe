# 真實生成 + 生成過程視覺化 — 設計文件

日期：2026-07-09
狀態：已核准
背景：原工具只對輸入句子做一次前向傳播，模型沒有生成任何回答。使用者要求：輸入問題 →
模型實際生成回答 → 生成的每個 token 也要有逐層點亮視覺化。

## 需求定案

- **問答模式**：有 chat template 的模型套 template（使用者句子作為 user 訊息、
  `add_generation_prompt=True`）；無 template 的模型（如 GPT-2）自動退回純續寫。
- **貪婪解碼**（`do_sample=False`）：同一問題示範結果可重現。
- `max_new_tokens` 白名單 **{8, 24, 48}**，預設 24；非白名單值回 400。
- 輸入（含 template）上限維持 64 tokens；生成部分不計入上限。
- dense 與 MoE **各自生成各自的回答**，並排顯示。

## 後端（server/extract.py、server/main.py）

### extract.py

- 新增 `GEN_CHOICES = (8, 24, 48)`。
- 新增 `_build_input_ids(tokenizer, sentence, device)`：套 chat template 或退回純 encode；
  超長拋 `SentenceTooLong`。
- 新增 `generate_ids(model, tokenizer, sentence, max_new_tokens) -> (full_ids, n_input, generated_text)`：
  `model.generate(ids, attention_mask=全 1, max_new_tokens, do_sample=False,
  pad_token_id=pad 或 eos)`；`generated_text` 為新增段 `skip_special_tokens=True` 解碼。
- `extract_dense` / `extract_moe` 各加選用參數 `input_ids=None`：給定時跳過 `_encode`，
  直接對該序列跑前向擷取（tokens 逐一 decode）。既有呼叫方式與回傳結構不變。

### main.py

- `InferReq` 加 `max_new_tokens: int = 24`；不在 `GEN_CHOICES` 內回 400（中文訊息）。
- `infer` 於 `_INFER_LOCK` 內：每側先 `generate_ids` 再以 `input_ids` 呼叫 extract；
  回應每側新增 `n_input_tokens: int`、`generated_text: str`。

## 前端（static/index.html、app.js、style.css）

- 輸入列加「生成長度」下拉 `#gen-len`（8/24/48，預設 24）；`run()` 帶入請求。
- 每側面板加 `#dense-answer` / `#moe-answer`：「💬 回答：{generated_text}」。
- token 條：`buildTokenStrip(id, tokens, nInput)`——生成段 token 加 class `gen`
  （綠色描邊強調）；輸入段維持原樣式。`highlightToken` 改用 `classList.toggle`
  以保留 `gen` class。
- 執行鈕 loading 文案改「推論+生成中…」。
- 播放引擎、參數計數器、expert 身分色不變（序列變長而已）。

## 測試

- `test_extract`：新增 gpt2 續寫 fallback 測試（無 template → 純續寫；
  `full_ids == n_input + 8`、`generated_text` 非空、`extract_dense(input_ids=...)`
  的 tokens 長度 = 全序列）。既有測試不動、必須照常通過。
- `test_api`：新增 `max_new_tokens=999 → 400`；`test_concurrent_infer_serialized`
  同步 monkeypatch `generate_ids`。
- `test_integration`：改用 `max_new_tokens=8`，加斷言 `generated_text` 非空、
  `n_input_tokens < len(tokens)`、routing 覆蓋全序列。
- `test_e2e`：加斷言回答文字出現（`#dense-answer` 非空）與生成 token 樣式（`.gen`）存在。

## 錯誤處理

- 輸入過長 / 空句 / 未載入 / 非白名單長度 → 400 中文訊息（沿用既有路徑）。
