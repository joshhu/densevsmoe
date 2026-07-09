# Dense vs MoE 推論視覺化 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立網頁工具：選一個 dense + 一個 MoE 模型（HuggingFace 策展清單）、輸入句子，以左右同步動畫視覺化真實推論時各層點亮過程。

**Architecture:** FastAPI 後端（uv venv）載入模型、跑一次前向傳播並擷取每層 × 每 token 的 activation / routing 資料成單一 JSON；原生 JS + SVG 前端以該 JSON 重播動畫（暫停/倒帶/調速皆前端進行，不重新推論）。

**Tech Stack:** Python 3.11+、uv、FastAPI、PyTorch（cu130）、transformers、原生 JS + SVG、pytest、pytest-playwright。

**Spec:** `docs/superpowers/specs/2026-07-09-densevsmoe-viz-design.md`

## Global Constraints

- **禁用 pip**：一律 `uv add` / `uv run`。
- PyTorch 需從 `https://download.pytorch.org/whl/cu130` index 安裝（DGX Spark：arm64 + CUDA 13 + GB10）。
- UI 文案一律繁體中文（台灣用語）、深色主題。
- 記憶體檢查讀 `/proc/meminfo` 的 `MemAvailable`（UMA 環境，不用 nvidia-smi）。
- 句子上限 64 tokens；超過回 HTTP 400 與中文錯誤訊息。
- pytest markers：`model`（需下載實跑模型）、`e2e`（需伺服器+瀏覽器）；預設 `addopts = "-m 'not e2e'"`。
- 測試中的模型一律用小模型：dense = `openai-community/gpt2`、MoE = `ibm-granite/granite-3.1-1b-a400m-instruct`。
- 每個 task 完成即 commit；訊息用 conventional commits。

---

### Task 1: 專案腳手架（uv + 依賴 + CUDA 驗證）

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `server/__init__.py`（空檔）
- Create: `tests/__init__.py`（空檔）

**Interfaces:**
- Produces: 可 `uv run` 的專案環境；後續所有 task 依賴此環境。

- [ ] **Step 1: 建立 pyproject.toml**

```toml
[project]
name = "densevsmoe"
version = "0.1.0"
description = "Dense vs MoE 推論視覺化"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "torch>=2.9",
    "transformers>=4.51",
    "accelerate>=1.0",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "httpx>=0.27",
    "pytest-playwright>=0.5",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.pytest.ini_options]
markers = [
    "model: 需下載並實跑模型",
    "e2e: 需啟動伺服器與瀏覽器",
]
addopts = "-m 'not e2e'"
```

- [ ] **Step 2: 建立 .gitignore**

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.pyc
.playwright/
```

- [ ] **Step 3: 建立空套件檔**

建立空的 `server/__init__.py` 與 `tests/__init__.py`。

- [ ] **Step 4: 同步依賴並驗證 CUDA**

Run: `uv sync && uv run python -c "import torch, transformers, fastapi; print(torch.__version__, torch.cuda.is_available())"`
Expected: 印出 torch 版本與 `True`（若 cu130 index 解析失敗，改試 `cu129`；CUDA 不可用時印 `False` 也可繼續，推論會退回 CPU）。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore server/__init__.py tests/__init__.py
git commit -m "chore: 專案腳手架（uv + FastAPI + torch cu130）"
```

---

### Task 2: models.py — 策展清單與模型載入管理

**Files:**
- Create: `server/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `ModelInfo` dataclass：`id, name, kind("dense"|"moe"), size_gb, n_layers, params_total, params_active, n_experts=0, top_k=0`
  - `MODEL_CATALOG: list[ModelInfo]`
  - `mem_available_gb() -> float`
  - `ModelManager(loader=None, mem_check=mem_available_gb)`：
    - `.catalog_entry(model_id) -> ModelInfo | None`（staticmethod）
    - `.request_load(model_id) -> dict`（未知 ID 拋 `KeyError`；回 `{"state": ...}`，背景執行緒載入）
    - `.status: dict[str, dict]`（`{model_id: {"state": "idle"|"loading"|"ready"|"error", "detail": str}}`）
    - `.get(kind, model_id) -> (model, tokenizer) | None`
    - `.loaded: dict[str, tuple[str, model, tokenizer]]`（kind → entry）

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_models.py
"""ModelManager 單元測試（用假 loader，不下載模型）。"""
import time

import pytest

from server.models import MODEL_CATALOG, ModelManager


def fake_loader(model_id):
    return f"model:{model_id}", f"tok:{model_id}"


def make_mgr(mem_gb=999.0):
    return ModelManager(loader=fake_loader, mem_check=lambda: mem_gb)


def wait_ready(mgr, model_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mgr.status.get(model_id, {}).get("state") in ("ready", "error"):
            return mgr.status[model_id]
        time.sleep(0.02)
    raise TimeoutError(model_id)


def test_catalog_has_both_kinds():
    kinds = {m.kind for m in MODEL_CATALOG}
    assert kinds == {"dense", "moe"}
    moe = [m for m in MODEL_CATALOG if m.kind == "moe"]
    assert all(m.n_experts > 0 and m.top_k > 0 for m in moe)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        make_mgr().request_load("nope/nope")


def test_load_and_ready():
    mgr = make_mgr()
    mid = "openai-community/gpt2"
    st = mgr.request_load(mid)
    assert st["state"] in ("loading", "ready")
    assert wait_ready(mgr, mid)["state"] == "ready"
    model, tok = mgr.get("dense", mid)
    assert model == f"model:{mid}" and tok == f"tok:{mid}"


def test_mem_insufficient():
    mgr = make_mgr(mem_gb=0.1)
    st = mgr.request_load("openai-community/gpt2")
    assert st["state"] == "error"
    assert "記憶體不足" in st["detail"]


def test_switch_unloads_old():
    mgr = make_mgr()
    a, b = "openai-community/gpt2", "Qwen/Qwen2.5-0.5B-Instruct"
    mgr.request_load(a)
    wait_ready(mgr, a)
    mgr.request_load(b)
    wait_ready(mgr, b)
    assert mgr.loaded["dense"][0] == b
    assert mgr.get("dense", a) is None
    assert mgr.status[a]["state"] == "idle"
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL（`ModuleNotFoundError: server.models`）

- [ ] **Step 3: 實作 server/models.py**

```python
"""模型載入管理：策展清單、記憶體控管、背景載入與狀態追蹤。"""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    kind: str            # "dense" | "moe"
    size_gb: float       # bf16 權重約略大小，供記憶體檢查
    n_layers: int
    params_total: int    # 總參數量
    params_active: int   # 每 token 動用參數量（dense = params_total）
    n_experts: int = 0
    top_k: int = 0


MODEL_CATALOG: list[ModelInfo] = [
    ModelInfo("openai-community/gpt2", "GPT-2 (124M)", "dense",
              0.3, 12, 124_000_000, 124_000_000),
    ModelInfo("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B（中文佳）", "dense",
              1.0, 24, 494_000_000, 494_000_000),
    ModelInfo("ibm-granite/granite-3.1-1b-a400m-instruct", "Granite MoE 1B（A400M）",
              "moe", 2.6, 24, 1_300_000_000, 400_000_000, n_experts=32, top_k=8),
    ModelInfo("allenai/OLMoE-1B-7B-0924-Instruct", "OLMoE 1B-7B（64 experts 選 8）",
              "moe", 14.0, 16, 6_900_000_000, 1_300_000_000, n_experts=64, top_k=8),
]


def mem_available_gb() -> float:
    """UMA 環境下讀系統可用記憶體（GB）。"""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


class ModelManager:
    """一次最多常駐一個 dense + 一個 MoE 模型；載入於背景執行緒。"""

    def __init__(self, loader=None, mem_check=mem_available_gb):
        self._loader = loader or self._default_loader
        self._mem_check = mem_check
        self._lock = threading.Lock()
        self.loaded: dict[str, tuple] = {}   # kind -> (model_id, model, tokenizer)
        self.status: dict[str, dict] = {}    # model_id -> {"state", "detail"}

    @staticmethod
    def catalog_entry(model_id: str) -> ModelInfo | None:
        return next((m for m in MODEL_CATALOG if m.id == model_id), None)

    @staticmethod
    def _default_loader(model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model, tok

    def request_load(self, model_id: str) -> dict:
        info = self.catalog_entry(model_id)
        if info is None:
            raise KeyError(model_id)
        with self._lock:
            state = self.status.get(model_id, {}).get("state")
            if state == "loading":
                return {"state": "loading"}
            if self.loaded.get(info.kind, (None,))[0] == model_id:
                return {"state": "ready"}
            free = self._mem_check()
            if free < info.size_gb * 1.2:
                msg = (f"可用記憶體不足：需要約 {info.size_gb * 1.2:.1f} GB，"
                       f"目前僅 {free:.1f} GB。請先釋放記憶體再試。")
                self.status[model_id] = {"state": "error", "detail": msg}
                return self.status[model_id]
            self.status[model_id] = {"state": "loading", "detail": "載入中"}
        threading.Thread(target=self._load_worker, args=(info,), daemon=True).start()
        return {"state": "loading"}

    def _load_worker(self, info: ModelInfo):
        try:
            self._unload(info.kind)
            model, tok = self._loader(info.id)
            with self._lock:
                self.loaded[info.kind] = (info.id, model, tok)
                self.status[info.id] = {"state": "ready", "detail": ""}
        except Exception as e:  # noqa: BLE001 — 載入失敗需回報前端
            with self._lock:
                self.status[info.id] = {"state": "error", "detail": f"載入失敗：{e}"}

    def _unload(self, kind: str):
        with self._lock:
            old = self.loaded.pop(kind, None)
            if old:
                self.status[old[0]] = {"state": "idle", "detail": ""}
        if old:
            del old
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get(self, kind: str, model_id: str):
        entry = self.loaded.get(kind)
        if entry and entry[0] == model_id:
            return entry[1], entry[2]
        return None
```

- [ ] **Step 4: 執行測試確認通過**

Run: `uv run pytest tests/test_models.py -v`
Expected: 5 項全 PASS

- [ ] **Step 5: Commit**

```bash
git add server/models.py tests/test_models.py
git commit -m "feat: 模型策展清單與載入管理（記憶體控管、背景載入）"
```

---

### Task 3: extract.py — Dense 激活擷取

**Files:**
- Create: `server/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `ModelManager._default_loader`（Task 2）取得真實 model/tokenizer。
- Produces:
  - `MAX_TOKENS = 64`
  - `class SentenceTooLong(ValueError)`
  - `extract_dense(model, tokenizer, sentence, n_bins=32) -> dict`：
    `{"tokens": [str], "n_layers": int, "activations": [[[float×n_bins]×n_tokens]×n_layers]}`，值域 0..1。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_extract.py
"""extract 擷取邏輯測試 — 實跑小模型（gpt2 / granite MoE）。"""
import pytest

from server.extract import SentenceTooLong, extract_dense
from server.models import ModelManager


@pytest.fixture(scope="session")
def gpt2():
    return ModelManager._default_loader("openai-community/gpt2")


@pytest.mark.model
def test_dense_structure(gpt2):
    model, tok = gpt2
    r = extract_dense(model, tok, "The weather is nice today")
    n_tok = len(r["tokens"])
    assert n_tok >= 4
    assert r["n_layers"] == 12
    assert len(r["activations"]) == 12
    assert all(len(layer) == n_tok for layer in r["activations"])
    assert all(len(cell) == 32
               for layer in r["activations"] for cell in layer)
    flat = [v for layer in r["activations"] for cell in layer for v in cell]
    assert min(flat) >= 0 and max(flat) <= 1
    assert max(flat) > 0.5  # 每層有正規化，應存在接近 1 的值


@pytest.mark.model
def test_dense_sentence_too_long(gpt2):
    model, tok = gpt2
    with pytest.raises(SentenceTooLong):
        extract_dense(model, tok, "word " * 100)
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `uv run pytest tests/test_extract.py -v -m model`
Expected: FAIL（`ModuleNotFoundError: server.extract`）；首次執行會下載 gpt2（約 0.5GB）。

- [ ] **Step 3: 實作 extract_dense**

```python
# server/extract.py
"""推論與擷取：dense 各層 MLP 激活強度、MoE 各層 router 路由。"""
from __future__ import annotations

import re

import torch

MAX_TOKENS = 64


class SentenceTooLong(ValueError):
    pass


def _encode(tokenizer, sentence: str, device):
    enc = tokenizer(sentence, return_tensors="pt")
    n = enc.input_ids.shape[1]
    if n > MAX_TOKENS:
        raise SentenceTooLong(f"句子過長：{n} tokens（上限 {MAX_TOKENS}）")
    tokens = [tokenizer.decode([t]) for t in enc.input_ids[0]]
    return {k: v.to(device) for k, v in enc.items()}, tokens


def _find_act_modules(model) -> list[torch.nn.Module]:
    """各層 MLP 的激活函數模組（GPT-2: mlp.act；Llama/Qwen 系: mlp.act_fn），依層序排列。"""
    mods = []
    for name, module in model.named_modules():
        if name.endswith(("mlp.act", "mlp.act_fn")):
            m = re.search(r"\.(\d+)\.", name)
            if m:
                mods.append((int(m.group(1)), module))
    mods.sort(key=lambda x: x[0])
    return [module for _, module in mods]


def extract_dense(model, tokenizer, sentence: str, n_bins: int = 32) -> dict:
    enc, tokens = _encode(tokenizer, sentence, model.device)
    acts: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        acts.append(out.detach().abs().float().cpu())  # (1, seq, intermediate)

    handles = [m.register_forward_hook(hook) for m in _find_act_modules(model)]
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    layers = []
    for a in acts:
        a = a[0]                                   # (seq, intermediate)
        seq, inter = a.shape
        pad = (-inter) % n_bins
        if pad:
            a = torch.nn.functional.pad(a, (0, pad))
        layers.append(a.reshape(seq, n_bins, -1).mean(-1))   # (seq, n_bins)

    stacked = torch.stack(layers)                  # (L, seq, n_bins)
    maxv = stacked.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    stacked = ((stacked / maxv) * 1000).round() / 1000       # 逐層正規化 0..1
    return {
        "tokens": tokens,
        "n_layers": int(stacked.shape[0]),
        "activations": stacked.tolist(),
    }
```

- [ ] **Step 4: 執行測試確認通過**

Run: `uv run pytest tests/test_extract.py -v -m model`
Expected: 2 項 PASS

- [ ] **Step 5: Commit**

```bash
git add server/extract.py tests/test_extract.py
git commit -m "feat: dense 各層 MLP 激活擷取（forward hook + neuron bins）"
```

---

### Task 4: extract.py — MoE routing 擷取

**Files:**
- Modify: `server/extract.py`（新增 `extract_moe`）
- Modify: `tests/test_extract.py`（新增 MoE 測試）

**Interfaces:**
- Produces: `extract_moe(model, tokenizer, sentence) -> dict`：
  `{"tokens": [str], "n_layers": int, "n_experts": int, "top_k": int, "routing": [[[{"expert": int, "weight": float}×top_k]×n_tokens]×n_layers]}`

- [ ] **Step 1: 寫失敗測試（附加到 tests/test_extract.py）**

```python
from server.extract import extract_moe  # 檔頭 import 併入既有那行

GRANITE = "ibm-granite/granite-3.1-1b-a400m-instruct"


@pytest.fixture(scope="session")
def granite():
    return ModelManager._default_loader(GRANITE)


@pytest.mark.model
def test_moe_structure(granite):
    model, tok = granite
    r = extract_moe(model, tok, "今天天氣真好")
    n_tok = len(r["tokens"])
    assert n_tok >= 2
    assert r["n_experts"] == 32 and r["top_k"] == 8
    assert len(r["routing"]) == r["n_layers"] > 0
    for layer in r["routing"]:
        assert len(layer) == n_tok
        for cell in layer:
            assert len(cell) == r["top_k"]
            assert all(0 <= e["expert"] < r["n_experts"] for e in cell)
            total = sum(e["weight"] for e in cell)
            assert 0 < total <= 1.001  # softmax 後取 top-k，總和不超過 1
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `uv run pytest tests/test_extract.py::test_moe_structure -v -m model`
Expected: FAIL（`ImportError: extract_moe`）；首次執行會下載 granite（約 2.6GB）。

- [ ] **Step 3: 實作 extract_moe（附加到 server/extract.py）**

```python
def extract_moe(model, tokenizer, sentence: str) -> dict:
    enc, tokens = _encode(tokenizer, sentence, model.device)
    with torch.no_grad():
        out = model(**enc, output_router_logits=True)

    cfg = model.config
    top_k = int(getattr(cfg, "num_experts_per_tok", 8))
    n_experts = int(getattr(cfg, "num_experts", 0)
                    or getattr(cfg, "num_local_experts"))
    seq = enc["input_ids"].shape[1]

    routing = []
    for logits in out.router_logits:               # 每層 (batch*seq, n_experts)
        probs = torch.softmax(logits.detach().float().cpu(), dim=-1)
        probs = probs.reshape(seq, -1)
        weights, indices = probs.topk(top_k, dim=-1)
        routing.append([
            [{"expert": int(e), "weight": round(float(w), 4)}
             for e, w in zip(idx_row, w_row)]
            for idx_row, w_row in zip(indices, weights)
        ])
    return {
        "tokens": tokens,
        "n_layers": len(routing),
        "n_experts": n_experts,
        "top_k": top_k,
        "routing": routing,
    }
```

- [ ] **Step 4: 執行測試確認通過**

Run: `uv run pytest tests/test_extract.py -v -m model`
Expected: 3 項全 PASS

- [ ] **Step 5: Commit**

```bash
git add server/extract.py tests/test_extract.py
git commit -m "feat: MoE router 路由擷取（output_router_logits + top-k）"
```

---

### Task 5: main.py — FastAPI API

**Files:**
- Create: `server/main.py`
- Create: `static/index.html`（暫時佔位，Task 6 完整實作；本 task 只需讓 StaticFiles mount 不噴錯）
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `ModelManager`、`MODEL_CATALOG`（Task 2）；`extract_dense`、`extract_moe`、`SentenceTooLong`（Task 3/4）。
- Produces:
  - `GET /api/models` → 目錄陣列（ModelInfo 欄位 + `state`）
  - `POST /api/load` `{model_id}` → 202 `{state}`；未知 ID → 400
  - `GET /api/status` → `{model_id: {state, detail}}`
  - `POST /api/infer` `{dense_model, moe_model, sentence}` → `{"dense": {...extract_dense 結果, model_id, params_per_token}, "moe": {...extract_moe 結果, model_id, active_params_per_token, total_params}}`
  - module-level `manager`（測試以 monkeypatch 替換）

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_api.py
"""API 功能測試 — 以假 loader 注入，不下載模型。"""
import time

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server.models import ModelManager

GPT2 = "openai-community/gpt2"
GRANITE = "ibm-granite/granite-3.1-1b-a400m-instruct"


@pytest.fixture()
def client(monkeypatch):
    mgr = ModelManager(loader=lambda mid: (object(), object()),
                       mem_check=lambda: 999.0)
    monkeypatch.setattr(main, "manager", mgr)
    return TestClient(main.app)


def load_and_wait(client, model_id):
    assert client.post("/api/load", json={"model_id": model_id}).status_code == 202
    for _ in range(100):
        st = client.get("/api/status").json().get(model_id, {})
        if st.get("state") == "ready":
            return
        time.sleep(0.02)
    raise TimeoutError(model_id)


def test_models_list(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()
    assert {m["kind"] for m in models} == {"dense", "moe"}
    assert all("state" in m for m in models)


def test_load_unknown_model(client):
    r = client.post("/api/load", json={"model_id": "nope/nope"})
    assert r.status_code == 400


def test_load_flow(client):
    load_and_wait(client, GPT2)


def test_infer_empty_sentence(client):
    r = client.post("/api/infer", json={
        "dense_model": GPT2, "moe_model": GRANITE, "sentence": "  "})
    assert r.status_code == 400


def test_infer_requires_loaded_models(client):
    r = client.post("/api/infer", json={
        "dense_model": GPT2, "moe_model": GRANITE, "sentence": "hi"})
    assert r.status_code == 400
    assert "尚未載入" in r.json()["detail"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL（`ModuleNotFoundError: server.main`）

- [ ] **Step 3: 實作 server/main.py 與佔位 index.html**

```python
# server/main.py
"""FastAPI 進入點：serve 前端靜態檔與 API。"""
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import extract
from .models import MODEL_CATALOG, ModelManager

app = FastAPI(title="Dense vs MoE 視覺化")
manager = ModelManager()

STATIC_DIR = Path(__file__).parent.parent / "static"


class LoadReq(BaseModel):
    model_id: str


class InferReq(BaseModel):
    dense_model: str
    moe_model: str
    sentence: str


@app.get("/api/models")
def list_models():
    return [{**asdict(m),
             "state": manager.status.get(m.id, {}).get("state", "idle")}
            for m in MODEL_CATALOG]


@app.post("/api/load", status_code=202)
def load_model(req: LoadReq):
    try:
        return manager.request_load(req.model_id)
    except KeyError:
        raise HTTPException(400, "未知的模型 ID（僅支援策展清單內的模型）")


@app.get("/api/status")
def status():
    return manager.status


@app.post("/api/infer")
def infer(req: InferReq):
    if not req.sentence.strip():
        raise HTTPException(400, "請輸入句子")
    dense = manager.get("dense", req.dense_model)
    moe = manager.get("moe", req.moe_model)
    if dense is None or moe is None:
        raise HTTPException(400, "模型尚未載入，請先載入兩側模型")
    d_info = manager.catalog_entry(req.dense_model)
    m_info = manager.catalog_entry(req.moe_model)
    try:
        d = extract.extract_dense(dense[0], dense[1], req.sentence)
        m = extract.extract_moe(moe[0], moe[1], req.sentence)
    except extract.SentenceTooLong as e:
        raise HTTPException(400, str(e))
    d.update(model_id=req.dense_model, params_per_token=d_info.params_active)
    m.update(model_id=req.moe_model,
             active_params_per_token=m_info.params_active,
             total_params=m_info.params_total)
    return {"dense": d, "moe": m}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

佔位 `static/index.html`（Task 6 會整份改寫）：

```html
<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>Dense vs MoE</title></head>
<body>建置中</body></html>
```

- [ ] **Step 4: 執行測試確認通過**

Run: `uv run pytest tests/test_api.py -v`
Expected: 6 項全 PASS

- [ ] **Step 5: Commit**

```bash
git add server/main.py static/index.html tests/test_api.py
git commit -m "feat: FastAPI API（models/load/status/infer + 靜態檔）"
```

---

### Task 6: 前端 — 版面、控制流程與點亮動畫

**Files:**
- Modify: `static/index.html`（整份改寫）
- Create: `static/style.css`
- Create: `static/app.js`
- Create: `static/viz.js`

**Interfaces:**
- Consumes: Task 5 的四支 API（回傳格式見 Task 5 Interfaces）。
- Produces: e2e 測試（Task 7）依賴的 DOM id：`#dense-select #moe-select #btn-load-dense #btn-load-moe #dense-status #moe-status #sentence #btn-run #dense-panel #moe-panel #btn-play #speed #scrub #dense-counter #moe-counter #moe-ratio #dense-tokens #moe-tokens #heatmap`。狀態文案：未載入／載入中…／`✓ 已載入`。

- [ ] **Step 1: 撰寫 static/index.html**

```html
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dense vs MoE 推論視覺化</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>Dense vs MoE 推論視覺化</h1>
  <p class="subtitle">看見句子流過模型時，每一層真正被「點亮」的部分 — 真實推論資料</p>
</header>

<section id="controls">
  <div class="model-pick">
    <label>Dense 模型 <select id="dense-select"></select></label>
    <button id="btn-load-dense">載入</button>
    <span id="dense-status" class="status">未載入</span>
  </div>
  <div class="model-pick">
    <label>MoE 模型 <select id="moe-select"></select></label>
    <button id="btn-load-moe">載入</button>
    <span id="moe-status" class="status">未載入</span>
  </div>
  <div class="sentence-row">
    <input id="sentence" placeholder="輸入一個句子"
           value="今天天氣真好，我們去公園散步吧。">
    <button id="btn-run" disabled>執行推論</button>
  </div>
</section>

<main>
  <div class="panel dense">
    <h2 id="dense-title">Dense</h2>
    <p class="panel-note">每個 token 動用<strong>全部</strong>神經元（32 個 neuron 區段，亮度 = 激活強度）</p>
    <svg id="dense-panel"></svg>
    <div class="tokens" id="dense-tokens"></div>
    <div class="counter">已動用參數：<span id="dense-counter">0</span></div>
  </div>
  <div class="panel moe">
    <h2 id="moe-title">MoE</h2>
    <p class="panel-note">每層只點亮被 router 選中的少數 experts（亮度 = 路由權重）</p>
    <svg id="moe-panel"></svg>
    <div class="tokens" id="moe-tokens"></div>
    <div class="counter">已動用參數：<span id="moe-counter">0</span>
      <span id="moe-ratio" class="ratio"></span></div>
    <div class="heatmap-wrap">
      <h3>Expert 使用熱度（累計被選次數）</h3>
      <svg id="heatmap"></svg>
    </div>
  </div>
</main>

<section id="playback">
  <button id="btn-play" disabled>▶ 播放</button>
  <label>速度
    <select id="speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
    </select>
  </label>
  <input type="range" id="scrub" min="0" max="1000" value="0">
</section>

<footer>PyTorch + transformers 真實推論 · DGX Spark</footer>
<script type="module" src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 撰寫 static/style.css**

```css
:root {
  --bg: #0f1117; --panel: #171b26; --border: #2a3040; --dim: #232838;
  --fg: #e6e9f0; --muted: #8b93a7;
  --dense: #ff9a28; --moe: #3cdcff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: "Noto Sans TC", system-ui, sans-serif;
}
header { text-align: center; padding: 1.2rem 1rem 0.4rem; }
h1 { margin: 0; font-size: 1.5rem; }
.subtitle { color: var(--muted); margin: 0.3rem 0 0; font-size: 0.9rem; }

#controls {
  display: flex; flex-wrap: wrap; gap: 0.8rem; justify-content: center;
  align-items: center; padding: 0.8rem 1rem;
}
.model-pick { display: flex; align-items: center; gap: 0.5rem; }
select, input, button {
  background: var(--panel); color: var(--fg);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 0.4rem 0.7rem; font-size: 0.9rem;
}
button { cursor: pointer; }
button:hover:not(:disabled) { border-color: var(--fg); }
button:disabled { opacity: 0.45; cursor: not-allowed; }
.sentence-row { display: flex; gap: 0.5rem; flex: 1 1 100%; justify-content: center; }
#sentence { width: min(480px, 70vw); }
.status { font-size: 0.85rem; color: var(--muted); min-width: 5rem; }
.status.ready { color: #6ee787; }
.status.error { color: #ff6b6b; }

main {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
  padding: 0 1rem; max-width: 1400px; margin: 0 auto;
}
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
.panel {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.8rem;
}
.panel.dense h2 { color: var(--dense); }
.panel.moe h2 { color: var(--moe); }
.panel h2 { margin: 0 0 0.2rem; font-size: 1.05rem; }
.panel-note { color: var(--muted); font-size: 0.8rem; margin: 0 0 0.6rem; }
.panel svg { width: 100%; height: auto; display: block; }
.layer-label { fill: var(--muted); font-size: 9px; }

.tokens { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 0.6rem; }
.tokens span {
  padding: 2px 6px; border-radius: 4px; background: var(--dim);
  font-size: 0.8rem; cursor: pointer; color: var(--muted);
}
.tokens span.done { color: var(--fg); }
.tokens span.current { background: var(--fg); color: var(--bg); font-weight: 700; }

.counter { margin-top: 0.5rem; font-size: 0.95rem; }
.counter span { font-variant-numeric: tabular-nums; font-weight: 700; }
.ratio { color: var(--muted); font-size: 0.8rem; font-weight: 400 !important; }
.heatmap-wrap h3 { font-size: 0.85rem; color: var(--muted); margin: 0.7rem 0 0.3rem; }

#playback {
  display: flex; gap: 1rem; align-items: center; justify-content: center;
  padding: 1rem;
}
#scrub { width: min(500px, 60vw); accent-color: var(--fg); padding: 0; }
footer { text-align: center; color: var(--muted); font-size: 0.75rem; padding: 0.5rem 0 1.2rem; }
```

- [ ] **Step 3: 撰寫 static/viz.js**

```js
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
```

- [ ] **Step 4: 撰寫 static/app.js**

```js
// app.js — 應用狀態、API 呼叫與播放引擎
import { buildDensePanel, buildHeatmap, buildMoePanel } from "./viz.js";

const $ = (id) => document.getElementById(id);
const state = {
  data: null, dense: null, moe: null, heat: null,
  playing: false, progress: 0, speed: 1, lastTs: 0,
};

const STATE_TEXT = { idle: "未載入", loading: "載入中…", ready: "✓ 已載入" };

async function init() {
  const models = await (await fetch("/api/models")).json();
  for (const m of models) {
    const sel = $(m.kind === "dense" ? "dense-select" : "moe-select");
    sel.add(new Option(`${m.name}（${m.size_gb}GB）`, m.id));
  }
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
  state.data = data;
  state.dense = buildDensePanel($("dense-panel"), data.dense);
  state.moe = buildMoePanel($("moe-panel"), data.moe);
  state.heat = buildHeatmap($("heatmap"), data.moe.n_experts);
  $("dense-title").textContent = `Dense — ${data.dense.model_id}`;
  $("moe-title").textContent = `MoE — ${data.moe.model_id}`;
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
  requestAnimationFrame(tick);
}

function play() {
  if (state.progress >= 1) state.progress = 0;
  state.playing = true;
  state.lastTs = 0;
  $("btn-play").textContent = "⏸ 暫停";
  requestAnimationFrame(tick);
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
```

- [ ] **Step 5: 手動 smoke 驗證**

Run: `uv run uvicorn server.main:app --port 8000 &`，然後 `curl -s http://127.0.0.1:8000/ | grep -o "Dense vs MoE 推論視覺化" | head -1 && curl -s http://127.0.0.1:8000/api/models | head -c 200`，完成後 kill 伺服器。
Expected: 標題字樣與模型清單 JSON。

- [ ] **Step 6: Commit**

```bash
git add static/
git commit -m "feat: 前端左右同步點亮動畫（SVG、參數計數器、expert 熱度圖）"
```

---

### Task 7: 整合測試（真實模型）與 e2e（Playwright）

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/test_e2e.py`

**Interfaces:**
- Consumes: Task 5 API、Task 6 DOM id 與狀態文案（`✓ 已載入`）。

- [ ] **Step 1: 寫整合測試（真模型走 API 全流程）**

```python
# tests/test_integration.py
"""真實模型整合測試：載入 gpt2 + granite，走 API infer 全流程。"""
import time

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server.models import ModelManager

GPT2 = "openai-community/gpt2"
GRANITE = "ibm-granite/granite-3.1-1b-a400m-instruct"


@pytest.mark.model
def test_infer_full_flow(monkeypatch):
    mgr = ModelManager()
    monkeypatch.setattr(main, "manager", mgr)
    client = TestClient(main.app)

    for mid in (GPT2, GRANITE):
        assert client.post("/api/load", json={"model_id": mid}).status_code == 202
    deadline = time.time() + 600
    while time.time() < deadline:
        status = client.get("/api/status").json()
        states = {status.get(m, {}).get("state") for m in (GPT2, GRANITE)}
        if states == {"ready"}:
            break
        assert "error" not in states, status
        time.sleep(1)
    else:
        pytest.fail("模型載入逾時")

    r = client.post("/api/infer", json={
        "dense_model": GPT2, "moe_model": GRANITE, "sentence": "今天天氣真好"})
    assert r.status_code == 200, r.text
    body = r.json()
    d, m = body["dense"], body["moe"]
    assert d["n_layers"] == 12 and len(d["activations"]) == 12
    assert d["params_per_token"] > 0
    assert m["n_experts"] == 32 and m["top_k"] == 8
    assert len(m["routing"]) == m["n_layers"]
    assert m["active_params_per_token"] < m["total_params"]
```

- [ ] **Step 2: 執行整合測試**

Run: `uv run pytest tests/test_integration.py -v -m model`
Expected: PASS（模型已在 Task 3/4 下載並快取）

- [ ] **Step 3: 安裝 Playwright 瀏覽器**

Run: `uv run playwright install chromium`
Expected: chromium 下載完成（arm64 支援）。

- [ ] **Step 4: 寫 e2e 測試**

```python
# tests/test_e2e.py
"""e2e：真伺服器 + 真瀏覽器 + 真模型，走完整使用者流程。"""
import socket
import subprocess
import time

import pytest
from playwright.sync_api import expect

PORT = 8321
URL = f"http://127.0.0.1:{PORT}"
GPT2 = "openai-community/gpt2"
GRANITE = "ibm-granite/granite-3.1-1b-a400m-instruct"


@pytest.fixture(scope="session")
def server():
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "server.main:app", "--port", str(PORT)])
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", PORT), 1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail("伺服器啟動逾時")
        yield URL
    finally:
        proc.terminate()
        proc.wait()


@pytest.mark.e2e
def test_full_user_flow(server, page):
    page.goto(server)
    page.select_option("#dense-select", GPT2)
    page.select_option("#moe-select", GRANITE)
    page.click("#btn-load-dense")
    page.click("#btn-load-moe")
    expect(page.locator("#dense-status")).to_have_text("✓ 已載入", timeout=600_000)
    expect(page.locator("#moe-status")).to_have_text("✓ 已載入", timeout=600_000)

    page.fill("#sentence", "今天天氣真好")
    expect(page.locator("#btn-run")).to_be_enabled(timeout=10_000)
    page.click("#btn-run")

    expect(page.locator("#dense-panel rect").first).to_be_visible(timeout=120_000)
    expect(page.locator("#moe-panel rect").first).to_be_visible()
    expect(page.locator("#heatmap rect").first).to_be_visible()
    page.wait_for_timeout(3000)  # 播放中
    assert page.text_content("#dense-counter") != "0"
    assert page.text_content("#moe-counter") != "0"
    assert (page.locator("#dense-tokens span").count() > 0
            and page.locator("#moe-tokens span").count() > 0)
```

- [ ] **Step 5: 執行 e2e**

Run: `uv run pytest tests/test_e2e.py -v -m e2e --override-ini addopts=""`
Expected: PASS（含真模型載入，可能需數分鐘）

- [ ] **Step 6: 全套測試確認**

Run: `uv run pytest -v`
Expected: 單元 + API + model 測試全 PASS（e2e 預設排除）

- [ ] **Step 7: Commit**

```bash
git add tests/test_integration.py tests/test_e2e.py
git commit -m "test: 真模型整合測試與 Playwright e2e"
```

---

### Task 8: README 與 GitHub 發佈

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 全部前置 task。

- [ ] **Step 1: 撰寫 README.md**

```markdown
# densevsmoe — Dense vs MoE 推論視覺化

從 HuggingFace 策展清單各選一個 dense 與一個 MoE 模型，輸入一個句子，
以左右同步動畫視覺化**真實推論**時每一層被「點亮」的過程：

- **Dense 側**：每層 32 個 neuron 區段全部點亮，亮度 = MLP 激活強度 —— 每個 token 動用全部參數。
- **MoE 側**：每層只點亮 router 選中的 top-k experts，亮度 = 路由權重 —— 稀疏性一目了然。
- 即時「已動用參數量」計數器、expert 使用熱度圖、播放／暫停／調速／點 token 跳轉。

## 需求

- Python 3.11+、[uv](https://docs.astral.sh/uv/)
- NVIDIA GPU（開發環境：DGX Spark，arm64 + CUDA 13）；無 GPU 會退回 CPU（較慢）

## 快速開始

​```bash
uv sync
uv run uvicorn server.main:app --port 8000
​```

開啟 http://localhost:8000 ，兩側各選模型按「載入」，輸入句子按「執行推論」。

## 支援模型

| 類型 | 模型 | 大小 (bf16) |
|------|------|------|
| Dense | openai-community/gpt2 | 0.3GB |
| Dense | Qwen/Qwen2.5-0.5B-Instruct | 1GB |
| MoE | ibm-granite/granite-3.1-1b-a400m-instruct | 2.6GB |
| MoE | allenai/OLMoE-1B-7B-0924-Instruct | 14GB |

## 原理

推論只跑一次：後端（FastAPI + PyTorch）對句子做一次前向傳播，
以 forward hook 擷取 dense 各層 MLP 激活強度、以 `output_router_logits=True`
擷取 MoE 各層 router top-k 路由，打包成 JSON；前端（原生 JS + SVG）重播動畫，
暫停／倒帶／調速都不需重新推論。

## 測試

​```bash
uv run pytest                                            # 單元 + API + 真模型測試
uv run pytest tests/test_e2e.py -m e2e --override-ini addopts=""  # e2e（Playwright）
​```
```

（注意：實際寫檔時把 ​``` 換成一般三反引號。）

- [ ] **Step 2: Commit 並發佈**

```bash
git add README.md
git commit -m "docs: README（安裝、使用、原理、測試）"
gh repo create densevsmoe --public --source . --push
```

Expected: repo 建立並推送成功，`gh repo view densevsmoe --web` 可開啟。

---

## Self-Review 紀錄

- **Spec 覆蓋**：策展清單（Task 2）、dense/MoE 擷取（Task 3/4）、四支 API 與錯誤處理（Task 5）、左右同步動畫＋計數器＋熱度圖＋互動（Task 6）、單元／API／整合／e2e 測試（Task 2–7）、README＋GitHub（Task 8）。spec 中「expert 網格 8×8」在實作採每層一列 n_experts 格（1×64），與 dense 側每層一列對齊、視覺對比更直接 —— 屬呈現細節微調。
- **Placeholder 掃描**：無 TBD/TODO；所有程式碼完整給出。
- **型別一致性**：`extract_dense`/`extract_moe` 回傳鍵與 Task 5 `infer` 組裝、Task 6 前端讀取（`activations`、`routing`、`params_per_token`、`active_params_per_token`、`total_params`、`n_experts`、`top_k`）逐一核對相符；DOM id 與 e2e 選擇器相符；狀態字串（idle/loading/ready/error 與「✓ 已載入」）前後端一致。
