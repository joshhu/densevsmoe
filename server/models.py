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
