"""測試共用設定：MoE forward 需要可寫的 triton cache（~/.triton 權限受限）。"""
import os
import tempfile

os.environ.setdefault(
    "TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton-cache"))

# 註：先前這裡有「預先匯入 transformers」的因應措施，用來規避 ModelManager
# 背景執行緒並行首次 import transformers 的 lazy-module 競態。該競態已在
# server/models.py 用 _LOAD_LOCK 從產品碼源頭修掉（整個 _loader 呼叫，含
# import，皆已序列化），故此處不再需要，已移除。
