"""測試共用設定：MoE forward 需要可寫的 triton cache（~/.triton 權限受限）。"""
import os
import tempfile

os.environ.setdefault(
    "TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton-cache"))

# 預先匯入 transformers，避免 ModelManager 背景執行緒同時載入 dense/moe 模型時，
# 對 transformers 的 lazy-module 造成競態（並行首次 import 觸發已知的執行緒不安全問題）。
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402,F401
