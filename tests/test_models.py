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
