"""ModelManager 單元測試（用假 loader，不下載模型）。"""
import threading
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


def test_concurrent_same_kind_load_rejected():
    def slow_loader(model_id):
        time.sleep(0.3)
        return f"model:{model_id}", f"tok:{model_id}"

    mgr = ModelManager(loader=slow_loader, mem_check=lambda: 999.0)
    a, b = "openai-community/gpt2", "Qwen/Qwen2.5-0.5B-Instruct"

    st_a = mgr.request_load(a)
    assert st_a["state"] == "loading"

    st_b = mgr.request_load(b)
    assert st_b["state"] == "error"
    assert "載入中" in st_b["detail"]

    assert wait_ready(mgr, a)["state"] == "ready"
    assert mgr.loaded["dense"][0] == a
    assert mgr.status[a]["state"] == "ready"


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


def test_cross_kind_loads_serialized():
    """dense 與 moe 屬不同 kind，_loading_kinds 機制允許兩者同時被接受排隊，
    但實際載入必須經 _LOAD_LOCK 序列化，避免並行 from_pretrained/import 競態。"""
    intervals = []
    lock = threading.Lock()

    def slow_loader(model_id):
        start = time.time()
        time.sleep(0.2)
        end = time.time()
        with lock:
            intervals.append((start, end))
        return f"model:{model_id}", f"tok:{model_id}"

    mgr = ModelManager(loader=slow_loader, mem_check=lambda: 999.0)
    dense_id = "openai-community/gpt2"
    moe_id = "ibm-granite/granite-3.1-1b-a400m-instruct"

    t_dense = threading.Thread(target=mgr.request_load, args=(dense_id,))
    t_moe = threading.Thread(target=mgr.request_load, args=(moe_id,))
    t_dense.start()
    t_moe.start()
    t_dense.join()
    t_moe.join()

    assert wait_ready(mgr, dense_id)["state"] == "ready"
    assert wait_ready(mgr, moe_id)["state"] == "ready"

    assert len(intervals) == 2
    (s1, e1), (s2, e2) = sorted(intervals)
    assert e1 <= s2, f"載入區間重疊，未序列化：{intervals}"
