"""API 功能測試 — 以假 loader 注入，不下載模型。"""
import threading
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


def test_load_memory_insufficient_returns_409(monkeypatch):
    mgr = ModelManager(loader=lambda mid: (object(), object()),
                       mem_check=lambda: 0.1)
    monkeypatch.setattr(main, "manager", mgr)
    c = TestClient(main.app)
    r = c.post("/api/load", json={"model_id": GPT2})
    assert r.status_code == 409
    assert "記憶體不足" in r.json()["detail"]


def test_concurrent_infer_serialized(client, monkeypatch):
    load_and_wait(client, GPT2)
    load_and_wait(client, GRANITE)

    intervals = []
    lock = threading.Lock()

    def fake_extract_dense(model, tok, sentence, n_bins=32):
        start = time.monotonic()
        time.sleep(0.2)
        end = time.monotonic()
        with lock:
            intervals.append((start, end))
        return {"tokens": ["a"], "n_layers": 1, "activations": [[[0.0]]]}

    def fake_extract_moe(model, tok, sentence):
        start = time.monotonic()
        time.sleep(0.2)
        end = time.monotonic()
        with lock:
            intervals.append((start, end))
        return {"tokens": ["a"], "n_layers": 1, "n_experts": 1, "top_k": 1,
                "routing": [[[{"expert": 0, "weight": 1.0}]]]}

    monkeypatch.setattr(main.extract, "extract_dense", fake_extract_dense)
    monkeypatch.setattr(main.extract, "extract_moe", fake_extract_moe)

    results = []

    def do_infer():
        r = client.post("/api/infer", json={
            "dense_model": GPT2, "moe_model": GRANITE, "sentence": "hi"})
        results.append(r)

    threads = [threading.Thread(target=do_infer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2
    assert all(r.status_code == 200 for r in results)
    assert len(intervals) == 4
    intervals.sort()
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        assert e1 <= s2, f"重疊區間：({s1}, {e1}) vs ({s2}, {e2})"


def test_load_concurrent_same_kind_returns_409(monkeypatch):
    def slow_loader(mid):
        time.sleep(0.3)
        return object(), object()

    mgr = ModelManager(loader=slow_loader, mem_check=lambda: 999.0)
    monkeypatch.setattr(main, "manager", mgr)
    c = TestClient(main.app)
    r1 = c.post("/api/load", json={"model_id": GPT2})
    assert r1.status_code == 202
    r2 = c.post("/api/load", json={"model_id": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert r2.status_code == 409
    assert "載入中" in r2.json()["detail"]
