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


def test_load_memory_insufficient_returns_409(monkeypatch):
    mgr = ModelManager(loader=lambda mid: (object(), object()),
                       mem_check=lambda: 0.1)
    monkeypatch.setattr(main, "manager", mgr)
    c = TestClient(main.app)
    r = c.post("/api/load", json={"model_id": GPT2})
    assert r.status_code == 409
    assert "記憶體不足" in r.json()["detail"]


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
