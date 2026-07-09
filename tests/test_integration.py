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

    # 依序載入（等前一個 ready 再載下一個）：transformers 的 from_pretrained
    # 非執行緒安全，兩條背景執行緒並行載入會競態污染 torch 預設 dtype，
    # 導致 tied lm_head 以錯誤 dtype 初始化、forward 時 dtype 不合
    # （詳見 task-7-report.md）。
    for mid in (GPT2, GRANITE):
        assert client.post("/api/load", json={"model_id": mid}).status_code == 202
        deadline = time.time() + 600
        while time.time() < deadline:
            status = client.get("/api/status").json()
            state = status.get(mid, {}).get("state")
            if state == "ready":
                break
            assert state != "error", status
            time.sleep(1)
        else:
            pytest.fail(f"模型載入逾時：{mid}")

    r = client.post("/api/infer", json={
        "dense_model": GPT2, "moe_model": GRANITE,
        "sentence": "今天天氣真好", "max_new_tokens": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    d, m = body["dense"], body["moe"]
    assert d["n_layers"] == 12 and len(d["activations"]) == 12
    assert d["params_per_token"] > 0
    assert m["n_experts"] == 32 and m["top_k"] == 8
    assert len(m["routing"]) == m["n_layers"]
    assert m["active_params_per_token"] < m["total_params"]
    # 生成：兩側都有回答文字，token 序列 = 輸入 + 生成
    for side in (d, m):
        assert side["generated_text"].strip()
        assert 0 < side["n_input_tokens"] < len(side["tokens"])
    # routing 覆蓋含生成段的完整序列
    assert all(len(layer) == len(m["tokens"]) for layer in m["routing"])
