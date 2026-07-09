"""extract 擷取邏輯測試 — 實跑小模型（gpt2 / granite MoE）。"""
import pytest

from server.extract import SentenceTooLong, extract_dense, extract_moe
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


def test_dense_no_act_modules_raises():
    import torch
    import torch.nn as nn

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.device = "cpu"

        def forward(self, input_ids=None, attention_mask=None, **kw):
            return None

    class DummyTok:
        def __call__(self, s, return_tensors=None):
            from transformers import BatchEncoding
            return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])},
                                 tensor_type="pt")

        def decode(self, ids):
            return "x"

    with pytest.raises(RuntimeError, match="MLP"):
        extract_dense(Dummy(), DummyTok(), "abc")


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
