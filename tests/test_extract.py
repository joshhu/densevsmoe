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
