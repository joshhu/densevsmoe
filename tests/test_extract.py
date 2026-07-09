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
