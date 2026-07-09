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


def test_moe_fallback_router_logits():
    """找不到 *TopKRouter 子模組時，退回 output_router_logits=True 路徑
    （目前僅 granite 有真模型測試涵蓋 hook 路徑，這條 fallback 路徑用 stub
    model 驗證 shape 假設與 top-k 選擇邏輯）。"""
    import types

    import torch
    import torch.nn as nn
    from transformers import BatchEncoding

    # 固定 logits，每列的最大值位置已知，用來驗證 top-k 選對
    logits_l0 = torch.tensor([
        [5.0, 1.0, 2.0, 3.0],   # argmax = 0
        [1.0, 5.0, 2.0, 3.0],   # argmax = 1
        [1.0, 2.0, 5.0, 3.0],   # argmax = 2
    ])
    logits_l1 = torch.tensor([
        [3.0, 1.0, 5.0, 2.0],   # argmax = 2
        [5.0, 3.0, 1.0, 2.0],   # argmax = 0
        [2.0, 5.0, 1.0, 3.0],   # argmax = 1
    ])
    expected_argmax = [[0, 1, 2], [2, 0, 1]]

    class Dummy(nn.Module):
        """無任何名稱含 TopKRouter 的子模組，模擬找不到 router 模組時的情況。"""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.device = "cpu"
            self.config = types.SimpleNamespace(
                num_experts_per_tok=2, num_local_experts=4)  # 不設 num_experts

        def forward(self, input_ids=None, attention_mask=None,
                    output_router_logits=False, **kw):
            return types.SimpleNamespace(router_logits=(logits_l0, logits_l1))

    class DummyTok:
        def __call__(self, s, return_tensors=None):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])},
                                 tensor_type="pt")

        def decode(self, ids):
            return "x"

    r = extract_moe(Dummy(), DummyTok(), "abc")

    assert r["n_layers"] == 2
    assert r["n_experts"] == 4
    assert r["top_k"] == 2
    assert len(r["routing"]) == 2
    for li, layer in enumerate(r["routing"]):
        assert len(layer) == 3
        for ti, cell in enumerate(layer):
            assert len(cell) == 2
            assert all(0 <= e["expert"] < 4 for e in cell)
            total = sum(e["weight"] for e in cell)
            assert 0 < total <= 1.001
            assert cell[0]["expert"] == expected_argmax[li][ti]


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
