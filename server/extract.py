"""推論與擷取：dense 各層 MLP 激活強度、MoE 各層 router 路由。"""
from __future__ import annotations

import re

import torch

MAX_TOKENS = 64


class SentenceTooLong(ValueError):
    pass


def _encode(tokenizer, sentence: str, device):
    enc = tokenizer(sentence, return_tensors="pt")
    n = enc.input_ids.shape[1]
    if n > MAX_TOKENS:
        raise SentenceTooLong(f"句子過長：{n} tokens（上限 {MAX_TOKENS}）")
    tokens = [tokenizer.decode([t]) for t in enc.input_ids[0]]
    return {k: v.to(device) for k, v in enc.items()}, tokens


def _find_act_modules(model) -> list[torch.nn.Module]:
    """各層 MLP 的激活函數模組（GPT-2: mlp.act；Llama/Qwen 系: mlp.act_fn），依層序排列。"""
    mods = []
    for name, module in model.named_modules():
        if name.endswith(("mlp.act", "mlp.act_fn")):
            m = re.search(r"\.(\d+)\.", name)
            if m:
                mods.append((int(m.group(1)), module))
    mods.sort(key=lambda x: x[0])
    return [module for _, module in mods]


def extract_dense(model, tokenizer, sentence: str, n_bins: int = 32) -> dict:
    enc, tokens = _encode(tokenizer, sentence, model.device)
    acts: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        acts.append(out.detach().abs().float().cpu())  # (1, seq, intermediate)

    handles = [m.register_forward_hook(hook) for m in _find_act_modules(model)]
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    layers = []
    for a in acts:
        a = a[0]                                   # (seq, intermediate)
        seq, inter = a.shape
        pad = (-inter) % n_bins
        if pad:
            a = torch.nn.functional.pad(a, (0, pad))
        layers.append(a.reshape(seq, n_bins, -1).mean(-1))   # (seq, n_bins)

    if not layers:
        raise RuntimeError(
            f"找不到 MLP 激活模組（模型 {type(model).__name__} 的命名不符 mlp.act/mlp.act_fn）")

    stacked = torch.stack(layers)                  # (L, seq, n_bins)
    maxv = stacked.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    stacked = ((stacked / maxv) * 1000).round() / 1000       # 逐層正規化 0..1
    return {
        "tokens": tokens,
        "n_layers": int(stacked.shape[0]),
        "activations": stacked.tolist(),
    }


def _find_router_modules(model) -> list[torch.nn.Module]:
    """各層 MoE router 模組（class 名稱以 TopKRouter 結尾，如 GraniteMoeTopKRouter/
    OlmoeTopKRouter），依模型內部登記序排列（即層序）。"""
    return [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")]


def _route_from_logits(raw_logits_by_layer, seq: int, top_k: int) -> list:
    """對每層原始 router logits（shape (batch*seq, n_experts)）做 softmax 後取
    top-k，回傳 [[{expert, weight}×top_k]×seq]×n_layers。"""
    routing = []
    for logits in raw_logits_by_layer:
        probs = torch.softmax(logits.detach().float().cpu(), dim=-1)
        probs = probs.reshape(seq, -1)
        weights, indices = probs.topk(top_k, dim=-1)
        routing.append([
            [{"expert": int(e), "weight": round(float(w), 4)}
             for e, w in zip(idx_row, w_row)]
            for idx_row, w_row in zip(indices, weights)
        ])
    return routing


def extract_moe(model, tokenizer, sentence: str) -> dict:
    enc, tokens = _encode(tokenizer, sentence, model.device)

    cfg = model.config
    top_k = int(getattr(cfg, "num_experts_per_tok", 8))
    n_experts = int(getattr(cfg, "num_experts", 0)
                    or getattr(cfg, "num_local_experts"))
    seq = enc["input_ids"].shape[1]

    # transformers 5.13 重構後，GraniteMoe 的 output_router_logits 已失效：
    # GraniteMoePreTrainedModel._can_record_outputs 只登記 hidden_states/
    # attentions，router_logits 在 GraniteMoeMoE.forward 內以 `_` 直接丟棄，
    # 導致 model(**enc, output_router_logits=True) 回傳的 out.router_logits
    # 恆為 None。改用 forward hook 直接擷取各層 router（*TopKRouter）forward
    # 回傳 tuple 中形狀為 (..., n_experts) 的原始 logits 張量（Granite 是第 3
    # 個回傳值；Olmoe 等架構則是第 1 個，故以形狀而非位置索引辨識，較穩健），
    # 再沿用與 output_router_logits 路徑相同的 softmax+top-k 邏輯。
    # 對仍支援 output_router_logits 的架構則保留原機制作為後援。
    router_mods = _find_router_modules(model)

    if router_mods:
        captured: list[torch.Tensor] = []

        def hook(_mod, _inp, out):
            for t in out:
                if isinstance(t, torch.Tensor) and t.shape[-1] == n_experts:
                    captured.append(t)
                    break

        handles = [m.register_forward_hook(hook) for m in router_mods]
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in handles:
                h.remove()
        routing = _route_from_logits(captured, seq, top_k)
    else:
        with torch.no_grad():
            out = model(**enc, output_router_logits=True)
        routing = _route_from_logits(out.router_logits, seq, top_k)

    return {
        "tokens": tokens,
        "n_layers": len(routing),
        "n_experts": n_experts,
        "top_k": top_k,
        "routing": routing,
    }
