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

    stacked = torch.stack(layers)                  # (L, seq, n_bins)
    maxv = stacked.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    stacked = ((stacked / maxv) * 1000).round() / 1000       # 逐層正規化 0..1
    return {
        "tokens": tokens,
        "n_layers": int(stacked.shape[0]),
        "activations": stacked.tolist(),
    }
