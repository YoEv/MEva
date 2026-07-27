"""MusicGen delay-pattern teacher-forcing (HF transformers path).

历史背景:本仓所有历代 loss/entropy 抽取(losscal/*, scripts/loss/*,
scripts/features/*)都以"并行口径"前向——对齐 codes 直接喂
``model.decoder``,不施加 MusicGen 训练时的 delay 交错。该口径模型内部
自洽(论文 Small 结果不受影响),但绝对值跨模型不可比且非真似然
(A/B 实测抬高 1.1–2.4 nat,详见 MEva doc/20260719-2350_medium_sanity_findings.md)。

本模块提供 delay-correct 前向:按 MusicGen 原生 delay pattern
(codebook k 右移 k 步,前导位填 pad/special token)构造输入,单次前向后
把每个 codebook 的 logits 对齐回音频帧时间轴——与 audiocraft
``LMModel.compute_predictions``/beat_emergence 管线同口径。
"""
from __future__ import annotations

import torch


def delayed_aligned_logits(model, codes: torch.Tensor) -> torch.Tensor:
    """codes ``[1, C, K, T]`` → 帧对齐 logits ``[C, K, T-1, V]``。

    返回值第 t 列(t=0..T-2)是"给定 delay 交错上下文,预测音频帧 t+1"
    的 logits——与并行口径的帧索引完全对齐,可直接替换。
    覆盖性:S-1 = T+K-2 ≥ T+k-1 对所有 k<K 成立,故每帧每 codebook
    都有预测,无缺失位。
    """
    pad_id = int(model.decoder.config.pad_token_id)
    bsz, channels, K, T = codes.shape
    assert bsz == 1, "delayed_aligned_logits 仅支持 bsz=1(与既有抽取循环一致)"
    S = T + K - 1
    outs = []
    for ch in range(channels):
        c = codes[0, ch]  # [K, T]
        dl = torch.full((K, S), pad_id, dtype=torch.long, device=c.device)
        for k in range(K):
            dl[k, k : k + T] = c[k]
        lg = model.decoder(dl[None, :, :-1], output_hidden_states=False).logits
        if lg.dim() == 4:
            lg = lg[0]  # [K, S-1, V]
        aligned = torch.empty((K, T - 1, lg.shape[-1]), dtype=lg.dtype, device=lg.device)
        for k in range(K):
            # seq 位置 s 预测 dl[k, s+1] = 音频帧 s+1-k;帧 1..T-1 ⇔ s = 帧+k-1
            aligned[k] = lg[k, k : k + T - 1, :]
        outs.append(aligned)
    return torch.stack(outs, dim=0)  # [C, K, T-1, V]
