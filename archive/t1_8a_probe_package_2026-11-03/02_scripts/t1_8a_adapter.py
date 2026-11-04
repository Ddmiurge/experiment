#!/usr/bin/env python3
"""T1.8a §1b · FasterVLM 选格适配器（独立实现，只依赖视觉编码器前向）

冻结口径（与 /tmp/fastervlm @f5d5f12 逐行等价）：
    attentions[select_layer] → 取 CLS 行（query 0）对 576 个 patch 列 → 跨头算术平均
    → topk(k=T) → 保留格号（升序返回）

用法：
    from t1_8a_adapter import select_cells
    keep = select_cells(model.get_model().vision_tower, px)   # px: [1,3,336,336]
"""
import torch

NUM_PATCH = 576
SELECT_LAYER = -2       # vanilla LLaVA-1.5-7B config: mm_vision_select_layer=-2
KEEP = 144              # 75% 档：576*(1-0.75)


@torch.no_grad()
def select_cells(vision_tower, px, keep=KEEP, select_layer=SELECT_LAYER):
    """返回升序的保留格号列表（长度 keep）。

    vision_tower: 仓库的 CLIPVisionTower（或其 .vision_tower，本函数两种都接受）
    px:           [B,3,336,336]，模型实际消费的像素张量
    """
    vt = vision_tower
    if hasattr(vt, "vision_tower"):          # CLIPVisionTower 包装
        vt = vt.vision_tower
        dev, dtype = vt.device, vt.dtype
    else:
        dev, dtype = next(vt.parameters()).device, next(vt.parameters()).dtype

    out = vt(px.to(device=dev, dtype=dtype), output_attentions=True, output_hidden_states=True)
    att = out.attentions[select_layer]      # [B, heads, 577, 577]
    att = att[:, :, 0, 1:]                  # CLS 行对 patch 列 → [B, heads, 576]
    att = att.to(px.dtype)                  # 仓库先 .to(images.dtype)（顺序很关键：先转 fp16 再求均值，
                                            # fp16 累加与 fp32 累加在边界格上会翻，实测 Jaccard 0.959 vs 1.000）
    att = att.mean(dim=1)                   # 跨头平均 → [B, 576]
    idx = torch.topk(att, k=keep, dim=1)[1] # [B, keep]
    return sorted(int(i) for i in idx[0].tolist())
