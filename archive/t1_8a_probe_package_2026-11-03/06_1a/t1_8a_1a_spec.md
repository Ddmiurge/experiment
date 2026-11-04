# T1.8a §1a · FasterVLM 选择逻辑说明书（服务器实读，含行号自证）

> 实读日期：2026-11-03 · 仓库克隆于 `/tmp/fastervlm`、`/tmp/vispruner`（`--depth 1`）

## 0 · 版本登记（SHA + 日期）

| 仓库 | commit | 提交日期 | 用途 |
|---|---|---|---|
| `Theia-4869/FasterVLM` | **f5d5f12037e105a51865ecb4b28a73cd86eb4921** | 2025-06-29 | **本轮 V1/V2 采用**（协议默认：旧版 FasterVLM） |
| `Theia-4869/VisPruner` | **aefa01adc7c7ce6334e880c88225e90cede760d1** | 2025-07-01 | 仅登记，不跑（协议 §1a） |

## 1 · 选择逻辑（逐行）

代码位置：`llava/model/llava_arch.py` **L137–L157**，函数 `LlamaMetaForCausalLM.encode_images`。

```python
138  def encode_images(self, images):
142      image_features, image_attentions = self.get_model().get_vision_tower()(images)  # (B,576,C), (B,H,576)
147      image_attentions = image_attentions.mean(dim=1)      # ① 头维平均 → (B,576)
148/149  visual_token_num = self.get_visual_token_num()       # ② 固定保留数 T
149      token_indices = torch.topk(image_attentions, k=visual_token_num, dim=1)[1]
151-153  index_masks = zeros(B,N,bool); index_masks.scatter_(1, token_indices, True)   # ③ 布尔掩码
155      image_features = self.get_model().mm_projector(image_features)                  # ④ 投影在掩码前
```

注意力张量的来源同文件 `llava/model/multimodal_encoder/clip_encoder.py`：
- **L44–L56 `feature_select`**：`hidden_states[select_layer]` 与 `attentions[select_layer]`，
  `select_feature=='patch'` 时取 `image_features[:,1:]`、`image_attentions[:,:,0,1:]`
  —— 即 **CLS 行（query 0）对 576 个 patch 列的注意力**。
- **L51–L67 `forward`**：`self.vision_tower(..., output_attentions=True, output_hidden_states=True)`。

**逐项回答协议三问：**

| 问题 | 实测答案 |
|---|---|
| **剪在哪** | LLM **之外**、视觉编码器 + `mm_projector` **之后**；`prepare_inputs_labels_for_multimodal` L239 `image_feature = image_feature[index_mask]` 才真正丢弃 |
| **保留格数** | **固定 T = `visual_token_num`**，不是阈值；README L92 明示 `576×(1−R)=T`，75% ⇒ **T=144** |
| **丢弃格如何处置** | **纯删除**（`image_feature[index_mask]`），**没有任何聚合 token** ⇒ 进 LLM 的视觉 token 数 = **144**（不是 144+1=145） |
| 打分层 | `attentions[select_layer]` 的 CLS 行，**跨头算术平均**（非 max/sum），再 `topk` 取最大 |
| 选层 | 由模型 config 决定；vanilla LLaVA-1.5-7B config 为 `mm_vision_select_layer=-2`、`mm_vision_select_feature=patch` |

## 2 · 与协议描述的出入（TRIM 先例：以仓库为准，不纠正）

协议 §1a 写的是"**[CLS] 行注意力 → 动态阈值 τ → 相似度去重补多样性 → 最终保留名单**"。
**旧版 FasterVLM（f5d5f12）里没有 τ、没有去重**，只有 `topk` 一步。

τ/去重那套出现在**新版 VisPruner**（aefa01a）：
`llava/model/llava_arch.py` L150–L151 把预算拆成
`important_token_num = int(T × important_ratio)` 与 `diverse_token_num = T − important_token_num`，
README Overview 明写"先按 visual attention 选显著 token，再从剩余里按相似度去重补多样 token"。

⇒ **V1 按旧版仓库原样跑（纯 topk）**；本条出入进报告披露节，不改代码、不"纠正"。

## 3 · 与我们栈的差异清单（fork 差异，供 V2 接线）

| 项 | 我们的栈（haotian-liu @f2032236） | FasterVLM（f5d5f12） |
|---|---|---|
| `CLIPVisionTower.forward` 返回 | `image_features`（单值） | `(image_features, image_attentions)` 二元组 |
| 剪枝位置 | TRIM 在 `token_reduction`（视觉塔内，进 projector 之前） | `encode_images` 内，**projector 之后、进 LLM 之前** |
| 进 LLM 的视觉 token 数 | 144 + 1 聚合 = **145** | **144**（纯删除） |
| 掩码语义 | `token_reduction` 内部 topk | `index_masks` 传入 `prepare_inputs...` 做布尔索引 |

⇒ V2（我们的框）必须改为：用它的名单选 144 格 + 其余 432 格均值聚合 = 145，
其余（conv/greedy/seed/512/判官）与 T1.6 §3 逐项相同。

## 4 · 一句话总结

**FasterVLM = "视觉编码器里 CLS 对 patch 的注意力，跨头平均后取 top-T（75% 档 T=144），
在投影后按布尔掩码纯删除，无阈值、无去重、无聚合"。**
