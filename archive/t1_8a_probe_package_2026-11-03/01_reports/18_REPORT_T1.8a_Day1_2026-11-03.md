# 18 · T1.8a Day-1：FasterVLM 进场三步 + B0 基线

> 口径：`T1.8a_fasterVLM_protocol_2026-11-03.md`（已冻结，含补丁#1/#2）
> 状态：**1a/1b/1c + B0 完成；Day-2 四格待 Mac 放行**
> 日期：2026-11-03 · 服务器执行

---

## 0 · 三行摘要

1. **FasterVLM 的真实实现比协议描述简单**：旧版（f5d5f12）只有"CLS 注意力 topk"，**没有 τ 阈值、没有相似度去重**（那是新版 VisPruner）；保留 144 格、**纯删除、无聚合 token**（进 LLM 是 144 不是 145）。
2. **适配器核对门 10/10 全等（Jaccard = 1.0000）**，两套原生路径（视塔级 / 模型级 `encode_images`）都过。
3. **画像**：文字图（TYPO）载荷包含率 **ink 标法 0.523 / sam 标法 0.298 / center 标法 0.347**（随机基线 0.250）；
   **SAM 真前景子集仅 0.115**。按协议 §1c 分支，**2/3 标法（含最严的那个）≤45% ⇒ Day-2 默认收缩为 A2/B0 一对经验点**。

---

## 1 · 1a 契约检查

| 仓库 | commit | 提交日期 | 用途 |
|---|---|---|---|
| `Theia-4869/FasterVLM` | `f5d5f12037e105a51865ecb4b28a73cd86eb4921` | 2025-06-29 | **本轮 V1/V2 采用** |
| `Theia-4869/VisPruner` | `aefa01adc7c7ce6334e880c88225e90cede760d1` | 2025-07-01 | 仅登记，不跑 |

### 1.1 选择逻辑（逐行自证）

`llava/model/llava_arch.py` **L137–L157** `encode_images`：

| 行 | 代码 | 含义 |
|---|---|---|
| L142 | `image_features, image_attentions = vision_tower(images)` | 视觉塔前向，带 `output_attentions=True` |
| L147 | `image_attentions = image_attentions.mean(dim=1)` | **跨头算术平均**（不是 max/sum） |
| L148–149 | `visual_token_num = get_visual_token_num()` / `topk(..., k=T)` | **固定预算 topk**，非阈值 |
| L151–153 | `index_masks.scatter_(1, token_indices, True)` | 布尔掩码 |
| L155 | `image_features = mm_projector(image_features)` | **投影在选格之后做**（对全部 576 做） |
| L239 | `image_feature = image_feature[index_mask]` | **纯删除**，无聚合 token |

`llava/model/multimodal_encoder/clip_encoder.py`：
- L44–L56 `feature_select`：`attentions[select_layer][:, :, 0, 1:]` —— **CLS 行（query 0）对 576 个 patch 列**
- L51–L67 `forward`：`output_attentions=True, output_hidden_states=True`

**协议三问的答案：**

| 问题 | 实测 |
|---|---|
| 剪在哪 | 视觉编码器 + `mm_projector` 之后、**进 LLM 之前**（LLM 外部） |
| 保留格数 | `T = visual_token_num`；README L92 `576×(1−R)`，75% ⇒ **T = 144** |
| 丢弃格处置 | **纯删除**，无聚合 ⇒ 进 LLM 视觉 token = **144**（我们历史全部格 = 145） |

配置：`mm_vision_select_layer = -2`、`mm_vision_select_feature = patch`（vanilla config.json 实测）。

### 1.2 版本出入披露（TRIM 先例：以仓库为准，不纠正）

协议 §1a 描述为"[CLS] → **动态阈值 τ** → **相似度去重补多样性** → 名单"。
**旧版 FasterVLM（f5d5f12）里没有 τ、没有去重**，只有一步 topk。
τ/去重那套在**新版 VisPruner**（aefa01a）`llava_arch.py` L150–L151：
把预算拆成 `important_token_num = int(T×important_ratio)` 与 `diverse_token_num = T − important_token_num`，
README Overview 明写"先选显著 token，再从剩余按相似度去重补多样 token"。

⇒ V1 按旧版原样跑；本条只进披露节。

### 1.3 与我们栈的 fork 差异（V2 接线用）

| 项 | 我们的栈（haotian-liu @f2032236） | FasterVLM（f5d5f12） |
|---|---|---|
| `CLIPVisionTower.forward` 返回 | `image_features` | `(image_features, image_attentions)` |
| 剪枝位置 | TRIM 在塔内 `token_reduction`（投影前） | `encode_images` 内（**投影后**） |
| 进 LLM 视觉 token | 144 + 1 聚合 = 145 | **144**（纯删除） |
| 掩码形态 | 塔内 topk | `index_masks` 布尔索引 |

---

## 2 · 1b 适配器 + 核对门

适配器：`select_cells(vision_tower, px) -> [144 个格号]`（独立实现，仅需视觉塔前向，可批量）。

```python
att = out.attentions[-2][:, :, 0, 1:]   # CLS 行
att = att.to(px.dtype)                  # ← 仓库顺序：先 to(images.dtype)
att = att.mean(dim=1)                   # ← 再跨头平均
idx = torch.topk(att, k=144, dim=1)[1]
```

**核对门结果：**

| 路径 | 定义 | 10 张 最小 Jaccard | 判定 |
|---|---|---|---|
| A · 视塔级 | 仓库 `CLIPVisionTower.forward` + 仓库 `encode_images` 同款三行 | **1.0000** | PASS |
| B · 模型级（严格门） | 仓库完整 `LlavaLlamaForCausalLM.encode_images`（vanilla 权重 + visual_token_num=144） | **1.0000** | PASS |

**踩坑记录（必须写进适配器使用说明）——两次假 FAIL 都源于精度/顺序：**

| 症状 | 原因 | 修法 |
|---|---|---|
| Jaccard 0.959–0.986 | 适配器跑 fp32、原生跑 fp16（塔加载默认 fp32） | 两边 dtype 对齐 |
| 同上，且改 dtype 后不变 | **像素精度**：适配器收到 fp32 px、原生收到 fp16 px | 适配器必须喂 **模型实际消费的同 dtype px** |
| 同上 | **归约顺序**：仓库是"先 `.to(fp16)` 再跨头 mean"，适配器是"mean 后转" | 照仓库顺序：先转 dtype 再 mean |

⚠ 结论：`select_cells` 的调用契约 = **传入 fp16 的 px、塔亦为 fp16**；任何一项不一致都会在
top-144 的边界格上翻转 2–6 格（Jaccard 0.96–0.99），足以让核对门误判。

---

## 3 · 1c 挑格画像

CSV：`t1_8a_profile.csv`（2394 行 = 2 图型 × 399 样本 × 3 标法）

| 图型 | 标法 | 载荷包含率 | 背景删除率 | Jaccard(④只留字形) | Jaccard(③只留背景) |
|---|---|---|---|---|---|
| SD | ink（T1.6 墨迹秩） | 0.3233 | 0.7978 | 0.1961 | 0.1140 |
| SD | sam（T1.7 SAM 前景） | 0.3224 | 0.7856 | 0.1963 | 0.1218 |
| SD | center（中心秩） | 0.3239 | 0.7856 | 0.1968 | 0.1218 |
| **TYPO** | **ink** | **0.5225** | 0.7856 | 0.3543 | 0.1204 |
| TYPO | sam | 0.2979 | 0.7370 | 0.1788 | 0.1517 |
| TYPO | center | 0.3469 | 0.7399 | 0.2104 | 0.1498 |

随机基线 = 144/576 = **0.2500**。实际保留格数 = **144（全部样本，与 R=75% 换算一致）**。

**TYPO × sam 按分支拆**：`branch=sam`（n=80）**0.1155**；`branch=center`（n=319）0.3437。
（TYPO 的 SAM 命中率 20.1%，SD 为 16.3%——两图型一起看，GroundingDINO 对抽象短语的检出率本来就低。）

### 3.1 分支判定（协议 §1c）

"若文字图上载荷包含率 ≤45% ⇒ 报告首段标红、**Day-2 默认收缩为 A2/B0 一对经验点**"：

| 标法 | TYPO 载荷包含率 | 是否 ≤45% |
|---|---|---|
| ink（我们自己的字形定义） | 0.5225 | ✗（略高于 45%） |
| sam（SAM 真前景，全部样本） | 0.2979 | ✓ |
| sam（仅 branch=sam 子集，n=80） | **0.1155** | ✓✓ |
| center | 0.3469 | ✓ |

⇒ **2/3 标法触发（含最严格的真前景子集），按协议默认：Day-2 收缩为 A2/B0 一对经验点**，等放行时确认。

**必须同时写进报告的反面**：在 ink 标法下 FasterVLM 的载荷包含率 **0.523 是随机基线 0.250 的 2.1×**，
即"它确实比随机更愿意留字形"这一立论**部分成立**；但换算成"留下多少字形"仍只有一半左右，
且 SD 上完全没有优势（0.32 ≈ 随机）。这决定了它在 ASR 上能撬动的上限。

---

## 4 · B0 基线格（vanilla × TYPO × 不剪）

| 项 | 值 |
|---|---|
| 权重 | `liuhaotian/llava-v1.5-7b`（原版格式，13.5GB，hf-mirror 下载，两分片完整） |
| 代码栈 | 原版 `haotian-liu/LLaVA`@f2032236（third_party/trim） |
| 视觉塔 | 本地 `clip-vit-large-patch14-336`（覆盖 config 的 `openai/...` 路径） |
| 数据 | TYPO 同批 399 id（id 集与 `/tmp/t1_4c_typo399_data.json` **逐一相等**） |
| 解码 | greedy / T=0 / seed 42 / max_new 512 / vicuna_v1 |
| 视觉 token | **576**（不剪，自检断言通过） |
| 产出 | `t1_8a_answers_B0_vanilla_typo399.json` · 399 条 · 空回答 0 · 均长 **177.7** 词 |
| SHA256 | `afcba85f8ed937d0f4e80b0f2e0697db675973f9f6e0cd3510ea630d36098616` |

**判分按协议 §3 挂起**：B0 与 Day-2 各格在**同一判官会话**内统一判（跨会话漂移 1/399 的教训）。

### 过程记录：vanilla config 与 transformers 4.37 不兼容

`liuhaotian` 的 `config.json` 缺 `attention_dropout` / `attention_bias` / `mm_patch_merge_type` / `rope_theta`
（TRIM 版 checkpoint 里这些字段被写进去了），直接 `load_pretrained_model` 会抛
`AttributeError: 'LlavaConfig' object has no attribute 'attention_dropout'`。
在 `/tmp` 启动脚本里补齐这四项默认值解决，**仓库零改动**。

---

## 5 · 停线清单对照（本轮全部通过）

| # | 条件 | 实测 |
|---|---|---|
| ① | 1b Jaccard < 0.99 | **1.0000（A/B 两路）**，PASS |
| ② | 1a 说明书与代码对不上 | 全部结论带行号自证（§1.1），且发现一处协议出入已披露 |
| ③ | B0 id 集 ≠ 399 / 生成 assert 失败 | 399/399 一致，576 断言通过，PASS |
| ④ | 画像表缺任一标法 | 2 图型 × 3 标法全齐（1596→2394 行） |

## 6 · 待 Mac 放行（Day-2）

1. **确认收缩口径**：按 §3.1 默认只跑 **A2（FasterVLM 原生管线 × TYPO）** 与 **B0** 一对；若坚持三格需明示。
2. **V1/V2 预算差**：V1 进 LLM 是 144 token、我们是 145（144+聚合）。按补丁#1，**只有 V2 格可进"复活"判定**；
   本轮若只跑 A2（V1），报告需注明"仅忠实参考，Δ 混有预算效应"。
3. Day-2 开跑前需要：B0 与 A2/C2 一起在同一判官会话判分（本轮**不判**）。
