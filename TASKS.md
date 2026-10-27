# TASKS · 当前轮：剪枝方向（H1）与权重（H2）——两格定生死

> 背景与全部证据见 `experiments/t1_5_2026-10-27/README_诊断_2026-10-27.md`（必读，里面有原文行号与出处）。
> 常驻规则仍以 `Agent.md` 为准。
> 本轮**不改 `third_party/`、不改 `src/`**——脚本已在 `experiments/t1_5_2026-10-27/scripts/`，pull 即可跑。
> 上一轮「判官可重复性」**降级保留**，见 §三。

---

## 零、为什么改方向

上一轮把复现失败归到「判官噪声」。核过 TRIM 官方源码后发现一个**更靠前、可一次性解释六组读数**的原因：

```python
# third_party/trim/llava/model/multimodal_encoder/clip_encoder.py（锁定 SHA，与 main 无差异）
86  similarities = matmul(proj_image_features, text_features)   # 图像 token 与文本的相似度
89  similarities = -similarities                                # ← 取负
90  similarities = F.softmax(similarities, dim=-1)
111 topk(similarities, num_tokens_to_keep, largest=True)         # → 保留 s 最小的 144 个
```

TRIM 原文 §3.2 要求 **保留高相似度（前景/与指令相关）**；官方代码**保留低相似度**。
我们的复刻逐行照抄，所以两个栈都在跑**反向着色**的剪枝：
把与指令最相关（含 TYPO 图上的关键词条带）的 token 删掉。

这正好解释三件事：
1. 我们的 TRIM 落在原文 **Random 那一行**（56.75），不在引导式那一行（65.87）；
2. TYPO 上 TRIM **显著降低** ASR（−5.01pp，p=0.0045）——载荷被剪掉了；
3. SD 上 TRIM 与无剪枝几乎不动（−1.75pp，不显著）。

原文 Table 9 自己给了坐标：**Foreground removed = 50.00（−9.4pp）/ Background removed = 65.87（=TRIM 行）**。
我们的读数方向跟的是 50.00 那一行，永远到不了 65.87。

---

## 一、本轮任务（按顺序，第 1 步不过就别往下跑）

### 1 · 探针：保留集到底是不是反的（15 分钟，不需要 LLaVA，不要生成）

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/t1_5_2026-10-27/scripts/t1_5_trim_probe.py \
    --split SD --ids-from /tmp/t1_4_answers_orig_sd_noprun399.json
CUDA_VISIBLE_DEVICES=6 python experiments/t1_5_2026-10-27/scripts/t1_5_trim_probe.py \
    --split TYPO --ids-from /tmp/t1_4c_typo399_data.json
```

**必须打印的五读数**（脚本已经打，照原样贴回来）：

| 读数 | 预测（若 H1 成立） |
|---|---|
| 官方保留集与丢弃集的相似度均值差 `sim_diff` | **< 0**（保留的是低相似度） |
| `sim_diff < 0` 的样本占比 | ≈ 1.000（逐样本都反） |
| Jaccard(官方保留集, 高相似度保留集) | 明显 < 0.25（两套选择确实不同） |
| Jaccard(官方保留集, 随机子集) | ≈ 0.25 → 「反向着色在 SD 上近似随机」= 我们读数落在 Random 行的原因 |
| TYPO 底部条带被保留的 token 占比 | **< 0.25** → 关键词条带被系统性删除 |

> `--ids-from` 的文件若在 `/tmp` 已被清掉，改用 `--limit 399`（脚本自带按类目分层抽样），
> 但要在报告里写明「id 换了批」——跨格子可比性会断。

### 2 · 权重与配置比对（几分钟，CPU）

```bash
python experiments/t1_5_2026-10-27/scripts/t1_5_weights_diff.py \
    --a /home/huangqixu01/models/llava-1.5-7b-hf \
    --b /home/huangqixu01/models/llava-v1.5-7b-TRIM --sig-filter mm_projector
```

顺带把 `config.json` 的 `image_aspect_ratio` 贴回来：
**若是 `square`，则原版栈的 `process_images` 根本没走 `expand2square`，
T1.4 报告「预处理已修」这条对 TYPO 不成立，要重核。**

### 3 · 格子矩阵（核心；每格 399 条约 20 分钟，两卡并行约 1 小时）

**A. vanilla 权重（llava-hf 栈）**

```bash
python experiments/t1_5_2026-10-27/scripts/t1_5_generate.py --split SD --prune trim_highsim --ids-from /tmp/t1_4_answers_orig_sd_noprun399.json
python experiments/t1_5_2026-10-27/scripts/t1_5_generate.py --split SD --prune random       --ids-from /tmp/t1_4_answers_orig_sd_noprun399.json
# 已有格子（none=0.5789 / trim_official=0.5739）不必重跑；若 /tmp 文件已清，补跑 none 与 trim_official
```

**B. TRIM-FT 权重（原版栈，`third_party/trim` 原样 + 内存 monkeypatch 去掉取负）**

```bash
CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
    experiments/t1_5_2026-10-27/scripts/t1_5_orig_highsim.py --split SD
```

脚本开跑前有 5 条自检（打印两套打分的 `sim(保留集)` 与 Jaccard，均值不升就 assert 退出），
**不改 `third_party/` 磁盘文件**，patch 只活在进程里。

必须凑齐的矩阵（✓=已有）：

| | none | trim_official（低相似度） | trim_highsim（高相似度） | random |
|---|---|---|---|---|
| vanilla | 0.5789 ✓ | 0.5739 ✓ | **新** | **新** |
| TRIM-FT | 0.5840 ✓ | 0.5664 ✓ | **新** | — |

`random` 是本轮新增的**阴性对照**——原文 Random 行 56.75 vs TRIM 行 65.87 的差才是「剪什么」的证据；
只比 none vs trim 分不清「引导性」与「少喂 token」。

### 3.1 顺手把原文的配对口径算出来（不要额外跑格子）

诊断已确认：**SAP 原文 Table 2 的 TRIM 行是 TRIM-FT 微调 checkpoint + 剪枝，
Original 行是 vanilla 不剪枝**（MMBench 64.60→66.92 的指纹，见诊断 §2.1）。
所以除了同模型内的配对比较，还要给一行「原文的配对」：

```
Δ_total(原文 +6.50pp) = Δ_checkpoint + Δ_pruning
  Δ_checkpoint = TRIM-FT|none − vanilla|none            实测 +0.51pp
  Δ_pruning    = TRIM-FT|trim − TRIM-FT|none            实测 −1.76pp（官方符号）
```

**翻盘只可能来自 `Δ_pruning` 变正**，所以 `trim_highsim` 两格是本轮唯一的关键格子，先跑它们。

### 4 · 判分与配对分析

```bash
python experiments/t1_5_2026-10-27/scripts/t1_5_judge.py --split SD --ref none \
  --file none=/tmp/t1_5_answers_none_sd399.json \
  --file trim_official=/tmp/t1_5_answers_trim_official_sd399.json \
  --file trim_highsim=/tmp/t1_5_answers_trim_highsim_sd399.json \
  --file random=/tmp/t1_5_answers_random_sd399.json
```

脚本会同时给出：每组 ASR、每组 vs `none` 的 2×2 与 **McNemar 精确 p**、
**严格类目 / 通用类目分层**（严格类目 42% 样本饱和在 1.0，没有上升空间，效应只可能在通用类目）、
解析失败率与其两个方向的敏感性。逐判官原始输出留在 `/tmp/t1_5_judge_raw_<tag>.jsonl`。

### 5 · 若第 4 步出现正方向 → 补两件事再收

- 同一配置扩到**全量 1680**（±5pp → ±2.4pp），约每格 75 分钟；
- TYPO 分片同四格（TYPO 是载荷在图上的那一类，效应应更明显）。

---

## 二、验收与判定（照本表写结论，不要含糊）

| 探针 sim_diff | trim_highsim − none | random − none | 结论 |
|---|---|---|---|
| < 0 | **> 0 且 p<0.05** | ≈ 0 | **H1 成立**：官方代码方向反了。复现成功 = 在「按原文规范实现的 TRIM」上复现出 +Δ；交付物必须写明官方代码与原文矛盾这一事实 |
| < 0 | > 0 但不显著 | ≈ 0 | 方向对了但功效不够 → 扩全量 1680 再判 |
| < 0 | ≈ 0 | ≈ 0 | 剪什么不重要 → H1 解释力下降，回 H3（总体/判官）+H4（样本量），跑 SD+TYPO 合并总体全量 |
| < 0 | > 0 | **> 0 同量级** | 任何 75% 剪枝都抬高 ASR → 与原文 Table 11（Random 几乎不动）冲突 → 判总体/判官问题，别声称复现了引导性剪枝 |
| **> 0** | — | — | **我读码读错了**，H1 死。立刻停，把探针 jsonl 贴回来，转 H3/H4 |

> 表里的 `trim_highsim − none` 指**同一行（同一套权重）内**的比较，vanilla 与 TRIM-FT 各算一次，
> 两次都要报。若只有 TRIM-FT 那行为正，说明「剪枝增益」与「微调 checkpoint」绑定，
> 这本身要写进结论（原文的配对恰恰就是这一种）。

---

## 三、上一轮「判官可重复性」怎么处理

跑，但**降级**，且别按原验收表下结论：

- A vs B（同输入重判两遍）测的是 **temperature=0 的确定性**，不是那 31 条翻转的噪声底。
  那 31 条是**两段不同文本**得到不同判定，属于判官对措辞敏感，不是判官抖动。
- 所以「A vs B ≥99% → 效应不存在」这条推论**不成立**，场上还有一个未检验的 H1。
- 顺手做两件有价值的：`max_tokens=60` 截断检查（`parse_verdict` 失败会兜底判 safe，
  会悄悄压低 ASR）；判官原始输出留档。

---

## 四、不要做

| 不要 | 因为 |
|---|---|
| 改 `third_party/trim` | 硬规则。`trim_highsim` 用本目录脚本里的独立实现（同口径打分，只差符号），不动仓库 |
| 只报 `trim_highsim` 一格的读数 | 必须四格齐报，含 `trim_official` 与 `random`。这是预声明的变体集，不是调参 |
| 为了对上 65.87 去扫保留率 / 层 / 文本编码方式 | 调参凑数。保留率固定 0.25 |
| 把 `+Δ` 直接写成「复现了原文」 | 我们复现的是「按 TRIM 原文规范实现的文本引导剪枝」；官方代码的产物是另一回事，两句话都要写 |
| 动 GPU 7 的部署 | 只调 API |
| 用非确定性判官 / 不记分辨率 | `Agent.md` 已定 |

---

## 五、回传

`results/archive/10_REPORT_T1.5_<日期>.md`，必须含：

1. **结论先行**：按 §二 判定表落在哪一行，一句话
2. 探针五读数原样（含 TYPO 的条带保留占比）
3. 权重比对的 config 表 + 同名张量相等/不等计数（含 `image_aspect_ratio`）
4. 四格表：ASR、vs none 的 b/c、McNemar p、严格/通用分层两行
5. 分类目 ASR 全表
6. `max_tokens=60` 截断结论
7. 所有结果块（4 块 `kind=asr` + 探针块 `kind=smoke`）

**当天回传。** 10-28 的硬决策要的就是 §二 那一行结论：
要么「配置对了、效应不存在」升级为「官方代码与原文规范矛盾、按原文规范可复现方向」，
要么 H1 被探针直接否掉——两种都比现在的收尾有分量。
