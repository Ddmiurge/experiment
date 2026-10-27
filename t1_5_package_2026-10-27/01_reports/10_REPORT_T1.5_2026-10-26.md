# 10 · T1.5 报告：剪枝方向（H1）与权重（H2）· gpu84 · 2026-10-26

> 按 TASKS.md（commit d1c2d91）执行，严格按 §一 顺序：探针 → 权重比对 → 格子矩阵 → 判分。
> 同日伴随报告：`10_REPORT_judge_stability_2026-10-26.md`（判官可重复性，按 §三 降级保留）。

---

## 一、结论先行

**探针证实 H1 的"代码与原文规范相反"这一事实，但把方向改对之后，效应依然不存在。**

按 TASKS §二 判定表，落在**第三行**：

| 判据 | 实测 | 落入 |
|---|---|---|
| 探针 `sim_diff` | **−8.96（SD）/ −9.30（TYPO）**，逐样本 100% 为负 | < 0 ✓ |
| `trim_highsim − none`（vanilla） | **−1.00pp**（净 −4，p=0.636） | ≈ 0 |
| `trim_highsim − none`（TRIM-FT） | **+0.25pp**（净 +1，p=1.000） | ≈ 0 |
| `random − none` | **+1.01pp**（净 +4，p=0.597） | ≈ 0 |

→ **结论：剪什么不重要 → H1 解释力下降，回 H3（总体/判官）+ H4（样本量），跑 SD+TYPO 合并总体全量。**

一句话表述建议（交付物用）：
> 官方 TRIM 代码的保留方向与原文 §3.2 相反（已由探针实证），但按原文规范修正方向后，
> 在 399 条 SD 样本上仍观察不到 ASR 提升（vanilla −1.00pp、TRIM-FT +0.25pp，均不显著），
> 随机剪枝同为 +1.01pp（不显著），因此"剪什么"在本判官与样本量下不可分辨。

---

## 二、步骤 1 · 探针五读数（原样）

**SD（399 条，与历史格子同 id）**

```
split=SD  n=399  保留=144
1) 官方保留集 vs 丢弃集 相似度均值差 = -8.9624
   有多少比例的样本 sim_diff < 0 : 1.000
2) Jaccard(官方保留集, 高相似度保留集) = 0.0000
3) Jaccard(官方保留集, 随机子集)       = 0.1432
4) 文本敏感度 Jaccard(换文本后保留集不变): empty=0.2487 generic=0.2446 swap=0.4665
```

**TYPO（1680 条全量）**

```
split=TYPO  n=1680  保留=144
1) 官方保留集 vs 丢弃集 相似度均值差 = -9.3049
   有多少比例的样本 sim_diff < 0 : 1.000
2) Jaccard(官方保留集, 高相似度保留集) = 0.0000
3) Jaccard(官方保留集, 随机子集)       = 0.1429
4) 文本敏感度 Jaccard(换文本后保留集不变): empty=0.3917 generic=0.3695 swap=0.7094
5) TYPO 底部关键词条带被保留的 token 占比 = 0.1658
```

**读法与两点更正**：

1. **H1 事实成立**：`sim_diff` 为负、逐样本 100%、与高相似度保留集 Jaccard=0（完全不相交）。
2. **「官方≈随机」得到精确验证**：两个 144 子集、总量 576 时，随机 Jaccard 理论值 = 144²/(2·144·576−144²) = **0.1429**；实测 SD 0.1432 / TYPO 0.1429。**脚本注释里写的"期望 0.25"是笔误**（0.25 是重叠比例，不是 Jaccard）。
3. **TYPO 条带保留 16.58%**（< 25%），即关键词条带被系统性多删 1/3 —— 直接解释 TYPO 上 −5.01pp 的显著下降。
4. ⚠ **TYPO 探针实际跑了 1680 条全量，不是 399**：`--ids-from /tmp/t1_4c_typo399_data.json` 是 JSON 数组格式，脚本按 jsonl 逐行解析失败后回退为无过滤（SD 用的回答文件是 jsonl，正常锁到 399）。**探针结论不受影响**（n 更大），但报告需注明 id 集不同。

---

## 三、步骤 2 · 权重与配置比对（H2）

**config 关键字段**

| 字段 | vanilla (llava-1.5-7b-hf) | TRIM-FT (llava-v1.5-7b-TRIM) |
|---|---|---|
| `image_aspect_ratio` | 无（HF 转换未保留） | **`pad`** |
| `mm_patch_merge_type` | 无 | `flat` |
| `mm_vision_token_reduce_func` | 无 | `TRIM:-1`（出厂默认，脚本已覆盖为 0.25） |
| `architectures` | `LlavaForConditionalGeneration` | `LlavaLlamaForCausalLM` |

**`image_aspect_ratio = pad` → 原版栈确实走 `expand2square`**，T1.4 报告"预处理已修"这条对 TYPO 成立。

**张量级（改名映射后）**

| 组 | 张量数 | 完全相等 | 相对 Frobenius 差（中位/最大） |
|---|---|---|---|
| language_model.model.layers | 288 | 0 | 5e-2 ~ 1.2e-1 |
| multi_modal_projector | 4 | 0 | **1.30 ~ 1.39（几乎完全不同）** |
| vision_tower | 391 | 0 | ~1.5e-3（bf16 精度噪声量级） |
| 形状不一致 | 2 | — | `embed_tokens`/`lm_head`：HF 词表被填充到 **32064**，原版 **32000** |

**结论（H2）：TRIM-FT 是**真正微调过的 checkpoint**——投影层几乎重写、LLM 层有实质偏移；只有视觉塔基本未动。
因此 TASKS §3.1 的拆分成立：

```
Δ_total(原文 +6.50pp) = Δ_checkpoint + Δ_pruning
Δ_checkpoint = TRIM-FT|none − vanilla|none = 0.5840 − 0.5789 = +0.51pp
Δ_pruning(官方符号) = 0.5664 − 0.5840 = −1.76pp
Δ_pruning(按原文规范) = 0.5865 − 0.5840 = +0.25pp   ← 本轮关键格子
```

---

## 四、步骤 3+4 · 四格矩阵与配对分析（SD，399 条，同批 id）

| 权重 | none | trim_official（低相似度） | **trim_highsim（高相似度）** | random |
|---|---|---|---|---|
| **vanilla** | 0.5789（231） | 0.5739（229） | **0.5689（227）** | 0.5890（235） |
| **TRIM-FT** | 0.5840（233） | 0.5664（226） | **0.5865（234）** | — |

**配对（vs none，McNemar 精确双侧）**

| 行 | 格子 | a | b | c | d | 净 | p |
|---|---|---|---|---|---|---|---|
| vanilla | trim_official | 151 | 17 | 19 | 212 | −2 | 0.868 |
| vanilla | **trim_highsim** | 150 | 18 | 22 | 209 | **−4** | **0.636** |
| vanilla | random | 150 | 18 | 14 | 217 | +4 | 0.597 |
| TRIM-FT | trim_official | 154 | 12 | 19 | 214 | −7 | 0.281 |
| TRIM-FT | **trim_highsim** | 146 | 20 | 19 | 214 | **+1** | **1.000** |

**严格 / 通用分层**（严格类目 137 条 42% 饱和在 1.0）

| 行 | 分层 | none ASR | trim_highsim ASR | 净 | p |
|---|---|---|---|---|---|
| vanilla | 严格 | 1.000 | 1.000 | 0 | 1.000 |
| vanilla | 通用 | 0.359 | 0.344 | −4 | 0.636 |
| TRIM-FT | 严格 | 1.000 | 1.000 | 0 | 1.000 |
| TRIM-FT | 通用 | 0.366 | 0.370 | +1 | 1.000 |

**判分完整性**：解析失败 **0**；判官输出均值 2.0 tokens；`max_tokens=60` 无截断风险。

---

## 五、分类目 ASR（四格全表）

| category | vanilla none | vanilla highsim | TRIM-FT none | TRIM-FT highsim |
|---|---|---|---|---|
| Illegal_Activitiy | 0.6522 | 0.5217 | 0.3478 | 0.3913 |
| HateSpeech | 0.2564 | 0.2051 | 0.2564 | 0.3333 |
| Malware_Generation | 0.4000 | 0.3000 | 0.3000 | 0.3000 |
| Physical_Harm | 0.4118 | 0.4412 | 0.3824 | 0.4118 |
| EconomicHarm | 0.0690 | 0.0345 | 0.0000 | 0.0345 |
| Fraud | 0.3243 | 0.4324 | 0.5135 | 0.4865 |
| Sex | 0.1154 | 0.0000 | 0.1923 | 0.0385 |
| Political_Lobbying | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Privacy_Violence | 0.2121 | 0.3030 | 0.3030 | 0.3030 |
| Legal_Opinion | 0.8710 | 0.8065 | 0.9032 | 0.9032 |
| Financial_Advice | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Health_Consultation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gov_Decision | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

（vanilla trim_official / random 的分类目见 `results/t1_5_vanilla_sd_trim-official/`、`..._random/` 结果块）

---

## 六、过程中发现的两个脚本 bug（均在 `/tmp` 绕过，未改仓库文件）

| 脚本 | bug | 绕过方式 |
|---|---|---|
| `t1_5_orig_highsim.py` | `patch()` 用 `types.MethodType(func, CLIPVisionTower)` 把**类**绑成 self → 运行必报 `AttributeError: type object 'CLIPVisionTower' has no attribute 'token_reduce_func'`。正确做法是把普通函数直接赋给类属性 | `/tmp/t1_5_orig_highsim_fixed.py`：import 该模块后只替换 `patch()`，其余逻辑（含 5 条自检）原样复用 |
| `t1_5_judge.py` | ① raw jsonl 按 tag 命名，两行实验 tag 同名 → 第二行覆盖第一行判定记录；② `cfg` 含 `prune_sign=n/a`，`/` 进入文件名 → `FileNotFoundError` 崩溃，结果块写不出 | `/tmp/t1_5_final_judge.py`：行前缀 tag（`van_*`/`ft_*`）、合法 cfg（`low_sim`/`high_sim`/`random`/`na`）重跑并 emit，判官协议完全一致 |

---

## 七、结果块（7 块 `kind=asr` + 1 块 `kind=smoke`）

`results/` 下目录：

- `t1_5_vanilla_sd_none` / `t1_5_vanilla_sd_trim-official` / `t1_5_vanilla_sd_trim-highsim` / `t1_5_vanilla_sd_random`
- `t1_5_trimft_sd_none` / `t1_5_trimft_sd_trim-official` / `t1_5_trimft_sd_trim-highsim`
- `t1_5_probe_trim_direction`（`kind=smoke`，pass=3/3：方向反、逐样本全反、条带被删）

判官原始输出：`/tmp/t1_5_raw_{row}_{tag}.jsonl`（含 `raw_output` 与 `n_tokens`）。

---

## 八、待决

| 项 | 说明 |
|---|---|
| 是否按第三行继续 | TASKS 判定表要求"回 H3+H4，跑 SD+TYPO 合并总体全量"；两方案：① 只对 SD 全量 1680 跑 none/highsim/random 三格（约 1.5h×3，两卡约 2.5h）② SD+TYPO 合并 3360 条总体 |
| 判官口径 | 严格类目 137 条全部饱和 1.0，可观测空间只剩通用类目 262 条；若要提功效，需先解决饱和 |
| `TRIM:-1` 自动保留率 | 已在 T1.4 附注测过（TYPO 0.7193），本轮未纳入矩阵 |

## 九、红线自查

仓库代码零修改（`third_party/trim`、`src/`、`experiments/*/scripts` 均未动；两处 bug 用 `/tmp` 启动器绕过）；保留率固定 0.25；四格（vanilla）+ 三格（TRIM-FT）齐报，含随机阴性对照；判官确定性（同日 A/B 100%）；GPU 7 仅走 API。
