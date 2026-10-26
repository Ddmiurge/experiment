# T1.4 系列实验包（2026-10-26）

复现目标：Token-Pruning 论文 **Table 2「Main Results」MM-Safety 列**
`Original 59.37 → TRIM 65.87（+6.50pp）`

---

## 一、结论速览

**唯一基线对上原文的配置（原版栈 + SD）：**

| 条件 | ASR | 目标 | 偏差 |
|---|---|---|---|
| 无剪枝 | 0.5840（233/399） | 0.5937 | **−0.97pp ✓** |
| TRIM 0.25 | 0.5664（226/399） | 0.6587 | −9.23pp |
| 差值 | **−1.75pp** | +6.50pp | 方向 ✗ |

flip：a=154 / b=12 / c=19 / d=214，McNemar p=0.281（不显著）。

**六组配置全部为负**（−0.50 ~ −5.01pp），详见 `reports/09_REPORT_T1.4d_2026-10-26.md` §2.1。

按 TASKS §四判定表 → 「配置对了、效应不存在」→ 停止复现，转交付物 1 改写。

---

## 二、目录结构

```
reports/      05(首轮SD) 06(TYPO) 07(状况与计划) 08(定位:预处理根因) 09(T1.4d 主报告)
scripts/      全部实验脚本（14 个，均放在 /tmp 运行，仓库代码零修改）
results/      每组的 kind=asr 结果块（.block.txt + .json）
answers/      每组 399 条的完整生成回答（复现材料，可用于重新判分）
summaries/    分类目 ASR 汇总、flip 计数
env_*.txt     两个 venv 的 pip freeze（环境锁定）
```

## 三、脚本清单与用途

| 脚本 | 用途 | 运行环境 |
|---|---|---|
| `t1_4_generate.py` | 我们的管线（llava-hf）生成，支持 `--split SD/TYPO`、`--condition noprun/trim` | `~/venvs/exp` |
| `t1_4_judge.py` / `t1_4b_judge.py` | 官方判官协议判分 + 分类目 ASR | `~/venvs/exp` |
| `t1_4c_hook_trim.py` | 方案A：forward 内部替换（hook `get_image_features`）A/B 对比 | `~/venvs/exp` |
| `t1_4c_orig_trim.py` | 方案B：原版栈 TRIM（官方代码 + 合并权重） | `~/venvs/trim-orig` |
| `t1_4c_orig_noprun.py` | 原版栈无剪枝 | 同上 |
| `t1_4c_orig_trimauto.py` | 原版栈 `TRIM:-1`（出厂默认，仅参考） | 同上 |
| `t1_4c_orig_sd_{noprun,trim}.py` | 原版栈 + SD 两格（vicuna_v1） | 同上 |
| `t1_4d_sd_lv1_{noprun,trim}.py` | 原版栈 + SD 两格（llava_v1，官方默认模板） | 同上 |
| `t1_4d_offw_{noprun,trim}.py` | TYPO + 官方 `Rephrased Question` 两格 | 同上 |
| `t1_4c_judge.py` | 原版栈判分 + 三方对比 | `~/venvs/exp` |

## 四、复跑要点

1. **原版栈环境**：`torch==2.1.2 + transformers==4.37.2`（见 `env_trim_orig_pip_freeze.txt`），代码走 `third_party/trim`（未修改）
2. **合并权重**：`FreedomIntelligence/llava-v1.5-7b-TRIM`，其 `config.json` 的 `mm_vision_tower` 是作者机器上的相对路径，需改为本机 CLIP 路径
3. **剪枝率**：脚本里 `model.get_model().vision_tower.token_reduce_func = "TRIM:0.25"`（出厂为 `TRIM:-1`）
4. **预处理**：原版栈走 `process_images` + `expand2square`（关键！llava-hf 默认 `do_center_crop=True` 会把 1024×180 的文字条裁掉 82%）
5. **数据**：MM-SafetyBench parquet（`~/data/MM-SafetyBench`）；官方问法取自 `third_party/mm-safetybench/data/processed_questions/*.json` 的 `Rephrased Question` 字段
6. **同批样本**：固定 399 个 `(cat, qid)`，由 `--ids-from` 锁定，跨所有实验可比

## 五、关键发现清单

1. **图像预处理 bug**（我们的管线）：llava-hf 中心裁剪只保留 17.6% 宽度 → TYPO 结论全部作废，修正后无剪枝 ASR 从 56.64 → 75.94
2. **拼接方式无问题**：hook 内部替换 vs 手工拼 embeds，399/399 回答完全一致
3. **权重/环境无问题**：官方代码 + 合并权重，方向依旧为负
4. **问法差异**：HF 数据集三种 split 全用 SD 问法；官方 TYPO 用 "The bottom of the image shows…"，1680/1680 不同；换官方问法重跑仍为 −5.01pp
5. **模板差异**：官方默认 `llava_v1`，`vicuna_v1` 只差系统提示词一个词；两者结果同量级（−1.50 / −1.75pp）
6. **剪枝效应**：六组全部为负，翻转样本仅 6–12% 且双向抵消；SD 上 p≈0.28/0.31，属"未检出"而非"证明有害"

## 六、待 Mac 决策

1. 是否跑 **原版栈 + SD 全量 1680×2**（约 1.5–2h，置信区间 ±5pp → ±2.4pp），把"未检出"钉成"确实没有"
2. 交付物 1 的改写口径（判官差异 Qwen3-8B vs GPT-4o-mini 需在文中声明）
3. HF 版 TYPO 图像被裁成纯文字条（1024×180）与官方"画布+底部文字"的差异，是否需要在交付物中记录
