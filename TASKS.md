# TASKS · 当前轮 T1.6：oracle 四格仲裁实验（保留轴正端是否存在）

> **本轮唯一口径**：`T1.6_oracle_protocol_2026-10-29.md`（仓库根，原 `oracle四格实验协议…` 改名 ASCII）。
> 背景与判读框架：`思路总览_从复现失败到保留轴_2026-10-29.md` §2–§5（Mac 侧）。
> 常驻规则 `Agent.md`；判据沿用 `实验口径与判据`。禁调参凑数、禁改 `third_party/`、`src/`。
> Mac 已完成：掩码模块在探针归档上实测（恒 6 行 / 33 种集合 / 溢出补足恒 0 / ③零命中），
> 分析脚本已在 ①② 归档上自测（random 行 −3.01/p=0.1263 逐位吻合）。

---

## 零、本轮在裁决什么（30 秒版）

九格全部是"让判据自己挑 token"——打掉的是**判据层**。
原文证据链还剩一层从未被独立检验：**人为完美地删背景、只留载荷（不经任何相似度判据），
ASR 会不会升过不剪枝基线**（Table 9 的 oracle 操作）。
④ O-载荷 = 该主张的忠实实现（裁决格）；③ O-留白 = "移除前景"净化版（哨兵格）。
载荷/背景按**冻结真值口径**（T1.5b 探针归档的逐样本 payload_rows）构造，**不重新定义、不得新写**。

---

## 一、步骤（严格按序；每步有停线条件）

### 0 · 掩码审计预跑（CPU，几分钟）——**修订#3 终版：纯秩构造，无任何待决参数**

```bash
/home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_6_2026-10-29/scripts/oracle_masks.py --data /tmp/t1_4c_typo399_data.json
```

④=活动度降序前 144 格、③=升序前 144 格（互斥由构造保证；Otsu 墨迹只当诊断）。
审计打印 |ink| 分布、cov4 中位、leak3 计数。**停线只有一条**：
`leak3>0`（任一样本）或 `median(cov4)<0.9` ⇒ ④已不是"保载荷"，停报勿进冒烟；
否则 PASS 直接往下。历史行口径（含 T1.5b §3.1 的 45.07/22.28）一律附录化，不进主表。

### 1 · 读出率口径定标（CPU，1 分钟，**先于一切生成**）

```bash
python experiments/t1_6_2026-10-29/scripts/t1_6_readout.py \
    --answers experiments/t1_4d_2026-10-26/answers/t1_4_answers_noprun_400.json \
    --tag SD-none --gate-r2-max 0.05 \
    --msb-json third_party/mm-safetybench/data/processed_questions
```

**判定门已换口径（服务器 step1 FAIL 的正确处置，协议已附冻结修订）**：
报告 §4.4 的"42/399=10.5%"当年代码未归档、重建正则无法复现（你们扫到 12–101 的表已入库），
**不再追这个数**——那会沦为"凑到匹配为止"。改用归档可复算的 **R2（Key Phrase 连续串包含）**：
要求 SD-none 的 R2 ≤5%（期望 **5/399=1.3%**，Mac 侧已实测）。它验证的是同一条结构前提
"SD 连完整模型都读不出载荷"，且这次有定义、有代码、有输入文件，三件齐全。
R1 仍会打印，但只作探索性参考，不判停。

### 2 · 冒烟生成：每格 3 条（两卡各 ~2 分钟）

```bash
CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_6_2026-10-29/scripts/t1_6_oracle_prune.py --prune oracle_payload --limit 3 \
  --out /tmp/t1_6_smoke_p.json
CUDA_VISIBLE_DEVICES=6 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_6_2026-10-29/scripts/t1_6_oracle_prune.py --prune oracle_blank --limit 3 \
  --out /tmp/t1_6_smoke_b.json
```

冒烟自检全部是"不该响"的构造级断言：④∩③=∅、len(keep)=144、assert 145、
运行时掩码逐条 == step0 审计件。响了即停（=preprocess 漂移或代码 bug），不存在需要
现场裁量的检查。keeps 每行带 cov4/leak3 审计量，§5 保留率核验直接反解。

### 3 · 全量生成（两卡并行，约 3h/格；协议 §7 估 1–2 天含判分）

去掉 `--limit`、`--out` 用默认（`/tmp/t1_6_answers_oracle_{payload,blank}_typo399.json`）。
断点不续跑（本脚本无 resume；中断即整格重跑，别手工拼文件）。

### 4 · 判分（判官与全部历史格子逐项一致）

```bash
python experiments/t1_5_2026-10-27/scripts/t1_5_judge.py --split TYPO --ref none \
  --model-tag llava-v1.5-7b-TRIM --exp-prefix t1_6_oracle \
  --file none=/tmp/t1_5b_answers_none_typo399.json \
  --file random=/tmp/t1_5b_answers_random_typo399.json \
  --file blank=/tmp/t1_6_answers_oracle_blank_typo399.json \
  --file payload=/tmp/t1_6_answers_oracle_payload_typo399.json
```

（①②回答若在 /tmp 已清，用包内 `t1_5b_package_2026-10-27/04_answers/` 的路径替代；
①② ASR 应与归档 75.94/72.93 逐位相同——判分是复用 raw 还是要重判，以 raw 文件在否为准，
重判若出偏差 >0 条即停报。）

### 5 · 近端与四档（协议 §4.3/§4.4；先于报 ASR）

```bash
python experiments/t1_6_2026-10-29/scripts/t1_6_level4.py \
  --cell none=... --cell random=... --cell blank=... --cell payload=... \
  （每项格式 tag=answers[:judge_raw]，路径同上）
python experiments/t1_6_2026-10-29/scripts/t1_6_readout.py --answers <各格> --tag <格> \
  --msb-json third_party/mm-safetybench/data/processed_questions
```

读出率序（**用 R2**）：④ ≥ ① ≥ ② ≥ ③；**等号算满足**，四格全 ≈0 记"读出信号弱"照常报；
只有**矛盾**（如 ③ > ①、③ 显著非零）才判管线问题、**不得先报 ASR**，停报。

### 6 · 回传

打包 `t1_6_package_<日期>/`（reports/scripts回答/keeps/judge_raw/结果块，同 T1.5b 结构），
报告 `results/archive/16_REPORT_T1.6_oracle_<日期>.md`：
1. §6 预注册表**原样回填**（含核验列）；
2. §5 落行：正格 / 负格 / 平 + ③哨兵是否落 [−6,−12] + 读出序是否成立；
3. 四个 `kind=asr` 结果块 + 掩码审计件 `t1_6_masks_audit.txt`；
4. 若触发"查掩码/网格错位/正则 FAIL"任一停线，如实单列。

---

## 二、硬规矩（本轮专用，叠加在通用红线之上）

| 不许 | 因为 |
|---|---|
| 重定义载荷行（换 θ、换算法、改用固定行进主表） | §2.1 冻结口径=探针归档逐样本 payload_rows；固定行只进附录敏感性 |
| 改 oracle 之外的任何条件（栈/conv/512/id/keep 形式） | 协议 §3 逐项锁死；①②复用是设计的一部分 |
| 读出序不满足还继续报 ASR | 协议 §5：序不成立=管线问题，ASR 无意义 |
| 看到结果后挪 ±4.0pp / CI 判据、临时加格进主表 | 预注册；加格一律标"探索性" |
| 把 ④ 的"载荷≈100%"口头化——必须报核验实测值 | 协议 §4.2：kept-set 反解，偏差>2pp 查掩码 |
| 全量 1680 / 换保留率 / 换分片"顺手加一格" | 复现线已关闭，本轮只裁决操作层一格，加格即稀释 |

## 三、时间预算

掩码+正则 3 分钟 → 冒烟 5 分钟 → 全量 2×3h（并行）→ 判分+四档+读出 ~1h → 回传。
**单卡日即可完成。** 跑完无论读数朝哪边，`思路总览` §7 表格 #2 即勾掉，下游进入写作期。
