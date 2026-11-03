# archive · 实验回传包索引

> 归档日期:2026-11-03（`git mv` 迁移，改名记录已暂存待提交）
> 规则:`t1_*_package_*`（服务器回传报告包）与 `t1_*_step*_fail_*`（冒烟停线记录包）一律进本目录；`experiments/`（轮次脚本）与根目录协议文件**不动**——服务器命令与历史文档仍按原路径引用它们。
> 历史文档（复现实验报告/思路总览/研究大纲等）中出现的 `code/t1_*_package_*` 旧路径，一律按本表映射读取。

| 旧路径（历史文档中的写法） | 现路径 |
|---|---|
| `code/t1_5_package_2026-10-27/` | `code/archive/t1_5_package_2026-10-27/` |
| `code/t1_5b_package_2026-10-27/` | `code/archive/t1_5b_package_2026-10-27/` |
| `code/t1_6_package_2026-10-30/` | `code/archive/t1_6_package_2026-10-30/` |
| `code/t1_7_package_2026-11-02/` | `code/archive/t1_7_package_2026-11-02/` |
| `code/t1_6_step1_fail_2026-10-29/` | `code/archive/t1_6_step1_fail_2026-10-29/` |
| `code/t1_6_step2_fail_2026-10-29/` | `code/archive/t1_6_step2_fail_2026-10-29/` |

## 各包一句话档案

| 包 | 轮次 | 内容 | 关键读数与判定 |
|---|---|---|---|
| **t1_5_package_2026-10-27** | T1.5 判据层·SD | 方向×权重矩阵（10_REPORT_T1.5）、判官稳定性验证、10-27 诊断 | SD 基线对齐 58.40 vs 59.37；best-combo highsim +0.25pp（3.99σ 拒绝 +6.50） |
| **t1_5b_package_2026-10-27** | T1.5b 判据层·TYPO + 探针 | TYPO 四条件（none/official/highsim/random）、`06_probe/`（Jaccard=0、逐行保留率、相似度剖面）、`07_judge_raw/` 原始判定 | highsim −9.77pp（p<0.0001，7.21σ）；"判据选反"剖面的全部原始证据在此 |
| **t1_6_package_2026-10-30** | T1.6 操作层·文字介质（含 E4 前半） | 16_REPORT、rev#3 纯秩掩码（`06_masks/`）、四格 keeps/answers/judge_raw、配对统计（`08_stats/paired_stats.json`） | ④载荷保留 +2.76pp 平（+6.50 单侧 p=0.011 排除）；③留背景 −23.81；读出序单调；判官跨会话漂移 1/399 |
| **t1_6_step1_fail_2026-10-29** | 停线记录 | 读出正则不可复算的取证包（12–101 扫描表、`t1_6_readout_SD-none.jsonl`） | 促成 R2 口径（1.3%）与"不追旧数"原则 |
| **t1_6_step2_fail_2026-10-29** | 停线记录 | 几何不一致诊断（`t1_6_geom_diag.json` 399 条逐样本、IoU/位移测量） | 拦住"两格含义互换"事故；促成 rev#2/#3 cell 级纯秩构造 |
| **t1_7_package_2026-11-02** | T1.7 场景介质（E1+E1'+E4） | 17_REPORT、前景掩码+类目×branch 审计表（B1 16.3%）、五格 answers/keeps/judge_raw、`07_lguard/` LG3 verdicts、`09_stats/` 配对与两判官对照 | E1 文字地板 36.34%（图贡献 **+22.06pp**）；④前景保留 −0.50pp CI[−3.28,+2.28]——**第四枚判定闭合**；两判官族符号/排序一致 |

## 完整性说明

- 包内 `05_judge_raw/`/`06_judge_raw/` 与 `07_judge_raw/` 等判定原始记录**只增不删**；重算核对一律从包内 raw 文件出发，不复用报告汇总数（项目纪律）。
- `results/inbox/`（结果块流水）与 `results/archive/`（阶段文档）保持原位，未进本目录。
