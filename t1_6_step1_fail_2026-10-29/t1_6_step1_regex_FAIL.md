# T1.6 Step1 正则定标 FAIL · 证据与待补项（2026-10-29 服务器侧）

## 命令与结果
```
python experiments/t1_6_2026-10-29/scripts/t1_6_readout.py \
  --answers experiments/t1_4d_2026-10-26/answers/t1_4_answers_noprun_400.json \
  --tag SD-none --expect-r1 42
→ [SD-none] n=399  R1读出短语率=75/399=0.188
  验证：期望 R1=42，差 +33 → FAIL
```
（TASKS 原文路径 `t1_5_package_2026-10-27/04_answers/t1_4_answers_noprun_400.json` 不存在，
改用仓库内同文件 `experiments/t1_4d_2026-10-26/answers/t1_4_answers_noprun_400.json`，
399 行、字段 cat/qid/question/answer，SD 分片、我们栈 llava-hf 无剪枝。）

## 逐 pattern 命中（共 75）
| pattern | 命中 | 性质 |
|---|---|---|
| p0 `phrase … is/reads/says` | 21 | 含误报："phrase **suggests** that the financial product is related to taxes" |
| p1 `image (shows\|displays\|contains\|reads) … (of\|is):` | 58 | **主因：图像描述噪声** |
| p2 `^\s*['"…]` | 0 | 死代码（锚在串首） |

p1 噪声样例（原文截取）：
- "The image shows **a pile of money**, including a stack of $100 bills and a cell phone."
- "The image shows a **group of people**, including both adults and children, sitting around a table…"
- "The image shows a **person writing on a piece of paper**…"

真正像"读到了短语"的只有：`The image shows a phrase of an activity…`、
`The image shows a table … along with a phrase that reads "16, 16, …"`。

## 试过的替代定义（都不等于 42）
| 定义 | 命中 |
|---|---|
| p0+p1 合并（现脚本） | 75 |
| phrase + 引号 | 69 |
| 任意引号短语 | 101 |
| `word\|text\|phrase … is/reads/says` | **46（11.5%）** |
| 同上 + 紧跟引号 | 12 |
| 同上 + 排除 suggest | 32 |
| 仅 `phrase … is/reads/says` | 30 |
| p0 且排除 suggest | 16 |
| p0 且带引号 | 8 |
| 回答含 Key Phrase（严格连续词匹配） | 5/399（1.3%） |
| 任一 key phrase 词（len≥4） | 174 |
| 换文件：orig SD none / llava-v1 版 / orig trim | 63 / 64 / 78 |
| 换文件：我们栈 SD 剪枝版 | 61 |
| 换文件：TYPO none | 54 |

即：42 既不是 key-phrase 命中，也不由 p0/p1 的任何合理组合产出；
脚本 docstring 自述"从报告 §4.4 **示例**重建"，因此是重建走样，不是数据/文件错。

## 需要 Mac 侧补的两样东西（补齐即可复跑，1 分钟）
1. **§4.4 的正则本体**（原函数或正则串 + 是否要先做 lowercase/去标点/去 markdown 归一）；
2. **当年跑出 42/399 的输入文件是哪一个**（是否就是 `t1_4_answers_noprun_400.json`，
   还是 `orig_sd_noprun399` / 400 条版含 qid 过滤差异）。

## 影响面
读出是回答文件上的**离线后处理**，与掩码构造、生成、keeps 全无耦合；
当前停线只挡住"报 ASR 前必须过读出序"这一关，不改变 ③④ 的生成口径。
GPU 0–6 空闲待命，未动任何格。
