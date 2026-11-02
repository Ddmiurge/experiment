# TASKS · 当前轮 T1.7：E1 文字地板 + E1' 场景操作层 + E4 判官免疫

> 唯一口径：`T1.7_E1_E1prime_protocol_2026-11-02.md`（预注册判读全部在其中）。
> 上游：`研究大纲_2026-11-02.md` §4.2/§4.3。常驻规则 `Agent.md`；不改 `third_party/`、不改 `src/`。
> 脚本在 `experiments/t1_7_2026-11-02/scripts/`（Mac 已过语法与逻辑自测）。
> 目标一句话：把"四枚判定"的最后一枚（操作层×场景介质）与地板/判官两环补齐——**这是填空，不是开线**。

---

## 一、顺序与命令

### 1 · E1 文字地板（今天，~45 分钟全程）
```bash
CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_7_2026-11-02/scripts/t1_7_textfloor_gen.py --data /tmp/t1_4c_sd399_data.json
# 判分（复用 t1_5_judge，--split SD --ref none 指向既有 SD-none 回答；none 那份若 /tmp 已清用包内）
```
判读=协议 §2 表（E1 落 [53.4,58.4] ⇒ SD 图贡献 ≤5pp；<53.4 ⇒ E1' 升级为主证）。

### 2 · E1' 掩码（今天并行；装不上 GD/SAM 就 `--force-b2`，协议允许）
```bash
python experiments/t1_7_2026-11-02/scripts/t1_7_fg_masks.py \
  --data /tmp/t1_4c_sd399_data.json --msb-json third_party/mm-safetybench/data/processed_questions
```
必须回传**类目×branch 审计表**与 B1 率（<20% 且未声明 B2-only ⇒ 停线）。
冒烟再试生成：`t1_7_oracle_prune.py --prune payload|blank --limit 3`
（assert 145；打印 branch/fg_n/keep=144；`blank` 的 keeps 与 payload 的 fg 交集应为 0）。

### 3 · E1' 全量三格（明天，两卡轮转，各 ~25–30 分钟）
```bash
... t1_7_oracle_prune.py --prune payload
... t1_7_oracle_prune.py --prune blank
... t1_7_oracle_prune.py --prune random
```
判分同 T1.6 流程（`--exp-prefix t1_7_oracle`），①SD-none(58.40) 用归档判定不重判。
概念读出：`t1_6_readout.py` 对四格跑 R2（协议 §3.3 哨兵序 ④≥①≥②≥③）。

### 4 · E4 LlamaGuard-3（与 3 并行或其后，1 GPU 时）
部署：vLLM 起 llama-guard-3-8B（**确认自带官方 chat template**；脚本走 /completions 自拼模板，
开跑前拿 3 条已知答案核对一次模板拼装，核对结果贴回来）。
```bash
python experiments/t1_7_2026-11-02/scripts/t1_7_rescore_lguard.py \
  --base http://127.0.0.1:8005/completions --model llama-guard-3-8b \
  --file none=/tmp/t1_5b_answers_none_typo399.json \
  --file random=/tmp/t1_5b_answers_random_typo399.json \
  --file blank=... --file payload=... \
  --qwen-map none=<该格 judge_raw> --qwen-map payload=<该格 judge_raw> ...
```
文件清单：T1.6 四格 + T1.7 四格（E1/②③④）+ T1.5 两格 SD 对照——**全部回答文件在
`t1_5*/t1_6_package_*/本轮` 内可寻**；LG3 绝对值不进表（协议 §4 资格限定已写死）。

## 二、停线（预先穷举，无现场裁量）
| 条件 | 动作 |
|---|---|
| E1 assert 混入 image token / 判分解析失败>0.5% | 停报 |
| 掩码 B1 率 <20% 且未跑 `--force-b2` 声明 | 停报（先定 headline 模式） |
| ③ 与 ④ keeps 交集非空（任一样本） | 构造 bug，停报 |
| LG3 三条模板核对不过 | E4 整节搁置，E1/E1' 照常（E4 是免疫性检验，非判定环） |

## 三、回传
`t1_7_package_<日期>/`（01_reports/17_REPORT_T1.7_<日期>.md + answers/keeps/masks审计/judge_raw/
结果块/LG3 verdicts）；报告必含：协议 §2 落点、§3.3 预注册判读三选一（含 B1-only 敏感性）、
§3.2 审计表、E4 主配对符号表。**跑完这轮，四枚判定集齐，项目全面转入写作。**
