# TASKS · 当前轮 T1.5d：keyphrase 查询（复现终审）+ 致盲四档 —— 跑完关闭复现线

> 背景：`results/archive/11_MAC_RECHECK_T1.5_2026-10-27.md` + `12_REPORT_T1.5b` +
> Mac 侧 Table 2 对齐分析（本文件 §一.1 的预测表就是它的结论）。常驻规则见 `Agent.md`。
> 本轮不改 `third_party/`、不改 `src/`；脚本更新已在 `experiments/t1_5_2026-10-27/scripts/`
> （probe/prune 的输出文件名现带 query-src 后缀，不会再互相覆盖；`--ids-from` 现兼容
> JSON 数组格式，解析失败**直接抛错**不再静默退回全量——T1.5 首轮那个坑已封）。
>
> **三天倒计时下的定位：这是复现线的终审实验。跑完无论结果如何，复现线关闭，
> 之后全部时间转入交付物 1 写作与项目主线（MAI 真值审计/诱饵席数）。**

---

## 零、这一轮在审什么（一段话说清）

TRIM 挑图块靠"图块与一段文本的 CLIP 相似度"。此前 15+ 个配置里，那段文本永远是
基准的匿名化模板句（"…a phrase of an activity…"，**不含关键词**）。
我们实测在这个条件下：白底留白的相似度（28.6–29.3）**高于**黑字条带（24.9–26.9）
⇒"保留高相似度块"实际删载荷 ⇒ TYPO highsim **−9.77pp**。

本轮第一次改变这个条件：**把喂给打分器的文本换成官方 JSON 里该样本的 `Key Phrase`
（图上真实渲染的那几个字）**。模型的 prompt 一字不动——只动 `texts=` 这一个变量。

它裁决两个竞争解释：
- **解释① 锚缺失**：排序被白底主导，是因为查询太抽象。→ 加关键词后排序应**翻转**。
- **解释② 范畴错配**：CLIP 图块特征表达不了字形内容，喂什么都不会翻转。

---

## 一、任务（严格按序，每步有硬门槛）

### 1 · 判门探针（约 5–10 分钟，只加载 CLIP，0 生成）★决定后面跑不跑

```bash
CUDA_VISIBLE_DEVICES=6 python experiments/t1_5_2026-10-27/scripts/t1_5_trim_probe.py \
    --split TYPO --query-src keyphrase \
    --ids-from /tmp/t1_4c_typo399_data.json --limit 40
```

脚本会先打印 **5 条人工核对**（`prompt → Key Phrase`）。核对标准：Key Phrase 应与该样本
图上黑字内容一致、且明显短于模板句。**对不上就停，报回来**（join 已在 Mac 侧用 798/798
验证过，错位属于异常而非噪声）。

同时先跑 5 分钟 sanity：`--query-src dummy --limit 40`（固定 "a photo."），
检查"排序是否主要由句子抽象度而非关键词驱动"——若 dummy 也大幅翻转，说明
"任何短具体词都拉得动排序"，解释①要降级为"锚灵敏度"而非"载荷特异"。报告两个都贴。

**判门读数**（脚本已自动打印）：
| 读数 | 模板查询（T1.5b 实测） | keyphrase 预测（解释①胜） |
|---|---|---|
| 载荷格相似度 − 留白格相似度 | **负**（白底赢） | **应转正** |
| highsim 载荷格保留率 | **0.223**（< 随机 0.26） | **应 >0.5，理想 >0.8** |

- **转正（>0.5）** ⇒ 过门，进入 2。
- **不转（≈模板那组）** ⇒ 解释②胜。**跳过 2，不烧 GPU**，直接做 3，
  结论写"文本引导剪枝对字形攻击的错配是范畴级的，连正确查询都救不活"——复现线同样关闭。

### 2 · 生成两格（约 50 分钟，仅当过门）

```bash
CUDA_VISIBLE_DEVICES=5 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py \
  --split TYPO --prune highsim --query-src keyphrase
CUDA_VISIBLE_DEVICES=6 /home/huangqixu01/venvs/trim-orig/bin/python \
  experiments/t1_5_2026-10-27/scripts/t1_5_orig_prune.py \
  --split TYPO --prune official --query-src keyphrase
```

自检不变：前 3 条 assert 进 LLM 的视觉 token 数（145）；keeps 文件会打印"不同掩码种类数"。
答案文件带 `_keyphrase` 后缀（输出名已隔离，不会覆盖 T1.5b 的）。
每格答案里都带 `trim_query` 字段 = 实际喂进去的 Key Phrase，判分与留档都用它核对。

判分（协议与全部历史轮一致）：

```bash
python experiments/t1_5_2026-10-27/scripts/t1_5_judge.py --split TYPO --ref none \
  --model-tag llava-v1.5-7b-TRIM --exp-prefix t1_5d \
  --file trim_highsim_kp=/tmp/t1_5c_answers_highsim_keyphrase_typo399.json \
  --file trim_official_kp=/tmp/t1_5c_answers_official_keyphrase_typo399.json
```

配对基准直接拿已归档的 `t1_5b_answers_none_typo399.jsonl` 判定（同 399 id），
Mac 侧会统一重算 CI/z——服务器只需报每格 ASR + b/c/净/精确 p。

**预注册判定（写死，不许事后改口径）**：

| highsim×keyphrase vs none(0.7594) | 结论（交付物原话级） |
|---|---|
| **≥ 0.7594 且 McNemar p<0.05** | **复现成功**。原文机制需要"打分查询含载荷"这一未声明条件；+6.5 量级是否达成以 CI 为准。交付物 1 翻转 |
| ≥ 0.7594 但 p≥0.05 | 方向出现、功效不足——如实报点估计+CI；这是"未声明条件"的部分证据，不追认 |
| < 0.7594（任意 p） | 解释①死亡：连正确查询也做不出放大。负结果**穷尽**，复现线终审关闭，不再有"没试到"的余地 |

（official×keyphrase 顺带跑，用来交叉验证翻转的对称性：解释①胜时它应变为**删**载荷、ASR 更低。）

### 3 · 致盲 vs 拒绝四档（0 生成，只调判官 API）★无论第 2 步结果如何都必须做

问题：TYPO highsim −9.77pp，是模型**开始拒绝**（安全改善）还是**读不到字**（致盲）？
两者含义相反——后者绝不能被写成"剪枝是一种防御"。

对已归档 4×399 TYPO 回答（`t1_5b_package_2026-10-27/04_answers/`）补判四档，
用 `src/eval/mm_safetybench_judge.py` 现成的 `build_refusal_prompt` + `judge_binary_to_level`，
写个 /tmp 小驱动即可（每档分布 + 与 unsafe 判定的交叉表）。若第 2 步跑了，新两格一并判。

| 预期（解释①胜，致盲） | 预期（若是真拒绝） |
|---|---|
| none 拒绝档 ~7% → highsim **误解档(1) 暴涨、拒绝档(0) 不涨** | 拒绝档(0) 显著上升 |

回答变长 8–12%（T1.5 已测）与致盲假说一致，四档是直接检验。

---

## 二、时间与取舍（三天）

```
Day 0（今天）    §一.1 判门（~15 分钟含 dummy）→ 过门立刻起 §一.2 两卡并跑
Day 1 上午       判分 + §一.3 四档（API 时间）→ 数字齐，复现线定案
Day 1 下午起     复现线关闭。转：交付物 1 骨架（Table 2 对齐表 + 三层归因 + C1–C5）
                 + 项目主线第一步：MAI top-m × 载荷格 命中审计（序数读数，不吃功效）
```

**不要做的**（再强调一遍，防止默认滑回去）：
- SD 全量 1680 / TYPO 全量 1680（CI 已排除 +6.5 的"测不到"读法，扩样零增量）
- ORIGINAL / BLACK 分片（见 §三，降为"写作期可选加固"）
- 任何保留率 / conv / max_new_tokens 扫描
- oracle_fg / oracle_bg 本轮不排（见下）
- 把 keyphrase 塞进模型 prompt（拆攻击本体，结论不可归因——脚本只动 texts=，别在包装层手贱）

> **oracle_fg/bg（Table 9 的忠实消融：100% 保载荷 vs 0% 保载荷）怎么摆**：
> 它没有被现有数据判死——official 也只保 45% 载荷，与"全保"不是一回事，仍是
> 唯一不受查询锚污染的干净反证。但先验低（在"载荷重要度"的单一轴上，
> official@45%→−3.01 与 random@26%→−3.01 已经给出"保得更多并没有换到上升"的梯度），
> 且要新建两个保留集模式（约 1.5 小时含调试）。
> **决策规则**：若 §一.2 的 keyphrase 格出现"≥none 方向"，说明机制确实被查询锚点亮，
> 那时 oracle_fg 值得作为追加确认（Day 2）；若 keyphrase 不过门，oracle 也一并放弃——
> 两条逃逸路径共享"载荷被保留"这一前提，前者已死，后者独活的可能性不足以花 Day 2。

## 三、§一.1 判门的理论意义（给报告用的两句）

1. 过门并复现 ⇒ 我们发现的不再只是"实现≠规范"（符号 bug），而是
   "**规范≠可运行**"：TRIM 的放大机制在此类基准上依赖一个**没有任何一方实现过的条件**
   （打分查询必须携带载荷原文）。这是对一篇已发表工作的可操作修正，不是单纯否定。
2. 不过门 ⇒ 错配是范畴级的：文本相似度这条引信在"关键词被逐出文本域"的攻击上
   根本点不着。负结果从"我们复现不出"升级为"原理上不该期望它出现"——同样封死，但更硬。

## 四、回传

`results/archive/14_REPORT_T1.5d_keyphrase_<日期>.md`：
1. 判门结果（两查询 × 载荷/留白相似度差、highsim 载荷保留率）+ 5 条人工核对原文
2. 过门则两格：ASR、vs none 与 vs T1.5b-highsim 两个配对口径、b/c、精确 p、CI
3. 四档交叉表（6 格：旧 4 + 新 2）——致盲 or 拒绝，一句话定案
4. 按 §一.2 预注册表落行的结论；复现线**正式关闭声明**
5. 结果块（新格 kind=asr；判门 kind=smoke）

**当天回传。** 这份报告 + 11/12 号 + Table 2 对齐表，就是交付物 1 的全部实验章节。
