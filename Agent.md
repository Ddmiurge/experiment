# Agent.md · 服务器侧操作手册

> 读这份文件的是**跑在服务器上的 Agent**，不是人。人的手册是 `README.md`。
> 项目背景与完整判据在 Mac 上（`代码任务清单_2026-10-19.md`、`执行计划_2026-10-19.md`、
> `实验口径与判据_2026-10-19.md`），服务器上只有本文件。

---

## 一、你的位置

你在服务器上。服务器**能 pull、不能 push**。

**这意味着：你写的任何东西都出不来。**

代码从 Mac 经 git 下来，结果靠你打印成"结果块"、由用户复制回 Mac。
没有别的通道。不要试图建立别的通道。

---

## 二、三条硬规则

### 规则 1 · 不写正式代码

你只能写 `results/` 和临时目录（`/tmp`）。

要改 `src/` 或 `scripts/`，**把补丁打印出来交给用户**，让他在 Mac 上改、push、你再 pull。
不要直接改 —— 你改的版本回不到 Mac，会变成孤儿，而且和 Mac 上的版本静默分叉。

临时 debug 脚本放 `/tmp`，不要留在仓库里。

### 规则 2 · 不改 `third_party/`

那里是按 `third_party.lock` 检出的只读副本。所有适配写在外层（`src/pruning/` 等）。

如果某个仓库的代码必须改才能跑通：**停下来，报告**。不要改。

### 规则 3 · 每次实验跑完必须打印结果块

用 `scripts/result_block.py`。没有结果块的读数等于没跑。

```bash
python scripts/result_block.py emit \
    --exp stage1_table1_noprune \
    --levels 0030012300... \
    --cfg pruner=none,defense=none,seed=0,res=336
```

它会同时打印结果块、写 `results/<exp>/<exp>__<slug>.json` 和 `.block.txt`。
**把结果块原样贴给用户**，并告诉他 `cat` 哪个文件能拿到。

---

## 三、环境

1. 复制路径模板并填写：
   ```bash
   cp env/paths.example.yaml env/paths.yaml
   # 按实际路径改
   ```

2. 拉第三方仓库：
   ```bash
   bash env/fetch_third_party.sh
   ```
   首次跑会解析 commit SHA 写进 `third_party.lock`。**这个文件要提交**（由用户操作）。
   `TRIM` 那行目前是 TODO，T1.2 之前必须确认是哪个仓库。

3. 自检：
   ```bash
   python scripts/result_block.py selftest      # 结果块往返
   bash scripts/check_egress.sh <仓库URL>       # 出网边界
   ```

---

## 四、开工前必做（T0.1 的一部分）

```bash
nvidia-smi                      # 每张卡的实际可用显存
python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"
```

**服务器上有常驻的 Qwen-7B-Nothinking 不可动，7 张 4090 的实际可用显存必须实测，不要按标称算。**
把每张卡的可用显存报给用户 —— 它决定后面怎么分卡。

---

## 五、任务表

按"越早失败越便宜"排。**T0.4 是唯一的"一票否决"项。**

| ID | 任务 | 通过判据 |
|---|---|---|
| **T0.1** | 环境盘点 | 版本表 + 每卡可用显存 |
| **T0.2** | LLaVA-1.5-7B 跑通 | 3 张图给出非退化描述 |
| **T0.3** | LLaMA-Guard-3-8B 跑通 | 10 有害 + 10 良性；同输入两次判定一致 20/20 |
| **T0.4** | **注意力 hook 验证** | `scripts/t0_4_attention_hook.py` 三条判据全过 |
| **T1.1** | 加载 MM-SafetyBench | 按类目样本数，与原文对齐 |
| **T1.2** | 集成 TRIM | `prune(inputs, ratio) -> keep_indices`；75% 时保留集大小符合预期 |
| **T1.3** | ASR 评测脚本 | 10 条人工已知结果，判定 10/10 一致 |
| **T1.4** | 跑 无剪枝 vs 被动剪枝 | 无剪枝 ≈ 54.01；TRIM ≈ 62.33（±2pp） |
| **T2.1** | v_safe 抽取 | 逐层隐藏状态 shape 与层数一致 |
| **T2.2** | MAI | 给出 top-m 索引与对应 MSA 值 |
| **T2.3** | BTR | 从 V_drop 均匀采样 k 替换 V_keep 最低的 k；序列长度不变 |
| **T2.4** | AR | 锚点注意力 ×(1−λ) 后整个分布总和仍为 1 |
| **T2.5** | 端到端 | TRIM + SAP 三基准均值 ≈ 50.19（±2pp） |
| **T3.1** | MSA 分布分析 | 直方图 + top-m 入选取值线（把 bin 计数报给用户，别搬 PNG） |
| **T3.2** | 诱饵候选区识别 | 语义无害但可被推到高 MSA 的区域 + 基线 MSA |
| **T4.1–4.3** | 诱饵 + 四格 + 消融 | 四格读数 + 理解度四档 + 任务效用 |

**T0.4 不过 → 立刻停，报告，不要往下走。**
**T1.4 偏离 ±2pp → 先查管线（提示词模板、图像预处理、判官口径），不要往下走。**

---

## 六、结果块格式

```
=== RESULT BLOCK v1 ===
exp:    stage1_table1_noprune
time:   2026-10-21T14:32:07+08:00
env:    torch=2.4.1 transformers=4.44.2 cuda=12.1 model=llava-1.5-7b attn=eager git=abc123
cfg:    pruner=none defense=none seed=0 res=336
n:      1200
asr:    0.5401
dist:   0=0.4100 1=0.0300 2=0.0200 3=0.5400
data:
  0030012300...          ← 逐样本理解度四档，每样本一位
=== END ===
```

- `data` **必须最后**，到 `=== END ===` 之间全是载荷。不要改动这个顺序。
- 四档编码：`0`=拒绝 `1`=误解 `2`=部分合规 `3`=合规
- `data` 长度必须等于 `n`。解析端会校验，截断会被抓住。
- 图的 bin 计数当成普通字段报（如 `hist: 12 34 56 ...`），不要搬 PNG。

**完整 json 留在服务器**，作为复现材料。交付物里写路径 + `env` 行指纹。

---

## 七、不要做的事

| 不要 | 因为 |
|---|---|
| 改 `src/` 或 `scripts/` | 改动回不到 Mac，会静默分叉 |
| 改 `third_party/` | 破坏复现依据 |
| 用 FlashAttention-2 | 静默返回 `None` 注意力，不报错 |
| 一次上三个基准 | 先用 MM-SafetyBench 把管线跑通 |
| 用非确定性判官 | temperature 必须 0 + 固定 seed，否则噪声盖过要测的效应 |
| 不记分辨率就跑 | 视觉 token 数随分辨率变，不同格之间不可比 |
| 建立自己的上行通道 | 违反规定。只走结果块 |

---

## 八、出错时

**报给用户，不要自己想办法绕。**

特别是这三种：

1. **T0.4 拿不到注意力** → 立刻停。这是命题的生死线，需要用户决策换实现还是换命题。
2. **某个 third_party 仓库跑不通** → 停。可能拉错了仓库（TRIM 的 URL 还没确认）。
3. **T1.4 数字对不上** → 停。先和用户一起查管线，别急着调参去凑数字。

**调参凑数字是最严重的问题** —— 它会让复现结果失去意义。

---

*本文件随代码经 git 同步到服务器。改动请在 Mac 上做。*
