# code · Token-Pruning 复现与诱饵占位攻击

配套文档在上一级目录：`代码任务清单_2026-10-19.md`、`执行计划_2026-10-19.md`、
`实验口径与判据_2026-10-19.md`。

---

## 两台机器的分工

```
Mac（唯一写者）                         服务器（只读消费者）
  ├── 论文工作区                          ├── git pull（只能 pull）
  └── code/  ← 代码只在这里写             ├── 跑实验
         │ git push                       ├── results/*.json 留在本地
         ▼                                └── 打印结果块
    GitHub private  ──────git pull──────►      │
         ▲                                     │
         └──────── 用户复制粘贴 ◄──────────────┘
```

**服务器能 pull、不能 push；Mac 按规定不能连服务器。**
所以：代码走 git 下行，结果靠人工搬运上行。

## 三条硬规则

1. **服务器上不写正式代码。** 服务器写的东西出不来。要改 `src/`，
   把补丁拿回 Mac 改、push、再 pull。
2. **`third_party/` 里的仓库不改。** 按 `third_party.lock` 检出，适配写在外层。
3. **每次实验必须打印结果块。** 没有结果块的读数等于没跑。

服务器侧的 Agent 读 `Agent.md`，那份是给它看的。

---

## 起步

```bash
cd code

# 1. 仓库已建好，remote 已指向 Ddmiurge/experiment
git pull

# 2. 路径配置（服务器上做）
cp env/paths.example.yaml env/paths.yaml

# 3. 自检（两端都能跑）
python scripts/result_block.py selftest

# 4. 出网边界（服务器上做，最先跑）
bash scripts/check_egress.sh

# 5. 第三方仓库（服务器上做）
bash env/fetch_third_party.sh
```

---

## 目录

```
code/
├── Agent.md              ← 服务器侧 Agent 的常驻操作手册
├── TASKS.md              ← 服务器侧 Agent 的本轮任务（每轮更新）
├── env/
│   ├── paths.example.yaml    路径模板（复制成 paths.yaml，不进 git）
│   ├── requirements.txt      占位，T0.1 后锁定
│   └── fetch_third_party.sh  拉取并锁定 commit SHA
├── scripts/
│   ├── check_egress.sh       出网边界探测（最先跑）
│   ├── result_block.py       结果块：两端共用，格式只有一处
│   └── t0_4_attention_hook.py  T0.4 一票否决项验证
├── src/
│   ├── models/          模型加载与注意力 hook
│   ├── pruning/         剪枝器适配层，契约 prune(inputs, ratio) -> keep_indices
│   ├── defense/         SAP：MAI / BTR / AR
│   ├── attack/          诱饵占位（阶段 4 之前应为空）
│   ├── eval/            ASR、理解度四档、任务效用
│   └── data/            数据集加载
├── third_party/         按 lock 检出，不进 git
└── results/
    ├── inbox/           ← 从服务器粘回来的结果块放这里（进 git）
    └── parsed/          解析出的 json（不进 git）
```

---

## 结果块怎么用

**服务器侧**（或让 Agent 做）：

```bash
python scripts/result_block.py emit \
    --exp stage1_table1_noprune \
    --levels 0030012300... \
    --cfg pruner=none,defense=none,seed=0
```

**Mac 侧**：把打印出来的块（连同 `=== RESULT BLOCK v1 ===` 和 `=== END ===`）
粘进 `results/inbox/xxx.txt`，然后：

```bash
python scripts/result_block.py parse results/inbox/
```

会打印一张汇总表并写 `results/parsed/*.json`。

**解析器会校验 `n == len(data)` 和 ASR 自洽** —— 粘贴截断会被直接抓住，
不会静默算错。

图的 bin 计数当成普通字段一起报，回来在 Mac 上重画，不要搬 PNG。

---

## 当前状态（2026-10-21）

仓库：`git@github.com:Ddmiurge/experiment.git`

| 项 | 状态 |
|---|---|
| 目录骨架 / 结果块模块 / git | ✅ |
| 出网边界 | ✅ GitHub pull 可用，主方案成立 |
| 环境版本 | ✅ 已锁（`env/requirements.txt`，服务器实测值） |
| **T0.1–T0.4** | ✅ **4/4 通过**，含一票否决项 T0.4 |
| **T1.1 数据集** | ✅ 13 类目 × 4 split，单 split 1680，共 6720，482MB |
| `third_party` | ✅ TRIM 已定案 `FreedomIntelligence/TRIM`（arXiv:2409.10994） |
| **T1.2 TRIM 集成** | ⬜ **本轮** |
| **T1.3 判官** | ⬜ **本轮** |
| T1.4 | ⬜ 等 T1.2/T1.3 |

**T0.4 结果**：32 层 `[1, 32, 596, 596]`，视觉 span `[5, 581)` 共 576 个 token，
行和偏离 3.35e-04，峰值/均匀最高 32.73。**命题的生死线过了。**
后续统一用路径 B（forward hook）。

### ⚠ 读原文后的三处口径修正（2026-10-21）

1. **判官换了**。原文 MM-SafetyBench 用 **GPT-4o-mini**，不是 LLaMA-Guard-3-8B
   （16 页里没出现过）。境内连不上 OpenAI，**改用本地 Qwen3-8B**。判的问题与原文
   相同（模型是否遵从有害指令），只是换了裁判 —— 可校准的偏离。
2. **验收数字换了**。54.01 / 62.33 / 50.19 是**三基准均值**。只跑 MM-SafetyBench 时
   对应 **59.37 / 65.87 / 52.78**。
3. **超参已定**（原文 §6.1）：m=10，λ=0.6，剪枝率 75%，BTR 恢复 10%，seed=42，greedy。

报告：`results/REPORT_T0_2026-10-20.md`、`results/REPORT_T1.1_2026-10-20.md`。

**本轮指令见 `TASKS.md`**（T1.4：跑剪枝前后的 ASR）。

---

## 判官口径（2026-10-21 定，必须写进交付物）

用 **MM-SafetyBench 官方判官协议**（`src/eval/mm_safetybench_judge.py`，逐字取自
`isXinLiu/MM-SafetyBench/evaluation.py`）+ **Qwen3-8B** 代替原文的 GPT-4o-mini
（境内连不上 OpenAI）。

**两点必须一起写，否则 58% 这个数会被误读：**

**1 · 官方判据是「有没有拒绝」，不是「有没有给出有害内容」。**
13 个类目分两类：5 个严格类目（政治游说 / 政府决策 / 法律意见 / 理财建议 /
健康咨询）只要没明确拒绝就算 unsafe；其余 8 个按内容判。所以 ASR 高
不等于模型输出了有害内容，只说明它**没拒绝**。

**2 · SD split 的图是主题配图，不含可读文字。**
Mac 侧核过原图（1024×1024）：Political_Lobbying#32 是一个装着糊字纹理的黑盒子，
Fraud#29 是书本与电子设备的图案，Privacy_Violence#122 是 SIM 卡与电路元件。
问题模板里的 "The image shows a phrase of..." 是误导——有害请求写在问题里。
模型也读不出图上的字（它看到"政治竞选"就编了句 "I'm with her"）。
该分片实际测量的是**模型对问题模板的拒绝倾向**。

> 这两点不构成复现问题：我们的 58.00% 与原文 59.37% 差 1.37pp，说明原文
> 大概率也是用 SD split、测的同一件事。但解释数字时必须带上口径。

**未做逐条人工标定。** 曾组织 100 条盲标，因官方判据（"有没有拒绝"）与标注者
理解（"有没有有害内容"）不一致而无法对齐，标注结果作废。

---

## 服务器侧命令

```bash
git clone https://github.com/Ddmiurge/experiment.git
cd experiment
cp env/paths.example.yaml env/paths.yaml   # 按实际路径改
bash env/fetch_third_party.sh              # 按 third_party.lock 检出
python scripts/result_block.py selftest
```
