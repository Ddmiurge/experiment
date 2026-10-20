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
├── Agent.md              ← 服务器侧 Agent 的操作手册
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

## 当前状态（2026-10-20）

仓库：`git@github.com:Ddmiurge/experiment.git`

| 项 | 状态 |
|---|---|
| 目录骨架 | ✅ |
| 结果块模块 | ✅ 往返自检通过；含截断检测 + kind 语义 |
| git 仓库 | ✅ 已推送 |
| 出网边界 | ✅ 实测通过——GitHub pull 可用，主方案成立 |
| `third_party` | ✅ 已锁 SHA（`third_party.lock`）；⚠️ `TRIM` 仍是 TODO |
| 环境版本 | ✅ 已锁（`env/requirements.txt`，服务器实测值） |
| **T0.1–T0.4** | ✅ **4/4 通过**，含一票否决项 T0.4 |
| T1.1 数据集 | ❌ **未下载**——当前卡点 |
| T1.2 剪枝器 | ❌ 等 TRIM URL 确认 |
| `src/` 实际代码 | ⬜ 待 T1 开始 |

**T0.4 结果**：32 层 `[1, 32, 596, 596]`，视觉 span `[5, 581)` 共 576 个 token，
行和偏离 3.35e-04，峰值/均匀最高 32.73。**命题的生死线过了。**
后续统一用路径 B（forward hook）。

**下一步的两个卡点**：① MM-SafetyBench 等数据集一个都没下；
② TRIM 仓库 URL 未确认。这两件事不解决，T1 开不了工。

服务器报告原文：`results/REPORT_T0_2026-10-20.md`。

---

## 服务器侧命令

```bash
git clone https://github.com/Ddmiurge/experiment.git
cd experiment
cp env/paths.example.yaml env/paths.yaml   # 按实际路径改
bash env/fetch_third_party.sh              # 按 third_party.lock 检出
python scripts/result_block.py selftest
```
