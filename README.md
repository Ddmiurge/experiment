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

# 1. 建仓库（三选一，还没做）
git init && git add -A && git commit -m "骨架"
git remote add origin <你的仓库URL>      # GitHub private 或 Gitee
git push -u origin main

# 2. 路径配置（服务器上做）
cp env/paths.example.yaml env/paths.yaml

# 3. 自检（两端都能跑）
python scripts/result_block.py selftest

# 4. 出网边界（服务器上做，最先跑）
bash scripts/check_egress.sh <仓库URL>

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

| 项 | 状态 |
|---|---|
| 目录骨架 | ✅ |
| 结果块模块 | ✅ 往返自检通过 |
| 出网探测脚本 | ✅ 待服务器上实跑 |
| T0.4 验证脚本 | ✅ 待服务器上实跑（**一票否决项**） |
| `third_party` URL | ⚠️ 未逐个核验，`TRIM` 仍是 TODO |
| git 仓库 | ⚠️ 尚未 `git init` |
| `src/` 实际代码 | ⬜ T0.2 之后开始 |

**下一步**：在服务器上跑 `check_egress.sh` 和 `t0_4_attention_hook.py`。
后者不过，整个计划停摆。
