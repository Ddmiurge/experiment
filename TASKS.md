# TASKS · 当前轮

> **上一轮**：T0 阶段 4/4 通过，含一票否决项 T0.4。报告见 `results/REPORT_T0_2026-10-20.md`。
> **本轮**：解掉 T1 的两个卡点（数据集 / TRIM）。
> 读完这份再动手。常驻规则仍以 `Agent.md` 为准。

---

## 一、先做：收尾（5 分钟）

### A1 · 拉新代码

```bash
git pull
python scripts/result_block.py selftest
```

**验收**：自检输出 4 条 ✓，最后一条是「kind 语义（smoke 不产出 ASR）」。
上一轮 T0.2 / T0.3 报的 `asr: 1.0000` 是模块缺陷，已在 Mac 侧修复。

### A2 · 把 lock 文件挪出仓库目录

上一轮你把 `env/requirements.lock.txt` 写进了仓库目录。`env/` 下只有 `paths.yaml` 被
gitignore，所以这个文件会让 `git status` 变脏，而且 Mac 侧将来提交同名文件后
`git pull` 会直接失败（untracked working tree file would be overwritten）。

```bash
mv env/requirements.lock.txt results/          # 或 /tmp/，但要保留在 results/ 更好查
git status                                     # 验收：干净
```

**验收**：`git status` 无输出。

> `env/paths.yaml` 是 gitignored 的，**那个可以随便改**，不算违反规则 1。

---

## 二、拉 MM-SafetyBench（T1.1）

镜像上确认可用：`PKU-Alignment/MM-SafetyBench`（PKU 官方账号）。

### B1 · 下载

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 若 hf 命令不存在，先装：
#   pip install -U "huggingface_hub[cli]"

hf download PKU-Alignment/MM-SafetyBench --repo-type dataset \
    --local-dir /home/huangqixu01/data/MM-SafetyBench
```

老版本 CLI 用 `huggingface-cli download`，参数相同。

**四个 split 全下**（`SD` / `SD_TYPO` / `TYPO` / `Text_only`），全量约 0.5GB。
Mac 侧还在确认 Token-Pruning 原文用的是哪个 split —— **先全下，等口径定了再选**，
别猜。

### B2 · 盘点

```bash
python - <<'EOF'
import glob, pandas as pd
tot = 0
for f in sorted(glob.glob("/home/huangqixu01/data/MM-SafetyBench/data/*/*.parquet")):
    n = len(pd.read_parquet(f, columns=["id"]))
    tot += n
    print(f"{f.split('MM-SafetyBench/')[-1]:45} {n:5}")
print(f"{'合计':45} {tot:5}")
EOF
```

需要 `pandas` + `pyarrow`。装不上就退回用文件大小 + 数据集 README 里声明的
`num_examples`。

### B3 · 更新 paths.yaml

把 `data.mm_safetybench` 改成实际路径。`paths.yaml` 不进 git，改它不影响复现。

**回传**：13 类目 × 4 split 的样本数表。这是 T1.4 能不能对齐原文的前提。

---

## 三、查 TRIM 的仓库 URL（解 T1.2 卡点）

`env/fetch_third_party.sh` 里 `trim` 那行还是 `TODO`。T1.2 要接剪枝器，没这个接不上。

**这是查证任务，不是写代码任务。**

1. 找 Token-Pruning 原文里 TRIM 的引用，确认是哪个仓库
2. 用 `git ls-remote <URL> main` 验证可达（GitHub 出网已确认可用）
3. **把 URL 报回来，不要改脚本**（规则 1）

**回传**：仓库 URL + 一句话说明你怎么确认它是原文用的那个。

> 如果查不到确切的：**报"查不到"，不要随便挑一个近似的仓库填上。**
> 拉错剪枝器会让 T1.4 的数字毫无意义，而且很难发现。

---

## 四、顺带：分辨率 → 视觉 token 数

避坑清单第 4 条：视觉 token 数随输入分辨率变化，不记录则不同格之间不可比。

用已经跑通的注意力 hook，把 LLaVA-1.5-7B 在几个分辨率下的视觉 token 数测出来
（至少 336 和 448），记进结果块或报告。

**这个很便宜，但能消掉 T1.4 的一个变量。**

---

## 五、不要做

| 不要 | 因为 |
|---|---|
| 开始 T1.2 / T1.3 / T1.4 | 数据集 split 口径未定 + TRIM 未定 |
| 改 `src/` 或 `scripts/` | 规则 1。包括填 TRIM 的 URL —— 报回来 |
| 猜 MM-SafetyBench 用哪个 split | 复现口径不能猜，猜错了 T1.4 全废 |
| 挑一个"看起来像"的 TRIM 仓库 | 同上 |
| 碰 GPU 7 | 常驻 Qwen，不可动 |

GPU 5 / 6 仍全空，本轮任务基本不占卡。

---

## 六、回传

写一份 `results/REPORT_T1.1_<日期>.md`，格式参照上一轮的 T0 报告：

1. 结论先行（本轮几项完成）
2. 收尾结果（selftest 输出、`git status` 是否干净）
3. MM-SafetyBench 样本数表
4. TRIM 查证结果（URL 或"查不到"）
5. 分辨率 → 视觉 token 数
6. 遇到的问题 / 待决

有数字就用结果块（`kind=smoke` 或 `kind=utility`，**不是 asr**）。
纯查证结论写在报告里就行。
