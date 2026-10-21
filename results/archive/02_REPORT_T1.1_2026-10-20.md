# 服务器实验报告 · T1.1 轮 · gpu84 · 2026-10-20

> 按 TASKS.md（commit bc3364b）执行。红线自查：仓库代码零修改（`git status` 干净）、TRIM 只查证未填 URL、未猜 split、未碰 GPU 7。

## 结论先行

本轮 TASKS.md 四项任务全部完成：

1. ✅ 收尾（A1/A2）：selftest 4 条 ✓（含新 kind 语义）；lock 文件移出 `env/`，`git status` 干净
2. ✅ MM-SafetyBench 全量下载 + 盘点（B1–B3）：13 类目 × 4 split，**单 split 1680 样本，共 6720**
3. ✅ TRIM 仓库查证：**https://github.com/FreedomIntelligence/TRIM**（细节见下）
4. ✅ 分辨率 → 视觉 token 数实测：**恒为 576**（处理器固定 336×336，输入分辨率不改变）

---

## 一、收尾结果

- `git pull` → bc3364b（fast-forward）
- `result_block.py selftest`：4 条 ✓（往返一致 / 折行容错 / 截断检测 / **kind 语义（smoke 不产出 ASR）**）
- `env/requirements.lock.txt` 已移至 `results/requirements.lock.txt`；`git status` 无输出 ✅
- 上轮遗留的本地 `third_party.lock` 与远端冲突已处理（本地生成物让位远端提交版，服务器侧备份在 `/tmp/third_party.lock.server.bak`）

## 二、MM-SafetyBench 样本数表（T1.1 前提）

来源：`PKU-Alignment/MM-SafetyBench`（hf-mirror 匿名下载，54 文件，482MB，45 秒）
位置：`/home/huangqixu01/data/MM-SafetyBench`（paths.yaml 已更新）
结构：`data/<类目>/{SD,SD_TYPO,TYPO,Text_only}.parquet`，列 = `id / question / image`（image 为 JPEG bytes）
**四个 split 每类目样本数完全相同**：

| 类目 | 样本数/split | | 类目 | 样本数/split |
|---|---|---|---|---|
| EconomicHarm | 122 | | Legal_Opinion | 130 |
| Financial_Advice | 167 | | Malware_Generation | 44 |
| Fraud | 154 | | Physical_Harm | 144 |
| Gov_Decision | 149 | | Political_Lobbying | 153 |
| HateSpeech | 163 | | Privacy_Violence | 139 |
| Health_Consultation | 109 | | Sex | 109 |
| Illegal_Activitiy | 97 | | **合计/split** | **1680** |

**总计：1680 × 4 split = 6720**。（注意：类目名 `Illegal_Activitiy` 是数据集原始拼写，非笔误）

> ⚠️ Mac 侧待决：Token-Pruning 原文用哪个 split。按指示全量下载未做选择。

## 三、TRIM 仓库查证（解 T1.2 卡点）

**结论：https://github.com/FreedomIntelligence/TRIM**（main @ `f2032236583b5d16ca4aec687e1fc2f6b962cdf3`，`git ls-remote` 实测可达）

确认依据：
- 论文《Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs》（COLING 2025，arXiv:2409.10994），**TRIM = Token Reduction using CLIP Metric**，缩写逐字对应
- 基于 **LLaVA-1.5** 做视觉 token 缩减（核心改动在 `llava/model/multimodal_encoder/clip_encoder.py`），与本项目 T0.2 跑通的底座模型完全一致；带保留比例 τ，与 `prune(inputs, ratio) -> keep_indices` 契约吻合
- 排除法：GitHub 全搜 "TRIM token pruning" 仅三个候选——`arielshaulov/TokenTrim`（长视频生成，方向不符）、`hanxunyu/VisionTrim`（ICLR 2026 另一方法，名称是 VisionTrim 非 TRIM）、`FreedomIntelligence/TRIM`（唯一匹配）

保留意见：服务器上没有 Token-Pruning 原文（在 Mac 侧），无法逐字核对引用条目。以上是从"缩写逐字对应 + LLaVA-1.5 底座 + 剪枝契约"三条证据链得出的结论，**建议 Mac 侧翻原文引用复核一次**再定。按 TASKS.md 要求，**未改 `env/fetch_third_party.sh`**，URL 只报回。

## 四、分辨率 → 视觉 token 数（消 T1.4 变量）

用 T0.4 同款定位法（`<image>` token id=32000 在 input_ids 中的展开数）实测：

| 输入分辨率 | input_ids 总长 | 视觉 token 数 |
|---|---|---|
| 224px | 596 | **576** |
| 336px | 596 | **576** |
| 448px | 596 | **576** |
| 512px | 596 | **576** |

**结论：LLaVA-1.5-7B 的 image_processor 固定 `shortest_edge=336` + `crop 336×336`，任何输入分辨率都会被归一到 336，视觉 token 恒为 576（24×24 patch）。** "不同分辨率导致视觉 token 数不可比"的坑在本底座上不存在——但若换 LLaVA-1.6/NeXT 等任意分辨率底座，此结论不成立。

## 五、结果块（粘入 results/inbox/）

**t1_1_mm_safetybench_inventory（kind=utility）**
```
=== RESULT BLOCK v1 ===
exp:    t1_1_mm_safetybench_inventory
time:   2026-10-20T17:38:47+08:00
env:    torch=2.14.1+cu130 cuda=13.0 transformers=5.20.2 py=3.10 host=gpu84 git=bc3364bbe4f7 source=hf-mirror
cfg:    dataset=MM-SafetyBench source=hf-mirror_PKU-Alignment split=all4 categories=13
kind:   utility
n:      4
asr:    n/a
dist:   0=0.0000 1=0.0000 2=0.0000 3=1.0000
n_economicharm: 122
n_financial_advice: 167
n_fraud: 154
n_gov_decision: 149
n_hatespeech: 163
n_health_consultation: 109
n_illegal_activitiy: 97
n_legal_opinion: 130
n_malware_generation: 44
n_physical_harm: 144
n_political_lobbying: 153
n_privacy_violence: 139
n_sex:   109
total_per_split: 1680
total_all_splits: 6720
data:
  3333
=== END ===
```

**t1_1_res_to_visual_tokens（kind=utility）**
```
=== RESULT BLOCK v1 ===
exp:    t1_1_res_to_visual_tokens
time:   2026-10-20T17:38:47+08:00
env:    torch=2.14.1+cu130 cuda=13.0 transformers=5.20.2 py=3.10 host=gpu84 model=/home/huangqixu01/models/llava-1.5-7b-hf git=bc3364bbe4f7
cfg:    model=llava-1.5-7b-hf processor=shortest_edge=336,crop=336
kind:   utility
n:      4
asr:    n/a
dist:   0=0.0000 1=0.0000 2=0.0000 3=1.0000
n_vis_224: 576
n_vis_336: 576
n_vis_448: 576
n_vis_512: 576
note:    processor 固定缩放到 336x336，视觉 token 恒为 576=24x24
data:
  3333
=== END ===
```

## 六、遇到的问题 / 待决

| 问题 | 状态 |
|---|---|
| GitHub 直连间歇性超时（HTTP2 framing / connect timeout） | 已用重试循环绕过；本轮 pull 成功。属出口网络抖动，非凭据问题 |
| 上轮本地 `third_party.lock` 阻塞 pull（untracked 冲突） | 已让位远端提交版；教训：**服务器侧不要往仓库目录写任何非 results/ 的文件**（TASKS A2 已固化为规则） |
| 结果块 extra 字段名只允许小写字母数字下划线；cfg 值不能含 `/`（slug 生成文件名） | 临时脚本已适配；若 Mac 侧要把类目名原样进块，需注意这两条约束 |
| **split 口径未定**（SD / SD_TYPO / TYPO / Text_only） | 等 Mac 侧翻 Token-Pruning 原文确定；已全量就位 |
| **TRIM 残余不确定性** | URL 已查证并报回，建议 Mac 复核原文引用后回填 `fetch_third_party.sh`（回填动作在 Mac 做） |
| T0 报告中 asr 字段 | 系旧版模块缺陷（Mac 已修复加 kind），T1 起的块均带 `kind` |

## 服务器侧留存

- 数据集：`~/data/MM-SafetyBench`（482MB，4 split 全量）
- 结果块：`~/experiment/results/t1_1_mm_safetybench_inventory/`、`results/t1_1_res_to_visual_tokens/`（.block.txt + .json）
- 临时脚本：`/tmp/t1_1_emit_blocks.py`、`/tmp/t1_res_tokens.py`（不进仓库）
- 本报告：`~/experiment/results/REPORT_T1.1_2026-10-20.md`
