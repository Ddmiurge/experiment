# T1.5 实验包 · 剪枝方向（H1）与权重（H2）· 2026-10-27

对照 TASKS：`01_reports/TASKS_T1.5.md`（commit d1c2d91）
诊断原文：`01_reports/DIAGNOSIS_2026-10-27.md`

---

## 一、结论

**探针证实"官方 TRIM 的保留方向与原文规范相反"，但把方向改对后，效应依然不存在。**

按 TASKS §二 判定表落在**第三行**：

| 判据 | 实测 |
|---|---|
| 探针 `sim_diff` | −8.96（SD，399 条）/ −9.30（TYPO，1680 条），逐样本 100% 为负 ✅ |
| `trim_highsim − none`（vanilla） | **−1.00pp**（净 −4，p=0.636） |
| `trim_highsim − none`（TRIM-FT） | **+0.25pp**（净 +1，p=1.000） |
| `random − none` | **+1.01pp**（净 +4，p=0.597） |

→ **剪什么不重要 → H1 解释力下降，回 H3（总体/判官）+ H4（样本量）**

---

## 二、目录结构

```
01_reports/
  ├── 10_REPORT_T1.5_2026-10-26.md          ← 本轮主报告（结论、探针、权重、四格表、分类目）
  ├── 10_REPORT_judge_stability_2026-10-26.md ← 同日判官可重复性（按 TASKS §三 降级保留）
  ├── DIAGNOSIS_2026-10-27.md               ← Mac 侧诊断原件（H1 出处与原文行号）
  └── TASKS_T1.5.md                         ← 本轮指令快照

02_mac_scripts/           Mac 侧提供的 5 个脚本（原样，服务器未改动）
  ├── t1_5_trim_probe.py        探针（不生成）
  ├── t1_5_weights_diff.py      权重/配置比对
  ├── t1_5_generate.py          vanilla 栈生成（none / trim_official / trim_highsim / random）
  ├── t1_5_orig_highsim.py      TRIM-FT 栈 + 高相似度剪枝（**有 bug，见第四节**）
  └── t1_5_judge.py             判分 + 配对 + 分层（**有 bug，见第四节**）

03_server_wrappers/       服务器侧在 /tmp 写的绕过/复跑脚本（未改仓库文件）
  ├── t1_5_orig_highsim_fixed.py   修 patch() 绑定后调用原脚本 main()
  ├── t1_5_final_judge.py          行前缀 tag + 合法 cfg，重跑判分并 emit 结果块
  └── t1_5_h2_weights.py           改名映射后的张量级比对（H2）

04_answers/               本轮新生成的回答（各 399 条，同批 id）
  ├── t1_5_answers_trim_highsim_sdfull.json     vanilla + trim_highsim（0.5689）
  ├── t1_5_answers_random_sdfull.json           vanilla + random（0.5890）
  ├── t1_5_answers_orig_sd_highsim.json         TRIM-FT + trim_highsim（0.5865）
  └── t1_5_keeps_*.jsonl                        每样本实际保留的 token 索引

05_probe/                 探针逐样本明细（sim_diff、保留集、条带覆盖率）
  ├── t1_5_probe_sd.jsonl       399 条
  └── t1_5_probe_typo.jsonl     1680 条

06_judge_raw/             判官原始输出（含 raw_output 与 n_tokens），7 个格子
07_result_blocks/         7 块 kind=asr + 1 块 kind=smoke
```

## 三、四格矩阵（SD，399 条，同一批 id）

| 权重 | none | trim_official（低相似度） | trim_highsim（高相似度） | random |
|---|---|---|---|---|
| vanilla | 0.5789（231） | 0.5739（229） | **0.5689（227）** | 0.5890（235） |
| TRIM-FT | 0.5840（233） | 0.5664（226） | **0.5865（234）** | — |

配对（vs none）：vanilla highsim b=18/c=22 净 −4 p=0.636；TRIM-FT highsim b=20/c=19 净 +1 p=1.000；
random b=18/c=14 净 +4 p=0.597。解析失败 0，判官输出均值 2.0 tokens。

Δ 拆分：
```
Δ_checkpoint        = TRIM-FT|none − vanilla|none      = +0.51pp
Δ_pruning(官方符号)  = TRIM-FT|trim − TRIM-FT|none      = −1.76pp
Δ_pruning(原文规范)  = TRIM-FT|highsim − TRIM-FT|none   = +0.25pp
```

## 四、两个脚本 bug（需 Mac 修）

1. **`t1_5_orig_highsim.py`**
   ```python
   clip_encoder.CLIPVisionTower.token_reduction = \
       types.MethodType(token_reduction_highsim, clip_encoder.CLIPVisionTower)
   ```
   把**类**绑成了 self → 运行必报
   `AttributeError: type object 'CLIPVisionTower' has no attribute 'token_reduce_func'`。
   修法：直接 `CLIPVisionTower.token_reduction = token_reduction_highsim`（普通函数走描述符绑定）。

2. **`t1_5_judge.py`**
   - raw jsonl 用 `tag` 命名，两行实验 tag 同名 → 第二行**覆盖**第一行的判定记录（第一行数据丢失）。
     修法：文件名加行前缀。
   - `cfg` 里 `prune_sign=n/a` 含 `/` → `result_block.emit` 生成非法路径，`FileNotFoundError` 崩溃，
     结果块写不出。修法：用 `na` / `low_sim` / `high_sim` / `random`。

## 五、探针五读数（原样，详见主报告 §二）

```
SD (n=399):   sim_diff=-8.9624  frac<0=1.000  J(off,high)=0.0000  J(off,rand)=0.1432
              text-sens: empty=0.2487 generic=0.2446 swap=0.4665
TYPO(n=1680): sim_diff=-9.3049  frac<0=1.000  J(off,high)=0.0000  J(off,rand)=0.1429
              text-sens: empty=0.3917 generic=0.3695 swap=0.7094
              TYPO 底部关键词条带保留占比 = 0.1658
```

注：① 随机 Jaccard 理论值是 **0.1429**（脚本注释写 0.25 为笔误）；
② TYPO 探针实际跑了**全量 1680**（`--ids-from` 传 JSON 数组时脚本解析失败后回退无过滤），不影响结论。

## 六、待决

1. TASKS 判定表要求"回 H3+H4，跑 SD+TYPO 合并总体全量"——是否执行、跑哪种范围
2. 严格类目 137 条（42%）全部饱和 1.0，可观测空间只剩通用类目 262 条；提功效前需先解决饱和
3. 判官（Qwen3-8B）对措辞敏感但确定（同日 A/B 100%），交付物需声明
