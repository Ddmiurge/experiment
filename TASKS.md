# TASKS · 当前轮 T1.8a-Day2：FasterVLM 三格（A2/C2/B0 同会话判分）

> 放行单：`archive/t1_8a_probe_package_2026-11-03/01_reports/MAC_review_Day1_release_2026-11-04.md`
> （先读它：SD 两格已放弃、C2 必跑、判分同会话）。口径仍是冻结协议+补丁#1/#2。
> 三件事：①Day-1 待办的掩码指认与改名（CPU 2 分钟）；②版本-论文对应核对材料随包回（文献活由 Mac 做，
> 你们只回贴 README/paper 引用线索）；③生成 C2 + 同会话判分。
> 老规矩：不改 third_party/、不改 src/；适配器 fp16 契约务必遵守。

## 步骤
1. `t1_8a_profile.py` 修正参照集（TYPO 行→T1.6 masks，SD 行→T1.7 masks），重出 jac4/jac3 两列并改名
   `04_masks_typo/` 内文件为 `t1_8a_refmasks_typo_*`；cover4 不动。
2. C2 生成：FasterVLM 适配器名单 → t1_7_oracle_prune 外部掩码通道（144+聚合=145，逐样本 assert
   keep==适配器输出）；A2 生成：仓库原生管线（144 纯删）。两卡并行 ~40 分钟/格。
3. 同会话判分：B0 + A2 + C2（+既有 TYPO 参照格不重判）；LG3 重判可选；预注册判读只看 C2。
4. 回传 `t1_8a_package_<日期>/`：19_REPORT（§4 落点、V1−V2 差、jac 修正表）+ 全套 raw。

## 停线
C2 keep 名单与适配器逐样本不一致 → 停；其余沿用协议 §5。
