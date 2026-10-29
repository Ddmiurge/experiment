#!/usr/bin/env python3
"""T1.6 冻结修订#2 · oracle 载荷集的**模型几何 cell 级**构造（决策记录见协议文末）

为什么不是 A（行级重算）也不是沿用探针归档行（=C 的前身）：
  · 探针归档 payload_rows 活在裸 CLIP 路径的几何里（条带满高 24 行、字形在 11–16），
    原版栈把条带压到 ~4.2 行、贴顶（实测 IoU 0.028、92.6% 取到空白）——
    跨几何搬运行号必错，服务器 step2 冒烟 3/3 拦住，正确。
  · 行粒度在原版栈几何里语义崩坏：条带 5/7/9/11 行高矮不一，top-6 行
    在矮样本混入带外白行、在高样本只装得下 6/11 行 ⇒ 载荷份额逐样本漂 45–55%，
    且"6×24=144 恰好等于预算"在模型几何不成立（399/399 触发溢出或补足）。
  · cell 级把这些问题全消掉：预算天然=144，选择直接按"每格笔画活动度"，
    语义 = 保留最像字形墨迹的 144 格 = "载荷在场"的最细可用口径。

冻结口径（生成时就地算，杜绝一切几何漂移；审计件由 oracle_masks.py 用**同一函数**产出）：
  1) act(s,i)   = 模型消费的 px（原版栈 process_images 后的 [1,3,336,336]）第 i 个
                  14×14×3 cell 跨通道的 std                          —— 与探针 cell 同法
  2) ink(s)     = {i: act_i > median + 6·MAD}；退化时( |ink|<8 或 >200 )取 act 前 12%
                  （与探针 cell 级 MAD 检测同参数，Mac 合成图 precision/recall=1.00）
  3) ④ O-载荷   = |ink|≥144: ink 内按 act 降序取 144（stable）;
                  |ink|<144: 全部 ink + 非 ink 中均匀随机补足到 144（default_rng(42)）
     ③ O-留白   = 非 ink 中均匀随机 144（同 rng）⇒ ③ 含 ink 恒 0（构造保证，留核验）
  4) 进 LLM：keep(144，格序升序=原空间序) + 被丢 432 均值聚合 token = 145 —— 与全部历史条件同形

随机性说明：rng 每样本 default_rng(42) 重建（与既有 random 臂同实现）⇒ 补足/③ 的选取
在同尺寸样本间可复现；(cat,qid) 顺序固定 ⇒ 全程确定。

本模块被两处 import：t1_6_oracle_prune.py（运行时构建，权威）与 oracle_masks.py
（预跑审计 + 落盘 masks 供冒烟比对，同一 process_images 路径）。
"""
import numpy as np

NROW = NCOL = 24
BUDGET = 144
SEED = 42


def cell_activity(px) -> np.ndarray:
    """px: 模型实际消费的张量 [1,3,H,W]（336）。返回 [576] 每格活动度。"""
    x = np.asarray(px[0], dtype=np.float64)                    # [3,H,W]
    x = x.transpose(1, 2, 0)                                   # [H,W,3]
    h = x.shape[0] // NROW                                     # 14
    x = x.reshape(NROW, h, NCOL, h, x.shape[2])
    x = x.transpose(0, 2, 1, 3, 4).reshape(NROW * NCOL, -1)
    return x.std(axis=1)


def ink_mask(act: np.ndarray):
    """MAD 阈值；返回 (ink_idx sorted, mode)。"""
    med = np.median(act)
    mad = np.median(np.abs(act - med))
    mask = act > med + 6.0 * mad
    n = int(mask.sum())
    mode = "mad"
    if n < 8 or n > 200:
        k = int(round(0.12 * len(act)))
        mask = np.zeros(len(act), dtype=bool)
        mask[np.argsort(-act, kind="stable")[:k]] = True
        mode = "fallback_top12pct"
    return np.flatnonzero(mask), mode


def build_keep(act: np.ndarray, which: str):
    """which ∈ {payload, blank}。返回 (keep sorted list, ink_count, note)。"""
    ink, mode = ink_mask(act)
    pool = np.setdiff1d(np.arange(len(act)), ink)
    note = mode
    if which == "payload":
        if len(ink) >= BUDGET:
            order = ink[np.argsort(-act[ink], kind="stable")]
            keep = order[:BUDGET]
            note += f";trim{len(ink) - BUDGET}"
        else:
            rng = np.random.default_rng(SEED)
            extra = rng.choice(pool, BUDGET - len(ink), replace=False)
            keep = np.concatenate([ink, extra])
            note += f";pad{BUDGET - len(ink)}"
    elif which == "blank":
        assert len(pool) >= BUDGET
        keep = np.random.default_rng(SEED).choice(pool, BUDGET, replace=False)
    else:
        raise ValueError(which)
    return sorted(int(k) for k in keep), int(len(ink)), note


def audit_row(keep, ink_count):
    """keeps 文件里随行落盘的核验量。"""
    rows = [k // NCOL for k in keep]
    return {"keep_n": len(keep), "ink_n": ink_count,
            "kept_rows": sorted(set(rows)), "rows_spread": max(rows) - min(rows) + 1}
