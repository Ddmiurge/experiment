#!/usr/bin/env python3
"""T1.6 · oracle 保留集构造（修订#3 · 终版：**无阈值、无随机、模型几何内**）

为什么又改一次（诚实记录）：修订#2 的 ink=MAD 阈值在真实 TYPO 图上必然退化——
白底占 ~80% 格、活动度≈0（归一化后常数块），零膨胀分布下 median+6·MAD≈0，
ink 集合爆量→落 fallback→"上限怎么办"变成一个待决问题。**干预的定义里不该有任何
待决旋钮**，故终版改为纯秩定义：

    ④ O-载荷 = 576 格按模型 px 的逐格活动度降序取前 144（并列按格序，稳定排序）
    ③ O-留白 = 同秩升序取后 144 —— 与④严格互斥、纯白优先，是 highsim"留最白"的端点版
    两格无阈值、无 rng、无 pad/trim 分支；聚合 token 与 145 规则不变。

ink/MAD/pad/trim/fallback 全部废除为**干预输入**；墨迹格数只作为审计诊断量
（Otsu 自动阈值，独立于干预，不进任何 keep 集合）。审计口径：
    ink_n(④覆盖核验)= |④∩otsu墨迹| / |otsu墨迹|   设计=1（墨迹≤144 时）
    ③泄漏 = |③∩otsu墨迹|                          设计=0
停线条件（全部写死在 TASKS step0，不留现场判断）：
    ③泄漏>0 任意样本 ⇒ 构造 bug；或 中位覆盖率 <0.9 ⇒ "④≉保载荷"前提破了，停报。

本模块是**唯一实现处**：生成器运行时调用，oracle_masks.py 审计件调用同一函数逐条比对。
"""
import numpy as np

NROW = NCOL = 24
BUDGET = 144


def cell_activity(px) -> np.ndarray:
    """px: 模型实际消费的张量 [1,3,H,W]（336）。返回 [576] 每格跨通道 std。"""
    x = np.asarray(px[0], dtype=np.float64).transpose(1, 2, 0)     # [H,W,3]
    h = x.shape[0] // NROW
    x = x.reshape(NROW, h, NCOL, h, x.shape[2]).transpose(0, 2, 1, 3, 4)
    return x.reshape(NROW * NCOL, -1).std(axis=1)


def _order(act):
    """稳定秩：活动度降序（并列按格序升序）⇒ ④ 取前 BUDGET、③ 取后 BUDGET。"""
    return np.lexsort((np.arange(len(act)), -act))


def build_keeps(act):
    """返回 (keep_payload, keep_blank, audit dict)。纯秩构造，无任何可调参数。"""
    order = _order(act)
    kp = sorted(int(i) for i in order[:BUDGET])
    kb = sorted(int(i) for i in order[-BUDGET:])
    t = otsu_threshold(act)
    ink = np.flatnonzero(act > t)
    inkset = set(int(i) for i in ink)
    audit = {
        "otsu_t": float(t), "ink_n": int(len(ink)),
        "cov4": (len(inkset & set(kp)) / len(inkset)) if len(ink) else float("nan"),
        "leak3": int(len(inkset & set(kb))),
    }
    return kp, kb, audit


def otsu_threshold(v, bins=64):
    """Otsu 仅用于审计墨迹计数（不进干预）。零膨胀下退化则返回 +inf（ink_n=0，审计会标）。"""
    v = np.asarray(v, dtype=np.float64)
    if v.max() - v.min() < 1e-9:
        return float("inf")
    hist, edges = np.histogram(v, bins=bins)
    p = hist / hist.sum()
    mid = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); mu0 = np.cumsum(p * mid); mu = (p * mid).sum()
    w1 = 1 - w0; mu1 = mu - mu0
    den = (w0 * w1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu0 * w1 - mu1 * w0) ** 2 / den
    sb[den == 0] = 0.0
    return float(mid[int(np.argmax(sb))])


# ---- 兼容旧接口名（生成器 import build_keep；一次调用返回一格的 keep）----
def build_keep(act, which):
    kp, kb, au = build_keeps(act)
    keep = kp if which == "payload" else kb
    note = f"rank;ink{au['ink_n']};cov4{au['cov4']:.2f}" if which == "payload" \
        else f"rank;ink{au['ink_n']};leak3{au['leak3']}"
    return keep, au["ink_n"], note
