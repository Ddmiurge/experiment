#!/usr/bin/env python3
"""T1.5b highsim/random 兜底启动器（/tmp 临时脚本，不改仓库）。

仓库版 t1_5_orig_prune.py 的 KEEP_LOG 留档行有 bug：
    token_mask[0].nonzero(as_tuple=True)[1]   # token_mask[0] 是 1-D → 返回 1 元组
→ IndexError: tuple index out of range（highsim / random 两条路径必然崩溃；
   none / official 不写 KEEP_LOG 所以不受影响）。

本启动器 import 仓库脚本，运行时用字符串替换修正该行、在模块命名空间里重定义函数，
其余逻辑（自检断言、生成、输出路径、--prune 分支）完全沿用。
"""
import importlib.util
import inspect
import sys

SCRIPT = ("/home/huangqixu01/experiment/experiments/t1_5_2026-10-27/scripts/"
          "t1_5_orig_prune.py")

spec = importlib.util.spec_from_file_location("t1_5_orig_prune", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["t1_5_orig_prune"] = mod
spec.loader.exec_module(mod)

src = inspect.getsource(mod.token_reduction_mode)
bad = "token_mask[0].nonzero(as_tuple=True)[1].tolist()"
good = "token_mask[0].nonzero(as_tuple=False).flatten().tolist()"
assert bad in src, "没找到待修正的行，仓库脚本可能已变"
src = src.replace(bad, good)
exec(src, mod.__dict__)
print("[patch-fixed] KEEP_LOG 留档行已修正（1-D nonzero 索引）", flush=True)

mod.main()
