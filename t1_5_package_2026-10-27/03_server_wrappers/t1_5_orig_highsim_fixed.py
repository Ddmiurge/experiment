#!/usr/bin/env python3
"""T1.5 步骤3B 兜底启动器（/tmp 临时脚本，不改仓库任何文件）。

原因：experiments/t1_5_2026-10-27/scripts/t1_5_orig_highsim.py 的 patch() 用
    types.MethodType(func, CLIPVisionTower)
把**类**绑成了 self，于是实例调用时 self 仍是类对象，报
    AttributeError: type object 'CLIPVisionTower' has no attribute 'token_reduce_func'
正确做法是把普通函数赋给类属性（描述符协议会自动绑定实例）。

本启动器 import 该模块、只替换它的 patch()，其余逻辑（含 5 条自检、生成、输出路径）完全沿用。
"""
import importlib.util
import sys
import types

SCRIPT = ("/home/huangqixu01/experiment/experiments/t1_5_2026-10-27/scripts/"
          "t1_5_orig_highsim.py")

spec = importlib.util.spec_from_file_location("t1_5_orig_highsim", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)          # __name__ != "__main__"，不会跑 main()


def patch_fixed():
    sys.path.insert(0, mod.TRIM_REPO)
    from llava.model.multimodal_encoder import clip_encoder
    cls = clip_encoder.CLIPVisionTower
    cls.token_reduction_official = cls.token_reduction
    cls.token_reduction = mod.token_reduction_highsim   # ← 关键：普通函数，走描述符绑定
    print("[patch-fixed] CLIPVisionTower.token_reduction → highsim 版"
          "（官方原函数存为 *_official）", flush=True)


mod.patch = patch_fixed
mod.main()
