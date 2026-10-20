#!/usr/bin/env python3
"""T0.4 · 注意力 hook 验证 —— 唯一的"一票否决"任务。

在写任何 SAP 代码或攻击代码之前必须通过。理由：SAP 的 MAI 与诱饵判据
都要读 post-softmax 跨模态注意力，拿不到则整条路走不通。

通过判据（三条全过）：
  1. 能拿到 [layers, heads, seq, seq] 的注意力张量
  2. 每行（softmax 后）和为 1
  3. **视觉 token 那几列不是全零、不是近似均匀分布**

第 3 条是这份脚本存在的主要理由。形状对、行和为 1 的**全零列**同样能通过
第 1、2 条 —— 这是最可能出现的假阳性，必须单独查。

用法：
    python scripts/t0_4_attention_hook.py --model /path/to/llava-v1.5-7b
    python scripts/t0_4_attention_hook.py --model ... --image /path/to/x.jpg

不传 --image 时用一张合成的灰噪声图，不依赖外部文件。
"""

from __future__ import annotations

import argparse
import sys

import torch


# ── 探针 ─────────────────────────────────────────────────────
PROMPT = "USER: <image>\nDescribe this image in one sentence.\nASSISTANT:"


def make_image(size: int = 336):
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(20261020)
    arr = rng.integers(0, 256, (size, size, 3), dtype="uint8")
    return Image.fromarray(arr, "RGB")


def find_image_token_id(processor) -> int | None:
    tok = processor.tokenizer
    for name in ("<image>", "<image_token>", "<|image|>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and i != tok.unk_token_id:
            return i
    return None


def visual_span(input_ids: torch.Tensor, img_id: int) -> tuple[int, int] | None:
    pos = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return None
    # 视觉 token 通常是连续一段；取最长连续段
    gaps = (pos[1:] - pos[:-1]).tolist() if pos.numel() > 1 else [1]
    start = 0
    best = (0, 0, 1)
    for i, g in enumerate(gaps):
        if g != 1:
            if i + 1 - start > best[2]:
                best = (start, i + 1, i + 1 - start)
            start = i + 1
    if pos.numel() - start > best[2]:
        best = (start, pos.numel(), pos.numel() - start)
    return int(pos[best[0]]), int(pos[best[1] - 1]) + 1


# ── 两条抓注意力的路 ─────────────────────────────────────────
def grab_via_output_attentions(out):
    att = getattr(out, "attentions", None)
    if att is None:
        return None, "outputs.attentions 是 None"
    if len(att) == 0:
        return None, "outputs.attentions 是空 tuple"
    if att[0] is None:
        return None, "outputs.attentions[0] 是 None"
    return list(att), f"{len(att)} 层，来自 output_attentions"


def grab_via_hooks(model):
    """在 self_attn 模块上挂 forward hook 抓第二个返回值。

    比 output_attentions 稳 —— 不依赖 transformers 把参数一路传下去。
    """
    captured: list = []
    handles = []

    def mk(idx):
        def hook(mod, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                captured.append((idx, out[1].detach()))
        return hook

    for i, mod in enumerate(
        m for m in model.modules() if type(m).__name__ in ("LlamaAttention", "LlamaSdpaAttention")
    ):
        handles.append(mod.register_forward_hook(mk(i)))

    return captured, handles


# ── 判据 ─────────────────────────────────────────────────────
def check(attn_layers, span, tag: str) -> bool:
    lo, hi = span
    n_vis = hi - lo
    ok_all = True

    print(f"\n\033[1m── {tag} ──\033[0m")
    a0 = attn_layers[0]
    print(f"  张量形状: {tuple(a0.shape)}  (期望 [batch, heads, seq, seq])")
    print(f"  层数:     {len(attn_layers)}")
    print(f"  视觉 span: [{lo}, {hi})  共 {n_vis} 个 token")
    print(f"  seq 长度:  {a0.shape[-1]}")
    print()

    # 判据 1 · 形状
    if a0.dim() != 4:
        print("  \033[31m✗\033[0m 判据1 形状不对，不是 4 维")
        return False
    print("  \033[32m✓\033[0m 判据1 拿到 [layers, heads, seq, seq]")

    # 判据 2 · 行和为 1
    rowsum = attn_layers[-1].float().sum(-1)
    lo_s, hi_s, mean_s = rowsum.min().item(), rowsum.max().item(), rowsum.mean().item()
    dev = max(abs(lo_s - 1.0), abs(hi_s - 1.0))
    if dev > 1e-3:
        print(f"  \033[31m✗\033[0m 判据2 行和偏离 1（min={lo_s:.4f} max={hi_s:.4f} mean={mean_s:.4f}）")
        print("        → 这不是 post-softmax 注意力。可能拿的是 logits 或半成品。")
        ok_all = False
    else:
        print(f"  \033[32m✓\033[0m 判据2 行和为 1（偏离 {dev:.2e}）→ 确认是 post-softmax")

    # 判据 3 · 视觉列不是全零、不是均匀 ← 最容易假阳性
    print()
    print("  逐层看视觉 token 上的注意力（文本 query 行）：")
    print(f"    {'层':>4}  {'质量':>7}  {'峰值/均匀':>9}  {'零占比':>7}  熵比")
    flags = []
    for li, a in enumerate(attn_layers):
        a = a.float()
        q = a[0, :, hi:, lo:hi]           # [heads, 文本query, 视觉token]
        if q.numel() == 0:
            continue
        mass = q.sum(-1).mean().item()    # 每个文本位置分给视觉的质量
        peak = q.max(-1).values.mean().item() * n_vis   # 相对均匀的倍数，1.0 = 均匀
        zero = (q == 0).float().mean().item()
        p = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
        ent_ratio = ent / torch.tensor(float(n_vis)).log().item()
        flags.append((mass, peak, zero, ent_ratio))
        if li < 4 or li >= len(attn_layers) - 2:
            print(f"    {li:>4}  {mass:>7.4f}  {peak:>9.2f}  {zero:>7.3f}  {ent_ratio:.3f}")
        elif li == 4:
            print(f"    {'…':>4}")

    uniform_mass = n_vis / a0.shape[-1]
    best_peak = max(f[1] for f in flags) if flags else 0.0
    best_mass = max(f[0] for f in flags) if flags else 0.0
    print()
    print(f"  均匀基线: 质量应≈{uniform_mass:.4f}，峰值/均匀≈1.00，熵比≈1.00")
    print(f"  实测最好: 质量={best_mass:.4f}，峰值/均匀={best_peak:.2f}")

    if best_peak < 1.5 and abs(best_mass - uniform_mass) < 0.05:
        print("  \033[31m✗\033[0m 判据3 视觉注意力近似均匀 —— 拿到了张量但没有跨模态信号")
        print("        → 可能是 hook 位置不对，或拿到了 mask 后的中间量")
        ok_all = False
    elif max(f[2] for f in flags) > 0.999:
        print("  \033[31m✗\033[0m 判据3 视觉列几乎全零 —— 形状对、行和对，但是空的")
        print("        → 这是最隐蔽的失败模式。查 mask 和视觉 token 定位。")
        ok_all = False
    else:
        print("  \033[32m✓\033[0m 判据3 视觉列有非平凡信号（非全零、非均匀）")

    return ok_all


# ── 主流程 ───────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="T0.4 注意力 hook 验证")
    ap.add_argument("--model", required=True, help="LLaVA-1.5-7B 路径或 HF id")
    ap.add_argument("--image", default=None, help="不传则用合成图")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-hooks", action="store_true", help="只试 output_attentions")
    a = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]

    print(f"加载 {a.model}  (attn_implementation=eager, dtype={a.dtype})")
    print("  注意：用 FlashAttention-2 会静默返回 None，不报错。")

    processor = AutoProcessor.from_pretrained(a.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        a.model,
        torch_dtype=dtype,
        attn_implementation="eager",   # ← 硬要求
        low_cpu_mem_usage=True,
    ).to(a.device).eval()

    image = Image.open(a.image).convert("RGB") if a.image else make_image()
    inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(a.device)
    print(f"  input_ids: {tuple(inputs['input_ids'].shape)}")

    img_id = find_image_token_id(processor)
    if img_id is None:
        print("\033[31m找不到图像占位 token，无法定位视觉 span。\033[0m")
        print("请手动确认 tokenizer 里的图像 token 名，改 find_image_token_id()。")
        return 1
    span = visual_span(inputs["input_ids"], img_id)
    if span is None:
        print(f"\033[31minput_ids 里找不到 token id {img_id}，视觉 span 定位失败。\033[0m")
        return 1

    # 路径 A：output_attentions
    captured, handles = ([], []) if a.no_hooks else grab_via_hooks(model)
    try:
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
    finally:
        for h in handles:
            h.remove()

    layers, src = grab_via_output_attentions(out)
    if layers:
        print(f"\n\033[32m路径 A 成功\033[0m：{src}")
    else:
        print(f"\n\033[33m路径 A 失败\033[0m：{src}")

    if captured:
        captured.sort(key=lambda x: x[0])
        print(f"\033[32m路径 B 成功\033[0m：hook 抓到 {len(captured)} 层 self_attn 输出")
    else:
        print("\033[33m路径 B 失败\033[0m：hook 没抓到任何 self_attn 输出")

    if not layers and not captured:
        print("\n\033[31m两条路都没拿到注意力。T0.4 不通过。\033[0m")
        print("下一步：确认 model.config._attn_implementation，检查 transformers 版本，")
        print("或改成自己实现 attention 层（见 代码任务清单 第六节）。")
        return 1

    ok_a = check(layers, span, "路径 A · output_attentions") if layers else False
    ok_b = check([c[1] for c in captured], span, "路径 B · forward hook") if captured else False

    print("\n" + "=" * 56)
    if ok_a or ok_b:
        print("\033[32mT0.4 通过。\033[0m可以开始写 SAP 与诱饵代码。")
        if ok_a and ok_b:
            print("两条路都通 —— 建议后续统一用路径 B（不受 transformers 版本影响）。")
        print("\n环境指纹（记进结果块的 env 行）：")
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from result_block import fingerprint, _kv

        print("  " + _kv(fingerprint(model=a.model, attn_impl="eager", device=a.device)))
        return 0

    print("\033[31mT0.4 不通过。停止全部后续工作。\033[0m")
    print("拿不到可用的跨模态注意力 → 整条路走不通。")
    print("立刻决策：换模型实现方式（自己改 attention 层），还是换命题。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
