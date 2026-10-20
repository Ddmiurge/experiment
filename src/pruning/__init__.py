"""剪枝器适配层 —— 统一接口，不改 third_party 里的任何仓库。

契约：
    prune(inputs, ratio) -> keep_indices

T1.2 先只接 TRIM 一个。其余（SparseVLM / FasterVLM / PruMerge / LLaVA-Mini）
留到 11 月扩模型时再说。
"""
