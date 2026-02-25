# Multi-GPU Inference Guide

## 概述

`generate_step_multi_gpu.py` 已修改为支持多 GPU 数据并行推理，可以显著加速 STEP 文件生成。

## 工作原理

- **数据并行**：每个 GPU 加载完整的模型，处理不同的数据子集
- **自动数据分配**：2047 个样本会自动分配到 5 个 GPU 上（每个 GPU 约 409-410 个样本）
- **独立处理**：每个 GPU 独立处理自己的数据，无需 GPU 间通信

## 使用方法

### 方法 1：使用提供的脚本（推荐）

```bash
# 使用所有 5 个 GPU（默认）
bash /home/group/cad_codebased/eval_ckpt/run_multi_gpu.sh

# 或指定 GPU 数量
bash /home/group/cad_codebased/eval_ckpt/run_multi_gpu.sh 3
```

### 方法 2：直接使用 torchrun

```bash
# 激活环境
conda activate /home/group/cad_codebased/cad_llm3

# 使用 5 个 GPU
torchrun --nproc_per_node=5 \
    /home/group/cad_codebased/eval_ckpt/generate_step_multi_gpu.py

# 或使用 3 个 GPU
torchrun --nproc_per_node=3 \
    /home/group/cad_codebased/eval_ckpt/generate_step_multi_gpu.py
```

### 方法 3：单 GPU 模式（向后兼容）

```bash
# 直接运行（不使用 torchrun），会自动使用单 GPU 模式
python /home/group/cad_codebased/eval_ckpt/generate_step_multi_gpu.py
```

## 性能预期

- **单 GPU**：处理 2047 个样本，假设每个样本 30 秒，总时间约 17 小时
- **5 GPU**：理论上可达到接近 5 倍加速，总时间约 3.4 小时

实际加速比取决于：
- 模型加载时间（每个 GPU 都需要加载模型）
- 数据 I/O 时间
- GPU 利用率

## 输出说明

每个 GPU 进程会输出带 `[GPU X]` 前缀的日志，例如：
```
[GPU 0] Processing 410 samples (indices: 0 to 2046 step 5)
[GPU 1] Processing 410 samples (indices: 1 to 2046 step 5)
[GPU 2] Processing 409 samples (indices: 2 to 2046 step 5)
...
```

## 注意事项

1. **显存要求**：每个 GPU 需要足够的显存来加载完整模型（Qwen2.5-3B 约需要 6-8GB）
2. **文件冲突**：代码已处理文件写入，不同 GPU 处理不同的数据，不会冲突
3. **随机种子**：所有 GPU 使用相同的随机种子（42），确保数据一致性
4. **跳过已存在**：如果某个样本的输出已存在，会自动跳过

## 故障排除

### 如果遇到 NCCL 错误
```bash
# 尝试设置环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

### 如果某个 GPU 内存不足
减少 GPU 数量：
```bash
torchrun --nproc_per_node=3 generate_step_multi_gpu.py
```

### 检查 GPU 使用情况
在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

## 参考

- [Unsloth Multi-GPU 文档](https://docs.unsloth.ai/basics/multi-gpu-training-with-unsloth)
- PyTorch Distributed: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

