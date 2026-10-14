# FastGPT

EasyGPT 的单卡加速改版：模型结构、参数命名、权重格式与上游完全一致，
产出 checkpoint 可直接被原版 chat.py 加载，训练耗时大幅降低（3060 实测目标 12h → 5-6h，V100 目标 ~3h）。

上游：[87owo/EasyGPT](https://github.com/87owo/EasyGPT)（注意上游为自定义许可证，本仓库仅供个人学习）

## Transformer/

| 文件 | 说明 |
|---|---|
| `train.py` | 加速版训练（替换原版 train.py 使用） |
| `chat.py` | 对话推理（兼容上游 checkpoint，亦兼容本仓训练产物） |

## train.py 相对上游的优化

1. 注意力走 `F.scaled_dot_product_attention` fused kernel，不再物化 T×T 中间矩阵
   （训练路径 `is_causal=True`；pad mask 是多余的：右填充+因果+loss ignore_index 三重屏蔽）
2. 优化器默认 torch 原生 fused AdamW（`optimizer: "8bit"` 可回退 bitsandbytes）
3. Dataset 持有文件句柄，不再每个样本 open 一次
4. 指标在 GPU 累积，每 `sync_every`（默认 20）步才回 CPU
5. `torch.compile` 默认开（失败自动回退 eager）
6. DataLoader 加 `prefetch_factor` + `drop_last`
7. TF32 按设备能力自动开关（sm80+ 才生效）；`amp_dtype: "bfloat16"` 可切 bf16

## 显存与 batch_size 建议

| 模型 | 3060 8G | V100 16G |
|---|---|---|
| 421M（hidden 1024） | 4~6 | 8~16 |
| 77M（hidden 512） | 16 | 32 |

## 运行

```bash
pip install torch safetensors tqdm colorama   # bitsandbytes 仅 optimizer=8bit 时需要
cd Transformer
python train.py    # 训练（改文件底部的 stages / default_config）
python chat.py     # 对话（改文件底部 model_dir 指向 checkpoint）
```
