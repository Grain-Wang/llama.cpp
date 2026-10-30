# π0.5 模型 Contract

## 1. 源模型

| 字段 | 值 |
| --- | --- |
| Hugging Face repo | `lerobot/pi05_base` |
| Revision | `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0` |
| Model type | `pi05` |
| Base model | `google/paligemma-3b-pt-224` |
| Backbone variant | `gemma_2b` |
| Action Expert variant | `gemma_300m` |
| Source dtype | F32 |
| Tensor count | 812 |
| Parameter count | 3,616,757,520 |

源模型目录只读保存，不得直接修改 `config.json`、拆分唯一权重文件或删除 Vision Tensor。所有转换中间文件必须写入独立工作目录。

## 2. 输入协议

每次 Observation 包含三路图像、一个文本任务和一个机器人状态向量。

| 输入 | 源字段 | Shape / 约束 |
| --- | --- | --- |
| 固定视角图像 | `observation.images.base_0_rgb` | `3 x 224 x 224` |
| 左腕图像 | `observation.images.left_wrist_0_rgb` | `3 x 224 x 224` |
| 右腕图像 | `observation.images.right_wrist_0_rgb` | `3 x 224 x 224` |
| 机器人状态 | `observation.state` | 32 个 F32 值 |
| 文本任务 | `task` | PaliGemma tokenizer，右侧 padding，最大 200 tokens |

三路图像的预处理、Token 顺序和缺失图像策略必须与参考实现一致。转换和 Runtime 不得静默丢弃任意一路图像。

## 3. 输出与 Flow Matching

| 字段 | 值 |
| --- | --- |
| Action dimension | 32 |
| Action chunk size | 50 |
| Action steps | 50 |
| Flow inference steps | 10 |
| Time beta alpha | 1.5 |
| Time beta beta | 1.0 |
| Minimum period | 0.004 |
| Maximum period | 4.0 |

Runtime 输出形状固定为 `50 x 32`。数值对齐必须固定三路图像、文本、状态、初始 Action Noise 和 10 个 Flow timestep。

## 4. 归一化边界

源 `policy_preprocessor.json` 和 `policy_postprocessor.json` 使用：

```text
VISUAL -> IDENTITY
STATE  -> QUANTILES
ACTION -> QUANTILES
```

当前 base checkpoint 的 `features` 为空，不包含特定机器人数据集的状态与动作统计量。因此当前模型可以用于 Runtime 和 OpenPI 数值对齐，但不能据此声明已经完成某个真实机器人数据集的动作反归一化或任务部署。

## 5. 计划中的 GGUF 边界

首版计划保持 llama.cpp 的模型与多模态投影边界：

```text
pi05-model-f16.gguf
├── Language/Prefix Backbone
├── Action Expert
├── Action input/output projection
└── Time MLP

mmproj-pi05-f16.gguf
├── SigLIP Vision Encoder
└── Multimodal Projector
```

GGUF architecture、variant metadata 和全部 Tensor 名称将在映射表完成后冻结。转换器不得通过猜测 Tensor 是否存在来替代必需的模型 variant 元数据。

当前主模型转换器使用 `general.architecture = pi0` 和 `pi0.variant = pi05`。主模型映射结果为 371 个 Tensor；视觉 GGUF 包含 439 个 Tensor。另外明确排除 PaliGemma 最终 Norm 和 Action Expert 的 `lm_head`。Gemma Norm 保留 checkpoint 中的原始值，`1 + weight` 语义由 Runtime 实现。

π0.5 checkpoint 不附带 PaliGemma tokenizer。完整转换需要从对应的 `google/paligemma-3b-pt-224` revision 准备 tokenizer 文件，并通过通用参数 `--target-model-dir` 指向该目录；转换器不会把 tokenizer 写回源模型目录。

## 6. 第一层验收条件

- 源文件 SHA256 与 `source-manifest.sha256` 一致。
- 812 个源 Tensor 全部进入映射表或有明确的排除理由。
- 转换器遇到未知 Tensor 时失败，不能静默跳过。
- 主模型 GGUF 与 mmproj GGUF 均可重新解析。
- 所有目标 Tensor 的名称、shape、dtype 和参数量可以追溯到源 Tensor。
- 转换过程不修改源 checkpoint。

