# π0.5 转换说明

## 1. 当前边界

转换器直接读取 LeRobot π0.5 checkpoint，不修改源 `config.json`，输出两个文件：

```text
pi05-model-f16.gguf       371 tensors，约 6.1 GB
mmproj-pi05-f16.gguf      439 tensors，约 832 MB
```

主模型包含 PaliGemma 文本主干、Action Expert、Action Projection 和 Time MLP。视觉文件包含 SigLIP Vision Encoder 和线性 Multimodal Projector。

## 2. 先验证映射

```bash
conda activate vla

python scripts/pi05/validate_main_conversion.py \
  ~/work/search_for_jobs/models/pi05/source/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0
```

预期输出：

```text
source tensors: 812
main GGUF tensors: 371
vision GGUF tensors: 439
mapping validation: OK
```

该检查只读取 Safetensors metadata，不会把 14 GB 权重整体载入内存。

## 3. 准备 tokenizer

LeRobot checkpoint 没有附带 tokenizer。主模型转换需要与 `google/paligemma-3b-pt-224` 对应的以下文件：

```text
tokenizer.model
tokenizer_config.json
special_tokens_map.json（如果上游 revision 提供）
added_tokens.json（如果上游 revision 提供）
```

PaliGemma 仓库受许可证约束。先在 Hugging Face 接受许可证并登录，再把这些小文件下载到独立目录。不要下载第二份 PaliGemma 权重，也不要把文件写进 π0.5 源目录。

## 4. Dry-run

```bash
SOURCE=~/work/search_for_jobs/models/pi05/source/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0
TOKENIZER=~/work/search_for_jobs/models/pi05/tokenizer

python convert_hf_to_gguf.py "$SOURCE" \
  --target-model-dir "$TOKENIZER" \
  --outtype f16 \
  --outfile /tmp/pi05-model-f16.gguf \
  --dry-run

python convert_hf_to_gguf.py "$SOURCE" \
  --mmproj \
  --outtype f16 \
  --outfile /tmp/mmproj-pi05-f16.gguf \
  --dry-run
```

## 5. 正式转换

确认 dry-run 的 Tensor 数量和预计大小正确后，去掉 `--dry-run`，并把输出放到独立目录：

```bash
mkdir -p ~/work/search_for_jobs/models/pi05/gguf

python convert_hf_to_gguf.py "$SOURCE" \
  --target-model-dir "$TOKENIZER" \
  --outtype f16 \
  --outfile ~/work/search_for_jobs/models/pi05/gguf/pi05-model-f16.gguf

python convert_hf_to_gguf.py "$SOURCE" \
  --mmproj \
  --outtype f16 \
  --outfile ~/work/search_for_jobs/models/pi05/gguf/mmproj-pi05-f16.gguf
```

F16 是第一版数值对齐基线。量化应当在 F16 Runtime 对齐以后进行。
