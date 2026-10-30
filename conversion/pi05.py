from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger


_LANGUAGE_PREFIX = "paligemma_with_expert.paligemma."
_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert."
_VISION_MARKERS = (".vision_tower.", ".multi_modal_projector.")

_TEXT_LAYERS = 18
_EXPERT_LAYERS = 18
_VOCAB_SIZE = 257152
_TEXT_HIDDEN = 2048
_TEXT_INTERMEDIATE = 16384
_EXPERT_HIDDEN = 1024
_EXPERT_INTERMEDIATE = 4096
_HEAD_DIM = 256


def _read_policy_config(dir_model: Path) -> dict[str, Any] | None:
    config_path = dir_model / "config.json"
    if not config_path.is_file():
        return None
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return None
    return config if config.get("type") == "pi05" else None


def _is_pi05_checkpoint(dir_model: Path) -> bool:
    return (dir_model / "model.safetensors").is_file() and _read_policy_config(dir_model) is not None


@ModelBase.register_hparams_loader(_is_pi05_checkpoint)
def _load_pi05_hparams(dir_model: Path) -> dict[str, Any]:
    policy = _read_policy_config(dir_model)
    assert policy is not None

    if policy.get("paligemma_variant") != "gemma_2b":
        raise ValueError(f"Unsupported PI0.5 PaliGemma variant: {policy.get('paligemma_variant')!r}")
    if policy.get("action_expert_variant") != "gemma_300m":
        raise ValueError(f"Unsupported PI0.5 action expert variant: {policy.get('action_expert_variant')!r}")

    image_features = [
        feature for feature in policy.get("input_features", {}).values() if feature.get("type") == "VISUAL"
    ]
    state_features = [
        feature for feature in policy.get("input_features", {}).values() if feature.get("type") == "STATE"
    ]
    action = policy.get("output_features", {}).get("action", {})
    if len(image_features) != 3 or any(feature.get("shape") != [3, 224, 224] for feature in image_features):
        raise ValueError("This converter supports exactly three 3x224x224 PI0.5 image inputs")
    if len(state_features) != 1 or state_features[0].get("shape") != [32]:
        raise ValueError("This converter supports a 32-dimensional PI0.5 state")
    if action.get("shape") != [32]:
        raise ValueError("This converter supports a 32-dimensional PI0.5 action")

    return {
        **policy,
        "architectures": ["Pi05ForConditionalGeneration"],
        "num_hidden_layers": _TEXT_LAYERS + _EXPERT_LAYERS,
        "max_position_embeddings": 8192,
        "hidden_size": _TEXT_HIDDEN,
        "intermediate_size": _TEXT_INTERMEDIATE,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": _HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": _VOCAB_SIZE,
        "vision_config": {
            "image_size": 224,
            "patch_size": 14,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "layer_norm_eps": 1e-6,
        },
    }


def _expected_policy_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        f"{_LANGUAGE_PREFIX}lm_head.weight": (_VOCAB_SIZE, _TEXT_HIDDEN),
        f"{_LANGUAGE_PREFIX}model.language_model.norm.weight": (_TEXT_HIDDEN,),
        f"{_EXPERT_PREFIX}lm_head.weight": (_VOCAB_SIZE, _EXPERT_HIDDEN),
        f"{_EXPERT_PREFIX}model.norm.dense.weight": (3072, _EXPERT_HIDDEN),
        f"{_EXPERT_PREFIX}model.norm.dense.bias": (3072,),
        "action_in_proj.weight": (_EXPERT_HIDDEN, 32),
        "action_in_proj.bias": (_EXPERT_HIDDEN,),
        "action_out_proj.weight": (32, _EXPERT_HIDDEN),
        "action_out_proj.bias": (32,),
        "time_mlp_in.weight": (_EXPERT_HIDDEN, _EXPERT_HIDDEN),
        "time_mlp_in.bias": (_EXPERT_HIDDEN,),
        "time_mlp_out.weight": (_EXPERT_HIDDEN, _EXPERT_HIDDEN),
        "time_mlp_out.bias": (_EXPERT_HIDDEN,),
    }

    for layer in range(_TEXT_LAYERS):
        prefix = f"{_LANGUAGE_PREFIX}model.language_model.layers.{layer}."
        shapes.update({
            prefix + "input_layernorm.weight": (_TEXT_HIDDEN,),
            prefix + "post_attention_layernorm.weight": (_TEXT_HIDDEN,),
            prefix + "self_attn.q_proj.weight": (_TEXT_HIDDEN, _TEXT_HIDDEN),
            prefix + "self_attn.k_proj.weight": (_HEAD_DIM, _TEXT_HIDDEN),
            prefix + "self_attn.v_proj.weight": (_HEAD_DIM, _TEXT_HIDDEN),
            prefix + "self_attn.o_proj.weight": (_TEXT_HIDDEN, _TEXT_HIDDEN),
            prefix + "mlp.gate_proj.weight": (_TEXT_INTERMEDIATE, _TEXT_HIDDEN),
            prefix + "mlp.up_proj.weight": (_TEXT_INTERMEDIATE, _TEXT_HIDDEN),
            prefix + "mlp.down_proj.weight": (_TEXT_HIDDEN, _TEXT_INTERMEDIATE),
        })

    for layer in range(_EXPERT_LAYERS):
        prefix = f"{_EXPERT_PREFIX}model.layers.{layer}."
        shapes.update({
            prefix + "input_layernorm.dense.weight": (3072, _EXPERT_HIDDEN),
            prefix + "input_layernorm.dense.bias": (3072,),
            prefix + "post_attention_layernorm.dense.weight": (3072, _EXPERT_HIDDEN),
            prefix + "post_attention_layernorm.dense.bias": (3072,),
            prefix + "self_attn.q_proj.weight": (_TEXT_HIDDEN, _EXPERT_HIDDEN),
            prefix + "self_attn.k_proj.weight": (_HEAD_DIM, _EXPERT_HIDDEN),
            prefix + "self_attn.v_proj.weight": (_HEAD_DIM, _EXPERT_HIDDEN),
            prefix + "self_attn.o_proj.weight": (_EXPERT_HIDDEN, _TEXT_HIDDEN),
            prefix + "mlp.gate_proj.weight": (_EXPERT_INTERMEDIATE, _EXPERT_HIDDEN),
            prefix + "mlp.up_proj.weight": (_EXPERT_INTERMEDIATE, _EXPERT_HIDDEN),
            prefix + "mlp.down_proj.weight": (_EXPERT_HIDDEN, _EXPERT_INTERMEDIATE),
        })

    vision_prefix = f"{_LANGUAGE_PREFIX}model.vision_tower.vision_model."
    projector_prefix = f"{_LANGUAGE_PREFIX}model.multi_modal_projector.linear."
    shapes.update({
        projector_prefix + "weight": (_TEXT_HIDDEN, 1152),
        projector_prefix + "bias": (_TEXT_HIDDEN,),
        vision_prefix + "embeddings.patch_embedding.weight": (1152, 3, 14, 14),
        vision_prefix + "embeddings.patch_embedding.bias": (1152,),
        vision_prefix + "embeddings.position_embedding.weight": (256, 1152),
        vision_prefix + "post_layernorm.weight": (1152,),
        vision_prefix + "post_layernorm.bias": (1152,),
    })
    for layer in range(27):
        prefix = vision_prefix + f"encoder.layers.{layer}."
        shapes.update({
            prefix + "layer_norm1.weight": (1152,),
            prefix + "layer_norm1.bias": (1152,),
            prefix + "layer_norm2.weight": (1152,),
            prefix + "layer_norm2.bias": (1152,),
            prefix + "mlp.fc1.weight": (4304, 1152),
            prefix + "mlp.fc1.bias": (4304,),
            prefix + "mlp.fc2.weight": (1152, 4304),
            prefix + "mlp.fc2.bias": (1152,),
            prefix + "self_attn.q_proj.weight": (1152, 1152),
            prefix + "self_attn.q_proj.bias": (1152,),
            prefix + "self_attn.k_proj.weight": (1152, 1152),
            prefix + "self_attn.k_proj.bias": (1152,),
            prefix + "self_attn.v_proj.weight": (1152, 1152),
            prefix + "self_attn.v_proj.bias": (1152,),
            prefix + "self_attn.out_proj.weight": (1152, 1152),
            prefix + "self_attn.out_proj.bias": (1152,),
        })
    return shapes


@ModelBase.register("Pi05ForConditionalGeneration")
class Pi05Model(TextModel):
    model_arch = gguf.MODEL_ARCH.PI0

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]]:
        # Keep the complete source inventory. Vision tensors are validated here
        # and are exported by the separate multimodal converter later.
        return item

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_checkpoint()

    def _validate_checkpoint(self) -> None:
        expected = _expected_policy_shapes()
        actual_names = set(self.model_tensors)
        missing = sorted(set(expected) - actual_names)
        unexpected = sorted(actual_names - set(expected))
        vision_names = [name for name in actual_names if any(marker in name for marker in _VISION_MARKERS)]
        if missing or unexpected or len(vision_names) != 439 or len(actual_names) != 812:
            raise ValueError(
                "PI0.5 checkpoint inventory mismatch: "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
                f"vision={len(vision_names)} (expected 439), total={len(actual_names)} (expected 812)"
            )

        for name, expected_shape in expected.items():
            tensor = self.model_tensors[name]()
            actual_shape = tuple(tensor.shape)
            if actual_shape != expected_shape:
                raise ValueError(f"PI0.5 tensor {name!r} has shape {actual_shape}, expected {expected_shape}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"PI0.5 tensor {name!r} has dtype {tensor.dtype}, expected torch.float32")

        logger.info("gguf: validated PI0.5 checkpoint: 812 tensors, including 439 vision tensors")

    def set_vocab(self):
        tokenizer_dir = self.target_model_dir or self.dir_model
        if not (tokenizer_dir / "tokenizer.model").is_file():
            raise FileNotFoundError(
                "PI0.5 does not include its PaliGemma tokenizer. Supply a directory containing the "
                "matching tokenizer.model with --target-model-dir."
            )
        source_dir, self.dir_model = self.dir_model, tokenizer_dir
        try:
            self._set_vocab_sentencepiece()
        finally:
            self.dir_model = source_dir
        self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(_TEXT_HIDDEN)
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(_TEXT_INTERMEDIATE)
        self.gguf_writer.add_head_count(8)
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(1e-6)
        self.gguf_writer.add_key_length(_HEAD_DIM)
        self.gguf_writer.add_value_length(_HEAD_DIM)
        self.gguf_writer.add_rope_freq_base(10000.0)
        self.gguf_writer.add_file_type(self.ftype)

        self.gguf_writer.add_variant("pi05")
        self.gguf_writer.add_action_expert_embedding_length(_EXPERT_HIDDEN)
        self.gguf_writer.add_action_expert_feed_forward_length(_EXPERT_INTERMEDIATE)
        self.gguf_writer.add_action_expert_block_count(_EXPERT_LAYERS)
        self.gguf_writer.add_inference_step_count(int(self.hparams["num_inference_steps"]))
        self.gguf_writer.add_action_step_count(int(self.hparams["n_action_steps"]))
        self.gguf_writer.add_state_dimension(int(self.hparams["max_state_dim"]))
        self.gguf_writer.add_action_dimension(int(self.hparams["max_action_dim"]))
        self.gguf_writer.add_vision_input_count(3)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        del name, bid, n_dims
        if new_name.startswith((
            "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "output_norm_dense",
        )) or ".attn_norm_dense." in new_name or ".ffn_norm_dense." in new_name:
            return gguf.GGMLQuantizationType.F32
        return False

    @staticmethod
    def _map_block_tensor(layer: int, suffix: str) -> str:
        names = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        try:
            return f"blk.{layer}.{names[suffix]}"
        except KeyError:
            raise ValueError(f"Unsupported PI0.5 block tensor suffix: {suffix!r}") from None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        if any(marker in name for marker in _VISION_MARKERS):
            return

        if name == f"{_LANGUAGE_PREFIX}lm_head.weight":
            yield "token_embd.weight", data_torch
            return
        if name == f"{_LANGUAGE_PREFIX}model.language_model.norm.weight":
            return

        language_match = re.fullmatch(
            re.escape(_LANGUAGE_PREFIX) + r"model\.language_model\.layers\.(\d+)\.(.+)", name
        )
        if language_match is not None:
            layer, suffix = language_match.groups()
            yield self._map_block_tensor(int(layer), suffix), data_torch
            return

        if name == f"{_EXPERT_PREFIX}lm_head.weight":
            return
        if name.startswith(f"{_EXPERT_PREFIX}model.norm.dense."):
            yield "output_norm_dense." + name.rsplit(".", 1)[-1], data_torch
            return

        expert_match = re.fullmatch(
            re.escape(_EXPERT_PREFIX) + r"model\.layers\.(\d+)\.(.+)", name
        )
        if expert_match is not None:
            layer_text, suffix = expert_match.groups()
            layer = int(layer_text) + _TEXT_LAYERS
            dense_match = re.fullmatch(r"(input_layernorm|post_attention_layernorm)\.dense\.(weight|bias)", suffix)
            if dense_match is not None:
                norm, tensor_suffix = dense_match.groups()
                target = "attn_norm_dense" if norm == "input_layernorm" else "ffn_norm_dense"
                yield f"blk.{layer}.{target}.{tensor_suffix}", data_torch
            else:
                yield self._map_block_tensor(layer, suffix), data_torch
            return

        if name.startswith(("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")):
            yield name, data_torch
            return

        raise ValueError(f"Unsupported PI0.5 tensor: {name!r}")


@ModelBase.register("Pi05ForConditionalGeneration")
class Pi05VisionModel(MmprojModel):
    @classmethod
    def filter_tensors(
        cls, item: tuple[str, Callable[[], Tensor]]
    ) -> tuple[str, Callable[[], Tensor]] | None:
        name, tensor_loader = item
        prefix = f"{_LANGUAGE_PREFIX}model."
        if not name.startswith(prefix) or not any(marker in name for marker in _VISION_MARKERS):
            return None
        return name.removeprefix(prefix), tensor_loader

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor_config.setdefault("image_mean", [0.5, 0.5, 0.5])
        self.preprocessor_config.setdefault("image_std", [0.5, 0.5, 0.5])
        expected = {
            name.removeprefix(f"{_LANGUAGE_PREFIX}model."): shape
            for name, shape in _expected_policy_shapes().items()
            if any(marker in name for marker in _VISION_MARKERS)
        }
        actual = set(self.model_tensors)
        if actual != set(expected):
            raise ValueError(
                "PI0.5 vision tensor inventory mismatch: "
                f"missing={sorted(set(expected) - actual)[:3]}, unexpected={sorted(actual - set(expected))[:3]}"
            )
        for name, expected_shape in expected.items():
            tensor = self.model_tensors[name]()
            if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float32:
                raise ValueError(
                    f"PI0.5 vision tensor {name!r} has shape {tuple(tensor.shape)} and dtype {tensor.dtype}; "
                    f"expected {expected_shape} and torch.float32"
                )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PI0)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_vision_use_gelu(True)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".embeddings." in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "multi_modal_projector.linear.weight":
            yield "mm.linear.weight", data_torch
            return
        if name == "multi_modal_projector.linear.bias":
            yield "mm.linear.bias", data_torch
            return
        yield from super().modify_tensors(data_torch, name, bid)
