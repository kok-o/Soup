"""Exercise the local Qwen2-VL CLI path with the real MLX backend."""

import json
import math
import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml
from mlx.utils import tree_flatten
from mlx_vlm import load
from mlx_vlm.models.qwen2_vl import Model, ModelConfig, TextConfig, VisionConfig
from mlx_vlm.trainer import get_peft_model
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import SoupConfig
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.mlx_vision_sft import (
    MLXVisionSFTTrainerWrapper,
    _collate,
    _load_dense_model,
    _prepare_row,
    _vision_loss,
)


def fixture(root):
    model_dir = root / "qwen2"
    model_dir.mkdir()
    tokens = [
        "[UNK]", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
        "<|image_pad|>", "<|video_pad|>", "user", "assistant", "Describe", "this", "image",
        "Red", "Blue", "square", ".", "Then", "what",
    ]
    vocab = {token: idx for idx, token in enumerate(tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        extra_special_tokens=tokens[1:7],
    )
    fast.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{% for part in message['content'] %}"
        "{% if part['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ part['text'] }}{% endif %}"
        "{% endfor %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    fast.save_pretrained(model_dir)
    text = dict(
        model_type="qwen2_vl", hidden_size=32, num_hidden_layers=1, intermediate_size=48,
        num_attention_heads=2, num_key_value_heads=2, rms_norm_eps=1e-6,
        vocab_size=len(vocab), rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        tie_word_embeddings=True,
    )
    vision = dict(
        model_type="qwen2_vl", depth=1, embed_dim=32, hidden_size=32, num_heads=4,
        patch_size=14, spatial_patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
    )
    config = ModelConfig(
        model_type="qwen2_vl", text_config=TextConfig(**text), vision_config=VisionConfig(**vision),
        image_token_id=5, video_token_id=6, vision_start_token_id=3, vocab_size=len(vocab),
        eos_token_id=[2],
    )
    mx.random.seed(27)
    model = Model(config)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(model_dir / "model.safetensors"), weights)
    (model_dir / "config.json").write_text(json.dumps({
        **text, "text_config": text, "vision_config": vision,
        "image_token_id": 5, "video_token_id": 6, "vision_start_token_id": 3,
        "eos_token_id": [2], "architectures": ["Qwen2VLForConditionalGeneration"],
    }), encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text(json.dumps({
        "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2,
        "min_pixels": 784, "max_pixels": 3136,
    }), encoding="utf-8")
    Image.new("RGB", (28, 28), (245, 15, 30)).save(root / "red.png")
    Image.new("RGB", (28, 28), (15, 30, 245)).save(root / "blue.png")
    Image.new("RGB", (56, 28), (15, 30, 245)).save(root / "wide.png")
    pairs = [
        ("red.png", "Red."), ("blue.png", "Red."), ("red.png", "Red square."),
        ("blue.png", "Blue."), ("red.png", "Red square."), ("blue.png", "Blue square."),
    ]
    rows = [{
        "image": image, "conversations": [
            {"from": "human", "value": "<image>\nDescribe this image."},
            {"from": "gpt", "value": answer},
        ],
    } for image, answer in pairs]
    (root / "vision.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    return {
        "base": "./qwen2", "backend": "mlx", "task": "sft", "modality": "vision",
        "data": {"train": "./vision.jsonl", "format": "llava", "max_length": 256, "val_split": 0.3},
        "training": {
            "quantization": "none", "epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 3,
            "seed": 41, "save_steps": 1, "logging_steps": 1, "eval_steps": 1,
            "gradient_checkpointing": True, "lora": {"r": 2, "alpha": 4, "dropout": 0.0},
        },
        "output": "./out",
    }


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.chdir(root)
        spec = fixture(root)
        (root / "soup.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
        cfg = SoupConfig.model_validate(spec)
        model, processor, _ = _load_dense_model(cfg.base)
        data = load_dataset(cfg.data)
        row = data["train"][0]
        rows = [_prepare_row(
            {**row, "image": image}, root=root, processor=processor,
            model_config=model.config, data=cfg.data,
        ) for image in ("red.png", "blue.png")]
        assert rows[0]["input_ids"].tolist() == rows[1]["input_ids"].tolist()
        assert rows[0]["image_grid_thw"].tolist() == rows[1]["image_grid_thw"].tolist()
        assert rows[0]["pixel_values"].shape == rows[1]["pixel_values"].shape
        pad = processor.tokenizer.pad_token_id
        get_peft_model(model, linear_layers=["v_proj"], rank=2, alpha=4, dropout=0.0, verbose=False)
        loss_and_grad = nn.value_and_grad(model, _vision_loss)
        losses = []
        gradients = []
        for example in rows:
            batch = _collate([example], max_length=256, pad_token_id=pad)
            (loss, metrics), grads = loss_and_grad(model, batch)
            losses.append(loss.item())
            gradients.append(dict(tree_flatten(grads)))
            assert metrics["num_vision_tokens"].item() > 0
        assert abs(losses[0] - losses[1]) > 1e-5, losses
        assert any(
            mx.max(mx.abs(gradients[0][name] - gradients[1][name])).item() > 1e-5
            for name in gradients[0]
        )

        # Compare one full batch to an unequal-token accumulation and compare
        # two rows of equal geometry with a third of different patch length.
        examples = rows + [_prepare_row(
            {**row, "image": "wide.png", "messages": [
                *row["messages"][:-1], {"role": "assistant", "content": "Blue square."},
            ]},
            root=root, processor=processor, model_config=model.config, data=cfg.data,
        )]
        full = _collate(examples, max_length=256, pad_token_id=pad)
        (full_loss, full_metrics), full_grads = loss_and_grad(model, full)
        parts = [
            loss_and_grad(model, _collate(part, max_length=256, pad_token_id=pad))
            for part in (examples[:2], examples[2:])
        ]
        sizes = [item[0][1]["num_tokens"].item() for item in parts]
        assert sizes[0] != sizes[1]
        assert full_metrics["num_tokens"].item() == sum(sizes)
        expected = sum(e["completion_mask"][1:].sum().item() for e in examples)
        assert expected == sum(sizes)
        assert full_metrics["num_vision_tokens"].item() == 4
        assert math.isclose(
            full_loss.item(), sum(item[0][0].item() * size for item, size in zip(parts, sizes))
            / sum(sizes), rel_tol=2e-4,
        )
        split_gradients = [dict(tree_flatten(item[1])) for item in parts]
        for name, full_gradient in tree_flatten(full_grads):
            expected = sum(float(size) * grad[name] for size, grad in zip(sizes, split_gradients))
            np.testing.assert_allclose(full_gradient, expected / sum(sizes), atol=2e-4, rtol=2e-3)

        runner = CliRunner()
        dry = runner.invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
        assert dry.exit_code == 0, (dry.output, dry.exception)
        trained = runner.invoke(
            app, ["train", "--config", "soup.yaml", "--yes"], catch_exceptions=False,
        )
        assert trained.exit_code == 0, trained.output
        assert (root / "out/0000002_adapters.safetensors").is_file()
        metadata = json.loads((root / "out/mlx_vision_run.json").read_text())
        assert metadata["optimizer"]["total_updates"] == 2
        assert metadata["vision_tokens"] > 0
        weights = mx.load(str(root / "out/adapters.safetensors"))
        assert weights and all(name.startswith("language_model.model.layers.") for name in weights)
        assert any(mx.any(tensor != 0).item() for name, tensor in weights.items()
                   if name.endswith(".lora_b"))
        reloaded, _ = load(root / "qwen2", adapter_path=root / "out")
        wrapper = MLXVisionSFTTrainerWrapper(
            cfg.model_copy(update={"output": str(root / "direct")})
        )
        wrapper.setup(data)
        assert wrapper.train()["total_steps"] == 2
        batch = _collate(rows, max_length=256, pad_token_id=pad)

        def logits(policy):
            return policy(
                batch["input_ids"], batch["pixel_values"], batch["attention_mask"],
                image_grid_thw=batch["image_grid_thw"],
            ).logits

        wrapper.model.eval()
        reloaded.eval()
        np.testing.assert_allclose(logits(wrapper.model), logits(reloaded), atol=2e-4, rtol=2e-3)
        loss, metrics = _vision_loss(reloaded, batch)
        assert math.isfinite(loss.item()) and metrics["num_sequences"].item() == 2
        print(json.dumps({
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "backend": str(mx.default_device()), "untrained_image_losses": losses,
            "metal_available": mx.metal.is_available(),
            "reloaded_batch_loss": loss.item(), "saved_tensors": list(weights),
            "gradient_parity_target_counts": sizes, **metadata,
        }, indent=2))


if __name__ == "__main__":
    main()
