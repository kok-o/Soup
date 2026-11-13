"""Dense Qwen2-VL supervised fine-tuning through MLX-VLM."""

from __future__ import annotations

import json
import math
import os
import stat
import time
from contextlib import ExitStack, contextmanager
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from rich.console import Console

from soup_cli.config.schema import validate_mlx_vision_sft
from soup_cli.data.vision import _vision_messages_with_image_parts
from soup_cli.trainer.loss_summary import summarize_training_loss
from soup_cli.trainer.mlx_masking import build_response_mask
from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer
from soup_cli.utils.paths import (
    atomic_write_bytes_group,
    is_network_or_device_path,
    is_under,
    open_no_follow,
    require_regular_file,
)
from soup_cli.utils.seeding import resolve_training_seed

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig

console = Console()
MLX_VLM_VERSION = "0.7.12"


def _open_row_image(value: Any, root: Path):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes) and value["bytes"]:
            with BytesIO(value["bytes"]) as stream, Image.open(stream) as image:
                return image.convert("RGB")
        value = value.get("path")
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("expected exactly one image as a path, PIL image or bytes struct")
    if is_network_or_device_path(value):
        raise ValueError("network and device image paths are unsupported")
    path = os.path.abspath(root / value)
    if not is_under(path, root):
        raise ValueError("image path is outside the data directory")
    require_regular_file(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = open_no_follow(path, flags)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("image path is not a regular file")
        with Image.open(stream) as image:
            return image.convert("RGB")


class _ExpandedVisionTokenizer:
    """Render Soup's response boundaries with the same expanded image tokens."""

    def __init__(self, processor, image_tokens: int):
        self.processor = processor
        self.image_tokens = image_tokens

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )
        image_token = self.processor.image_token
        prompt = prompt.replace(image_token, image_token * self.image_tokens, 1)
        return self.processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]


def _prepare_row(row, *, root, processor, model_config, data):
    import mlx.core as mx

    messages = _vision_messages_with_image_parts(row.get("messages") or [], 1)
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("expected system, user or assistant messages")
        for part in message["content"]:
            if part.get("type") == "image" and message["role"] == "user":
                continue
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise ValueError("only text and one user image are supported")
            if any(token in part["text"] for token in processor.tokenizer.all_special_tokens):
                raise ValueError("message text contains a reserved tokenizer token")
    if not any(
        message["role"] == "assistant"
        and any(p.get("text", "").strip() for p in message["content"])
        for message in messages
    ):
        raise ValueError("expected a non-empty assistant response")
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if prompt.count(processor.image_token) != 1:
        raise ValueError("chat template must render exactly one image placeholder")
    with _open_row_image(row.get("image"), root) as image:
        inputs = processor(
            images=[image], text=[prompt], add_special_tokens=False,
            return_attention_mask=True, return_token_type_ids=False,
        )
    token_ids, pixels, grid = (
        mx.array(inputs[key]) for key in ("input_ids", "pixel_values", "image_grid_thw")
    )
    if grid.shape != (1, 3):
        raise ValueError("expected one image grid with three dimensions")
    frames, height, width = grid[0].tolist()
    merge = model_config.vision_config.spatial_merge_size
    if frames != 1 or min(height, width) <= 0 or height % merge or width % merge:
        raise ValueError("image grid is incompatible with the model's spatial merge")
    patch_count = frames * height * width
    image_tokens = patch_count // (merge * merge)
    if pixels.ndim != 2 or pixels.shape[0] != patch_count:
        raise ValueError("image pixels and patch grid disagree")
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("processor must produce one token sequence per row")
    ids = token_ids[0].tolist()
    image_id = model_config.image_token_id
    if len(ids) < 2 or ids.count(image_id) != image_tokens or ids[-1] == image_id:
        raise ValueError("expanded image placeholders and patch grid disagree")
    if len(ids) > data.max_length:
        raise ValueError(
            f"expanded sequence has {len(ids)} tokens, above data.max_length={data.max_length}; "
            "shorten the conversation or resize the image"
        )
    if data.train_on_responses_only:
        tokenizer = _ExpandedVisionTokenizer(processor, image_tokens)
        masked_ids, mask = build_response_mask(messages, tokenizer, mask_history=data.mask_history)
        if masked_ids != ids:
            raise ValueError("response mask tokenization disagrees with the vision processor")
    else:
        mask = [1] * len(ids)
    vision_markers = {
        image_id, model_config.vision_start_token_id, model_config.video_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    mask = [keep and token not in vision_markers for token, keep in zip(ids, mask, strict=True)]
    if not any(mask[1:]):
        raise ValueError("row has no supervised causal target")
    return {
        "input_ids": token_ids[0],
        "completion_mask": mx.array(mask, dtype=mx.int32),
        "pixel_values": pixels,
        "image_grid_thw": grid,
    }


def _collate(rows, *, max_length, pad_token_id):
    import mlx.core as mx
    from mlx_vlm.trainer.vlm.sft.runtime import collate_vlm_batch

    batch = collate_vlm_batch(
        [{"input_ids": row["input_ids"], "completion_mask": row["completion_mask"]}
         for row in rows],
        max_length,
        pad_to_multiple=1,
        pad_token_id=pad_token_id,
    )
    # Qwen2-VL expects a flat patch stream even when two grids have equal shape.
    for key in ("pixel_values", "image_grid_thw"):
        batch[key] = mx.concatenate([row[key] for row in rows], axis=0)
    return batch


def _vision_loss(model, batch):
    from mlx_vlm.trainer.vlm.sft.runtime import vlm_sft_loss

    return vlm_sft_loss(model, batch, train_on_completions=True, assistant_id=None)


@contextmanager
def _checkpoint_language_layers(layers):
    from mlx_vlm.trainer.common.utils import enable_gradient_checkpointing

    original = {type(layer): type(layer).__call__ for layer in layers}
    try:
        enable_gradient_checkpointing(layers)
        yield
    finally:
        for layer_class, call in original.items():
            layer_class.__call__ = call


def _load_dense_model(base: str):
    import mlx.nn as nn
    from mlx_vlm import load
    from mlx_vlm.utils import get_model_path, load_config

    path = Path(base)
    if not path.is_dir():
        path = get_model_path(base, allow_patterns=["config.json"])
    metadata = load_config(path)
    text_config = metadata.get("text_config") or {}
    if metadata.get("model_type") != "qwen2_vl":
        raise ValueError("MLX vision SFT currently supports Qwen2-VL only.")
    if metadata.get("model_file") or metadata.get("auto_map"):
        raise ValueError("MLX vision SFT requires a built-in Qwen2-VL model and processor.")
    if any(
        config.get(key) is not None
        for config in (metadata, text_config)
        for key in ("quantization", "quantization_config")
    ):
        raise ValueError("MLX vision SFT requires a dense, unquantized Qwen2-VL base.")

    revision = None
    if not Path(base).is_dir():
        # The metadata-only Hub fetch resolves a commit. Use it for both weights
        # and processor files so a moving Hub branch cannot change either.
        revision = path.name
        path = get_model_path(base, revision=revision)
    model, processor = load(path, trust_remote_code=False)
    if any(isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
           for module in model.modules()):
        raise ValueError("MLX vision SFT loaded quantized weights; use a dense Qwen2-VL base.")
    if not getattr(processor, "chat_template", None):
        raise ValueError("Qwen2-VL checkpoint must supply a processor chat template.")
    if processor.image_token_id != model.config.image_token_id:
        raise ValueError("Qwen2-VL processor and checkpoint disagree on the image token.")
    return model, processor, revision


class MLXVisionSFTTrainerWrapper:
    """Use MLX-VLM's model, processor and loss with token-weighted Soup updates."""

    def __init__(self, config: SoupConfig, **kwargs: Any) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self._trainer_options = kwargs

    def setup(self, dataset: dict) -> None:
        cfg = self.config
        if (cfg.backend, cfg.task, cfg.modality) != ("mlx", "sft", "vision"):
            raise ValueError("MLX vision training requires backend=mlx, task=sft, modality=vision.")
        validate_mlx_vision_sft(cfg)
        if any(self._trainer_options.get(name) for name in ("deepspeed_config", "fsdp_config")):
            raise ValueError("MLX vision SFT does not support DeepSpeed or FSDP.")
        if self._trainer_options.get("report_to") not in (None, "none"):
            raise ValueError("MLX vision SFT supports Soup tracking only.")
        if self._trainer_options.get("trust_remote_code"):
            raise ValueError("MLX vision SFT does not accept --trust-remote-code.")
        try:
            if version("mlx-vlm") != MLX_VLM_VERSION:
                raise ImportError(f"MLX vision SFT requires mlx-vlm=={MLX_VLM_VERSION}.")
            import mlx.core as mx
        except (ImportError, PackageNotFoundError) as exc:
            raise ImportError(
                'Install MLX vision SFT with: pip install "soup-cli[mlx-vision]"'
            ) from exc
        if mx.distributed.init().size() != 1:
            raise ValueError("MLX vision SFT runs in a single process.")
        if not dataset.get("train"):
            raise ValueError("MLX vision SFT requires a non-empty training split.")
        validation_expected = cfg.data.val_split > 0 or cfg.training.eval_steps is not None
        if validation_expected and not dataset.get("val"):
            raise ValueError("MLX vision SFT was configured for validation but its split is empty.")
        self.model, self.processor, self.base_revision = _load_dense_model(cfg.base)
        self.root = Path(cfg.data.image_dir or Path(cfg.data.train).parent).absolute()
        self.dataset = dataset

    def _batch(self, rows, split):
        prepared = []
        for index, row in rows:
            try:
                prepared.append(_prepare_row(
                    row, root=self.root, processor=self.processor,
                    model_config=self.model.config, data=self.config.data,
                ))
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise ValueError(f"MLX vision {split} row {index + 1}: {exc}") from exc
        tokenizer = self.processor.tokenizer
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Qwen2-VL tokenizer must define a pad or EOS token.")
        return _collate(prepared, max_length=self.config.data.max_length, pad_token_id=pad)

    def _evaluate(self, batch_size):
        self.model.eval()
        total_loss = total_tokens = 0
        rows = self.dataset.get("val", [])
        try:
            for start in range(0, len(rows), batch_size):
                batch = self._batch(enumerate(rows[start:start + batch_size], start), "val")
                loss, metrics = _vision_loss(self.model, batch)
                count = int(metrics["num_tokens"].item())
                total_loss += loss.item() * count
                total_tokens += count
        finally:
            self.model.train()
        if not math.isfinite(total_loss) or total_tokens < 1:
            raise ValueError("MLX vision validation did not produce a finite supervised loss.")
        return total_loss / total_tokens

    def _save(self, destination: Path, metadata: dict, *, final: bool) -> None:
        from mlx_vlm.trainer import save_adapter

        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".mlx-vision-", dir=destination.parent) as staging:
            weights = Path(staging) / "adapters.safetensors"
            save_adapter(self.model, weights)
            config = weights.with_name("adapter_config.json").read_bytes()
            outputs = [
                (weights.read_bytes(), str(destination), "MLX vision weights"),
                (config, str(destination.with_name("adapter_config.json")), "MLX vision adapter"),
            ]
            if final:
                outputs.append((
                    json.dumps(metadata, indent=2).encode("utf-8") + b"\n",
                    str(destination.with_name("mlx_vision_run.json")),
                    "MLX vision run metadata",
                ))
            atomic_write_bytes_group(outputs)

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten, tree_map
        from mlx_vlm.trainer import get_peft_model

        from soup_cli.trainer.mlx_sft import resolve_mlx_target_keys

        if self.model is None or self.processor is None:
            raise RuntimeError("Call setup(dataset) before MLX vision SFT training.")
        cfg = self.config
        tcfg = cfg.training
        batch_size = 1 if tcfg.batch_size == "auto" else tcfg.batch_size
        rows = self.dataset["train"]
        window_size = batch_size * tcfg.gradient_accumulation_steps
        updates_per_epoch = math.ceil(len(rows) / window_size)
        total_updates = tcfg.epochs * updates_per_epoch
        plan = plan_optimizer(
            lr=tcfg.lr, optimizer=tcfg.optimizer, scheduler=tcfg.scheduler,
            warmup_ratio=tcfg.warmup_ratio, weight_decay=tcfg.weight_decay,
            total_updates=total_updates,
        )
        for warning in plan.warnings:
            console.print(warning, markup=False)

        seed = resolve_training_seed(tcfg)
        mx.random.seed(seed)
        rng = np.random.default_rng(seed if tcfg.data_seed is None else tcfg.data_seed)
        targets = []
        decoder_linears = [
            name for name, module in self.model.named_modules()
            if name.startswith("language_model.model.layers.") and isinstance(module, nn.Linear)
        ]
        for target in resolve_mlx_target_keys(tcfg.lora):
            matches = [
                name for name in decoder_linears
                if name == target or name.endswith("." + target)
            ]
            if not matches:
                raise ValueError(f"MLX vision LoRA target {target!r} matches no decoder linear.")
            targets.extend(matches)
        get_peft_model(
            self.model, linear_layers=targets, rank=tcfg.lora.r, alpha=tcfg.lora.alpha,
            dropout=tcfg.lora.dropout, verbose=False,
        )
        trainable = dict(tree_flatten(self.model.trainable_parameters()))
        if not trainable or any(
            not key.startswith("language_model.model.layers.")
            or not key.endswith((".lora_a", ".lora_b"))
            for key in trainable
        ):
            raise ValueError("MLX vision LoRA must target only language decoder layers.")
        if resume_from_checkpoint is not None:
            weights = mx.load(str(resume_from_checkpoint))
            if weights.keys() != trainable.keys() or any(
                value.shape != trainable[key].shape for key, value in weights.items()
            ):
                raise ValueError("MLX vision checkpoint tensors do not match the selected LoRA.")
            self.model.load_weights(list(weights.items()), strict=False)

        optimizer = build_optimizer(plan)
        loss_and_grad = nn.value_and_grad(self.model, _vision_loss)
        output_dir = Path(cfg.output)
        output_path = output_dir / "adapters.safetensors"
        losses = []
        val_loss = None
        totals = {"supervised_tokens": 0, "processed_tokens": 0, "vision_tokens": 0}
        eval_every = tcfg.eval_steps or total_updates
        val_enabled = bool(self.dataset.get("val"))
        started = time.perf_counter()
        self.model.train()
        with ExitStack() as cleanup:
            if display is not None:
                display.start(total_updates)
                cleanup.callback(display.stop)
            if tcfg.gradient_checkpointing:
                cleanup.enter_context(_checkpoint_language_layers(self.model.layers))
            for epoch in range(tcfg.epochs):
                order = rng.permutation(len(rows)).tolist()
                for start in range(0, len(rows), window_size):
                    indices = order[start:start + window_size]
                    grads = None
                    loss_sum = token_count = 0
                    for offset in range(0, len(indices), batch_size):
                        selected = indices[offset:offset + batch_size]
                        batch = self._batch(((i, rows[i]) for i in selected), "train")
                        (loss, metrics), current = loss_and_grad(self.model, batch)
                        count = int(metrics["num_tokens"].item())
                        weighted = tree_map(lambda g: g.astype(mx.float32) * count, current)
                        grads = weighted if grads is None else tree_map(
                            lambda total, add: total + add, grads, weighted
                        )
                        mx.eval(grads, loss)
                        loss_sum += loss.item() * count
                        token_count += count
                        totals["processed_tokens"] += int(metrics["num_processed_tokens"].item())
                        totals["vision_tokens"] += int(metrics["num_vision_tokens"].item())
                    if not token_count or not math.isfinite(loss_sum):
                        raise ValueError("MLX vision SFT found a non-finite or empty update loss.")
                    grads = tree_map(lambda grad: grad / token_count, grads)
                    grads, norm = optim.clip_grad_norm(grads, tcfg.max_grad_norm)
                    if not math.isfinite(norm.item()):
                        raise ValueError("Non-finite MLX vision gradient before optimizer update.")
                    updated = optimizer.apply_gradients(grads, self.model.trainable_parameters())
                    self.model.update(updated)
                    mx.eval(self.model.state, optimizer.state)
                    totals["supervised_tokens"] += token_count
                    losses.append({"loss": loss_sum / token_count})
                    step = len(losses)
                    if val_enabled and (step % eval_every == 0 or step == total_updates):
                        val_loss = self._evaluate(batch_size)
                    if step % max(1, tcfg.logging_steps) == 0 or step == total_updates:
                        report = dict(
                            step=step, epoch=step / updates_per_epoch, loss=losses[-1]["loss"],
                            lr=float(optimizer.learning_rate), val_loss=val_loss,
                        )
                        if display is not None:
                            display.update(**report)
                        if tracker is not None and run_id:
                            tracker.log_metrics(run_id=run_id, **report)
                    if tcfg.save_steps > 0 and step % tcfg.save_steps == 0:
                        self._save(output_path.with_name(f"{step:07d}_{output_path.name}"), {},
                                   final=False)
        duration = time.perf_counter() - started
        self._save(output_path, {
            "base": cfg.base,
            "base_revision": self.base_revision,
            "mlx_vlm": MLX_VLM_VERSION,
            "max_length": cfg.data.max_length,
            "train_on_responses_only": cfg.data.train_on_responses_only,
            "mask_history": cfg.data.mask_history,
            "batch_size": batch_size,
            "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
            "optimizer": plan.as_metadata(),
            **totals,
        }, final=True)
        return {
            **summarize_training_loss(losses),
            **totals,
            "val_loss": val_loss,
            "total_steps": len(losses),
            "duration_secs": duration,
            "output_dir": str(output_dir),
        }
