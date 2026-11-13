import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.mlx_routing import resolve_trainer


def vision_config(**overrides):
    config = {
        "base": "local-qwen2-vl",
        "backend": "mlx",
        "task": "sft",
        "modality": "vision",
        "data": {"train": "./vision.jsonl", "format": "llava", "val_split": 0},
        "training": {"quantization": "none", "lora": {"r": 2, "alpha": 4}},
    }
    for name, values in overrides.items():
        if isinstance(values, dict):
            config[name].update(values)
        else:
            config[name] = values
    return SoupConfig.model_validate(config)


@pytest.mark.parametrize("fmt", ["llava", "sharegpt4v"])
def test_mlx_vision_config_routes_to_a_vision_trainer(fmt):
    cfg = vision_config(data={"format": fmt})
    trainer, kwargs = resolve_trainer(cfg, {"device": "mlx"})
    assert trainer.__name__ == "MLXVisionSFTTrainerWrapper"
    assert kwargs == {"device": "mlx"}
    assert resolve_trainer(vision_config(backend="transformers"))[0] is None
    text_cfg = vision_config(modality="text", data={"format": "chatml"})
    assert resolve_trainer(text_cfg)[0].__name__ == "MLXSFTTrainerWrapper"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"data": {"format": "video"}}, "data.format"),
        ({"data": {"streaming": True}}, "data.streaming"),
        ({"data": {"chat_template": "chatml"}}, "data.chat_template"),
        ({"training": {"quantization": "4bit"}}, "training.quantization"),
        ({"training": {"lora": {"r": 2, "use_rslora": True}}}, "training.lora.use_rslora"),
        ({"training": {"gradient_checkpointing": "selective"}}, "training.gradient_checkpointing"),
    ],
)
def test_mlx_vision_rejects_settings_it_does_not_apply(overrides, field):
    with pytest.raises(ValueError, match=field):
        vision_config(**overrides)


def test_nonvalidated_vision_config_cannot_bypass_checks_in_dispatch():
    cfg = vision_config()
    cfg.training.quantization = "4bit"
    with pytest.raises(ValueError, match="training.quantization"):
        resolve_trainer(cfg)


def test_decode_vision_image_refuses_path_outside_data_directory(tmp_path):
    from soup_cli.trainer.mlx_vision_sft import _open_row_image

    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "image.png"
    outside.touch()
    with pytest.raises(ValueError, match="outside"):
        _open_row_image(str(outside), root)


def test_decode_vision_image_handles_rgb_conversion_and_embedded_bytes(tmp_path):
    from io import BytesIO

    from PIL import Image

    from soup_cli.trainer.mlx_vision_sft import _open_row_image

    path = tmp_path / "gray.png"
    Image.new("L", (11, 9), 23).save(path)
    with _open_row_image(path.name, tmp_path) as image:
        assert image.mode == "RGB" and image.getpixel((0, 0)) == (23, 23, 23)
    stream = BytesIO()
    Image.new("RGBA", (2, 3), (8, 16, 24, 32)).save(stream, format="PNG")
    with _open_row_image({"bytes": stream.getvalue(), "path": "../other"}, tmp_path) as image:
        assert image.mode == "RGB" and image.getpixel((0, 0)) == (8, 16, 24)
