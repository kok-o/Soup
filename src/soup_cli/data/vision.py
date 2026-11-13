"""Shared message normalization for processor-aware vision training."""

from __future__ import annotations

from typing import Any


def _vision_messages_with_image_parts(
    messages: list[dict[str, Any]], image_count: int
) -> list[dict[str, Any]]:
    """Convert Soup's legacy ``<image>`` messages to HF multimodal content.

    LLaVA JSON rows reach the trainer with string ``content`` fields. Modern
    processors such as Idefics3 only preserve an image placeholder when the
    chat message contains a structured ``{"type": "image"}`` part. Passing
    the legacy string directly makes ``apply_chat_template`` silently drop the
    prompt text and image marker, then the processor rejects the accompanying
    image because the rendered text contains zero image tokens (#302).

    Existing structured messages are preserved. When an otherwise valid
    vision row omits the literal marker, place its image(s) at the start of the
    first user turn. Refuse excess markers rather than handing the processor a
    misleading image/token-count mismatch.
    """
    if image_count < 0:
        raise ValueError("image_count must be non-negative")
    if not messages:
        raise ValueError("Vision sample has no messages")

    converted: list[dict[str, Any]] = []
    represented_images = 0
    for message in messages:
        converted_message = dict(message)
        content = converted_message.get("content", "")
        parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    copied = dict(part)
                else:
                    copied = {"type": "text", "text": str(part)}
                parts.append(copied)
                if copied.get("type") == "image":
                    represented_images += 1
        elif isinstance(content, str):
            for index, chunk in enumerate(content.split("<image>")):
                if index:
                    parts.append({"type": "image"})
                    represented_images += 1
                text = chunk.strip()
                if text:
                    parts.append({"type": "text", "text": text})
        else:
            parts.append({"type": "text", "text": str(content)})
        converted_message["content"] = parts
        converted.append(converted_message)

    if represented_images > image_count:
        raise ValueError(
            "Vision sample contains "
            f"{represented_images} image placeholder(s) but only {image_count} image(s)"
        )
    missing_images = image_count - represented_images
    if missing_images:
        target = next(
            (message for message in converted if message.get("role") == "user"),
            converted[0],
        )
        target["content"] = [
            *({"type": "image"} for _ in range(missing_images)),
            *target["content"],
        ]
    return converted
