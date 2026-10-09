"""Visual inspection of synthetic annotations; never applies label repairs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def review_crop_box(
    bbox: Sequence[float], polygons: Sequence[Sequence[float]],
    image_size: tuple[int, int], *, padding: int = 32,
) -> tuple[int, int, int, int]:
    """Enclose the bbox and every polygon fragment, clipped to the source image."""
    x, y, w, h = bbox
    xs = [p for polygon in polygons for p in polygon[::2]]
    ys = [p for polygon in polygons for p in polygon[1::2]]
    width, height = image_size
    return (
        max(0, math.floor(min(x, *xs) - padding)),
        max(0, math.floor(min(y, *ys) - padding)),
        min(width, math.ceil(max(x + w, *(p + 1 for p in xs)) + padding)),
        min(height, math.ceil(max(y + h, *(p + 1 for p in ys)) + padding)),
    )


def render_review_case(
    image: Image.Image, annotation: dict[str, Any], case: dict[str, Any],
) -> Image.Image:
    """Show the full tray, untouched crop and annotated crop with one shared crop box."""
    raw = image.convert("RGB")
    crop_box = review_crop_box(annotation["bbox"], annotation["segmentation"], raw.size)
    overlay = raw.copy()
    draw = ImageDraw.Draw(overlay)
    x, y, w, h = annotation["bbox"]
    line_width = max(3, round(max(raw.size) / 360))
    draw.rectangle((x, y, x + w, y + h), outline=(235, 65, 65), width=line_width)
    for polygon in annotation["segmentation"]:
        points = list(zip(polygon[::2], polygon[1::2], strict=True))
        draw.line(points + points[:1], fill=(0, 180, 225), width=line_width)

    panel_size = (360, 480)
    card = Image.new("RGB", (1168, 720), "#f5f5f5")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw = ImageDraw.Draw(card)
    title = (f"Task {case['source_task']} | Ann {case['annotation_id']} | "
             f"Image {case['image_id']} | SKU {case['rpc_category_id']} | IoU {case['iou']:.4f}")
    draw.text((24, 18), title, fill="#202020", font=title_font)
    draw.text((24, 58), "Stored bbox: red", fill=(235, 65, 65), font=font)
    draw.text((250, 58), "Polygon boundary: cyan", fill=(0, 140, 180), font=font)
    draw.text((24, 92), "Inspect the original pixels; these outlines are existing labels, not repaired masks.",
              fill="#444444", font=font)

    panels = [("Full tray", overlay), ("Original crop", raw.crop(crop_box)),
              ("Same crop + labels", overlay.crop(crop_box))]
    for index, (label, panel) in enumerate(panels):
        left = 24 + index * 380
        draw.text((left, 133), label, fill="#202020", font=font)
        fitted = ImageOps.contain(panel, panel_size, Image.Resampling.LANCZOS)
        card.paste(fitted, (left + (360 - fitted.width) // 2, 168 + (480 - fitted.height) // 2))
    draw.text((24, 670), f"Crop in source pixels: {crop_box} | bbox: {annotation['bbox']}",
              fill="#444444", font=font)
    return card
