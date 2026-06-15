from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (69, 123, 157),
    (131, 56, 236),
    (255, 183, 3),
    (32, 201, 151),
]


def draw_detections(
    image: Image.Image,
    boxes_xyxy: list[list[float]],
    scores: list[float],
    text_labels: list[str],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 350))

    for idx, (box, score, label) in enumerate(zip(boxes_xyxy, scores, text_labels)):
        color = PALETTE[idx % len(PALETTE)]
        x0, y0, x1, y1 = _clamp_box(box, width, height)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)

        caption = f"{label} {score:.2f}"
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]

        pad_x = 4
        pad_y = 3
        label_x0 = x0
        label_y0 = max(0, y0 - text_h - 2 * pad_y)
        label_x1 = min(width, label_x0 + text_w + 2 * pad_x)
        label_y1 = min(height, label_y0 + text_h + 2 * pad_y)

        if label_x1 - label_x0 < text_w + 2 * pad_x:
            label_x0 = max(0, label_x1 - text_w - 2 * pad_x)

        draw.rectangle((label_x0, label_y0, label_x1, label_y1), fill=color)
        draw.text((label_x0 + pad_x, label_y0 + pad_y), caption, fill=(255, 255, 255), font=font)

    return canvas


def _clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = int(max(0, min(width - 1, round(x0))))
    y0 = int(max(0, min(height - 1, round(y0))))
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1
