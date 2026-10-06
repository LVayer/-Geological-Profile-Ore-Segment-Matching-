"""Run repeatable layout validation on the two reviewed experiment drawings.

The output images keep the original drawing visible and add only thin overlays:
green = rectangular plot extent, cyan = detected geological content boundary,
magenta = internal blank holes, yellow = coloured legend swatches.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOFTWARE_ROOT.parent
sys.path.insert(0, str(SOFTWARE_ROOT))

from stratamatch.layout import detect


INPUT = ROOT / "输入数据" / "实验"
OUTPUT = ROOT / "输出数据" / "experiment_validation"


def _draw_polygon(draw: ImageDraw.ImageDraw, points, colour, width: int) -> None:
    if len(points) >= 3:
        draw.line([tuple(p) for p in points] + [tuple(points[0])], fill=colour, width=width)


def validate(path: Path) -> dict:
    proposal = detect(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(5, round(max(image.size) / 1500))

    if proposal["roi"]:
        x, y, w, h = proposal["roi"]
        draw.rectangle((x, y, x + w, y + h), outline=(0, 220, 40), width=line_width * 2)
    for ring in proposal["content_rings"]:
        _draw_polygon(draw, ring["points"], (255, 0, 190) if ring["hole"] else (0, 235, 255), line_width)
    for index, entry in enumerate(proposal["legend"], 1):
        x, y, w, h = entry["detected_box"]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 220, 0), width=line_width * 2)
        draw.text((x + w + line_width, y), str(index), fill=(180, 40, 0), stroke_width=2, stroke_fill="white")

    # A review-sized overlay avoids creating another multi-megabyte copy while
    # retaining enough detail to judge plot and legend positions.
    scale = min(1.0, 2400 / max(image.size))
    preview = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    overlay_path = OUTPUT / f"{path.stem}_layout_overlay.jpg"
    preview.save(overlay_path, quality=92, subsampling=0)
    return {
        "image": str(path.relative_to(ROOT)),
        "size": list(image.size),
        "roi": proposal["roi"],
        "roi_candidates": proposal["roi_candidates"],
        "content_outer_count": sum(not r["hole"] for r in proposal["content_rings"]),
        "content_hole_count": sum(r["hole"] for r in proposal["content_rings"]),
        "legend_candidate_count": len(proposal["legend"]),
        "inferred_blank_count": sum(item.get("inferred_blank", False) for item in proposal["legend"]),
        "legend_boxes": [item["detected_box"] for item in proposal["legend"]],
        "status": proposal["status"],
        "warnings": proposal["warnings"],
        "overlay": str(overlay_path.relative_to(ROOT)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [validate(path) for path in sorted(INPUT.glob("*")) if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    report = {
        "meaning": "Detector proposals for manual review; an interior blank cell is inferred only between aligned coloured swatches.",
        "results": results,
    }
    report_path = OUTPUT / "layout_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
