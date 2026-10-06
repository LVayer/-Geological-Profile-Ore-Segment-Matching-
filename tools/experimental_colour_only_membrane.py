"""纯颜色清杂 + 拓扑安全膜实验。

本实验刻意不使用连通域面积、细长形态、边界保护、文字/网格几何检测或
不确定线条保护。主图范围内每个像素只按图例颜色距离分类，随后交给气球膜。
它不接入正式流程，用来验证“唯一依据为颜色”在 P64/P17 上的实际效果。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
SOFTWARE = PROJECT / "软件"
sys.path.insert(0, str(SOFTWARE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratamatch.staged_reconstruction import _load_and_classify  # noqa: E402
from experimental_topology_safe_membrane import _fill_holes, _components, _perimeter, _render, _sheet, _simulate  # noqa: E402


CONFIG_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "配置"
OUTPUT_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "纯颜色气球实验"
CASES = ("22-P64", "23-P17")


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.float32([[rgb / 255.0]]), cv2.COLOR_RGB2LAB)[0, 0]


def classify_colour_only(crop: np.ndarray, colours: list[np.ndarray], cfg: dict,
                         allowed: np.ndarray) -> tuple[np.ndarray, dict]:
    """逐像素纯颜色分类；不进行任何形态、面积或路径判断。"""
    prototypes = np.asarray([_lab(np.asarray(c, np.float32)) for c in colours], np.float32)
    line_prototypes = []
    for item in (cfg.get("legend_catalog") or {}).get("items", []):
        line = item.get("line") or {}
        rgb = line.get("dominant_rgb")
        if line.get("present") and isinstance(rgb, list) and len(rgb) == 3:
            line_prototypes.append(_lab(np.asarray(rgb, np.float32)))

    h, w = crop.shape[:2]
    labels = np.full((h, w), -1, np.int16)
    best_error = np.full((h, w), np.inf, np.float32)
    winning_margin = np.zeros((h, w), np.float32)
    tolerance = float(cfg.get("lab_tolerance", 12))
    margin_limit = float(cfg.get("lab_margin", 4))
    line_wins = 0
    for y0 in range(0, h, 192):
        y1 = min(h, y0 + 192)
        lab = cv2.cvtColor(crop[y0:y1].astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        best = np.full(lab.shape[:2], np.inf, np.float32)
        second = best.copy()
        winner = np.zeros(lab.shape[:2], np.int16)
        for index, prototype in enumerate(prototypes):
            distance = np.linalg.norm(lab - prototype, axis=2)
            better = distance < best
            second = np.where(better, best, np.minimum(second, distance))
            best = np.where(better, distance, best)
            winner[better] = index
        eligible = (best < tolerance) & ((second - best) > margin_limit)
        if line_prototypes:
            line_best = np.full(lab.shape[:2], np.inf, np.float32)
            for prototype in line_prototypes:
                line_best = np.minimum(line_best, np.linalg.norm(lab - prototype, axis=2))
            rejected = eligible & (line_best + 2 < best)
            line_wins += int(rejected.sum())
            eligible &= ~rejected
        eligible &= allowed[y0:y1]
        labels[y0:y1] = np.where(eligible, winner, -1)
        best_error[y0:y1] = best
        winning_margin[y0:y1] = second - best

    audit = {
        "method": "pixelwise_colour_only",
        "lab_tolerance": tolerance,
        "lab_margin": margin_limit,
        "lithology_colour_count": len(prototypes),
        "non_lithology_line_colour_count": len(line_prototypes),
        "classified_pixels": int(np.count_nonzero(labels >= 0)),
        "background_pixels": int(np.count_nonzero(labels < 0)),
        "line_colour_wins": line_wins,
        "geometry_rules_used": False,
        "component_area_rules_used": False,
        "boundary_or_uncertain_protection_used": False,
        "mean_accepted_delta_e": float(best_error[labels >= 0].mean()) if np.any(labels >= 0) else None,
        "mean_accepted_margin": float(winning_margin[labels >= 0].mean()) if np.any(labels >= 0) else None,
    }
    return labels, audit


def run_case(name: str) -> None:
    cfg = json.loads((CONFIG_ROOT / f"{name}.json").read_text(encoding="utf-8-sig"))
    # 复用同一图例色估计和主图坐标，但完全丢弃其经过几何清杂的 labels。
    crop, _, colours, _, _, _, allowed, _, _, _ = _load_and_classify(cfg)
    labels, colour_audit = classify_colour_only(crop, colours, cfg, allowed)
    obstacle = labels >= 0
    palette = np.asarray(colours, np.uint8)
    cleaned = np.full_like(crop, 255)
    cleaned[obstacle] = palette[labels[obstacle]]

    membranes, audit = _simulate(obstacle)
    out_h, out_w = membranes[0].shape
    top = (out_h - crop.shape[0]) // 2
    left = (out_w - crop.shape[1]) // 2
    padded_cleaned = np.full((out_h, out_w, 3), 255, np.uint8)
    padded_cleaned[top:top + crop.shape[0], left:left + crop.shape[1]] = cleaned
    padded_obstacle = np.zeros((out_h, out_w), bool)
    padded_obstacle[top:top + obstacle.shape[0], left:left + obstacle.shape[1]] = obstacle

    # 只允许整张膜均匀外移吸收缩放误差，禁止直接焊接单个像素。
    membranes = [_fill_holes(cv2.dilate(m.astype(np.uint8),
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool))
                 for m in membranes]
    missing = [int(np.count_nonzero(padded_obstacle & ~m)) for m in membranes]
    perimeters = [_perimeter(m) for m in membranes]
    areas = [int(m.sum()) for m in membranes]
    components = [_components(m)[0] for m in membranes]
    nested = all(not np.any(membranes[i + 1] & ~membranes[i]) for i in range(3))
    hole_free = all(np.array_equal(m, _fill_holes(m)) for m in membranes)
    monotonic = all(perimeters[i + 1] <= perimeters[i] + 1e-6 for i in range(3))
    audit.update({
        "stage_perimeters_full_px": [round(x, 3) for x in perimeters],
        "stage_areas_full_px": areas,
        "stage_component_counts": components,
        "nested_contraction": nested,
        "hole_free": hole_free,
        "perimeter_nonincreasing": monotonic,
        "missing_target_pixels_full_by_stage": missing,
        "geometry_valid": bool(nested and hole_free and max(components) == 1 and not any(missing)),
        "physical_process_valid": bool(nested and hole_free and max(components) == 1 and
                                       not any(missing) and monotonic),
        "input_definition": "pixelwise legend colours only",
        "colour_only_audit": colour_audit,
        "experimental_only": True,
        "production_code_modified": False,
    })
    audit["valid"] = audit["physical_process_valid"]

    titles = [
        f"初始圆形气球｜周长 {perimeters[0]:.0f}px",
        f"减压中期｜周长 {perimeters[1]:.0f}px",
        f"低压贴合｜周长 {perimeters[2]:.0f}px",
        f"纯颜色最终轮廓｜周长 {perimeters[3]:.0f}px",
    ]
    frames = [_render(padded_cleaned, membrane, title)
              for membrane, title in zip(membranes, titles)]
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cleaned).save(out / "00_纯颜色清杂图.png")
    for filename, frame in zip(("01_初始圆形气球.png", "02_减压中期.png",
                                "03_低压贴合.png", "04_最终总体轮廓.png"), frames):
        Image.fromarray(frame).save(out / filename)
    Image.fromarray(_sheet(frames)).save(out / "05_放气过程对照.png")
    (out / "实验审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for case in CASES:
        print(f"processing {case}", flush=True)
        run_case(case)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
