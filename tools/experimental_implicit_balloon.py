"""稳定的隐式弹性气球总体轮廓实验。

气球内部用二值区域表示，因此闭合膜不会自交或产生单点突刺。膜从外接圆开始
逐像素收缩；由目标颜色形成的刚性地层始终不可被排除。膜张力用尺度自适应的
圆盘闭运算建立平衡包络：窄凹槽被连续膜桥接，宽缓凹部仍可进入。
"""
from __future__ import annotations

import json
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
SOFTWARE = PROJECT / "软件"
sys.path[:0] = [str(SOFTWARE), str(Path(__file__).resolve().parent)]
from experimental_fullres_colour_membrane import (  # noqa: E402
    CONFIG_ROOT, classify_full_resolution, _supported_colour_contact,
)
from stratamatch.io import local  # noqa: E402

OUTPUT_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "连续弹性气球实验"
CASES = ("22-P64", "23-P17")


def _external_fill(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("没有可用于气球接触的目标颜色区域")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
    return filled.astype(bool), contour


def _perimeter(mask: np.ndarray) -> float:
    _, contour = _external_fill(mask)
    return float(cv2.arcLength(contour, True))


def _circle_mask(shape: tuple[int, int], cx: float, cy: float, radius: float) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    cv2.circle(out, (round(cx), round(cy)), max(1, round(radius)), 1, cv2.FILLED,
               cv2.LINE_8)
    return out.astype(bool)


def _clean_boundary_speckles(contact: np.ndarray, obstacle: np.ndarray):
    """只在粗边界带清除缺少二维支撑的零散目标色像素。

    第一遍接触集负责全图清杂；第二遍先用轻微开闭运算取得不受单像素毛刺影响的
    粗边界，再检查该边界附近和外侧。内部像素无条件保留，避免误删内部薄层。
    """
    h, w = contact.shape
    diagonal = float(np.hypot(h, w))
    rough_radius = max(4, round(diagonal * 0.0020))
    rough_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rough_radius + 1, 2 * rough_radius + 1))
    rough_source = cv2.morphologyEx(contact.astype(np.uint8), cv2.MORPH_OPEN,
                                    rough_kernel)
    rough_source = cv2.morphologyEx(rough_source, cv2.MORPH_CLOSE, rough_kernel)
    rough_inside, _ = _external_fill(rough_source.astype(bool))

    edge = cv2.morphologyEx(rough_inside.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(bool)
    band_radius = max(8, round(diagonal * 0.0040))
    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_radius + 1, 2 * band_radius + 1))
    edge_band = cv2.dilate(edge.astype(np.uint8), band_kernel).astype(bool)
    inspection_zone = edge_band | ~rough_inside

    support_radius = max(3, round(diagonal * 0.0016))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * support_radius + 1, 2 * support_radius + 1))
    density = cv2.filter2D(contact.astype(np.float32), -1,
                           support_kernel.astype(np.float32)
                           / float(support_kernel.sum()),
                           borderType=cv2.BORDER_CONSTANT)
    # 阈值低于第一遍的 0.36，只针对明显缺乏二维面积的散点，不重复大范围清杂。
    weak = density < 0.18
    removed = contact & inspection_zone & weak
    cleaned = contact & ~removed
    if not np.any(cleaned):
        raise RuntimeError("边界零散像素清理删除了全部接触集")
    audit = {
        "rough_boundary_radius_work_px": rough_radius,
        "inspection_band_radius_work_px": band_radius,
        "local_support_radius_work_px": support_radius,
        "minimum_local_support": 0.18,
        "inspected_contact_pixels": int(np.count_nonzero(contact & inspection_zone)),
        "removed_boundary_speckle_pixels": int(removed.sum()),
        "retained_contact_pixels": int(cleaned.sum()),
        "interior_pixels_exempt_from_second_pass": int(
            np.count_nonzero(contact & ~inspection_zone)),
    }
    return cleaned, removed, inspection_zone, audit


def _equilibrium_envelope(contact: np.ndarray, first_circle: np.ndarray):
    """以膜张力尺度闭合目标色块，得到单一且完整包含目标色块的平衡包络。"""
    h, w = contact.shape
    radius = int(np.clip(round(min(h, w) * 0.035), 12, 72))
    chosen = None
    component_count = None
    for trial in (radius, round(radius * 1.35), round(radius * 1.8),
                  round(radius * 2.4)):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * trial + 1, 2 * trial + 1))
        closed = cv2.morphologyEx(contact.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        closed = (closed.astype(bool) & first_circle)
        candidate, _ = _external_fill(closed)
        count, _ = cv2.connectedComponents(closed.astype(np.uint8), 8)
        component_count = count - 1
        if np.all(candidate[contact]):
            chosen = (candidate, trial)
            break
    if chosen is None:
        raise RuntimeError("张力包络未能完整包含目标颜色像素")
    return chosen[0], chosen[1], component_count


def _contract(first: np.ndarray, final: np.ndarray):
    """从首次接触状态逐像素收缩；每一步均保持最终刚性包络在膜内。"""
    region = first.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    snapshots = [region.copy()]
    areas = [int(region.sum())]
    max_steps = max(region.shape)
    for step in range(1, max_steps + 1):
        shrunk = cv2.erode(region.astype(np.uint8), kernel, iterations=1).astype(bool)
        region = shrunk | final
        if step % 12 == 0:
            snapshots.append(region.copy())
            areas.append(int(region.sum()))
        if np.array_equal(region, final):
            break
    if not np.array_equal(region, final):
        raise RuntimeError("隐式气球在最大迭代数内未收缩到平衡包络")
    target_area = 0.5 * (int(first.sum()) + int(final.sum()))
    middle = snapshots[int(np.argmin(np.abs(np.asarray(areas) - target_area)))]
    return middle, step


def _render(base: np.ndarray, region: np.ndarray) -> np.ndarray:
    """只在逐像素清杂底图上叠加轮廓，不重着色、不填充区域。"""
    image = base.copy()
    _, contour = _external_fill(region)
    cv2.drawContours(image, [contour], -1, (15, 45, 230), 3, cv2.LINE_AA)
    return image


def run_case(name: str, config_path=None, output_root=None) -> None:
    cfg = json.loads(Path(config_path or CONFIG_ROOT / f"{name}.json").read_text(encoding="utf-8-sig"))
    labels, colours, obstacle, _, full_labels, _, colour_audit = classify_full_resolution(cfg)
    contact_first, _, support_audit = _supported_colour_contact(obstacle)
    contact, removed_edge, inspection_zone, edge_cleanup_audit = \
        _clean_boundary_speckles(contact_first, obstacle)
    # 显示底图必须来自原图相同位置。分类标签只决定“保留还是清除”，不能用图例
    # 标准色替换原像素，更不能把一个工作网格单元整体涂成地层色。
    image_path = local(cfg["image"])
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        x, y, rw, rh = map(int, cfg["roi"])
        original_roi = np.asarray(source.crop((x, y, x + rw, y + rh)))
    if original_roi.shape[:2] != full_labels.shape:
        raise RuntimeError("原始 ROI 与逐像素分类标签尺寸不一致")
    pixel_clean = np.full_like(original_roi, 255)
    confirmed = full_labels >= 0
    # 仅把第二遍检查区内被拒绝的工作网格映射回原分辨率；内部区域沿用第一遍
    # 逐像素结果。映射只控制保留/清除，不生成任何颜色。
    display_keep_work = obstacle.copy()
    display_keep_work[inspection_zone] &= contact[inspection_zone]
    display_keep_full = cv2.resize(
        display_keep_work.astype(np.uint8),
        (full_labels.shape[1], full_labels.shape[0]),
        interpolation=cv2.INTER_NEAREST).astype(bool)
    confirmed &= display_keep_full
    pixel_clean[confirmed] = original_roi[confirmed]
    # 缩放仅用于显示和轮廓坐标统一；颜色分类早已在原分辨率逐像素完成。
    # 最近邻保证显示图中的底图像素只能来自原图或纯白；面积插值会在边缘制造
    # 原图中不存在的淡色混合像素，容易被误认为颜色回填。
    sampled_original = cv2.resize(original_roi, (labels.shape[1], labels.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    base = cv2.resize(pixel_clean, (labels.shape[1], labels.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    changed_from_original = np.any(base != sampled_original, axis=2)
    output_nonwhite = np.any(base != 255, axis=2)
    # 发生变化但结果不是白色，才可能是新增颜色或改色；正常清杂只能把像素变白。
    colour_pixels_added_or_changed = int(np.count_nonzero(
        changed_from_original & output_nonwhite))
    pixels_cleared_to_white = int(np.count_nonzero(
        changed_from_original & ~output_nonwhite))

    ys, xs = np.nonzero(contact)
    (cx, cy), object_radius = cv2.minEnclosingCircle(
        np.column_stack((xs, ys)).astype(np.float32))
    pad = max(40, int(math.ceil(object_radius * 0.16)))
    contact = np.pad(contact, pad, mode="constant")
    base = np.pad(base, ((pad, pad), (pad, pad), (0, 0)),
                  mode="constant", constant_values=255)
    cx, cy = cx + pad, cy + pad

    first = _circle_mask(contact.shape, cx, cy, object_radius + 2.0)
    initial = _circle_mask(contact.shape, cx, cy,
                           object_radius + max(30.0, object_radius * 0.08))
    final, tension_radius, source_components = _equilibrium_envelope(contact, first)
    middle, contraction_steps = _contract(first, final)
    stages = (initial, first, middle, final)
    stage_areas = [int(stage.sum()) for stage in stages]
    stage_perimeters = [_perimeter(stage) for stage in stages]
    missed = [int(np.count_nonzero(contact & ~stage)) for stage in stages]
    components = [cv2.connectedComponents(stage.astype(np.uint8), 8)[0] - 1
                  for stage in stages]
    nested = all(np.all(~stages[i + 1] | stages[i]) for i in range(3))
    valid = nested and not any(missed) and components == [1, 1, 1, 1]

    out = Path(output_root or OUTPUT_ROOT) / name
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base[pad:pad + labels.shape[0],
                         pad:pad + labels.shape[1]]).save(
        out / "00_第二遍边界零散像素清除.png")
    names = ("01_起始.png", "02_最开始接触.png",
             "03_最开始接触到最终.png", "04_最终.png")
    for filename, stage in zip(names, stages):
        Image.fromarray(_render(base, stage)).save(out / filename)
    # 最终轮廓同时映射回原图，便于核查薄层端部和浅色区域是否漏包。
    final_crop = final[pad:pad+labels.shape[0], pad:pad+labels.shape[1]]
    contours, _ = cv2.findContours(final_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_contours = []
    for contour in contours:
        points = contour.astype(np.float64)
        points[:,:,0] *= original_roi.shape[1]/labels.shape[1]
        points[:,:,1] *= original_roi.shape[0]/labels.shape[0]
        original_contours.append(np.rint(points).astype(np.int32))
    overlay = original_roi.copy()
    cv2.drawContours(overlay, original_contours, -1, (15,45,230), max(3,round(original_roi.shape[1]/700)), cv2.LINE_AA)
    Image.fromarray(overlay).save(out/'05_原图总体轮廓.png')
    Image.fromarray(pixel_clean).save(out/'06_原分辨率清杂.png')
    Image.fromarray(final_crop.astype(np.uint8)*255).save(out/'07_总体轮廓掩膜.png')
    rendered_final = _render(base, final)
    Image.fromarray(rendered_final[pad:pad+labels.shape[0],pad:pad+labels.shape[1]]).save(out/'08_清杂后总体轮廓.png')
    (out/'轮廓坐标.json').write_text(json.dumps({'roi':cfg['roi'], 'coordinate_system':'original image pixels',
        'rings':[(c[:,0,:]+np.array(cfg['roi'][:2])).tolist() for c in original_contours]},ensure_ascii=False),encoding='utf-8')

    audit = {
        "case": name,
        "model": "implicit_elastic_balloon_v2",
        "representation": "single_connected_binary_region",
        "contraction_steps_after_first_contact": contraction_steps,
        "tension_radius_work_px": tension_radius,
        "source_components_after_tension_close": source_components,
        "stage_area_work_px": stage_areas,
        "stage_perimeter_work_px": [round(v, 3) for v in stage_perimeters],
        "stage_missing_contact_pixels": missed,
        "stage_connected_component_counts": components,
        "nested_monotone_contraction": bool(nested),
        "self_intersection_possible": False,
        "colour_refill_used": False,
        "colour_pixels_added_or_changed_after_cleanup": colour_pixels_added_or_changed,
        "pixels_cleared_to_white": pixels_cleared_to_white,
        "computational_binary_region_used_for_contour_only": True,
        "display_pixels": "original ROI RGB retained only where full-resolution classification is confirmed",
        "display_resize": "nearest-neighbour; introduces no blended colours",
        "legend_palette_recolour_used": False,
        "annotation_box_used": False,
        "valid": bool(valid),
        "colour_audit": colour_audit,
        "spatial_support_audit": support_audit,
        "boundary_speckle_cleanup_audit": edge_cleanup_audit,
    }
    (out / "实验审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description='从任意剖面配置导出气球总体轮廓')
    parser.add_argument('--config', nargs='+')
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    for case in (args.config or CASES):
        print(f"processing {case}", flush=True)
        run_case(Path(case).stem if args.config else case, case if args.config else None, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
