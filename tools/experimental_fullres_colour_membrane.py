"""原始分辨率分块颜色分类，再运行气球膜实验。

RGB 图像绝不在颜色判定前缩小。为控制内存，原始 ROI 按行分块转 Lab；完成
分类后才把离散类别掩膜汇聚到气球工作网格，避免 Lanczos 将蓝线与地层混色。
该文件是独立实验，不接入正式流程。
"""
from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


PROJECT = Path(__file__).resolve().parents[2]
SOFTWARE = PROJECT / "软件"
sys.path.insert(0, str(SOFTWARE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratamatch.io import local  # noqa: E402
from stratamatch.staged_reconstruction import _polygon_mask  # noqa: E402
from experimental_topology_safe_membrane import (_components, _fill_holes, _perimeter,  # noqa: E402
                                                   _dilate, _render, _sheet, _simulate)


CONFIG_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "配置"
OUTPUT_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "原分辨率纯颜色气球实验"
CASES = ("22-P64", "23-P17")


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.float32([[rgb / 255.0]]), cv2.COLOR_RGB2LAB)[0, 0]


def _legend_colours(full: np.ndarray, cfg: dict) -> list[np.ndarray]:
    colours = []
    for entry in cfg["legend"]:
        if "swatch" in entry:
            x, y, w, h = map(int, entry["swatch"])
            patch = full[max(0, y):max(y + 1, y + h), max(0, x):max(x + 1, x + w)]
            patch_lab = cv2.cvtColor(patch.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
            keep = patch_lab[:, :, 0] > 45
            pixels = patch[keep] if np.any(keep) else patch.reshape(-1, 3)
            colour = np.median(pixels, axis=0)
        else:
            colour = np.asarray(entry["rgb"])
        colours.append(np.asarray(colour, np.uint8))
    return colours


def _normalise_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value or "").lower()


def _colour_limits(full, cfg, prototypes):
    """从色块内部估计 JPEG 波动；排除文字后设定每类的自适应色差上限。"""
    ceiling = float(cfg.get('lab_tolerance', 12))
    limits = []
    for entry, centre in zip(cfg['legend'], prototypes):
        limit = ceiling
        if 'swatch' in entry:
            x, y, w, h = map(int, entry['swatch'])
            patch = full[max(0,y):y+h, max(0,x):x+w]
            if patch.size:
                lab = cv2.cvtColor(patch.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
                distances = np.linalg.norm(lab-centre, axis=2)
                # 仅用接近填充底色的像素估计噪声，黑字和边框不能放宽阈值。
                core = distances[distances < ceiling]
                if core.size:
                    # 图例与绘图区可能使用略不同的填充色，保留可配置的域内偏色余量。
                    floor = min(float(cfg.get('lab_min_adaptive_tolerance',8)),ceiling)
                    limit = float(np.clip(np.percentile(core, 95)+2., floor, ceiling))
        limits.append(limit)
    return np.asarray(limits, np.float32)


def classify_full_resolution(cfg: dict, target_max_dimension: int = 2400):
    """在原图 ROI 逐块分类，并把类别掩膜保守汇聚到工作网格。"""
    image_path = local(cfg["image"])
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        full = np.asarray(source)
    original_h, original_w = full.shape[:2]
    x, y, rw, rh = map(int, cfg["roi"])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(original_w, x + rw), min(original_h, y + rh)
    roi = full[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]

    colours = _legend_colours(full, cfg)
    all_prototypes = np.asarray([_lab(np.asarray(c, np.float32)) for c in colours], np.float32)
    adaptive = cfg.get('adaptive_colour_cleanup', True)
    colour_limits = _colour_limits(full, cfg, all_prototypes)
    line_prototypes = []
    line_names = set()
    for item in (cfg.get("legend_catalog") or {}).get("items", []):
        line = item.get("line") or {}
        rgb = line.get("dominant_rgb")
        if line.get("present") and isinstance(rgb, list) and len(rgb) == 3:
            line_prototypes.append(_lab(np.asarray(rgb, np.float32)))
            line_names.add(_normalise_name(item.get("name", "")))

    # 只允许确认属于色块的图例进入目标色。若同名项目在图例目录中具有线条
    # 特征（例如“露天开采境界线”），从地层色目标中排除。
    non_lithology_indices = []
    pale_ambiguous_indices = []
    active_indices = []
    for index, (entry, prototype) in enumerate(zip(cfg["legend"], all_prototypes)):
        name = _normalise_name(entry.get("lithology", ""))
        is_line_item = bool(name and name in line_names)
        is_background_equal = bool(prototype[0] >= 97 and np.linalg.norm(prototype[1:]) < 3)
        if is_line_item:
            non_lithology_indices.append(index)
        elif is_background_equal:
            pale_ambiguous_indices.append(index)
        else:
            active_indices.append(index)
    prototypes = all_prototypes[active_indices]
    if not active_indices:
        raise ValueError('没有可区分于背景的地层图例色块')

    allowed = _polygon_mask((rh, rw), cfg.get("plot_content_rings"), 1.0, 1.0, (x0, y0))
    # 图例可位于主图矩形内部；排除框只屏蔽页面家具，不手绘地层外轮廓。
    for bx, by, bw, bh in cfg.get('exclude_boxes', []):
        allowed[max(0,by-y0):max(0,min(rh,by+bh-y0)),
                max(0,bx-x0):max(0,min(rw,bx+bw-x0))] = False
    labels = np.full((rh, rw), -1, np.int16)
    ambiguous = np.zeros((rh, rw), np.uint8)
    tolerance = float(cfg.get("lab_tolerance", 12))
    margin_limit = float(cfg.get("lab_margin", 4))
    accepted = 0
    line_colour_rejected = 0
    adaptive_rejected = 0
    # 每块只产生几十 MB 临时数组；不为整幅大图分配 Lab 浮点副本。
    for row0 in range(0, rh, 96):
        row1 = min(rh, row0 + 96)
        lab = cv2.cvtColor(roi[row0:row1].astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        best = np.full(lab.shape[:2], np.inf, np.float32)
        second = best.copy()
        winner_local = np.zeros(lab.shape[:2], np.int16)
        for local_index, prototype in enumerate(prototypes):
            distance = np.linalg.norm(lab - prototype, axis=2)
            better = distance < best
            second = np.where(better, best, np.minimum(second, distance))
            best = np.where(better, distance, best)
            winner_local[better] = local_index
        winner = np.asarray(active_indices, np.int16)[winner_local]
        eligible = (best < tolerance) & ((second - best) > margin_limit)
        if adaptive:
            # 纸张白色是竞争类别，防止浅色地层容差吞入页面背景。
            paper_distance = np.linalg.norm(lab-np.array([100.,0.,0.],np.float32), axis=2)
            supported = (best < colour_limits[winner]) & (best + 1.5 < paper_distance)
            adaptive_rejected += int(np.count_nonzero(eligible & ~supported & allowed[row0:row1]))
            eligible &= supported
        if line_prototypes:
            line_best = np.full(lab.shape[:2], np.inf, np.float32)
            for prototype in line_prototypes:
                line_best = np.minimum(line_best, np.linalg.norm(lab - prototype, axis=2))
            rejected = eligible & (line_best + 2 < best)
            line_colour_rejected += int(rejected.sum())
            eligible &= ~rejected
        eligible &= allowed[row0:row1]
        labels[row0:row1] = np.where(eligible, winner, -1)
        accepted += int(eligible.sum())
        if pale_ambiguous_indices:
            pale_distance = np.full(lab.shape[:2], np.inf, np.float32)
            for index in pale_ambiguous_indices:
                pale_distance = np.minimum(pale_distance,
                                           np.linalg.norm(lab - all_prototypes[index], axis=2))
            ambiguous[row0:row1] = ((pale_distance < tolerance) & allowed[row0:row1]).astype(np.uint8)

    # 释放完整 RGB，后续只处理离散标签。
    del roi, full, allowed
    gc.collect()

    scale = min(1.0, target_max_dimension / max(rh, rw))
    ww, wh = max(1, round(rw * scale)), max(1, round(rh * scale))
    best_coverage = np.zeros((wh, ww), np.float32)
    working_labels = np.full((wh, ww), -1, np.int16)
    for value in range(len(colours)):
        # INTER_AREA 只汇聚已经判定好的类别占比，不再混合 RGB 颜色。
        coverage = cv2.resize((labels == value).astype(np.float32), (ww, wh),
                              interpolation=cv2.INTER_AREA)
        better = coverage > best_coverage
        working_labels[better] = value
        best_coverage[better] = coverage[better]
    support_coverage = cv2.resize((labels >= 0).astype(np.float32), (ww, wh),
                                  interpolation=cv2.INTER_AREA)
    # 单个孤立像素不应在工作网格放大成硬障碍；低覆盖像素保留为不确定证据。
    # 气球工作网格只把多数面积确认为地层的单元作为刚性接触。旧值 0.12 会把
    # 以白色为主、只含少量地层色的单元整体扩张为障碍。
    minimum_coverage = 0.50
    obstacle = support_coverage >= minimum_coverage
    sparse_uncertain = (support_coverage > 0) & ~obstacle
    ambiguous_working = (cv2.resize(ambiguous.astype(np.float32), (ww, wh),
                                    interpolation=cv2.INTER_AREA) >= 0.35) | sparse_uncertain
    working_labels[~obstacle] = -1
    del best_coverage, support_coverage
    gc.collect()

    audit = {
        "method": "full_resolution_tiled_colour_classification",
        "source_image_shape_hw": [original_h, original_w],
        "original_roi_shape_hw": [rh, rw],
        "balloon_input_shape_hw": [wh, ww],
        "rgb_resized_before_classification": False,
        "classification_chunk_rows": 96,
        "lab_tolerance": tolerance,
        "lab_margin": margin_limit,
        "classified_original_pixels": accepted,
        "line_colour_rejected_original_pixels": line_colour_rejected,
        "active_lithology_indices": active_indices,
        "excluded_non_lithology_indices": non_lithology_indices,
        "background_equal_ambiguous_indices": pale_ambiguous_indices,
        "background_equal_ambiguous_original_pixels": int(ambiguous.sum()),
        "mask_downsample_scale": scale,
        "mask_downsample_method": "per-class area coverage; RGB never resized",
        "minimum_working_coverage": minimum_coverage,
        "sparse_uncertain_working_pixels": int(sparse_uncertain.sum()),
        "geometry_rules_used": False,
        "adaptive_colour_cleanup": bool(adaptive),
        "per_class_lab_limits": colour_limits.tolist(),
        "adaptive_rejected_original_pixels": adaptive_rejected,
    }
    return working_labels, colours, obstacle, ambiguous_working, labels, ambiguous.astype(bool), audit


def _supported_colour_contact(obstacle: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """把有二维色块支撑的像素作为膜接触集。

    颜色仍是进入候选的唯一语义依据；这里不按线名、方向或连通域面积猜类别。
    但单像素/窄线即使碰巧与某个地层色相同，也不能证明其是地层区域。先在很小
    的圆邻域检查二维覆盖，再只恢复紧邻二维色块的同色候选。被拒绝的候选保留为
    不确定证据，不允许它把总体轮廓拉成长针。
    """
    h, w = obstacle.shape
    diagonal = float(np.hypot(h, w))
    radius = max(8, round(diagonal * 0.0045))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    density = cv2.filter2D(obstacle.astype(np.float32), -1,
                           kernel.astype(np.float32) / float(kernel.sum()),
                           borderType=cv2.BORDER_CONSTANT)
    # 总体轮廓只需由二维区域支撑。较大的邻域可拒绝粗网格、引线和字符笔画；
    # 这一步不用于删除或合并后续地层，只决定哪些像素能够阻挡外部膜。
    dense_seed = density >= 0.36
    supported_zone = _dilate(dense_seed, radius)
    contact = obstacle & supported_zone
    rejected = obstacle & ~contact
    audit = {
        "support_radius_work_px": radius,
        "minimum_local_colour_coverage": 0.36,
        "candidate_pixels": int(obstacle.sum()),
        "supported_contact_pixels": int(contact.sum()),
        "unsupported_colour_pixels": int(rejected.sum()),
        "unsupported_pixels_policy": "uncertain; excluded from membrane contact",
    }
    return contact, rejected, audit


def _refill_narrow_gaps(labels: np.ndarray, contact: np.ndarray,
                        inside: np.ndarray) -> tuple[np.ndarray, dict]:
    """用最近的可靠地层色回填窄空洞，宽空白仍保持不确定。

    距离传播同时自然处理两侧颜色不同的窄线：像素归入距离更近的一侧。最大传播
    距离很小，因此不会跨越大块白色区域，也不会把背景同色地层武断改成邻层。
    """
    distance, nearest = ndimage.distance_transform_edt(~contact, return_indices=True)
    nearest_label = labels[nearest[0], nearest[1]]
    max_distance = max(3, round(float(np.hypot(*labels.shape)) * 0.0014))
    unknown = inside & ~contact
    # 只有完全封闭在膜内部的孔洞可以回填。与总体轮廓相连的白区可能是真背景，
    # 即使离地层色很近也禁止染色，避免把错误凸起伪装成地层。
    count, component, _, _ = cv2.connectedComponentsWithStats(unknown.astype(np.uint8), 8)
    inner_edge = unknown & (cv2.dilate((~inside).astype(np.uint8),
                                       np.ones((3, 3), np.uint8)).astype(bool))
    boundary_ids = np.unique(component[inner_edge])
    boundary_connected = np.isin(component, boundary_ids[boundary_ids > 0]) if count > 1 else np.zeros_like(unknown)
    enclosed_unknown = unknown & ~boundary_connected
    fillable = enclosed_unknown & (distance <= max_distance) & (nearest_label >= 0)
    filled = labels.copy()
    filled[fillable] = nearest_label[fillable]
    return filled, {
        "maximum_refill_distance_work_px": max_distance,
        "refilled_work_pixels": int(fillable.sum()),
        "wide_unknown_pixels_preserved": int(np.count_nonzero(inside & (filled < 0))),
        "boundary_connected_unknown_pixels_never_refilled": int(boundary_connected.sum()),
        "method": "nearest supported lithology within a narrow distance band",
    }


def run_case(name: str) -> None:
    cfg = json.loads((CONFIG_ROOT / f"{name}.json").read_text(encoding="utf-8-sig"))
    labels, colours, obstacle, ambiguous_working, full_labels, full_ambiguous, colour_audit = classify_full_resolution(cfg)
    palette = np.asarray(colours, np.uint8)
    cleaned = np.full((*labels.shape, 3), 255, np.uint8)
    cleaned[obstacle] = palette[labels[obstacle]]
    yy, xx = np.indices(labels.shape)

    contact, unsupported_colour, support_audit = _supported_colour_contact(obstacle)
    # 清杂结果只显示硬接触色块；不确定候选另图保存，不能继续以原色混入清杂图。
    cleaned[:] = 255
    cleaned[contact] = palette[labels[contact]]
    uncertain_working = ambiguous_working | unsupported_colour
    uncertain_view = cleaned.copy()
    hatch = uncertain_working & (((xx + yy) % 12) < 2)
    uncertain_view[hatch & ~contact] = (165, 165, 165)

    membranes, audit = _simulate(contact)
    out_h, out_w = membranes[0].shape
    top = (out_h - cleaned.shape[0]) // 2
    left = (out_w - cleaned.shape[1]) // 2
    padded_cleaned = np.full((out_h, out_w, 3), 255, np.uint8)
    padded_cleaned[top:top + cleaned.shape[0], left:left + cleaned.shape[1]] = cleaned
    padded_obstacle = np.zeros((out_h, out_w), bool)
    padded_obstacle[top:top + contact.shape[0], left:left + contact.shape[1]] = contact
    membranes = [_fill_holes(cv2.dilate(m.astype(np.uint8),
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool))
                 for m in membranes]
    missing = [int(np.count_nonzero(padded_obstacle & ~m)) for m in membranes]
    perimeters = [_perimeter(m) for m in membranes]
    components = [_components(m)[0] for m in membranes]
    nested = all(not np.any(membranes[i + 1] & ~membranes[i]) for i in range(3))
    hole_free = all(np.array_equal(m, _fill_holes(m)) for m in membranes)
    monotonic = all(perimeters[i + 1] <= perimeters[i] + 1e-6 for i in range(3))
    valid = bool(nested and hole_free and max(components) == 1 and not any(missing) and monotonic)

    # 将最终工作轮廓映回原 ROI，并只用均匀外移吸收坐标量化误差；不焊接孤立像素。
    working_final = membranes[-1][top:top + labels.shape[0], left:left + labels.shape[1]]
    refilled_labels, refill_audit = _refill_narrow_gaps(labels, contact, working_final)
    refilled_view = np.full((*labels.shape, 3), 255, np.uint8)
    refilled_known = (refilled_labels >= 0) & working_final
    refilled_view[refilled_known] = palette[refilled_labels[refilled_known]]
    refill_contours, _ = cv2.findContours(working_final.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
    if refill_contours:
        cv2.polylines(refilled_view, [max(refill_contours, key=cv2.contourArea)], True,
                      (15, 45, 230), 3, cv2.LINE_AA)
    full_h, full_w = full_labels.shape
    full_membrane = cv2.resize(working_final.astype(np.uint8), (full_w, full_h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    # 原分辨率复核只要求包含由工作网格确认具有二维支撑的接触区域；所有被判为
    # 不确定的同色细线不参与拉膜。最近邻映射只用于建立复核带，不重做颜色判定。
    full_contact_gate = cv2.resize(contact.astype(np.uint8), (full_w, full_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
    full_obstacle = (full_labels >= 0) & full_contact_gate
    refinement_steps = 0
    while np.any(full_obstacle & ~full_membrane) and refinement_steps < 8:
        full_membrane = cv2.dilate(full_membrane.astype(np.uint8),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
        refinement_steps += 1
    missing_full = int(np.count_nonzero(full_obstacle & ~full_membrane))
    full_contours, _ = cv2.findContours(full_membrane.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
    full_cleaned = np.full((full_h, full_w, 3), 255, np.uint8)
    full_cleaned[full_obstacle] = palette[full_labels[full_obstacle]]
    fy, fx = np.indices((full_h, full_w))
    full_candidate = full_labels >= 0
    full_uncertain = full_ambiguous | (full_candidate & ~full_contact_gate)
    full_hatch = full_uncertain & (((fx + fy) % 24) < 3)
    full_uncertain_view = full_cleaned.copy()
    full_uncertain_view[full_hatch & ~full_obstacle] = (165, 165, 165)
    if full_contours:
        cv2.polylines(full_cleaned, [max(full_contours, key=cv2.contourArea)], True,
                      (15, 45, 230), max(3, round(max(full_h, full_w) / 1800)), cv2.LINE_AA)
    valid = bool(valid and audit.get("physical_process_valid", False) and missing_full == 0)
    audit.update({
        "case": name,
        "input_definition": "original-resolution tiled pixel colour classification",
        "colour_audit": colour_audit,
        "spatial_support_audit": support_audit,
        "narrow_gap_refill_audit": refill_audit,
        "posthoc_spur_removal_used": False,
        "stage_perimeters_px": [round(x, 3) for x in perimeters],
        "stage_component_counts": components,
        "nested_contraction": nested,
        "hole_free": hole_free,
        "perimeter_nonincreasing": monotonic,
        "missing_target_pixels_by_stage": missing,
        "full_resolution_refinement_steps": refinement_steps,
        "missing_confirmed_pixels_full_resolution": missing_full,
        "valid": valid,
        "experimental_only": True,
        "production_code_modified": False,
    })
    titles = [
        f"初始圆形气球｜原图分块分类 {colour_audit['original_roi_shape_hw'][1]}×{colour_audit['original_roi_shape_hw'][0]}",
        f"减压中期｜类别掩膜 {colour_audit['balloon_input_shape_hw'][1]}×{colour_audit['balloon_input_shape_hw'][0]}",
        f"低压贴合｜周长 {perimeters[2]:.0f}px",
        f"最终轮廓｜RGB分类前未缩放｜周长 {perimeters[3]:.0f}px",
    ]
    frames = [_render(padded_cleaned, membrane, title)
              for membrane, title in zip(membranes, titles)]
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    # 该图是气球工作网格上的类别显示，不能称为“原分辨率逐像素清除”。
    Image.fromarray(cleaned).save(out / "00_工作网格接触集.png")
    # 真正的清除图必须重新读取原始 ROI：已确认目标色像素保留原始 RGB，其他
    # 像素逐一置白。禁止标签降采样、整格着色、图例标准色替换和颜色回填。
    image_path = local(cfg["image"])
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        x, y, rw, rh = map(int, cfg["roi"])
        original_roi = np.asarray(source.crop((x, y, x + rw, y + rh)))
    pixel_clean = np.full_like(original_roi, 255)
    pixel_confirmed = full_labels >= 0
    pixel_clean[pixel_confirmed] = original_roi[pixel_confirmed]
    Image.fromarray(pixel_clean).save(out / "00_原分辨率分类后的清杂图.png")
    Image.fromarray(pixel_clean).save(out / "00A_原分辨率逐像素清除_保留原色.png")
    Image.fromarray(uncertain_view).save(out / "00B_不确定颜色证据.png")
    Image.fromarray(full_cleaned).save(out / "06_原分辨率最终轮廓.png")
    Image.fromarray(full_uncertain_view).save(out / "07_原分辨率不确定证据.png")
    Image.fromarray(refilled_view).save(out / "08_窄空洞回填与最终轮廓.png")
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
