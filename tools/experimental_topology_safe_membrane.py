"""拓扑安全的放气膜实验。

该实验不接入正式识别流程。它用二值隐式区域表示气球内部，因此边界始终是
单一、无孔的闭合曲线；确定地层色块和 100% 置信线段作为不可穿透接触集。
膜先在较大曲率尺度下收缩，再根据色块间断裂距离降低尺度，以兼顾整体张力
和凹部贴合。这个实现用于替代会产生自交和尖刺的显式节点冻结模型。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[2]
SOFTWARE = PROJECT / "软件"
sys.path.insert(0, str(SOFTWARE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratamatch.staged_reconstruction import _load_and_classify  # noqa: E402


CONFIG_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "配置"
OUTPUT_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "拓扑安全放气膜实验"
CASES = ("22-P64", "23-P17")


def _font(size: int):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return count - 1, labels, stats


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """只保留最大外部闭环并填孔，保证膜没有内部回环。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask, np.uint8)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return out.astype(bool)


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """圆盘闭运算对应有限弯曲半径的膜；加边框避免图像边缘破坏广延性。"""
    radius = max(1, int(radius))
    pad = radius + 4
    work = np.pad(mask.astype(np.uint8), pad, mode="constant")
    closed = cv2.morphologyEx(work, cv2.MORPH_CLOSE, _disk(radius), borderType=cv2.BORDER_CONSTANT)
    return closed[pad:-pad, pad:-pad].astype(bool)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)).astype(bool)


def _perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(cv2.arcLength(max(contours, key=cv2.contourArea), True)) if contours else 0.0


def _contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), np.float32)
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)


def _one_component_radius(mask: np.ndarray, lower: int, upper: int) -> int:
    """寻找连接全部目标色接触块的最小尺度，不擅自删除小目标色块。"""
    n, _, _ = _components(mask)
    if n <= 1:
        return lower
    lo, hi = lower, upper
    while lo < hi:
        mid = (lo + hi) // 2
        if _components(_dilate(mask, mid))[0] <= 1:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _simulate(obstacle_full: np.ndarray, _regularity_override: int | None = None,
              _attempted_radii: tuple[int, ...] = ()) -> tuple[list[np.ndarray], dict]:
    """在降采样网格求解隐式膜，最后用原分辨率接触集做保守校正。"""
    h, w = obstacle_full.shape
    scale = min(1.0, 2400.0 / max(h, w))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    # 面积重采样后只要工作像素覆盖任一目标色像素就设为接触点，避免细薄地层
    # 在降采样时被最近邻漏掉，进而让膜穿入真实地层。
    obstacle = cv2.resize(obstacle_full.astype(np.float32), (sw, sh),
                          interpolation=cv2.INTER_AREA) > 0.0

    # 第一阶段已经完成杂项清除；此处必须逐像素保留其全部目标色结果。
    contact = obstacle

    diagonal = float(np.hypot(sw, sh))
    # 清杂接触集已经通过二维覆盖检查，基础弯曲尺度无需再承担去除网格细线的
    # 职责。使用较小尺度让外膜继续穿过白色背景向可靠色块收紧。
    base = max(5, round(diagonal * 0.0035))
    cap = max(base, round(diagonal * 0.070))
    connect_radius = _one_component_radius(contact, base, cap)
    # 轮廓必须使用“刚好能连接可靠接触集”的最小尺度。旧实现还强制使用
    # diagonal*0.015（本组数据约 41~44 px），会跨过真实白色背景并制造凸起。
    # connect_radius 已通过连通性搜索得到；不再人为放大张力半径。
    search_upper = min(cap, max(base, connect_radius, round(diagonal * 0.0075)))
    regularity_radius = min(cap, max(base, connect_radius,
                                     _regularity_override or 0))
    equilibrium = _fill_holes(_closing(contact, regularity_radius))

    # 闭运算理论上广延；离散圆盘在边缘处可能丢一圈像素，因此显式并回接触集再闭合。
    equilibrium = _fill_holes(_closing(equilibrium | contact, base))
    missing_contact = int(np.count_nonzero(contact & ~equilibrium))

    # 从膨胀态到平衡态的三个严格嵌套阶段；每阶段仍是一张无孔单连通膜。
    outer_r = max(round(diagonal * 0.16), regularity_radius * 3)
    middle_r = max(round(outer_r * 0.48), regularity_radius * 2)
    late_r = max(round(outer_r * 0.17), regularity_radius)
    # 膨胀阶段可能位于原图之外。先扩展计算域，避免轮廓被图像边界裁成悬空直线。
    domain_pad = outer_r + 8
    equilibrium_padded = np.pad(equilibrium, domain_pad, mode="constant")
    middle = _fill_holes(_dilate(equilibrium_padded, middle_r))
    late = _fill_holes(_dilate(equilibrium_padded, late_r))
    # 第一阶段是真正的圆形充气膜，而不是色块的等距膨胀轮廓。
    ys, xs = np.nonzero(middle)
    (circle_cx, circle_cy), circle_radius = cv2.minEnclosingCircle(
        np.column_stack((xs, ys)).astype(np.float32))
    circle_radius += max(6, round(diagonal * 0.012))
    # 若最小包围圆偏心，固定外边距仍可能截到圆。按圆的实际包围盒继续扩域。
    eh, ew = equilibrium_padded.shape
    extra_pad = max(0, int(np.ceil(max(circle_radius - circle_cx,
                                       circle_radius - circle_cy,
                                       circle_cx + circle_radius - (ew - 1),
                                       circle_cy + circle_radius - (eh - 1)))) + 4)
    if extra_pad:
        equilibrium_padded = np.pad(equilibrium_padded, extra_pad, mode="constant")
        middle = np.pad(middle, extra_pad, mode="constant")
        late = np.pad(late, extra_pad, mode="constant")
        circle_cx += extra_pad
        circle_cy += extra_pad
        domain_pad += extra_pad
    yy, xx = np.ogrid[:equilibrium_padded.shape[0], :equilibrium_padded.shape[1]]
    initial_circle = ((xx - circle_cx) ** 2 + (yy - circle_cy) ** 2) <= circle_radius ** 2
    stages = [initial_circle, middle, late, equilibrium_padded]

    perimeters = [_perimeter(x) / scale for x in stages]
    areas = [int(x.sum() / (scale * scale)) for x in stages]
    nesting_ok = all(not np.any(stages[i + 1] & ~stages[i]) for i in range(3))
    component_counts = [_components(x)[0] for x in stages]
    hole_free = all(np.array_equal(x, _fill_holes(x)) for x in stages)
    perimeter_nonincreasing = all(perimeters[i + 1] <= perimeters[i] + 1e-6 for i in range(3))
    geometry_valid = bool(nesting_ok and hole_free and max(component_counts) == 1 and missing_contact == 0)

    attempted_radii = _attempted_radii + (regularity_radius,)
    # 从紧到松搜索最小可行尺度。若轮廓过于贴合导致末阶段周长反增或几何约束
    # 失败，只增加少量曲率尺度重算；不同图片不再共享一个偏大的固定半径。
    if (not geometry_valid or not perimeter_nonincreasing) and regularity_radius < search_upper:
        next_radius = min(search_upper, regularity_radius + 4)
        return _simulate(obstacle_full, next_radius, attempted_radii)

    # 映回原图坐标。最近邻保持接触侧，不通过曲线插值穿过细障碍。
    full_w = round((sw + 2 * domain_pad) / scale)
    full_h = round((sh + 2 * domain_pad) / scale)
    full_stages = [cv2.resize(x.astype(np.uint8), (full_w, full_h),
                              interpolation=cv2.INTER_NEAREST).astype(bool) for x in stages]
    audit = {
        "model": "topology_safe_implicit_deflation_membrane_v2",
        "representation": "filled binary membrane with curvature-scale closing",
        "working_scale": scale,
        "working_shape_hw": [sh, sw],
        "base_curvature_radius_work_px": base,
        "contact_connect_radius_work_px": connect_radius,
        "regularity_radius_work_px": regularity_radius,
        "regularity_selection": "minimum scale that keeps the supported contact set connected",
        "regularity_search_attempted_work_px": list(attempted_radii),
        "regularity_search_upper_work_px": search_upper,
        "stage_dilation_radii_work_px": [outer_r, middle_r, late_r, 0],
        "initial_circle_centre_work_xy": [round(circle_cx, 3), round(circle_cy, 3)],
        "initial_circle_radius_work_px": round(circle_radius, 3),
        "initial_circle_extra_padding_work_px": extra_pad,
        "domain_padding_work_px": domain_pad,
        "domain_padding_full_px": round(domain_pad / scale),
        "stage_perimeters_full_px": [round(x, 3) for x in perimeters],
        "stage_areas_full_px": areas,
        "stage_component_counts": component_counts,
        "nested_contraction": nesting_ok,
        "hole_free": hole_free,
        "perimeter_nonincreasing": perimeter_nonincreasing,
        "missing_contact_work_pixels": missing_contact,
        "geometry_valid": geometry_valid,
        # 严格物理有效还要求每个观测阶段的周长均不增加；几何有效不能替代该检查。
        "physical_process_valid": bool(geometry_valid and perimeter_nonincreasing),
        "valid": bool(geometry_valid and perimeter_nonincreasing),
    }
    return full_stages, audit


def _render(cleaned: np.ndarray, membrane: np.ndarray, title: str) -> np.ndarray:
    """只显示图例地层色和白色背景，不把原图杂项以透明方式叠回。"""
    display = cleaned.copy()
    points = _contour(membrane)
    if len(points):
        cv2.polylines(display, [np.rint(points).astype(np.int32).reshape((-1, 1, 2))], True,
                      (15, 45, 230), max(3, round(max(display.shape[:2]) / 1800)), cv2.LINE_AA)
    pil = Image.fromarray(display)
    draw = ImageDraw.Draw(pil)
    size = max(24, round(max(display.shape[:2]) / 115))
    box_w = min(display.shape[1] - 20, round(size * 30))
    draw.rounded_rectangle((12, 12, box_w, 12 + size * 2), radius=9,
                           fill="white", outline=(20, 20, 20), width=2)
    draw.text((26, 20), title, fill=(10, 10, 10), font=_font(size))
    return np.asarray(pil)


def _sheet(frames: list[np.ndarray]) -> np.ndarray:
    width = 1200
    thumbs = [cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
              for f in frames]
    height = max(x.shape[0] for x in thumbs)
    out = np.full((2 * height + 24, 2 * width + 24, 3), 244, np.uint8)
    for i, thumb in enumerate(thumbs):
        x, y = (i % 2) * (width + 24), (i // 2) * (height + 24)
        out[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    return out


def run_case(name: str) -> None:
    cfg = json.loads((CONFIG_ROOT / f"{name}.json").read_text(encoding="utf-8-sig"))
    crop, labels, colours, _, _, _, allowed, _, _, stage1_audit = _load_and_classify(cfg)
    # 新流程的气球输入严格等于第一阶段保留下来的图例地层色像素。
    # 不读取旧流程置信线，也不再次按连通域面积删除细薄地层。
    obstacle = (labels >= 0) & allowed
    cleaned = np.full_like(crop, 255)
    palette = np.asarray(colours, np.uint8)
    cleaned[obstacle] = palette[labels[obstacle]]
    membranes, audit = _simulate(obstacle)
    # 将原剖面居中放入扩展计算域，使中间阶段的完整闭合膜可见。
    out_h, out_w = membranes[0].shape
    top = (out_h - crop.shape[0]) // 2
    left = (out_w - crop.shape[1]) // 2
    padded_cleaned = np.full((out_h, out_w, 3), 255, np.uint8)
    padded_cleaned[top:top + cleaned.shape[0], left:left + cleaned.shape[1]] = cleaned
    padded_obstacle = np.zeros((out_h, out_w), bool)
    padded_obstacle[top:top + obstacle.shape[0], left:left + obstacle.shape[1]] = obstacle
    # 映回原分辨率的量化误差只能通过整张膜的均匀亚像素外移补偿。禁止把漏包
    # 像素直接 OR 回膜内；后者会绕过张力约束，制造不可能的单像素突刺。
    correction_radius = 2
    membranes = [_fill_holes(_dilate(membrane, correction_radius)) for membrane in membranes]
    missing_full = [int(np.count_nonzero(padded_obstacle & ~membrane)) for membrane in membranes]
    full_perimeters = [_perimeter(membrane) for membrane in membranes]
    full_areas = [int(membrane.sum()) for membrane in membranes]
    full_components = [_components(membrane)[0] for membrane in membranes]
    full_hole_free = all(np.array_equal(membrane, _fill_holes(membrane)) for membrane in membranes)
    full_nested = all(not np.any(membranes[i + 1] & ~membranes[i]) for i in range(3))
    full_perimeter_nonincreasing = all(full_perimeters[i + 1] <= full_perimeters[i] + 1e-6
                                       for i in range(3))
    audit["stage_perimeters_full_px"] = [round(x, 3) for x in full_perimeters]
    audit["stage_areas_full_px"] = full_areas
    audit["stage_component_counts"] = full_components
    audit["nested_contraction"] = full_nested
    audit["hole_free"] = full_hole_free
    audit["perimeter_nonincreasing"] = full_perimeter_nonincreasing
    audit["missing_target_pixels_full_by_stage"] = missing_full
    audit["uniform_resampling_correction_radius_full_px"] = correction_radius
    audit["direct_pixel_union_forbidden"] = True
    audit["full_resolution_contact_contained"] = not any(missing_full)
    audit["geometry_valid"] = bool(full_nested and full_hole_free and max(full_components) == 1
                                   and not any(missing_full))
    audit["physical_process_valid"] = bool(audit["geometry_valid"] and full_perimeter_nonincreasing)
    audit["valid"] = audit["physical_process_valid"]
    titles = [
        f"初始膨胀膜｜周长 {audit['stage_perimeters_full_px'][0]:.0f}px",
        f"减压中期｜周长 {audit['stage_perimeters_full_px'][1]:.0f}px",
        f"低压贴合｜周长 {audit['stage_perimeters_full_px'][2]:.0f}px",
        f"最终总体轮廓｜周长 {audit['stage_perimeters_full_px'][3]:.0f}px",
    ]
    frames = [_render(padded_cleaned, membrane, title)
              for membrane, title in zip(membranes, titles)]
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    names = ("01_初始膨胀膜.png", "02_减压中期.png", "03_低压贴合.png", "04_最终总体轮廓.png")
    for filename, frame in zip(names, frames):
        Image.fromarray(frame).save(out / filename)
    Image.fromarray(_sheet(frames)).save(out / "00_放气过程对照.png")
    audit.update({"case": name, "experimental_only": True, "production_code_modified": False,
                  "input_definition": "stage1 legend-lithology pixels only; no trusted arcs",
                  "stage1_audit": stage1_audit,
                  "contact_pixels_full": int(obstacle.sum())})
    (out / "实验审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for case in CASES:
        print(f"processing {case}", flush=True)
        run_case(case)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
