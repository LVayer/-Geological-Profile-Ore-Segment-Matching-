"""Physical deflation-snake experiment, isolated from the production pipeline.

The model follows the parametric active-contour equation used by Kass et al.
and Cohen's balloon model:

    dC/dt = alpha*C_ss - beta*C_ssss + pressure*N + contact/image forces

The initial curve is a genuinely free 2-D contour.  Nodes are re-sampled at an
arc-length spacing of at most two working pixels, as recommended by Cohen &
Cohen (PAMI 1993), so an edge cannot be skipped by a long contour segment.
Retained lithology pixels and trusted outer arcs form a hard contact surface.

This file is an experiment only; it is not imported by application code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[2]
SOFTWARE = PROJECT / "软件"
sys.path.insert(0, str(SOFTWARE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratamatch.staged_reconstruction import _load_and_classify, reconstruct_confident_outline  # noqa: E402
from experimental_balloon_shrinkwrap import _significant_support  # noqa: E402


CONFIG_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "配置"
OUTPUT_ROOT = PROJECT / "输出数据" / "总体轮廓测验" / "物理放气气球实验"
CASES = ("22-P64", "23-P17")


def _font(size: int):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _lengths(points: np.ndarray) -> tuple[np.ndarray, float]:
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    return lengths, float(lengths.sum())


def _resample_closed(points: np.ndarray, spacing: float = 0.8) -> np.ndarray:
    """Equal-arc-length nodes; no segment is deliberately left longer than 2 px."""
    edge_lengths, perimeter = _lengths(points)
    count = max(64, int(np.ceil(perimeter / spacing)))
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    closed = np.vstack((points, points[0]))
    samples = np.linspace(0.0, perimeter, count, endpoint=False)
    index = np.searchsorted(cumulative, samples, side="right") - 1
    index = np.clip(index, 0, len(points) - 1)
    denominator = np.maximum(edge_lengths[index], 1e-6)
    fraction = ((samples - cumulative[index]) / denominator)[:, None]
    return (closed[index] + fraction * (closed[index + 1] - closed[index])).astype(np.float32)


def _normals(points: np.ndarray) -> np.ndarray:
    """Inward unit normal for the clockwise image-coordinate contour."""
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    norm = np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    tangent /= norm
    return np.column_stack((-tangent[:, 1], tangent[:, 0])).astype(np.float32)


def _sample_field(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, field.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, field.shape[0] - 1)
    return field[y, x]


def _initial_circle(obstacle: np.ndarray, margin: float = 55.0) -> tuple[np.ndarray, tuple[float, float], float]:
    ys, xs = np.nonzero(obstacle)
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    radius = float(np.max(np.hypot(xs - cx, ys - cy)) + margin)
    count = max(128, int(np.ceil(2 * np.pi * radius / 2.0)))
    theta = np.arange(count, dtype=np.float32) * (2 * np.pi / count)
    points = np.column_stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta))).astype(np.float32)
    return points, (cx, cy), radius


def _simulate(obstacle: np.ndarray, maximum_iterations: int = 2300) -> tuple[list[np.ndarray], dict]:
    """Evolve a dense free contour under pressure, tension, rigidity and contact."""
    # Two-pixel contact shell plus subpixel motion prevents the curve from
    # tunnelling through a one-pixel trusted arc.
    hard = cv2.dilate(obstacle.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    free = (~hard).astype(np.uint8)
    distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    points, centre, radius = _initial_circle(hard)
    points = _resample_closed(points, 0.8)
    initial = points.copy()
    snapshots = [initial.copy()]
    snapshot_iterations = []
    history = []
    rejected_topology = 0
    local_self_contact_events = 0
    stable = 0

    # Coefficients are expressed for ~2 px node spacing.  Pressure is clipped
    # below one pixel/iteration, so even the thinnest hard barrier cannot be
    # crossed between collision tests.
    alpha, beta, pressure = 0.20, 0.018, 0.70
    for iteration in range(1, maximum_iterations + 1):
        # Reparameterise every step. A nominal 1 px interval leaves enough
        # margin for differential contact motion while keeping every resulting
        # segment below the two-pixel limit.
        points = _resample_closed(points, 0.8)
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)
        laplacian = previous + following - 2 * points
        biharmonic = (np.roll(points, 2, axis=0) - 4 * previous + 6 * points
                      - 4 * following + np.roll(points, -2, axis=0))
        normal = _normals(points)
        local_distance = _sample_field(distance, points)
        contact = local_distance <= 2.2
        # Pressure switches off locally at contact. Internal elastic forces may
        # still move a node tangentially or away from the obstacle.
        force = alpha * laplacian - beta * biharmonic
        force += (pressure * (~contact).astype(np.float32))[:, None] * normal
        magnitude = np.linalg.norm(force, axis=1)
        scale = np.minimum(1.0, 0.50 / np.maximum(magnitude, 1e-6))
        proposal = points + force * scale[:, None]

        # Hard surface collision. A midpoint test covers diagonal movement and
        # rejects the entire local step rather than projecting through a wall.
        midpoint = 0.5 * (points + proposal)
        blocked = _sample_field(hard, proposal) | _sample_field(hard, midpoint)
        proposal[blocked] = points[blocked]

        # A dense curve can only cross itself after two non-neighbouring arcs
        # approach within a few pixels. Treat this as membrane self-contact:
        # freeze those local nodes and their immediate elastic neighbourhood,
        # while allowing the rest of the balloon to continue deflating.
        pairs = cKDTree(proposal).query_pairs(r=2.35, output_type="ndarray")
        if len(pairs):
            separation = np.abs(pairs[:, 0] - pairs[:, 1])
            cyclic = np.minimum(separation, len(proposal) - separation)
            nonlocal_pairs = pairs[cyclic > 5]
            if len(nonlocal_pairs):
                bad = np.unique(nonlocal_pairs.ravel())
                expanded = np.unique(np.concatenate([(bad + offset) % len(proposal)
                                                      for offset in range(-5, 6)]))
                proposal[expanded] = points[expanded]
                local_self_contact_events += int(len(nonlocal_pairs))

        # A deflating contour should not reverse its global orientation. Large
        # signed-area reversal is a cheap self-intersection/collapse safeguard;
        # the dense nodes and subpixel step handle local crossings.
        x, y = proposal[:, 0], proposal[:, 1]
        signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if signed_area <= 0:
            proposal = points
            rejected_topology += 1

        motion = np.linalg.norm(proposal - points, axis=1)
        points = proposal.astype(np.float32)
        _, perimeter = _lengths(points)
        history.append({
            "iteration": iteration,
            "perimeter_px": perimeter,
            "mean_motion_px": float(motion.mean()),
            "max_motion_px": float(motion.max(initial=0.0)),
            "contact_fraction": float(contact.mean()),
            "nodes": len(points),
        })
        if iteration in (500, 1100):
            snapshots.append(points.copy())
            snapshot_iterations.append(iteration)
        if iteration > 900 and float(motion.mean()) < 0.012:
            stable += 1
            if stable >= 35:
                break
        else:
            stable = 0

    while len(snapshots) < 3:
        snapshots.append(points.copy())
    snapshots.append(points.copy())
    segment_lengths, final_perimeter = _lengths(points)
    audit = {
        "model": "parametric_deflation_snake_free_xy_v1",
        "equation": "C_t = alpha*C_ss - beta*C_ssss + pressure*N + hard_contact",
        "iterations": iteration,
        "initial_radius_px": round(radius, 3),
        "initial_nodes": len(initial),
        "final_nodes": len(points),
        "maximum_final_segment_px": round(float(segment_lengths.max(initial=0.0)), 4),
        "mean_final_segment_px": round(float(segment_lengths.mean()), 4),
        "initial_perimeter_px": round(_lengths(initial)[1], 3),
        "final_perimeter_px": round(final_perimeter, 3),
        "perimeter_reduction_fraction": round(1 - final_perimeter / _lengths(initial)[1], 6),
        "rejected_topology_steps": rejected_topology,
        "coefficients": {"tension_alpha": alpha, "rigidity_beta": beta,
                         "inward_pressure": pressure, "max_step_px": 0.50,
                         "target_node_spacing_px": 0.8},
        "local_self_contact_events": local_self_contact_events,
        "checkpoints": [history[0], *[history[min(i - 1, len(history) - 1)] for i in snapshot_iterations], history[-1]],
        "history_stride_10": history[::10],
        "centre_xy": [round(centre[0], 3), round(centre[1], 3)],
    }
    return snapshots[:4], audit


def _render(crop: np.ndarray, obstacle: np.ndarray, contour: np.ndarray,
            centre: tuple[float, float], radius: float, title: str) -> np.ndarray:
    side = min(3200, int(np.ceil(2 * radius + 100)))
    display_scale = side / (2 * radius + 100)
    canvas = np.full((side, side, 3), 255, np.uint8)
    faint = crop.copy()
    faint[~obstacle] = np.rint(0.18 * faint[~obstacle] + 0.82 * 255).astype(np.uint8)
    small = cv2.resize(faint, (round(crop.shape[1] * display_scale), round(crop.shape[0] * display_scale)),
                       interpolation=cv2.INTER_AREA)
    ox = round(side / 2 - centre[0] * display_scale)
    oy = round(side / 2 - centre[1] * display_scale)
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(side, ox + small.shape[1]), min(side, oy + small.shape[0])
    canvas[y0:y1, x0:x1] = small[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    draw_points = np.rint(np.column_stack((side / 2 + (contour[:, 0] - centre[0]) * display_scale,
                                             side / 2 + (contour[:, 1] - centre[1]) * display_scale))).astype(np.int32)
    cv2.polylines(canvas, [draw_points.reshape((-1, 1, 2))], True, (15, 45, 230), 5, cv2.LINE_AA)
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.rounded_rectangle((20, 18, 920, 90), radius=12, fill="white", outline=(20, 20, 20), width=2)
    draw.text((38, 31), title, fill=(10, 10, 10), font=_font(32))
    return np.asarray(pil)


def _sheet(frames: list[np.ndarray]) -> np.ndarray:
    width = 1100
    thumbs = [cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])),
                         interpolation=cv2.INTER_AREA) for frame in frames]
    height = max(x.shape[0] for x in thumbs)
    out = np.full((2 * height + 30, 2 * width + 30, 3), 242, np.uint8)
    for i, thumb in enumerate(thumbs):
        x, y = (i % 2) * (width + 30), (i // 2) * (height + 30)
        out[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    return out


def run_case(name: str) -> None:
    cfg = json.loads((CONFIG_ROOT / f"{name}.json").read_text(encoding="utf-8-sig"))
    crop, labels, _, _, _, _, allowed, _, _, _ = _load_and_classify(cfg)
    _, trusted, _ = reconstruct_confident_outline(cfg)
    support = _significant_support(labels, allowed)
    obstacle = support | cv2.dilate(trusted.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    contours, audit = _simulate(obstacle)
    centre = tuple(audit["centre_xy"])
    radius = float(audit["initial_radius_px"])
    checkpoint = audit["checkpoints"]
    titles = [
        f"初始充气圆｜节点{audit['initial_nodes']}｜周长{audit['initial_perimeter_px']:.0f}px",
        f"自由放气第{checkpoint[1]['iteration']}步｜接触{checkpoint[1]['contact_fraction']:.1%}｜周长{checkpoint[1]['perimeter_px']:.0f}px",
        f"自由放气第{checkpoint[2]['iteration']}步｜接触{checkpoint[2]['contact_fraction']:.1%}｜周长{checkpoint[2]['perimeter_px']:.0f}px",
        f"最终｜节点{audit['final_nodes']}｜最大分段{audit['maximum_final_segment_px']:.2f}px｜周长{audit['final_perimeter_px']:.0f}px",
    ]
    frames = [_render(crop, obstacle, contour, centre, radius, title)
              for contour, title in zip(contours, titles)]
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()
    names = ("01_初始充气圆.png", "02_自由放气中期.png", "03_自由放气后期.png", "04_最终总体轮廓.png")
    for filename, frame in zip(names, frames):
        Image.fromarray(frame).save(out / filename)
    Image.fromarray(_sheet(frames)).save(out / "00_物理放气过程对照.png")
    audit.update({"case": name, "experimental_only": True, "production_code_modified": False,
                  "support_pixels": int(support.sum()), "trusted_arc_pixels": int(trusted.sum()),
                  "hard_contact_pixels": int(obstacle.sum())})
    (out / "实验审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for case in CASES:
        run_case(case)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
