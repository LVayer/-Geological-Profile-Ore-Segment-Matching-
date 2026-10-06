"""边界梯度方向和沿断层缓变错移的关键行为。"""
import numpy as np

from stratamatch.band_registration import _band, _solve
from stratamatch.global_layer_matching import _registration_models
from stratamatch.gradient_orientation import GradientOrientation


def _part(pid, x, y):
    point = np.array([float(x), float(y)])
    return {'id': pid, 'lith': 1, 'centre': point,
            'axis': np.array([1., 0.]), 'centreline': point+[
                [-4., 0.], [4., 0.]],
            'tangents': np.array([[1., 0.], [1., 0.]]),
            'width_profile': np.array([9., 9.])}


def test_gradient_follows_horizontal_layer_boundary():
    labels = np.full((110, 150), -1, np.int16)
    labels[40:65, 15:135] = 1
    field = GradientOrientation(labels)
    tangent, quality = field.sample(_part(1, 70, 50), [70, 40])
    assert quality > .3
    assert abs(float(tangent @ np.array([1., 0.]))) > .9


def test_affine_displacement_is_candidate_when_shift_changes_along_cut():
    cut = {'id': -1, 'axis': np.array([0., -1.]),
           'centre': np.array([0., 50.])}
    left_y = [5., 35., 65., 95.]
    right_y = [20.+1.3*y for y in left_y]
    parts = [_part(i, -12, y) for i, y in enumerate(left_y)]
    parts += [_part(i+4, 12, y) for i, y in enumerate(right_y)]
    by_id = {p['id']: p for p in [cut, *parts]}
    edges = []
    for i, y in enumerate(left_y):
        for j, yy in enumerate(right_y):
            edges.append({'left_part': i, 'right_part': j+4,
                          'score': .12 if i == j else .5,
                          'curve_work_pixels': [[-12., y], [12., yy]]})
    band = _band(-1, edges, by_id)
    models = _registration_models(band, by_id)
    assert any(abs(slope-.3) < .06 for _, slope in models)
    # 与整套最优常数位移相比，允许缓变错移应降低序列代价。
    affine_cost = min(_solve(band, lambda p, a=a, b=b:
                             a+b*float((p-band['centre'])@band['axis']),
                             [], by_id, .38)[0]
                      for a, b in models if abs(b) > .1)
    constant_cost = min(_solve(band, a, [], by_id, .38)[0]
                        for a, b in models if abs(b) < 1e-8)
    assert affine_cost < constant_cost
