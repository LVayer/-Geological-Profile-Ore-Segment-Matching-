"""逐对控制匹配的权重、缺层和多切断带回归测试。"""
import numpy as np

from stratamatch.band_registration import _band
from stratamatch.global_layer_matching import _progressive_band, select_progressive_groups


def part(pid, lith, x, y, width=10.):
    point = np.array([x, y], float)
    return {'id': pid, 'lith': lith, 'centre': point,
            'axis': np.array([1., 0.]),
            'sample': point + [[-4., 0.], [4., 0.]],
            'centreline': point + [[-4., 0.], [4., 0.]],
            'tangents': np.array([[1., 0.], [1., 0.]]),
            'width_profile': np.array([width, width])}


def cutter(pid, x):
    return {'id': pid, 'lith': -1, 'centre': np.array([x, 20.]),
            'axis': np.array([0., -1.])}


def edge(u, v, cut, x0, x1, y0, y1):
    return {'left_part': u, 'right_part': v, 'left_index': u,
            'right_index': v, 'left_tip': 1, 'right_tip': 0,
            'score': .12, 'individual_score': .12,
            'costs': {'morphology': .1}, 'cut_part_ids': [cut],
            'curve_work_pixels': [[x0, y0], [x1, y1]],
            'contact_evidence_sufficient': True}


def test_affected_three_layers_use_sequence_and_relative_thickness_weights():
    ds = [part(0, 9, 10, 20), cutter(-1, 10)]
    ds += [part(i, 1, 0, i*20) for i in range(1, 4)]
    ds += [part(i, 1, 20, i*20+15) for i in range(4, 7)]
    by_id = {d['id']: d for d in ds}
    edges = [edge(u, v, -1, 0, 20, u*20, (v-3)*20+15)
             for u in range(1, 4) for v in range(4, 7)]
    band = _band(-1, edges, by_id)
    proposed = _progressive_band(band, by_id, [(1, 4)], None, {})
    assert proposed
    weights = proposed[0]['features']['weights']
    assert np.isclose(weights['sequence'], .4)
    assert np.isclose(weights['relative_thickness'], .3)
    assert np.isclose(weights['corrected_geometry'], .1)
    assert np.isclose(weights['morphology'], .05)


def test_multiple_cutting_bands_add_exactly_one_pair_each_round():
    ds = [part(i, 1, x, y) for i, (x, y) in enumerate(
        [(0, 0), (0, 20), (20, 10), (20, 30), (40, 5), (40, 25)])]
    ds += [cutter(-1, 10), cutter(-2, 30)]
    edges = [edge(u, v, cut, x0, x1, y0, y1)
             for cut, x0, x1, left, right in
             [(-1, 0, 20, [(0, 0), (1, 20)], [(2, 10), (3, 30)]),
              (-2, 20, 40, [(2, 10), (3, 30)], [(4, 5), (5, 25)])]
             for u, y0 in left for v, y1 in right]
    audit = {'bands': [{'cut_part': -1, 'joint_evidence_sufficient': True},
                       {'cut_part': -2, 'joint_evidence_sufficient': True}]}
    groups, chosen, info = select_progressive_groups(edges, ds, audit)
    assert len(info['rounds']) == len(chosen)
    assert len({step['round'] for step in info['rounds']}) == len(chosen)
    assert info['selected_bands'] == 2
    assert groups[0] == groups[2] == groups[4]
    assert groups[1] == groups[3] == groups[5]
