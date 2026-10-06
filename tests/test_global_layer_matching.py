"""验证跨两条切断带的证据能推翻第一条带的局部较优错配。"""
import numpy as np

from stratamatch import global_layer_matching as matching


def _part(pid, x, y):
    centre = np.array([x, y], float)
    return {'id': pid, 'lith': 1, 'centre': centre,
            'axis': np.array([1., 0.]), 'sample': centre + [[-3., 0.], [3., 0.]],
            'width_profile': np.array([8., 8.])}


def _cut(pid, x):
    return {'id': pid, 'lith': -1, 'centre': np.array([x, 10.]),
            'axis': np.array([0., -1.])}


def _edge(u, v, cut, left_x, right_x, left_y, right_y):
    return {'left_part': u, 'right_part': v, 'left_index': u, 'right_index': v,
            'left_tip': 1, 'right_tip': 0, 'score': .2, 'cut_part_ids': [cut],
            'curve_work_pixels': [[left_x, left_y], [right_x, right_y]],
            'contact_evidence_sufficient': True}


def test_later_block_resolves_earlier_repeated_lithology(monkeypatch):
    ds = [_part(0, 0, 0), _part(1, 0, 20), _part(2, 20, 0),
          _part(3, 20, 20), _part(4, 40, 0), _part(5, 40, 20),
          _cut(-1, 10), _cut(-2, 30)]
    edges = [_edge(u, v, -1, 0, 20, uy, vy)
             for u, uy in ((0, 0), (1, 20))
             for v, vy in ((2, 0), (3, 20))]
    edges += [_edge(2, 4, -2, 20, 40, 0, 0),
              _edge(3, 5, -2, 20, 40, 20, 20)]
    # 长距离候选只提供闭合证据，不能单独合组。
    for u, v in ((0, 4), (1, 5)):
        edge = _edge(u, v, 0, 0, 40, 0, 0)
        edge['cut_part_ids'] = []
        edge['score'] = .1
        edge['contact_evidence_sufficient'] = False
        edges.append(edge)

    def alternatives(band, *_):
        if band['cut'] == -1:
            return [
                {'pairs': [(0, 3, .16), (1, 2, .16)], 'gain': 1.2, 'spread': 0.},
                {'pairs': [(0, 2, .21), (1, 3, .21)], 'gain': 1.1, 'spread': 0.},
            ]
        return [{'pairs': [(2, 4, .26), (3, 5, .26)],
                 'gain': 1., 'spread': 0.}]

    monkeypatch.setattr(matching, '_band_options', alternatives)
    audit = {'bands': [{'cut_part': -1, 'joint_evidence_sufficient': True},
                       {'cut_part': -2, 'joint_evidence_sufficient': True}]}
    groups, chosen, info = matching.select_global_groups(edges, ds, audit)
    assert groups[0] == groups[2] == groups[4]
    assert groups[1] == groups[3] == groups[5]
    assert groups[0] != groups[1]
    assert info['selected_bands'] == 2
