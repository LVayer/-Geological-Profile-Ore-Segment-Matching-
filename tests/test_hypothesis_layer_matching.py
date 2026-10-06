"""验证单条切断层的序列隔离、完整权重及跨两条切断带的逐对选择。"""
import numpy as np

from stratamatch.band_registration import _band
from stratamatch.hypothesis_layer_matching import (
    _band_options, _bridge_options, select_hypothesis_groups)

from test_adaptive_progressive import cutter, edge, part


def test_one_cut_uses_the_requested_four_weights():
    ds = [part(i, 1, x, y) for i, (x, y) in enumerate(
        [(0, 0), (0, 20), (20, 10), (20, 30)])] + [cutter(-1, 10)]
    by_id = {d['id']: d for d in ds}
    edges = [edge(u, v, -1, 0, 20, y0, y1)
             for u, y0 in [(0, 0), (1, 20)]
             for v, y1 in [(2, 10), (3, 30)]]
    band = _band(-1, edges, by_id)
    options = _band_options(band, by_id, [], None, {})
    assert options
    feature = next(iter(options[0]['features'].values()))
    assert feature['relative_thickness'] is None
    assert feature['weights'] == {
        'sequence': .40, 'relative_thickness': .30,
        'corrected_geometry': .25, 'morphology': .05}
    assert np.isclose(sum(feature['weights'].values()), 1.)


def test_global_alternatives_resolve_two_cuts_one_pair_per_round():
    ds = [part(i, 1, x, y) for i, (x, y) in enumerate(
        [(0, 0), (0, 20), (20, 10), (20, 30), (40, 5), (40, 25)])]
    ds += [cutter(-1, 10), cutter(-2, 30)]
    edges = [edge(u, v, cut, x0, x1, y0, y1)
             for cut, x0, x1, left, right in
             [(-1, 0, 20, [(0, 0), (1, 20)], [(2, 10), (3, 30)]),
              (-2, 20, 40, [(2, 10), (3, 30)], [(4, 5), (5, 25)])]
             for u, y0 in left for v, y1 in right]
    groups, selected, audit = select_hypothesis_groups(edges, ds, {})
    assert len(audit['rounds']) == len(selected) == 4
    assert groups[0] == groups[2] == groups[4]
    assert groups[1] == groups[3] == groups[5]
    assert groups[0] != groups[1]


def test_multiple_cut_path_can_use_two_confirmed_neighbour_layers():
    ds = [part(i, lith, x, y) for i, (lith, x, y) in enumerate(
        [(1, 0, 0), (2, 0, 20), (3, 0, 40),
         (2, 20, 20), (3, 20, 40), (1, 20, 0)])]
    ds.append(cutter(-1, 10))
    edges = [edge(1, 3, -1, 0, 20, 20, 20),
             edge(2, 4, -1, 0, 20, 40, 40)]
    bridge = edge(0, 5, -1, 0, 20, 0, 0)
    bridge['cut_part_ids'] = [-1, -2]
    bridge['score'] = .40
    bridge['band_unselected'] = True
    bridge['sequence_alignment'] = {'matched_layers': [
        {'left': 1, 'right': 3, 'axis': 'normal'},
        {'left': 2, 'right': 4, 'axis': 'normal'}]}
    edges.append(bridge)
    groups, selected, audit = select_hypothesis_groups(edges, ds, {})
    assert groups[1] == groups[3]
    assert groups[2] == groups[4]
    assert groups[0] == groups[5]
    assert bridge['multi_cut_sequence_support'] == 2
    assert bridge['decision'] == 'grouped'


def test_long_bridge_is_softly_penalised_and_controls_reduce_penalty():
    ds = [part(i, lith, x, y) for i, (lith, x, y) in enumerate(
        [(1, 0, 0), (1, 20, 0), (2, 0, 20), (2, 20, 20)])]
    by_id = {d['id']: d for d in ds}
    bridge = edge(0, 1, -1, 0, 20, 0, 0)
    bridge['contact_similarity'] = .5
    bridge['neighbour_confidence'] = .6
    bridge['sequence_alignment'] = {'matched_layers': [
        {'left': 2, 'right': 3, 'axis': 'normal'}]}
    bridge['cut_part_ids'] = list(range(-1, -13, -1))
    long_cost = _bridge_options([bridge], by_id, [], {})[0]['pairs'][0][2]
    bridge['cut_part_ids'] = [-1, -2]
    short_cost = _bridge_options([bridge], by_id, [], {})[0]['pairs'][0][2]
    assert long_cost > short_cost
    bridge['cut_part_ids'] = list(range(-1, -13, -1))
    supported_cost = _bridge_options([bridge], by_id, [(2, 3)], {})[0]['pairs'][0][2]
    assert supported_cost < long_cost
