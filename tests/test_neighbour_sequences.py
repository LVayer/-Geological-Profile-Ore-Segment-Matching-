import numpy as np
from stratamatch.neighbour_sequences import (align_costs, compare_neighbourhoods,
    short_axis_penalty, joint_band_ranking, neighbourhood)


def test_alignment_skips_missing_layer_and_preserves_order():
    a, b = [1,2,3,1], [1,3,1]
    costs = np.array([[0. if x == y else 1.2 for y in b] for x in a])
    cost, pairs = align_costs(costs)
    assert pairs == [(0,0),(2,1),(3,2)]
    assert abs(cost-.32) < 1e-8


def signature(sequence):
    return [{'axis':axis, 'side':side, 'layers':[
        {'lith':lith, 'part':i, 'relative_width':1., 'angle':0., 'quality':.8}
        for i,lith in enumerate(sequence)] if axis == 'normal' and side == 1 else []}
        for axis in ('normal','tangent') for side in (-1,1)]


def test_missing_neighbour_is_tolerated_but_wrong_order_loses_support():
    a = signature([1,2,3,4])
    missing = compare_neighbourhoods(a, signature([1,3,4]))
    reversed_order = compare_neighbourhoods(a, signature([4,3,2,1]))
    assert missing[2] > reversed_order[2]
    assert missing[3] > reversed_order[3]
    assert len(missing[4]['matched_layers']) == 3
    empty = compare_neighbourhoods(signature([]), signature([]))
    assert empty[2] == empty[3] == 0


def descriptor(width):
    return {'width_profile':np.array([width]), 'centreline':np.array([[0.,0.],[100.,0.]]),
            'axis':np.array([1.,0.])}


def test_short_axis_penalty_decreases_with_width_length_ratio():
    thin, thick = descriptor(5.), descriptor(80.)
    thin_side, _ = short_axis_penalty(thin, thin, np.array([0.,100.]))
    thick_side, _ = short_axis_penalty(thick, thick, np.array([0.,100.]))
    along, _ = short_axis_penalty(thin, thin, np.array([100.,0.]))
    assert thin_side > thick_side > along


def test_joint_band_prefers_consistent_order_with_missing_layer():
    ds = [{'id':0, 'axis':np.array([0.,1.]), 'centre':np.array([0.,0.])}]
    for pid,x,y in [(1,-10,0),(2,-10,10),(3,-10,20),(4,10,0),(5,10,20)]:
        ds.append({'id':pid, 'centre':np.array([x,y],float)})
    pairs = [(1,4,.25),(1,5,.24),(3,4,.24),(3,5,.25)]
    edges = [{'left_part':a,'right_part':b,'score':s,'cut_part_ids':[0]} for a,b,s in pairs]
    joint_band_ranking(edges,ds)
    scores = {(e['left_part'],e['right_part']):e['score'] for e in edges}
    assert scores[1,4] < scores[1,5]
    assert scores[3,5] < scores[3,4]


def test_alignment_can_leave_all_candidates_unmatched():
    _, pairs = align_costs(np.full((3,2),1.2))
    assert pairs == []


def test_neighbourhood_reads_beyond_direct_contact_and_stops_at_cut():
    ds = [{'id':i,'lith':i,'centre':np.array([0.,10.*i]),'axis':np.array([1.,0.])}
          for i in range(5)]
    graph = {i:{} for i in range(5)}
    for i in range(4):
        edge = {'point':np.array([0.,10.*i+5.]),'length':100}
        graph[i][i+1] = edge; graph[i+1][i] = edge
    frame = lambda d,p:(np.array([1.,0.]),10.)
    angle = lambda a,b:0.
    chains = neighbourhood(0,{4},graph,ds,frame,angle)
    chain = next(c for c in chains if c['axis']=='normal' and c['side']==1)
    assert [e['part'] for e in chain['layers']] == [1,2,3]
