import numpy as np
from stratamatch.band_registration import _ratio_cost, register_bands


def part(pid, lith, x, y, width=10):
    line=np.array([[x-4.,y],[x+4.,y]])
    return {'id':pid,'lith':lith,'centre':np.array([x,y],float),
            'axis':np.array([1.,0.]) if pid else np.array([0.,1.]),
            'centreline':line,'tangents':np.array([[1.,0.],[1.,0.]]),
            'width_profile':np.array([width,width],float)}


def candidate(u,v,y0,y1,cost=.15):
    return {'left_part':u,'right_part':v,'score':cost,'cut_part_ids':[0],
            'curve_work_pixels':[[-10.,float(y0)],[10.,float(y1)]],
            'weights':{'direction':.4,'position':.1},
            'costs':{'direction':.5,'position':.5},'total_cut_thickness_pixels':20.,
            'contact_evidence_sufficient':True}


def test_consistent_shift_is_jointly_selected_despite_missing_layer():
    ds=[part(0,9,0,30),part(1,1,-10,0),part(2,2,-10,20),part(3,3,-10,40),
        part(4,1,10,30),part(5,3,10,70)]
    edges=[candidate(1,4,0,30),candidate(3,5,40,70)]
    audit=register_bands(edges,ds)
    assert audit['bands'][0]['matched_layers']==2
    assert abs(abs(audit['bands'][0]['apparent_shift_along_cut_pixels'])-30)<1
    assert all(e['band_registration']['selected'] for e in edges)


def test_relative_thickness_uses_paired_same_lithology_reference():
    ds=[part(0,9,0,0),part(1,2,-10,0,20),part(2,2,-10,20,10),
        part(3,2,10,30,30),part(4,2,10,50,15)]
    by_id={d['id']:d for d in ds}
    positions={('left',1):np.array([-10.,0.]),('left',2):np.array([-10.,20.]),
               ('right',3):np.array([10.,30.]),('right',4):np.array([10.,50.])}
    band={'positions':positions,'axis':np.array([0.,1.])}
    assert _ratio_cost(1,3,[(2,4)],band,by_id)<1e-9
    by_id[3]['width_profile'][:]=12
    assert _ratio_cost(1,3,[(2,4)],band,by_id)>.5
    assert _ratio_cost(1,3,[],band,by_id) is None


def test_whole_band_beats_cheaper_inconsistent_repeated_lithology_pairs():
    ds=[part(0,9,0,35)]
    ds += [part(i,1,-10,(i-1)*20) for i in range(1,4)]
    ds += [part(i,1,10,30+(i-4)*20) for i in range(4,7)]
    edges=[candidate(i,j,(i-1)*20,30+(j-4)*20,
                     .30 if j==i+3 else .12)
           for i in range(1,4) for j in range(4,7)]
    audit=register_bands(edges,ds)
    selected={(e['left_part'],e['right_part']) for e in edges
              if e['band_registration']['selected']}
    assert selected=={(1,4),(2,5),(3,6)}
    assert abs(abs(audit['bands'][0]['apparent_shift_along_cut_pixels'])-30)<1


def test_one_pair_cannot_establish_shared_displacement():
    ds=[part(0,9,0,20),part(1,1,-10,0),part(2,2,-10,40),
        part(3,1,10,30),part(4,2,10,70)]
    edge=candidate(1,3,0,30,.01)
    weak=candidate(2,4,40,70,1.5)
    audit=register_bands([edge,weak],ds)
    assert audit['bands'][0]['matched_layers']==1
    assert not audit['bands'][0]['joint_evidence_sufficient']
    assert 'band_registration' not in edge
