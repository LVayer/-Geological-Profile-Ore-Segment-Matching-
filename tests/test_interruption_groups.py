import numpy as np
from stratamatch.layer_groups import recognize_layer_groups
from stratamatch.interruption_groups import interruption_weights


def test_cut_count_and_thickness_shift_evidence_weights():
    thin=interruption_weights(1,.2)
    thick=interruption_weights(1,6)
    two=interruption_weights(2,.2)
    assert thin['direction']>thin['contact']+thin['morphology']
    for weights in (thick,two):
        assert weights['contact']+weights['morphology']>weights['direction']
        assert abs(sum(weights.values())-1)<1e-8


def test_two_cut_units_restore_offset_layer_using_neighbours():
    labels=np.full((140,340),2,np.int16)
    labels[65:,:110]=3;labels[105:,220:]=3
    labels[45:65,:110]=0;labels[85:105,220:]=0
    labels[:,110:165]=1;labels[:,165:220]=4
    _,_,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    candidates=[e for e in audit['pairs'] if e.get('cut_layer_count')==2 and e.get('contact_similarity',0)>.5]
    assert candidates
    assert any(e['evidence_mode']=='band_displacement_supported' and e['decision']=='grouped' for e in candidates)
    assert any(b['matched_layers']>=3 and b['shift_spread_pixels']<1
               for b in audit['band_registration']['bands'])
    groups=[g for g in audit['groups'] if g['lithology_index']==0]
    assert len(groups)==1


def test_parallel_interlayer_is_not_counted_as_a_transverse_cut():
    labels=np.full((120,300),1,np.int16)
    labels[20:35,10:290]=0;labels[75:90,10:290]=0
    _,_,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    assert len([g for g in audit['groups'] if g['lithology_index']==0])==2
    assert not [p for p in audit['pairs'] if p.get('cut_layer_count',0)>0]
