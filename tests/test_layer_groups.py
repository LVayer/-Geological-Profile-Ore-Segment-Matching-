import numpy as np
import cv2
from stratamatch.layer_groups import recognize_layer_groups


def test_offset_cut_parts_group_without_repainting():
    labels=np.full((120,240),-1,np.int16)
    labels[35:47,10:100]=0;labels[40:52,135:225]=0
    labels[10:100,100:135]=1
    before=labels.copy()
    instances,records,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    groups=[g for g in audit['groups'] if g['lithology_index']==0]
    assert len(groups)==1 and len(groups[0]['parts'])==2
    assert np.array_equal(labels,before)
    assert np.all(instances[35:47,100:135]!=groups[0]['parts'][0])


def test_parallel_same_colour_beds_stay_separate():
    labels=np.full((120,240),-1,np.int16)
    labels[25:37,10:220]=0;labels[60:72,10:220]=0
    _,_,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    assert audit['group_count']==2


def test_equal_competing_continuations_are_not_forced():
    labels=np.full((120,240),-1,np.int16)
    labels[45:55,10:90]=0;labels[36:46,120:215]=0;labels[54:64,120:215]=0
    _,_,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    assert audit['group_count']==3


def test_different_lithology_never_groups():
    labels=np.full((100,200),-1,np.int16)
    labels[40:50,5:80]=0;labels[40:50,100:190]=1
    _,_,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    assert audit['group_count']==2 and not audit['pairs']


def test_thick_intrusion_does_not_break_group_by_fixed_gap():
    labels=np.full((180,520),-1,np.int16)
    labels[50:64,10:130]=0;labels[67:81,310:490]=0
    labels[10:150,130:310]=1
    _,_,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    groups=[g for g in audit['groups'] if g['lithology_index']==0]
    assert len(groups)==1 and len(groups[0]['parts'])==2


def test_changed_strike_can_follow_curved_continuation():
    labels=np.full((220,420),-1,np.int16)
    cv2.line(labels,(15,90),(115,90),0,14)
    cv2.line(labels,(230,110),(355,185),0,14)
    labels[:,130:215]=1
    _,_,audit=recognize_layer_groups(labels,np.ones(labels.shape,bool))
    groups=[g for g in audit['groups'] if g['lithology_index']==0]
    assert len(groups)==1
