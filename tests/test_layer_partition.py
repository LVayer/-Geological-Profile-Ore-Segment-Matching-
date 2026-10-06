import numpy as np

from stratamatch.layer_partition import (detect_fault_mask, _colour_components,
                                         classify_unassigned_by_neighbours,
                                         _render_colour_checked_boundaries,
                                         _micro_water_cleanup,
                                         smooth_filled_boundaries)


def _config():
    return {'legend_catalog':{'items':[
        {'id':'fault','name':'断层及编号','category':'geological_boundary','confidence':.95,
         'line':{'present':True,'dominant_rgb':[237,23,23],'mean_width_px':3}},
        {'id':'other','name':'其他红橙线','category':'other','confidence':.9,
         'line':{'present':True,'dominant_rgb':[222,103,0],'mean_width_px':3}},
    ]}}


def test_only_strict_fault_colour_can_split_a_complete_path():
    rgb=np.full((90,140,3),[180,210,80],np.uint8)
    rgb[:,68:72]=[237,23,23]       # complete fault
    rgb[20:24,:45]=[222,103,0]     # competing legend colour
    fault,audit=detect_fault_mask(rgb,_config(),[np.array([180,210,80]),np.array([222,103,0])],(1.,1.))
    assert audit['enabled'] and np.any(fault[:,69])
    assert not np.any(fault[21,:45])
    labels=np.zeros((90,140),np.int16);domain=np.ones_like(labels,bool)
    _,records=_colour_components(labels,domain,fault,12)
    assert len(records)==2


def test_incomplete_fault_does_not_split_a_layer():
    labels=np.zeros((90,140),np.int16);domain=np.ones_like(labels,bool)
    partial=np.zeros_like(domain);partial[:55,69:72]=True
    _,records=_colour_components(labels,domain,partial,12)
    assert len(records)==1


def test_unassigned_component_uses_immediate_ring_without_merging_colour_classes():
    labels=np.zeros((30,40),np.int16);domain=np.ones_like(labels,bool)
    labels[:,25:]=1
    labels[8:14,8:14]=-1                 # surrounded only by class 0
    labels[17:22,24:28]=-1               # touches both, class 1 has longer ring
    filled,audit=classify_unassigned_by_neighbours(labels,domain)
    assert np.all(filled[8:14,8:14]==0)
    assert np.all(filled[17:22,24:28]==1)
    assert audit['same_colour_components']==1
    assert audit['majority_colour_components']==1
    assert audit['unresolved_pixels']==0


def test_boundary_is_cancelled_for_equal_side_colours():
    labels=np.zeros((20,30),np.int16);domain=np.ones_like(labels,bool)
    instances=np.zeros_like(labels,np.int32)
    image=_render_colour_checked_boundaries(labels,instances,[[230,230,230],[30,90,160]],domain,2)
    assert np.all(image[10,15]==[230,230,230])
    labels[:,15:]=1
    instances[:,15:]=1
    image=_render_colour_checked_boundaries(labels,instances,[[230,230,230],[30,90,160]],domain,2)
    assert np.all(image[10,15]==0)


def test_micro_fragment_without_source_colour_is_dissolved_on_same_fault_side():
    crop=np.full((40,50,3),[205,180,120],np.uint8)
    labels=np.zeros((40,50),np.int16)
    labels[18:21,24:27]=1
    domain=np.ones_like(labels,bool);fault=np.zeros_like(domain)
    cleaned,audit=_micro_water_cleanup(
        crop,labels,domain,fault,
        [np.array([205,180,120]),np.array([40,80,160])],minimum_area=1)
    assert not np.any(cleaned==1)
    decision=next(item for item in audit['details'] if item['area_pixels']==9)
    assert decision['decision']=='partition_no_source_support'
    assert decision['fault_side_partitioned_pixels']==9


def test_boundary_smoothing_removes_burr_but_preserves_thin_continuity_and_fault_edge():
    labels=np.zeros((30,40),np.int16);labels[:,20:]=1
    labels[14,19]=1                         # one-pixel boundary burr
    labels[5:25,7]=1                        # genuine one-pixel continuous bed
    labels[10,19]=1                         # protected because it touches fault
    domain=np.ones_like(labels,bool);fault=np.zeros_like(domain)
    fault[10,20]=True
    smoothed,audit=smooth_filled_boundaries(labels,domain,fault,passes=2)
    assert smoothed[14,19]==0
    assert np.all(smoothed[5:25,7]==1)
    assert smoothed[10,19]==1
    assert audit['changed_pixels_total']>=1
