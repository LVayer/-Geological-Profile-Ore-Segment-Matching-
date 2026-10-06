import numpy as np
from PIL import Image
from stratamatch.fault_lines import detect_annotated_fault_lines, fault_line_candidates
from stratamatch.layer_groups import recognize_layer_groups


def descriptor(pid,lith,x,y):
    line=np.array([[x-15.,y],[x+15.,y]])
    return {'id':pid,'lith':lith,'centre':np.array([x,y]),'axis':np.array([1.,0.]),
            'length':30.,'width':12.,'centreline':line,'tangents':np.array([[1.,0.],[1.,0.]]),
            'width_profile':np.array([12.,12.])}


def test_red_fault_annotation_supplies_missing_cross_gap_candidates(tmp_path):
    picture=np.full((140,240,3),255,np.uint8)
    picture[10:125,119:121]=[255,0,0]
    picture[70:73,20:22]=[255,0,0]  # A short annotation cannot become a fault.
    source=tmp_path/'fault.png';Image.fromarray(picture).save(source)
    lines,audit=detect_annotated_fault_lines(source,[0,0,240,140],(140,240))
    assert audit['merged_candidate_lines']==1
    instances=np.full((140,240),-1,np.int32)
    for y0,y1,u,v in [(12,58,1,3),(76,121,2,4)]:
        instances[y0:y1,65:115]=u;instances[y0:y1,125:175]=v
    descriptors=[descriptor(1,1,90,35),descriptor(2,2,90,98),
                 descriptor(3,1,150,35),descriptor(4,2,150,98)]
    candidates,virtual,_=fault_line_candidates(lines,instances,descriptors)
    assert len(virtual)==1
    assert {frozenset((e['left_part'],e['right_part'])) for e in candidates}=={
        frozenset((1,3)),frozenset((2,4))}
    assert all(not e['contact_evidence_sufficient'] for e in candidates)


def test_fault_line_splits_instance_without_erasing_colour_pixels():
    labels=np.full((100,200),-1,np.int16)
    labels[15:85,15:185]=1
    domain=labels>=0
    instances,records,audit=recognize_layer_groups(labels,domain,{
        'fragment_connection_radius':0,'fragment_min_area':20,
        'annotated_fault_lines':[[[100,5],[100,95]]]})
    assert len(records)==2
    assert instances[40,50]!=instances[40,150]
    assert np.array_equal(instances>=0,domain)
    assert audit['fault_barrier_pixels']>0


def test_thin_valid_layer_keeps_all_pixels_after_fault_split():
    labels=np.full((90,180),-1,np.int16)
    labels[40:44,20:160]=1
    domain=labels>=0
    instances,records,_=recognize_layer_groups(labels,domain,{
        'fragment_connection_radius':0,'fragment_min_area':25,
        'annotated_fault_lines':[[[90,10],[90,75]]]})
    assert len(records)==2
    assert np.array_equal(instances>=0,domain)
