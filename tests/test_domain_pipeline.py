import numpy as np
from PIL import Image,ImageDraw

from stratamatch.artifacts import (classify_line_paths,reconnect_artifact_corridors,
                                   expand_confirmed_corridors,inpaint_artifact_labels,
                                   classify_catalogue_strokes)
from stratamatch.domain import infer_domain
from stratamatch.recognition import recognize


def test_complete_path_decision_separates_interface_from_technical_line():
    interface_labels=np.zeros((100,120),np.int16);interface_labels[:,60:]=1
    vertical=np.zeros(interface_labels.shape,bool);vertical[:,58:62]=True
    artifact,boundary,uncertain,audit=classify_line_paths(interface_labels,vertical,10)
    assert boundary.sum()>artifact.sum() and audit['boundary_paths']>=1

    uniform=np.zeros((100,120),np.int16)
    technical=np.zeros(uniform.shape,bool);technical[48:52,5:115]=True
    artifact,boundary,uncertain,audit=classify_line_paths(uniform,technical,10)
    assert artifact.sum()>boundary.sum() and audit['artifact_paths']>=1


def test_corridor_repair_reconnects_only_unique_lithology_proposals():
    labels=np.full((80,100),-1,np.int16)
    labels[10:35,:48]=0;labels[10:35,53:]=0
    labels[45:70,:48]=1;labels[45:70,53:]=1
    corridor=np.zeros(labels.shape,bool);corridor[:,48:53]=True
    repaired,audit=reconnect_artifact_corridors(labels,corridor,max_gap=8,angle_step=15)
    assert np.all(repaired[15:30,48:53]==0)
    assert np.all(repaired[50:65,48:53]==1)
    assert np.all(repaired[36:44,48:53]<0)
    assert audit['recovered_pixels']>0


def test_corridor_repair_never_crosses_unconfirmed_blank_space():
    labels=np.full((40,80),-1,np.int16)
    labels[10:30,:30]=0;labels[10:30,50:]=0
    corridor=np.zeros(labels.shape,bool);corridor[10:30,38:42]=True
    repaired,audit=reconnect_artifact_corridors(labels,corridor,max_gap=24)
    assert np.all(repaired[:,30:38]<0)
    assert np.all(repaired[:,42:50]<0)
    assert audit['recovered_pixels']==0


def test_broad_borehole_expansion_is_pale_and_width_bounded():
    rgb=np.full((60,100,3),(35,120,180),np.uint8)
    rgb[:,43:57]=245
    centre=np.zeros((60,100),bool);centre[:,49:51]=True
    expanded,audit=expand_confirmed_corridors(rgb,centre,max_half_width=10)
    assert np.all(expanded[:,43:57])
    assert not np.any(expanded[:,:38]) and not np.any(expanded[:,62:])
    assert audit['added_pixels']>0


def test_artifact_inpaint_fills_broad_split_and_preserves_class_seam():
    labels=np.full((60,100),-1,np.int16)
    labels[:28,:43]=0;labels[:28,57:]=0
    labels[32:,:43]=1;labels[32:,57:]=1
    corridor=np.zeros(labels.shape,bool);corridor[:,43:57]=True
    repaired,seam,audit=inpaint_artifact_labels(labels,corridor,[0,1])
    assert np.all(repaired[:28,43:57]==0)
    assert np.all(repaired[32:,43:57]==1)
    assert np.all(repaired[:,43:57]>=0)
    assert seam.any() and audit['unresolved_pixels']==0


def test_domain_is_built_from_cleaned_support_and_fills_enclosed_white():
    labels=np.full((140,180),-1,np.int16)
    labels[25:115,25:155]=0;labels[50:90,65:115]=-1
    prior=np.ones(labels.shape,bool)
    domain,audit=infer_domain(labels,prior,(),np.zeros(labels.shape,bool))
    assert domain[30,30] and domain[70,90]       # enclosed white region retained
    assert not domain[5,5]                       # remote page excluded
    assert audit['excluded_prior_pixels']>0


def test_legend_colour_precedes_topology_for_fault_and_technical_line():
    rgb=np.full((180,220,3),(190,175,130),np.uint8)
    labels=np.zeros((180,220),np.int16);labels[:,110:]=1
    # Red fault lies on a true class change; magenta section line crosses it.
    rgb[:,107:113]=(237,23,23)
    rgb[88:96,15:205]=(255,35,253)
    catalogue={'items':[
        {'id':'G','name':'fault','category':'geological_boundary','subtype':'line','confidence':.95,
         'line':{'present':True,'dominant_rgb':[237,23,23],'mean_width_px':6,'shape':'vertical_straight'}},
        {'id':'T','name':'section line','category':'other','subtype':'line','confidence':.95,
         'line':{'present':True,'dominant_rgb':[255,35,253],'mean_width_px':8,'shape':'horizontal_straight'}}]}
    geological,technical,uncertain,audit=classify_catalogue_strokes(rgb,labels,catalogue,1.,14,10)
    assert geological[:,108:112].sum()>300
    assert technical[89:95,20:200].sum()>500
    assert audit['prototype_count']==2


def test_colour_path_continues_through_label_but_does_not_consume_same_colour_stratum():
    rgb=np.full((180,260,3),(185,180,145),np.uint8);labels=np.zeros((180,260),np.int16)
    magenta=(255,35,253)
    rgb[35:43,15:235]=magenta;rgb[30:49,112:130]=(0,0,0)  # label obscures the line
    rgb[90:165,35:225]=magenta                              # same-colour broad stratum
    catalogue={'items':[
        {'category':'stratum_swatch','appearance':{'fill_rgb':list(magenta)}},
        {'id':'T','name':'open-pit boundary','category':'other','subtype':'line','confidence':.95,
         'line':{'present':True,'dominant_rgb':list(magenta),'mean_width_px':8,'shape':'horizontal_straight'}}]}
    geological,technical,uncertain,audit=classify_catalogue_strokes(rgb,labels,catalogue,1.,14,10)
    assert technical[36:42,112:130].sum()>40       # directional continuation through the label
    assert technical[105:150,60:200].mean()<.05    # broad same-colour geology is not erased
    assert audit['directional_extension_pixels']>0


def test_boundary_support_cannot_grow_domain_from_remote_legend_line():
    labels=np.full((120,180),-1,np.int16);labels[35:95,45:135]=0
    boundary=np.zeros(labels.shape,bool);boundary[33:97,43]=True;boundary[5:8,5:90]=True
    domain,audit=infer_domain(labels,np.ones(labels.shape,bool),(),None,boundary)
    assert domain[60,44]
    assert not domain[6,20]
    assert audit['accepted_boundary_support_pixels']>0


def test_thin_verified_layer_survives_area_floor_and_text_crossing(tmp_path):
    im=Image.new('RGB',(340,200),'white');draw=ImageDraw.Draw(im)
    blue=(146,185,210);red=(220,70,85)
    draw.rectangle([10,10,269,179],fill=blue)
    draw.polygon([(35,145),(235,35),(238,40),(38,150)],fill=red)
    draw.rectangle([124,90,150,105],fill='black')  # label crossing the thin bed
    draw.rectangle([295,30,314,49],fill=blue);draw.rectangle([295,75,314,94],fill=red)
    path=tmp_path/'thin-text-crossing.png';im.save(path)
    cfg={'id':'T','image':str(path),'frame':'x','units':'m','station':0,'bounds':[0,0,260,170],
         'roi':[10,10,260,170],'registration_verified':True,'complete':False,'layout_reviewed':True,
         'min_area_pixels':2000,'min_component_fraction':0,'auto_text_boxes':[[124,90,27,16]],
         'plot_content_rings':[{'points':[[10,10],[269,10],[269,179],[10,179]],'hole':False}],
         'legend':[{'lithology':'host','swatch':[297,32,16,16],'verified':True},
                   {'lithology':'thin-bed','swatch':[297,77,16,16],'verified':True}]}
    result,_=recognize(cfg)
    thin=[layer for layer in result['layers'] if layer['lithology']=='thin-bed']
    assert len(thin)==1
    assert 'thin_layer_multi_evidence' in thin[0]['geometry_flags']
    assert result['recognition']['thin_layer_retained']>=1
