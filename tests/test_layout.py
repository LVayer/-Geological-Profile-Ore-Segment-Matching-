from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw
import pytest
from stratamatch.layout import detect,attach_text,apply_proposal
from stratamatch.io import read,local
from stratamatch.recognition import recognize

def fixture_image(tmp_path,bottom=False,two_plots=False,legend=True):
    im=Image.new('RGB',(640,420),'white');d=ImageDraw.Draw(im)
    colors=[(146,185,210),(236,193,96),(159,203,151)]
    width=245 if two_plots else 400
    for i,c in enumerate(colors):d.rectangle([30,30+i*70,30+width-1,99+i*70],fill=c)
    if two_plots:
        for i,c in enumerate(colors):d.rectangle([350,30+i*70,594,99+i*70],fill=c)
    elif legend:
        for i,c in enumerate(colors):
            x,y=(40+i*155,320) if bottom else (490,55+i*55)
            d.rectangle([x,y,x+23,y+17],fill=c)
    p=tmp_path/'image.png';im.save(p);return p

def inset_legend_image(tmp_path):
    im=Image.new('RGB',(500,360),'white');d=ImageDraw.Draw(im)
    colors=[(146,185,210),(236,193,96),(159,203,151)]
    # Connected ring-shaped geology surrounds a blank annotation bay inside its bbox.
    d.rectangle([20,20,470,79],fill=colors[0]);d.rectangle([20,80,130,279],fill=colors[1]);d.rectangle([20,280,470,339],fill=colors[2])
    d.rectangle([430,80,470,279],fill=colors[1])
    for i,c in enumerate(colors):d.rectangle([250,120+i*45,273,137+i*45],fill=c)
    p=tmp_path/'inset.png';im.save(p);return p

@pytest.mark.parametrize('bottom',[False,True])
def test_plot_and_legend_layouts(tmp_path,bottom):
    result=detect(fixture_image(tmp_path,bottom=bottom))
    assert result['roi']==[30,30,400,210]
    assert len(result['legend'])==3
    assert all(e['verified'] is False and e['lithology']=='UNKNOWN' for e in result['legend'])

def test_two_equal_plots_require_review(tmp_path):
    result=detect(fixture_image(tmp_path,two_plots=True))
    assert result['roi'] is None and len(result['roi_candidates'])==2

def test_missing_legend_not_invented(tmp_path):
    result=detect(fixture_image(tmp_path,legend=False));assert result['roi'] and not result['legend']

def test_grayscale_and_blank_do_not_guess(tmp_path):
    for color in ['white','#aaaaaa']:
        p=tmp_path/'gray.png';Image.new('RGB',(400,300),color).save(p)
        result=detect(p);assert result['roi'] is None and result['status']=='manual_review'

def test_reviewed_settings_not_overwritten(tmp_path):
    section={'roi':[0,0,10,10],'legend':[{'lithology':'shale','verified':True}],'layout_reviewed':True}
    result=apply_proposal(section,detect(fixture_image(tmp_path)))
    assert result['roi']==section['roi'] and result['legend']==section['legend'] and result['auto_layout']

def test_ocr_association_and_no_automatic_authority():
    legend=[{'swatch':[10,10,20,20],'lithology':'UNKNOWN','verified':False}, {'swatch':[10,60,20,20],'lithology':'shale','verified':True}]
    lines=[{'words':[{'text':'砂岩','box':[40,12,30,16]}]}, {'words':[{'text':'错误文字','box':[40,62,35,16]}]}]
    result=attach_text(legend,lines)
    assert result[0]['lithology']=='砂岩' and not result[0]['verified']
    assert result[1]['lithology']=='shale' and result[1]['verified']

def test_unreviewed_layout_prevents_high_quality():
    cfg=read('data/synthetic/images/continuous/manifest.json')['sections'][0]
    cfg['layout_reviewed']=False
    section,_=recognize(cfg)
    assert section['layers'] and all(l['quality']<=.65 for l in section['layers'])

def test_legend_inside_rectangular_roi_is_found_and_hole_preserved(tmp_path):
    result=detect(inset_legend_image(tmp_path))
    assert result['roi']==[20,20,451,320] and len(result['legend'])==3
    assert result['legend_search_scope']=='full_image_including_roi'
    assert any(r['hole'] for r in result['content_rings'])
    assert all(20<=e['swatch'][0]<=470 for e in result['legend'])

def test_content_curve_excludes_inset_legend_from_strata(tmp_path):
    p=inset_legend_image(tmp_path);proposal=detect(p)
    section={'id':'A','image':str(p),'frame':'x','units':'m','station':0,'bounds':[0,0,451,320],
        'registration_verified':True,'complete':False,'layout_reviewed':False,'legend':[]}
    section=apply_proposal(section,proposal);section['layout_reviewed']=True
    names=['shale','sandstone','limestone']
    for e,name in zip(section['legend'],names):e['lithology']=name;e['verified']=True
    recognized,labels=recognize(section)
    assert recognized['recognition']['content_mask_used']
    # Inset legend centers map inside ROI but remain outside the curved plot content.
    x,y,_,_=section['roi']
    for e in section['legend']:
        sx,sy,sw,sh=e['swatch'];assert labels[sy-y+sh//2,sx-x+sw//2]==-1

def test_redetection_replaces_stale_automatic_legend_but_preserves_reviewed(tmp_path):
    proposal=detect(fixture_image(tmp_path))
    stale={'source':'auto_layout','lithology':'UNKNOWN','verified':False,'swatch':[1,1,5,5],'detected_box':[1,1,5,5]}
    reviewed={'lithology':'人工确认','verified':True,'swatch':[600,380,10,10]}
    updated=apply_proposal({'legend':[stale,reviewed]},proposal)
    assert stale not in updated['legend'] and reviewed in updated['legend']
    assert len(updated['legend'])==4

def test_reviewed_experiment_images_layout_regression():
    """The user's two real drawings remain the detector acceptance baseline."""
    expected=read('data/实验/expected_layout.json')['images']
    for name,rules in expected.items():
        result=detect(local('data/实验')/name);x,y,w,h=result['roi']
        assert rules['roi_x_range'][0]<=x<=rules['roi_x_range'][1]
        assert rules['roi_y_range'][0]<=y<=rules['roi_y_range'][1]
        assert rules['roi_right_range'][0]<=x+w<=rules['roi_right_range'][1]
        assert rules['roi_bottom_range'][0]<=y+h<=rules['roi_bottom_range'][1]
        assert len(result['legend'])==rules['legend_count']
        assert sum(e.get('inferred_blank',False) for e in result['legend'])==rules['inferred_blank_count']
        assert all(rules['legend_x_range'][0]<=e['detected_box'][0]<=rules['legend_x_range'][1] for e in result['legend'])

def test_confirmed_white_unit_can_use_enclosed_pale_region(tmp_path):
    im=Image.new('RGB',(210,140),'white');d=ImageDraw.Draw(im)
    d.rectangle([10,10,129,109],fill=(146,185,210));d.rectangle([45,35,85,85],fill='white')
    d.rectangle([160,20,179,39],fill=(146,185,210));d.rectangle([160,60,179,79],fill='white',outline='black')
    p=tmp_path/'white-unit.png';im.save(p)
    cfg={'id':'W','image':str(p),'frame':'x','units':'m','station':0,'bounds':[0,0,120,100],
        'roi':[10,10,120,100],'registration_verified':True,'complete':False,'layout_reviewed':True,
        'plot_content_rings':[{'points':[[10,10],[129,10],[129,109],[10,109]],'hole':False},
                              {'points':[[45,35],[85,35],[85,85],[45,85]],'hole':True}],
        'legend':[{'lithology':'blue','swatch':[162,22,16,16],'verified':True},
                  {'lithology':'white-unit','swatch':[162,62,16,16],'detected_box':[160,60,20,20],'verified':True}]}
    result,labels=recognize(cfg)
    assert labels[50,55]>=0
    assert result['recognition']['confirmed_pale_regions_enabled']
    assert any(layer['lithology']=='white-unit' for layer in result['layers'])

def test_white_drill_trace_is_removed_without_erasing_white_geology(tmp_path):
    """A long white borehole trace must not divide a coloured stratum.

    The compact white body is real geology because its legend item is reviewed;
    it must survive the same artifact pass that removes the technical trace.
    """
    im=Image.new('RGB',(260,180),'white');d=ImageDraw.Draw(im)
    blue=(146,185,210)
    d.rectangle([10,10,189,159],fill=blue)
    d.rectangle([92,10,99,159],fill='white')       # pale drill casing
    d.line([95,10,95,159],fill=(150,150,150),width=2)  # neutral technical centreline
    d.rectangle([125,55,160,105],fill='white')    # compact white geological body
    d.rectangle([215,25,234,44],fill=blue)
    d.rectangle([215,75,234,94],fill='white',outline='black')
    p=tmp_path/'white-drill-trace.png';im.save(p)
    cfg={'id':'D','image':str(p),'frame':'x','units':'m','station':0,'bounds':[0,0,180,150],
        'roi':[10,10,180,150],'registration_verified':True,'complete':False,'layout_reviewed':True,
        'artifact_max_gap_pixels':16,'bridge_unclassified_gap_pixels':20,
        'plot_content_rings':[{'points':[[10,10],[189,10],[189,159],[10,159]],'hole':False}],
        'legend':[{'lithology':'blue','swatch':[217,27,16,16],'verified':True},
                  {'lithology':'white-unit','swatch':[217,77,16,16],'detected_box':[215,75,20,20],'verified':True}]}
    result,labels=recognize(cfg)
    blue_layers=[layer for layer in result['layers'] if layer['lithology']=='blue']
    white_layers=[layer for layer in result['layers'] if layer['lithology']=='white-unit']
    assert len(blue_layers)==1 and len(white_layers)==1
    assert labels[20,85]==labels[20,95]             # trace no longer splits blue
    assert labels[70,130]>=0                        # white geology remains assigned
    assert result['recognition']['post_filter_gap_fill']['filled_pixels']>=0

def test_pixel_width_coloured_axis_is_not_a_stratum(tmp_path):
    im=Image.new('RGB',(260,180),'white');d=ImageDraw.Draw(im)
    blue=(146,185,210);red=(220,50,50)
    d.rectangle([10,10,189,159],fill=blue)
    # Crossing labels split the guide into collinear fragments in real plans.
    d.line([95,10,95,65],fill=red,width=1);d.line([95,82,95,159],fill=red,width=1)
    d.rectangle([215,25,234,44],fill=blue);d.rectangle([215,75,234,94],fill=red)
    p=tmp_path/'coloured-axis.png';im.save(p)
    cfg={'id':'C','image':str(p),'frame':'x','units':'m','station':0,'bounds':[0,0,180,150],
        'roi':[10,10,180,150],'registration_verified':True,'complete':False,'layout_reviewed':True,
        'plot_content_rings':[{'points':[[10,10],[189,10],[189,159],[10,159]],'hole':False}],
        'legend':[{'lithology':'blue','swatch':[217,27,16,16],'verified':True},
                  {'lithology':'red','swatch':[217,77,16,16],'verified':True}]}
    result,labels=recognize(cfg)
    assert len(result['layers'])==1 and result['layers'][0]['lithology']=='blue'
    assert labels[40,84]==labels[40,86]>=0
    assert result['recognition']['artifact_preprocessing']['coloured_axis_pixels']>0

def test_large_image_uses_memory_bounded_working_resolution(tmp_path):
    im=Image.new('RGB',(1800,1200),'white');d=ImageDraw.Draw(im)
    d.rectangle([100,100,1599,549],fill=(146,185,210));d.rectangle([100,550,1599,999],fill=(236,193,96))
    d.rectangle([1650,200,1690,240],fill=(146,185,210));d.rectangle([1650,300,1690,340],fill=(236,193,96))
    p=tmp_path/'large.jpg';im.save(p,quality=95)
    cfg={'id':'M','image':str(p),'frame':'x','units':'m','station':0,'bounds':[0,0,1500,900],
        'roi':[100,100,1500,900],'registration_verified':True,'complete':False,'layout_reviewed':True,
        'max_recognition_pixels':250000,'min_component_fraction':.00005,
        'legend':[{'lithology':'blue','swatch':[1655,205,30,30],'verified':True},
                  {'lithology':'yellow','swatch':[1655,305,30,30],'verified':True}]}
    result,labels=recognize(cfg)
    assert labels.size<=255000
    assert result['recognition']['memory_bounded_classification']
    assert len(result['layers'])==2
