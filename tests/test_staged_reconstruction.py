import numpy as np
from PIL import Image,ImageDraw

from stratamatch.staged_reconstruction import _load_and_classify,reconstruct_stages


def test_four_stage_reconstruction_keeps_hard_outer_domain_and_fills_gaps(tmp_path):
    image=Image.new('RGB',(360,230),'white');draw=ImageDraw.Draw(image)
    blue=(90,170,220);orange=(225,145,45)
    # One section body with a black contact, a label-sized void and grid line.
    draw.polygon([(20,35),(270,35),(320,175),(55,195)],fill=blue)
    draw.polygon([(145,35),(270,35),(320,175),(180,185)],fill=orange)
    draw.line([(145,35),(180,185)],fill='black',width=5)
    draw.rectangle((105,95,135,115),fill='black')
    draw.line([(20,75),(300,75)],fill=(200,200,200),width=3)
    draw.rectangle((325,35,344,54),fill=blue);draw.rectangle((325,80,344,99),fill=orange)
    path=tmp_path/'staged.png';image.save(path)
    cfg={'image':str(path),'roi':[0,0,320,220],'max_recognition_pixels':1_000_000,
         'plot_content_rings':[{'points':[[20,35],[270,35],[320,175],[55,195]],'hole':False}],
         'legend':[{'lithology':'blue','swatch':[327,37,16,16],'verified':True},
                   {'lithology':'orange','swatch':[327,82,16,16],'verified':True}]}
    result=reconstruct_stages(cfg)
    assert result['stage1'].shape==result['final'].shape
    assert result['audit']['final_invariants']=={'outside_domain_pixels':0,'unfilled_inside_pixels':0}
    assert result['audit']['stage3']['filled_pixels']>0
    assert result['audit']['stage4']['filled_pixels']>0
    assert np.all(result['labels'][~result['domain']]==-1)


def test_unreadable_white_swatch_is_deferred_as_background_ambiguous(tmp_path):
    """OCR confidence must not decide whether white equals page background."""
    image=Image.new('RGB',(180,100),'white');draw=ImageDraw.Draw(image)
    blue=(70,150,220)
    draw.rectangle((10,20,120,80),fill=blue)
    draw.rectangle((55,30,75,70),fill='white')
    draw.rectangle((140,20,159,39),fill=blue)
    draw.rectangle((140,60,159,79),fill='white')
    path=tmp_path/'unreadable-white.png';image.save(path)
    cfg={'image':str(path),'roi':[0,0,130,100],'max_recognition_pixels':100_000,
         'legend':[{'lithology':'blue','swatch':[140,20,20,20],'verified':True},
                   {'lithology':'UNKNOWN','swatch':[140,60,20,20],'verified':False}]}
    _,labels,_,_,pale,pale_candidate,_,_,_,audit=_load_and_classify(cfg)
    assert pale==[1]
    assert audit['ambiguous_pale_pixels']>0
    assert np.any(pale_candidate)
    assert not np.any(labels==1)
