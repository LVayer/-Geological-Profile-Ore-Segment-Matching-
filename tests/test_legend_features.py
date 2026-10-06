import cv2
import numpy as np
from PIL import Image

from stratamatch.legend_features import analyze_legend


def test_complete_legend_catalogue_uses_geometry_colour_line_and_text(tmp_path):
    rgb=np.full((500,600,3),255,np.uint8);cells=[]
    for x in (90,340):
        for y in (90,180,270):
            cv2.rectangle(rgb,(x,y),(x+90,y+55),(150,150,150),2);cells.append([x,y,91,56])
    cv2.rectangle(rgb,(93,93),(178,142),(210,80,30),-1)
    cv2.rectangle(rgb,(93,183),(178,232),(40,190,80),-1)
    cv2.line(rgb,(100,315),(170,280),(235,25,25),4);cv2.putText(rgb,'F1',(138,320),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,25,25),1)
    for x in range(350,420,18):cv2.line(rgb,(x,115),(min(x+10,425),115),(20,30,230),4)
    cv2.putText(rgb,'I',(378,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.ellipse(rgb,(385,315),(35,22),0,190,350,(0,0,0),2)
    path=tmp_path/'legend.png';Image.fromarray(rgb).save(path)
    known=[{'lithology':'A','swatch':[94,94,84,48],'detected_box':[90,90,91,56],'verified':True},
           {'lithology':'B','swatch':[94,184,84,48],'detected_box':[90,180,91,56],'verified':True}]
    descriptions=[('岩层A',[185,103,55,24]),('岩层B',[185,193,55,24]),('断层及编号',[185,283,110,24]),
                  ('纵剖面线',[435,103,90,24]),('矿体编号',[435,193,90,24]),('地质界线',[435,283,90,24])]
    ocr={'status':'available','lines':[{'words':[{'text':text,'box':box}]} for text,box in descriptions]}
    result=analyze_legend(str(path),known,ocr_result=ocr)
    assert result['cell_count']==6
    assert result['counts']['stratum_swatch']==2
    assert result['counts']['geological_boundary']==2
    fault=next(item for item in result['items'] if item['name']=='断层及编号')
    assert fault['line']['present'] and fault['line']['dominant_rgb'][0]>150
    assert fault['line']['mean_width_px']>0
    assert any(item['name']=='纵剖面线' and item['category']=='other' for item in result['items'])
