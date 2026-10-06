"""Controlled validation: a legend inside the rectangular plot ROI stays non-stratal."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from PIL import Image,ImageDraw
from stratamatch.io import local,write
from stratamatch.layout import detect,apply_proposal
from stratamatch.recognition import recognize

colors=[(146,185,210),(236,193,96),(159,203,151)]
im=Image.new('RGB',(500,360),'white');d=ImageDraw.Draw(im)
d.rectangle([20,20,470,79],fill=colors[0]);d.rectangle([20,80,130,279],fill=colors[1])
d.rectangle([430,80,470,279],fill=colors[1]);d.rectangle([20,280,470,339],fill=colors[2])
for i,c in enumerate(colors):d.rectangle([250,120+i*45,273,137+i*45],fill=c)
path=local('data/layout_validation/inset_legend.png');im.save(path)
proposal=detect(path)
section=apply_proposal({'id':'INSET','image':str(path.relative_to(local('.'))),'frame':'pixel_fixture','units':'px','station':0,
    'bounds':[0,0,451,320],'registration_verified':True,'complete':False,'layout_reviewed':False,'legend':[]},proposal)
section['layout_reviewed']=True
for entry,name in zip(section['legend'],['shale','sandstone','limestone']):entry['lithology']=name;entry['verified']=True
recognized,labels=recognize(section);x,y,_,_=section['roi']
centers=[]
for entry in section['legend']:
    sx,sy,sw,sh=entry['swatch'];centers.append({'point':[sx+sw//2,sy+sh//2],'label_value':int(labels[sy-y+sh//2,sx-x+sw//2])})
overlay=im.copy();draw=ImageDraw.Draw(overlay)
rx,ry,rw,rh=proposal['roi'];draw.rectangle([rx,ry,rx+rw,ry+rh],outline='#40e49b',width=2)
for ring in proposal['content_rings']:
    draw.line(ring['points']+[ring['points'][0]],fill='#ff4cab' if ring['hole'] else '#00a5e8',width=3)
for entry in proposal['legend']:
    sx,sy,sw,sh=entry['detected_box'];draw.rectangle([sx,sy,sx+sw,sy+sh],outline='#ffad00',width=3)
overlay.save(local('outputs/inset-legend-validation.png'))
write('outputs/inset-legend-validation.json',{'proposal':proposal,'legend_centers':centers,
    'all_legend_centers_unassigned':all(v['label_value']==-1 for v in centers),'recognized_layers':len(recognized['layers']),
    'warning':'Controlled fixture only; validates geometry logic, not real-data accuracy.'})
print(json.dumps({'roi':proposal['roi'],'legend_count':len(proposal['legend']),'rings':len(proposal['content_rings']),
    'all_legend_centers_unassigned':all(v['label_value']==-1 for v in centers)}))
