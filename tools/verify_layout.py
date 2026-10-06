"""Record actual local OCR results on a fully labeled Chinese export fixture."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont
from stratamatch.io import local,write
from stratamatch.layout import prepare_image

im=Image.new('RGB',(850,430),'white');d=ImageDraw.Draw(im)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
d.text((40,8),'规范彩色剖面：自动定位验证图',font=font,fill='black')
colors=[(146,185,210),(236,193,96),(159,203,151)];names=['页岩','砂岩','石灰岩']
for i,(color,name) in enumerate(zip(colors,names)):
    d.rectangle([40,60+i*90,539,149+i*90],fill=color)
    y=80+i*75;d.rectangle([620,y,651,y+23],fill=color);d.text((670,y-4),name,font=font,fill='black')
d.line([30,50,30,340,550,340],fill='black',width=2)
p=local('data/layout_validation/chinese_legend.png');p.parent.mkdir(parents=True,exist_ok=True);im.save(p)
result=prepare_image(str(p));result['id']='AUTO_DEMO';result['station']=0
write('outputs/layout-validation.json',{'fixture':str(p),'truth':{'roi':[40,60,500,270],'lithologies':names},'proposal':result,
    'warning':'Single controlled fixture; not real-data accuracy. OCR proposals remain unverified.'})
print('ROI:',result['roi'],'Legend:',[(e['lithology'],e['verified']) for e in result['legend']],'OCR:',result['auto_layout'].get('ocr'))
