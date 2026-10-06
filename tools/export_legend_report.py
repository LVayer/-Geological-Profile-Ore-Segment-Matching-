"""Run complete legend preprocessing and export an auditable Chinese report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stratamatch.io import local
from stratamatch.legend_features import analyze_legend


TYPE_NAMES={'stratum_swatch':'地层色块','geological_boundary':'地层/断层界限',
            'other.label':'其他-标签','other.line':'其他-线条','other.label_line':'其他-标签加线条'}
BOX_COLOURS={'stratum_swatch':(35,190,65),'geological_boundary':(230,45,45),
             'other.label':(60,120,240),'other.line':(30,195,210),'other.label_line':(245,145,30)}


def type_key(item):return item['category'] if item['category']!='other' else f"other.{item['subtype']}"


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(local(args.config).read_text(encoding='utf-8-sig'))
    result=analyze_legend(cfg['image'],cfg.get('legend',[]),cfg.get('page',0))
    output=local(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'P101_图例特征.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

    rows=[]
    for item in result['items']:
        line=item['line'];appearance=item['appearance'];key=type_key(item)
        rows.append({'id':item['id'],'name':item['name'],'type':TYPE_NAMES[key],'subtype':item['subtype'],
            'confidence':item['confidence'],'box':item['box'],'main_rgb':appearance.get('fill_rgb',appearance.get('dominant_rgb')),
            'embedded_label_fraction':appearance.get('embedded_label_fraction',''),
            'line_rgb':line.get('dominant_rgb',''),'line_width_px':line.get('mean_width_px',''),
            'line_shape':line.get('shape',''),'orientation_deg':line.get('orientation_degrees',''),
            'symbol_text':item['symbol_text'],'review_flags':'|'.join(item['review_flags'])})
    with (output/'P101_图例特征.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

    with Image.open(local(cfg['image'])) as source:rgb=np.asarray(source.convert('RGB'))
    x,y,w,h=result['panel'];crop=rgb[y:y+h,x:x+w].copy()
    for item in result['items']:
        bx,by,bw,bh=item['box'];p1=(bx-x,by-y);p2=(bx-x+bw,by-y+bh);colour=BOX_COLOURS[type_key(item)]
        cv2.rectangle(crop,p1,p2,colour,5,lineType=cv2.LINE_8)
        cv2.putText(crop,item['id'],(p1[0],max(18,p1[1]-7)),cv2.FONT_HERSHEY_SIMPLEX,.62,colour,2,cv2.LINE_AA)
    Image.fromarray(crop).save(output/'P101_图例识别.png')

    summary=[]
    for key in ['stratum_swatch','geological_boundary','other.label','other.line','other.label_line']:
        summary.append(f"- {TYPE_NAMES[key]}：{result['counts'].get(key,0)} 项")
    table=[]
    for row in rows:
        colour=row['main_rgb'] if row['type']=='地层色块' else row['line_rgb']
        line=f"{row['line_width_px']} px / {row['line_shape']}" if row['line_width_px']!='' else '—'
        table.append(f"| {row['id']} | {row['name']} | {row['type']} | {colour} | {line} | {row['confidence']} |")
    report=f"""# P101 图例预处理报告

## 处理流程

1. 以自动检出的地层色块列作为锚点，向整张图片扩展图例搜索区，不排除主图矩形内部区域。
2. 通过矩形边框、重复尺寸、列对齐和行间距检测完整图例单元；孤立矩形和图框坐标不作为图例。
3. 对每个单元分别读取符号区和右侧名称。OCR 只提供名称建议，不单独决定类别。
4. 地层色块提取填充主色、颜色分布、均匀度及色块内部标签墨迹比例。
5. 线状符号提取主色、像素线宽、方向、直线/折线/曲线/虚线形态、线段数量和符号内文字。
6. 综合单元所在列、已确认色块、图形证据和地质语义，将图例分为地层色块、地层/断层界限及其他；其他继续分为标签、线条、标签加线条。
7. 缺少名称或图形证据冲突时保留复核标记，不强行作为地质界限。

## 本图结果

- 共识别 **{result['cell_count']}** 个图例单元，OCR 状态：`{result['ocr_status']}`。
{chr(10).join(summary)}

| 编号 | OCR 名称 | 分类 | 主要颜色 RGB | 线宽/形态 | 置信度 |
|---|---|---|---|---|---:|
{chr(10).join(table)}

## 对后续识别的直接作用

- 9 个地层色块提供正式颜色原型；色块中的 Fe2、Fe3、Fe4 等文字被记录为标签特征，可用于清除主图内同类文字，但不参与地层颜色距离。
- “断层及编号”和“地质界线”保存为地质边界原型，主图中只有颜色、线宽和形态与这些原型相符并具有连续地质拓扑证据时才允许作为边界。
- 纵剖面线、露天开采境界线、采矿权范围及资源量计算边界保存为其他线条原型；它们不应成为地层界面。
- 钻孔位置、取样位置和面积编号保存为标签加线条原型；矿体编号、方位角、孔深、厚度/品位保存为标签原型。

## 当前限制

OCR 名称仍有少数字符误识别，例如品位公式和钻孔说明，因此名称保持人工复核状态。颜色、线宽和形态来自原始像素，可直接用于后续主图判别。图例特征只是强证据来源；主图中与图例相似但拓扑冲突的对象仍应标为不确定。
"""
    (output/'P101_图例预处理报告.md').write_text(report,encoding='utf-8')
    enhanced=dict(cfg);enhanced['legend_catalog']=result
    (output/'P101_图例增强配置.json').write_text(json.dumps(enhanced,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cells':result['cell_count'],'counts':result['counts'],'output':str(output)},ensure_ascii=False))


if __name__=='__main__':main()
