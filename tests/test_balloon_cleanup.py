"""气球颜色清理：拒绝近色杂项，但不按宽度删除真实薄层。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from experimental_fullres_colour_membrane import classify_full_resolution


def test_adaptive_colour_rejects_near_colour_and_keeps_thin_support(tmp_path):
    rgb = np.full((100,160,3),255,np.uint8)
    rgb[10:30,10:30] = [222,103,0]  # 图例纯底色
    rgb[40:80,40:95] = [222,103,0]
    rgb[55:57,95:130] = [222,103,0]  # 连续两像素薄层不能被颜色清理删除
    rgb[40:80,135:150] = [222,118,0]  # 接近图例但不属于色块自身的颜色波动
    path = tmp_path/'colour.png'; Image.fromarray(rgb).save(path)
    cfg={'image':str(path),'roi':[0,0,160,100],
         'legend':[{'swatch':[12,12,16,16]}], 'lab_tolerance':12,
         'exclude_boxes':[[10,10,20,20]]}
    old=classify_full_resolution({**cfg,'adaptive_colour_cleanup':False})
    new=classify_full_resolution(cfg)
    assert np.all(old[4][45:75,138:147]>=0)
    assert np.all(new[4][45:75,138:147]<0)
    assert np.all(new[4][55:57,95:130]>=0)
    assert np.all(new[4][10:30,10:30]<0)
    assert new[-1]['adaptive_rejected_original_pixels']>0


def test_paper_is_not_a_pale_lithology(tmp_path):
    rgb=np.full((60,80,3),255,np.uint8)
    rgb[10:50,20:60]=[255,231,231]
    path=tmp_path/'pale.png';Image.fromarray(rgb).save(path)
    result=classify_full_resolution({'image':str(path),'roi':[0,0,80,60],
        'legend':[{'rgb':[255,231,231]}]})
    assert np.all(result[4][:10]<0)
    assert np.all(result[4][10:50,20:60]==0)


def test_drawing_fill_can_differ_from_legend(tmp_path):
    # P68 的同类绿色填充与图例约有 7.56 Lab 色差，不能当成杂色删除。
    rgb=np.full((60,80,3),255,np.uint8)
    rgb[2:12,2:12]=[0,204,127]
    rgb[20:50,20:60]=[50,217,127]
    path=tmp_path/'green.png';Image.fromarray(rgb).save(path)
    result=classify_full_resolution({'image':str(path),'roi':[0,0,80,60],
        'legend':[{'swatch':[2,2,10,10]}]})
    assert np.all(result[4][20:50,20:60]==0)
