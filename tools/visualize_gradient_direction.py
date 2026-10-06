"""在实际岩性图上标出被采用连接两端的边界梯度方向，供人工复核。"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratamatch.gradient_orientation import GradientOrientation
from stratamatch.layer_groups import _descriptor


def draw(folder):
    folder = Path(folder)
    audit = json.loads((folder/'分组依据.json').read_text(encoding='utf-8'))
    labels = np.load(folder/'回填前后标签.npz')['after']
    instances = np.load(folder/'区域实例.npy')
    by_id = {record['instance']: _descriptor(instances, record)
             for record in audit['parts']}
    field = GradientOrientation(labels)
    image = np.asarray(Image.open(folder/'岩性分类.png').convert('RGB')).copy()
    zoom = 2
    image = cv2.resize(image, None, fx=zoom, fy=zoom,
                       interpolation=cv2.INTER_NEAREST)
    visited = set()
    for edge in audit['pairs']:
        if edge.get('decision') != 'grouped':
            continue
        points = edge.get('curve_work_pixels', [])
        if len(points) < 2:
            continue
        for part, point in ((edge['left_part'], points[0]),
                            (edge['right_part'], points[-1])):
            if part not in by_id:
                continue
            point = np.asarray(point, float)
            key = (part, round(float(point[0])/12), round(float(point[1])/12))
            if key in visited:
                continue
            visited.add(key)
            tangent, quality = field.sample(by_id[part], point)
            if quality < .15:
                continue
            centre = point*zoom
            half = tangent*17
            p = tuple(np.rint(centre-half).astype(int))
            q = tuple(np.rint(centre+half).astype(int))
            color = (15, 110, 10) if quality >= .5 else (235, 120, 10)
            cv2.line(image, p, q, (255, 255, 255), 7, cv2.LINE_AA)
            cv2.line(image, p, q, color, 3, cv2.LINE_AA)
    cv2.rectangle(image, (12, 12), (1320, 100), (255, 255, 255), -1)
    cv2.putText(image, 'LOCAL BEDDING DIRECTION FROM LITHOLOGY BOUNDARIES',
                (28, 48), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(image, f'GREEN: STRONG  ORANGE: WEAK  SAMPLES: {len(visited)}',
                (28, 84), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2,
                cv2.LINE_AA)
    out = folder/'岩性边界梯度方向.png'
    Image.fromarray(image).save(out)
    print(out, len(visited), field.reliable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', nargs='+', required=True)
    args = parser.parse_args()
    for folder in args.folder:
        draw(folder)
