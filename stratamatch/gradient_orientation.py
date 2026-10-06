"""从岩性指示函数的边界梯度估计局部地层方向，不使用任意岩性编号差。"""
import cv2
import numpy as np


class GradientOrientation:
    def __init__(self, labels):
        self.labels = labels
        self.cache = {}
        self.samples = 0
        self.reliable = 0

    def sample(self, descriptor, point):
        """返回无向切线和结构张量各向异性；大片纯色内部退回原中心线。"""
        point = np.asarray(point, float)
        key = (descriptor['id'], round(float(point[0])/2), round(float(point[1])/2))
        if key in self.cache:
            return self.cache[key]
        line = descriptor['centreline']
        nearest = int(np.argmin(np.sum((line-point)**2, axis=1)))
        fallback = descriptor['tangents'][nearest]
        width = float(descriptor['width_profile'][min(nearest, len(descriptor['width_profile'])-1)])
        radius = int(np.clip(round(width*1.5), 12, 48))
        x, y = map(lambda value: int(round(float(value))), point)
        h, w = self.labels.shape
        x0, x1 = max(0, x-radius), min(w, x+radius+1)
        y0, y1 = max(0, y-radius), min(h, y+radius+1)
        self.samples += 1
        if x1-x0 < 7 or y1-y0 < 7:
            answer = (fallback, 0.)
        else:
            indicator = (self.labels[y0:y1, x0:x1] == descriptor['lith']).astype(np.float32)
            if indicator.min() == indicator.max():
                answer = (fallback, 0.)
            else:
                smooth = cv2.GaussianBlur(indicator, (0, 0), max(1.1, radius/9))
                gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)/8
                gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)/8
                xx, yy = np.meshgrid(np.arange(x0, x1)-point[0],
                                     np.arange(y0, y1)-point[1])
                weight = np.exp(-(xx*xx+yy*yy)/(2*(radius*.65)**2))
                jxx = float(np.sum(weight*gx*gx))
                jxy = float(np.sum(weight*gx*gy))
                jyy = float(np.sum(weight*gy*gy))
                energy = jxx+jyy
                contrast = np.hypot(jxx-jyy, 2*jxy)
                quality = float(contrast/max(energy, 1e-9) * min(1., energy/.16))
                normal_angle = .5*np.arctan2(2*jxy, jxx-jyy)
                tangent = np.array([-np.sin(normal_angle), np.cos(normal_angle)])
                # 交叉边界和文字附近方向不稳定；与实例长轴近乎垂直时不覆盖它。
                if abs(float(tangent@fallback)) < .35:
                    quality *= .25
                if quality < .15:
                    answer = (fallback, 0.)
                else:
                    if tangent@fallback < 0:
                        tangent = -tangent
                    blended = (1-quality)*fallback+quality*tangent
                    blended /= max(np.linalg.norm(blended), 1e-9)
                    answer = (blended, quality)
                    self.reliable += 1
        self.cache[key] = answer
        return answer
