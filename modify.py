from plantcv import plantcv as pcv 
import cv2
import numpy as np
import math
from skimage.filters import median
from skimage.morphology import disk, skeletonize

class Options:
    def __init__(self):
        self.image = "./HV281_0516.jpg" 
        self.debug = "plot"
        self.result_angle = "HV281_0516_angle.txt"
        
args = Options()

# 讀取和縮放影像
img = cv2.imread(args.image)
img = cv2.resize(img, (math.floor(img.shape[1]/5), math.floor(img.shape[0]/5)), interpolation=cv2.INTER_AREA)

# 灰階處理和閾值
v = pcv.rgb2gray_hsv(rgb_img=img, channel='v')
v_thresh = pcv.threshold.binary(gray_img=v, threshold=120, max_value=255, object_type='light')
l = pcv.rgb2gray_lab(rgb_img=img, channel='l')
l_thresh = pcv.threshold.binary(gray_img=l, threshold=155, max_value=255, object_type='light')
combined_mask = pcv.logical_or(v_thresh, l_thresh)

# 遮罩應用和分割
masked = pcv.apply_mask(img=img, mask=combined_mask, mask_color='black')
id_objects, obj_hierarchy = pcv.find_objects(img=masked, mask=combined_mask)
roi1, roi_hierarchy = pcv.roi.rectangle(img=masked, x=0, y=0, h=img.shape[0]//2, w=img.shape[1])

# HSV和LAB處理以突出特徵
s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
s_thresh = pcv.threshold.binary(gray_img=s, threshold=60, max_value=255, object_type='light')
s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
combined_mask2 = pcv.logical_or(s_mblur, l_thresh)
masked2 = pcv.apply_mask(img=img, mask=combined_mask2, mask_color='white')

# 降噪與骨架化
closed_ab = pcv.closing(gray_img=pcv.fill(bin_img=pcv.threshold.binary(masked2, threshold=125, max_value=255, object_type='dark'), size=100))
dilated = pcv.dilate(closed_ab, ksize=5)
skeleton = pcv.morphology.skeletonize(mask=dilated)

# 剪枝與角度計算
img1, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=110, mask=dilated)

# 儲存結果影像和角度計算
cv2.imwrite(args.result_angle, seg_img)

# 計算莖與分支之間的角度
stem_y, stem_x = min((y, x) for (y, x) in np.argwhere(seg_img) if seg_img[y, x] != 0)
angle_results = []

for color, (max_y, max_x) in dicmax_y.items():
    dx = stem_x - max_x
    dy = stem_y - max_y
    if dx == 0:
        dx, dy = max_x - dicmin_x[color], max_y - dicmin_y[color]
    angle = math.degrees(math.atan2(dy, dx)) % 180
    angle_results.append((max_y, angle))

# 輸出角度結果
with open(args.result_angle, 'w') as f:
    for y, angle in angle_results:
        f.write(f"{y}, {angle}\n")
