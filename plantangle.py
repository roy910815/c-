from plantcv import plantcv as pcv 
from PIL import Image
import cv2
import timeit
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
from skimage.morphology import medial_axis
import math
class options:
    def __init__(self):
        self.image = "./HV281_0516.jpg" 
        self.image1 = "./HV281_0516_0.jpg"
        self.image2 = "./HV281_0516_1.jpg"
        self.image3 = "./HV281_0516_2.jpg"
        self.debug = "plot"
        self.writeimg= False
        self.result1 ="HV281_0516_3.txt"
        self.result2 = "HV281_0516_angle.txt"
        self.outdir = "." # Store the output to the current directory           
# Get options
args = options()
img = cv2.imread(args.image)
img = cv2.resize(img, (math.floor(img.shape[1]/5), math.floor(img.shape[1]/5)), interpolation=cv2.INTER_AREA)
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
cv2.imshow('My Image', img)
plt.imshow(img)
#pcv.params.debug = args.debug

#pcv.params.debug = args.debug

img2, path, filename = pcv.readimage(filename=args.image)
img = cv2.resize(img2, (math.floor(img2.shape[1]/5), math.floor(img2.shape[1]/5)), interpolation=cv2.INTER_AREA)

v = pcv.rgb2gray_hsv(rgb_img=img, channel='v')
v_thresh = pcv.threshold.binary(gray_img=v, threshold=120, max_value=255, object_type='light') #
v_mblur = pcv.median_blur(gray_img=v_thresh, ksize=5)
gaussian_imgv = pcv.gaussian_blur(img=v_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)

l = pcv.rgb2gray_lab(rgb_img=img, channel='l')
l_thresh = pcv.threshold.binary(gray_img=l, threshold=155, max_value=255, object_type='light')#
#b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
#cv2.imshow('My gaussian_imgv', gaussian_imgv)

#b_thresh = pcv.threshold.binary(gray_img=b, threshold=160, max_value=255,  object_type='light')
bs = pcv.logical_and(bin_img1=v_thresh, bin_img2=l_thresh)
#cv2.imshow('My bs', bs)
bs_xor = pcv.logical_xor(bin_img1=bs, bin_img2=l_thresh)
bs_or = pcv.logical_or(bin_img1=bs, bin_img2=l_thresh)
#cv2.imshow('My bs_or', bs_or)#
masked = pcv.apply_mask(img=img, mask=bs_or, mask_color='black')
#cv2.imshow('My masked', masked)#
id_objects, obj_hierarchy = pcv.find_objects(img=masked, mask=bs_or)
roi1, roi_hierarchy= pcv.roi.rectangle(img=masked, x=0, y=0, h=math.floor(img2.shape[1]/5)*1/2, w=math.floor(img2.shape[1]/5))
roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1, 
                                                               roi_hierarchy=roi_hierarchy, 
                                                               object_contour=id_objects, 
                                                               obj_hierarchy=obj_hierarchy,
                                                               roi_type='cutto')
# 

v = pcv.rgb2gray_hsv(rgb_img=img, channel='v')
v_thresh = pcv.threshold.binary(gray_img=v, threshold=124, max_value=255, object_type='light') #
s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
s_thresh = pcv.threshold.binary(gray_img=s, threshold=60, max_value=255, object_type='light')
s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
gaussian_img = pcv.gaussian_blur(img=s_thresh, ksize=(3, 3), sigma_x=0, sigma_y=None)
b = pcv.rgb2gray_lab(rgb_img=img, channel='l')

# Threshold the blue channel image 
b_thresh = pcv.threshold.binary(gray_img=b, threshold=155, max_value=255, 
                                object_type='light')
#roi1, roi_hierarchy= pcv.roi.rectangle(img=b_thresh, x=100, y=100, h=200, w=200)
#roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1,  roi_hierarchy=roi_hierarchy, object_contour=id_objects, obj_hierarchy=obj_hierarchy,roi_type='partial')

bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
masked = pcv.apply_mask(img=img, mask=bs, mask_color='white')
masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=125, max_value=255, object_type='dark')
maskeda_thresh=pcv.closing(maskeda_thresh,kernel=None)
ab_fill = pcv.fill(bin_img=maskeda_thresh, size=100)
closed_ab = pcv.closing(gray_img=ab_fill)
#masked_l= pcv.rgb2gray_lab(rgb_img=masked, channel='l')
maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=125, max_value=255, object_type='light')

bs_and = pcv.logical_and(bin_img1=maskeda_thresh1, bin_img2=closed_ab)
add=pcv.image_add(closed_ab,bs_and)

add2=pcv.image_add(add,kept_mask)

dilate1=pcv.dilate(add2,ksize=5, i=1)
mask1 = cv2.medianBlur(dilate1,1)
cv2.imshow('My kept_mask', mask1)
#mask1_p = plt.imshow(mask1)
#mask1_p.set_cmap('plasma')
#plt.savefig('mask1.jpg', format="jpg", bbox_inches='tight')
#

key=cv2.waitKey(0)
cv2.imwrite(args.image1, mask1)
###############################################################################################
IMG, path, filename = pcv.readimage(filename=args.image1)


# # 影像侵蝕 erode 與影像膨脹 dilate
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(IMG, kernel, iterations = 3)
#iteration 進行多次腐蝕。
kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(erosion, kernel, iterations = 1)
ret,dilation1 = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY)

SK_Median = median(dilation, disk(4), mode='mirror', cval=0.0)

#cv2.imshow("cv2 median", Opencv_Median) # 噪點依舊太多
#cv2.imshow("Using skimage median", SK_Median) #醬噪成功 但依舊無法擬和
ret,sk_bc = cv2.threshold(SK_Median, 0, 255, cv2.THRESH_BINARY) #強制轉黑白
cv2.imshow("sk_bc", sk_bc)

cv2.imshow("Using HV050_0210_1 ", sk_bc) 
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(args.image2, sk_bc)
 #降造
################################################################################################

img, path, filename = pcv.readimage(filename=args.image2)

cropped_mask = img 
cropped_mask[cropped_mask > 0] = 1

mask1 = cv2.medianBlur(img,7)
skeleton1 = pcv.morphology.skeletonize(mask=mask1) 
img1, seg_img, edge_objects= pcv.morphology.prune(skel_img=skeleton1, size=110, mask=mask1)

cv2.imshow('My result', seg_img)

key=cv2.waitKey(0)
plt.axis('on')
#plt.colorbar()
seg_img_p = plt.imshow(seg_img)
seg_img_p.set_cmap('plasma')
plt.savefig(args.image3, format="jpg", bbox_inches='tight')
#plt.savefig(args.result_txt, format="txt", bbox_inches='tight')
#print(seg_img.shape)  
#
path = args.result1
f = open(path, 'w')

for y in range(math.floor(img.shape[1])):
    for x in range(math.floor(img.shape[1])):
        b,g,r=seg_img[y,x]
        if ((b == 0) and (g == 0)and(r == 0)) or((b == 1) and (g == 1)and(r == 1)):
           print(x,y)
        else:
           print(x,y,b,g,r)
           #line =str(x),str(y),str(b),str(g),str(r)
           line=',',x,y,b,g,r,','
           f.writelines(str(line))
           f.write('\n')
f.close()
############################################################################################
txt1=open(args.result1, 'r')
f = open(args.result2, 'w')

dicmax_x = {}
dicmax_y = {}
dicmin_x = {}
dicmin_y = {}
for line in txt1.readlines():
    s = line.split(',')
    x=int(s[2])
    y=int(s[3])
    b=s[4]
    g=s[5]
    r=s[6]
    color=int(b)+int(g)+int(r)
    #print(x,y,color)
    if color in dicmax_y:
            #print('a')
            if dicmax_y[color]>(y):
                dicmax_x[color]=x
                dicmax_y[color]=y
                #print('b')
    else:
            dicmax_x[color]=x
            dicmax_y[color]=y


    if color in dicmin_y:
            #print('c')
            if dicmin_y[color]<(y):
                dicmin_x[color]=x
                dicmin_y[color]=y
                #print('d')
    else:
            dicmin_y[color]=y
            dicmin_x[color]=x
for color in dicmax_y:
      dicmin_y[color]=250-dicmin_y[color]
      dicmax_y[color]=250-dicmax_y[color]
print("max x")
print(dicmax_x)
print("min x") 
print(dicmin_x)
print("max y")     
print(dicmax_y)          
print("min y")
print(dicmin_y)

stemy=250
stemx=200
for y in dicmax_y:
      if dicmax_y.get(y)<stemy:
            stemy=dicmax_y.get(y)
            stemx=dicmax_x.get(y) 
s_dic={}
print(stemx,stemy)
for color in dicmax_y:
    dx=stemx-dicmax_x.get(color)
    dy=stemy-dicmax_y.get(color)
    if dx==0:
          dx=dicmax_x.get(color)-dicmin_x.get(color)
          dy=dicmax_y.get(color)-dicmin_y.get(color)
    if dx==0:
          dx=1
          dy=1
    s=dy/dx
    s_dic[dicmax_y.get(color)]=s
print(s_dic)
sm=250
for a in s_dic: #找莖
        if a < sm:
              sm=a   
print(sm)             
for y in s_dic: #和莖的夾角
        if y == sm:
            print('0')   
        else:
            t=s_dic.get(y)
            stem=s_dic.get(sm)
            tan1=(stem-t)/(1+t*stem)
            an=math.atan(tan1)* 180 /math.pi
            if an<0:
                  an=an+180
            else:
                  an=180-an 
            st=250-y,an
            print(250-y,an)     
            f.writelines(str(st))
            f.write('\n')
           
