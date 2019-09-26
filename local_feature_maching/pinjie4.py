# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 21:11:20 2019

@author: 咸鸡
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

GOOD_POINTS_LIMITED = 0.99

src = 'spring_1.jpg'
des = 'spring_2.jpg'

img1_3 = cv.imread(src,1)# 基准图像
img2_3 = cv.imread(des,1)# 拼接图像


'''
img1_3 = cv.imread('3.jpg')# 基准图像
img2_3 = cv.imread('4.jpg')# 拼接图像
'''

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1_3,None)
kp2, des2 = orb.detectAndCompute(img2_3,None)




bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

'''
FLANN_INDEX_KDTREE=0   #建立FLANN匹配器的参数

indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5) #配置索引，密度树的数量为5

searchParams=dict(checks=50)    #指定递归次数
flann=cv.FlannBasedMatcher(indexParams,searchParams)  #建立匹配器
matches=flann.knnMatch(des1,des2,k=2)
'''


matches = sorted(matches, key = lambda x:x.distance)

goodPoints =[]
for i in range(len(matches)-1):
    if matches[i].distance < GOOD_POINTS_LIMITED * matches[i+1].distance:
        goodPoints.append(matches[i])

# goodPoints = matches[:20] if len(matches) > 20   else matches[:]
print(goodPoints)

img3 = cv.drawMatches(img1_3,kp1,img2_3,kp2,goodPoints, flags=2,outImg=None )

src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)

M, mask = cv.findHomography( dst_pts,src_pts, cv.RHO)

# 获取原图像的高和宽
h1,w1,p1 = img2_3.shape
h2,w2,p2 = img1_3.shape

h = np.maximum(h1,h2)
w = np.maximum(w1,w2)

_movedis = int(np.maximum(dst_pts[0][0][0],src_pts[0][0][0]))
imageTransform = cv.warpPerspective(img2_3,M,(w1+w2-_movedis,h))

M1 = np.float32([[1, 0, 0], [0, 1, 0]])
h_1,w_1,p = img1_3.shape
dst1 = cv.warpAffine(img1_3,M1,(w1+w2-_movedis, h))

dst = cv.add(dst1,imageTransform)
dst_no = np.copy(dst)

dst_target = np.maximum(dst1,imageTransform)

fig = plt.figure (tight_layout=True, figsize=(8, 18))
gs = gridspec.GridSpec (6, 2)
ax = fig.add_subplot (gs[0, 0])
ax.imshow(img1_3)
ax = fig.add_subplot (gs[0, 1])
ax.imshow(img2_3)
ax = fig.add_subplot (gs[1, :])
ax.imshow(img3)

cv.imwrite('pinjie.jpg',img3)

ax = fig.add_subplot (gs[2, :])
ax.imshow(imageTransform)
ax = fig.add_subplot (gs[3, :])
ax.imshow(dst1)
ax = fig.add_subplot (gs[4, :])
ax.imshow(dst_no)
ax = fig.add_subplot (gs[5, :])
ax.imshow(dst_target)
ax.set_xlabel ('The smooth method is SO FAST !!!!')
plt.show()