# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:40:29 2019

@author: 咸鸡
"""

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('bandian.jpg',0)
surf = cv2.xfeatures2d.SURF_create(1000)
#surf = cv2.SURF(400)
kp, des = surf.detectAndCompute(img,None)
#surf.hessianThreshold = 50000
kp, des = surf.detectAndCompute(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),2)
plt.imshow(img2),plt.show()
cv2.imwrite('surfbandain.jpg',img2)