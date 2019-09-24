# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:24:13 2019

@author: 咸鸡
"""

import cv2

img = cv2.imread('bandian.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

#kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
#img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('siftbandian.jpg',img)



