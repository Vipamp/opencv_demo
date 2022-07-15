'''
   @author: HeQingsong
   @date: 2022-02-23 10:15
   @filename: tst.py
   @project: opencv_demo
   @python version: 3.7 by Anaconda
   @description: 
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((1080, 1920, 3), np.uint8)
area1 = np.array([[250, 200], [300, 100], [750, 800], [100, 1000]])
area2 = np.array([[1000, 200], [1500, 200], [1500, 400], [1000, 400]])

cv2.fillPoly(img, [area1, area2], (255, 255, 255))

plt.imshow(img)
plt.show()
