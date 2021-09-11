from cv2 import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def scanLine4e(f,I,loc):
   row_num=len(f)
   col_num=len(f.T)
   if loc=='column':
       if I>=col_num or I<0:
           print('索引超过图像范围！')
       else:
           return f[:,I] 
   elif loc=='row':
       if I>=row_num or I<0:
           print('索引超过图像范围！')
       else:
           return f[I,:]
       
##############################################################
dirs = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(dirs, 'cameraman.tif')
print(file)
cam=cv2.imread(r'einstein.tif',0)
print(type(cam))
plt.imshow(cam)
plt.show()
cam_row,cam_col = cam.shape
#先提取列
loc='column'
Icc=math.ceil(cam_col/2)-1
cam_center_col=scanLine4e(cam,Icc,loc)
#再提取行
loc='row'
Icr=math.ceil(cam_row/2)-1
cam_center_row=scanLine4e(cam,Icr,loc)
#################################################################
ein=cv2.imread('einstein.tif',0)
ein_row=len(ein)
ein_col=len(ein.T)
#先提取列
loc='column'
Iec=math.ceil(ein_col/2)-1
ein_center_col=scanLine4e(ein,Iec,loc)
#再提取行
loc='row'
Ier=math.ceil(ein_row/2)-1
ein_center_row=scanLine4e(ein,Ier,loc)
###################################################################
#可视化
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.title("cammeraman.tif中心列灰度扫描图")
plt.xlabel("位置(行)")
plt.ylabel("灰度值")
plt.plot(cam_center_col)
plt.grid('true')
plt.show()

plt.figure(2)
plt.title("cammeraman.tif中心行灰度扫描图")
plt.xlabel("位置(列)")
plt.ylabel("灰度值")
plt.plot(cam_center_row)
plt.grid('true')
plt.show()
plt.figure(3)
plt.title("einstein.tif中心列灰度扫描图")
plt.xlabel("位置(行)")
plt.ylabel("灰度值")
plt.plot(ein_center_col)
plt.grid('true')
plt.show()
plt.figure(4)
plt.title("einstein.tif中心行灰度扫描图")
plt.xlabel("位置(列)")
plt.ylabel("灰度值")
plt.plot(ein_center_row)
plt.grid('true')
plt.show()

