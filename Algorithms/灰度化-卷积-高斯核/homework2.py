""" 图像处理中的一个常见问题是将彩色 RGB 图像转换成单色灰度图像，第一种常用的方法是
取三个元素 R，G，B 的均值。第二种常用的方式，又称为 NTSC 标准，考虑了人类的彩色
感知体验，对于 R,G,B 三通道分别采用了不同的加权系数，分别是 R 通道 0.2989，G 通道
0.5870，B 通道 0.1140. 实现一个函数 g = rgb1gray(f, method). 函数功能是将一幅 24 位的
RGB 图像, f, 转换成灰度图像, g. 参数 method 是一个字符串，当其值为’average’ 时，采用
第一种转换方法，当其值为’NTSC’时，采用第二种转换方法。将’NTSC’做为缺省方式。
调用该函数，将提供的图像 mandril_color.tif 和 lena512color.tiff 用上述两种方法转换成单色
灰度图像，对于两种方法的结果进行简短比较和讨论。 """

import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def rgb1gray(f, method="NTSC"): ##灰度化函数
    if method == 'average':
        gray = (f[:,:,0]+f[:,:,1]+f[:,:,2])/3
    else:
        gray = 0.2989*f[:,:,0]+0.5870*f[:,:,1]+0.1140*f[:,:,2]
    return gray

if __name__ == '__main__':
    img = mpimg.imread(r'mandril_color.tif')  ##读取图片
    gray1 = rgb1gray(img,'average') #第一种平均值灰度化方法
    gray2 = rgb1gray(img) #第二种灰度化方法
    img1 = mpimg.imread(r'lena512color.tiff')  ##读取图片
    gray11 = rgb1gray(img1,'average') #第一种平均值灰度化方法
    gray12 = rgb1gray(img1) #第二种灰度化方法
    ##图像显示
    plt.subplot(2,3,1)
    plt.imshow(img) #原始图像
    plt.title('origin')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,3,2)
    plt.imshow(gray1,cmap="gray") #第一种灰度化图像
    plt.title('gary_average')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,3,3)
    plt.imshow(gray2,cmap="gray") #第二种灰度化图像
    plt.title('gray_NTSC')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,3,4)
    plt.imshow(img1) #原始图像
    plt.title('origin')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,3,5)
    plt.imshow(gray11,cmap="gray") #第一种灰度化图像
    plt.title('gary_average')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,3,6)
    plt.imshow(gray12,cmap="gray") #第二种灰度化图像
    plt.title('gray_NTSC')
    plt.xticks([])
    plt.yticks([])
    plt.show()