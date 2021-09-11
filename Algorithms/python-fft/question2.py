""" 问题 2 图像二维快速傅里叶逆变换 （10 分）
实现一个函数 f=idft2D(F), 其中 F 是一个灰度图像的傅里叶变换，f 是其对应的二维快速傅
里叶逆变换 (IFFT)图像，也就是灰度源图像. 具体实现要求按照课上的介绍通过类似正向变
换的方式实现。 """

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from cmath import exp 
from math import pi
from numpy.fft import fft,ifft,fft2
from question1 import dft2D

def idft2D(F):
    M,N = F.shape
    f = (dft2D(F.conjugate())/(M*N)).conjugate()
    return f

if __name__ == "__main__":
    img = cv2.imread(r'rose512.tif',0)  ##读取图片
    img = img/255
    F = dft2D(img)
    f = idft2D(F)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(1,3,1)
    plt.title('原图片')
    plt.imshow(img,cmap="gray") #卷积处理输出图片
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.title('傅里叶变换')
    plt.imshow(abs(F),cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.title('傅里叶逆变换')
    plt.imshow(f.real,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    