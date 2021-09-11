""" 问题 3 测试图像二维快速傅里叶变换与逆变换 （8 分）
对于给定的输入图像 rose512.tif, 首先将其灰度范围通过归一化调整到[0,1]. 将此归一化的
图像记为 f. 首先调用问题 1 下实现的函数 dft2D 计算其傅里叶变换，记为 F。然后调用问题
2 下的函数 idft2D 计算 F 的傅里叶逆变换，记为 g. 计算并显示误差图像 d = f-g. """

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from question1 import dft2D
from question2 import idft2D
from numpy.fft import fft,ifft,fft2,fftshift

if __name__ == "__main__":
    img = cv2.imread(r'rose512.tif',0)  ##读取图片
    f = img/255
    F = dft2D(f)
    g = idft2D(F)
    g = g.real
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    err = cv2.absdiff(g,f)
    err = np.array(err,dtype='uint8')
    plt.subplot(2,2,1)
    plt.title('归一化图片f')
    plt.imshow(f,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,2)
    plt.title('傅里叶变换F')
    plt.imshow(np.log(1+abs(fftshift(F))),cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,3)
    plt.title('傅里叶逆变换g')
    plt.imshow(g,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,4)
    plt.title('误差图像d')
    plt.imshow(err,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()

