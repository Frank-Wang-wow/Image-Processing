""" 问题1 通过计算一维傅里叶变换实现 图像 二维 快速傅里叶变换（10 分）
实现一个函数 F=dft2D(f), 其中 f 是一个灰度源图像，F 是其对应的二维快速傅里叶变换(FFT)图像. 具体实现要求按照课上的介绍通过两轮一维
傅里叶变换实现。也就是首先计算源图像每一行的一维傅里叶变换，然后对于得到的结果计算其每一列的一维傅里叶变换。如果实现采用 MATLAB, 
可以直接调用函数 fft 计算一维傅里叶变换。如果采用其他语言，请选择并直接调用相应的一维傅里叶变换函数。 """

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import math
from numpy.fft import fft,ifft,fft2,fftshift
def dft2D(f):
    w,h = f.shape
    F_row = np.zeros((w,h),dtype = complex)  ###逐行FFT图初始化
    F = np.zeros((w,h),dtype = complex) ###逐列FFT图初始化
    for i in range(w):
        F_row[i]=fft(f[i])
    for j in range(h):
        F[:,j]=fft(F_row[:,j])
    return F

if __name__ == "__main__":
    img = cv2.imread(r'rose512.tif',0)  ##读取图片
    img = img/255
    img_F = abs(fftshift(dft2D(img)))
    print(img_F)
#    img_F = img_F//(np.max(img_F))
    img_ff = abs(fftshift(fft2(img)))
    print(img_ff)
    plt.title('origin image')
    plt.imshow(img,cmap="gray") #卷积处理输出图片
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.title('my fft')
    plt.imshow(img_F,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.title('numpys fft')
    plt.imshow(img_ff,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print(np.max(img_F))
    