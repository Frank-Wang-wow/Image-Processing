""" 问题 4 计算图像的中心化二维快速傅里叶变换与谱图像 (12 分)
我们的目标是复现下图中的结果。首先合成矩形物体图像，建议图像尺寸为 512×512，矩
形位于图像中心，建议尺寸为 60 像素长，10 像素宽，灰度假设已归一化设为 1. 对于输入
图像 f 计算其中心化二维傅里叶变换 F。然后计算对应的谱图像 S=log(1+abs(F)). 显示该谱
图像。 """

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from question1 import dft2D
from question2 import idft2D
from numpy.fft import fft,ifft,fft2,fftshift

def ff_center(F):
    M,N = F.shape
    M = int(M/2)
    N = int(N/2)
    return np.vstack((np.hstack((F[M:,N:],F[M:,:N])),np.hstack((F[:M,N:],F[:M,:N]))))

if __name__ == "__main__":
    ####合成图像#####
    f = np.zeros((512,512))
    f[226:286,251:265]=1
    F = dft2D(f)
    F_C = ff_center(F)
    S = np.log(1+abs(F_C))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(2,2,1)
    plt.title('原图像f')
    plt.imshow(f,cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,2)
    plt.title('傅里叶变换F')
    plt.imshow(abs(F),cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,3)
    plt.title('中心化傅里叶变换F_C')
    plt.imshow(abs(F_C),cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,4)
    plt.title('对数傅里叶变换S')
    plt.imshow(abs(S),cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()