""" 选做题 测试 更多图像的二维快速傅里叶变换 (10 分)
计算其他 5 幅图像的二维快速傅里叶变换：house.tif, house02.tif, lena_gray_512.tif,
lunar_surface.tif, characters_test_pattern.tif。注意，有些图像的尺寸不是 2 的整数次幂，需要
进行相应的像素填补处理。如果图像有多个通道可以选择其中的一个通道进行计算。 """

from question1 import dft2D
from question4 import ff_center
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft,fft2,fftshift
from math import log2,floor,ceil

def pad(img):
    w,h = img.shape
    if log2(w)==ceil(log2(w)) and log2(h)==ceil(log2(h)):
        return img
    else:
        wp = (2**(ceil(log2(w)))-w)//2
        hp = (2**(ceil(log2(h)))-h)//2
    row_pad = np.zeros((wp,h))
    col_pad = np.zeros((w+2*wp,hp))
    img_p = np.row_stack((row_pad,img,row_pad))
    img_pad = np.column_stack((col_pad,img_p,col_pad))
    return img_pad

if __name__ == "__main__":
    pictures = ['house.tif','house02.tif','lena_gray_512.tif','lunar_surface.tif','characters_test_pattern.tif']
    img = cv2.imread(pictures[1],0)
    for picture in pictures:
        img = cv2.imread(picture,0)
        img = pad(img)
        F = np.log(1+abs(fftshift(dft2D(img))))
        plt.subplot(1,2,1)
        plt.title(picture+'_oringin')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img,cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(F,cmap="gray")
        plt.xticks([])
        plt.yticks([])  
        plt.title(picture+'_FFT')
        plt.show()