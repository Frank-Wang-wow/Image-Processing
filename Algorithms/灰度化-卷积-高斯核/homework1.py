""" 问题 1 黑白图像灰度扫描
实现一个函数 s = scanLine4e(f, I, loc), 其中 f 是一个灰度图像，I 是一个整数，loc 是一个字
符串。当 loc 为’row’时，I 代表行数。当 loc 为’column’时，I 代表列数。输出 s 是对应的相
关行或者列的像素灰度矢量。
调用该函数，提取 cameraman.tif 和 einstein.tif 的中心行和中心列的像素灰度矢量并将扫描
得到的灰度序列绘制成图。 """

import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def scanLine4e(f, I, loc):
    if loc == 'row':
        s = f[I]
    if loc == 'column':
        s = f[:,I]
    return s
if __name__ == '__main__':
    gray = cv2.imread(r'einstein.tif',0)  ##读取图片
    w,h = gray.shape ##获取灰度图片的行数和列数
    s1 = scanLine4e(gray, w//2, 'row')#获取中心行序列
    s2 = scanLine4e(gray, h//2, 'column')#中心列
    X1 = range(h)
    X2 = range(w)
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  ##设置中午显示
    plt.plot(X1, s1,marker='.',label='中心行')  ##显示中心行序列
    plt.plot(X2, s2,marker='.',label='中心列')
    plt.title("gray line")
    plt.legend()
    plt.show()


