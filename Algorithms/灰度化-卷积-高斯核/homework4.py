''' 实现一个高斯滤波核函数 w = gaussKernel(sig，m)，其中 sig 对应于高斯函数定义中的σ,w
的大小为 m×m。请注意，这里如果 m 没有提供，需要进行计算确定。如果 m 已提供但过小，
应给出警告信息提示。w 要求归一化，即全部元素加起来和为 1。 '''

import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import math
from homeworkf import twodConv

def gaussKernel(sig,*m): #高斯滤波核函数
    if  len(m):         #判断m是否提供
        m = m[0]           #获取m
        if m < (round(3*sig)*2+1):  #判断m是否太小
            print("warning:sigma is too small!")
    else:
        m = round(3*sig)*2+1 #计算确定m

    k = m//2
    G = np.zeros((m,m))
    s = 2*(sig**2)
    for i in range(m):
        for j in range(m):
            x = i - k
            y = j - k
            G[i,j] = np.exp(-(x**2+y**2)/s)/(s*np.pi)
    return G/np.sum(G)

if __name__ == '__main__':
    img = cv2.imread(r'mandril_color.tif',0)  ##读取图片
    G = gaussKernel(1)  ##调用高斯卷积核生成函数
    img_out = twodConv(img,G,"replicate") ##调用卷积操作函数
    ''' 显示原始图片及卷积操作后的图片 '''
    plt.subplot(1,2,1)
    plt.imshow(img,cmap="gray") #原始图像
    plt.title('origin')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(img_out,cmap="gray") #卷积处理输出图片
    plt.title('img_out')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    #cv2库
    # G1X = cv2.getGaussianKernel(7, 1)
    # G1Y = cv2.getGaussianKernel(7, 1)
    # G1 = np.multiply(G1X,G1Y.T)
    # img_out1 = twodConv(img,G1,"replicate")
    # err = cv2.absdiff(img_out,img_out1)
    # err = np.array(err,dtype='uint8')
    # plt.imshow(err,cmap="gray")
    # plt.show() 
    """ plt.imshow(abs(img_out-img_out1),cmap="gray")
    plt.show() """