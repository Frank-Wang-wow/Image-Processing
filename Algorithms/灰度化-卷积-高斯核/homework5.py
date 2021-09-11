''' 调用上面实现的函数，对于问题 1 和 2 中的灰度图像（cameraman, einstein, 以及 lena512color
和 mandril_color 对应的 NTSC 转换后的灰度图像）进行高斯滤波，采用σ=1，2，3，5。任
选一种像素填补方案。
对于σ=1 下的结果，与直接调用相关函数的结果进行比较（可以简单计算差值图像）。然后，
任选两幅图像，比较其他参数条件不变的情况下像素复制和补零下滤波结果在边界上的差
别。 '''

import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import math
from homeworkf import twodConv,  gaussKernel, rgb1gray
 
if __name__ == '__main__':
    img = cv2.imread(r'lena512color.tiff')  ##读取图片
    img = rgb1gray(img,'NTSC')
    w,h = img.shape

    ###########σ=1，2，3，5#############
    for i,sig in enumerate([1,2,3,5]):
        G = gaussKernel(sig)  ##调用高斯卷积核生成函数
        img_out = twodConv(img,G,"replicate") ##调用卷积操作函数
        ##显示原始图片及卷积操作后的图片 
        plt.subplot(2,2,i+1)
        plt.imshow(img_out,cmap="gray") #卷积处理输出图片
        title = 'sig='+str(sig)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show() 

    #######cv2.getGaussianKernel库函数对比#################
    G = gaussKernel(1)  ##调用高斯卷积核生成函数
    img_out = twodConv(img,G,"replicate") ##调用卷积操作函数
    img_out1 = cv2.GaussianBlur(img,(7,7),1)
    err = cv2.absdiff(img_out,img_out1)
    err = np.array(err,dtype='uint8')
    #img_out1 = twodConv(img,G1,"replicate") ##调用卷积操作函数
    plt.subplot(1,2,1)
    plt.imshow(img_out1,cmap="gray") #卷积处理输出图片
    plt.title("cv2_conv")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(err,cmap="gray") #卷积处理输出图片
    print(np.max(err))
    plt.title("error")
    plt.xticks([])
    plt.yticks([])
    plt.show() 

    # ########比较补零和复制##########
    G1 = gaussKernel(1)  ##调用高斯卷积核生成函数
    img_out11 = twodConv(img,G1,"replicate") ##调用卷积操作函数, 复制方式
    img_out12 = twodConv(img,G1) ##调用卷积操作函数, 补零方式
    err1 = cv2.absdiff(img_out11,img_out12)

    bound_err_up = err1[0]
    bound_err_down = err1[w-1]
    bound_err_left = err1[:,0]
    bound_err_right = err1[:,h-1]
    x = np.linspace(0,len(bound_err_up),len(bound_err_up))
    plt.plot(x,bound_err_up,label='row0')
    plt.plot(x,bound_err_down,label='row512')
    plt.title('boundry difference')
    plt.xlabel('piex')
    plt.ylabel('diffence')
    plt.legend()
    plt.show()
    x = np.linspace(0,len(bound_err_left),len(bound_err_left))
    plt.plot(x,bound_err_left,label='column0')
    plt.plot(x,bound_err_right,label='column512')
    plt.title('boundry difference')
    plt.xlabel('piex')
    plt.ylabel('diffence')
    plt.legend()
    plt.show()
    plt.subplot(1,3,1)
    plt.imshow(img_out11,cmap="gray") #卷积处理输出图片
    plt.title("conv_replicate")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(img_out12,cmap="gray") #卷积处理输出图片
    plt.title("conv_zero")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.imshow(img_out12-img_out11,cmap="gray") #卷积处理输出图片
    plt.title("error")
    plt.xticks([])
    plt.yticks([])
    plt.show()  
    ''' G1X = cv2.getGaussianKernel(7, 1)
    G1Y = cv2.getGaussianKernel(7, 1)
    G1 = np.multiply(G1X,G1Y.T)
    print("ku",G1,np.sum(G1)) '''