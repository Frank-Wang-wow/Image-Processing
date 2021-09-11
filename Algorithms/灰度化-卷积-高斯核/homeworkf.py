import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import math

def scanLine4e(f, I, loc):
    if loc == 'row':
        s = f[I]
    if loc == 'column':
        s = f[:,I]
    return s

def rgb1gray(f, method="NTSC"): ##灰度化函数
    if method == 'average':
        gray = (f[:,:,0]+f[:,:,1]+f[:,:,2])/3
    else:
        gray = 0.2989*f[:,:,0]+0.5870*f[:,:,1]+0.1140*f[:,:,2]
    return gray

#####卷积函数##############################
def twodConv(f, W, pad="zero"):
    L,h = f.shape
    wW,hW = W.shape
    wp,hp = wW//2,hW//2

    """ padding """
    row_pad_up = np.zeros((wp,h))
    row_pad_down = np.zeros((wp,h))
    col_pad_left = np.zeros((L+2*wp,hp))
    col_pad_right = np.zeros((L+2*wp,hp)) ##扩充的行列块初始化全为零
    if pad == "replicate":
        for i in range(wp):
            row_pad_up[i] = f[0]
            row_pad_down[i] = f[L-1]
        img_p = np.row_stack((row_pad_up,f,row_pad_down))
        for i in range(hp):
            col_pad_left[:,i] = img_p[:,0]
            col_pad_right[:,i] = img_p[:,h-1]
        img_p = np.column_stack((col_pad_left,img_p,col_pad_right))
#        img_p = np.pad(f,((wp,wp),(hp,hp)),'edge') 
    else:
        img_p = np.row_stack((row_pad_up,f,row_pad_down))
        img_p = np.column_stack((col_pad_left,img_p,col_pad_right))
#        img_p = np.pad(f,pad_width=((wp,wp),(hp,hp)),mode="constant",constant_values=(0, 0))

    """ 卷积操作 """
    out_img = np.zeros((L,h)) #输出图像初始化
    w_img = np.zeros((wW,hW)) #定义一个原始图像中与卷积核对应矩阵
    W = np.flip(W)
    W = np.flip(W,1) ##卷积核旋转180度
    for i in range(L):
        for j in range(h):
            w_img = img_p[i:i+wW,j:j+hW]
            out_img[i,j] = np.sum(W*w_img)
    out_img = np.clip(out_img,0,255)
    return out_img

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