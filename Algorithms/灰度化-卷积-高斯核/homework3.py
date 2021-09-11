""" 实现一个函数 g = twodConv(f, w), 其中 f 是一个灰度源图像，w 是一个矩形卷积核。要求输
出图像 g 与源图像 f 大小（也就是像素的行数和列数）一致。请注意，为满足这一要求，对
于源图像f需要进行边界像素填补(padding)。这里请实现两种方案。第一种方案是像素复制，
对应的选项定义为’replicate’，填补的像素拷贝与其最近的图像边界像素灰度。第二种方案是
补零，对应的选项定义为’zero’, 填补的像素灰度为 0. 将第二种方案设置为缺省选择。 """


import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

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
    #W = np.flip(W,1) ##卷积核旋转180度
    for i in range(L):
        for j in range(h):
            w_img = img_p[i:i+wW,j:j+hW]
            out_img[i,j] = np.sum(W*w_img)
    out_img = np.clip(out_img,0,255)
    return out_img

if __name__ == '__main__':
    img = cv2.imread(r'lena512color.tiff',0)  ##读取图片
    print(type(img))
    """ [[0,0,0],[0,1,0],[0,0,0]],[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]平均值滤波 [[-1,0,1],[-1,0,1],[-1,0,1]]水平梯度
    [[-1,-1,-1],[0,0,0],[1,1,1]]垂直梯度 [[1,2,1],[2,4,2],[1,2,1]]/16高斯平滑 [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
    """
    
    conv = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    # conv = np.array([[-1,-1,-1],[0,0,0],[1,1,1]]) #卷积核定义
    img_out = twodConv(img,conv,"replicate")
    img_out1 = twodConv(img,conv)
    conv = np.flip(conv)
    print(conv,img_out1)
    res_cv2 = cv2.filter2D(img,-1,conv)    #调用cv卷积核库函数结构
    plt.subplot(2,2,1)
    plt.imshow(img,cmap="gray") #原始图像
    plt.title('origin')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,2)
    plt.imshow(img_out,cmap="gray") #卷积处理输出图片
    plt.title('con_replicate')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,3)
    plt.imshow(img_out1,cmap="gray") #卷积处理输出图片
    plt.title('con_zero')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,2,4)
    plt.imshow(res_cv2,cmap="gray") #卷积处理输出图片
    plt.title('con_cv2')
    plt.xticks([])
    plt.yticks([])
    plt.show()


