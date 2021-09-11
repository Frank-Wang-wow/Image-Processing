""" 以lena图像为例，编程实现小波域维纳滤波（具体算法见十二讲ppt）
小波变换可以使用matlab自带的dwt2. """
""" 'gaussian'  Gaussian-distributed additive noise.  
    - 'localvar'  Gaussian-distributed additive noise, with specified  
                  local variance at each point of `image`.  
    - 'poisson'   Poisson-distributed noise generated from the data.  
    - 'salt'      Replaces random pixels with 1.  
    - 'pepper'    Replaces random pixels with 0 (for unsigned images) or  
                  -1 (for signed images).  
    - 's&p'       Replaces random pixels with either 1 or `low_val`, where  
                  `low_val` is 0 for unsigned images or -1 for signed  
                  images.  
    - 'speckle'   Multiplicative noise using out = image + n*image, where  
                  n is uniform noise with specified mean & variance.   """

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import cv2
from skimage import morphology,util
import skimage
from pywt import dwt2,idwt2

def winner(img):
    cA,(cH,cV,cD)=dwt2(img,'haar')
    deta_n = (np.median(abs(cD))/0.6745)**2
    M = cA.shape[0]*cA.shape[1]
    deta_h = np.sum(cH**2)/M
    deta_v = np.sum(cV**2)/M
    deta_d = np.sum(cD**2)/M
    deta_a = np.sum(cA**2)/M
    x_h = (deta_h/(deta_h+deta_n))*cH
    x_v = (deta_v/(deta_v+deta_n))*cV
    x_d = (deta_d/(deta_d+deta_n))*cD
    x_a = (deta_a/(deta_a+deta_h))*cA
    rimg = idwt2((cA,(x_h,x_v,x_d)),'haar')
    return rimg

if __name__ == "__main__":
    img = cv2.imread(r'lena512color.tiff',0)  ##读取图片
    print(np.max(img))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(3,2,1)
    plt.title('原图')
    plt.imshow(img,cmap='gray')
    plt.subplot(3,2,2)
    plt.hist(img.ravel(),256,[0,256])
    plt.title('灰度直方图')
    img_noise = util.random_noise(img, mode='gaussian', seed=None, clip=True)*255
    print(np.max(img_noise))
    plt.subplot(3,2,3)
    plt.title('含噪声图片')
    plt.imshow(img_noise,cmap='gray')
    plt.subplot(3,2,4)
    plt.hist(img_noise.ravel(),256,[0,256])
    plt.title('灰度直方图')
    cA,(cH,cV,cD)=dwt2(img,'haar')
    rimg = winner(img_noise)
    plt.subplot(3,2,5)
    plt.title('维纳滤波结果')
    plt.imshow(rimg,cmap='gray')
    plt.subplot(3,2,6)
    plt.hist(rimg.ravel(),256,[0,256])
    plt.title('灰度直方图')
    plt.show()

    img_log = np.log2(img)
    plt.imshow(int(img_log),cmap='gray')
    plt.show()


    
    