# -*- coding: utf-8 -*-
"""

Adaptive Shifted Weighted Mean Filter (ASWMF)

"""

from ReWeightedMean import reweightmean
import numpy as np
import cv2
from skimage.util import random_noise
from sewar import psnr,uqi,vifp,mse

def ASWMF(im):
    h = 3;
    I = np.pad(im, (h,h), 'symmetric')
    for i in range(h + 1,I.shape[0] - h):
        for j in range(h + 1,I.shape[1] - h):
            if I[i,j] == 0 or I[i,j] == 255:
                I[i,j] = reweightmean(I, i, j, h) 

    oim = I[h + 1:I.shape[0] - h, h + 1:I.shape[1] - h]
    return oim

#Test the image

input_image = cv2.imread('C:/Users/Abhi/Downloads/IMG_9476.jpg')
image_gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
n = np.uint8(random_noise(image_gray, mode = 's&p',amount = .2) * 255)

a = ASWMF(n)

#To avoid the 2 pixel loss 
d = (a.shape[1] + 1,a.shape[0] + 1)
a = cv2.resize(a,d,interpolation = cv2.INTER_LINEAR)

cv2.imwrite('Denoised_image_20%.png',a)