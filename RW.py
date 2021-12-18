# -*- coding: utf-8 -*-
"""

ReWeighted Mean function for ASWMF Filter

"""

import numpy as np


def reweightmean(img,ii,jj,hh):
    a = 10
    WD = img[ii-hh:ii+hh, jj-hh:jj+hh]
    d1 = diag1(WD)
    d2 = diag2(WD)
    d3 = othel(WD)
    newgrayvalue = np.uint8((sum(d1)+sum(d2)+a*sum(d3))/(np.size(d1)+np.size(d2)+a*np.size(d3)))
    return newgrayvalue

def diag1(img):
   res1 = np.array([])
   sz = img.shape[0]
   for ind in range(0,sz):
       
           if (img[ind,ind] != 0 or img[ind,ind] != 255):
               res1 = np.array([img[ind,ind]])
   return res1

def diag2(img):
    res2 = np.array([])
    sz = img.shape[0]
    for ind in range(0,sz):
        if (img[ind,sz-ind - 1] != 0 or img[ind,sz-ind - 1] != 255):
            res2 = np.array([img[ind,sz - ind - 1]])
    return res2
            
def othel(img):
    res3 = np.array([])
    sz = img.shape[0]
    for p in range(0,sz):
        for q in range(0,sz):
            if (img[p,q] != 0 and img[p,q] != 255 and p != q and p != sz - q + 1):
                res3 = np.array([img[p,q]])
    return res3

