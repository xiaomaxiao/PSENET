import threading
import numpy as np 
import csv
import os
import shutil
import cv2
import glob
from copy import deepcopy
from numba import autojit

class BatchIndices():
    def __init__(self,total,batchsize,trainable=True):
        self.n = total
        self.bs = batchsize
        self.shuffle = trainable
        self.lock = threading.Lock()
        self.reset()
    def reset(self):
        self.index = np.random.permutation(self.n) if self.shuffle==True else np.arange(0,self.n)
        self.curr = 0
    
    def __next__(self):
        with self.lock:
            if self.curr >=self.n:
                self.reset()
            rn = min(self.bs,self.n - self.curr)
            res = self.index[self.curr:self.curr+rn]
            self.curr += rn
            return res

def del_allfile(path):
    '''
    del all files in the specified directory
    '''
    filelist = glob.glob(os.path.join(path,'*.*'))
    for f in filelist:
        os.remove(os.path.join(path,f))



def convert_label_to_id(label2id,labelimg):
    '''
    convert label image to id npy
    param:
    labelimg - a label image with 3 channels
    label2id  - dict eg.{(0,0,0):0,(0,255,0):1,....}
    '''

    h,w = labelimg.shape[0],labelimg.shape[1]
    npy = np.zeros((h,w),'uint8')
    
    for i,j in label2id.items():
        idx = ((labelimg == i) * 1)
        idx = np.sum(idx,axis=2) >=3
        npy = npy + idx * j

    return npy


def convert_id_to_label(id,label2id):
    '''
    convet id numpy to label image 
    param:
    id          : numpy
    label2id  - dict eg.{(0,0,0):0,(0,255,0):1,....}
    return labelimage 
    '''
    h,w = id.shape[0],id.shape[1]

    labelimage = np.ones((h,w,3),'uint8') * 255
    for i,j in label2id.items():
        labelimage[np.where(id==j)] = i 

    return labelimage
 



@autojit
def ufunc_4(S1,S2,TAG):
    #indices 四邻域 x-1 x+1 y-1 y+1，如果等于TAG 则赋值为label
    for h in range(1,S1.shape[0]-1):
        for w in range(1,S1.shape[1]-1):
            label = S1[h][w]
            if(label!=0):
                if(S2[h][w-1] == TAG):                          
                    S2[h][w-1] = label
                if(S2[h][w+1] == TAG):                            
                    S2[h][w+1] = label
                if(S2[h-1][w] == TAG):                            
                    S2[h-1][w] = label
                if(S2[h+1][w] == TAG):                           
                    S2[h+1][w] = label
                    
def scale_expand_kernel(S1,S2):
    TAG = 10240                     
    S2[S2==255] = TAG
    mask = (S1!=0)
    S2[mask] = S1[mask]
    cond = True 
    while(cond):  
        before = np.count_nonzero(S1==0)
        ufunc_4(S1,S2,TAG)  
        S1[S2!=TAG] = S2[S2!=TAG]  
        after = np.count_nonzero(S1==0)
        if(before<=after):
            cond = False
       
    return S1

def scale_expand_kernels(kernels):
    '''
    args:
        kernels : S(0,1,2,..n) scale kernels , Sn is the largest kernel
    '''
    S = kernels[0]
    num_label,labelimage = cv2.connectedComponents(S.astype('uint8'))
    for Si in kernels[1:]:
        labelimage = scale_expand_kernel(labelimage,Si)
    return num_label,labelimage   

def fit_minarearectange(num_label,labelImage):
    rects= []
    for label in range(1,num_label+1):
        points = np.array(np.where(labelImage == label)[::-1]).T

        rect = cv2.minAreaRect(points)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        area = cv2.contourArea(rect)
        if(area<10):
            continue
        rects.append(rect)
    return rects

def save_MTWI_2108_resault(filename,rects,scalex=1.0,scaley=1.0):
    with open(filename,'w',encoding='utf-8') as f:
        for rect in rects:
            line = ''
            for r in rect:
                line += str(r[0] * scalex) + ',' + str(r[1] * scaley) + ','
            line = line[:-1] + '\n'
            f.writelines(line)

def fit_boundingRect(num_label,labelImage):
    rects= []
    for label in range(1,num_label+1):
        points = np.array(np.where(labelImage == label)[::-1]).T
        #rect = cv2.minAreaRect(points)
        #rect = cv2.boxPoints(rect)
        #rect = np.int0(rect)
        x,y,w,h = cv2.boundingRect(points)
        rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        rects.append(rect)
    return rects