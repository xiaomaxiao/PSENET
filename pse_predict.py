import numpy as np 
import cv2
import keras 
from models.psenet import psenet
from tool.utils import ufunc_4 , scale_expand_kernels ,fit_minarearectange,fit_boundingRect_2,text_porposcal

inputs = keras.layers.Input(shape = (None,None,3))
outputs = psenet(inputs)
model = keras.models.Model(inputs,outputs)
model.load_weights('resent50-190219_BLINEAR-iou8604.hdf5')

MIN_LEN = 640 
MAX_LEN = 1240

def predict(images):
    h,w = images.shape[0:2]
    if(w<h):
        if(h<MIN_LEN):
            scale = 1.0 * MIN_LEN / h
            w = w * scale 
            h = MIN_LEN
        elif(h>MAX_LEN):
            scale = 1.0 * MAX_LEN / h 
            w = w * scale 
            h = MAX_LEN
    elif(h<=w ):
        if(w<MIN_LEN):
            scale = 1.0 * MIN_LEN /w
            h = scale * h 
            w = MIN_LEN 
        elif(w>MAX_LEN):
            scale = 1.0 * MAX_LEN /w
            h = scale * w 
            h = MAX_LEN


    w = int(w //32 * 32)
    h = int(h//32 * 32)

    scalex = images.shape[1] / w
    scaley = images.shape[0] / h

    images = cv2.resize(images,(w,h),cv2.INTER_AREA)
    images = np.reshape(images,(1,h,w,3))   

    res = model.predict(images[0:1,:,:,:])

    res1 = res[0]
    res1[res1>0.9]= 1
    res1[res1<=0.9]= 0
    newres1 = []
    for i in [2,4]:
        n = np.logical_and(res1[:,:,5],res1[:,:,i]) * 255
        newres1.append(n)
    newres1.append(res1[:,:,5]*255)

    num_label,labelimage = scale_expand_kernels(newres1,filter=False)
    rects = fit_boundingRect_2(num_label,labelimage)

    im = np.copy(images[0])
    for rt in rects:
        cv2.rectangle(im,(rt[0]*2,rt[1]*2),(rt[2]*2,rt[3]*2),(0,255,0),2)
   
    g = text_porposcal(rects,res1.shape[1],max_dist=8,threshold_overlap_v=0.3)
    rects = g.get_text_line()

    for rt in rects:
        cv2.rectangle(im,(rt[0]*2,rt[1]*2-2),(rt[2]*2,rt[3]*2),(0,0,255),2)
    cv2.imwrite('test.png',im)

    return rects 


images = cv2.imread(r'2019_05_05_17_45_06_268.png')

images = cv2.imread(r'2019_05_05_17_45_06_273.png')

rects = predict(images)
