import os
import sys
import cv2
import time
import numpy as np 
import tensorflow as tf 

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
#sys.path.append(os.getcwd() + '/psenet')

from utils import scale_expand_kernels ,text_porposcal , fit_boundingRect_cpp ,fit_minarearectange_cpp
from utils import calc_vote_angle , fit_boundingRect_warp_cpp

def predict(images,angle = False):
    a = time.time()
    MIN_LEN = 32
    MAX_LEN = 1500
    h, w = images.shape[0:2]
    if(w<h):
        if(w<MIN_LEN):
            scale = 1.0 * MIN_LEN / w
            h = h * scale 
            w = MIN_LEN
        elif(h>MAX_LEN):
            scale = 1.0 * MAX_LEN / h 
            w = w * scale if w * scale > MIN_LEN else MIN_LEN
            h = MAX_LEN
    elif(h<=w ):
        if(h<MIN_LEN):
            scale = 1.0 * MIN_LEN / h
            w = scale * w
            h = MIN_LEN 
        elif(w>MAX_LEN):
            scale = 1.0 * MAX_LEN / w
            h = scale * h if scale * h >  MIN_LEN else MIN_LEN
            w = MAX_LEN

    w = int(w //32 * 32)
    h = int(h//32 * 32)

    scalex = images.shape[1] / w
    scaley = images.shape[0] / h

    images = cv2.resize(images,(w,h),cv2.INTER_AREA)
    images = np.reshape(images,(1,h,w,3))   

    res = sess.run([op], feed_dict={ip:images})
    b =time.time()
    print('pse的模型预测耗时：', str(b-a))
    res1 = res[0][0]
    res1[res1>0.9]= 1
    res1[res1<=0.9]= 0
    newres1 = []
    for i in range(0,5):
        n = np.logical_and(res1[:,:,5],res1[:,:,i]) * 255
        n = n.astype('int32')
        newres1.append(n)
    newres1.append((res1[:,:,5]*255).astype('int32'))

    num_label,labelimage = scale_expand_kernels(newres1,filter=False)

    if(angle == False):
        rects = fit_boundingRect_cpp(num_label-1,labelimage)
        g = text_porposcal(rects,res1.shape[1],max_dist=20,threshold_overlap_v=0.5)
        rects = g.get_text_line()
    else:
        #计算角度
        angle = calc_vote_angle(newres1[-1])
        h,w = labelimage.shape[0:2]
        M = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
        neg_M = cv2.getRotationMatrix2D((w/2,h/2),-angle,1.0)

        rects = fit_boundingRect_warp_cpp(num_label-1,labelimage,M)
        g = text_porposcal(rects,res1.shape[1],max_dist=20,threshold_overlap_v=0.5)
        rects = g.get_text_line()
        rects = np.array(rects)
        if(rects.shape[0]>0):
            rects = cv2.transform(np.array(rects),neg_M)

    if(rects.shape[0]>0):
        rects = rects.reshape(-1,8)

    c = time.time()
    print('pse的连接部分耗时：', str(c-b))
    results = []
    for rt in rects:
        rt[0] = rt[0] * 2 * scalex
        rt[1] = rt[1] * 2 * scaley
        rt[2] = rt[2] * 2 * scalex 
        rt[3] = rt[3] * 2 * scaley
        rt[4] = rt[4] * 2 * scalex
        rt[5] = rt[5] * 2 * scaley
        rt[6] = rt[6] * 2 * scalex
        rt[7] = rt[7] * 2 * scaley
        results.append(rt)
    return results

def draw_boxes(img, boxes):
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    return img

output_graph_def = tf.GraphDef()
with open('psenet/psenet.pb','rb') as f :
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name='')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ip = sess.graph.get_tensor_by_name("input_1:0")
op = sess.graph.get_tensor_by_name("activation_55/Sigmoid:0")


if __name__ == '__main__':
    images = cv2.imread('/data/ocr_train_ctpn/page1.jpg')
    print(images.shape)
    sess, IP, OP = get_ip_op()
    rects = predict(images, sess, IP, OP)

    draw_img = draw_boxes(images, rects)
    cv2.imwrite('imag.jpg', draw_img)
   
    cv2.namedWindow("image", 0)
    cv2.resizeWindow('image', 800, 900)
    cv2.imshow('image', draw_img)
    cv2.waitKey(0)

