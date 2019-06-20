import cv2
import numpy as np 


sess = None
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open('psenet.pb','rb') as f :
        output_graph_def.ParseFromString(f.read())
        _= tf.import_graph_def(output_graph_def,name='')
        
    sess =  tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    ip = sess.graph.get_tensor_by_name("input_1:0")
    op = sess.graph.get_tensor_by_name("activation_55/Sigmoid:0")




def predict(images):
    h,w = images.shape[0:2]
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
    h = int(h //32 * 32)

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

