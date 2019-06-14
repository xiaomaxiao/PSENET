#coding = utf-8

import argparse
import os 
import numpy as np 
import cv2
import keras 
import keras.backend as K
import tensorflow as tf
from models.psenet import psenet
from tool.utils import ufunc_4 , scale_expand_kernels ,fit_minarearectange,fit_boundingRect_2,text_porposcal



MIN_LEN = 640 
MAX_LEN = 1500

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

# 用于判断文件后缀
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    rsp = make_response(render_template('index.html'))
    rsp.headers['Access-Control-Allow-Origin'] = '*'
    return rsp



@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file==None or allowed_file(file.filename)==False:
                return jsonify([{'size':0,'text':'just suport jpg png'}])
            res = predict_file(file)
        except Exception as e :
            return jsonify([{'error':e}])
        return jsonify(res)
    return index()


def parser_argument():
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--gpu',
        type = str,
        default ='1'
    )
    parser.add_argument(
        '--port',
        type = int,
        default = '6000'
    )
    parser.add_argument(
        'modelfile',
        type = str,
        default ='./tf/resnet50.hdf5'
    )
    return parser.parse_args()

def __name__ == '__main__':
    args = parser_argument()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    K.set_session(session)

    inputs = keras.layers.Input(shape = (None,None,3))
    outputs = psenet(inputs)
    model = keras.models.Model(inputs,outputs)
    model.load_weights(args.modelfile)
    print('load weights from {} '.format(args.modelfile))
    
    port = int(os.environ.get('PORT'),args.port)
    app.run(host='0.0.0.0',port = port,threaded=True)

