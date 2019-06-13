#coding = utf-8
"""
https://www.kaggle.com/pestipeti/explanation-of-scoring-metric
https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss
评价指标：mIou 这个加到metric里
每个epoch 再重新算一下每类的miou 和 precision，计算方法看链接

"""

from keras import backend as K
import numpy as np
import config 

def iou(y_true,y_pred,label:int):
    """
    return the intersection over union for a given label.
    """
    y_true = y_true[:,:,:,-1:]
    y_pred = y_pred[:,:,:,-1:]

    y_true = K.cast(K.equal(y_true,label),K.floatx())
    y_pred = y_pred > 0.5
    y_pred = K.cast(y_pred,K.floatx())
    y_pred = K.cast(K.equal(y_pred,label),K.floatx())
    if(config.metric_iou_batch):
        intersecion = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersecion
    else:
        intersecion = K.sum(y_true * y_pred,axis=(1,2,3))
        union = K.sum(y_true,axis=(1,2,3)) + K.sum(y_pred,axis=(1,2,3)) - intersecion

    return K.mean((intersecion+1e-5)/(union+1e-5))


def build_iou(label:int,name:str=None):
    """
    build an intersection over union metric for labels list or label

    Args:
        label:a label list or label int
        name: an optional name for label

    Returns:
        a keras metric to evaluate Iou for the given label
    
    Note:
        label and name support list inputs for multiple labels
    """

    if isinstance(label,list):
        if(isinstance(name,list)):
            return [build_iou(l,n) for (l,n) in zip(label,name)]
        return [build_iou for i in label]
    
    def label_iou(y_true,y_pred):
        return iou(y_true,y_pred,label)

    if name is None:
        name = label
    label_iou.__name__ = 'iou_{}'.format(name)
    return label_iou



def mean_iou(y_true,y_pred):
    num_labels = K.int_shape(y_pred)[-1]
    total_iou = K.variable(0)
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true,y_pred,label)
    return total_iou / num_labels





