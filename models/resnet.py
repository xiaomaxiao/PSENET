


import keras 
from keras.applications import ResNet50
from keras.layers import Lambda
import tensorflow as tf 
import numpy as np 


#order B G R 
_VGG_MEANS = [103.939, 116.779, 123.68]

def mean_substraction(input_tensor, means=_VGG_MEANS):
    return tf.subtract(input_tensor, np.array(means)[None, None, None, :], name='MeanSubstraction')

def resnet_v1_50_fn(input,include_top=False,weight=None):

    target_shape = keras.backend.int_shape(input)
    input = Lambda(mean_substraction,output_shape=target_shape[1:4])(input) 

    resnet = ResNet50(input_tensor=input,include_top=include_top , weights =weight)

    ##set bn layer trainable = False
    #for layer in resnet.layers:
    #    if isinstance(layer,keras.layers.normalization.BatchNormalization):
    #        layer.trainable = False

    resnet.load_weights(r'models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # get layers from block 1 2 3 4 5
    b5 = resnet.get_layer('activation_49').output
    b4 = resnet.get_layer('activation_37').output
    b3 = resnet.get_layer('activation_19').output
    b2 = resnet.get_layer('activation_7').output
    b1 = resnet.get_layer('activation_1').output
    return [b5,b4,b3,b2,b1]


def resnet_v1_101_fn(input,include_top=False,weight=None):
    target_shape = keras.backend.int_shape(input)
    input = Lambda(mean_substraction,output_shape=target_shape[1:4])(input) 
    

    resnet = ResNet101(include_top=False,input_tensor=input,weights = None)
    resnet.load_weights(r'models\resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5')
    b5 = resnet.get_layer('res5c_relu').output
    b4 = resnet.get_layer('res4b22_relu').output
    b3 = resnet.get_layer('res3b2_relu').output
    b2 = resnet.get_layer('res2c_relu').output
    b1 = resnet.get_layer('conv1_relu').output
    return resnet,[b5,b4,b3,b2,b1]

