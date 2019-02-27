


import tensorflow as tf 
import keras
from keras.layers import Conv2D , ReLU , BatchNormalization , Input
from keras.layers import Concatenate,Activation,Reshape,Lambda,UpSampling2D,Add
from keras.models import Model
from keras import regularizers
from models.resnet import resnet_v1_50_fn 
import config 
import numpy as np 
class resize_image(keras.layers.Layer):
    
    def __init__(self,target_tensor_shape,target_int_shape,*args,**kwargs):
        self.target_tensor_shape = target_tensor_shape
        self.target_int_shape = target_int_shape
        super(resize_image,self).__init__(*args,**kwargs)
    
    def call(self,input_tensor,**kwargs):
        print(self.target_int_shape)
        return tf.image.resize_images(input_tensor,(self.target_tensor_shape[0],self.target_tensor_shape[1]),method = tf.image.ResizeMethod.BILINEAR)

    def compute_output_shape(self,input_shape):       
        return (input_shape[0],) + (self.target_int_shape[0],self.target_int_shape[1]) + (input_shape[-1],)


def conv_bn_relu(input_tensor,filters , kernel_size = 3 ,bn = True ,
                relu= True , isTraining = True,weight_decay=1e-6):
    '''
    conv2d + bn + relu
    notice : 
        isTraining : if finetune model should set False
        ? wether add l2 regularizer?
    '''
    x = Conv2D(filters,kernel_size,strides=(1,1),
               padding='same',kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    if(bn):
        x = BatchNormalization(axis=-1)(x)

    if(relu):
        x = Activation('relu')(x)
    return x



def upsample_conv(input_tensor,concat_tensor,filters,type='resize',kernel_size=3):
    '''
    upsmaple : use resize image + conv  or transpose conv or unsmaple + conv
    '''

    t_tensor_shape = keras.backend.shape(concat_tensor)[1:3]
    t_int_shape = keras.backend.int_shape(concat_tensor)[1:3]

    if(type == 'resize'):
        output_image = resize_image(t_tensor_shape,t_int_shape)(input_tensor)
    else:
        raise ValueError('upsample_conv type not in [resize,...]')
    
    #todo conv again?

    #concat two layers
    output_image = Concatenate(axis=3)([output_image,concat_tensor])
    #output_image =Add()([output_image,concat_tensor])

    output_image = conv_bn_relu(output_image,filters)

    return output_image
    

def FPN(blocks,type= 'resize'): 
    #force max channels to config.max_depth(default 256) to reduce memory usage
    for i,l in enumerate(blocks):
        if(l.shape.as_list()[-1] > config.max_depth):
            blocks[i] = Conv2D(config.max_depth,kernel_size=1,padding='same')(l)
 
    # get four 256 channels featuremaps (i.e. P2, P3, P4, P5) from the backbone.
    PN = []
    output_tensor = blocks[0]
    PN.append(output_tensor)
    for i in range(1,len(blocks)):
        output_tensor = upsample_conv(output_tensor,blocks[i],config.upsample_filters[i-1],type)
        PN.append(output_tensor)

    #[P5,P4,P3,P2]
    return PN
    
def FC_SN(PN):

    # we fuse the four feature maps to get feature map F with 1024 channels via the function
    #C(·) as: F = C(P2, P3, P4, P5) = P2 || Up×2(P3) || Up×4(P4) || Up×8 (P5), where “k” refers to the concatenation and Up×2
    #(·), Up×4 (·), Up×8 (·) refer to 2, 4, 8 times upsampling
    P2 = PN[-1]
    t_tensor_shape = keras.backend.shape(P2)[1:3]
    t_int_shape = keras.backend.int_shape(P2)[1:3]

    for i in range(len(PN)-1):
        PN[i] = resize_image(t_tensor_shape,t_int_shape)(PN[i])

    F = Concatenate(-1)(PN)

    #F is fed into Conv(3, 3)-BN-ReLU layers and is reduced to 256 channels.
    F = conv_bn_relu(F,256)

    #Next, it passes through multiple Conv(1, 1)-Up-Sigmoid layers and produces n segmentation results
    #S1, S2, ..., Sn. 
    SN = Conv2D(config.SN,(1,1))(F)

    scale = 1
    if(config.ns == 2):
        scale = 1

    new_shape = t_tensor_shape
    new_shape *= tf.constant(np.array([scale, scale], dtype='int32'))
    if t_int_shape[0] is None:
        new_height = None
    else:
        new_height = t_int_shape[0] * scale
    if t_int_shape[1] is None:
        new_width = None
    else:
        new_width = t_int_shape[1] * scale

    SN = resize_image(new_shape,(new_height,new_width))(SN)
    SN = Activation('sigmoid')(SN)

    return SN


def psenet(input_tensor,backbone = 'resnet50'):
    if backbone=='resnet50':
        blocks = resnet_v1_50_fn(input_tensor)
    PN = FPN(blocks)
    SN = FC_SN(PN)
    return SN



#input  = Input((1184,800,3))
#input  = Input((1184,800,3))
#output = psenet(input)

#model = Model(input,output)

#model.summary()

#import time
#import numpy as np 
#img = np.random.randn(1,1184,800,3)
#for i in range(10):
#    t0  = time.time()
   
#    model.predict(img)
    
#    print(time.time()-t0)







