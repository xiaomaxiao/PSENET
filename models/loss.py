import sys
sys.path.insert(0,r'E:\psenet-MTWI\PSENET')
import config
import keras.backend as K
import tensorflow as tf 

def build_loss(y_true,y_pred):
    '''
    build psenet loss refer to  section 3.4 

    Arg:
        y_true: the ground truth label. [batchsize,h,w,config.SN]
        y_pred : the predict label 

    return total loss
    '''
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    total_loss = 0.0
    Lc_loss = 0.0
    Ls_loss = 0.0

    y_true_Lc = y_true[:,:,:,-1:]
    y_pred_Lc = y_pred[:,:,:,-1:]

    y_true_Ls = y_true[:,:,:,:-1]
    y_pred_Ls = y_pred[:,:,:,:-1]

    #adopt ohem to Lc
    M = ohem_batch(y_true_Lc,y_pred_Lc)

    Lc_loss = 1 - dice_loss(y_true_Lc * M , y_pred_Lc * M)

    #ignore the pixels of non-text region 
    #in the segmentation result Sn to avoid a certain redundancy.
    W = y_pred_Lc > 0.5
    pos_mask = tf.cast(y_true_Lc,tf.bool)
    W =  tf.logical_or(pos_mask, W)
    W = tf.cast(W,tf.float32)

    for i in range(config.SN-1):
        Ls_loss += dice_loss(y_true_Ls[:,:,:,i:i+1] * W , y_pred_Ls[:,:,:,i:i+1] * W)
    Ls_loss = 1.0 - Ls_loss / (config.SN-1)

    #Ls_loss = tf.print(Ls_loss,['lc_loss.->',Lc_loss,'ls_loss->',Ls_loss])

    # λ balances the importance between Lc and Ls.
    total_loss = config.rate_lc_ls * Lc_loss + (1-config.rate_lc_ls) * Ls_loss
    return total_loss


def ohem_batch(y_true_Lc,y_pred_Lc):
    M = tf.map_fn(ohem_single,(y_true_Lc,y_pred_Lc),dtype = tf.float32)
    return tf.stack(M)

def ohem_single(s_Lc):
    s_y_ture_Lc,s_y_pred_Lc = s_Lc
    n_pos = K.sum(s_y_ture_Lc)
    #n_pos = tf.print(n_pos,['n_pos->',n_pos])

    def has_pos():
        n_max_neg = K.sum(tf.cast(s_y_ture_Lc>-1.0,tf.int32))

        #n_max_neg = tf.print(n_max_neg,['n_max_neg',n_max_neg])
        n_neg  = n_pos * config.rate_ohem
        n_neg = tf.cast(n_neg,tf.int32)
        n_neg = K.minimum(n_neg,n_max_neg)
        
        pos_mask = tf.cast(s_y_ture_Lc,tf.bool)
        neg_mask = tf.cast(tf.equal(pos_mask,False),tf.float32)
        neg = neg_mask * s_y_pred_Lc

        vals,_  = tf.nn.top_k(K.reshape(neg,(1,-1)),k = n_neg)
        threshold = vals[0][-1]
        
        #threshold = tf.print(threshold,['threshold->',threshold,
        #                                'n_neg>threshold->',K.sum(tf.cast(neg>0,tf.int32)),
        #                                's_y_pred_Lc',K.sum(tf.cast(s_y_pred_Lc>0.0,tf.int32)),
        #                                'neg->',neg,
        #                                'neg shape->',K.shape(neg)])

        mask =  tf.logical_or(pos_mask, neg>threshold)

        return tf.cast(mask,tf.float32) 
    def no_pos():
        mask = K.zeros_like(s_y_ture_Lc)
        return tf.cast(mask,tf.float32)

    return tf.cond(n_pos>0,has_pos,no_pos)

def dice_loss(y_true,y_pred,smooth = 1.0):
    intersection = K.sum(y_true * y_pred)
    #intersection = tf.print(intersection,[intersection],'intersection:')
    return  (2.0 * intersection + smooth ) / (K.sum(y_true) + K.sum(y_pred) + smooth)



#import numpy as np 

#y_random = np.zeros((2,640,640,6))

#y_true  = np.copy(y_random)
#y_true[:,320:420,320:420,:] = 1.0

#y_pred  = np.copy(y_random)
#y_pred[:,270:370,270:370,:] = 0.8



#loss = build_loss(y_true,y_pred)

#sess = tf.Session()
#print('loss:',sess.run(loss))



