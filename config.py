
max_depth = 256
upsample_filters = [128,128,128,128]  #from buttom to top 
SN = 6           # number of kernel scales

rate_lc_ls = 0.7    #balances the importance between Lc and Ls 
rate_ohem = 3       #positive ：negtive = 1:rate_ohem

m = 0.5  #the minimal scale ration , which the value is (0,1]
n = 6   #the number of kernel scales

ns = 2  #“1s”, “2s” and “4s” means the width and height of the output map are 1/1, 1/2 and 1/4 of the input


MTWI_2018_TXT_DIR = r'E:\psenet-MTWI\document\mtwi_2018_train\txt_train'
MTWI_2018_IMG_DIR = r'E:\psenet-MTWI\document\mtwi_2018_train\image_train'
MIWI_2018_TRAIN_LABEL_DIR = r'E:\psenet-MTWI\document\mtwi_2018_train\train_label'
MIWI_2018_TEST_LABEL_DIR = r'E:\psenet-MTWI\document\mtwi_2018_train\test_label'

label_to_id = {(255,255,255):0,(0,0,255):1}

data_gen_min_scales = 0.8
data_gen_max_scales = 2.0
data_gen_itter_scales = 0.3

#随机剪切 文字区域最小面积
data_gen_clip_min_area = 20*100


#dice loss
batch_loss = True

#metric iou 
metric_iou_batch = True


