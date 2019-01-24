import sys
sys.path.insert(0,r'E:\psenet-MTWI\PSENET')

import os 
import glob
import config 



def read_txt(file):
    with open(file,'r',encoding='utf-8') as f :
        lines = f.read()
    lines = lines.split('\n')
    gtbox =[]
    for line in lines:
        if(line==''):
            continue
        pts = line.split(',')[0:8]
        #convert str to int 
        x1 = round(float(pts[0]))
        y1 = round(float(pts[1]))
        x2 = round(float(pts[2]))
        y2 = round(float(pts[3]))
        x3 = round(float(pts[4]))
        y3 = round(float(pts[5]))
        x4 = round(float(pts[6]))
        y4 = round(float(pts[7]))

        gtbox.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    return gtbox

def read_dataset():
    files = glob.glob(os.path.join(config.MTWI_2018_TXT_DIR,'*.txt'))
    dataset={}
    for file in files:
        basename = '.'.join(os.path.basename(file).split('.')[:-1])
        imgname = os.path.join(config.MTWI_2018_IMG_DIR,basename+'.jpg')
        gtbox = read_txt(file)
        dataset[imgname] = gtbox
    return dataset


