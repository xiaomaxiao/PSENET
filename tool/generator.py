import sys
sys.path.insert(0,r'C:\jianweidata\ocr\psenet')


import threading
import os
import glob
import numpy as np 
import cv2
import config
from tool.utils import BatchIndices


class Generator():
    def __init__(self,dir,batch_size = 2 , istraining = True,
                 num_classes = 2,mirror=True,reshape=(640,640)):
        self.dir = dir 
        self.lock = threading.Lock()
        self.batch_size = batch_size
        self.shuffle =  istraining
        self.num_classes = num_classes
        self.mirror = mirror
        self.reshape = reshape  #(h,w)
        self.imagelist,self.labellist = self.list_dir(self.dir)
        self.batch_idx = BatchIndices(self.imagelist.shape[0],self.batch_size,self.shuffle)
    def num_classes(self):
        return self.num_classes

    def num_samples(self):
        return len(self.imagelist)

    def list_dir(self,dir):

        image =[]
        npy =[]

        imagesfile = glob.glob(os.path.join(dir,'*.jpg'))
        for i in imagesfile:
            npyfile = os.path.join(dir,'.'.join(os.path.basename(i).split('.')[:-1])+'.npy')
            imagefile = os.path.join(dir,i)
            if(os.path.exists(npyfile)):
                image.append(imagefile)
                npy.append(npyfile)
                
        return np.array(image),np.array(npy)



    def __next__(self):
        idx = next(self.batch_idx)
        try:
            images = []
            labels = []
            for i,j in zip(self.labellist[idx],self.imagelist[idx]):
                l = np.load(i).astype(np.uint8)
                img = cv2.imread(j)
                if(self.reshape):
                    lreshape = (int(self.reshape[0]/config.ns),int(self.reshape[1]/config.ns))
                    lns = np.zeros((lreshape[0],lreshape[1],config.n))
                    for c in range(config.n):
                        lns[:,:,c] =cv2.resize(l[:,:,c],(lreshape[1],lreshape[0]),interpolation=cv2.INTER_NEAREST)
                    l = lns
                    img = cv2.resize(img,(self.reshape[1],self.reshape[0]),interpolation=cv2.INTER_AREA)

                images.append(img)
                labels.append(l)

            images = np.array(images)
            labels = np.array(labels)
        
            seed = np.random.randint(0,100)
            if(self.mirror and  seed >90):
                images = images[:,::-1,::-1,:]
                labels = labels[:,::-1,::-1,:]
            elif(self.mirror and seed > 70):
                images = images[:,::-1,:,:]
                labels = labels[:,::-1,:,:]
            elif(self.mirror and seed > 50):
                images = images[:,:,::-1,:]
                labels = labels[:,:,::-1,:]
                
            return images, labels
        except Exception as e :
            print(e,self.imagelist[idx])
            self.__next__()

def test():
    gen = Generator(config.MIWI_2018_TEST_LABEL_DIR)

    images,labels = next(gen)
    import matplotlib.pyplot as plt 

    plt.imshow(images[1][:,:,::-1])

    plt.imshow(labels[0][:,:,5])


    z0 = np.count_nonzero(labels==0)
    z1 = np.count_nonzero(labels==1)
    print(z0+z1 == 2 * 320 * 320 * 6)

