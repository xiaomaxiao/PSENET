
import numpy as np 
import keras 

def cal_iou_rectangle(boxes,gtboxes):
    """
    calculates the intersection over union(IOU) between one predict boxes and some(k) grouth boxes
    # Arguments:
        boxes: np.array (K,4), (x1,y1,x2,y2)
        gtboxes : np.array (m,4) where m is the number of gtboxes
    return : the max iou of predict boxes
    """

    area_boxes = np.abs((boxes[:0] - boxes[:2]) *(boxes[:1] - boxes[:3]))
    area_gtboxes = np.abs((boxes[:0] - boxes[:2]) *(boxes[:1] - boxes[:3]))

    def cal_iou(box1,box1_area, boxes2,boxes2_area):
        """
        box1 [x1,y1,x2,y2]
        boxes2 [m][x1,y1,x2,y2]
        """
        x1 = np.maximum(box1[0],boxes2[:,0])
        x2 = np.minimum(box1[2],boxes2[:,2])
        y1 = np.maximum(box1[1],boxes2[:,1])
        y2 = np.minimum(box1[3],boxes2[:,3])
        intersection = np.maximum(x2-x1,0) * np.maximum(y2-y1,0)
        iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
        return iou

    overlaps = np.zeros((boxes.shape[0],gtboxes.shape[0]))

    for id , box in enumerate(boxes):
        overlaps[id][:] = cal_iou(box,area_boxes[id],gtboxes,area_gtboxes)

    return np.max(overlaps,axis=-1)


def PR(boxes,gtboxes,min_overlap=0.3):
    """
    calculates precision / recall
    # Arguments:
        boxes: np.array (K,4), (x1,y1,x2,y2)
        gtboxes : np.array (m,4) where m is the number of gtboxes
        min_overlap : the minmun overlap 
    return : PR
    """

    iou = cal_iou_rectangle(boxes,gtboxes)
    
    tp = 0 # true positive
    fp = 0 # false positve
    
    iou[iou>min_overlap] = 1
    iou[iou<=min_overlap] = 0 
    tp = np.count_nonzero(iou)
    fp = iou.shape[0] - tp 

    precision = (1.0 * tp) / (tp + fp)
    recall = (1.0 * tp) / gtboxes.shape[0]

    return precesion,recall


