
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def t_param2bbox(anc_box, t_param):
    
    anc_box_h = anc_box[:,2] - anc_box[:,0]  
    anc_box_w = anc_box[:,3] - anc_box[:,1]
    anc_box_x = anc_box[:,0] + anc_box_h/2 
    anc_box_y = anc_box[:,1] + anc_box_w/2
    
    gt_box_x = anc_box_x + anc_box_h*t_param[:,0] 
    gt_box_y = anc_box_y + anc_box_w*t_param[:,1] 
    gt_box_h = anc_box_h * np.exp(t_param[:,2])
    gt_box_w = anc_box_w * np.exp(t_param[:,3])
    
    gt_box_x_min = (gt_box_x - gt_box_h / 2).reshape([-1, 1])
    gt_box_y_min = (gt_box_y - gt_box_w / 2).reshape([-1, 1])
    gt_box_x_max = (gt_box_x + gt_box_h / 2).reshape([-1, 1])
    gt_box_y_max = (gt_box_y + gt_box_w / 2).reshape([-1, 1])
    
    gt_box = np.concatenate([gt_box_x_min, gt_box_y_min, gt_box_x_max, gt_box_y_max], axis=1)
    
    return gt_box


# In[9]:


def bbox2t_param(anc_box, gt_box):   #anchor bounding box, ground truth bounding box
    
    anc_h = anc_box[:, 2] - anc_box[:, 0] + 1.0
    anc_w = anc_box[:, 3] - anc_box[:, 1] + 1.0
    anc_ctr_x = anc_box[:, 0] + 0.5 * anc_h               #center x, y 
    anc_ctr_y = anc_box[:, 1] + 0.5 * anc_w

    gt_h = gt_box[:, 2] - gt_box[:, 0] + 1.0
    gt_w = gt_box[:, 3] - gt_box[:, 1] + 1.0
    gt_ctr_x = gt_box[:, 0] + 0.5 * gt_h
    gt_ctr_y = gt_box[:, 1] + 0.5 * gt_w

    dx = ((gt_ctr_x - anc_ctr_x) / anc_h).reshape([-1,1])
    dy = ((gt_ctr_y - anc_ctr_y) / anc_w).reshape([-1,1])
    dh = np.log(gt_h / anc_h).reshape([-1,1])
    dw = np.log(gt_w / anc_w).reshape([-1,1])
    t_param = np.concatenate([dx, dy, dh, dw], axis=1)
    
    return t_param

def bbox_iou(bbox1, bbox2):            #anchor, gt_bbox
    
    top_left = np.maximum(bbox1[:,None,:2], bbox2[:,:2])        # (N1,N2,2)
    bottom_right = np.minimum(bbox1[:,None,2:], bbox2[:,2:])    # (N1,N2,2)

    area_inter = np.prod(bottom_right-top_left,axis=2) * (top_left < bottom_right).all(axis=2)  # (N1,N2)
    area_1 = np.prod(bbox1[:,2:]-bbox1[:,:2], axis=1)   # (N1,)
    area_2 = np.prod(bbox2[:,2:]-bbox2[:,:2], axis=1)   # (N2,)
    iou = area_inter / (area_1[:,None] + area_2 - area_inter)   # (N1, N2)
    return iou

