
# coding: utf-8

# In[ ]:


import numpy as np
from model.utils.bbox import bbox_iou, bbox2t_param

class ProposalTargetCreator(object):
    # p:possitive n:negative r:rate t:threshold num:number
    def __init__(self):
        self.n_sample = 128
        self.p_r = 0.25
        self.p_t = 0.5
        self.n_t_high = 0.5
        self.n_t_low = 0.0
        
    def make_proposal_target(self, roi, gt_bbox, gt_label):
        
        # concate gt_bbox as part of roi to be chose
        roi = np.concatenate((roi, gt_bbox), axis=0)   

        num_p = int(self.n_sample * self.p_r)      #128 * 0.25

        iou = bbox_iou(roi, gt_bbox)          #(N1 * N2)
        bbox_index_for_roi = iou.argmax(axis=1)   #roi belongs to which gt-bbox
        max_iou_for_roi = iou.max(axis=1)         #the max iou value

        # note that bbox_label_for_roi include background, class 0 stand for backdround
        # object class change from 0 ~ n_class-1 to 1 ~ n_class
        bbox_label_for_roi = gt_label[bbox_index_for_roi] + 1
        
        # Select foreground(positive) RoIs as those with >= pos_iou_thresh IoU.
        p_idx = np.where(max_iou_for_roi >= self.p_t)[0]     #positive index of roi
        num_p_real = int(min(num_p, len(p_idx)))        #num of possitive roi > thresh
        if num_p_real > 0:
            p_idx = np.random.choice(p_idx, size=num_p_real, replace=False)
        
        # Select background(negative) RoIs as those within [neg_iou_thresh_low, neg_iou_thresh_high).
        n_idx = np.where((max_iou_for_roi >= self.n_t_low) & (max_iou_for_roi < self.n_t_high))[0]
        n_neg = self.n_sample - num_p_real
        num_n_real = int(min(n_neg, len(n_idx)))
        if num_n_real > 0:
            n_idx = np.random.choice(n_idx, size=num_n_real, replace=False)
        
        retain_idx = np.append(p_idx, n_idx)
        sample_roi = roi[retain_idx]           #selected roi, num = 128
        bbox_label_for_sample_roi = bbox_label_for_roi[retain_idx]
        bbox_label_for_sample_roi[num_p_real:] = 0   # set negative sample's label to background 0

        target_delta_for_sample_roi = bbox2t_param(sample_roi, gt_bbox[bbox_index_for_roi[retain_idx]])

        target_delta_for_sample_roi = (target_delta_for_sample_roi - np.array([0., 0., 0., 0.])) / np.array([0.1, 0.1, 0.2, 0.2])
        return sample_roi, target_delta_for_sample_roi, bbox_label_for_sample_roi

