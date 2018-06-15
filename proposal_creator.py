
# coding: utf-8

# In[1]:


import numpy as np
from model.utils.bbox import t_param2bbox
from model.utils.nms import nms
from model.utils.anchor_target_creator import AnchorTargetCreator

class ProposalCreator(object):

    def __init__(self):
        self.nms_thresh = 0.7
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_post_nms = 300
        self.min_roi_size = 16

    def make_proposal(self, anchor, t_param, score, image_size, is_training):
 
        if is_training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        
        # 1. clip the roi into the size of image
        roi = t_param2bbox(anchor, t_param)
        roi[:,slice(0,4,2)] = np.clip(roi[:,slice(0,4,2)], a_min=0, a_max=image_size[0])
        roi[:,slice(1,4,2)] = np.clip(roi[:,slice(1,4,2)], a_min=0, a_max=image_size[1])
        
        # 2. remove roi where H or W is less than min_roi_size
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        mask = np.where((hs >= self.min_roi_size) & (ws >= self.min_roi_size))[0]
        roi = roi[mask, :]
        score = score[mask]

        # 3. keep top n_pre_nms rois according to score, and the left roi are sorted according to score
        order = score.argsort()[::-1]
        order = order[:n_pre_nms]
        roi = roi[order,:]
        
        # 4. apply nms, ans keep top n_post_nms roi
        # note that roi is already sorted according to its score value
        keep = nms(roi, self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep,:]

        return roi




