
# coding: utf-8

# In[8]:


import numpy as np
from model.utils.bbox import bbox_iou, bbox2t_param


# In[84]:


class AnchorTargetCreator(object):

    def __init__(self):
        self.n_sample = 256
        self.pos_iou_thresh = 0.7
        self.neg_iou_thresh = 0.3
        self.pos_ratio =0.5

    def make_anchor_target(self, anchor, gt_bbox, image_size):

        img_H, img_W = image_size
        n_anchor = len(anchor)

        index_inside_image = np.where((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) &                                      (anchor[:, 2] <= img_H) & (anchor[:, 3] <= img_W))[0]

        anchor = anchor[index_inside_image] # rule out anchors that are not fully included inside the image

        bbox_index_for_anchor, anchor_label = self._assign_target_and_label_for_anchor(anchor, gt_bbox)

        # create targer delta for bbox regression
        target_delta = bbox2t_param(anchor, gt_bbox[bbox_index_for_anchor])

        # expand the target_dalta and label to match original length of anchor
        target_delta = self._to_orignal_length(target_delta, n_anchor, index_inside_image, fill=0)
        anchor_label = self._to_orignal_length(anchor_label, n_anchor, index_inside_image, fill=-1)
        
        return target_delta, anchor_label


    def _assign_target_and_label_for_anchor(self, anchor, gt_bbox):
        
        #assign a label for each anchor, and the targer bbox index(with max iou) for each anchor.
        #label: 1 is positive, 0 is negative, -1 is don't care

        label = np.ones(anchor.shape[0]).astype(int)*-1   # init label with -1
        
        bbox_index_for_anchor, max_iou_for_anchor, anchor_index_for_bbox = self._anchor_bbox_ious(anchor, gt_bbox)

        # 1. assign anchor with 0 whose max_iou is small than neg_iou_thresh
        # 3. assign anchor with 1 whose max_iou is large than pos_iou_thresh
        label[max_iou_for_anchor < self.neg_iou_thresh] = 0
        label[max_iou_for_anchor>self.pos_iou_thresh] = 1

        #for each gt_bbox, assign anchor with 1 who has max iou with the gt_bbox
        label[anchor_index_for_bbox] = 1

        n_pos = int(self.n_sample * self.pos_ratio)  # default: 128
        n_neg = int(self.n_sample - np.sum(label==1))
        label = self._subsample_labels(label,n_pos, 1 )
        label = self._subsample_labels(label,n_neg, 0)
        
        return bbox_index_for_anchor, label
    
    def _subsample_labels(self,label,threshold,choice):
        index = np.where(label == choice)[0]
        if len(index) > threshold:
            disable_index = np.random.choice(index, size=(len(index) - threshold), replace=False)
            label[disable_index] = -1
        return label

    def _anchor_bbox_ious(self, anchor, gt_bbox):
        iou = bbox_iou(anchor, gt_bbox)
        
        anchor_index_for_bbox = iou.argmax(axis=0)
        bbox_index_for_anchor = iou.argmax(axis=1)
        max_iou_for_anchor = iou.max(axis=1)

        return bbox_index_for_anchor, max_iou_for_anchor, anchor_index_for_bbox

    def _to_orignal_length(self, data, length, index, fill):
        shape = list(data.shape)
        shape[0] = length
        ret = np.empty(shape, dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
        return ret

