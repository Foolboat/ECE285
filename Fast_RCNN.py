
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# In[ ]:


from model.utils.RoiPooling import roipooling
from model.utils.loss_func import reg_loss
from model.utils.bbox import t_param2bbox
from model.utils.nms import nms


# In[ ]:


class Fast_RCNN(nn.Module):
    def __init__(self, n_class_bg, roip_size, classifier):
#        n_class_bg: n_class plus background = n_class + 1
        super(Fast_RCNN, self).__init__()
        self.n_class_bg = n_class_bg  #21
        self.roip_size = roip_size    #7*7

        self.roip = roipooling(roip_size)
        self.classifier = classifier
        self.delta = nn.Linear(in_features=4096, out_features=n_class_bg*4)    # Note: predice a delta for each class
        self.score = nn.Linear(in_features=4096, out_features=n_class_bg)

        self._normal_init(self.delta, 0, 0.001)
        self._normal_init(self.score, 0, 0.01)

    def forward(self, feature_map, rois, image_size):          #(features, sample_roi, new_image_size) for loss


        feature_image_scale = feature_map.shape[2] / image_size[0]    #1/16
        
        # meet roi_pooling's input requirement
        temp = np.zeros((rois.shape[0], 1), dtype=rois.dtype)    #rois.shape[0], num of rois
        rois = np.concatenate([temp, rois], axis=1) 

        rois = Variable(torch.FloatTensor(rois))
        if torch.cuda.is_available():
            rois = rois.cuda()

        roipool_out = self.roip(feature_map, rois, scaling=feature_image_scale)  

        roipool_out = roipool_out.view(roipool_out.size(0), -1) # (N, 25088)   N=num of rois, 25088 = 512 * 7 * 7
        if torch.cuda.is_available():
            roipool_out = roipool_out.cuda()

        mid_output = self.classifier(roipool_out)   # (N, 4096)
        delta_per_class = self.delta(mid_output)    # (N, n_class_bg*4)
        score = self.score(mid_output)      # (N, n_class_bg)
       
        return delta_per_class, score

    def loss(self, score, delta_per_class, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi):

#         score: (N, 2)
#         delta_per_class: (N, 4*n_class_bg)   #delta from roi to 21 classes
#         target_delta_for_sample_roi: (N, 4)
#         bbox_bg_label_for_sample_roi: (N,)

       
        target_delta_for_sample_roi = Variable(torch.FloatTensor(target_delta_for_sample_roi))
        bbox_bg_label_for_sample_roi = Variable(torch.LongTensor(bbox_bg_label_for_sample_roi))
        if torch.cuda.is_available():
            target_delta_for_sample_roi = target_delta_for_sample_roi.cuda()
            bbox_bg_label_for_sample_roi = bbox_bg_label_for_sample_roi.cuda()

        n_sample = score.shape[0]
        delta_per_class = delta_per_class.view(n_sample, -1, 4)

        # get delta for roi w.r.t its corresponding bbox label
        index = torch.arange(0, n_sample).long()
        if torch.cuda.is_available():
            index = index.cuda()
        delta = delta_per_class[index, bbox_bg_label_for_sample_roi.data]   #roi 到最接近的gt_bbox所属类的delta

        head_reg_loss = reg_loss(delta, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi, 1)
        head_class_loss = F.cross_entropy(score, bbox_bg_label_for_sample_roi)

        return head_reg_loss + head_class_loss

    def predict(self, roi, delta_per_class, score, image_size, prob_threshold=0.5):
       
        roi = torch.FloatTensor(roi)
        if torch.cuda.is_available():
            roi = roi.cuda()
        delta_per_class = delta_per_class.data
        prob = F.softmax(score, dim=1).data  #(N, n_class_bg)

        delta_per_class = delta_per_class.view(-1, self.n_class_bg, 4)  #(N * 21 * 4)?
        
        delta_per_class = delta_per_class * torch.cuda.FloatTensor([0.1, 0.1, 0.2, 0.2]) + torch.cuda.FloatTensor([0., 0., 0., 0.])
        
        roi = roi.view(-1,1,4).expand_as(delta_per_class)
        bbox_per_class = t_param2bbox(roi.cpu().numpy().reshape(-1,4), delta_per_class.cpu().numpy().reshape(-1,4))
        bbox_per_class = torch.FloatTensor(bbox_per_class) #(N * 4)

        bbox_per_class[:,0::2] = bbox_per_class[:,0::2].clamp(min=0, max=image_size[0])  #0::2  :choose 0 and 2 cloumn (choose 0,2,4,6...)
        bbox_per_class[:,1::2] = bbox_per_class[:,1::2].clamp(min=0, max=image_size[1])  #clamp:choose between min and max, if beyond, choose max,min
                                                               #so the bbox will not outside the image,is a torch function
        bbox_per_class = bbox_per_class.numpy().reshape(-1,self.n_class_bg,4)
        prob = prob.cpu().numpy()
        
        # suppress:
        bbox_out = []
        class_out = []
        prob_out = []
        # skip class_id = 0 because it is the background class
        for t in range(1, self.n_class_bg):  #each class
            bbox_for_class_t = bbox_per_class[:,t,:]    #(N, 4)
            prob_for_class_t = prob[:,t]                #(N,)
            mask = prob_for_class_t > prob_threshold    #(N,)
            
            left_bbox_for_class_t = bbox_for_class_t[mask]  #(N2,4)
            left_prob_for_class_t = prob_for_class_t[mask]  #(N2,)
            keep = nms(left_bbox_for_class_t, score=left_prob_for_class_t)
            bbox_out.append(left_bbox_for_class_t[keep])  #keep bbox for class t
            prob_out.append(left_prob_for_class_t[keep])
            class_out.append((t-1)*np.ones(len(keep)))

        bbox_out = np.concatenate(bbox_out, axis=0).astype(np.float32)
        prob_out = np.concatenate(prob_out, axis=0).astype(np.float32)   #score
        class_out = np.concatenate(class_out, axis=0).astype(np.int32)   #label
        
        return bbox_out, class_out, prob_out
    

    def _normal_init(self, m, mean, stddev, truncated=False):

        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

