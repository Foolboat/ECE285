
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


# own model

# In[ ]:


from model.utils.generate_anchor import generate_anchor
from model.utils.proposal_creator import ProposalCreator
from model.utils.anchor_target_creator import AnchorTargetCreator
from model.utils.loss_func import reg_loss


# In[ ]:


class rpn(nn.Module):
    def __init__(self):    #in_channel = 512 mid_channel = 512
        super(rpn, self).__init__()

        
        self.K = 9    # default: 9 : 9 ahcnors per spatial channel in feature maps

        self.mid_layer = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1) 
        self.score_layer = nn.Conv2d(512, 2*self.K, kernel_size=1, stride=1, padding=0)
        self.delta_layer = nn.Conv2d(512, 4*self.K, kernel_size=1, stride=1, padding=0)
        
        self._normal_init(self.mid_layer, 0, 0.01)
        self._normal_init(self.score_layer, 0, 0.01)
        self._normal_init(self.delta_layer, 0, 0.01)

        self.proposal_creator = ProposalCreator()
        self.anchor_target_creator = AnchorTargetCreator()

    def forward(self, features, image_size):


        _, _, feature_height, feature_width = features.shape
        image_height, image_width = image_size[0], image_size[1]

        mid_features = F.relu(self.mid_layer(features))   #3 * 3 convolution
        
        delta = self.delta_layer(mid_features)
        delta = delta.permute(0,2,3,1).contiguous().view([feature_height*feature_width*self.K, 4])   #1 * 1 convolution,  num of anchor * 4
        
        score = self.score_layer(mid_features)
        score = score.permute(0,2,3,1).contiguous().view([feature_height*feature_width*self.K, 2])

        # ndarray: (feature_height*feature_width*K, 4) = number of anchor
        anchor = generate_anchor(feature_height, feature_width, image_size)  #anchors in original image
        
        return delta, score, anchor

    def loss(self, delta, score, anchor, gt_bbox, image_size):
        
        target_delta, anchor_label = self.anchor_target_creator.make_anchor_target(anchor, gt_bbox, image_size)
        target_delta = Variable(torch.FloatTensor(target_delta))
        anchor_label = Variable(torch.LongTensor(anchor_label))
        if torch.cuda.is_available():
            target_delta, anchor_label = target_delta.cuda(), anchor_label.cuda()

        rpn_reg_loss = reg_loss(delta, target_delta, anchor_label, 3)    #delta: feature through head_net  target_delta: anchor with cloest bbox
        
        rpn_class_loss = F.cross_entropy(score, anchor_label, ignore_index=-1)   # ignore loss for label value -1

        return rpn_reg_loss + rpn_class_loss

    def predict(self, delta, score, anchor, image_size):
        
        delta = delta.data.cpu().numpy()
        score = score.data.cpu().numpy()
        score_fg = score[:,1]   #score.shape == (feature_height*feature_width*self.K, 2)
        roi = self.proposal_creator.make_proposal(anchor, delta, score_fg, image_size, is_training=self.training)  #after nms, num of roi = 2000 
        
        return roi


    def _normal_init(self, m, mean, stddev, truncated=False):

        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

