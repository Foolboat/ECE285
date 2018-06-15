
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# own model

# In[ ]:


from model.RPN import rpn
from model.utils.proposal_target_creator import ProposalTargetCreator
from model.utils.transform import image_size_transform, bbox_resize, image_flip, image_normalize


# In[ ]:


class Faster_RCNN(nn.Module):
    def __init__(self, feature_extractor, rpn, head):
        super(Faster_RCNN, self).__init__()
        self.feature_extractor = feature_extractor
        self.rpn = rpn
        self.head = head
        self.proposal_target_creator = ProposalTargetCreator()

    def forward(self):
        raise NotImplementedError("No Forward!")


    def loss(self, image, gt_bbox, gt_bbox_label):

#         image: (C=3,H,W), pixels should be in range 0~1 and normalized.
#         gt_bbox: (N2,4)
#         gt_bbox_label: (N2,)

        if self.training == False:
            raise Exception("Only in train mode!")

        original_image_size = image.shape[1:]    #height and width
        image, gt_bbox = image_flip(image, gt_bbox, h_ran=True)
        
        image = image_size_transform(image)
        new_image_size = image.shape[1:]        # new H and W
        gt_bbox = bbox_resize(gt_bbox, original_image_size, new_image_size)

        image = image_normalize(image)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

        image = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            image = image.cuda()

        features = self.feature_extractor(image)

        # rpn loss
        delta, score, anchor = self.rpn.forward(features, new_image_size)
        rpn_loss = self.rpn.loss(delta, score, anchor, gt_bbox, new_image_size)  #rpn_delta_loss + rpn_class_loss


        # head loss:
        roi = self.rpn.predict(delta, score, anchor, new_image_size)  #umber of roi after nms = 2000
        sample_roi, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi = self.proposal_target_creator.make_proposal_target(roi, gt_bbox, gt_bbox_label)   #gt_bbox_label: 20 classes     sample_roi = 256


        delta_per_class, score = self.head.forward(features, sample_roi, new_image_size)


        head_loss = self.head.loss(score, delta_per_class, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi)

        return rpn_loss + head_loss



    def predict(self, image, prob_threshold=0.5):

        
        if self.training == True:
            raise Exception("Only in eval mode!")
        original_image_size = image.shape[1:]        
        image = image_size_transform(image)
        new_image_size = image.shape[1:]
        image = image_normalize(image)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])   #batch_size, channel, high, width
        
        image = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            image = image.cuda()

        features = self.feature_extractor(image)    #channel=512
        image_size = image.shape[2:]

        delta, score, anchor = self.rpn.forward(features, image_size)
        roi = self.rpn.predict(delta, score, anchor, image_size)
        
 
        delta_per_class, score = self.head.forward(features, roi, image_size)  #contain roi pooling and fully connected layer     
        bbox_out, class_out, prob_out = self.head.predict(roi, delta_per_class, score, image_size, prob_threshold=prob_threshold)
        
        bbox_out = bbox_resize(bbox_out, new_image_size, original_image_size)
        
        return bbox_out, class_out, prob_out

    def get_optimizer(self, is_adam=False):
        lr = 0.001
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        if False:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

