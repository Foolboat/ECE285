
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np



def reg_loss(pred_delta, target_delta, anchor_label, sigma):
    
    weight = torch.zeros(target_delta.shape)
    index = (anchor_label.data > 0).view(-1,1).expand_as(weight)       #expand to the size of weight (N,4)
    weight[index.cpu()] = 1
    weight = Variable(weight)
    if torch.cuda.is_available():
        weight = weight.cuda()

    diff = weight*(pred_delta-target_delta)
    abs_diff = diff.abs()
#     print(abs_diff)
    mask = (abs_diff.data < (1.0/sigma**2)).float() # do not back propagat on flag
    mask = Variable(mask, requires_grad=False)
    if torch.cuda.is_available():
        mask = mask.cuda()
        
    loss = (mask*(0.5*sigma**2)*(abs_diff*abs_diff)+(1 - mask)*(abs_diff-0.5/sigma**2)).sum()

    
    loss = loss/(anchor_label.data>=0).sum().float()
    return loss

