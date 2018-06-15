
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision

# In[2]:s


from model.Faster_RCNN import Faster_RCNN
from model.Fast_RCNN import Fast_RCNN
from model.RPN import rpn


# In[ ]:


def faster_rcnn(n_class, vgg16_net):
    

    features = list(vgg16_net.features)[0:30]
    
    for layer in features[0:10]:    # freeze top 4 conv2d layers
        for p in layer.parameters():
            p.requires_grad = False
            
    extractor = nn.Sequential(*features)
    

    classifier = list(vgg16_net.classifier)
    del(classifier[6])  # delete last fc layer
    classifier = nn.Sequential(*classifier)     # classifier : (N,25088) -> (N,4096); 25088 = 512*7*7 = C*H*W
    if torch.cuda.is_available():
        classifier = classifier.cuda()
        
    head = Fast_RCNN(n_class_bg=n_class+1, roip_size=(7,7), classifier=classifier)   #n_class_bg = 20 + 1 (background)
    if torch.cuda.is_available():
        extractor, head = extractor.cuda(), head.cuda()
        
    rpn_net = rpn()
    
    model = Faster_RCNN(extractor, rpn_net, head)   #extractor: feature extractor
    return model

