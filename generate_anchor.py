
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


def generate_anchor(feature_height ,feature_width, image_size):
    anchor_ratio=[0.5, 1, 2]
    anchor_size = [128, 256, 512]
    
    anchor_dim = len(anchor_ratio)*len(anchor_size)
    corner_dim = 4
    stride_x = image_size[0] / feature_height
    stride_y = image_size[1]/ feature_width
    
    anchor_base = np.array([[-size*math.sqrt(ratio)/2, -size*math.sqrt(1/ratio)/2,                         size*math.sqrt(ratio)/2, size*math.sqrt(1/ratio)/2]                    for ratio in anchor_ratio for size in anchor_size])
    
    anchors = np.zeros([feature_height, feature_width, anchor_dim, corner_dim])
    for i in range(feature_height):
        for j in range(feature_width):
            x = i*stride_x + stride_x/2
            y = j*stride_y + stride_y/2
            shift = [x,y,x,y]
            anchors[i, j] = anchor_base+shift 
    anchors = anchors.reshape([-1,4])
    return anchors




    

