
# coding: utf-8

# In[1]:


import numpy as np

def nms(roi, thresh=0.7, score=None):
    x_min = roi[:, 0]
    y_min = roi[:, 1]
    x_max = roi[:, 2]
    y_max = roi[:, 3] 

    zone = (x_max - x_min + 1) * (y_max - y_min + 1)
    if score is None:    # roi are already sorted in large --> small order
        sort = np.arange(roi.shape[0])
    else:               # roi are not sorted
        sort = score.argsort()[::-1]
    retain = []
    while sort.size > 0:
        i = sort[0]
        retain.append(i)
        x_g_max = np.maximum(x_min[i], x_min[sort[1:]])
        y_g_max = np.maximum(y_min[i], y_min[sort[1:]])
        x_g_min = np.minimum(x_max[i], x_max[sort[1:]])
        y_g_min = np.minimum(y_max[i], y_max[sort[1:]])

        w = np.maximum(0.0, x_g_min - x_g_max + 1)
        h = np.maximum(0.0, y_g_min - y_g_max + 1)
        cross = w * h
        iou = cross / (zone[i] + zone[sort[1:]] - cross)

        inds = np.where(iou <= thresh)[0]
        sort = sort[inds + 1]

    return retain # list of index of kept roi

