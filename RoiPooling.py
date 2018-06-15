
# coding: utf-8

# In[33]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class roipooling(nn.Module):
    def __init__(self, roi_size):
        super(roipooling, self).__init__()
        self.roip_h = int(roi_size[0])
        self.roip_w = int(roi_size[1])
        

    def forward(self, feature_map, rois, scaling): 
        # feature_map:(1,c,h,w)
        #rois: (N, 5); 5=[roi_index, x1, y1, x2, y2]
        batch_size, num_channels, feature_h, feature_w = feature_map.size()
#         print(rois.size())
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.roip_h, self.roip_w))
        if torch.cuda.is_available():
            outputs = outputs.cuda()

        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data)
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(roi[1:].data.cpu().numpy()*scaling).astype(int)
            roi_w = max(roi_end_w - roi_start_w + 1, 1)
            roi_h = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_w) / float(self.roip_w)
            bin_size_h = float(roi_h) / float(self.roip_h)

            for ph in range(self.roip_h):
                h_start = int(np.floor(ph * bin_size_h))
                h_end = int(np.ceil((ph + 1) * bin_size_h))
                h_start = min(feature_h, max(0, h_start + roi_start_h))
                h_end = min(feature_h, max(0, h_end + roi_start_h))
                for pw in range(self.roip_w):
                    w_start = int(np.floor(pw * bin_size_w))
                    w_end = int(np.ceil((pw + 1) * bin_size_w))
                    w_start = min(feature_w, max(0, w_start + roi_start_w))
                    w_end = min(feature_w, max(0, w_end + roi_start_w))

                    is_empty = (h_end <= h_start)or(w_end <= w_start)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = feature_map[batch_ind]

                        data_pool = torch.max(data[:, h_start:h_end, w_start:w_end], 1)[0]
                        outputs[roi_ind, :, ph, pw] = torch.max(data_pool, 1)[0].view(-1)
        return outputs
 

