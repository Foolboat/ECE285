
# coding: utf-8

# In[ ]:


from torchvision import transforms
import torch
from skimage import transform as sktransform
import random

def image_normalize(image):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    nor_image = (normalize(torch.from_numpy(image))).numpy()
    return nor_image

def image_size_transform(image, min_size=600, max_size=1000):

    Channel, High, Width = image.shape
    trans1 = min_size / min(High, Width)
    trans2 = max_size / max(High, Width)
    trans = min(trans1, trans2)
    image_trans = sktransform.resize(image, (Channel, High * trans, Width * trans), mode='reflect')
    return image_trans

def bbox_resize(bbox, image_pre, image_post):
   
    bbox = bbox.copy()
    y_trans = float(image_post[0]) / image_pre[0]
    x_trans = float(image_post[1]) / image_pre[1]
    bbox[:, 0] = y_trans * bbox[:, 0]
    bbox[:, 2] = y_trans * bbox[:, 2]
    bbox[:, 1] = x_trans * bbox[:, 1]
    bbox[:, 3] = x_trans * bbox[:, 3]
    return bbox


def image_flip(image, bbox, v_ran=False, h_ran=False):
    vertical_flip, horizontal_flip = False, False
    H,W = image.shape[1], image.shape[2]
    bbox = bbox.copy()

    if v_ran:
        vertical_flip =  random.choice([True, False])
    if h_ran:
        horizontal_flip = random.choice([True, False])

    if vertical_flip:
        image = image[:,::-1,:]             #in every channel, flip in vertical
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max

    if horizontal_flip:
        image = image[:,:,::-1]
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    
    return image, bbox
