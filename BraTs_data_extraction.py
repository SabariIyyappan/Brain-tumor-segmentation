#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import random
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps
import nibabel as nib
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
# from tensorflow.keras.layers import preprocessing
from tensorflow.keras.layers import Normalization, Rescaling, Resizing


# ### -----TRAINING DATA EXTRACTION-----

# In[2]:


#TRAIN_DATASET_PATH = "E:/RESEARCH NEW/Brats2020/BraTS2020_TrainingData/"
TRAIN_DATASET_PATH = "/nfsshare/knr/knrtrain/renu/Brats2020/BraTS2020_TrainingData/"


# In[9]:


t2_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*/*_t2.nii"))
t1ce_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*/*_t1ce.nii"))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*/*_flair.nii"))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + "*/*/*_seg.nii"))


# In[129]:


for img in range(len(t2_list)):
    print("working on image and mask number:", img)
    temp_img_t2 = nib.load(t2_list[img]).get_fdata()
    temp_img_t2 = scaler.fit_transform(temp_img_t2.reshape(-1, temp_img_t2.shape[-1])).reshape(temp_img_t2.shape)
    
    temp_img_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_img_t1ce = scaler.fit_transform(temp_img_t1ce.reshape(-1, temp_img_t1ce.shape[-1])).reshape(temp_img_t1ce.shape)
    
    temp_img_flair = nib.load(flair_list[img]).get_fdata()
    temp_img_flair = scaler.fit_transform(temp_img_flair.reshape(-1, temp_img_flair.shape[-1])).reshape(temp_img_flair.shape)
    
    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3
    
    temp_combined_imgs = np.stack([temp_img_t2, temp_img_t1ce, temp_img_flair], axis=3)
    
    temp_combined_imgs = temp_combined_imgs[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val,counts = np.unique(temp_mask, return_counts=True)
    print(counts)                     
    if (1 - counts[0]/counts.sum()) > 0.01:
        temp_mask = to_categorical(temp_mask, num_classes=4)
        '''np.save(TRAIN_DATASET_PATH+"Training_final_data/images/image_"+str(t2_list[img]).replace("\\","/").split('/')[5]+".npy",temp_combined_imgs)
        np.save(TRAIN_DATASET_PATH+"Training_final_data/masks/mask_"+str(mask_list[img]).replace("\\","/").split('/')[5]+".npy",temp_mask)'''
        np.save("/nfsshare/knr/knrtrain/sabari/Brats_cropped/training data/images/image_"+str(t2_list[img]).replace("\\","/").split('/')[5]+".npy",temp_combined_imgs)
        np.save("/nfsshare/knr/knrtrain/sabari/Brats_cropped/training data/masks/mask_"+str(mask_list[img]).replace("\\","/").split('/')[5]+".npy",temp_mask)


# ### -----VALIDATION DATA EXTRACTION------

# In[7]:


#VALIDATION_DATASET_PATH = "E:/RESEARCH NEW/Brats2020/BraTS2020_ValidationData/"
VALIDATION_DATASET_PATH = "/nfsshare/knr/knrtrain/renu/Brats2020/BraTS2020_ValidationData/"

# In[8]:


t2_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*/*_t2.nii"))
t1ce_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*/*_t1ce.nii"))
flair_list = sorted(glob.glob(VALIDATION_DATASET_PATH + "*/*/*_flair.nii"))


# In[126]:


for img in range(len(t2_list)):
    print("working on image and mask number:", img)
    temp_img_t2 = nib.load(t2_list[img]).get_fdata()
    temp_img_t2 = scaler.fit_transform(temp_img_t2.reshape(-1, temp_img_t2.shape[-1])).reshape(temp_img_t2.shape)
    
    temp_img_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_img_t1ce = scaler.fit_transform(temp_img_t1ce.reshape(-1, temp_img_t1ce.shape[-1])).reshape(temp_img_t1ce.shape)
    
    temp_img_flair = nib.load(flair_list[img]).get_fdata()
    temp_img_flair = scaler.fit_transform(temp_img_flair.reshape(-1, temp_img_flair.shape[-1])).reshape(temp_img_flair.shape)
    
    temp_combined_imgs = np.stack([temp_img_t2, temp_img_t1ce, temp_img_flair], axis=3)
    
    temp_combined_imgs = temp_combined_imgs[56:184, 56:184, 13:141]
    
    #np.save(VALIDATION_DATASET_PATH+"Validation_final_data/images/image_"+str(t2_list[img]).replace("\\","/").split('/')[5]+".npy",temp_combined_imgs)
    np.save("/nfsshare/knr/knrtrain/sabari/Brats_cropped/testing data/images/image_"+str(t2_list[img]).replace("\\","/").split('/')[5]+".npy",temp_combined_imgs)


# ### -----THE END OF DATA EXTRACTION-----

# In[3]:


# load .nii file as a numpy array
test_image_flair = nib.load(TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_flair.nii").get_fdata()
print("Shape: ", test_image_flair.shape)
print("Dtype: ", test_image_flair.dtype)


# In[4]:


print("Min: ", test_image_flair.min())
print("Max: ", test_image_flair.max())


# In[5]:


scaler = MinMaxScaler()


# In[33]:


test_set = sorted(glob.glob(TRAIN_DATASET_PATH + "BraTS20_Training_355/BraTS20_Training_355_*.nii"))

test_flair = nib.load(test_set[0]).get_fdata()
test_flair = scaler.fit_transform(test_flair.reshape(-1, test_flair.shape[-1])).reshape(test_flair.shape)

test_mask = nib.load(test_set[1]).get_fdata()

test_t1 = nib.load(test_set[2]).get_fdata()
test_t1 = scaler.fit_transform(test_t1.reshape(-1, test_t1.shape[-1])).reshape(test_t1.shape)

test_t1ce = nib.load(test_set[3]).get_fdata()
test_t1ce = scaler.fit_transform(test_t1ce.reshape(-1, test_t1ce.shape[-1])).reshape(test_t1ce.shape)

test_t2 = nib.load(test_set[4]).get_fdata()
test_t2 = scaler.fit_transform(test_t2.reshape(-1, test_t2.shape[-1])).reshape(test_t2.shape)



# In[83]:


val,counts = np.unique(test_mask, return_counts=True)
counts


# In[43]:


n_slice = random.randint(0,test_mask.shape[2])
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(test_flair[:,:,n_slice],cmap='gray')
plt.title("flair")
plt.subplot(2,3,2)
plt.imshow(test_mask[:,:,n_slice])
plt.title("mask")
plt.subplot(2,3,3)
plt.imshow(test_t1[:,:,n_slice],cmap='gray')
plt.title("t1")
plt.subplot(2,3,4)
plt.imshow(test_t1ce[:,:,n_slice],cmap='gray')
plt.title("t1ce")
plt.subplot(2,3,5)
plt.imshow(test_t2[:,:,n_slice],cmap='gray')
plt.title("t2")


# In[ ]:




