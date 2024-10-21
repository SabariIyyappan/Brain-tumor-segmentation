


import os
import numpy as np
from matplotlib import pyplot as plt
import random
import segmentation_models_3D as sm
import keras
import glob
import nbimporter
from simple_3D_Unet_architecture import simple_unet_model
import tensorflow as tf
from keras.metrics import MeanIoU


# In[2]:


def load_img(img_dir, img_list):
    
    images=[]
    for i, image_name in enumerate(img_list[:batch_size]):    
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
                      
            images.append(image)
    images = np.array(images)
    
    return(images)



batch_size = 2


# In[5]:


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size


# ### Defining directory paths

# In[6]:


TRAIN_DATASET_PATH = "E:/RESEARCH NEW/Brats2020/BraTS2020_TrainingData/"


# In[7]:


train_img_dir = TRAIN_DATASET_PATH+"Training_final_data/images/"
train_mask_dir = TRAIN_DATASET_PATH+"Training_final_data/masks/"
train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)


# In[8]:


VALIDATION_DATASET_PATH = "E:/RESEARCH NEW/Brats2020/BraTS2020_ValidationData/"


# In[9]:


val_img_dir = VALIDATION_DATASET_PATH+"Validation_final_data/images/"
val_img_list=os.listdir(val_img_dir)


# In[10]:


num_images = len(os.listdir(train_img_dir))


# In[11]:


train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)


img, msk = train_img_datagen.__next__()


# In[12]:


img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)


# In[15]:


n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(10, 6))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image t2')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image flair')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()


# ### Define loss, metrics and optimizer to be used for training

# In[22]:


### custom_iou function defined instead of default IOUScore metric because of different datatypes of gt, pr in some cases
def custom_iou(gt, pr):
    # Cast gt and pr to float64
    gt = tf.cast(gt, tf.float32)
    pr = tf.cast(pr, tf.float32)
    
    # Call the IoU calculation
    return sm.metrics.IOUScore(threshold=0.5)(gt, pr)


# In[17]:


wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', custom_iou]

LR = 0.0001
optim = keras.optimizers.Adam(LR)


# ### Fit the model 

# In[18]:


steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


# ### Model building

# In[19]:


model = simple_unet_model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=3, 
                          num_classes=4)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)




# In[23]:


history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=1,
          verbose=1
#           ,validation_data=val_img_datagen,
#           validation_steps=val_steps_per_epoch
          )


# In[ ]:


#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ### Testing 

# In[ ]:


batch_size=8 #Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())







