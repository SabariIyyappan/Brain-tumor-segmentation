#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda, UpSampling3D, Add
from keras.optimizers import Adam
from keras.metrics import MeanIoU


# In[2]:


def staircase_net_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    kernel_initializer = 'he_uniform'

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    # Step 1: Coarse segmentation
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    # Step 2: Refine segmentation
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    # Step 3: Further refinement
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    # Step 4: Deep refinement stage
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    # Bottleneck layer
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    # Progressive refinement steps (staircase-like steps)
    # Step 1 Refinement (From Bottleneck)
    u6 = UpSampling3D((2, 2, 2))(c5)
    r6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    r6 = concatenate([r6, c4])
    r6 = Dropout(0.2)(r6)
    r6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(r6)

    # Step 2 Refinement
    u7 = UpSampling3D((2, 2, 2))(r6)
    r7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    r7 = concatenate([r7, c3])
    r7 = Dropout(0.2)(r7)
    r7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(r7)

    # Step 3 Refinement
    u8 = UpSampling3D((2, 2, 2))(r7)
    r8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    r8 = concatenate([r8, c2])
    r8 = Dropout(0.1)(r8)
    r8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(r8)

    # Final Refinement (Coarse to Fine)
    u9 = UpSampling3D((2, 2, 2))(r8)
    r9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    r9 = concatenate([r9, c1])
    r9 = Dropout(0.1)(r9)
    r9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(r9)

    # Output Layer (Final Segmentation Map)
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(r9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()

    return model


# In[3]:


# Test the model
model = staircase_net_model(128, 128, 128, 3, 4)
print(model.input_shape)
print(model.output_shape)

