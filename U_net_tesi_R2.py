# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:49:47 2021

@author: Sara
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:47:01 2021

@author: Sara
"""

# -*- coding: utf-8 -*-

#clean variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

#import some libraries
from glob import glob
import shutil
import argparse
import zipfile
import hashlib
import requests
#from tqdm import tqdm
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error


from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 

import pydicom as dicom
import pandas as pd
import math

num_cores = 4

GPU=False
CPU=True

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

# path="/home/sarajoubbi/Desktop/Prove tesi/DATA/"
# dataset_path = path + "DICOM_PANCREAS_CROP/" # data directory
# mask_path = path + "MASCHERE/" # maschere directory
# maps_path = path + "MAPPE_HERNANDO/" # data directory
# folder=os.listdir(dataset_path)

path="C:/Users/Sara/Desktop/Tesi Laurea Magistrale/Pratica/DATA_SAMPLE/DATA/"
dataset_path = path + "DICOM_PANCREAS_CROP/" # data directory
mask_path = path + "MASCHERE/" # maschere directory
maps_path = path + "MAPPE_HERNANDO/" # data directory
folder=os.listdir(dataset_path)

imm=[]
im_size = 64
n=len(folder);
mask=np.zeros([im_size,im_size,3*n])
maps=np.zeros([im_size,im_size,n*3,2])
id_paz = np.zeros(n*3)
zone=[]
for i in range(len(folder)):
    imm.append(dicom.read_file(dataset_path+folder[i]+'/head.dcm',force=True).pixel_array)
    imm.append(dicom.read_file(dataset_path+folder[i]+'/body.dcm',force=True).pixel_array)
    imm.append(dicom.read_file(dataset_path+folder[i]+'/tail.dcm',force=True).pixel_array)
    mask[:,:,(i*3)]=cv2.imread(mask_path+folder[i]+'/head.tif',-1)
    mask[:,:,(i*3)+1]=cv2.imread(mask_path+folder[i]+'/body.tif',-1)
    mask[:,:,(i*3)+2]=cv2.imread(mask_path+folder[i]+'/tail.tif',-1)
    maps[:,:,(i*3),0]=cv2.imread(maps_path+folder[i]+'/R2/head.png',0)
    maps[:,:,(i*3)+1,0]=cv2.imread(maps_path+folder[i]+'/R2/body.png',0)
    maps[:,:,(i*3)+2,0]=cv2.imread(maps_path+folder[i]+'/R2/tail.png',0)
    maps[:,:,(i*3),1]=cv2.imread(maps_path+folder[i]+'/FF/head.png',0)
    maps[:,:,(i*3)+1,1]=cv2.imread(maps_path+folder[i]+'/FF/body.png',0)
    maps[:,:,(i*3)+2,1]=cv2.imread(maps_path+folder[i]+'/FF/tail.png',0)
    id_paz[(i*3):(i*3 + 3)] = int(i+1)
    zone.extend(['head','body','tail']);
  

plt.figure(figsize=(10, 10))
im_paz = imm[119]
for i in range (0,10):
    I = im_paz[i,:,:]
    r2_true=plt.subplot(5,5,i+1)
    plt.imshow(I, cmap = 'gray')
    
n_te = 10
images = []
Im = np.zeros([im_size,im_size,n_te])
Im_tot = []
for i in range(0,3*n):
     img = imm[i] #read image
     for j in range(0,n_te):
         Im[:,:,j] = img[j,:,:]
     Im_tot.append(Im)
     img1=tf.convert_to_tensor(Im) # convert to tensor
     images.append(img1)  # add image to list

from scipy import ndimage, misc
labels=[]
lab = []
lab_filt = []
for i in range(0,3*n):
       appoggio = maps[:,:,i,:]
       appoggio_filt = [ndimage.median_filter(maps[:,:,i,j], 5) for j in range(2)]
       lab.append(appoggio)
       lab_filt.append(appoggio_filt)
       appoggio = tf.convert_to_tensor(appoggio)
       labels.append(appoggio)

# # Errori
# def rmse (y_true, y_pred):
#     y_true = np.reshape(np.array(y_true),[im_size*im_size])
#     y_pred = np.reshape(np.array(y_pred),[im_size*im_size])
#     diff = np.subtract(y_true,y_pred)
#     sq = np.square(diff).mean()
#     res = math.sqrt(sq)
#     return res

# error = []
# for i in range(len(labels)):
#     true = lab[i]
#     true_filt = lab_filt[i]
#     error.append([rmse(true[:,:,0],true_filt[0]),rmse(true[:,:,1],true_filt[1])])
# print(np.mean(np.array(error)))

df = pd.ExcelFile('res_hippo.xlsx').parse('Sheet2') #you could add index_col=0 if there's an index
hippo=[]
hippo = list(zip(df['T2* HIPPO'],df['FF HIPPO']))


def custom_split(images, labels, ind, zone, hippo, test_size, seed = None):
    
    np.random.seed(seed)
    arr_rand = np.random.choice(range(len(images)), len(images), replace=False)
    train = round((1-2*test_size)*len(images))
    test = round(test_size*len(images))
    
    X_train = [images[i] for i in arr_rand[:train]]
    y_train = [labels[i] for i in arr_rand[:train]]
    ind_train = [ind[i] for i in arr_rand[:train]]
    zone_train = [zone[i] for i in arr_rand[:train]]
    hippo_train = [hippo[i] for i in arr_rand[:train]]
    X_val = [images[i] for i in arr_rand[train:(train+test)]]
    y_val = [labels[i] for i in arr_rand[train:(train+test)]]
    ind_val = [ind[i] for i in arr_rand[train:(train+test)]]
    zone_val = [zone[i] for i in arr_rand[train:(train+test)]]
    hippo_val = [hippo[i] for i in arr_rand[train:(train+test)]]
    X_test = [images[i] for i in arr_rand[(train+test):]]
    y_test = [labels[i] for i in arr_rand[(train+test):]]
    ind_test = [ind[i] for i in arr_rand[(train+test):]]
    zone_test = [zone[i] for i in arr_rand[(train+test):]]
    hippo_test = [hippo[i] for i in arr_rand[(train+test):]]
    
    return  X_train, X_val, X_test, y_train, y_val, y_test, ind_train, ind_val, ind_test, zone_train, zone_val, zone_test, hippo_train, hippo_val, hippo_test


from random import randint
import random
def createAugmentedData1(Xor,Yor,train_ind,train_zone,Im_tot,lab,inc):
    X = []
    Y = []
    ind = []
    zone = []
    dataSize=len(Xor)
    rng = np.random.default_rng(seed=42)
    for i in range(dataSize):
        ind_paz = train_ind[i]
        if(train_zone[i]=='head'):
            cost = 0
        elif (train_zone[i] == 'body'):
            cost = 1
        else:
            cost = 2
        X.append(Xor[i])
        Y.append(Yor[i])
        ind.append(ind_paz)
        zone.append(train_zone[i])
        for j in range(inc):
            p=rng.integers(1,6)
            
            if (p ==1):
                    ang =  randint(0, 20)
                    augImg = tf.keras.preprocessing.image.apply_affine_transform(Im_tot[((int(ind_paz)-1)*3+cost)], theta=ang, tx=0, ty=0, shear=0, zx=1, zy=1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
                    augLab = tf.keras.preprocessing.image.apply_affine_transform(lab[((int(ind_paz)-1)*3+cost)], theta=ang, tx=0, ty=0, shear=0, zx=1, zy=1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
            if (p ==2):
                    tr = random.uniform(1.0,1.6)
                    augImg = tf.keras.preprocessing.image.apply_affine_transform(Im_tot[((int(ind_paz)-1)*3+cost)], theta=0, tx=0, ty=0, shear=0, zx=tr, zy=tr, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
                    augLab = tf.keras.preprocessing.image.apply_affine_transform(lab[((int(ind_paz)-1)*3+cost)], theta=0, tx=0, ty=0, shear=0, zx=tr, zy=tr, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
            if (p ==3):
                    sh = random.uniform(0.1,3.0)
                    augImg = tf.keras.preprocessing.image.apply_affine_transform(Im_tot[((int(ind_paz)-1)*3+cost)], theta=0, tx=sh, ty=sh, shear=0, zx=1, zy=1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
                    augLab = tf.keras.preprocessing.image.apply_affine_transform(lab[((int(ind_paz)-1)*3+cost)], theta=0, tx=0, ty=sh, shear=sh, zx=1, zy=1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
            if (p ==4):
                    rfloat = 1+rng.random()
                    augImg = Im_tot[((int(ind_paz)-1)*3+cost)]*rfloat
                    augLab = Yor[i]
            if (p ==5):
                    rfloat = 50*rng.random()
                    augImg = Im_tot[((int(ind_paz)-1)*3+cost)]+rfloat
                    augLab = Yor[i]
            
            X.append(tf.convert_to_tensor(augImg))
            Y.append(tf.convert_to_tensor(augLab))
            ind.append(ind_paz)
            zone.append(train_zone[i])
    
    return X,Y,ind,zone
     
#from sklearn.model_selection import train_test_split

percVal=0.2 # validatio nset size
train_X1, val_X1, test_X1, train_y1, val_y1, test_y1, train_ind, val_ind, test_ind, train_zone, val_zone, test_zone, train_hippo, val_hippo, test_hippo = custom_split(images,labels,id_paz, zone, hippo, test_size=percVal, seed = 4)                               
batch_size = 8

train_ds = tf.data.Dataset.from_tensor_slices((train_X1,train_y1))  # combino immagini e label per il training
val_ds = tf.data.Dataset.from_tensor_slices((val_X1,val_y1))        # combino immagini e label per il validation
train_ds = train_ds.repeat().batch(batch_size) # devo aggiungere la dimensione per il batch
val_ds = val_ds.repeat().batch(batch_size)

train_ds.element_spec, val_ds.element_spec  # devono venire degli oggetti [None,64,64,10] [None,64,64,2]


kernel_size=3

def relu_mod(x):
    return K.relu(x,max_value=1.0)

################################### U-Net di Goldfarb et al
def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.8))(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = conv_block(decoder, num_filters)
  return decoder

img_shape = (im_size, im_size, n_te) 
first_encoder_channels=96

inputs = layers.Input(shape=img_shape)
encoder0_pool, encoder0 = encoder_block(inputs, first_encoder_channels)
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 2*first_encoder_channels)
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 4*first_encoder_channels)
center = conv_block(encoder2_pool, 8*first_encoder_channels)
decoder2 = decoder_block(center, encoder2, 4*first_encoder_channels)
decoder1 = decoder_block(decoder2, encoder1, 2*first_encoder_channels)
decoder0 = decoder_block(decoder1, encoder0, first_encoder_channels)

outputs = layers.Conv2D(2, (1,1), activation="linear", padding="same")(decoder0)

model = models.Model(inputs=[inputs], outputs=[outputs]) # definisco il modello

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss="mean_squared_error",
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"),
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)])


# training del modello
#from tensorflow.keras.callbacks import TensorBoard
#save_model_path = 'C:/Users/Sara/Desktop/Tesi Laurea Magistrale/Pratica/Esempi Keras/Pesi/weights.hdf5'
#cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss', mode='min', save_best_only=True)
#early = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=200, restore_best_weights=True)
num_train_examples = len(train_X1)
num_val_examples = len(val_X1)
epochs = 500
history = model.fit(train_ds, 
                    steps_per_epoch=max(int(0.1*np.ceil(num_train_examples / float(batch_size))),1),
                    epochs=epochs,
                    validation_data=val_ds,
                    validation_steps=max(int(0.1*np.ceil(num_val_examples / float(batch_size))),1),
                    #shuffle=True,
                    callbacks=[PlotLearning()])
                    #callbacks=[early, PlotLearning()])


loss =  history.history['loss']
accuracy = history.history['root_mean_squared_error']
val_loss =  history.history['val_loss']
val_accuracy = history.history['val_root_mean_squared_error']

epochs_range = range(epochs)
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(len(loss)), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(range(len(loss)), accuracy, label='Trainig RMSE')
plt.plot(range(len(loss)), val_accuracy, label='Validation RMSE')
plt.legend(loc='upper right')
plt.title('Metric')
plt.savefig('loss_metric.png')
plt.show()


import time
tic = time.time()
n = 3
imgtest = tf.expand_dims(test_X1[n], axis=0)
predictions = model.predict(imgtest)
r2_im=np.reshape(predictions[:,:,:,0],(im_size,im_size))
ff_im=np.reshape(predictions[:,:,:,1],(im_size,im_size))
true=test_y1[n]
toc = time.time()
print(toc-tic)

minmin_r2 = np.min([np.min(true[:,:,0]), np.min(r2_im)])
maxmax_r2 = np.max([np.max(true[:,:,0]), np.max(r2_im)])
minmin_ff = np.min([np.min(true[:,:,1]), np.min(ff_im)])
maxmax_ff = np.max([np.max(true[:,:,1]), np.max(ff_im)])

plt.figure(figsize=(10, 8))
r2_true=plt.subplot(2,2,1)
plt.title("True R2")
plt.imshow(np.reshape(true[:,:,0],(im_size,im_size)), vmin=minmin_r2, vmax=maxmax_r2)

r2_pred=plt.subplot(2,2,2)
plt.title("R2 Predicted")
plt.imshow(r2_im, vmin=minmin_r2, vmax=maxmax_r2)
plt.colorbar()

ff_true=plt.subplot(2,2,3)
plt.title("True FF")
plt.imshow(np.reshape(true[:,:,1],(im_size,im_size)), vmin=minmin_ff, vmax=maxmax_ff)

ff_pred=plt.subplot(2,2,4)
plt.title("FF Predicted")
plt.imshow(ff_im, vmin=minmin_ff, vmax=maxmax_ff)
plt.colorbar()
plt.savefig('random_test.png')

# #############################################
if (test_zone[n] == 'head'):
    cost = 0
elif (test_zone[n] == 'body'):
    cost = 1
else:
    cost = 2

mask_val = np.reshape(mask[:,:,((int(test_ind[n])-1)*3+cost)], [im_size,im_size])
mask_val = mask_val/255
mask_rif = cv2.multiply(np.reshape(true[:,:,0], [im_size,im_size]),mask_val)
r2_im = r2_im.astype('float64')
mask_res = cv2.multiply(r2_im,mask_val)
mask_rif_ff = cv2.multiply(np.reshape(true[:,:,1], [im_size,im_size]),mask_val)
ff_im = ff_im.astype('float64')
mask_res_ff = cv2.multiply(ff_im,mask_val)


minmin_r2 = np.min([np.min(mask_rif), np.min(mask_res)])
maxmax_r2 = np.max([np.max(mask_rif), np.max(mask_res)])
minmin_ff = np.min([np.min(mask_rif_ff), np.min(mask_res_ff)])
maxmax_ff = np.max([np.max(mask_rif_ff), np.max(mask_res_ff)])

plt.figure(figsize=(10, 8))
r2_true=plt.subplot(2,2,1)
plt.title("True R2")
plt.imshow(mask_rif, vmin=minmin_r2, vmax=maxmax_r2)

r2_pred=plt.subplot(2,2,2)
plt.title("R2 Predicted")
plt.imshow(mask_res, vmin=minmin_r2, vmax=maxmax_r2)
plt.colorbar()

r2_true=plt.subplot(2,2,3)
plt.title("True FF")
plt.imshow(mask_rif_ff, vmin=minmin_ff, vmax=maxmax_ff)

r2_pred=plt.subplot(2,2,4)
plt.title("FF Predicted")
plt.imshow(mask_res_ff, vmin=minmin_ff, vmax=maxmax_ff)
plt.colorbar()
plt.savefig('random_test_roi.png')

##################################################
diff_r2 = abs(np.reshape(true[:,:,0],(im_size,im_size)) - r2_im)
diff_ff = abs(np.reshape(true[:,:,1],(im_size,im_size)) - ff_im)
diff_r2_mask = abs(mask_rif - mask_res)
diff_ff_mask = abs(mask_rif_ff - mask_res_ff)

plt.figure(figsize=(10, 8))
r2_true=plt.subplot(2,2,1)
plt.title("Diff R2")
plt.imshow(diff_r2)
plt.colorbar()

r2_pred=plt.subplot(2,2,2)
plt.title("Diff FF")
plt.imshow(diff_ff)
plt.colorbar()

r2_true=plt.subplot(2,2,3)
plt.title("Diff R2 mask")
plt.imshow(diff_r2_mask)
plt.colorbar()

r2_pred=plt.subplot(2,2,4)
plt.title("Diff FF mask")
plt.imshow(diff_ff_mask)
plt.colorbar()
plt.savefig('random_test_diff.png')

import statistics
index = np.where(mask_val == 1)
true_val = [true[index[0][i]][index[1][i]][0] for i in range(0,len(index[0]))]
pred_val = [r2_im[index[0][i]][index[1][i]] for i in range(0,len(index[0]))]
true_val_ff = [true[index[0][i]][index[1][i]][1] for i in range(0,len(index[0]))]
pred_val_ff = [ff_im[index[0][i]][index[1][i]] for i in range(0,len(index[0]))]
true_r2 = statistics.median(np.array(true_val))
pred_r2 = statistics.median(pred_val)
true_ff = statistics.median(np.array(true_val_ff))
pred_ff = statistics.median(pred_val_ff)
print(test_ind[n])
print(test_zone[n])
print('True T2*',(1/true_r2)*1000)
print('T2* pred',(1/pred_r2)*1000)
print('True FF',true_ff)
print('FF pred',pred_ff)


################################# Analisi dei dati
import time
tic = time.time()
import statistics
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

def results(X, y, ind, zone, hippo, model, mask):
    res=[]
    ssim_r2 = []
    ssim_ff = []
    rmse_r2 = []
    rmse_ff = []
    rmse_r2_roi = []
    rmse_ff_roi = []
    for j in range(0,len(X)):
        
        imgtest = tf.expand_dims(X[j], axis=0)
        predictions = model.predict(imgtest)
        r2_im=np.reshape(predictions[:,:,:,0],(64,64))
        ff_im=np.reshape(predictions[:,:,:,1],(64,64))
        true=y[j]
        
        rmse_r2.append(mean_squared_error(true[:,:,0],r2_im, squared=False))
        rmse_ff.append(mean_squared_error(true[:,:,1],ff_im, squared=False))
        
        minmin_r2 = np.min([np.min(true[:,:,0]), np.min(r2_im)])
        maxmax_r2 = np.max([np.max(true[:,:,0]), np.max(r2_im)])
        minmin_ff = np.min([np.min(true[:,:,1]), np.min(ff_im)])
        maxmax_ff = np.max([np.max(true[:,:,1]), np.max(ff_im)])
        
        st = 'patient'+str(int(ind[j]))+'.png'
        
        plt.figure(figsize=(10, 8))
        r2_true=plt.subplot(2,2,1)
        plt.title("True R2*")
        plt.imshow(np.reshape(true[:,:,0],(im_size,im_size)), vmin=minmin_r2, vmax=maxmax_r2)

        r2_pred=plt.subplot(2,2,2)
        plt.title("R2* Predicted")
        plt.imshow(r2_im, vmin=minmin_r2, vmax=maxmax_r2)
        plt.colorbar()

        ff_true=plt.subplot(2,2,3)
        plt.title("True FF")
        plt.imshow(np.reshape(true[:,:,1],(im_size,im_size)), vmin=minmin_ff, vmax=maxmax_ff)

        ff_pred=plt.subplot(2,2,4)
        plt.title("FF Predicted")
        plt.imshow(ff_im, vmin=minmin_ff, vmax=maxmax_ff)
        plt.colorbar()
        #plt.savefig(st)
        
        ssim_r2.append(ssim(r2_im,np.reshape(true[:,:,0],(64,64)),data_range=(maxmax_r2-minmin_r2)))
        ssim_ff.append(ssim(ff_im,np.reshape(true[:,:,1],(64,64)),data_range=(maxmax_ff-minmin_ff)))
        
        if (zone[j] == 'head'):
            cost = 0
        elif (zone[j] == 'body'):
            cost = 1
        else:
            cost = 2
        
        mask_val = np.reshape(mask[:,:,((int(ind[j])-1)*3+cost)], [64,64])
        mask_val = mask_val/255
        index = np.where(mask_val == 1)
        
        true_val_r2 = [true[index[0][i]][index[1][i]][0] for i in range(0,len(index[0]))]
        pred_val_r2 = [r2_im[index[0][i]][index[1][i]] for i in range(0,len(index[0]))]
        true_val_ff = [true[index[0][i]][index[1][i]][1] for i in range(0,len(index[0]))]
        pred_val_ff = [ff_im[index[0][i]][index[1][i]] for i in range(0,len(index[0]))]
        
        rmse_r2_roi.append(mean_squared_error(true_val_r2,pred_val_r2, squared=False))
        rmse_ff_roi.append(mean_squared_error(true_val_ff,pred_val_ff, squared=False))

        
        true_r2 = statistics.median(np.array(true_val_r2))
        pred_r2 = statistics.median(pred_val_r2)
        true_ff = statistics.median(np.array(true_val_ff))
        pred_ff = statistics.median(pred_val_ff)
        res.append([ind[j], zone[j], round(hippo[j][0],2), round((1/true_r2)*1000,2), round((1/pred_r2)*1000,2), round(hippo[j][1],2), round(true_ff,2), round(pred_ff,2)])
        
    return res, ssim_r2, ssim_ff, rmse_r2, rmse_ff,rmse_r2_roi,rmse_ff_roi

    
res_test, ssim_r2_test, ssim_ff_test, rmse_r2_test, rmse_ff_test, rmse_r2_test_roi, rmse_ff_test_roi = results(test_X1, test_y1, test_ind, test_zone, test_hippo, model, mask)
toc = time.time()
print(toc-tic)
print('SSIM R2*: '+str(round(np.mean(np.array(ssim_r2_test)),2)))
print('SSIM FF: '+str(round(np.mean(np.array(ssim_ff_test)),2)))
print('RMSE R2*: '+str(round(np.mean(np.array(rmse_r2_test)),2)))
print('RMSE R2* ROI: '+str(round(np.mean(np.array(rmse_r2_test_roi)),2)))
print('RMSE FF: '+str(round(np.mean(np.array(rmse_ff_test)),2)))
print('RMSE FF ROI: '+str(round(np.mean(np.array(rmse_ff_test_roi)),2)))

col = ['Id','Zone','RMSE R2* tot','RMSE R2* ROI','Diff R2*','SSIM R2*','RMSE FF tot','RMSE FF ROI','Diff FF','SSIM FF']
id_p= [x[0] for x in res_test]
z = [x[1] for x in res_test]
r = list(zip(id_p,z,rmse_r2_test,rmse_r2_test_roi,r2_err,ssim_r2_test,rmse_ff_test,rmse_ff_test_roi,ff_err,ssim_ff_test))
# df1 = pd.DataFrame(res_train,columns=col)
# df1 = df1.set_index('Id')
# df2 = pd.DataFrame(res_val,columns=col)
# df2 = df2.set_index('Id')
df3 = pd.DataFrame(r,columns=col)
df3 = df3.set_index('Id')
# df4 = pd.DataFrame(res_tot,columns=col)
# df4 = df4.set_index('Id')

with pd.ExcelWriter('metriche_new.xlsx') as writer:  
    #df1.to_excel(writer, sheet_name='Training')
    #df2.to_excel(writer, sheet_name='Validation')
    df3.to_excel(writer, sheet_name='Test')
    #df4.to_excel(writer, sheet_name='Total')

######################## Plot tra errore R2* e rmse

r2_err = abs(np.array([x[3] for x in res_test])-np.array([x[4] for x in res_test]))
plt.figure(figsize=(5, 5))
plt.scatter(r2_err, rmse_r2_test)
plt.title('T2*')
plt.xlabel('RMSE [ms]')
plt.ylabel('Difference [ms]')
#plt.savefig('diff_t2.png')

ff_err = abs(np.array([x[6] for x in res_test])-np.array([x[7] for x in res_test]))
plt.figure(figsize=(5, 5))
plt.scatter(ff_err, rmse_ff_test)
plt.title('FF')
plt.xlabel('RMSE [%]')
plt.ylabel('Difference [%]')
#plt.savefig('diff_ff.png')


############################### Bland-altman plot
from statsmodels.graphics import utils
def mean_diff_plot_custom(m1, m2, perc = False, sd_limit=1.96, ax=None, scatter_kwds=None,
                    mean_line_kwds=None, limit_lines_kwds=None):
    
    fig, ax = utils.create_mpl_ax(ax)

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    if perc:
        diffs = np.divide((m1-m2),m1)*100
    else:
        diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds) # Plot the means against the diffs.
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate(f'-SD{sd_limit}: {lower:0.2g}',
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=14,
                    xycoords='axes fraction')
        ax.annotate(f'+SD{sd_limit}: {upper:0.2g}',
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=14,
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
    
    if perc:
        ax.set_ylabel('Difference [%]', fontsize=15)
        ax.set_xlabel('Means [ms]', fontsize=15)
    else:
        ax.set_ylabel('Difference [ms]', fontsize=15)
        ax.set_xlabel('Means [ms]', fontsize=15)
        
    # ax.set_ylabel('Difference [%]', fontsize=15)
    # ax.set_xlabel('Means [%]', fontsize=15)

    
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    return fig

f, ax = plt.subplots(1, figsize = (8,5))
mean_diff_plot_custom(np.array([x[3] for x in res_test]), np.array([x[4] for x in res_test]), ax = ax)
plt.title('T2* test')
plt.axhline(0,color='gray', linestyle='--')
ax.axvline(26,color='gray', linestyle='--')
plt.show()

f, ax = plt.subplots(1, figsize = (8,5))
mean_diff_plot_custom(np.array([x[6] for x in res_test]), np.array([x[7] for x in res_test]), ax = ax)
plt.title('FF test')
plt.axhline(0,color='gray', linestyle='--')
ax.axvline(6.5,color='gray', linestyle='--')
plt.show()

f, ax = plt.subplots(1, figsize = (8,5))
mean_diff_plot_custom(np.array([x[3] for x in res_test]), np.array([x[4] for x in res_test]),perc = True, ax = ax)
plt.title('T2* test relative')
plt.axhline(0,color='gray', linestyle='--')
ax.axvline(26,color='gray', linestyle='--')
plt.show()

f, ax = plt.subplots(1, figsize = (8,5))
mean_diff_plot_custom(np.array([x[6] for x in res_test]), np.array([x[7] for x in res_test]),perc = True, ax = ax)
plt.title('FF test relative')
plt.axhline(0,color='gray', linestyle='--')
ax.axvline(6.5,color='gray', linestyle='--')
plt.show()

############################################ Creazione file excel
col = ['Id','Zone','T2* HIPPO','T2* Hernando','T2* Unet','FF HIPPO','FF Hernando','FF Unet']
# df1 = pd.DataFrame(res_train,columns=col)
# df1 = df1.set_index('Id')
# df2 = pd.DataFrame(res_val,columns=col)
# df2 = df2.set_index('Id')
df3 = pd.DataFrame(res_test,columns=col)
df3 = df3.set_index('Id')
# df4 = pd.DataFrame(res_tot,columns=col)
# df4 = df4.set_index('Id')

with pd.ExcelWriter('output_new.xlsx') as writer:  
    #df1.to_excel(writer, sheet_name='Training')
    #df2.to_excel(writer, sheet_name='Validation')
    df3.to_excel(writer, sheet_name='Test')
    #df4.to_excel(writer, sheet_name='Total')
    

# ########################################### Salvataggio parametri
# # # serialize model to JSON
# # model_json = model.to_json()
# # with open("model.json", "w") as json_file:
# #     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_new.h5")
# print("Saved model to disk")

# # # load json and create model
# # from keras.models import model_from_json
# # json_file = open('model.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# # loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# #loaded_model.load_weights("model.h5")
# model.load_weights("model.h5")
# print("Loaded model from disk")

##################### Calcolo tempo
# import time
# tic = time.time()

# def results_time(X, y, model, tic):

#     for j in range(0,len(X)):
        
#         imgtest = tf.expand_dims(X[j], axis=0)
#         predictions = model.predict(imgtest)
#         r2_im=np.reshape(predictions[:,:,:,0],(64,64))
#         ff_im=np.reshape(predictions[:,:,:,1],(64,64))
#         true=y[j]
        
#     toc = time.time()  
#     return toc-tic

# time = results_time(images,labels,model,tic)
# print(time)