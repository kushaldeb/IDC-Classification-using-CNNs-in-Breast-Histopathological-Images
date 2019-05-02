#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:25:48 2019

@author: kushaldeb
"""

import os
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# declaring the constants

batch_size = 64
epochs = 20
img_width, img_height = 100, 100
learn_rate = 2e-4
ngpus = 2
nclasses = 2

# path to save the model

model_path = 'models-inceptionv3'
top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')

train_data_dir = '/home/rtx/kushal/bcd/Round-2/sample_dataset/Training'
validation_data_dir = '/home/rtx/kushal/bcd/Round-2/sample_dataset/Validation'

# Initializing the model

base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nclasses-1, activation='sigmoid')(x)
pmodel = Model(base_model.input, predictions)
#pmodel.summary()

# Preparing the training and validation set

train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

training_data = train_datagen.flow_from_directory(train_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  class_mode='binary')
validation_data = validation_datagen.flow_from_directory(validation_data_dir,
                                                         target_size=(img_width, img_height),
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         class_mode='binary')

print('Training incides are :')
print(training_data.class_indices)
train_steps = training_data.__len__()
validation_steps = validation_data.__len__()

model = multi_gpu_model(pmodel, ngpus)

for layer in model.layers:
    layer.trainable = True
nadam = Nadam(lr = learn_rate)
print('=> creating model replicas for distributed training across {ngpus} gpus <=')
model.compile(optimizer = nadam,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
print('=> done building model <=')

tensorboard = TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='batch')
callbacks_list = [ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
                  tensorboard, EarlyStopping(monitor='val_loss', patience=3, verbose=1)]
print('=> created callback objects <=')

print('=> initializing training loop <=')
history = model.fit_generator(training_data,
                              steps_per_epoch = train_steps,
                              epochs = epochs,
                              validation_data = validation_data,
                              validation_steps = validation_steps,
                              workers = 8,
                              #use_multiprocessing = True,
                              max_queue_size = 500,
                              callbacks = callbacks_list)

import matplotlib.pyplot as plt

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure(1, figsize=(10,5))
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.title('Train loss vs Validation loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure(2, figsize=(10,5))
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Train accuracy vs Validation accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.show()

# print('--------- Loading best weights ---------')
# pmodel.load_weights(final_weights_path)

print('---------- Saving final model ----------')
model.save(os.path.join(os.path.abspath(model_path), 'model.h5'))

print('---------- Saving final model -1 GPU----------')
pmodel.save(os.path.join(os.path.abspath(model_path), 'model-singleGPU.h5'))
