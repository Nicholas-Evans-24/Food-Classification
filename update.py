# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:41:58 2021

https://www.kaggle.com/hemangshrimali/foodpedia-v1-70-acc

@author: nicke
"""

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import time

def train():
    
    imageSize = 100
    batch_size = 50
    
    train_data_size = 75750
    test_data_size = 25250
    
    train_parent_dir = 'C:/Users/nicke/Desktop/Mini Project 2/Images/train_data'
    test_parent_dir = 'C:/Users/nicke/Desktop/Mini Project 2/Images/validation_data'
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, imageSize, imageSize)
    else:
        input_shape = (imageSize, imageSize, 3)
        
    NAME = "No Extra preprocessing/Only Scaling-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs3/{}'.format(NAME))
    print(NAME)
    
    reduce = ReduceLROnPlateau(monitor = 'val_loss',patience = 1)
    early_stop = EarlyStopping(monitor = 'val_loss',patience = 5,restore_best_weights = True)

    model = Sequential()
    
    model.add(Conv2D(8, (3,3), input_shape = input_shape,padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
   
    model.add(Conv2D(16, (3,3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(32, (3,3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Conv2D(64, (3,3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(128, (3,3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    print("\n\nAfter Conv2D(64)\n", model.input_shape)
    print(model.output_shape)
        
    model.add(Flatten())    
    
    print("\n\nAfter Flatten())\n", model.input_shape)
    print(model.output_shape)


    print("\n\nAfter Dense(8192))\n", model.input_shape)
    print(model.output_shape)

    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(101))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
        
    model.add(Dense(101))
    model.add(Activation('softmax'))
    
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['categorical_accuracy'])
    
    
    
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
        
    
    
    
    train_generator = train_datagen.flow_from_directory(
        train_parent_dir,
        target_size=(imageSize, imageSize),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')
    
    
    validation_generator = test_datagen.flow_from_directory(
        test_parent_dir,
        target_size=(imageSize, imageSize),
        batch_size=batch_size,
        class_mode='categorical')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=(train_data_size/50),
        epochs=50,
        validation_data=validation_generator,
        validation_steps=(test_data_size/50), 
        callbacks=[tensorboard],
        verbose=1)
    
    
train()