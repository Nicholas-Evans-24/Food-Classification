# -*- coding: utf-8 -*-
"""

Dataset: https://www.kaggle.com/kmader/food41

@author: Nicholas Evans
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
import cv2
import random
import time
import joblib


def openFile():
    
    directory = 'C:/Users/nicke/Desktop/Mini Project 2/archive/images'
    
    
    CAT = []
    
    #creates list of food names
    for name in os.listdir(directory):
        CAT.append(name)
        
    CAT = CAT[:50]
    
    imageSize = 100
    
    data = []
    
    for category in CAT:
        print("Reading images from: ", category)
        path = os.path.join(directory, category)
        foodNum = CAT.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (imageSize,imageSize))
            data.append([new_array, foodNum])
            
    print("Shuffling...")
    random.shuffle(data)
    print("Done Shuffling")
    
    x = []
    y = []
    
    for pics,label in data:
        x.append(pics)
        y.append(label)
        
    x = np.array(x,dtype=float).reshape(-1, imageSize, imageSize, 3)
    
    one,two,three,four = x.shape
    
    #Dataset is so large
    #It needed to be sorted through individualy
    print("Scaling...")
    i = 0
    for a in range(0, one):
        print("Image Number: ", i)
        i += 1
        for b in range(0, two):
            for c in range(0, three):
                for d in range(0, four):
                    x[a,b,c,d] = x[a,b,c,d]/255.0
                    
                    
    print("Scaling Finished")
    
    


   
    print("Saving Data...")
    
    out = open("x.joblib","wb")
    joblib.dump(x, out)
    out.close()
    
    out = open("y.joblib","wb")
    joblib.dump(y, out) 
    
    
    """
    out = open("x.pickle","wb")
    pickle.dump(x, out)
    out.close()
    
    out = open("y.pickle","wb")
    pickle.dump(y, out) 
    
    """
    print("Data saved")
    
def train():
    
    print("Loading Data...")
    x = joblib.load(open("x.joblib","rb"))
    y = joblib.load(open("y.joblib","rb"))
    print("loaded Data")
    
    
    y = np.array(y)
    
    
    labels = preprocessing.CategoryEncoding()
    labels.adapt(y)
    y = labels(y)
    
    class_Size = len(y[1])
    
   
    

    NAME = "ExtraConv and ExtraDense layers-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs3/{}'.format(NAME))
    print(NAME)
    
    reduce = ReduceLROnPlateau(monitor = 'val_loss',patience = 1)
    early_stop = EarlyStopping(monitor = 'val_loss',patience = 5,restore_best_weights = True)

    model = Sequential()
    
    model.add(Conv2D(8, (3,3), input_shape = x.shape[1:],padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
   
    model.add(Conv2D(16, (3,3), input_shape = x.shape[1:],padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(32, (3,3), input_shape = x.shape[1:],padding="same"))
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
        
        
    model.add(Flatten())    
    
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
    
    model.add(Dense(class_Size))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
        
    model.add(Dense(class_Size))
    model.add(Activation('softmax'))
    
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['categorical_accuracy'])
    
    model.fit(x, y, batch_size=32, epochs=50, validation_split=0.2, callbacks=[tensorboard])


#openFile()
train()