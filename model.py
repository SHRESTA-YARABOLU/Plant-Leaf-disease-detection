import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical # type: ignore
from keras.models import Model,Sequential,load_model# type: ignore
from keras import Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D# type: ignore
from keras.optimizers import Adam# type: ignore
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau# type: ignore
from keras.applications import DenseNet121# type: ignore
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image# type: ignore

def build_model(input_shape, num_classes):
    densenet = DenseNet121(weights='imagenet', include_top=False)
    input = Input(shape=input_shape)
    x = Conv2D(3, (3, 3), padding='same')(input)
    x = densenet(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(input, output)
    optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_model(X_train, Y_train, X_val, Y_val, input_shape, num_classes, epochs=6, batch_size=64):
    model = build_model(input_shape, num_classes)
    datagen = ImageDataGenerator(rotation_range=360, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
    datagen.fit(X_train)
    model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
              steps_per_epoch=X_train.shape[0] // batch_size,
              epochs=epochs,
              validation_data=(X_val, Y_val))
    return model

def predict_disease(model, image_path, disease_classes):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return disease_classes[class_index]
