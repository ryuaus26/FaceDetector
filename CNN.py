import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from keras.layers import Input, DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization
from keras.layers import ReLU, AvgPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.callbacks import ModelCheckpoint

import os
from tqdm import tqdm
import numpy as np

batch_size = 32

train_dir = "/home/ubuntu/Dataset/Train"
val_dir = "/home/ubuntu/Dataset/Val"

weights_save_path = "Weights"
os.makedirs(weights_save_path,exist_ok=True)

image_size =(224,224)
import os
import numpy as np
from keras.utils import load_img, img_to_array


img_size = (224, 224)  # Set the desired image size


train_ds  = tf.keras.utils.image_dataset_from_directory(
    "./Dataset/Train/",
    labels="inferred",
    class_names=["NotFace", "Face"],
    label_mode='int',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
)



val_ds  = tf.keras.utils.image_dataset_from_directory (
    "./Dataset/Val/",
    labels="inferred",
    class_names=["NotFace", "Face"],
    label_mode='int',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    
)

def mobilnet_block(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x
    
input = Input(shape=(224, 224, 3))
x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)

# main part of the model
x = mobilnet_block(x, filters=64, strides=1)
x = mobilnet_block(x, filters=128, strides=1)
x = mobilnet_block(x, filters=128, strides=2)
x = mobilnet_block(x, filters=256, strides=1)
x = mobilnet_block(x, filters=256, strides=2)
x = mobilnet_block(x, filters=512, strides=1)

for _ in range(5):
    x = mobilnet_block(x, filters=512, strides=1)
x = mobilnet_block(x, filters=1024, strides=1)
x = mobilnet_block(x, filters=1024, strides=1)

x = AvgPool2D(pool_size=7, strides=1, data_format='channels_first')(x)
x = Flatten()(x)
output = Dense(units=1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)

checkpoint = ModelCheckpoint(
    './Weights/weights.h5',  # Path to save the weights file
    monitor='val_loss',  # Metric to monitor for saving the weights
    save_best_only=True,  # Save only the best weights
    save_weights_only=True  # Save only the weights, not the entire model
)

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, callbacks=checkpoint, epochs=7, batch_size=batch_size,workers=30)
model.save('FaceDetectModel.h5')
