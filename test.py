import tensorflow as tf
from keras.models import load_model
import keras.utils as image
import numpy as np
from CNN import model

test_model = load_model('model.h5')
img_path = '/Users/ryuaus26/Desktop/Screen Shot 2023-06-28 at 8.14.28 PM.png'  # Replace with the path to your input image file
img = image.load_img(img_path, target_size=(224, 224))  # Set the target size according to your model's input shape
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0  # Normalize the image


predictions = test_model.predict(x)
if predictions > 0.75:
    print("FACE")
else:
    print("Not Face")