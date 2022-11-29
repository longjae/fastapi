import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing import image


def load_weight(img):
    new_model = tf.keras.models.load_model("./weights/se-resnet50.hdf5")
    print("load_weight ==> Made new_model")
    pred = new_model.predict(img)
    pred_classes = pred.argmax(axis=1)[0]
    print("pred_class: ", pred_classes)
    return pred_classes

def predict(file_dir):
    img_width, img_height = 225, 225
    img = image.load_img(file_dir, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    print("predict ==> Made image")
    pred = load_weight(img)
    return pred