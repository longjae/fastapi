import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing import image


def load_weight(img):
    class_names = ['.ipynb_checkpoints', '1. Eczema', '10. Warts Molluscum and other Viral Infections', '2. Melanoma', '3. Atopic Dermatitis', '4. Basal Cell Carcinoma (BCC)', '5. Melanocytic Nevi (NV)', '6. Benign Keratosis-like Lesions (BKL)', '7. Psoriasis pictures Lichen Planus and related diseases', '9. Tinea Ringworm Candidiasis and other Fungal Infections']
    new_model = tf.keras.models.load_model("./weights/se-resnet50.hdf5")
    print("load_weight ==> Made new_model")
    pred = new_model.predict(img)
    pred_classes = pred.argmax(axis=1)[0]
    pred_name = class_names[pred_classes]
    print("pred_class: ", pred_classes)
    print("pred_name:", pred_name)
    return pred_name

def predict(file_dir):
    img_width, img_height = 225, 225
    img = image.load_img(file_dir, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    print("predict ==> Made image")
    pred = load_weight(img)
    return pred