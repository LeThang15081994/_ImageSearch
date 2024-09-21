from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras.models import Model, load_model
import os

from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt


class feature_extract:
    def __init__(self):
        pass
    
    def get_model_extract(self): # make a model to extract features use pretrain of VGG16
        vgg16_model = VGG16(weights = 'imgaenet') 
        model_extract = Model(inputs = vgg16_model.input, outputs = vgg16_model.get_layer("fc1").output)
        return model_extract
    
    def img_preprocess(seft, img): # image preprocessing convert to tensor
        img = image.resize((224,224)) 
        img = img.convert('RGB') 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0) # add batch_size axis
        x = preprocess_input(x) # image normalized
        return x
    
    def vector_normalized(self, model, img_path): # extract vector and normalized
        print('processing...', img_path)
        img = Image.open(img_path)
        img_tensor = self.img_preprocess(img)

        vector = model.predict(img_tensor)[0] # get value from 2D to 1D
        vector = vector / np.linalg.norm(vector) # normalized
        return vector

    def save_model(self, model, save_path):
        model.save(save_path)
        print("Model saved to", save_path)

    def load_model(self, load_path):
        model = load_model(load_path)
        print("Model loaded from", load_path)
        return model
        



