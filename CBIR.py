from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_model_extract():
    vgg16_model = VGG16(weight='imgaenet') # sử dụng pretrain của VGG16
    model_extract = Model(input=vgg16_model.input, output = vgg16_model.get_layer("fc1").output)
    return model_extract