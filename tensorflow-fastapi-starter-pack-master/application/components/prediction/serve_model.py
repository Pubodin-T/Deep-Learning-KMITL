from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from rembg import remove
model = None


def load_model():
    model = tf.keras.models.load_model('C:/Users/oatxs/Desktop/Hocrox/tensorflow-fastapi-starter-pack-master/application/components/optimized-model')
    print("Model loaded")
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()
    
    
    
    image = image.resize((224, 224),Image.BILINEAR)
    image = remove(image)
    backgroud = Image.open("./backgroud.png")
    image = Image.composite(image, backgroud ,image)
    image.save('./test.png')
    image = np.array(image)
    image = np.reshape(image ,(1,224,224,3))
    predict = model.predict(image)
    print(predict)
    label = ['langsat','longgong','longan']


    response = {}
    for i in range(3):
        response[label[i]] = float(predict[0][i].item())
    
    return response


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
