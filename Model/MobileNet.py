import os
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , Conv2D, MaxPool2D
import matplotlib.pyplot as plt

# img1 = cv2.imread(r'H:/My Drive/DEEP_LEARNING/[Augmented]LangSad/test/Blur/LangSad_Blur_1.JPG')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(8,4))
# plt.subplot(121)
# plt.imshow(img1)
# plt.show()

trainpath = [r'H:/My Drive/DEEP_LEARNING/[Augmented]LangSad/train/',r'H:/My Drive/DEEP_LEARNING/[Augmented]LongGong/train/',r'H:/My Drive/DEEP_LEARNING/[Augmented]LumYai/train/']
valpath = [r'H:/My Drive/DEEP_LEARNING/[Augmented]LangSad/val/',r'H:/My Drive/DEEP_LEARNING/[Augmented]LongGong/val/',r'H:/My Drive/DEEP_LEARNING/[Augmented]LumYai/val/']
testpath = [r'H:/My Drive/DEEP_LEARNING/[Augmented]LangSad/test/',r'H:/My Drive/DEEP_LEARNING/[Augmented]LongGong/test/',r'H:/My Drive/DEEP_LEARNING/[Augmented]LumYai/test/']
trainImg = []
valImg = []
testImg = []
for i in trainpath:
  for f in listdir(i):
    if(f != 'desktop.ini'):
        trainImg.append(i+f+'/')
for i in valpath:
  for f in listdir(i):
    if(f != 'desktop.ini'):
        valImg.append(i+f+'/')
for i in testpath:
  for f in listdir(i):
    if(f != 'desktop.ini'):
        testImg.append(i+f+'/')

print(len(trainImg))
print(len(valImg))
print(len(testImg))
def img2data(path):
  rawImgs = []
  labels = []
  i=0
  for imagePath in (path):
    print(imagePath)
    for item in tqdm(listdir(imagePath)):
      file = join(imagePath, item)
      img = cv2.imread(file , cv2.COLOR_BGR2RGB)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
      rawImgs.append(img)
      if imagePath in path[:9]:
        labels.append([1,0,0])
      elif imagePath in path[9:18]:
        labels.append([0,1,0])
      elif imagePath in path[18:27]:
        labels.append([0,0,1])
  return rawImgs, labels

x_train, y_train = img2data(trainImg)
x_val, y_val = img2data(valImg)
x_test, y_test = img2data(testImg)