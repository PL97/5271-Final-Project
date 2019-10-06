#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import random
import numpy as np
import math
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


def GenData(d1, d2):
    randVal = random.randint(0,255)
    image = np.zeros([d1, d2])
    idx1 = random.randint(0, d1-1)
    idx2 = random.randint(0, d2-1)
    image[idx1, idx2] = randVal
    return image

# test
# x = GenData(5, 5)
# print(x)
# print(x[:2].flatten)


# In[3]:


def bitToLabel(bits, label_num):
    x = math.ceil(len(bits)/label_num)
    label = [0]*x
    for i in range(x):
        label[i] = int(bits[i*label_num:(i+1)*label_num], 2)
    return label
        
# bits = "0011010011"
# bitToLabel(bits, 3)
    


# In[4]:


# input: image
# return bits of image in one-dimention like
def getSecret(image):
    return np.unpackbits(image.flatten())
  
# image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
# getSecret(image)


# In[5]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

DIR = "cifar-10-batches-py"
FILE_LIST = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

data = unpickle(DIR + "/" + FILE_LIST[0])
for x in data:
    print(x)


# In[6]:


X = data[b'data']
flat = X[0]

def flatToImg(flat):
    img = np.reshape(flat, (3, 32, 32))
    img = np.transpose(img, (1, 2, 0))
    return img

# read and show image
# img = flatToImg(flat)
# print(img.shape)
# plt.imshow(img)
# plt.show()


# In[9]:


def getData(X):
    for x in X:
        img = flatToImg(x)
        s = getSecret(img)
        print(len(s))
        break
getData(X)


# In[28]:


def 




(s, toImg=False):
    if not toImg:
        return np.packbits(s)
    else:
        return flatToImg(np.packbits(s))

    
# s = getSecret(flat)
# img = decode(s, True)
# plt.imshow(img)
# plt.show()


# In[ ]:




