import os
import tensorflow as tf
from tensorflow.python.client import device_lib

from model import Vnet3dModule
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '4' #use GPU with ID=4
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75 # maximun alloc gpu 75% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)


def train():
    # Read  dataset images and masks
    csvmaskdata = pd.read_csv('train_Y.csv')
    csvimagedata = pd.read_csv('train_X.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    print(maskdata)
    imagedata = csvimagedata.iloc[:, :].values

    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    Vnet3d = Vnet3dModule(256, 256, 16, channels=1, costname=("dice coefficient",))
    Vnet3d.train(imagedata, maskdata, "vnet3d-model.pd", "logs/", 0.001, 0.7, 10, 1)


train()
