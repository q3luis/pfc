__author__ = 'luispeinado'


import scipy.io
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
from scipy import misc
import os

flag = True;
filepath='../data/configuration/ConfigurationVidriloPrecomputedFeatures.mat'

output_path='../data/datasets/'

image_path='../data/sequences/Sequence2/visualInformation/'

sequences=["Sequence1","Sequence2","Sequence3","Sequence4","Sequence5"]



def loadConfig():
    try:
        f = h5py.File(filepath)
    except:
        flag = False;
    try:
        if not flag:
            f = scipy.io.loadmat(filepath)
    except:
        print('Error')

    return f['Configuration']

def save(values):

    print "save"


def getConfigSequence(conf,seq_name):
    s1=conf[seq_name]
    visualPath=s1[0][0][0][0][0][0]
    object_class=s1[0][0][0][0][2][0]
    image_label=[(visualPath[i][0],object_class[i]) for i in range(0,len(visualPath))]
    return image_label


for i in sequences:
    conf= loadConfig()
    image_label=getConfigSequence(conf,i)
    print i
    file_ = open(output_path+i, 'w')
    for i in range(0,len(image_label)):
        (image,clase)= image_label[i]
        file_.write(''+image_path+str(i)+str(image)+' '+str(clase)+'\n')
    file_.close()

