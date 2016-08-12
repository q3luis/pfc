__author__ = 'luispeinado'



import os
#os.system('ls')#

# Inicializamos las librerias de cafe
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum


base_path = "../data/datasets/"
means_path = "../data/means/"

caffe_path="/opt/caffe-gpu"

caffe_convert_imaggen_tool= "/build/tools/convert_imageset"
caffe_compute_mean_tool="/build/tools/compute_image_mean"


resize_height_param="--resize_height"
new_height="240"
resize_width_param="--resize_width"
new_width="320"
sufle_param="--shuffle"

#base_path+output_train_file_name

def execute_comand(comand):
    print "run comand "+str(comand)
    os.system(comand)
    print "end run command "+str(comand)


def generate_lmbd_dataset(input_dataset,output_file,new_height,new_width):
    convert_comand="GLOG_logtostderr=1 "+caffe_path+caffe_convert_imaggen_tool+" " \
               +resize_height_param+"="+new_height+" "+resize_width_param+"="+new_width+" "+ \
               sufle_param+" /"+" "+input_dataset+" "+output_file+"_lmdb"
    execute_comand(convert_comand)
    return output_file+"_lmdb"

def generate_imagen_meadn(input_dataset,output_file):
    mean_comand=caffe_path+caffe_compute_mean_tool+" " \
                +input_dataset+" "+output_file+".binaryproto"
    execute_comand(mean_comand)
    transform_image_mean_to_np(output_file+".binaryproto",output_file)


def transform_image_mean_to_np(input_mean,output_mean):
    print "transform image mean to npy"
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( input_mean , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    np.save( output_mean, out )
    print "end transform image mean to npy"
