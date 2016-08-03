__author__ = 'luispeinado'


import os
#os.system('ls')#


base_path="../data/datasets/"

caffe_path="/opt/caffe-gpu"

caffe_convert_imaggen_tool= "/build/tools/convert_imageset"
caffe_compute_mean_tool="/build/tools/compute_image_mean"

train_sequences=["Sequence2","Sequence4","Sequence5"]
test_sequences=["Sequence1","Sequence3"]

output_train_file_name="Sequence_2_4_5"
output_test_file_name="Sequence_1_3"


resize_height_param="--resize_height"
new_height="240"
resize_width_param="--resize_width"
new_width="320"
sufle_param="--shuffle"


def fusion_sequences(origenes,path_destino):
    with open(path_destino, 'wb') as dest:
        for origen in origenes:
            with open(base_path+origen) as o:
                dest.write(o.read())


print "inicio la union de secuencias "
print "train "
fusion_sequences(train_sequences,base_path+output_train_file_name)
print "test"
fusion_sequences(test_sequences,base_path+output_test_file_name)

print "fin la union de secuencias "


convert_comand="GLOG_logtostderr=1 "+caffe_path+caffe_convert_imaggen_tool+" "\
               +resize_height_param+"="+new_height+" "+resize_width_param+"="+new_width+" "+\
               sufle_param+" /"+" "+base_path+output_train_file_name+" "+base_path+output_train_file_name+"_lmdb"

print convert_comand

#GLOG_logtostderr=1 /Users/luispeinado/caffe/build/tools/convert_imageset --resize_height=320 --resize_width=240 --shuffle /  sequence2_dataset.txt train_lmdb2


#/Users/luispeinado/caffe/build/tools/compute_image_mean train_lmdb2 mean_image_seq2.binaryproto