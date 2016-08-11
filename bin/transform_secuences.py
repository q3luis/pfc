__author__ = 'luispeinado'


import os
#os.system('ls')#


base_path = "../data/datasets/"
means_path = "../data/means/"

caffe_path="/opt/caffe-gpu"

caffe_convert_imaggen_tool= "/build/tools/convert_imageset"
caffe_compute_mean_tool="/build/tools/compute_image_mean"

train_sequences=["Sequence2","Sequence4","Sequence5"]
test_sequences=["Sequence1","Sequence3"]

output_train_file_name="Sequences_2_Oversample"
output_test_file_name="Sequence_1_3"


resize_height_param="--resize_height"
new_height="230"
resize_width_param="--resize_width"
new_width="310"
sufle_param="--shuffle"


def fusion_sequences(origenes,path_destino):
    with open(path_destino, 'wb') as dest:
        for origen in origenes:
            with open(base_path+origen) as o:
                dest.write(o.read())


print "inicio la union de secuencias "
print "train "
#fusion_sequences(train_sequences,base_path+output_train_file_name)
print "test"
#fusion_sequences(test_sequences,base_path+output_test_file_name)

print "fin la union de secuencias "


convert_comand="GLOG_logtostderr=1 "+caffe_path+caffe_convert_imaggen_tool+" "\
               +resize_height_param+"="+new_height+" "+resize_width_param+"="+new_width+" "+\
               sufle_param+" /"+" "+base_path+output_train_file_name+" "+base_path+output_train_file_name+"_lmdb"

print convert_comand
os.system(convert_comand)

#convert_comand="GLOG_logtostderr=1 "+caffe_path+caffe_convert_imaggen_tool+" " \
#               +resize_height_param+"="+new_height+" "+resize_width_param+"="+new_width+" "+ \
#               sufle_param+" /"+" "+base_path+output_test_file_name+" "+base_path+output_test_file_name+"_lmdb"

print convert_comand
#os.system(convert_comand)


mean_comand=caffe_path+caffe_compute_mean_tool+" " \
               +base_path+output_train_file_name+"_lmdb"+" "+means_path+output_train_file_name+".binaryproto"

print mean_comand
os.system(mean_comand)

#mean_comand=caffe_path+caffe_compute_mean_tool+" " \
#            +base_path+output_test_file_name+"_lmdb"+" "+means_path+output_test_file_name+".binaryproto"

print mean_comand
os.system(mean_comand)


