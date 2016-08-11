__author__ = 'luispeinado'

import lib.utils as utils

import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

#def run_test():

sequences = ["/vol/pfc/data/datasets/Sequence1_lmdb"]
models = ["/vol/pfc/data/models/train_50_batch/train_sequence_2_50_vs_seq3_lmdb_iter_20000.caffemodel"]

means = ["/vol/pfc/data/means/Sequence2_mean.npy"]

deploys=["/vol/pfc/prototxt/sequence2/train_sequence_25_lmdb_deploy.prototxt"]


len_model=len(models)
len_sequences=len(sequences)
print "inicio"
data_evaluate2=[]
for i in range(0,len_model):
    model=models[i]
    deploy=deploys[i]
    print 'evalua el  modelo: '+str(model)
    net=utils.load_model(model,deploy)
    transform=utils.load_transform(net,means[i])
    data_evaluate2.append([])
    for j in range(0,len_sequences):
        print 'evalua sequence '+str(sequences[j])
        data=utils.load_sequence(sequences[j])
        data_evaluate2[i].append(utils.process_sequence(net,data,transform))
    print "end "+str(model)


seq_model_label=['seq2_model']
seq_dataset_label=['seq1','seq2','seq3','seq4','seq5']


def get_accuracy(y):
    y_real=y[:,0]
    y_pred=y[:,1]
    metrics= y_real==y_pred
    accuracy =(np.sum(metrics)*100.0)/len(metrics)
    #print "accruracy "+"{:10.4f}".format(accuracy) +"%"
    return "{:10.4f}".format(accuracy)+"%"

len_data_evaluate_model=len(data_evaluate2)

for i in range(0,len_data_evaluate_model):
    len_dataset_evaluate=len(data_evaluate2[i])
    seq_for_model=data_evaluate2[i]
    print "model "+seq_model_label[i]
    result_line=""
    for j in range(0,len_dataset_evaluate):
        result_line+=" "+seq_dataset_label[j]+": "+(get_accuracy(np.array(seq_for_model[j])))

    print result_line

