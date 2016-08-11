# Inicializamos las librerias de cafe
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Funcion que no sirve para hacer la prediccion
def predict(image_in):
    net.blobs['data'].data[...] = image_in
    out = net.forward()
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    output_prob = out['prob'][0]
    #print 'predicted class is:', output_prob.argmax()
    return output_prob.argmax()

datum = caffe.proto.caffe_pb2.Datum()
# Caffe en modo gpu para evaluar mas rapido
caffe.set_mode_gpu()

def load_sequence(path):
    lmdb_env = lmdb.open(path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    return lmdb_cursor

def load_model(model_path,deploy_path):
    #deploy_path="/vol/pfc/prototxt/sequence1/train_sequence_25_lmdb_deploy.prototxt"    
    net = caffe.Net(deploy_path,model_path, caffe.TEST)
    return net
   
def load_transform(net,mean_path):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    return transformer


sequences=["/vol/pfc/data/datasets/Sequence_1_3_lmdb"]

models=["/vol/pfc/data/models/sequence_2_4_5_oversample_25/train_sequence_2_4_5_25_100000_lmdb_iter_200000.caffemodel" ]

means=["/vol/pfc/data/means/Sequence_2_4_5_oversampel_scikit_learn_mean.npy"]

deploys=["/vol/pfc/prototxt/sequence_2_4_5_oversample/train_sequence_25_lmdb_deploy.prototxt"]

def process_sequence(net,sequence,transform):
    data_array = []
    cnt=0
    for key, value in sequence:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        image = np.transpose(data,(1, 2, 0))
        prediccion = predict(transform.preprocess('data', image))
        data_array.append((label,prediccion)) 
        
    return data_array


import time
print "sleep" 
# Wait for 5 seconds
#time.sleep(14400)

print "init "


len_model=len(models)
len_sequences=len(sequences)
print "inicio"
data_evaluate2=[]
for i in range(0,len_model):
    model=models[i]
    deploy=deploys[i]
    print 'evalua el  modelo: '+str(model)
    net=load_model(model,deploy)
    transform=load_transform(net,means[i])
    data_evaluate2.append([])
    for j in range(0,len_sequences):
        print 'evalua sequence '+str(sequences[j])
        data=load_sequence(sequences[j])
        data_evaluate2[i].append(process_sequence(net,data,transform))
    print "end "+str(model)



X= np.array(data_evaluate2)

print X.dtype


def get_accuracy(y):
    y_real=y[:,0]
    y_pred=y[:,1]
    metrics= y_real==y_pred
   # accuracy =(np.sum(metrics)*100.0)/len(metrics)
    #print "accruracy "+"{:10.4f}".format(accuracy) +"%"
    accuracy = precision_score(y_real, y_pred, average='micro')
    return "{:10.4f}".format(accuracy)+"%"


len_data_evaluate_model=len(data_evaluate2)

seq_model_label=['seq1_model','seq2_model','seq4_model','seq5_model']
seq_dataset_label=['seq1','seq2','seq3','seq4','seq5']

for i in range(0,len_data_evaluate_model):
    len_dataset_evaluate=len(data_evaluate2[i])
    seq_for_model=data_evaluate2[i]
    print "model "+seq_model_label[i]
    result_line=""
    for j in range(0,len_dataset_evaluate):
        result_line+=" "+seq_dataset_label[j]+": "+(get_accuracy(np.array(seq_for_model[j])))
    
    print result_line

#np.savetxt("test.out", X, fmt="%u", delimiter=",")
print "end"
import pickle

output = open('test.pkl', 'wb')
pickle.dump(X, output)
output.close()



