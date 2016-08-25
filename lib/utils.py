__author__ = 'luispeinado'



import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



#Funcion que no sirve para hacer la prediccion
def predict(net,image_in):
    net.blobs['data'].data[...] = image_in
    out = net.forward()
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    output_prob = out['prob'][0]
    #print 'predicted class is:', output_prob.argmax()
    return output_prob.argmax()



datum = caffe.proto.caffe_pb2.Datum()
# Caffe en modo gpu para evaluar mas rapido
#caffe.set_mode_gpu()

caffe.set_mode_cpu()


def load_sequence(path):
    lmdb_env = lmdb.open(path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    return lmdb_cursor

def load_model(model_path,deploy_path):
    #deploy_path="/vol/pfc/prototxt/sequence1/train_sequence_25_lmdb_deploy.prototxt"
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_path,model_path, caffe.TEST)
    return net

def load_transform(net,mean_path):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    return transformer



def process_sequence(net,sequence,transform):
    data_array = []
    cnt=0
    for key, value in sequence:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        image = np.transpose(data,(1, 2, 0))
        #print "image shape "+str(transform.preprocess('data', image).shape)
        prediccion = predict(net,transform.preprocess('data', image))
        data_array.append((label,prediccion))
        print " label "+str(label)+" preiccion "+str(prediccion)
        #cnt+=1
        #if(cnt>10):
        #     break

    return data_array








