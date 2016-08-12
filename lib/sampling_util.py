__author__ = 'luispeinado'


from imblearn.over_sampling import RandomOverSampler
import numpy as np
from  collections import Counter


def _load_file(path):
    lines= [ line.rstrip("\n").split(" ") for line in open(path) ]
    return lines

def _to_file(X,y,path):
    file_ = open(path, 'w')
    for i in range(0,len(X)):
        file_.write(''+str(X[i][0])+' '+str(y[i])+'\n')

    file_.close()


# genera fichero nuevo de oversample de los datos de entrada

def generate_oversample_from_file(input_path_file,output_path_file):

    l= np.array(_load_file(input_path_file))
    X= [[i] for i in  l[:,0] ]
    y= l[:,1]
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_sample(X, y)
    c= Counter(y)
    print "old values "+str(c)
    c= Counter(y_resampled)
    print "new values"+str(c)

    _to_file(X_resampled,y_resampled,output_path_file)
    print "end"

