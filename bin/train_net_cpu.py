__author__ = 'luispeinado'

import sys

#caffe_root = '/Users/luispeinado/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
#sys.path.insert(0, caffe_root + 'python')

#from sklearn.datasets import load_iris
#import sklearn.metrics
#import numpy as np
#from sklearn.cross_validation import StratifiedShuffleSplit
#import matplotlib.pyplot as plt
import h5py
import caffe
#import caffe.draw

solver_prototxt_filename="solver.prototxt"

caffe.set_mode_cpu()

#caffe.set_device(0)
#caffe.set_mode_gpu()

solver_prototxt_filename = sys.argv[1]
solver = caffe.get_solver(solver_prototxt_filename)
solver.solve()


