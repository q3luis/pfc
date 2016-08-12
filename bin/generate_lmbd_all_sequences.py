__author__ = 'luispeinado'


import sys
sys.path.insert(0, '../lib/')
sys.path.append( '../lib')
import lib.utils as ut
import lib.caffe_utils as cut


base_path= "/vol/pfc/data/"

means_path="/vol/pfc/data/"



sequences=["Sequence1", "Sequence2","Sequence3","Sequence4","Sequence5"]



for seq in sequences:
    lmdb_path= cut.generate_lmbd_dataset(base_path+seq,base_path+seq+"_240_320",240,320)
    cut.generate_imagen_meadn(lmdb_path,means_path)




