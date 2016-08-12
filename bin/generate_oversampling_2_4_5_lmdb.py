__author__ = 'luispeinado'


### Generamos el oversampling de las clases 2 4 5 para entrenar contra la 3 y validar contra la 1


import sys
sys.path.insert(0, '../lib/')
sys.path.append( '../lib')
import lib.utils as ut
import lib.caffe_utils as cut
import lib.sampling_util as smpl


base_path= "/vol/pfc/data/datasets/"

means_path="/vol/pfc/data/means/"



sequences=[ "Sequence2","Sequence4","Sequence5"]



for seq in sequences:
    smpl.generate_oversample_from_file(base_path+seq,base_path+seq+"_oversampled")
    lmdb_path= cut.generate_lmbd_dataset(base_path+seq+"_oversampled",base_path+seq+"_oversampled"+"_240_320","240","320")
    cut.generate_imagen_meadn(lmdb_path,means_path+seq+"_240_320")