__author__ = 'luispeinado'

import sys
from zipfile import ZipFile
from urllib import urlretrieve
from tempfile import mktemp

url_base="http://www.rovit.ua.es/dataset/vidrilo/"
sequences=["Sequence1","Sequence2","Sequence3","Sequence4","Sequence5"]
destino_path="../data/sequences/"


#filename = mktemp('.zip')
#destDir = mktemp()
#theurl = 'http://www.rovit.ua.es/dataset/vidrilo/Sequence1.zip'
#name, hdrs = urlretrieve(theurl, filename)
#thefile=ZipFile(filename)
#thefile.extractall(destino_path)
#thefile.close()


def download_unzip(name):

    filename = '/vol/tmp/'+name+'.zip'
    destDir = mktemp()
    theurl = 'http://www.rovit.ua.es/dataset/vidrilo/'+name+'.zip'
    print "init download "+str(theurl)
    name, hdrs = urlretrieve(theurl, filename)
    thefile=ZipFile(filename)
    thefile.extractall(destino_path)
    thefile.close()
    print "end download "+str(theurl)


#for i in sequences:
i= sys.argv[1]

download_unzip(i)
