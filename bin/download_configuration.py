__author__ = 'luispeinado'

import urllib

testfile = urllib.URLopener()
testfile.retrieve("http://www.rovit.ua.es/dataset/vidrilo/ConfigurationVidriloPrecomputedFeatures.mat", "../data/configuration/ConfigurationVidriloPrecomputedFeatures.mat")