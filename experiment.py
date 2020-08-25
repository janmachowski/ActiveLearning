from __future__ import absolute_import
from sklearn import neural_network
from strlearn.streams import StreamGenerator
import strlearn as sl
import weles as ws
import warnings
import arff
import numpy as np
import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt
from strlearn.evaluators import TestThenTrain
from ALS import ALS

# Generate streams

# repeatings = 11
# for i in tqdm(range(1, repeatings)):
#     sl.streams.StreamGenerator(n_drifts=5, n_classes=2, n_features=5, n_chunks=1000, chunk_size=500,
#                                random_state=1410 + i).save_to_arff('strNRec' + str(i) + ".arff")
#     sl.streams.StreamGenerator(n_drifts=5, n_classes=2, n_features=5, n_chunks=1000, chunk_size=500,
#
#                                random_state=1410 + i).save_to_arff('strRec' + str(i) + ".arff")
# Set of parameters
# Non-recurrent drift
dbnames = [
    'strNRec1', 'strNRec2', 'strNRec3', 'strNRec4', 'strNRec5'
    'strNRec6', 'strNRec7', 'strNRec8', 'strNRec9', 'strNRec10'
]
# Recurrent drift
# dbnames = [
#     'strRec1', 'strRec2', 'strRec3', 'strRec4', 'strRec5'
#     'strRec6', 'strRec7', 'strRec8', 'strRec9', 'strRec10'
# ]

clfs = {
    "MLP100adam": neural_network.MLPClassifier(hidden_layer_sizes=(100,), solver='adam'),
    # "MLP100lbfgs": neural_network.MLPClassifier(hidden_layer_sizes=(100,), solver='lbfgs') AttributeError: partial_fit is only available for stochastic optimizer. lbfgs is not stochastic
    "MLP100sgd": neural_network.MLPClassifier(hidden_layer_sizes=(100,), solver='sgd')
}
# Experimental loop
for dbname in tqdm(dbnames, ascii=True, desc="DBS"):
    with open('datasets/%s.arff' % dbname, 'r') as stream:
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        X = data[:, :-1].astype(np.float)
        y = data[:, -1]
        length = len(y)
        evaluate_interval = length
        for clfname in clfs:
            clf = clfs[clfname]
            filename = 'results/%s_%s.csv' % (clfname, dbname)
            if os.path.isfile(filename):
                pass
            else:
                als = ALS(clf, budget=0.7, treshold=0.3)
                als.partial_fit(X=X, y=y, classes=np.unique(y))
                predict = als.predict(X=X)