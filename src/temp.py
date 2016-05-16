#!/usr/local/bin/python

import os
import sys
import numpy
import scipy.io

import gzip
import tarfile
import h5py
from PIL import Image
import six.moves.cPickle as pickle
from six.moves import urllib

import theano
import theano.tensor as T

def format_data(dataset):

    tar = tarfile.open(os.path.join(os.path.split(__file__)[0],
    '..',
    'data',
    dataset
    ), 'r:gz')

    data_file_split = os.path.splitext(dataset)[0]
    print data_file_split
    data_type = os.path.splitext(data_file_split)[0]
    print data_type


    def check_file(folder_name):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            '..',
            'data',
            folder_name)
        print new_path
        if (not os.path.exists(new_path)):
            tar.extractall(os.path.join(
                os.path.split(__file__)[0],
                '..',
                'data'))
            process_data()
    check_file(data_type)

# format_data('train.tar.gz')
exists = os.path.join(os.path.split(__file__)[0],  '..', 'data', 'train.pkl.gz')
print exists
print not os.path.isfile(os.path.join(os.path.split(__file__)[0],  '..', 'data', 'testpkl.gz'))
print not False
