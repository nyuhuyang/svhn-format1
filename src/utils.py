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

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue

    return shared_x, T.cast(shared_y, 'int32')

    # return shared_x, T.cast(shared_y, 'int32')

def load_data(theano_shared=True):
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    # Download the SVHN dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            '..',
            'data',
            dataset
        )
        if (not os.path.isfile(new_path)):
            origin = (
                'http://ufldl.stanford.edu/housenumbers/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    train_dataset = check_dataset('train.tar.gz')
    test_dataset = check_dataset('test.tar.gz')

    def format_data(dataset):

        tar = tarfile.open(os.path.join(os.path.split(__file__)[0],
        '..',
        'data',
        dataset
        ), 'r:gz')

        data_file_split = os.path.splitext(dataset)[0]
        data_type = os.path.splitext(data_file_split)[0]

        def check_file(folder_name):
            new_path = os.path.join(
                os.path.split(__file__)[0],
                '..',
                'data',
                folder_name)
            if (not os.path.isfile(new_path)):
                tar.extractall(os.path.join(
                    os.path.split(__file__)[0],
                    '..',
                    'data'))
                process_data()

        def process_data():
            print '... processing data (should only occur when downloading for the first time)'
            # Access label information in digitStruct.mat
            new_path = os.path.join(os.path.split(__file__)[0],
                '..',
                'data',
                data_type,
                'digitStruct.mat')
            f = h5py.File(new_path, 'r')

            digitStructName = f['digitStruct']['name']
            digitStructBbox = f['digitStruct']['bbox']

            def getName(n):
                return ''.join([chr(c[0]) for c in f[digitStructName[n][0]].value])

            def bboxHelper(attr):
                if (len(attr) > 1):
                    attr = [f[attr.value[j].item()].value[0][0] for j in range(len(attr))]
                else:
                    attr = [attr.value[0][0]]
                return attr

            def getBbox(n):
                bbox = {}
                bb = digitStructBbox[n].item()
                # bbox = bboxHelper(f[bb]["label"])
                bbox['height'] = bboxHelper(f[bb]["height"])
                bbox['label'] = bboxHelper(f[bb]["label"])
                bbox['left'] = bboxHelper(f[bb]["left"])
                bbox['top'] = bboxHelper(f[bb]["top"])
                bbox['width'] = bboxHelper(f[bb]["width"])
                return bbox

            def getDigitStructure(n):
                s = getBbox(n)
                s['name'] = getName(n)
                return s

            # Process labels
            print '... creating image box bound dict for %s data' % data_type
            image_dict = {}
            for i in range(len(digitStructName)):
                image_dict[getName(i)] = getBbox(i)
                if (i%1000 == 0):
                    print '     image dict processing: %i/%i complete' %(i,len(digitStructName))
            print '... dict processing complete'

            # Process the data
            print('... processing image data and labels')

            names = []
            for item in os.listdir(os.path.join(os.path.split(__file__)[0], '..', 'data', data_type)):
                if item.endswith('.png'):
                    names.append(item)

            y = []
            x = []
            for i in range(len(names)):
                path = os.path.join(os.path.split(__file__)[0], '..', 'data', data_type)
                y.append(image_dict[names[i]]['label'])
                image = Image.open(path + '/' + names[i])
                left = int(min(image_dict[names[i]]['left']))
                upper = int(min(image_dict[names[i]]['top']))
                right = int(max(image_dict[names[i]]['left'])) + int(max(image_dict[names[i]]['width']))
                lower = int(max(image_dict[names[i]]['top'])) + int(max(image_dict[names[i]]['height']))
                image = image.crop(box = (left, upper, right, lower))
                image = image.resize([32,32])
                image_array = numpy.array(image)
                x.append(image_array)
                if (i%1000 == 0):
                    print '     image processing: %i/%i complete' %(i,len(names))
            print '... image processing complete'

            # Save data
            print '... pickling data'
            out = {}
            out['names'] = names
            out['labels'] = y
            out['images'] = x
            output_file = data_type + 'pkl.gz'
            out_path = os.path.join(os.path.split(__file__)[0], '..', 'data', output_file)
            p = gzip.open(out_path, 'wb')
            pickle.dump(out, p)
            p.close()

            tar.close()

        check_file(data_type)

    # format_data('train.tar.gz')
    # format_data('test.tar.gz')

    # Load the dataset

    if (not os.path.isfile(os.path.join(os.path.split(__file__)[0],  '..', 'data', 'trainpkl.gz'))):
        format_data('train.tar.gz')

    f_train = gzip.open(os.path.join(os.path.split(__file__)[0],  '..', 'data', 'trainpkl.gz'), 'rb')
    train_set = pickle.load(f_train)
    f_train.close()

    if (not os.path.isfile(os.path.join(os.path.split(__file__)[0],  '..', 'data', 'testpkl.gz'))):
        format_data('test.tar.gz')

    f_test = gzip.open(os.path.join(os.path.split(__file__)[0],  '..', 'data', 'testpkl.gz'), 'rb')
    test_set = pickle.load(f_test)
    f_test.close()

    # Convert data format
    def convert_data_format(data):
        data['X'] = data.pop('images')
        data['X'] = numpy.array(data['X'])
        data['X'] = numpy.rollaxis(data['X'],0,4)
        data['y'] = data.pop('labels')

        X = numpy.reshape(data['X'],
                          (numpy.prod(data['X'].shape[:-1]), data['X'].shape[-1]),
                          order='C').T / 255.

        def process_sequence(labels):
            for i in range(len(labels)):
                l = len(labels[i])-1
                labels[i].insert(0,l)
                zeros = numpy.zeros(6-l-1).tolist()
                labels[i].extend(zeros)
            return numpy.array(labels)

        y = process_sequence(data['y'])
        # y = data['y'].flatten()
        # y[y == 10] = 0
        return (X,y)

    train_set = convert_data_format(train_set)
    test_set = convert_data_format(test_set)

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    # if ds_rate is not None:
    #     train_set_len = int(train_set_len // ds_rate)
    #     train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    else:
        rval = [train_set, valid_set, test_set]

    return rval

# print not os.path.isfile(os.path.join(os.path.split(__file__)[0],  '..', 'data', 'trainpkl.gz'))
if __name__ == '__main__':
    load_data(theano_shared=False)
