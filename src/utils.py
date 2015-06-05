import caffe
import numpy as np
import os
import time
import hickle as hkl


use_alexnet = True

feature_layers = ['fc7', 'fc6', 'pool5', 'conv4', 'conv3', 'pool2', 'pool1']
feature_dir = "../features"
compression_dir = "../compression"
distances_dir = "../distances"
db_dir = "../db"
caffe_root = '/home/eric/caffe/caffe-master/'


# database configuration
user = 'postgres'
password = 'asdfgh'
host = '127.0.0.1'
dbname = 'mydb'

import cv2

def get_dimension_options(layer, compression):
    """
    Returns an array of all the possible compression sizes for that layer / compression pair

    :type layer: str
    :param layer: feature layer

    :type compression: str
    :param compression: compression type identifier (pca, kpca, etc.)

    :rtype: array-like
    :return: dimensions
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    compresion_path = os.path.join(compression_dir, compression, layer)
    files = os.listdir(compresion_path)
    N = len(files)

    if N <= 1:
        raise ValueError('Path provided contained no stored algorithms : ' + compresion_path)

    # there is a holder file in each directory which needs to be removed
    files.remove('holder.txt')

    dimensions = []
    for file in files:
        name, dim, postfix = file.split('_')
        dimensions.append(int(dim))

    return dimensions


def load_english_labels():
    """
    Returns a dictionary from class # to the english label.

    :return: labels
    """
    imagenet_labels_filename = os.path.join('../caffe/synset_words.txt')
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        raise ValueError('Could not find synset_works in the correct place.')

    return labels

def load_db_labels():
    fo = open("../db/labels.txt", "r+")

    # remove the /n
    content = fo.read().splitlines()
    labels = {}

    for line in content:
        klass, label = line.split(' ')
        labels[int(klass)] = label

    fo.close()
    return labels


def load_compressor(layer, dimension, compression):
    """
    Loads the compression algorithm from the file system

    :type layer: str
    :param layer: feature layer

    :type dimension: int
    :param dimension: n_components of compressor

    :type compression: str
    :param compression: Compressional algorithm ID

    :return: Compression algorithm
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    compression_path = os.path.join(compression_dir, compression, layer)
    file_name = compression + '_' + str(dimension) + '_gzip.hkl'

    file_path = os.path.join(compression_path, file_name)

    return hkl.load(file_path, safe=False)


def batch_gen(data, batch_size):
    """
    Simple generator for looping over an array in batches

    :type data: array-like
    :param data:

    :type batch_size: int
    :param batch_size:

    :return: generator
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def load_network():
    """
    Loads the caffe network. The type of network loaded is specified in the utils file.

    :return: caffe network
    """
    if not use_alexnet:
        # Set the right path to your model definition file, pretrained model weights,
        # and the image you would like to classify.
        MODEL_FILE = '../caffe/bvlc_reference_caffenet/deploy.prototxt'
        PRETRAINED = '../caffe/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    else:
        # ALEXNET
        MODEL_FILE = '../caffe/bvlc_alexnet/deploy.prototxt'
        PRETRAINED = '../caffe/bvlc_alexnet/bvlc_alexnet.caffemodel'

    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')),
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(256, 256))

    blobs = [(k, v.data.shape) for k, v in net.blobs.items()]
    params = [(k, v[0].data.shape) for k, v in net.params.items()]

    print 'Blobs : ', blobs
    print 'Params : ', params

    net.set_phase_test()
    net.set_mode_gpu()

    return net, params, blobs

def load_scalar(layer):
    """
    Load the feature mean / variance scalar for the input layer

    :type layer: str
    :param layer: Feature layer

    :return: scalar
    """
    if not layer in feature_layers:
        raise NotImplementedError('Feature Layer Type Not Found.')

    features_path = os.path.join(feature_dir, layer)
    files = os.listdir(features_path)
    N = len(files)

    if N <= 1:
        raise ValueError('Path provided contained no features : ' + features_path)

    for file in files:
        sp = file.split('_')
        if 'scalar' in sp:
            scalar = hkl.load(os.path.join(features_path, file), safe=False)

    return scalar

def load_distance_matrix(layer):
    """
    Returns the distance matrix as defined by the features of the provided layer
    Note that this must be generated beforehand using generate_dist_func

    :type layer: str
    :param layer: Feature layer

    :return: numpy array
    """
    return hkl.load(os.path.join(distances_dir, 'dist_matrix_' + layer + '.hkl'))

def dump_feature_db(comp_fc7, ids, fc7_feats, pool5_feats):
    """
    Saves out the feature layer using hickle

    :return:
    """

    file_name = 'fc7_feats.hkl'
    file_path = os.path.join(db_dir, 'feats', file_name)
    print 'Saving : ', file_path
    hkl.dump(fc7_feats, file_path, mode='w', compression='gzip')

    file_name = 'ids.hkl'
    file_path = os.path.join(db_dir, 'feats', file_name)
    print 'Saving : ', file_path
    hkl.dump(ids, file_path, mode='w', compression='gzip')


    file_name = 'comp_fc7.hkl'
    file_path = os.path.join(db_dir, 'feats', file_name)
    print 'Saving : ', file_path
    hkl.dump(comp_fc7, file_path, mode='w', compression='gzip')


    file_name = 'pool5_feats.hkl'
    file_path = os.path.join(db_dir, 'feats', file_name)
    print 'Saving : ', file_path
    hkl.dump(pool5_feats, file_path, mode='w', compression='gzip')

    print 'Saved %s Images into Databse' % ids.shape[0]

def load_feature_db():
    """

    :return: comp_fc7, ids, fc7_feats, pool5_feats
    """

    features_path = os.path.join(db_dir, 'feats')
    files = os.listdir(features_path)
    N = len(files)

    if N <= 1:
        print 'No stored features in Database!'
        return (np.empty((0,256), dtype=np.float32),
               np.empty(shape=(0,1), dtype=np.int32),
               np.empty((0,4096), dtype=np.float32),
                np.empty((0,9216), dtype=np.float32))

    start_time = time.clock()

    fc7_feats = hkl.load(os.path.join(features_path, 'fc7_feats.hkl'))
    pool5_feats = hkl.load(os.path.join(features_path, 'pool5_feats.hkl'))
    ids = hkl.load(os.path.join(features_path, 'ids.hkl'))
    comp_fc7 = hkl.load(os.path.join(features_path, 'comp_fc7.hkl'))

    print 'Load Time Feat DB (s) : ', time.clock() - start_time
    print '%s Instances in DB' % ids.shape[0]

    return comp_fc7, ids, fc7_feats, pool5_feats

def save_image(np_img, inst, type):
    file_path = os.path.join(db_dir,type, 'img_%s.jpeg' % (inst))
    cv2.imwrite(file_path, np_img)
