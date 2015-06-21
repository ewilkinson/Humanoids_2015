#!/usr/bin/env python

import numpy as np
import time, sys

from generic_seg import GenericSegmenter
import cv2

from scipy.stats import pearsonr, spearmanr

import utils

if __name__ == '__main__':
    net, params, blobs = utils.load_network()

    fc7_compressor = utils.load_compressor(layer='fc7',
                                           dimension=128,
                                           compression='pca')

    fc7_scalar = utils.load_scalar(layer='fc7')
    pool5_scalar = utils.load_scalar(layer='pool5')

    cv2.startWindowThread()
    cv2.namedWindow("Query Window", 1)
    cv2.namedWindow("DCNN Window", 1)

    segmenter = GenericSegmenter(cluster_type="dbscan",
                                 use_gray=False,
                                 depth_max_threshold=4000,
                                 show_segmentation=False,
                                 show_cluster=False,
                                 show_mask=False,
                                 merge_boxes=True)

    comp_fc7, props, fc7_feats, pool5_feats = utils.load_feature_db()

    class_labels = np.zeros(shape=(len(props),), dtype=np.int32)
    aspect_labels = np.zeros(shape=(len(props),), dtype=np.int32)
    for i in range(len(props)):
        class_labels[i] = props[i]['type_id']
        aspect_labels[i] = props[i]['aspect_id']

    try:
        segmenter.listen()

        type_id = utils.query_id('TYPE')
        object_id = utils.query_id('OBJECT')

        print 'Type ID : ', type_id
        print 'Object ID : ', object_id

        while True:
            time.sleep(1)

            # segmenter can be a bit slow to start up
            if segmenter.rgb_imge is None or segmenter.boxes is None:
                continue

            cropped_img, box = utils.crop_segment(segmenter)
            cv2.imshow("Query Window", cropped_img)
            cv2.waitKey(1)

            if utils.query_accept() == -1:
                continue

            prop = {}
            prop['type_id'] = type_id
            prop['object_id'] = object_id

            aspect_id = utils.query_id('ASPECT')
            rotation = utils.query_id('ROTATION')
            prop['aspect_id'] = aspect_id
            prop['rotation'] = rotation

            dcnn_img = utils.trans_img_dcnn(cropped_img, box)
            cv2.imshow("DCNN Window", dcnn_img)
            cv2.waitKey(1)

            prediction = net.predict([dcnn_img], oversample=False)
            fc7 = net.blobs['fc7'].data[0].ravel()
            pool5 = net.blobs['pool5'].data[0].ravel()

            fc7 = fc7_scalar.transform(fc7)
            comp_feat = fc7_compressor.transform(fc7)

            fc7_feats = np.vstack((fc7_feats, fc7))
            comp_fc7 = np.vstack((comp_fc7, comp_feat))
            pool5_feats = np.vstack((pool5_feats, pool5))

            # assign a unique ID and save image
            unique_id = len(props)

            class_labels = np.append(class_labels, type_id)
            props.append(prop)

            # we have to dump each time otherwise images and indicies might get out of sync
            # say, if the program terminated but had already saved out images
            utils.save_image(segmenter.rgb_imge, unique_id, 'images')
            utils.save_image(np.asarray(dcnn_img * 255, dtype=np.uint8), unique_id, 'seg_images')
            utils.dump_feature_db(comp_fc7, props, fc7_feats, pool5_feats)

            time.sleep(1)

            if utils.query_should_continue() == -1:
                break


    except KeyboardInterrupt:
        print "Shutting down"
