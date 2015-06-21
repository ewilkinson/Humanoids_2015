#!/usr/bin/env python

import numpy as np
import time, sys
import matplotlib.pyplot as plt
import matplotlib

from generic_seg import GenericSegmenter
import cv2
from sklearn.neighbors import BallTree

from scipy.stats import pearsonr, spearmanr

import utils

if __name__ == '__main__':
    plt.ion()
    k_closest = 5  # retrieve the k closest points
    net, params, blobs = utils.load_network()

    fc7_compressor = utils.load_compressor(layer='fc7',
                                           dimension=128,
                                           compression='pca')

    fc7_scalar = utils.load_scalar(layer='fc7')
    pool5_scalar = utils.load_scalar(layer='pool5')

    cv2.startWindowThread()
    cv2.namedWindow("Query Window", 1)

    segmenter = GenericSegmenter(cluster_type="dbscan",
                                 use_gray=False,
                                 depth_max_threshold=4000,
                                 show_segmentation=False,
                                 show_cluster=False,
                                 show_mask=False,
                                 merge_boxes=True)

    # LOAD DATABASE
    comp_fc7, props, fc7_feats, pool5_feats = utils.load_feature_db()
    class_labels = np.zeros(shape=(len(props),), dtype=np.int32)
    aspect_labels = np.zeros(shape=(len(props),), dtype=np.int32)
    for i in range(len(props)):
        class_labels[i] = props[i]['type_id']
        aspect_labels[i] = props[i]['aspect_id']

    class_gmms = utils.load_feature_stats()
    labels = utils.load_db_labels()

    tree = BallTree(comp_fc7, leaf_size=5)

    try:
        segmenter.listen()

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

            dcnn_img = utils.trans_img_dcnn(cropped_img, box)

            cv2.imshow("Query Window", dcnn_img)
            cv2.waitKey(1)

            dcnn_img = (dcnn_img - dcnn_img.mean() ) / dcnn_img.std()
            dcnn_img = dcnn_img + abs(dcnn_img.min())
            dcnn_img = dcnn_img / dcnn_img.max()
            print dcnn_img.max()

            f = plt.figure()
            plt.imshow(dcnn_img)
            plt.axis('off')

            prediction = net.predict([dcnn_img], oversample=False)
            fc7 = net.blobs['fc7'].data[0].ravel()
            pool5 = net.blobs['pool5'].data[0].ravel()

            query_start_time = time.time()
            fc7 = fc7_scalar.transform(fc7)
            comp_feat = fc7_compressor.transform(fc7)
            distances, idxes = tree.query(comp_feat, k=k_closest)

            distances = distances[0]
            idxes = idxes[0]

            print 'Query Time :', time.time() - query_start_time

            print 'Closest Type Vectors:'
            for dist, idx in zip(distances, idxes):
                print labels[class_labels[idx]], dist

            print 'Probability For Each Class'
            unique_classes = np.unique(class_labels)
            scores = {}
            for k in unique_classes:
                gmm = class_gmms[k]
                scores[k] = np.exp(gmm.score(comp_feat) / comp_fc7.shape[1])
                print labels[k], scores[k]

            print 'Closest FC7 Inst : ', idxes[0]

            # plot the closest
            type_query_scores = []
            f = plt.figure()
            for i, idx in enumerate(idxes):
                f.add_subplot(1, k_closest, i + 1)
                plt.imshow(utils.load_image(idx, 'seg_images'))
                plt.text(0, 30, 'Prob: %.3f' % scores[class_labels[idx]],
                         fontdict={'weight': 'bold', 'size': 30},
                         color='b')
                type_query_scores.append(scores[class_labels[idx]])
                plt.axis('off')

            for i in range(len(type_query_scores) - 1, -1, -1):
                score = type_query_scores[i]
                if score[0] < 0.05:
                    idxes = np.delete(idxes, i)

            # find the closest in terms of pool 5 features
            # search over all classes returned and use the pearson r correlation
            pool5 = pool5 / pool5.max()

            query_start_time = time.time()
            pearson_rhos = []
            for i, idx in enumerate(idxes):
                class_p5_feats = pool5_feats[idx, :]

                p_rho, pval = pearsonr(class_p5_feats, pool5)
                pearson_rhos.append(p_rho)

            sort_idxs = np.argsort(pearson_rhos)[::-1]

            print 'Aspect Query Time :', time.time() - query_start_time

            f = plt.figure()
            for i in range(k_closest):
                if i >= len(sort_idxs):
                    f.add_subplot(1, k_closest, i + 1)
                    seg_img = np.ones(shape=(256, 256), dtype=np.uint8) * 255
                    plt.imshow(seg_img, cmap='gray', vmin=0, vmax=255)
                    plt.axis('off')
                else:
                    idx = sort_idxs[i]
                    f.add_subplot(1, k_closest, i + 1)
                    seg_img = utils.load_image(idxes[idx], 'seg_images')
                    plt.imshow(seg_img)
                    print 'Rho: %.3f' % pearson_rhos[idx]
                    plt.text(0, 30, 'Rho: %.3f' % pearson_rhos[idx],
                             fontdict={'weight': 'bold', 'size': 30},
                             color='b')
                    plt.axis('off')

            # PRINT THE CLASS LIKELIHOOD HISTOGRAM
            class_probs = []
            for k, v in scores.items():
                class_probs.append(v[0])

            x = range(len(scores))
            f = plt.figure()
            plt.bar(x, class_probs)
            matplotlib.rcParams.update({'font.size': 22})
            plt.axhline(0.05, color='red', linewidth=2)
            plt.ylim([0, 0.06])

            time.sleep(1)

            break

    except KeyboardInterrupt:
        print "Shutting down"

