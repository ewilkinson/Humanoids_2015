#!/usr/bin/env python

import numpy as np
import time, sys

from generic_seg import GenericSegmenter
import cv2
from PIL import Image

import utils

def trans_img_dcnn(img, box):
    """
    Center the crop in a image size set for the dcnn (255, 255). Convert to float32 and scale 0,1

    :param img: segmented image with 1 object
    :param box: The segment box

    :return: dcnn_img
    """
    x1, y1, x2, y2, mean_depth = box

    mean_x = x2 -x1 / 2.0
    mean_y = y2 - y1 / 2.0

    # its okay to have negatives since these  should wrap around to 0's
    x_min = mean_x - 200
    x_max = x_min + 400
    y_min = mean_y - 200
    y_max = y_min + 400

    resized_img = cv2.resize(img[x_min:x_max, y_min:y_max, :], (256, 256))

    return np.asarray(resized_img, dtype=np.float32) / 255.0




def query_user():
    var = raw_input("Accept Photo (y/n) ? : ")
    if var.upper() == 'N' or var.upper() == 'NO':
        return -1

    var = raw_input("Enter class ID (uint) : ")
    id = int(var)

    return id

def crop_segment(segmenter):
    """
    Crops the image from the segmenter so that only the closest object is visible

    :param segmenter:
    :return:
    """
    img = segmenter.rgb_imge
    boxes = segmenter.boxes

    closest_idx = 0
    closest_val = np.inf
    for i in range(len(boxes)):
        x1, y1, x2, y2, mean_depth = boxes[i]
        if mean_depth < closest_val:
            closest_idx = i
            closest_val = mean_depth

    x1, y1, x2, y2, mean_depth = boxes[closest_idx]

    x1 = max(0, x1)
    y1 = max(0,y1)
    x2 = min(x2, img.shape[0]-1)
    y2 = min(y2, img.shape[1]-1)

    cropped_img = np.zeros(img.shape, dtype=img.dtype)
    cropped_img[x1:x2, y1:y2, :] = img[x1:x2, y1:y2, :]

    return  cropped_img, boxes[closest_idx]

if __name__ == '__main__':
    net, params, blobs = utils.load_network()

    fc7_compressor = utils.load_compressor(layer='fc7',
                                               dimension=256,
                                               compression='lda')

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

    comp_fc7, ids, fc7_feats, pool5_feats = utils.load_feature_db()

    try:
        segmenter.listen()

        while True:
            time.sleep(1)

            # segmenter can be a bit slow to start up
            if segmenter.rgb_imge is None or segmenter.boxes is None:
                continue

            cropped_img, box = crop_segment(segmenter)

            cv2.imshow("Query Window", cropped_img)
            cv2.waitKey(1)


            class_id = query_user()

            if class_id == -1:
                continue

            print 'Class ID : ', class_id

            dcnn_img = trans_img_dcnn(cropped_img, box)

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
            unique_id = ids.shape[0]

            # append the id list with the latest class id
            ids = np.append(ids, class_id)

            # we have to dump each time otherwise images and indicies might get out of sync
            # say, if the program terminated but had already saved out images
            utils.save_image(segmenter.rgb_imge, unique_id, 'images')
            utils.save_image(dcnn_img*255, unique_id, 'seg_images')
            utils.dump_feature_db(comp_fc7, ids, fc7_feats, pool5_feats)

            time.sleep(1)

            should_continue = raw_input("Continue (y/n) ? : ")
            if should_continue.lower() == 'n' or should_continue.lower() == 'no':
                break


    except KeyboardInterrupt:
        print "Shutting down"