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


k_closest = 5
plt.ion()

fc7_compressor = utils.load_compressor(layer='fc7',
                                       dimension=128,
                                       compression='pca')

fc7_scalar = utils.load_scalar(layer='fc7')
pool5_scalar = utils.load_scalar(layer='pool5')


# LOAD MODEL DATABASE
comp_fc7, props, fc7_feats, pool5_feats = utils.load_feature_db()
type_ids = np.zeros(shape=(len(props),), dtype=np.int32)
aspect_ids = np.zeros(shape=(len(props),), dtype=np.int32)
for i in range(len(props)):
    type_ids[i] = props[i]['type_id']
    aspect_ids[i] = props[i]['aspect_id']

class_gmms = utils.load_feature_stats()
labels = utils.load_db_labels()

tree = BallTree(comp_fc7, leaf_size=5)

test_comp_fc7, test_props, test_fc7_feats, test_pool5_feats = utils.load_test_set_db()
test_type_ids = np.zeros(shape=(len(props),), dtype=np.int32)
test_aspect_ids = np.zeros(shape=(len(props),), dtype=np.int32)
for i in range(len(test_props)):
    test_type_ids[i] = test_props[i]['type_id']
    test_aspect_ids[i] = test_props[i]['aspect_id']



# RUN EVALUATION
type_query_times = []
aspect_comp_times = []
type_successes = []
aspect_successes = []
aspect_dists = []

top_2_success = []

for iter in range(len(test_props)):
    query_comp_fc7 = test_comp_fc7[iter, :]
    query_pool5 = test_pool5_feats[iter, :]
    query_type_id = test_type_ids[iter]
    query_aspect_id = test_aspect_ids[iter]

    query_start_time = time.time()
    distances, idxes = tree.query(query_comp_fc7, k=k_closest)

    distances = distances[0]
    idxes = idxes[0]

    R_type_ids = type_ids[idxes]
    R_unique_type_ids = np.unique(R_type_ids)
    R_aspect_ids = aspect_ids[idxes]

    # SCORE EACH TYPE IN R
    valid_types = []
    for i, type_id in enumerate(R_unique_type_ids):
        gmm = class_gmms[type_id]
        score = gmm.score(query_comp_fc7) / comp_fc7.shape[1]
        if score > -3.0:
            valid_types.append(type_id)

    # IF NO VALID TYPES, CHECK IF THAT WAS THE ANSWER
    # UNKNOWN TYPES IN TEST SET ARE ALL NEGATIVE
    if len(valid_types) == 0:
        if query_type_id < 0:
            type_successes.append(1)
        else:
            type_successes.append(0)

        type_query_times.append(time.time() - query_start_time)
        continue  # no need to go on from here

    type_successes.append(query_type_id in valid_types)
    type_mask = np.zeros(R_type_ids.shape, dtype=np.bool)
    for type_id in valid_types:
        type_mask = type_mask | R_type_ids == type_id

    aspect_idxes = idxes[type_mask]
    type_query_times.append(time.time() - query_start_time)


    # find the closest in terms of pool 5 features
    # search over all classes returned and use the pearson r correlation
    aspect_start_time = time.time()
    query_pool5 = query_pool5 / query_pool5.max()

    pearson_rhos = []
    for idx in aspect_idxes:
        pool5 = pool5_feats[idx, :]
        p_rho, pval = pearsonr(pool5, query_pool5)
        pearson_rhos.append(p_rho)

    aspect_sort_idxes = np.argsort(pearson_rhos)[::-1]
    print pearson_rhos, iter, query_aspect_id, aspect_ids[aspect_idxes[aspect_sort_idxes[0]]]

    if pearson_rhos[aspect_sort_idxes[0]] < 0.55:
        aspect_successes.append(0)
        continue

    else:
        ans_aspect_id = aspect_ids[aspect_idxes[aspect_sort_idxes[0]]]
        if query_aspect_id == ans_aspect_id:
            aspect_successes.append(1)
            top_2_success.append(1)
        else:
            aspect_successes.append(0)

            if pearson_rhos[aspect_sort_idxes[1]] < 0.55:
                ans_aspect_id_2 = aspect_ids[aspect_idxes[aspect_sort_idxes[1]]]
                if query_aspect_id == ans_aspect_id_2:
                    top_2_success.append(1)
                else:
                    top_2_success.append(0)
            else:
                top_2_success.append(0)

            print 'Fail!'

        dist = abs(query_aspect_id - ans_aspect_id)
        if dist > 4:
            dist = abs((dist + 3) % 8 - 3)
        aspect_dists.append(dist)

    aspect_comp_times.append(time.time() - aspect_start_time)

aspect_dists = np.asarray(aspect_dists)
type_query_times = np.asarray(type_query_times)
type_successes = np.asarray(type_successes)
aspect_successes = np.asarray(aspect_successes)
aspect_comp_times = np.asarray(aspect_comp_times)
top_2_success = np.asarray(top_2_success)

avg_type_query_time = np.mean(type_query_times)
avg_aspect_comp_time = np.mean(aspect_comp_times)
avg_type_success = np.mean(type_successes)
avg_aspect_success = np.mean(aspect_successes)
avg_aspect_dist = np.mean(aspect_dists)
avg_top_2_success = np.mean(top_2_success)

print 'Avg Type Query Time :', avg_type_query_time
print 'Avg Aspect Comp time : ', avg_aspect_comp_time
print 'Avg Type Success : ', avg_type_success
print 'Avg Aspect Success : ', avg_aspect_success
print 'Avg Aspect Dist : ', avg_aspect_dist
print 'Avg Top 2 Success', avg_top_2_success

#
# utils.dump_test_set_db(test_comp_fc7, test_props, test_fc7_feats, test_pool5_feats)
