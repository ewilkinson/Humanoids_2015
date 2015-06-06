__author__ = 'eric'

import utils
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection)
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import offsetbox


comp_fc7, ids, fc7_feats, pool5_feats = utils.load_feature_db()
classes = np.unique(ids)

# this is only to display the class ID
from sklearn import datasets
digits = datasets.load_digits(n_class=classes.shape[0])


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(ids[i]),
                 color=plt.cm.Set1(ids[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


n_neighbors = 6

# model = manifold.TSNE(n_components=2, perplexity=10, random_state=0)
# model = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')

# model = manifold.Isomap(n_neighbors, n_components=2)

model = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")

embed_X = model.fit_transform(comp_fc7)

# model = lda.LDA(n_components=2)
# embed_X = model.fit_transform(comp_fc7, ids)




plot_embedding(embed_X, 'LDA Embedding')
