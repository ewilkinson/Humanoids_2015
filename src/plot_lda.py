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

labels = utils.load_db_labels()


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, with_labels=True,   title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if with_labels is None:
            plt.scatter(X[i, 0], X[i, 1],
                        color=plt.cm.Set1(ids[i]*1.0 / len(classes)),
                        s = 32)
            continue

        elif not with_labels:
            plt.text(X[i, 0], X[i, 1], str(ids[i]),
                     color=plt.cm.Set1(ids[i]*1.0 / len(classes)),
                     fontdict={'weight': 'bold', 'size': 30})

        elif with_labels:
            plt.text(X[i, 0], X[i, 1], labels[ids[i]],
                 color=plt.cm.Set1(ids[i]*1.0 / len(classes)),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1, 1.1])
    plt.xlim([-0.1, 1.1])
    if title is not None:
        plt.title(title)


n_neighbors = 3

# model = manifold.TSNE(n_components=2, perplexity=5, random_state=0)
# model = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')

# model = manifold.Isomap(n_neighbors, n_components=2)

# model = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                       eigen_solver="arpack")

# embed_X = model.fit_transform(comp_fc7)

model = lda.LDA(n_components=2)
embed_X = model.fit_transform(comp_fc7, ids)




plot_embedding(embed_X, with_labels=None, title= '')
