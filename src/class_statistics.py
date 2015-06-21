import numpy as np
import utils
from sklearn import mixture
import hickle as hkl

comp_fc7, class_labels, fc7_feats, pool5_feats = utils.load_feature_db()

unique_classes = np.unique(class_labels)
gmms = {}
for k in unique_classes:
    class_idxes = np.where(class_labels == k)[0]

    X = comp_fc7[class_idxes, :]
    # X = X + np.random.randn(X.shape[0], X.shape[1])*1e-4
    gmm = mixture.GMM(n_components=1, covariance_type='diag')
    gmm.fit(X)
    gmms[k] = gmm

# utils.dump_feature_stats(gmms)



from divergence import gau_bh, gau_js, gau_kl, my_kl

num_classes = len(unique_classes)
divergence_matrix = np.zeros(shape=(num_classes, num_classes), dtype=np.float64)

for i in range(num_classes):
    pm = np.squeeze(gmms[i].means_)
    pv = np.matrix(np.squeeze(gmms[i]._get_covars()))
    for j in range(num_classes):
        qm = np.squeeze(gmms[j].means_)
        qv = np.matrix(np.squeeze(gmms[j]._get_covars()))
        divergence_matrix[i, j] = np.sqrt(gau_js(pm, pv, qm, qv))

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 22})
plt.figure()
plt.imshow(divergence_matrix, interpolation='none', cmap=plt.cm.get_cmap('Blues'))
plt.xlabel('Type Gaussian Model')
plt.ylabel('Type Gaussian Model')
plt.colorbar()

