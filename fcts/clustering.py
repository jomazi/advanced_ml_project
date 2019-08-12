import numpy as np
from sklearn import metrics


# function to measure quality of clustering
def bench_k_means(labels_, estimator):
    # metrics
    metrs = [metrics.homogeneity_score, metrics.completeness_score, metrics.v_measure_score,
             metrics.adjusted_rand_score, metrics.adjusted_mutual_info_score]
    # scores
    scores = [m(labels_, estimator.labels_) for m in metrs]

    return np.mean(scores)