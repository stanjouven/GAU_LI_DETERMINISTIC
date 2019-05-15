import networkx as nx
import numpy as np
import sys

import source_estimation as se
import operator

from multiprocessing import Pool

### Compute a batch in parallel
def ptva_li(graph, obs_time, distribution) :

    source = 3
    mu = distribution.mean()
    sigma = distribution.std()

    ### Run the estimation
    s_est, likelihoods, d_mu, cov = se.ml_estimate(graph, obs_time, sigma, mu, paths,
        path_lengths, max_dist=max_dist)

    ranked = sorted(likelihoods.items(), key=operator.itemgetter(1), reverse=True)

    return (s_est, ranked)



# -----------------------------------------------------------------------------
