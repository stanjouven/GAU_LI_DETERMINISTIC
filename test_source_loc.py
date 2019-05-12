import networkx as nx
import numpy as np
import sys

import diffusion as diff
import source_estimation as se
import source_est_tools as tl
import operator

from multiprocessing import Pool

### Compute a batch in parallel
def ptva_li(graph, obs_time, sigma, mu) :

    est_ranked = [[]]

    source = 3
    ### Preprocess the graph (computes edge weights, shortest paths & lengths, checks if graph is a tree)
    graph, is_tree, paths, path_lengths = se.preprocess(list(obs_time.keys()), sigma, mu, graph)

    ### Run the estimation
    s_est, likelihoods, d_mu, cov = se.ml_estimate(graph, obs_time, sigma, mu, paths,
        path_lengths, max_dist=max_dist)

    ranked = sorted(likelihoods.items(), key=operator.itemgetter(1), reverse=True)

    result[0].append(s_est)
    result[1].append(ranked)

    return est_ranked



# -----------------------------------------------------------------------------
