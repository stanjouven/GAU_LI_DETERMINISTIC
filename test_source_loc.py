import networkx as nx
import numpy as np
import sys

import PTVA_LI.source_estimation as se
import operator

from multiprocessing import Pool
from termcolor import colored

### Compute a batch in parallel
def ptva_li(graph, obs_time, distribution) :

    mu = distribution.mean()
    sigma = distribution.std()
    obs = np.array(list(obs_time.keys()))

    path_lengths = {}
    paths = {}
    for o in obs:
        path_lengths[o], paths[o] = nx.single_source_dijkstra(graph, o)
        print('path_lengths', o, ' = ', len(path_lengths[o]))
        print('nodes', len(list(graph.nodes())))
        print('path_lengths tab', sorted(path_lengths[o].items(), key=operator.itemgetter(0), reverse=True))
        #print('mean', np.min(path_lengths[o]))
        print('OBS', len(obs))
    ### Run the estimation
    s_est, likelihoods, d_mu, cov = se.ml_estimate(graph, obs_time, sigma, mu, paths,
        path_lengths)

    ranked = sorted(likelihoods.items(), key=operator.itemgetter(1), reverse=True)

    return (s_est, ranked)



# -----------------------------------------------------------------------------
