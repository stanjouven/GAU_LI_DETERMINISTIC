
"""This file contains some functions needed to estimate (via maximum
likelihood) the source of a SI epidemic process (with Gaussian edge delays).
The important function is
    s_est, likelihood = ml_estimate(graph, obs_time, sigma, is_tree, paths,
    path_lengths, max_dist)
where s_est is the list of nodes having maximum a posteriori likelihood and
likelihood is a dictionary containing the a posteriori likelihood of every
node.
"""
import math
import random
import networkx as nx
import numpy as np
import GAU_LI_DETERMINISTIC.source_est_tools as tl
import operator
import collections

import scipy.stats as st
from scipy.misc import logsumexp

def ml_estimate(graph, obs_time, sigma, mu, paths, path_lengths,
        max_dist=np.inf):
    """Returns estimated source from graph and partial observation of the
    process.
    - graph is a networkx graph
    - obs_time is a dictionary containing the observervations: observer -->
      time
    Output:
    - list of nodes having maximum a posteriori likelihood
    - dictionary: node -> a posteriori likelihood
    """
    ### Gets the sorted observers and the referential observer (closest one)
    sorted_obs = sorted(obs_time.items(), key=operator.itemgetter(1))
    sorted_obs = [x[0] for x in sorted_obs]
    random.shuffle(sorted_obs)
    ref_obs = sorted_obs[0]
    print('ref obs ', ref_obs)
    print('sorted obs ', sorted_obs)
    #ref_obs = random.choice(sorted_obs)

    ### Gets the nodes of the graph and initializes likelihood
    nodes = np.array(list(graph.nodes))
    loglikelihood = {n: -np.inf for n in nodes}

    # candidate nodes does not contain observers nodes by assumption
    candidate_nodes = np.array(list(set(nodes) - set(sorted_obs)))
    for s in nodes:
        print('s ', s)
        if path_lengths[ref_obs][s] < max_dist:
            ### BFS tree
            tree_s = likelihood_tree(paths, s, sorted_obs)
            ### Covariance matrix
            cov_d_s = tl.cov_mat(tree_s, graph, paths, sorted_obs, ref_obs)
            print('cov ', cov_d_s, flush = True)
            print('sigma**2 ', sigma**2, flush = True)
            cov_d_s = (sigma**2)*cov_d_s
            ### Mean vector
            mu_s = tl.mu_vector_s(paths, s, sorted_obs, ref_obs)
            print('mu ', mu_s)
            print('mu dist ', mu)
            mu_s = mu*mu_s
            ### Computes log-probability of the source being the real source
            likelihood, tmp = logLH_source_tree(mu_s, cov_d_s, sorted_obs, obs_time, ref_obs)
            loglikelihood[s] = likelihood


    ### Find the nodes with maximum loglikelihood and return the nodes
    # with maximum a posteriori likelihood

    ### Corrects a bias
    posterior = posterior_from_logLH(loglikelihood)

    scores = sorted(posterior.items(), key=operator.itemgetter(1), reverse=True)
    source_candidate = scores[0][0]

    return source_candidate, scores

#################################################### Helper methods for ml algo
def posterior_from_logLH(loglikelihood):
    """Computes and correct the bias associated with the loglikelihood operation.
    The output is a likelihood.
    Returns a dictionary: node -> posterior probability
    """
    bias = logsumexp(list(loglikelihood.values()))
    print('loglikelihood ', loglikelihood, flush = True)
    return dict((key, np.exp(value - bias))
            for key, value in loglikelihood.items())


def logLH_source_tree(mu_s, cov_d, obs, obs_time, ref_obs):
    """ Returns loglikelihood of node 's' being the source.
    For that, the probability of the observed time is computed in a tree where
    the current candidate is the source/root of the tree.
    - mu_s is the mean vector of Gaussian delays when s is the source
    - cov_d the covariance matrix for the tree
    - obs_time is a dictionary containing the observervations: observer --> time
    - obs is the ordered list of observers, i.e. obs[0] is the reference
    """
    assert len(obs) > 1

    ### Creates the vector for the infection times with respect to the referential observer
    obs_d = np.zeros((len(obs)-1, 1))

    ### Loops over all the observers (w/o first one (referential) and last one (computation constraint))
    #   Every time it computes the infection time with respect to the ref obs
    for l in range(1, len(obs)):
        obs_d[l-1] = obs_time[obs[l]] - obs_time[ref_obs]

    ### Computes the log of the gaussian probability of the observed time being possible
    exponent =  - (1/2 * (obs_d - mu_s).T.dot(np.linalg.inv(cov_d)).dot(obs_d -
            mu_s))
    print('det ', np.linalg.det(cov_d), flush = True)
    denom = math.sqrt(((2*math.pi)**(len(obs_d)-1))*np.linalg.det(cov_d))
    print('obs_d - mu_s ', obs_d - mu_s)
    print('denom ', denom)
    print('exponent ', exponent)
    return (exponent - np.log(denom))[0,0], obs_d - mu_s


def likelihood_tree(paths, s, obs):
    """Creates a BFS tree with only observers at its leaves.
    Returns a BFS tree
    """
    tree = nx.Graph()
    for o in obs:
        p = paths[o][s]
        tree.add_edges_from(zip(p[0:-1], p[1:]))
    return tree
