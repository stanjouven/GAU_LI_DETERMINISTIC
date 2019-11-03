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
import networkx as nx
import numpy as np
import PTVA_LI.source_est_tools as tl
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
    o1 = min(obs_time, key=obs_time.get)

    ### Gets the nodes of the graph and initializes likelihood
    nodes = np.array(list(graph.nodes))
    loglikelihood = {n: -np.inf for n in nodes}

    ### Print variables to be given to output to communicate intermediate results
    d_mu = collections.defaultdict(list)
    covariance = collections.defaultdict(list)

    ### Computes classes of nodes with same position with respect to all observers
    classes = tl.classes(path_lengths, sorted_obs)
    print('OBS ', len(sorted_obs))

    ### Iteration over all nodes per class
    #   nodes from same class will be attributed the average of their likelihoods
    #   likelihood
    for c in classes:

        tmp_lkl = [] # Used to compute mean of likelihoods of same class
        for s in c:
            if path_lengths[o1][s] < max_dist:
                ### BFS tree
                tree_s = likelihood_tree(paths, s, sorted_obs)
                ### Covariance matrix
                cov_d_s = tl.cov_mat(tree_s, graph, paths, sorted_obs)
                cov_d_s = (sigma**2)*cov_d_s
                ### Mean vector
                mu_s = tl.mu_vector_s(paths, s, sorted_obs)
                print('SHAPE ', np.array(mu_s).shape)
                mu_s = mu*mu_s
                ### Computes log-probability of the source being the real source
                likelihood, tmp = logLH_source_tree(mu_s, cov_d_s, sorted_obs, obs_time)
                tmp_lkl.append(likelihood)

                ## Save print values
                d_mu[s] = tmp
                covariance[s] = cov_d_s
        ### If the class was not empty
        if len(tmp_lkl)>0:
            for s in c:
                loglikelihood[s] = np.mean(tmp_lkl)

    ### Find the nodes with maximum loglikelihood and return the nodes
    # with maximum a posteriori likelihood

    ### Corrects a bias
    posterior = posterior_from_logLH(loglikelihood)

    max_lkl = max(posterior.values())
    source_candidates = list()
    ### Finds nodes with maximum likelihood
    for src, value in posterior.items():
        if np.isclose(value, max_lkl, atol= 1e-08):
            source_candidates.append(src)

    return source_candidates, posterior, d_mu, covariance

#################################################### Helper methods for ml algo
def posterior_from_logLH(loglikelihood):
    """Computes and correct the bias associated with the loglikelihood operation.
    The output is a likelihood.

    Returns a dictionary: node -> posterior probability

    """
    bias = logsumexp(list(loglikelihood.values()))
    return dict((key, np.exp(value - bias))
            for key, value in loglikelihood.items())


def logLH_source_tree(mu_s, cov_d, obs, obs_time):
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
        obs_d[l-1] = obs_time[obs[l]] - obs_time[obs[0]]

    ### Computes the log of the gaussian probability of the observed time being possible
    exponent =  - (1/2 * (obs_d - mu_s).T.dot(np.linalg.inv(cov_d)).dot(obs_d -
            mu_s))
    denom = math.sqrt(((2*math.pi)**(len(obs_d)-1))*np.linalg.det(cov_d))

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
