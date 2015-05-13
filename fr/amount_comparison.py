
import numpy as np
from numpy.random import dirichlet
import pandas as pd
import matplotlib.pyplot as plt
from stats_utils import *


def amount_comparison(a_value_counts, b_value_counts, conf = 95):
    alpha = np.ones(a_value_counts.shape)
    a_dist = dirichlet(a_value_counts + alpha, 10000).dot(np.array(a_value_counts.index).transpose())
    b_dist = dirichlet(b_value_counts + alpha, 10000).dot(np.array(b_value_counts.index).transpose())
    dists = pd.DataFrame.from_dict({'A':a_dist, 'B': b_dist})
    return print_rate_stats(dists, conf, True)


def print_rate_stats(dists, conf, plot):

    """
    Helper function to create a pandas datframe with rate statistics
    """

    if plot:
        plot_rate_dist(dists)
    result_df = pd.DataFrame()

    def f(d):
        rci = bayesian_ci(d, conf)
        return "(%0.6f, %0.6f)" % (rci[0], rci[1])

    result_df['CI'] = dists.apply(f)

    def f(d):
        return d.idxmax()
    best = dists.apply(f, axis=1)
    result_df['P(Winner)'] = best.value_counts() / best.shape[0]
    result_df = result_df.sort('P(Winner)', ascending=False)

    def f(d):
        ref_d = dists[result_df.index[0]]
        lift_ci = bayesian_ci(100.0 * ((ref_d - d) / d), conf)
        return "(%0.2f%%, %0.2f%%)" % (lift_ci[0], lift_ci[1])

    result_df['Winners Lift'] = dists.apply(f)

    return result_df[['P(Winner)', 'Winners Lift', 'CI']]
    


def plot_rate_dist(dists):
    """
    Helper function to plot the probability distribution over
    the donation rates (bayesian formalism)
    """
    fig, ax = plt.subplots(1, 1, figsize=(13, 3))

    bins = 50
    for name in dists.columns:
        ax.hist(dists[name], bins=bins, alpha=0.6, label=name)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))