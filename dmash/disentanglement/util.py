import numpy as np

from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def get_mutual_information(x, y, normalize=True, continuous_factors=True, kraskov=False):
    ''' Compute mutual information between two random variables

    :param x:      random variable
    :param y:      random variable
    '''
    if kraskov:
        if continuous_factors:
            return mutual_info_regression(y[:, None], x, random_state=0).item()
        else:
            return mutual_info_classif(y[:, None], x, random_state=0).item()
    else:
        if normalize:
            return normalized_mutual_info_score(x, y)
        else:
            return mutual_info_score(x, y)


def get_bin_index(x, nb_bins):
    ''' Discretize input variable

    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)
