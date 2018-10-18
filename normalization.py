import numpy as np
import pdb

def normalize_probs(probs):

    """takes an array of arrays of probabilities and normalizes each entry to sum to one"""
    # TODO: this only works if not 1D array but 1D case is easy to write 1 line so maybe don't implement
    probs_sums = np.array([p.sum() for p in probs])
    probs = np.array([p / p_sum for p, p_sum in zip(probs, probs_sums)])
    return probs
