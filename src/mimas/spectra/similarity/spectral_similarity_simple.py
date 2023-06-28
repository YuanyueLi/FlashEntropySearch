import numpy as np
import scipy.stats

def _weight_intensity_for_entropy(x):
    if sum(x) > 0:
        entropy_x = scipy.stats.entropy(x)
        if entropy_x >= 3:
            return x
        else:
            WEIGHT_START = 0.25
            WEIGHT_SLOPE = 0.25

            weight = WEIGHT_START + WEIGHT_SLOPE * entropy_x
            x = np.power(x, weight)
            x = x / sum(x)
            return x


def entropy_distance(p, q):
    p = _weight_intensity_for_entropy(p)
    q = _weight_intensity_for_entropy(q)

    merged = p + q
    entropy_increase = 2 * scipy.stats.entropy(merged) - scipy.stats.entropy(p) - scipy.stats.entropy(q)
    return entropy_increase
