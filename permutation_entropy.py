"""Permutation entropy for IMF complexity grouping."""
import itertools, math
import numpy as np

def permutation_entropy(x, embedding_dim=3, delay=1):
    x = np.asarray(x, dtype=float)
    patterns = list(itertools.permutations(range(embedding_dim)))
    index = {p:i for i,p in enumerate(patterns)}
    counts = np.zeros(math.factorial(embedding_dim), dtype=float)
    n_windows = len(x) - (embedding_dim - 1) * delay
    if n_windows <= 0:
        return np.nan
    for i in range(n_windows):
        w = x[i:i + embedding_dim * delay:delay]
        counts[index[tuple(np.argsort(w))]] += 1
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(math.factorial(embedding_dim))

def component_entropy(imfs, embedding_dim=3, delay=1):
    channels, components, _ = imfs.shape
    entropy = np.empty((channels, components))
    for c in range(channels):
        for k in range(components):
            entropy[c,k] = permutation_entropy(imfs[c,k], embedding_dim, delay)
    return np.nanmean(entropy, axis=0)

def entropy_groups(imfs, embedding_dim=3, delay=1):
    mean_entropy = component_entropy(imfs, embedding_dim, delay)
    threshold = np.nanmean(mean_entropy)
    high = np.where(mean_entropy > threshold)[0]
    low = np.where(mean_entropy <= threshold)[0]
    return high, low, mean_entropy, threshold
