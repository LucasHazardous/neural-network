def mean_and_scale_normalization(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)
