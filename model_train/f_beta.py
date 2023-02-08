def f_beta(tp, fp, fn, beta=2):
    return (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp)
