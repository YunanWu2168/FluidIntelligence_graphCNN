def stand_norm(x):
    mu = np.mean(x)
    sig = np.std(x)
    return (x-mu)/sig

def minmax_norm(x, low=0, high=1):
    min_x = np.min(x)
    max_x = np.max(x)
    return (high-low)*(x-min_x)/(max_x-min_x) + low