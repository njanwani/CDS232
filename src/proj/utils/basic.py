import numpy as np

def integrate(func, x, a, b, axis=0, dt=0.001):
    if a == b:
        return np.zeros(x.shape[1])
    t = np.linspace(a, b, num=x.shape[0])
    fs = func(x, t)
    return np.trapz(fs, dx=dt, axis=0)
