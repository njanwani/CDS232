from system import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def integrate(func, x, a, b, axis=0, dt=0.001):
    if a == b:
        return np.zeros(x.shape[1])
    t = np.linspace(a, b, num=x.shape[0])
    fs = func(x, t)
    return np.trapz(fs, dx=dt, axis=0)


def picard_iter(x, f, T, x0, t0=0, dt=0.01):
    x1 = x0 + np.array([integrate(lambda x, s: f(x, s, tf), x[:i], T[0], tf, dt=dt) for i, tf in enumerate(T)]) #integrate(lambda t: f(x0, t), t0, T)
    return x1

def picard_solve(f, T, x0, verbose=False, thresh=10**(-9), dt=0.01):
    xlast = x0
    xcurr = -np.inf
    xsols = []
    i = 0
    picard_dists = []
    while True:
        xcurr = picard_iter(xlast, f, T, x0, dt)
        xsols.append(xcurr.copy())
        picard_dists.append(np.linalg.norm(xlast - xcurr) )
        if verbose: print(picard_dists[-1])
        if picard_dists[-1] < thresh:
            break
        xlast = xcurr.copy()
        i += 1
        
    return xsols

np.random.seed(232)
DT = 0.02
t0, tfinal = 0, 20
T = np.linspace(t0, tfinal, num=int((tfinal - t0) / DT))
NEURONS = W.shape[0]
x0 = np.ones((T.shape[0], NEURONS)) + np.random.random()

xsols = picard_solve(f, T, x0, dt=DT, verbose=True)
    
PER_ROW = NEURONS
fig, axs = plt.subplots(nrows=np.ceil(NEURONS / PER_ROW).astype(int), ncols=PER_ROW)
axs = axs.flatten()
for n in range(NEURONS):
    for i, x in enumerate(xsols):
        style = None
        lw = (i+1)/(len(xsols) + 1)
        if lw < 1:
            style = 'dashed'
        axs[n].plot(T, x[:,n], color='red', lw=lw, linestyle=style)
        axs[n].set_xlabel('Time')
        axs[n].set_ylabel('Activation')
        axs[n].set_title(f'Neuron {n+1}')
        axs[n].set_ylim((-2, 2))
fig.set_size_inches((10, 5))
fig.tight_layout()
plt.show()

if NEURONS == 2 and plot2D:
    fig, ax = plt.subplots()
    ax.plot(xsols[-1][:, 0], xsols[-1][:, 1])
    fig.set_size_inches((5, 5))
    fig.tight_layout()
    plt.show()