import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from utils.picard import *

plot2D = True
DT = 0.01
t0, tfinal = 0, 6
tau = 1
T = np.linspace(t0, tfinal, num=int((tfinal - t0) / DT))
NEURONS = 2
theta = np.random.random(NEURONS)
W = np.random.random((NEURONS, NEURONS)) * 10 - 5
# W = np.eye(NEURONS)
x0 = np.zeros((T.shape[0], NEURONS)) + 0.01

# phi = lambda x: 1 / (1 + np.exp(-x))
phi = lambda x: np.tanh(x)
f = lambda x, s, t: 1 / tau * (-x + phi(np.array([W@v + 0*theta for v in x]))) 

if __name__ == '__main__':
    pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_A.pickle'
    with open(pickle_file_path, 'rb') as file:
        weight_matrix = pickle.load(file)
    pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_B.pickle'
    with open(pickle_file_path, 'rb') as file:
        bias_matrix = pickle.load(file)

    W = weight_matrix
    theta = bias_matrix

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
        
    
    