import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pickle

DT = 0.01
t0, tfinal = 0, 6
# theta = np.zeros(3)
tau = 1
T = np.linspace(t0, tfinal, num=int((tfinal - t0) / DT))
NEURONS = 3
# W = np.array([[]])
# W = np.random.random((NEURONS, NEURONS)) * 10 - 5
# W = W.astype(int)
x0 = [np.array([[-0.4988,  0.7565, -1.2760]]).repeat(T.shape[0], axis=0), #6th
      np.array([[2.3049, -1.0322,  1.8185]]).repeat(T.shape[0], axis=0), #3rd
      np.zeros((T.shape[0], NEURONS)) + 0.01
     ][0]

phi = lambda x: 1 / (1 + np.exp(-x))
f = lambda x, s, t: 1 / tau * (-x + phi(np.array([W@v + theta for v in x]))) #* np.expand_dims(np.exp(-(t-s)), axis=0).T

def integrate(func, x, a, b, axis=0, dt=0.001):
    if a == b:
        return np.zeros(x.shape[1])
    t = np.linspace(a, b, num=x.shape[0])
    fs = func(x, t)
    return np.trapz(fs, dx=dt, axis=0)


def picard_iter(x, f, T, x0, t0=0):
    x1 = x0 + np.array([integrate(lambda x, s: f(x, s, tf), x[:i], t0, tf, dt=DT) for i, tf in enumerate(T)]) #integrate(lambda t: f(x0, t), t0, T)
    # x1 = x0 + integrate2(f, x, T)
    return x1


if __name__ == '__main__':
    pickle_file_path = r'iris_weights_ultrasimple_weights.pickle'
    with open(pickle_file_path, 'rb') as file:
        weight_matrix = pickle.load(file)
    pickle_file_path = r'iris_weights_ultrasimple_bias.pickle'
    with open(pickle_file_path, 'rb') as file:
        bias_matrix = pickle.load(file)

    W = weight_matrix
    theta = bias_matrix
    # W = np.eye(3)
    # theta = np.zeros(3)

    xlast = x0
    xcurr = -np.inf
    xsols = []
    thresh = 10**(-9)
    i = 0
    picard_dists = []
    while True:
        xcurr = picard_iter(xlast, f, T, x0)
        xsols.append(xcurr.copy())
        picard_dists.append(np.linalg.norm(xlast - xcurr) )
        print(picard_dists[-1])
        if picard_dists[-1] < thresh:
            break
        xlast = xcurr.copy()
        i += 1
        
        
    PER_ROW = NEURONS
    fig, axs = plt.subplots(nrows=np.ceil(NEURONS / PER_ROW).astype(int), ncols=PER_ROW)
    axs = axs.flatten()
    for n in range(NEURONS):
        for i, x in enumerate(xsols):
            axs[n].plot(T, x[:,n], color='red', lw=(i / (len(xsols))))
            axs[n].set_xlabel('Time')
            axs[n].set_ylabel('Activation')
            axs[n].set_title(f'Neuron {n+1}')
            axs[n].set_ylim((-2, 2))
    fig.set_size_inches((10, 5))
    fig.tight_layout()
    plt.show()