from basic import integrate
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

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


if __name__ == '__main__':
    np.random.seed(232) #3 #200
    plot2D = True
    DT = 0.02
    t0, tfinal = 0, 20
    tau = 1.0
    T = np.linspace(t0, tfinal, num=int((tfinal - t0) / DT))
    NEURONS = 2
    theta = np.random.random(NEURONS) * 10 - 5
    W = np.random.random((NEURONS, NEURONS)) * 10 - 5
    # W = np.eye(NEURONS)
    x0 = np.ones((T.shape[0], NEURONS)) + np.random.random()

    # phi = lambda x: 1 / (1 + np.exp(-x))
    phi = lambda x: np.tanh(x)
    f = lambda x, s, t: 1 / tau * (-x + phi(np.array([W@v + theta for v in x]))) 

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