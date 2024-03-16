from system import *
import matplotlib.pyplot as plt
import time
import pickle
from scipy.integrate import solve_ivp
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from autograd import jacobian
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
np.random.seed(232)

import pickle

pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/stable_A_alpha10.pickle' #r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_A.pickle'
with open(pickle_file_path, 'rb') as file:
    weight_matrix = pickle.load(file)
pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/stable_b_alpha10.pickle' #r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_B.pickle'
with open(pickle_file_path, 'rb') as file:
    bias_matrix = pickle.load(file)
    

W = weight_matrix
theta = bias_matrix
plot2D = True
tau = 1.0
alpha = 10
phi = lambda x: alpha * np.tanh(x)
f = lambda x: 1 / tau * (-x + phi(W@x + theta))
jacobian_ = jacobian(f)
A = jacobian_(np.zeros(2))
Q = np.eye(2)
P = solve_continuous_lyapunov(A.T, -Q)
np.allclose(A.T@P + P@A + Q, np.zeros(2))

def integrate(func, a, b, axis=0, dt=0.001):
    if a == b:
        return np.zeros(x.shape[1])
    t = np.linspace(a, b, num=int((b - a) / dt))
    # print(func(t[0]))
    fs = [func(t_) for t_ in t]
    # print(dt)
    return np.trapz(fs, dx=dt, axis=0)

def inside(v):
    func = lambda tau: jacobian_(tau * v)
    integral = integrate(func, 0, 1)
    return np.linalg.norm(integral - jacobian_(np.zeros(2)))

# x = np.zeros(2)
# v = np.zeros(2)
# func = lambda tau: jacobian_(tau * np.zeros(2))
# integral = integrate(func, 0, 2 * np.pi)
# print(func(0))
# inside(np.zeros(2))
for r in np.linspace(0, 100, num=10):
    thetas = np.linspace(0,2*np.pi)
    val = inside(np.zeros(2))
    # vecs = np.array([r * np.cos(theta), r * np.sin(theta)]).T
    
    # # print(vecs)
    # maximum = -np.inf
    # for idx in range(vecs.shape[0]):
    #     # print(idx)
    #     print(vecs[idx,:])
    #     x = np.ones(2)
    #     v = np.ones(2)
    #     val = inside(np.zeros(2))
    #     if val > maximum:
    #         maximum = val
    
    # print(maximum)