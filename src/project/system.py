import pickle
import numpy as np

pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/asym_periodic_A_alpha10.pickle' #r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_A.pickle'
with open(pickle_file_path, 'rb') as file:
    weight_matrix = pickle.load(file)
pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/asym_periodic_b_alpha10.pickle' #r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_B.pickle'
with open(pickle_file_path, 'rb') as file:
    bias_matrix = pickle.load(file)
    

W = weight_matrix
theta = bias_matrix
plot2D = True
tau = 1.0

phi = lambda x: np.tanh(x)
f = lambda x, s, t: 1 / tau * (-x + phi(np.array([W@v + theta for v in x]))) 