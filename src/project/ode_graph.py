import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib

font = {'family' : 'Times',
        'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size
    )
    
np.random.seed(7) #232
fig, ax = plt.subplots(nrows = 1, ncols = 1)
font = {'fontname' : 'Times New Roman'}

plot2D = True
DT = 0.01
t0, tfinal = 0, 1
tau = 1
T = np.linspace(t0, tfinal, num=int((tfinal - t0) / DT))
NEURONS = 2
theta = np.random.random(NEURONS)
W = np.random.random((NEURONS, NEURONS)) * 10 - 5

periodics = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/asym_periodic_A_alpha10.pickle', r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/asym_periodic_b_alpha10.pickle'
stables = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/stable_A_alpha10.pickle', r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/stable_b_alpha10.pickle'
unstables = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/unstable_A_alpha10.pickle', r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/unstable_b_alpha10.pickle'
waves = r'/Users/neiljanwani/Documents/CDS232/src/project/learning/wave_A_alpha10.pickle', r'/Users/neiljanwani/Documents/CDS232/src/project/learning/wave_b_alpha10.pickle'
trains = lambda num_epochs: (rf'/Users/neiljanwani/Documents/CDS232/src/project/learning/stable_A_alpha10_{num_epochs}.pickle', rf'/Users/neiljanwani/Documents/CDS232/src/project/learning/stable_b_alpha10_{num_epochs}.pickle')
num_epochs = 35
dude = trains(num_epochs)
pickle_file_path = dude[0]
with open(pickle_file_path, 'rb') as file:
    weight_matrix = pickle.load(file)
pickle_file_path = dude[1]
with open(pickle_file_path, 'rb') as file:
    bias_matrix = pickle.load(file)

W = weight_matrix
theta = bias_matrix

# W = np.array([[0, -1], [1, 0]])
# theta = np.zeros((1, 2))
# W = np.eye(NEURONS)
mu = -1
# phi = lambda x: 1 / (1 + np.exp(-x))
alpha = 10
phi = lambda x: alpha *np.tanh(x)
f = lambda x: 1 / tau * (-x + phi(W@x + theta)) 

# def f(x):
#     theta = np.sin(x[0]) #np.arctan2(x[1], x[0]) + np.pi
#     return 5 * np.array([np.cos(theta), np.sin(theta)])
# f2 = lambda x: 1 / tau * (-x + phi(W1@x + theta)) 
# f = lambda x: phi(W@x + theta)

steps = 2000
x = np.zeros((steps,2))
X0 = []
NUM = 8
BOUND = 4
for a in np.linspace(-BOUND, BOUND, num=NUM):
    for b in np.linspace(-BOUND, BOUND, num=NUM):
        X0.append((a,b))
        
for a in np.linspace(-0.8, 0.8, num=3):
    for b in np.linspace(-0.8, 0.8, num=3):
        X0.append((a,b))
dt = 0.01

for x0 in X0:
    x[0] = np.array(x0)
    for i in range(1, steps):
        x[i] = f(x[i - 1]) * dt + x[i - 1]
    l, = ax.plot(x[:, 0], x[:, 1], color = 'red', lw=0.5)
    # add_arrow(l, position=x[20][0])
    # add_arrow(l, position=x[60][0])
    add_arrow(l, position=x[10 * steps // 100][0], size=5)
    # break
    
t = np.linspace(0, 2 * np.pi)
# ax.plot(np.cos(t), np.sin(t), color = 'black', lw=0.5)
    
# ax.set_xlim((-10, 10))
# ax.set_ylim((-10, 10))
# ax.set_aspect('equal')
fig.set_size_inches((6,6))
fig.suptitle(f'Epoch {num_epochs}')
fig.tight_layout()
plt.savefig(f'/Users/neiljanwani/Documents/CDS232/src/project/graphs/train_{num_epochs}.png', dpi=500)