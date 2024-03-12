from basic import integrate
import autograd.numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize
import pickle
from autograd import grad, jacobian
from tqdm import tqdm

if __name__ == '__main__':
    np.random.seed(232) #3 #232: BFGS #134: COBYLA
    tau = 1.0
    NEURONS = 2
    theta = np.random.random(NEURONS) * 10 - 5
    W = np.random.random((NEURONS, NEURONS)) * 10 - 5
    # W = np.eye(NEURONS)
    # x0 = np.ones(NEURONS) * np.random.random()
    x0 = np.zeros(NEURONS)

    # phi = lambda x: 1 / (1 + np.exp(-x))
    phi = lambda x: np.tanh(x)
    f = lambda x: 1 / tau * (-x + phi(W@x + theta))
    pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_A.pickle'
    with open(pickle_file_path, 'rb') as file:
        weight_matrix = pickle.load(file)
    pickle_file_path = r'/Users/neiljanwani/Documents/CDS232/src/periodic_pickles/circle_B.pickle'
    with open(pickle_file_path, 'rb') as file:
        bias_matrix = pickle.load(file)

    W = weight_matrix
    theta = bias_matrix
    jacobian_cost = jacobian(f)
    def func(x):
        global jacobian_cost
        return -np.linalg.norm(jacobian_cost(x))
    
    res = scipy.optimize.minimize(
        fun=func,
        x0=x0,
        method='SLSQP'
    )
    x_min = res.x
    print(x_min)
    print(-func(x_min))
    print(x0)
    
    import matplotlib.pyplot as plt

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    dt = 0.01
    X = np.arange(-1, 1, dt)
    Y = np.arange(-1, 1, dt)
    R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    # print(Z)
    Z = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        print(f'{i / len(X) * 100:.2f}%', end='\r')
        for j, y in enumerate(Y):
            v = np.array([x,y])
            Z[i,j] = -func(v)
            # if np.linalg.norm(x_min - v) < 0.5:
            #     print(f'({x}, {y})', Z[i,j])
    X, Y = np.meshgrid(X, Y)
    print(np.max(Z))
    ax.scatter(*x_min[::-1], -func(x_min), color='black', s=40)
    # Plot the surface.
    cmap = cm.coolwarm
    cmap = cmap(np.arange(cmap.N))
    cmap[:,-1] = 0.5
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(cmap)

    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$||DJ(x)||$')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.set_size_inches((10,8))
    fig.tight_layout()
    plt.show()