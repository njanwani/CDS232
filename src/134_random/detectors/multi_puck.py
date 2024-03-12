from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt

def gen_data(N, r):
    return np.random.random((N,2)) * r - r / 2

def alter_data(data, eps):
    return data + (np.random.random(data.shape) * eps - eps / 2)

data = gen_data(27, 1)
tree = KDTree(data, metric='minkowski')
fig, axs = plt.subplots(ncols=2)
axs = axs.flatten()
axs[0].scatter(data[:,0], data[:,1], s=2, color='black')
# axs[0].annotat/e([str(idx) for idx in np.arange(data.shape[0])], data)

newdata = alter_data(data, 0.01)
axs[0].scatter(newdata[:,0], newdata[:,1], s=2, color='red')

filter_data = np.zeros(newdata.shape)
for i, pt in enumerate(newdata):
    print(pt)
    dist, ind = tree.query(pt[np.newaxis, :], k=1)
    # filter_data[i] = np.array((tree.data[ind]))
    filter_data[i] = data[ind]
    
newdata = alter_data(data, 0.01)

for pt1, pt2 in zip(data, newdata):
    axs[1].scatter(*pt1, s=2, color='blue')
    axs[1].scatter(*pt2, s=2, color='blue')
    axs[1].plot((pt1[0], pt2[0]), (pt1[1], pt2[1]), color='blue', lw=2)


fig.set_size_inches((10,5))
plt.show()
