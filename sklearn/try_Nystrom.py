import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors._graph import kneighbors_graph
from sklearn.manifold._spectral_embedding import spectral_embedding

N = 100
theta = np.arange(0,2*np.pi,.001)
t = 2*np.pi*(np.arange(1/N, 1, 1/N))
X = np.array([np.cos(t), np.sin(t)])
X = X+np.random.randn(X.shape[0], X.shape[1])/70

plt.subplot(121)
plt.plot(X[0,:], X[1,:])

adjacency = kneighbors_graph(X.transpose(), 50)

adj1 = adjacency.toarray()
adj1 = adj1 + adj1.transpose()
adj1[adj1 > 1] = 1

embedding = spectral_embedding(adj1, n_components=3)

plt.subplot(122)
plt.plot(embedding[:,0], embedding[:,1])
plt.show()

print()