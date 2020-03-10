import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from load_data import*
from sklearn.neighbors import NearestNeighbors
from element_information import *
from sklearn.decomposition import PCA

path = '../data/'
names, X = load_features(path, 'proposition_of_features.txt', add_information = False, expand = False, degree = 2)
#names, x = select_elements_from_data(names, x, ["Pd"], remove = False)
#X = np.delete(X, np.arange(8,11), axis = 1)
X_transf = PCA(n_components = 2).fit_transform(X)


for n in np.arange(2,11,1):
    nbrs = NearestNeighbors(n_neighbors=n).fit(X_transf)
    distances, indices = nbrs.kneighbors(X_transf)

    decrescent_distance = sorted(distances[:,n-1], reverse=True)
    plt.plot(list(range(1,X.shape[0]+1)), decrescent_distance, label = str(n))
plt.xlabel("number of points")
plt.ylabel("k distance")
plt.title("k distance plot. k = " + str(n))
plt.legend()
plt.show()