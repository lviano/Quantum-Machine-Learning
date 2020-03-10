import numpy as np
import matplotlib.pyplot as plt
from element_information import*
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from validation import *
from load_data import*
from sklearn.preprocessing import scale
from sklearn import metrics
from model_validation import silhouette_score

path = '../data/'


names, x = load_features(path, 'proposition_of_features.txt', add_information = False, expand = False, degree = 2)
#x = np.delete(x, np.arange(8,11), axis = 1)
#names, x = select_elements_from_data(names, x, ["Pd"], remove = False)
#print(x.shape)
#x = scale(x)

#rank = estimate_rank( x, method = "IndividualF", percentage_missing = 0.25, k_fold_rows = 23, k_fold_col = 8, num_seeds = 1, plot = True)

#******PCA PROJECTION TO THE SPACE OF DIMENSIONS EQUAL TO RANK**************************#
#x_transf = PCA(n_components = int(rank)).fit_transform(x)
x_transf = PCA(n_components = 2).fit_transform(x)
#*************************CLUSTER IN THIS NEW SPACE************************************#
#eps_array_euclidean = np.array([0.1, 0.15, 0.15, 0.2, 0.243, 0.246, 0.249, 0.25, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27])
eps_array_euclidean = np.array([ 0.35]) # for 20, 30, 40
#eps_array_euclidean = np.array([0.82, 1.06, 1.23, 1.35, 1.42, 1.54, 1.59, 1.64, 1.72]) #For extended features
#eps_array_euclidean = np.array([0.15, 0.17, 0.17, 0.2, 0.2, 0.2, 0.21, 0.23, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27])  #correct k-dist
eps_array_manhattan = np.array([0.5, 0.6, 0.7, 0.7, 0.8, 0.8, 0.85, 0.85])
for ind, min_samples in enumerate(np.arange(30, 40, 10)):
   #for eps in np.arange(0.25, 0., 0.05):

    eps = eps_array_euclidean[ind]  #eps and min samples are extimated by k-distance plot
    #eps = 0.3
    #min_samples = 3
    db = DBSCAN(eps = eps, min_samples = min_samples, metric ='euclidean').fit(x_transf)
    labels = db.labels_
    unique_labels = set(labels)
    n_cluster = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print("Eps = " + str(eps) + " min_samples = " + str(min_samples))
    print("Number of found clusters is " + str(n_cluster))
    print("Silhouette Coefficient: %0.3f" %silhouette_score(x_transf, labels)) 
    #% metrics.silhouette_score(x_transf, labels, metric = 'euclidean'))
    print("============================================")
    #**********************FINAL PROJECTION IN 2D FOR VISUALIZATION***********************#

    x2d = PCA(n_components = 2).fit_transform(x)
    #Plot Cluster
    plt.figure()
    for k in unique_labels:

        if (k != -1):

            class_k_members_mask = (labels == k)

            points_k_class = x2d[class_k_members_mask]

            
            plt.plot(points_k_class[:,0], points_k_class[:,1], 'o', alpha = 1)

            #print("In class k = " + str(k) + "\n" + str(names[class_k_members_mask]) )

        #else: break
            
    plt.title('2D PCA Scatter Plot of clusters obtained by DBSCAN in ' + str(5) + " dimensions. " + '\n' + " Parameters. eps = " + str(eps) + " NumPointsMin = " + str(min_samples) + " for all the element" )

    noise_mask = (labels ==-1)
    noisy_points = x2d[noise_mask]
    #Plot Noise
    plt.figure()
    plt.plot(noisy_points[:,0], noisy_points[:,1], 'o')
    plt.title("Outliers Points obtained by DBSCAN in " + str(5) + " dimensions. " + '\n' + " Parameters. eps = " + str(eps) + " NumPointsMin = " + str(min_samples) )
plt.show()