import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from validation import *
from load_data import*

path = '../data/'
names, x = load_features(path + 'proposition_of_features.txt', expand = True, degree = 2)
#******************GABRIEL/F-TEST CROSS VALIDATION FOR RANK ESTIMATION**********************#
num_seeds = 1
k_fold_rows = 8
k_fold_col = 8
matrix_error_gabr, square_errors_gabr = gabrielHoldsOut(x, k_fold_rows, k_fold_col, num_seeds, plot = True)

#dimensions for F test
gabriel_covariate_dimensions = 1 + np.arange((np.ceil(x.shape[1] * (k_fold_col - 1) / k_fold_col)))
gabriel_covariates_rows_number = x.shape[0] / k_fold_rows

#*****************************FTEST ON SINGULAR CONFIGURATION OF GABRIEL********************#
#The following vector will store the number of significant p_values per each number of dimensions
number_of_significant_p_values = np.zeros(len(gabriel_covariate_dimensions)-1) 

fig, ax1 = plt.subplots()
plt.xlabel("number of dimensions")
ax1.set_ylabel("number of significant p_values", color = "r")
plt.title("P value tests. One for each different configuration \n of Gabriel Cross Validation algorithm")
ax2 = ax1.twinx()
ax2.set_ylabel("p-value", color = "b")
for id_column, column in enumerate(matrix_error_gabr.T):
    p_values, rank, index = F_test( column,gabriel_covariate_dimensions, gabriel_covariates_rows_number, plot = False)
    print(rank)
    number_of_significant_p_values[index] = number_of_significant_p_values[index] + 1  
    ax2.plot(gabriel_covariate_dimensions[1:], p_values, linewidth = 0.5, color = "b")
ax1.plot(gabriel_covariate_dimensions[1:], number_of_significant_p_values, linewidth = 3.0, color = "r")
ax2.set_ylim(0, 2)
fig.tight_layout()

#*************************FTEST ON THE AVERAGE OF THE ERROR********************************#
#Probably is theoretically incorrect
p_values, rank, index = F_test( square_errors_gabr, gabriel_covariate_dimensions, gabriel_covariates_rows_number, plot = True)
#************************WOLD CROSS VALIDATION FOR RANK ESTIMATION**********************#
square_errors_wold, rank_wold = woldHoldsOut(x,num_seeds, plot = True)
#******PCA PROJECTION TO THE SPACE OF DIMENSIONS EQUAL TO RANK**************************#
x = PCA(n_components = rank_wold).fit_transform(x)
#*************************CLUSTER IN THIS NEW SPACE************************************#
eps = 3  #eps and min samples are extimated by k-distance plot
min_samples = 56
db = DBSCAN(eps = eps, min_samples = min_samples).fit(x)
labels = db.labels_
unique_labels = set(labels)
n_cluster = len(unique_labels) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Number of found clusters is " + str(n_cluster))
#**********************FINAL PROJECTION IN 2D FOR VISUALIZATION***********************#

x2d = PCA(n_components = 2).fit_transform(x)
#Plot Cluster
plt.figure()
for k in unique_labels:

    if (k != -1):

        class_k_members_mask = (labels == k)

        points_k_class = x2d[class_k_members_mask]

        
        plt.plot(points_k_class[:,0], points_k_class[:,1], 'o')

        print("In class k = " + str(k) + "\n" + str(names[class_k_members_mask]) )

    else: break
        
plt.title('2D PCA Scatter Plot of clusters obtained by DBSCAN in ' + str(rank_wold) + " dimensions. " + '\n' + " Parameters. eps = " + str(eps) + " NumPointsMin = " + str(min_samples) )

noise_mask = (labels ==-1)
noisy_points = x2d[noise_mask]
#Plot Noise
plt.figure()
plt.plot(noisy_points[:,0], noisy_points[:,1], 'o')
plt.title("Outliers Points obtained by DBSCAN in " + str(rank_wold) + " dimensions. " + '\n' + " Parameters. eps = " + str(eps) + " NumPointsMin = " + str(min_samples) )
plt.show()