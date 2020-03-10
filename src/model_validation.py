# External libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

# Own libraries
import visualization as viz
import debug

def k_distance(x, visualization_options):
    """Function to produce the sorted k-dist graph. It is useful to set the DBSCAN parameters Eps and NumPoints.

    Args:
        x: data matrix. It must correspond to the features represented to the space chosen to perform the DBSCAN, e.g.,
            if for DBSCAN a bi-dimensional space is chosen, x must be the matrix resulting from the 2D PCA projection of the original data-matrix
        visualization_options (dict): contains the options for visualization (plotting)
    """
    # Specify whether the results shall be plotted
    # Default: False (no plotting)
    # Of course, it is only False for coherence. There is no sense in calling a
    # a plot-only function without plotting
    plot = viz.do_plot(visualization_options)

    if plot:
        k_distances_fig = plt.figure()

    for n in range(2, 10):
        neighbors = NearestNeighbors(n_neighbors=n).fit(x)
        distances, indices = neighbors.kneighbors(x)
        decreasing_distances = sorted(distances[:,n-1], reverse=True)  #sorting the points in decreasing k-dist order

        if plot:
            plt.plot(list(range(1, x.shape[0]+1)), decreasing_distances, label = str(n))

    if plot:
        plt.xlabel("Number of points")
        plt.ylabel("k distance")
        plt.title("k-distance plot")
        plt.legend()

        viz.visualize(k_distances_fig, 'k_distances', visualization_options)

def build_k_indices(num, k_fold, seed):
    """Build k indices for the k-fold rows or columns partition.

    Args:
        num (int): total number of rows or columns
        k_fold (int): number of subsets to be created
        seed (int): seed to generate the pseudo-random indices partition

    Returns:
        k_indices: matrix with k_fold rows. It contains in the i-th row the indices corresponding to the i-th subset of the partition
    """
    np.random.seed(seed)

    interval = int(num / k_fold)
    indices = np.random.permutation(num)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)

def plot_gabriel_holds_out(x_train, vector_error, visualization_options):
    """Plots the results found by Gabriel Holds out method

    Args:
        x_train (numpy array): train data
        vector_error (numpy array): associated error vector
        visualization_options (dict): contains the options for visualization (plotting)
    """
    dimensions = np.arange(1, x_train.shape[1] + 1)
    gabriel_fig = plt.figure()
    plt.plot(dimensions, vector_error)
    plt.title('Gabriel error vs dimensions')

    viz.visualize(gabriel_fig, 'gabriel_error', visualization_options)

def gabriel_holds_out(x, k_fold_rows, k_fold_cols, visualization_options, num_seeds = 1):
    """Function to perform a KL fold validation as described at page 101 of the thesis about cross validation for unsupervised learning

    Args:
        x (matrix): the data matrix which must be already normalized and cleaned by molecules name.
        num_seeds (int): number of different row and column shuffling that must be considered.
        k_fold_rows (int): rows subgroups number.
        k_fold_cols (int): columns subgroup number.
        visualization_options (dict): contains the options for visualization (plotting)

    Returns:
        matrix_error: 	matrix storing in position (k, index) the Sum of Square Errors of the k-rank approximation for the gabriel submatrices labelled by index.
    """
    # Random creation of the indices of the subgroups
    model_error = np.zeros([x.shape[1], k_fold_rows*k_fold_cols, num_seeds])

    for seed in range(num_seeds):
        np.random.seed(seed)
        k_indices_rows = build_k_indices(x.shape[0], k_fold_rows, seed)
        k_indices_columns = build_k_indices(x.shape[1], k_fold_cols, seed)

        # Nested loop over the 4 submatrices created by partition of rows and columns of x. They are:
        # 1) y_train contains the elements that are used as training sets labels
        # 2) x_train are the data to fit the least square model. Here used not to
        #    predict new responses by just to evaluate how much information is lost reducing dimensionality.
        # 3) x_test are the data considered unlabelled
        # 4) y_test are the entries that are considered unknown and that we aim to reconstruct

        for index_row, k_group_rows in enumerate(k_indices_rows):
            for index_col, k_group_columns in enumerate(k_indices_columns):

                y = x[:, k_group_columns]
                red_x = np.delete(x, k_group_columns, axis = 1)

                x_test = red_x[k_group_rows]
                y_test = y[k_group_rows]

                x_train = np.delete(red_x, k_group_rows, axis = 0)
                y_train = np.delete(y, k_group_rows, axis = 0)

                [_, _, V1] = np.linalg.svd(x_train, full_matrices = False)
                for dim in range(x_train.shape[1]):

                    # Detailed description of the following algebra at page 86 of
                    # Patrik O.Perry "Cross-Validation for Unsupervised Learning"
                    reducedV1 = V1[:dim, :]
                    reducedV1 = reducedV1.T

                    Z1 = np.dot(x_train, reducedV1)
                    Z2 = np.dot(x_test, reducedV1)

                    B = np.linalg.solve(np.dot(Z1.T,Z1),np.dot(Z1.T,y_train))

                    predicted_y_test = np.dot(Z2,B)
                    index = index_row * k_fold_cols + index_col
                    model_error[dim, index, seed] = np.linalg.norm(y_test - predicted_y_test)

    matrix_error = np.mean(model_error, axis = 2)
    vector_error = np.mean(matrix_error, axis = 1)
    matrix_error = matrix_error[:x_train.shape[1]]
    vector_error = vector_error[:x_train.shape[1]]

    if viz.do_plot(visualization_options):
        # Call to plot_gabriel_holds_out() to generate the appendix figure 5
        plot_gabriel_holds_out(x_train, vector_error, visualization_options)

    return matrix_error

def f_test(square_errors, dimensions, number_of_data, threshold = 0.05):
    """Function to perform an hypotesis testing under the null hypothesis "The i-th principal components doesn't affect the regression result"
    Args:
        square_errors: array that contains the error in position i the MSE committed with the regression
            that uses as a projection plane the column space spanned by a certain number of principal columns of the PCA transformed matrix
        dimensions: array that contains in position i the number of principal components considered in computing the i-th element of square_errors
        number_of_data: the number of samples (row of matrix X) considered in the test set of gabriel setup

        Args:
            square_errors (array): contains the error in position i the MSE
                committed with the regression that uses as a projection plane
                the column space spanned by a certain number of principal columns of the PCA transformed matrix.
            dimensions (array): contains in position i the number of principal components considered in computing the i-th element of square_errors.
            number_of_data (int): the number of samples (row of matrix X) considered in the test set of gabriel setup.
            plot (boolean): True for plotting the p_values of the test.
            threshold: level above that a p value is considered significant

        Returns:
            p_values (list): list storing the p-values for the performed tests.
            rank (int): stores the lowest dimension that showed a p-value larger than 5%.
    	    index_significant_p (list): stores the indices corresponding to the dimensions showing a p-value greater than the desired thershold.
    """
    diff_square_errors = (square_errors[:-1] - square_errors[1:])
    diff_dimensions = (dimensions[1:] - dimensions[:-1])

    # Implementation of equation (2) of the report
    ratio = (diff_square_errors / diff_dimensions) / ((square_errors[-1]) / (number_of_data - dimensions[-1]))

    # Implementation of equation (3) of the report
    p_values = [1 - f.cdf(ratio[i], diff_dimensions[i], number_of_data - dimensions[-1]) for i in range(len(dimensions) - 1)]

    debug.log('Not significant dimensions at level' + str(threshold) + ' are ' + str((np.argwhere(np.array(p_values) > threshold).T)) + '.')

    index_significant_p = np.nonzero(np.array(p_values) > threshold)

    rank = 0
    for n in range(len(dimensions) - 1):
        if (p_values[n] > threshold):
            rank = dimensions[n+1]
            break

    return p_values, rank, index_significant_p

def individual_f_tests(gabriel_error_matrix, gabriel_covariate_dimensions, gabriel_covariate_row_number, visualization_options, percentage_of_significant_p = 0.1 ):
    """Function that counts how many gabriel configuration shows a significant p-value. It estimates the rank finding the
    smallest dimension so that the number of significant p-values is higher than percentage_of_significant_p*total_number_of_gabriel_configurations.
    This is defined condition (1) in the following.

        Args:
            gabriel_error_matrix: matrix storing in position (k, index) the Sum of Square Errors
                of the k-rank approximation for the gabriel submatrices labelled by index.
                It is the output 'matrix_error' of the function gabriel_holds_out().
            gabriel_covariate_dimensions (int): vector stores the dimensions of the model
                that gives the prediction errors that are stored in the i-th row of gabriel_error_matrix in i-th position
            gabriel_covariate_row_number (int): the number of data, i.e. the number of rows of the data matrix x
            percentage_of_significant_p (float): onset percentage that defines the threshold percentage_of_significant_p * total_number_of_gabriel_configurations
            visualization_options (dict): contains the options for visualization (plotting)

        Returns:
            rank (int): smallest number of dimensions at which the number of significant p-values is larger than
                percentage_of_significant_p*total_number_of_gabriel_configurations. If this condition is never True, the
                maximum dimension stored in gabriel_covariate_dimensions is returned
    """
    # Specify whether this analysis shall be plotted
    # Default: False (no plotting)
    plot = viz.do_plot(visualization_options)

    # The following vector will store the number of significant p-values for each number of dimensions
    number_of_significant_p_values = np.zeros(len(gabriel_covariate_dimensions)-1)

    # Here, if specified we print on the fly so that we do not need to save otherwise not needed data
    # These plot instructions generate figure 1 of the report
    if plot:
        f_fig, f_ax1 = plt.subplots()
        plt.xlabel("Number of dimensions")
        f_ax1.set_ylabel("Number of significant p-values", color = "r")
        plt.title("P value tests. One for each different configuration \n of Gabriel Cross Validation algorithm")
        f_ax2 = f_ax1.twinx()
        f_ax2.set_ylabel("p-value", color = "b")

    # For loop ranging on the column of the data matrix. Each of them represents a gabriel configuration.
    # It is passed to the function F-test. The returned indices of the dimensions at which significant
    # p-values are observed, are used to increment the corresponding elements of the vector number_of_significant_p_values.
    # It stores in the i-th position the number of significant p-values at the dimension i.

    for id_column, column in enumerate(gabriel_error_matrix.T):
        p_values, _, index = f_test( column,gabriel_covariate_dimensions, gabriel_covariate_row_number)
        number_of_significant_p_values[index] = number_of_significant_p_values[index] + 1

        if plot:
            f_ax2.plot(gabriel_covariate_dimensions[1:], p_values, 'bo', alpha=0.05)

    if plot:
        f_ax1.plot(gabriel_covariate_dimensions[1:], number_of_significant_p_values, linewidth = 3.0, color = "r")
        f_ax2.set_ylim(0, 2)
        f_fig.tight_layout()

        # Visualization
        viz.visualize(f_fig, 'individual_F_tests', visualization_options)

    rank = 0

    # For loop that ranges over the vector number_of_significant_p_values. The rank is determined evaluating
    # for each element of number_of_significant_p_values the condition (1).
    for n in range(len(gabriel_covariate_dimensions)-1):
        if (number_of_significant_p_values[n] > percentage_of_significant_p * len(gabriel_error_matrix.T)):    #len(gabriel_error_matrix.T) is the number of
                                                                                                            #considered gabriel configuration
            rank = gabriel_covariate_dimensions[n+1]
            return rank

    return gabriel_covariate_dimensions[-1]  # return the largest number of dimensions if none of the elements of number_of_significant_p_values verifies the condition (1)

def estimate_rank(x, visualization_options, num_seeds = 1, k_fold_rows = 8, k_fold_cols = 8):
    """Function that calls gabriel_holds_out() at first for getting the prediction errors matrix.
    Then, the p-values are quantified calling individual_f_test that returns the rank estimation.

        Args:
            x: data matrix.
            num_seeds (int): number of times the gabriel cross validation scheme has to be repeated
            k_fold_rows (int): number of row-indices subsets to be considered.
            k_fold_cols (int): number of column-indices subsets to be considered.
            visualization_options (dict): contains the options for visualization (plotting)

        Returns:
            rank (int): rank estimated by individual_f_test
    """
    rank = 0
    gabriel_matrix_error = gabriel_holds_out(x, k_fold_rows, k_fold_cols, visualization_options, num_seeds = num_seeds)

    # Setting dimensions for F test
    gabriel_covariate_dimensions = 1 + np.arange((np.ceil(x.shape[1] * (k_fold_cols - 1) / k_fold_cols)))
    gabriel_covariate_row_number = x.shape[0] / k_fold_rows

    # Call individual_f_tests() to estimate rank
    rank = individual_f_tests(gabriel_matrix_error, gabriel_covariate_dimensions, gabriel_covariate_row_number, visualization_options)

    return rank


def silhouette_score(data_matrix, labels):
    """Function that calls the scikit tool to compute the silhouette score. It is useful to assess the homogeneity of the clusters.

    Args:
        data_matrix (array): matrix representing the data in the space where DBSCAN is performed.
        label (list): clusters label. Output of DBSCAN

    Returns:
        silhouette (float): silhouette score
    """
    silhouette = metrics.silhouette_score(data_matrix, labels)
    return silhouette
