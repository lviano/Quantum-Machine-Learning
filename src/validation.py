import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import f

def gabrielHoldsOut(x, k_fold_rows, k_fold_col, num_seeds = 1, plot = False):

    """Function to perform a KL fold validation as described at page 101 of the thesis about cross validation for unsupervised learning
    
    Args:
        x (matrix):     it must be already normalized and cleaned by molecules name.
        num_seeds:      number of different row and column shuffling that must be considered.
        k_fold_rows:    rows subgroups number.
	k_fold_column:  columns subgroup number.
	plot:		boolean flag. True for ploting.

    Returns:
        vector_error:   vector storing in position k the average Sum of Square Errors of the k-rank approximation gabriel sub-matrices.
	matrix_error: 	matrix storing in position (k, index) the Sum of Square Errors of the k-rank approximation for the gabriel submatrices labelled by index.
        
    """
   # random creation of the indices of the subgroups

    model_error = np.zeros([x.shape[1], k_fold_rows*k_fold_col, num_seeds])
   
    for seed in range(num_seeds):
        np.random.seed(seed)
        k_indices_rows = build_k_indices(x.shape[0], k_fold_rows, seed)
        k_indices_columns = build_k_indices(x.shape[1], k_fold_col, seed)

        # nested loop over the 4 submatrices created partitioning rows and columns of x. They are:
        # 1)y_tr contains the elements that are used as training sets labels
        # 2)x_tr are the data to fit the least square model. Here used not for aiming 
        # at predict new responses by just to evaluate how much information is lost reducing dimensionality
        # 3)x_te are the data considered unlabelled
        # 4)y_te are the entries that are considered unknown and that we aim to reconstruct
          
        for index_row, k_group_rows in enumerate(k_indices_rows):
            for index_col, k_group_columns in enumerate(k_indices_columns):

                y = x[:, k_group_columns]
                red_x = np.delete(x, k_group_columns, axis = 1)

                x_te = red_x[k_group_rows]
                y_te = y[k_group_rows]
            
                x_tr = np.delete(red_x, k_group_rows, axis = 0)
                y_tr = np.delete(y, k_group_rows, axis = 0)

                [U1, S1, V1] = np.linalg.svd(x_tr, full_matrices = False)
                for dim in range(x_tr.shape[1]):
                    
                    #detailed description of the following algebra at page 86 of 
                    # Patrik O.Perry "Cross-Validation for Unsupervised Learning "
                    reducedV1 = V1[:dim, :]
                    reducedV1 = reducedV1.T

                    Z1 = np.dot(x_tr, reducedV1)
                    Z2 = np.dot(x_te, reducedV1)

                    B = np.linalg.solve(np.dot(Z1.T,Z1),np.dot(Z1.T,y_tr))

                    predicted_y_te = np.dot(Z2,B)
                    index = index_row * k_fold_col + index_col
                    model_error[dim, index, seed] = np.linalg.norm(y_te - predicted_y_te)

    matrix_error = np.mean(model_error, axis = 2)
    vector_error = np.mean(matrix_error, axis = 1)
    matrix_error = matrix_error[:x_tr.shape[1]]
    vector_error = vector_error[:x_tr.shape[1]]

    if plot==True:
        dimensions = np.arange(1, x_tr.shape[1] + 1)
        plt.figure()
        plt.plot(dimensions, vector_error)
        plt.title('Gabriel error vs dimensions')
        plt.xlabel('Number of dimensions')
        plt.ylabel('Prediction error')

    return matrix_error, vector_error         


def build_k_indices(num_rows, k_fold, seed):
    """build k indices for k-fold."""
    np.random.seed(seed)

    interval = int(num_rows / k_fold)
    indices = np.random.permutation(num_rows)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)    


def F_test( square_errors, dimensions, number_of_data, plot = False):
    """Function to perform an hypotesis testing under the null hypothesis "The i-th principal components doesn't affect the regression result"

    Args:
        square_errors (array)   it contains the error in position i the MSE 
                                committed with the regression that uses as a projection plane
                                the column space spanned by a certain number of principal columns of the PCA transformed matrix.
        dimensions (array)      it contains in position i the number of principal components considered in computing the i-th element of square_errors.
        number_of_data          it is the number of samples (row of matrix X) considered in the test set of gabriel setup.
        plot (boolean)          True for plotting the p_values of the test.

    Returns:
        p-values                array storing the p-values for the performed tests.
        rank         		stores the lowest dimension that showed a p-value larger than 5%.
	index_significant_p	stores the indices corresponding to the dimensions showing a p-value greater than the desired thershold.
    """
    diff_square_errors = (square_errors[:-1] - square_errors[1:])
    
    diff_dimensions = (dimensions[1:] - dimensions[:-1])
    
    ratio = (diff_square_errors / diff_dimensions) / ((square_errors[-1]) / (number_of_data - dimensions[-1]))
    
    p_values = [1 - f.cdf(ratio[i], diff_dimensions[i], number_of_data - dimensions[-1]) for i in range(len(dimensions) - 1)]

    #print(' Not significant dimensions at level 0.05 are ' + str((np.argwhere(np.array(p_values) > 0.05).T)))

    index_significant_p = np.nonzero(np.array(p_values) > 0.05)
   
    rank = 0
    for n in range(len(dimensions) - 1):
        if (p_values[n] > 0.05):
            rank = dimensions[n+1]
            break
    if plot == True:
        plt.figure()
        plt.plot(dimensions[1:], p_values)
        plt.title("p_values vs Degree of Freedom" )

    return p_values, rank, index_significant_p
    
def woldHoldsOut(x, num_seeds, percentage_missing = 0.1, plot = False):
    """
    Function that gives an estimation of the rank of the data matrix considering the expectation-maximization 
    algorithm for the Wold Hold Out cross validation approach. (See page 92 Patrik O.Perry "Cross-Validation for Unsupervised Learning ")
    A random seed is used to generate indeces of elements considered as unknown. 
    They are replaced by the mean of non missing value in the same column and the SVD decomposition of this new matrix is computed (data_matrix = USV^T).
    The approximation of rank k is obtained considering the product keeping only the first k columns of U and V and elements in the diagonal of S. 
     

    Args:

        data_matrix:    array whose rows correspond to one molecule

        num_seeds:      number of different pseudo_randomly selected elements to be considered "missing"

        plot:           boolean flag that indicates whether a graph of Wold's PE estimate is needed or not 

        percentage_missing: ratio of value considered unknown

    Returns

        PE:             prediction error computed as Frobenius norm of the difference between the data matrix and its k-rank approximation. The latter 
                        is computed considering the SVD decomposition of the missing value matrix truncated at the k highest eigenvalues

        estimated_rank: k at which the PE is the lowest 

    """
    data_matrix = np.copy(x)
    PE = np.zeros([data_matrix.shape[1], num_seeds])
    for seed in range(num_seeds):
        missing_value_matrix = np.zeros_like(data_matrix)
        size = int(data_matrix.shape[0]*data_matrix.shape[1]*percentage_missing)
        np.random.seed(seed)
        mask = np.zeros_like(data_matrix, dtype = int)
        indices_row = np.random.randint(data_matrix.shape[0], size = (size,1))
        indices_col = np.random.randint(data_matrix.shape[1], size = (size,1))
        indices = np.c_[indices_row, indices_col]
        mask[indices[:,0], indices[:,1]] = 1
       
        missing_value_matrix = np.ma.array(data_matrix, mask = mask)
        means = np.mean(missing_value_matrix, axis = 0)
        opposite_mask = np.ones_like(mask)-mask  #mask of non missing elements
        indices_known = np.transpose(np.nonzero(opposite_mask))
        indices_unknown = np.transpose(np.nonzero(mask))       
        
        for column_id, column in enumerate(data_matrix.T): #along columns
            mask_column = mask[:,column_id]
            index = np.nonzero(mask_column)
            missing_value_matrix[index, column_id] = means[column_id]


        #for k in range(data_matrix.shape[1]):
        for k in np.arange(1,data_matrix.shape[1]):

            RSS_old = 0

            while True:
                [U, s, V] = np.linalg.svd(missing_value_matrix, full_matrices = False)
                U_k = U[:, :k]
                s = s[:k]
                V_k = V[:k,:]
                missing_value_matrix_k = U_k @ np.diag(s) @ V_k
                
                #RSS = np.linalg.norm(data_matrix - missing_value_matrix_k, 'fro')
                RSS = np.linalg.norm(data_matrix[indices_known[:,0],indices_known[:,1]] - missing_value_matrix_k[indices_known[:,0],indices_known[:,1]], 2)
                if (RSS - RSS_old < 1e-5):
                    break
                #missing_value_matrix[mask] = missing_value_matrix_k[mask]     #update missing value SVD prediction
                missing_value_matrix[np.transpose(np.nonzero(mask))] = missing_value_matrix_k[np.transpose(np.nonzero(mask))]
                RSS_old = RSS

            #PE[k,seed] = RSS
            PE[k, seed] = np.linalg.norm(data_matrix[indices_unknown[:,0],indices_unknown[:,1]] - missing_value_matrix_k[indices_unknown[:,0],indices_unknown[:,1]], 2)
    PE = np.mean(PE, axis = 1)
    estimated_rank = np.argmin(PE[1:])
    print("The estimated rank by Wold is " + str(estimated_rank))
    
    if plot == True:
        plt.figure()
        plt.plot(np.arange(1, data_matrix.shape[1]), PE[1:], 'r', label = 'Prediction error')
        plt.title('Wold Leave Out Prediction Error')
        plt.legend()        
    return PE, estimated_rank


def estimate_rank( x, method = "Wold", percentage_missing = 0.5, num_seeds = 1, k_fold_rows = 8, k_fold_col = 44, plot = False):
    #num_seeds = 1
    rank = 0
    if method == "AverageF" or method == "IndividualF":
        
        #k_fold_rows = 8
        #k_fold_col = 44
        matrix_error_gabr, square_errors_gabr = gabrielHoldsOut(x, k_fold_rows, k_fold_col, num_seeds, plot = plot)

        #dimensions for F test
        gabriel_covariate_dimensions = 1 + np.arange((np.ceil(x.shape[1] * (k_fold_col - 1) / k_fold_col)))
        gabriel_covariates_rows_number = x.shape[0] / k_fold_rows

        if method == "IndividualF":
           rank = individual_F_tests(matrix_error_gabr, gabriel_covariate_dimensions, gabriel_covariates_rows_number)
           return rank

        elif method == "AverageF":
            #FTEST ON THE AVERAGE OF THE ERROR
            #Probably is theoretically incorrect
            _, rank, _ = F_test( square_errors_gabr, gabriel_covariate_dimensions, gabriel_covariates_rows_number, plot = plot)
            return rank

    elif method == "Wold":
        #WOLD CROSS VALIDATION FOR RANK ESTIMATION
        _, rank = woldHoldsOut(x,num_seeds, percentage_missing = percentage_missing, plot = plot)
        return rank

    elif method == "CrossValidationWold":
        _, rank = woldHoldsOutCrossValidation(x, num_seeds, percentage_missing = percentage_missing, plot = plot)
        return rank
    else:
        print("Unknown method. \n Choose among 'Average F Test', 'Individual F tests', 'Wold', 'CrossValidationWold")


def individual_F_tests(matrix_error_gabr, gabriel_covariate_dimensions, gabriel_covariates_rows_number):
    """It estimates the rank of the data space finding the dimension for which more than 50% of the regression
        performed in the Gabriel frameset assumes a p-value larger than 0.05%


    """

    #The following vector will store the number of significant p_values per each number of dimensions
    number_of_significant_p_values = np.zeros(len(gabriel_covariate_dimensions)-1) 

    fig, ax1 = plt.subplots()
    plt.xlabel("Number of dimensions")
    ax1.set_ylabel("Number of significant p-values", color = "r")
    #plt.title("P value tests. One for each different configuration \n of Gabriel Cross Validation algorithm")
    ax2 = ax1.twinx()
    ax2.set_ylabel("p-value", color = "b")
    for id_column, column in enumerate(matrix_error_gabr.T):
        print(id_column)
        p_values, _, index = F_test( column,gabriel_covariate_dimensions, gabriel_covariates_rows_number, plot = False)
        number_of_significant_p_values[index] = number_of_significant_p_values[index] + 1  
        ax2.plot(gabriel_covariate_dimensions[1:], p_values, 'bo', alpha=0.05)
    ax1.plot(gabriel_covariate_dimensions[1:], number_of_significant_p_values, "--rx", markerSize = 10.0) #linewidth = 3.0, color = "r")
    ax2.set_ylim(0, 2)
    fig.tight_layout()
    #rank_vec = np.argwhere(number_of_significant_p_values > 0.5*len(matrix_error_gabr.T)).T
    #rank = np.array(rank_vec)[0]
    rank = 0
    for n in range(len(gabriel_covariate_dimensions)-1):
        if (number_of_significant_p_values[n] > 0.1*len(matrix_error_gabr.T)):
            rank = gabriel_covariate_dimensions[n+1]
            print("The estimated rank by Individual F tests is " + str(rank))
            return rank
    print("The estimated rank by Individual F tests is " + str(rank))        
    return gabriel_covariate_dimensions[-1]

def woldHoldsOutCrossValidation(x, num_seeds, percentage_missing = 0.25, plot = False):

    
    PE = np.zeros([x.shape[1], num_seeds, x.shape[0]*x.shape[1]])
    #model_error = np.zeros([x.shape[1], k_fold_rows*k_fold_col, num_seeds])
   
    for seed in range(num_seeds):
        data_matrix = np.copy(x)
        np.random.seed(seed)
        #size = int(data_matrix.shape[0]*data_matrix.shape[1]*percentage_missing)
        #print(size)
        k_indices_rows = build_k_indices(x.shape[0], int(np.sqrt(1/percentage_missing)), seed)
        k_indices_columns = build_k_indices(x.shape[1], int(np.sqrt(1/percentage_missing)), seed)
        for index_row, k_group_rows in enumerate(k_indices_rows):
            for index_col, k_group_columns in enumerate(k_indices_columns):
                missing_value_matrix = np.zeros_like(data_matrix)                     
                mask = np.zeros_like(data_matrix, dtype = int)
                indices = np.c_[np.tile(k_group_rows, len(k_group_columns)), np.repeat(k_group_columns, len(k_group_rows))]
                mask[indices[:,0], indices[:,1]] = 1
    
                missing_value_matrix = np.ma.array(data_matrix, mask = mask)
                means = np.mean(missing_value_matrix, axis = 0)
                opposite_mask = np.ones_like(mask)-mask  #mask of non missing elements
                indices_known = np.transpose(np.nonzero(opposite_mask))
                indices_unknown = np.transpose(np.nonzero(mask))
    
                for column_id, column in enumerate(data_matrix.T): #along columns
                    mask_column = mask[:,column_id]
                    index = np.nonzero(mask_column)
                    missing_value_matrix[index, column_id] = means[column_id]

                for k in np.arange(1, data_matrix.shape[1]):

                    RSS_old = 0

                    while True:
                        [U, s, V] = np.linalg.svd(missing_value_matrix, full_matrices = False)
                        U_k = U[:, :k]
                        s = s[:k]
                        V_k = V[:k,:]
                        missing_value_matrix_k = U_k @ np.diag(s) @ V_k
                        
                        
                        RSS = np.linalg.norm(data_matrix[indices_known[:,0],indices_known[:,1]] - missing_value_matrix_k[indices_known[:,0],indices_known[:,1]], 2)
                        if (RSS - RSS_old < 1e-10):
                            break
                        missing_value_matrix[np.transpose(np.nonzero(mask))] = missing_value_matrix_k[np.transpose(np.nonzero(mask))]     #update missing value SVD prediction
                        RSS_old = RSS

                    
                    PE[k,seed,index_row*x.shape[0]+index_col] = np.linalg.norm(data_matrix[indices_unknown[:,0],indices_unknown[:,1]] - missing_value_matrix_k[indices_unknown[:,0],indices_unknown[:,1]], 2)
                    print(PE[k,seed,index_row*x.shape[0]+index_col])
    PE = np.mean(PE, axis = 2)                
    PE = np.mean(PE, axis = 1)
    estimated_rank = np.argmin(PE[1:])
    print("The estimated rank by Wold is " + str(estimated_rank))
    
    if plot == True:
        plt.figure()
        plt.plot(np.arange(1,data_matrix.shape[1]), PE[1:], 'r', label = 'Prediction error')
        plt.title('Wold Leave Out Prediction Error')
        plt.legend()        
    return PE, estimated_rank
