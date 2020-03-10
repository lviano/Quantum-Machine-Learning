import numpy as np
import os.path

from sklearn.preprocessing import scale
from element_information import*
def load_features(path, filename, normalize = True, add_information = False, expand = False, degree = 2):
    """Loads the molecules matrix

    Args:
        path (string): file location
        normalize (boolean flag): true for performing normalization with function scale of scikit
        expand (boolean flag): true for considering also columns generated by elements wise power operation and interactions between
        degree (int): degree of the expansion

    Returns:
        x(array): data matrix
        names(array): array of names of molecules .xyz files
    """
    filepath = os.path.join(os.path.normpath(path), filename)

    names = np.genfromtxt(filepath, delimiter=",", skip_header=0, dtype=str, usecols=0)
    x = np.genfromtxt(filepath, delimiter=",", skip_header=0)
    x = x[:,1:]

    if add_information:
        x = add_element_information_to_data(names, x)

    if normalize:
        x = scale(x)

    if expand:
        column_powers = build_poly(x,degree)
        interactions = expand_features(x,degree)
        x = np.c_[column_powers, interactions]

    return names, x

def load_energies(path):
    """
    Loads the associated energies of the molecules

    Args:
        path (string): file location

    Returns:
        energies(array): energies of the molecule
        names(array): names of molecules
    """
    data = np.genfromtxt(path, dtype=str)
    names = data[:,0]
    energies = data[:,1].astype(float)
    return energies, names

def build_poly(tx, degree):
    """
    Builds for each column a of the input matrix tx the following powers : a, a^2, ..., a^degree and returns the original matrix tx along with the new features.

    Args:
        x (array): matrix of shape (N,D) (N is the number of data and D is the number of features)
        degree (int): degree of maximum powers

    Returns:
        matrix: of dimension (N, (degree+1)*D)
                The first degree+1 columns are the powers of first column of the input matrix tx, the columns with index between degree+1 and 2*degree+1 are powers of the second column of tx and so on
    """
    tx = np.array(tx)

    length = tx.shape[0]
    width = tx.shape[1]

    tx = np.repeat(tx, degree, axis=1)
    tx.shape = (length, (degree)*width)
    powers = np.tile(np.arange(1, degree + 1), width) #Builds
    x_poly = np.power(tx, powers)

    return x_poly

def expand_features(tx, degree):
    """
    Builds for each column a,b of the input matrix tx (with shape (N,D) ) the columns corresponding to the cross terms of the following polynomial : (a + b)^degree and returns the original matrix tx along with the new features columns.
    If degree is set to 1, then it makes for each column a,b of the input matrix tx a column-wise multiplication : a*b.

    Args:
        tx (array): input matrix of shape (N,D) (D is the number of features)
        degree (int): degree of the polynomial. If degree == 1, it makes column wise multiplication.
    Returns:
        array: original matrix tx with the new features columns
    """
    num_data = tx.shape[0]
    num_original_col = tx.shape[1]
    for i in range(num_original_col):
        for j in range(i):
            if degree == 1:
                feature = (tx[:,i]*tx[:,j])
                feature.shape = (num_data,1)
                tx = np.concatenate((tx, feature), axis=1)
            else:
                for k in np.arange(1,degree):
                    feature = tx[:,i]**(degree-k)*tx[:,j]**k
                    feature.shape = (num_data,1)
                    tx = np.concatenate((tx, feature), axis=1)

    return tx[:,num_original_col:]


def split_data(y, tx, ratio, seed=1):
    """
    Splits (and shuffles) the data of the output vector y and input matrix tx into two set : training set and test set

    Args:
        y (array): output vector of shape (N,) (N is the data size)
        tx (array): input matrix of shape (N,D) (D is the number of features)
	    ratio (float): percentage of data to be used for training
	    seed (int or float): for random number generation

    Returns:
	    tx_training (array): contains the elements of the original array that have been assigned to the train subset
        y_training (array): contains the elements of the original array that have been assigned to the train subset
	    tx_testing (array): contains the elements of the original array that have been assigned to the test subset
	    y_testing (array): contains the elements of the original array that have been assigned to the test subset

        """
    # set seed
    np.random.seed(seed)
    length = len(tx)
    random_index = np.random.permutation(length)
    q = int(np.floor(length*ratio))

    training_index = random_index[0:q]
    testing_index = random_index[q:]

    return tx[training_index], y[training_index], tx[testing_index], y[testing_index]
