import numpy as np
from boomdiff.autodiff import AD

"""Defines a series of loss functions to be called in various use-case
methods"""


def _rowsums(data, var_list):
    """Helper method for creating rows of data as AD objects using a format
    where data is an n x m array (no outcome data provided) and var_list is a
    length m list or broadcastable
    
    Parameters
    ----------
    data: np.array
        should be a n x m numpy array representing n observations and m
        features (excluding outcome)
    var_list: list or np.array
        list or array of AD objects
    
    Returns
    -------
    length n array representing rows of data matrix as AD objects
    """
    return AD.sum(data * np.array(var_list).reshape(1,-1), axis=1)

def linear_mse(data, var_list, outcome_ind):
    """Calculates mean squared error for AD objects. All objects passed to var_list
    must be instantiated AD objects. Highly preferable for data to be input as numpy
    array with observations as rows and features as columns.
    
    Parameters
    ----------
    data: np.array
        Data to be used in MSE function
    var_list: list; AD object
        list of AD objects to be included in function
    outcome_col:
        index of outcome in data
    
    Examples
    --------
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> v1 = AD(0.0, 'v1')
    >>> v2 = AD(0.0, 'v2')
    >>> varlist = [v1, v2]
    >>> linear_mse(x, varlist, 0)
    8.5 ({'v1': -22.0, 'v2': -27.0})
    """
    if not isinstance(outcome_ind, int):
        raise TypeError('outcome_ind should be an integer')
    try:
        # Step 1: Convert data to array format and confirm it is 2-dimensional
        # If not, could be converted by np.reshape(n,p) to get to n x p matrix
        data_arr = np.array(data)
        assert data_arr.ndim == 2, 'data must be convertible to 2-D array!'
        
        # Step 2: Separate out feature and outcome data
        # Note: Loss function could be constructed with data passed separately!
        # Then skip this step
        feat_list = list(np.arange(data_arr.shape[1]))
        feat_list.remove(outcome_ind)
        outcome_data = data_arr[:,outcome_ind]
        
        # Step 3: Calculate loss function - _rowsums() method can be used to
        # make x, beta to x*beta step
        sse = AD.sum((outcome_data - _rowsums(data_arr[:,feat_list], var_list)) ** 2)
        # Note: Linear MSE could be calculated in one step
        return (1/len(outcome_data))*sse
    
    except:
        raise IndexError('outcome_ind is not a valid column index in data')

def logistic_cross_entropy(data, var_list, outcome_ind):
    """Calculates the binary cross-entropy between true_label and predictions
    
    Parameters
    ----------
    data: np.array
        Data to be used in MSE function
    var_list: list; AD object
        list of AD objects to be included in function
    outcome_indl:
        index of outcome in data
    
    Returns
    -------
    Binary cross-entropy between labels
    
    Examples
    --------
    >>> x = np.array([[1, 2, 3], [0, 5, 6]])
    >>> v1 = AD(0.0, 'v1')
    >>> v2 = AD(0.0, 'v2')
    >>> varlist = [v1, v2]
    >>> logistic_cross_entropy(x, varlist, 0)
    0.6931471805599453 ({'v1': 0.75, 'v2': 0.75})
    >>> assert type(logistic_cross_entropy(x, varlist, 0)) == AD
    """
    if not isinstance(outcome_ind, int):
        raise TypeError('outcome_ind should be an integer')
    try:
        # Step 1: Convert data to array format and confirm it is 2-dimensional
        # If not, could be converted by np.reshape(n,p) to get to n x p matrix
        data_arr = np.array(data)
        assert data_arr.ndim == 2, 'data must be convertible to 2-D array!'
        
        # Step 2: Separate out feature and outcome data
        # Note: Loss function could be constructed with data passed separately!
        # Then skip this step
        feat_list = list(np.arange(data_arr.shape[1]))
        feat_list.remove(outcome_ind)
        outcome_data = data_arr[:,outcome_ind]
        
        # Step 2A (optional): Validate data. Because this is a logistic regression
        # output, we check that all data is zero or one
        for x in outcome_data:
            assert x in [0, 1], 'All outcomes must be 0 or 1!'
            
        # Step 3: Calculate loss function
        # Note: Because this is a logistic cross_entropy function,
        # we use AD.logisitc() to convert our linear row sums to probabilities
        pred_probs = AD.logistic(_rowsums(data_arr[:,feat_list], var_list))
        # Note: Factors are calculated separately to present information cleanly,
        # could be combined
        log_probs = AD.log(pred_probs)
        log_1_probs = AD.log(1 - pred_probs)
        mf = -(1/len(outcome_data))
        return mf*np.sum((outcome_data*log_probs) + (1 - outcome_data)*log_1_probs)
    
    except:
        raise IndexError('outcome_ind is not a valid column index in data')

