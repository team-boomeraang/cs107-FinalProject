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
    
    Examples
    -------
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> v1 = AD(0.0, 'v1')
    >>> v2 = AD(0.0, 'v2')
    >>> v3 = AD(0.0, 'v3')
    >>> print(_rowsums(x, [v1, v2, v3]))
    [0.0 ({'v1': 1.0, 'v2': 2.0, 'v3': 3.0})
     0.0 ({'v1': 4.0, 'v2': 5.0, 'v3': 6.0})]
    """
    return np.sum(data * np.array(var_list).reshape(1,-1), axis=1)

def mse(data, var_list, outcome_ind):
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
    >>> mse(x, varlist, 0)
    8.5 ({'v1': -22.0, 'v2': -27.0})
    """
    if not isinstance(outcome_ind, int):
        raise TypeError('outcome_ind should be an integer')
    try:
        data_arr = np.array(data)
        assert data_arr.ndim == 2, 'data must be convertible to 2-D array!'
        
        # Find column list and remove column for outcome index
        feat_list = list(np.arange(data_arr.shape[1])) #.remove(outcome_ind)
        feat_list.remove(outcome_ind)
        outcome_data = data_arr[:,outcome_ind]
        
        sse = np.sum((outcome_data - _rowsums(data_arr[:,feat_list], var_list)) ** 2)
        return (1/len(outcome_data))*sse
    
    except ValueError:
        return ValueError('outcome_ind is not a valid column index in data')

def cross_entropy(data, var_list, outcome_ind):
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
    >>> cross_entropy(x, varlist, 0)
    0.6931471805599453 ({'v1': 0.75, 'v2': 0.75})
    >>> assert type(cross_entropy(x, varlist, 0)) == AD
    """
    if not isinstance(outcome_ind, int):
        raise TypeError('outcome_ind should be an integer')
    try:
        data_arr = np.array(data)
        assert data_arr.ndim == 2, 'data must be convertible to 2-D array!'
        
        # Find column list and remove column for outcome index
        feat_list = list(np.arange(data_arr.shape[1])) #.remove(outcome_ind)
        feat_list.remove(outcome_ind)
        outcome_data = data_arr[:,outcome_ind]
        for x in outcome_data:
            assert x in [0, 1], 'All outcomes must be 0 or 1!'
            
        # Use logistic function to get prediction probabilities
        pred_probs = AD.logistic(_rowsums(data_arr[:,feat_list], var_list))
        log_probs = [AD.log(p) for p in pred_probs]
        log_1_probs = [AD.log(1 - p) for p in pred_probs]
        mf = -(1/len(outcome_data))
        return mf*np.sum((outcome_data*log_probs) + (1 - outcome_data)*log_1_probs)
    
    except ValueError:
        return ValueError('outcome_ind is not a valid column index in data')
    
def mae():
    pass
"""
if __name__ == '__main__':
    import doctest
    doctest.testmod()

    from sklearn.linear_model import LinearRegression
    x1 = np.array([[2, 3], [5, 6]])
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 4])
    l = LinearRegression(fit_intercept=False)
    l.fit(x1, y)
    v1 = AD(0.0, 'v1')
    v2 = AD(0.0, 'v2')    
    m = lambda: mse(x, [v1, v2], 0)
"""    