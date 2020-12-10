import boomdiff
from boomdiff import AD
import doctest
import pytest
import numpy as np


@pytest.fixture
def data():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    return x

@pytest.fixture
def v1():
    v1 = AD(0.0, 'v1')
    return v1

@pytest.fixture 
def v2():
    v2 = AD(0.0, 'v2')
    return v2

# Test loss function capabilities
def test_rowsum():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    v1 = AD(0.0, 'v1')
    v2 = AD(0.0, 'v2')
    v3 = AD(0.0, 'v3')
    f = boomdiff.loss_function._rowsums(x, [v1, v2, v3])
    assert f[0] == AD(0.0, {'v1': 1.0, 'v2': 2.0, 'v3':3.0})
    assert f[1] == AD(0.0, {'v1': 4.0, 'v2': 5.0, 'v3': 6.0})

# MSE
def test_mse_nonint():
    varlist = [v1, v2]
    with pytest.raises(TypeError):
        boomdiff.loss_function.linear_mse(data, varlist, '1')
    
def test_mse_outind():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    v1 = AD(0.0, 'v1')
    v2 = AD(0.0, 'v2')
    varlist = [v1, v2]
    with pytest.raises(IndexError):
        boomdiff.loss_function.linear_mse(x, varlist, 5)

# CROSS ENTROPY
def test_ce_nonint():
    varlist = [v1, v2]
    with pytest.raises(TypeError):
        boomdiff.loss_function.logistic_cross_entropy(data, varlist, '1')   

def test_ce_outind():
    varlist = [v1, v2]
    with pytest.raises(IndexError):
        boomdiff.loss_function.logistic_cross_entropy(data, varlist, 5)