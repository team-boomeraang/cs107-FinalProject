import boomdiff
from boomdiff import AD
import doctest
import pytest

## Test suite for boomdiff

# Test initalization of base class
def test_base():
    x = AD(1, {'x1':1})
    assert x.func_val == 1
    print(x.func_val)
    assert x.partial_dict['x1'] == 1
    assert len(x.partial_dict.keys()) == 1
    
def test_badbaseval():
    # Test initialize base class without float or integer type
    with pytest.raises(ValueError):
        x = AD('v')
        
def test_nonnum_dict_init():
    # Test values of passed dictionary are int or float
    with pytest.raises(ValueError):
        x = AD(3.6, {'x1': 'bad'})
        
def test_nondict_init():
    # Test that if der_dict passed, it is dictionary 
    with pytest.raises(ValueError):
        x = AD(1.5, 9.3)

def test_set_nonatt():
    # Test that set_params(att) must be one of 'func_val' or 'partial_dict'
    with pytest.raises(ValueError):
        x = AD(1.3)
        x.set_params('evaluation_point', 2)

def test_set_funcval():
    # If setting 'func_val', must be type float or int
    with pytest.raises(ValueError):
        x = AD(13.2)
        x.set_params('func_val', '2.3')

def test_set_dictionary():
    # If set 'partial_dict', must be dictionary
    with pytest.raises(ValueError):
        x = AD(13.2)
        x.set_params('partial_dict', 2)

def test_setparam_dictvals():
    # 'partial_dict' values must be integers or floats
    with pytest.raises(ValueError):
        x = AD(12)
        x.set_params('partial_dict', {'x1': '3'})

def doctesting():
    doctest.testmod()
