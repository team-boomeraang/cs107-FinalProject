# check out https://docs.pytest.org/en/stable/getting-started.html#getstarted
import boomdiff
import pytest
import numpy as np

#creating templates that will be used to test the functions
@pytest.fixture
def basic_cls():
    x = boomdiff.AD(np.pi/4, {'x1': 1.})
    return x

@pytest.fixture
def basic_var():
    x = np.pi/4
    return x

#testing basic functions
def test_init(basic_cls):
    assert basic_cls.func_val == np.pi/4
    assert basic_cls.partial_dict['x1'] == 1
    assert len(basic_cls.partial_dict.keys()) == 1

def test_sin(basic_cls, basic_var):
    f1 = boomdiff.AD.sin(basic_cls)
    f2 = boomdiff.AD.sin(basic_var)

    assert f1.func_val == np.sin(np.pi/4)
    assert f1.partial_dict['x1'] == np.cos(np.pi/4)
    assert f2 == np.sin(np.pi/4)

def test_cos(basic_cls, basic_var):
    f1 = boomdiff.AD.cos(basic_cls)
    f2 = boomdiff.AD.cos(basic_var)

    assert f1.func_val == np.cos(np.pi/4)
    assert f1.partial_dict['x1'] == -np.sin(np.pi/4)
    assert f2 == np.cos(np.pi/4)

def test_tan(basic_cls, basic_var):
    f1 = boomdiff.AD.tan(basic_cls)
    f2 = boomdiff.AD.tan(basic_var)

    assert f1.func_val == np.tan(np.pi/4)
    assert f1.partial_dict['x1'] == 1/(np.cos(np.pi/4)**2)
    assert f2 == np.tan(np.pi/4)

def test_log(basic_cls, basic_var):
    f1 = boomdiff.AD.log(basic_cls)
    f2 = boomdiff.AD.log(basic_var)

    assert f1.func_val == np.log(np.pi/4)
    assert f1.partial_dict['x1'] == 1/(np.pi/4)
    assert f2 == np.log(np.pi/4)
