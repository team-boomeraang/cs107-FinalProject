"""
Interface for BGD and SGD opimizer subclass

Batch gradient descent (BGD)
Stochastic Gradient Descent (SGD)

for more details see: https://ruder.io/optimizing-gradient-descent/
and https://www.geeksforgeeks.org/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/?ref=lbp
"""

__all__ = ['SGD', 'BGD']

import numpy as np

from autodiff import AD
from optimizer import Optimizer

class BGD(Optimizer):
    """
    Batch gradient descent (BGD) optimizer subclass
    inefficient but precise algorithm that makes evaluations for the whole dataset.

    Usage:
    ```python
    >>> # Instantiate an BGD optimizer
    >>> opt = BGD(learning_rate=0.1)

    >>> # 'loss' should be a callable that takes no arguments and output an AD instance
    >>> # should only use operations supported by AD class
    >>> loss = lambda: var1**2 + var2**2

    >>> # initialize the variables for the objective function
    >>> # make sure the name string in the dict is corresponding with the variable name
    >>> var1 = AD(1, {'var1': 1})
    >>> var2 = AD(2, {'var2': 1})

    >>> # Call step method, update the variables for one step, to minimize the loss value
    >>> # var_list are the variable lists that you want to update
    >>> # It can be part of the variables in loss callable.
    >>> # The step method will update the variables defined before
    >>> opt.step(loss, var_list=[var1, var2])

    >>> # The var1 and var2 will be updated, - learning_rate * grad
    >>> var1
    0.8 ({'var1': 1})
    >>> var2
    1.6 ({'var2': 1})

    >>> # Or you can call minimize method, to update multiple steps
    >>> # With user-specified learning_rate series
    >>> opt.minimize(loss, [var1, var2], steps=100)

    >>> # This will give some final optimization results if converged
    >>> print(var1, var2)
    5.879867611500617e-22 ({'var1': 1}) 1.3066372470001377e-22 ({'var2': 1})
    ```
    """

    def __init__(self, learning_rate=0.1, **kwargs):

        super().__init__(learning_rate)

    def _apply_gradient(self, loss, var_list, grad_dict):
        """
        Apply the gradient to update variables, with BGD algorithm
        BGD algorithm idea: evaluate gradient of the loss function at all points.
        x_i = x_i - learning_rate * grad(loss(x_i))
        """
        for var in var_list:
            var.func_val -= self.lr * grad_dict[var.name()[0]]

class SGD(Optimizer):
    """
    SGD optimizer subclass

    Usage:
    ```python
    >>> # Instantiate an SGD optimizer
    >>> opt = boomdiff.optimize.SGD(learning_rate=0.1)

    >>> # 'loss' should be a callable that takes no arguments and output an AD instance
    >>> # should only use operations supported by AD class
    >>> loss = lambda: var1**2 + var2**2

    >>> # initialize the variables for the objective function
    >>> # make sure the name string in the dict is corresponding with the variable name
    >>> var1 = AD(1, {'var1': 1})
    >>> var2 = AD(2, {'var2': 1})

    >>> # Call step method, update the variables for one step, to minimize the loss value
    >>> # var_list are the variable lists that you want to update
    >>> # It can be part of the variables in loss callable.
    >>> # The step method will update the variables defined before
    >>> opt.step(loss, var_list=[var1, var2])

    >>> # The var1 and var2 will be updated, - learning_rate * grad
    >>> var1
    0.8 ({'var1': 1})
    >>> var2
    1.6 ({'var2': 1})

    >>> # Or you can call minimize method, to update multiple steps
    >>> # With user-specified learning_rate series
    >>> opt.miminize(loss, var_list[var1, var2], learning_rates=np.linspace(0.1,0.01,100), steps=100)

    >>> # This will give some final optimization results if converged
    >>> var1
    0.0 ({'var1': 1})
    >>> var2
    0.0 ({'var2': 1})
    ```
    """

    def __init__(self, learning_rate=0.1, **kwargs):

        super().__init__(learning_rate)

        #TODO, other SGD-specific initializations, like momentum
        #...


    def _apply_gradient(self, loss, var_list, grad_dict):
        """Apply the gradient to update variables, with SGD algorithm
        SGD algorithm idea: pick a random single point with each iteration
        to compute the gradient.

        """
        # TODO, use SGD algorithms, update the variables in the var_list
        # Simple demo codes are:
        # >>> for var in var_list:
        # >>>       var.func_val -= self.lr * grad_dict[var.name()[0]]
        # self.lr is learning rate
        # var.name() is a new method added to AD class, return the keys list in the partial_dict, which is the variable name
        # The simple demo codes should already work to gradient descent the loss fucntion, by updating the vars
        # Need to add some codes to make this method more robust





# opt = BGD(learning_rate=0.2)
# var1 = AD(9, {'var1': 1})
# var2 = AD(2, {'var2': 1})
# loss = lambda: var1**2 + var2**2
# opt.minimize(loss, [var1, var2], steps=100)
#
# # opt.step(loss, [var1])
# # print(var1, var2)
# # opt.step(loss, [var1])
# # print(var1, var2)
# # opt.step(loss, [var1])
# print(var1, var2)
