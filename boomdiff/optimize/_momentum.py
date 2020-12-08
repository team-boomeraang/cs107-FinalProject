"""
Interface for Momentum accelerated Gradient Descent
"""

__all__ = ['Momentum']

import warnings

import numpy as np

from boomdiff.autodiff import AD
from ._optimizer import Optimizer

class Momentum(Optimizer):
    """
    Momentum accelerated Gradient Descent subclass 
    Adding a fraction gamma of the update vector of the past time step to the current update vector:
    v_t = gamma * v_{t-1} + lr * gradient
    theta = theta - v_t

    Usage:
    >>> a = AD(2, 'a')
    >>> loss = lambda: a**2
    >>> # for a installed package, the api is boomdiff.optimize.Momentum() 
    >>> opt_mom = Momentum(learning_rate=0.1, gamma=0.9)
    >>> opt_mom.minimize(loss, [a])
    >>> print(a.round(2))
    -0.01 ({'a': 1.0})
    """

    def __init__(self, learning_rate=0.1, gamma=0.9):
        """
        Parameters
        ----------
        learning_rate: int or float

        gamma: int or float
            momentum term
        """
        super(Momentum, self).__init__(learning_rate)
        self.gamma = 0.9

    def _apply_gradient(self, loss, var_list, grad_dict):
        """
        Apply the gradient to update variables, with Momentum algorithm
        v_t = gamma * v_{t-1} + lr * gradient
        theta = theta - v_t
        """
        if not hasattr(self,'last_update'):
            # If no history before, initialize one
            new_update_dict = {}
            for var in var_list:
                new_update_dict[var.name()[0]] = 0.
            self.last_update = new_update_dict

        assert isinstance(self.last_update, dict), "last update should be a dictionary!"

        new_update_dict = {}
        for var in var_list:
            grad = grad_dict[var.name()[0]]
            if abs(grad) > abs(var.func_val) * 10**6:
                warnings.warn("Gradient is too large: potential numerical instability")
            v_tm1 = self.last_update[var.name()[0]]
            v_t = self.gamma * v_tm1 + self.lr * grad
            
            # update the variable value
            var.func_val -= v_t

            # update the last_update dictionary
            new_update_dict[var.name()[0]] = v_t

        self.last_update = new_update_dict




        
