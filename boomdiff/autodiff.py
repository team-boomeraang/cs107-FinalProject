import numpy as np

class AD():

    def __init__(self, eval_pt, der_dict={'x1':1}):
        """Initializes class structure

        Parameters
        ----------
        eval_pt : float
            Value of the current function/variable
        der_dict : dict
            derivative value dictionary of all variables
        
        Returns
        -------
        None.
        
        Examples
        --------
        >>> x1 = AD(3.6, {'x1':1})
        >>> x1.func_val
        3.6
        >>> x1.partial_dict
        {'x1': 1}
        """
        
        # Set function string
        self.func_string = ''
        self.func_val = eval_pt
        
        # Set partial derivative dictionary
        # Will assume form of x_1, ..., x_n
        self.partial_dict = der_dict
        
    def set_params(self, params):
        pass # TODO

    def __add__(self, other):
        pass # TODO

    def __radd__(self, other):
        pass # TODO

    def __sub__(self, other):
        """Overload subtraction operation '-'

        Parameters
        ----------
        other : AD class instance or float
            Elements to be subtracted from self. Can be a AD class instance, which 
            will update function value and partial derivative dictionary; Or a con-
            -stant, which will only update function value.

        Returns
        -------
        A new AD class instance with updated information

        Examples
        --------
        >>> x1 = AD(3.6, {'x1': 1})
        >>> print(x1.func_val, x1.partial_dict)
        3.6 {'x1': 1}
        >>> f1 = x1 - 10
        >>> print(f1.func_val, f1.partial_dict)
        -6.4 {'x1': 1}
        >>> x2 = AD(2.0, {'x1': 3.4})
        >>> f2 = x1 - x2
        >>> print(f2.func_val, f2.partial_dict)
        1.6 {'x1': -2.4}
        """
        try: 
            # First try as other is an AD class instance
            # Give the variable list for self object
            self_var_keys = list(self.partial_dict.keys())
            # Give the variable list for other object
            other_var_keys = list(other.partial_dict.keys())
            # At this moment, we assume all have one variable, need to be fixed, TODO
            new_der_dict = {self_var_keys[0]: self.partial_dict[self_var_keys[0]] - other.partial_dict[self_var_keys[0]]}
            return AD(self.func_val-other.func_val, new_der_dict)
        except AttributeError:
            # If other is not an AD class instance, treat as a constant
            return AD(self.func_val-other, self.partial_dict)

    def __rsub__(self, other):
        """Overload to make sure commutativity of subtraction '-'

        Parameters
        ----------
        other : AD class instance or float
            Elements to be subtracted on the left, from self. Can be a AD class instance, which 
            will update function value and partial derivative dictionary; Or a con-
            -stant, which will only update function value.
        
        Returns
        -------
        A new AD class instance with updated information

        Examples
        --------
        >>> x1 = AD(3.6, {'x1': 1})
        >>> f1 = 10 - x1
        >>> print(f1.func_val, f1.partial_dict)
        6.4 {'x1': -1}
        """
        if isinstance(other, AD):
            # First try as other is an AD class instance
            return other.__sub__(self)
        else:
            # If other is not an AD class instance, treat as a constant
            self_var_keys = list(self.partial_dict.keys())
            # At this moment, we assume all have one variable, need to be fixed, TODO
            new_der_dict = {self_var_keys[0]: -self.partial_dict[self_var_keys[0]]} 
            return AD(other-self.func_val, new_der_dict)

    def __mul__(self, other):
        """Overload multiplication operation '*'

        Parameters
        ----------
        other : AD class instance or float
            Elements to be multiplied to self. Can be a AD class instance, which 
            will update function value and partial derivative dictionary; Or a con-
            -stant, which will only update function value.

        Returns
        -------
        A new AD class instance with updated information

        Examples
        --------
        >>> x1 = AD(1, {'x1': 1})
        >>> f1 = x1*10.0
        >>> print(f1.func_val, f1.partial_dict)
        10.0 {'x1': 10.0}
        >>> x2 = x1*3.4 - 1.4
        >>> print(x2.func_val, x2.partial_dict)
        2.0 {'x1': 3.4}
        >>> f2 = f1 * x2
        >>> print(f2.func_val, f2.partial_dict)
        20.0 {'x1': 54.0}
        """
        try: 
            # First try as other is an AD class instance
            # Give the variable list for self object
            self_var_keys = list(self.partial_dict.keys())
            # Give the variable list for other object
            other_var_keys = list(other.partial_dict.keys())
            # At this moment, we assume all have one variable, need to be fixed, TODO
            new_der_dict = {self_var_keys[0]: self.partial_dict[self_var_keys[0]]*other.func_val + self.func_val*other.partial_dict[self_var_keys[0]]}
            return AD(self.func_val*other.func_val, new_der_dict)
        except AttributeError:
            # If other is not an AD class instance, treat as a constant
            self_var_keys = list(self.partial_dict.keys())
            # At this moment, we assume all have one variable, need to be fixed, TODO
            new_der_dict = {self_var_keys[0]: self.partial_dict[self_var_keys[0]]*other}
            return AD(self.func_val*other, new_der_dict)

    def __rmul__(self, other):
        """Overload to make sure commutativity of operation '*'

        Parameters
        ----------
        other : AD class instance or float
            Elements to be multiplied to self. Can be a AD class instance, which 
            will update function value and partial derivative dictionary; Or a con-
            -stant, which will only update function value.

        Returns
        -------
        A new AD class instance with updated information

        Examples
        --------
        >>> x1 = AD(1, {'x1': 1})
        >>> f1 = 10.0*x1
        >>> print(f1.func_val, f1.partial_dict)
        10.0 {'x1': 10.0}
        >>> x2 = 3.4*x1 - 1.4
        >>> print(x2.func_val, x2.partial_dict)
        2.0 {'x1': 3.4}
        >>> f2 = f1 * x2
        >>> print(f2.func_val, f2.partial_dict)
        20.0 {'x1': 54.0}
        """
        if isinstance(other, AD):
            # First try as other is an AD class instance
            return other.__mul__(self)
        else:
            # if other is not an AD class instance, treat as a constant
            self_var_keys = list(self.partial_dict.keys())
            # At this moment, we assume all have one variable, need to be fixed, TODO
            new_der_dict = {self_var_keys[0]: other*self.partial_dict[self_var_keys[0]]}
            return AD(other*self.func_val, new_der_dict)

    def __truediv__(self, other):
        pass # TODO

    def __rtruediv__(self, other):
        pass # TODO 

    def __pow__(self, other):
        pass # TODO

    def __rpow__(self, other):
        pass # TODO

    @staticmethod
    def sin(x):
        pass # TODO

    @staticmethod
    def cos(x):
        pass # TODO

    @staticmethod
    def tan(x):
        pass # TODO

    @staticmethod
    def log(x):
        pass # TODO


if __name__ == '__main__':
    import doctest
    doctest.testmod()
