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
        pass # TODO

    def __rsub__(self, other):
        pass # TODO

    def __mul__(self, other):
        pass # TODO

    def __rmul__(self, other):
        pass # TODO

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
