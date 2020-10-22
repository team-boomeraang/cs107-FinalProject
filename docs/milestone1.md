## Introduction
This software package `Boomeraang_optimizer` implements an optimization of a user-supplied or pre-set objective function. Many scientific and social scientific fields rely on probabilistic and statistical methodologies that fundamentally concern the optimization of some function, often called an objective function or a cost function. Intuitively, these methods seek to find the best fit of some function given observed data. Two common point estimators in data-driven fields are Maximum Likelihood Estimator (MLE) from Frequentist's view and Maximum A Posterior (MAP) from Bayesian's view. In each case, the optimum points are identified by the stationary condition (first order derivative equals 0) and convexity check by higher order derivatives. Generally, a major branch of modern optimization methods, like Stochastic Gradient Descent (SGD) and Broyden–Fletcher–Goldfarb–Shanno algorithm(BFGS), are established via the first or higher order gradient of target function, thus heavily relies on an efficient gradient computation.

Auto Differentiation (AD) is one highly effective method for calculation of the gradient function at some point. The method, which balances the efficiency of numeric computation and the precision of symbolic derivatives, is commonly used for optimization applications. Our library solves the optimization problem described above via gradient descent, which is implemented on top of a forward-mode autodifferentiation. The advance of this method is that AD can effectively and efficiently compute the Jacobian matrix. For the optimization problems we consider, the multidimensional nature of the challenge necessarily lends itself to the use of AD.


## Background
At it’s heart, automatic differentiation (AD) seeks to calculate the derivative of some function, and evaluate both the function and the derivative, at a given point by iteratively applying the chain rule to a composition of elementary functions whose derivatives are well-known. This application of the chain rule [https://en.wikipedia.org/wiki/Chain_rule] is essential. By viewing a more complex function as simply the composition of many elementary functions, the calculation of the derivative becomes a series of steps, starting from the innermost function in the forward mode of AD. In each step, the chain rule is applied. For $z(y(x))$:

$$ \frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx} $$

Importantly, because $z(y)$ will be an elementary function (e.g. addition, subtraction, sine, cosine), its derivative can be easily calculated. As we have begun at the innermost derivative, that function $\frac{dy}{dx}$ is known and can be used to iteratively calculate the derivative of the composition. For the simple example above, there are only two elementary operations, but this method can be extended to cover many elementary operations. One only needs to keep track of the derivative of each ‘running’ piece.

The method can be implemented through a graph structure (see below for a simple example), with each node in the graph, which represents a single elementary operation. This computational graph is especially important when the function of interest relies on multiple elements (e.g. $f(x,y) = x^2 + \sin(y)$). One particular advantage of the elementary operations in this method is that each type of operation has a known derivative that can be calculated systematically and efficiently. At each step in the forward mode, the algorithm only needs to maintain the status of the derivative (potentially multiple partial derivatives in the case of multiple elements) as well as the current value of the function.

These storage requirements and iterative nature take advantage of a computer’s ability to store many values and perform many simple operations very quickly. One challenge for computers in calculating complex, symbolic derivatives is that symbolic differentiation may lead to enormous equations and syntactical rules that are highly complex to apply. For humans with a pen and paper, the AD approach may take too much time. The number of calculations, albeit simple, would certainly overwhelm the ability of most people to simply calculate the symbolic derivatives and implement a single evaluation. That concern is severely attenuated by computers. Additionally, AD is superior to the finite differences method of differentiation due to the fact that it is able to keep track of derivatives and functions at the level of machine precision. In scientific applications, this feature is incredibly important, as the sensitive systems measured or engineered will not be successful with only generalities.



## How to use *boom-diff* and *Boomeraang-optimizer*

*What should they import? *
They should import the entire autodiff module.


*How can they instantiate AD objects? *
Users will be able to instantiate objects in two ways:

1. A user will be able to implement a function and a set of points where it should be implemented. The autodiff module will then differentiate the function at the point supplied by the user.

2. Users can also instantiate classes directly as an AD class object. This will allow the user to iteratively build the autodifferentiation.

LinearRegression will take in its fit method arrays X, y and will store the coefficients  of the linear model in its coef_ member:
```
 >>> from boom-diff import autodiff
 >>> func = "x^2" #lambda x: x**2 
 >>> ad = autodiff.AD(func, {x:2})
 >>> ad.diff
     (4, 4)
```


## Software organization

What will the directory structure look like?
The implementation for our software will follow the structure laid out below:
```
cs107-FinalProject/
	boomeraang-optimizer/
		__init__.py
		[implement optimization].py
		setup.py
	boom-diff/
		__init__.py
		autodiff.py
		setup.py
					
	docs/
	    milestone1.md
	    documentation.md
	tests/
		unit_tests.py
```

 
- *What modules do you plan on including? What is their basic functionality?*
 The code will have two modules: first will implement automatic differentiation (boom-diff) and the second will perform optimization using the AD.
 
- *Where will your test suite live? Will you use TravisCI? CodeCov?*
 We are planning to use TravisCI and CodeCov to make sure that our code can be used by others. The test suites will be on GitHub, to ensure compatibility of different versions.

- *How will you distribute your package (e.g. PyPI)?*
 We should only need a simple package that could be installed using pip. In order to create one, we are planning to follow the tutorial from this website: https://python-packaging.readthedocs.io/en/latest/index.html.
  
- *How will you package your software? Will you use a framework? If so, which one and why? If not, why not?*
 We’re not planning to use the framework since our package is not going to be a web application. Also, the package should be basic enough and contain all required documentation. 


## Implementation

- Core data structures:
    - Class for objective function
    - Create function/AD class. With operation overloading, this will be used to 'recreate' the objective function from the inside out
    - Potentially class for specific methods (could also use a class method to construct objective function class)
    - Do we want to consider differences for scalars and vectors?
- Classes
    - ObjectiveFunction
        - Methods
            - set_function(): allows user to set objective function
                - Validation that function
            - optimize(method={'graddesc', etc): run optimization; return value of optimization
            - from_X(): where 'X' represents some preset function (may have multiple)
        - Attributes:
            - data
            - functional_form
    - DiffFunc()
        - Attributes:
            - point: location for differentiation
            - deriv: derivative at point
            - (do we need this?) trace_table: method for storing trace table...list of list?
        - Methods:
            - Overloading for common methods
            - Additional functions for implementation of elementary operations (do we want to restrict?)
- External dependencies
    - numpy
    - scipy
    - matplotlib
- Elementary functions:
    - Will rely on numpy execution for calling results of functions.
    - Define set of Python functions to handle basic derivatives
 



