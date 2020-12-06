# boomdiff: Software for optimization of objective functions

**Authors**: Minhuan Li, Oksana Makarova, Timothy Williamson, Kevin Hare *(Group #19)*


## Introduction
#### Overview
This software package `boomdiff` implements an optimization of a user-supplied or pre-set objective function. Many scientific and social scientific fields rely on probabilistic and statistical methodologies, of which function optimization is an integral part. This function, often called an objective function or a cost function, can take many arbitrary functional forms. Intuitively, the user observes some data and wishes to fit a model to the data subject to some constraint. Deviations from the constraint (i.e. the accuracy) can be viewed as the 'cost'. In minimizing this cost, the user retains the best model.

Two common point estimators in data-driven fields are Maximum Likelihood Estimator (MLE) from Frequentist's view and Maximum A Posterior (MAP) from Bayesian's view. In each case, the optimum points are identified by the stationary condition (first order derivative equals 0) and convexity check by higher order derivatives. Generally, a major branch of modern optimization methods, like Stochastic Gradient Descent (SGD) and Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS), are established via the first or higher order gradient of target function, thus heavily relies on an efficient gradient computation.

Auto Differentiation (AD) is one highly effective method for calculation of the gradient function at some arbitrary point. The method, which balances the efficiency of numeric computation and the precision of symbolic derivatives, is commonly used for optimization applications. Our library solves the optimization problem described above for users via gradient descent, a generalized class of algorithms that exploit the first derivative of a function. The optimization is implemented via forward-mode autodifferentiation, a computing technique that  can effectively and efficiently compute the Jacobian matrix. For the optimization problems we consider, the multidimensional nature of the challenge necessarily lends itself to the use of AD.

While the provided description of *boomdiff* primarily concerns the case where the user has some set of data and wishes to optimize a model against that data, the functionality of this package is not limited to that setting. Rather, *boomdiff* will suffice as a general optimization tool as well for an arbitrary function defined by the user.

#### Gradient-methods supported
- Gradient descent (also known as batch gradient descent)
- Stochastic gradient descent
- Mini-batch gradient descent
- TBD

## Background
#### Optimization
As briefly described in the introduction, nearly all scientific and social scientific fields rely require the optimization of objective (or cost) functions. Even realtively straightforward data analysis techniques such as ordinary least squares regression proceed by specifying a cost function ($(y-\hat{y})^2$ in the canoncial regression task), and then minimizing this cost function. The relative simplicity of this idea, however, belies the true difficulty of estimation of this function minimum. The case of OLS regression has a closed form [solution](https://en.wikipedia.org/wiki/Ordinary_least_squares); more computationally complex models such as neural networks do not. The most general approach for solving these complex sets of equations in lieu of an analytical solution is gradient descent. This iterative methodology operates in the following two general steps:
1. Evaluate the gradient of the objective function while fixing the parameters to be optimized.
2. Update the parameters based on the direction and magnitude of the gradient.

This procedure is continued until the parameters are optimized, which will intuitively occur when the gradient with respect to the parameter is equal to zero. Algorithmically, when the gradient is zero, the magnitude will be as well, and the parameters will terminate updates. Speaking mathematically, this occurs exactly when the stationarity condition has been satisfied (i.e. the first derivative, or gradient) is equal to zero.

This methodology, however, is quite [generalizable and flexible](https://ruder.io/optimizing-gradient-descent/). Two particular problems faced in practice revolve around computational cost and saddle points. When optimization involves significant quantities of data -- as is often the case for MLE or MAP procedures -- the [log-likelihood function](https://en.wikipedia.org/wiki/Likelihood_function), which we hope to minimize, may have hundreds or thousands of data points to be summed over. Doing so can be computationally overwhelming and expensive. To account for this, many techniques randomly select some subset of data, and iteratively optimize over each subset of data. In doing so, the optimization proceeds with smaller magnitude updates, but may more quickly realize the appropriate direction. This technique is commonly known as Stochastic Gradient Descent (SGD) or Mini-Batch Gradient Descent (MBGD) (see [here](https://developers.google.com/machine-learning/crash-course/reducing-loss/stochastic-gradient-descent) for more information). 

Second, for highly complex and non-convex functions, the gradient descent may become 'stuck' at a specific point, or a local minimum, rather than the desired global minimum. Overcoming this challenge once again requires the introduction of randomness. While SGD may be able to eventually overcome this challenge via selecting a subset of data with a gradient with sufficient magnitude to escape the point, a second general approach is to update the parameters using not only the single previous estimate, but many previous estimates. This concept is often referred to as 'momentum' (see [here](https://distill.pub/2017/momentum/) for a more detailed tutorial on the specific mechanics underlying momentum-based gradient descent algorithms).

In all of these methodologies, however, there is the notion of calculation of the gradient. For highly non-convex functions, analytically writing the gradient, or writing a function to evaluate it, may be impossible or highly time-consuming. One approach for solving this issue is to use automatic differentiation, which

#### Automatic Differentiation
At it’s heart, automatic differentiation (AD) seeks to calculate the derivative of some function, and evaluate both the function and the derivative, at a given point by iteratively applying the chain rule to a composition of elementary functions whose derivatives are well-known. This application of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) is essential. By viewing a more complex function as simply the composition of many elementary functions, the calculation of the derivative becomes a series of steps, starting from the innermost function in the forward mode of AD. In each step, the chain rule is applied. For $z(y(x))$:

$$ \frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx} $$

Importantly, because $z(y)$ will be an elementary function (e.g. addition, subtraction, sine, cosine), its derivative can be easily calculated. As we have begun at the innermost derivative, that function $\frac{dy}{dx}$ is known and can be used to iteratively calculate the derivative of the composition. For the simple example above, there are only two elementary operations, but this method can be extended to cover many elementary operations. One only needs to keep track of the derivative of each ‘running’ piece.

The method can be implemented through a graph structure (see below for a simple example), with each node in the graph, which represents a single elementary operation. This computational graph is especially important when the function of interest relies on multiple elements (e.g. $f(x,y) = x^2 + \sin(y)$). One particular advantage of the elementary operations in this method is that each type of operation has a known derivative that can be calculated systematically and efficiently. At each step in the forward mode, the algorithm only needs to maintain the status of the derivative (potentially multiple partial derivatives in the case of multiple elements) as well as the current value of the function.

These storage requirements and iterative nature take advantage of a computer’s ability to store many values and perform many simple operations very quickly. One challenge for computers in calculating complex, symbolic derivatives is that symbolic differentiation may lead to enormous equations and syntactical rules that are highly complex to apply. For humans with a pen and paper, the AD approach may take too much time. The number of calculations, albeit simple, would certainly overwhelm the ability of most people to simply calculate the symbolic derivatives and implement a single evaluation. That concern is severely attenuated by computers. Additionally, AD is superior to the finite differences method of differentiation due to the fact that it is able to keep track of derivatives and functions at the level of machine precision. In scientific applications, this feature is incredibly important, as the sensitive systems measured or engineered will not be successful with only generalities.


## Installation of boomdiff
Below are instructions for the download and installation of the *boomdiff* package. The instructions below are all designed to be run on a command-line interface (Windows, macOS, Linux). Please note that steps (2) and (3) assume that the user has installed [Python](https://www.python.org/). If the user has not installed Python, please see instructions [below](#installation_py)

**Method #1: Installation via PyPi**
Note: this is the preferred method for installing the most up-to-date release of *boomdiff*.
1. In a command-line interface, run the following command:

` pip install boomdiff`

** Method #2: Installation via GitHub**
If installation via PyPi is insufficient, we have provided the instructions below, desgined to download and install the development version of *boomdiff*, available through our GitHub repository.
1. **Download package**: 

    The boomdiff package is available at the GitHub address (https://github.com/team-boomeraang/cs107-FinalProject). To download, navigate on the command line to the desired installation location and run:

    `git clone https://github.com/team-boomeraang/cs107-FinalProject.git`
    
2. **Installation of dependencies**: 

    There are two ways to install dependencies, depending on the package manager used. If using `pip`, then run:

    `pip install -r requirements.txt`
    
    Else, if using Conda or Miniconda for package management, run:
    
    `conda install --file requirements.txt`
    
    Finally, if the desired use is to use `pip` to install the package through Conda or Miniconda, run the following sequence of commands:
    
    `conda install pip`
    
    `which pip`
    
    Ensure that the directory of pip to be used falls below 'anaconda' or 'miniconda' in the directory structure.
    
    `pip install -r requirements.txt`
    
3. **Set-up and install packages**: 
   
    Next, the `setup.py` file must be run to install boomdiff [boomdiff_optimizer not currently available, see Future Features section below]. Navigate to the newly cloned directory and run:

    `python setup.py install`
    
    

<a id='installation_py'></a>
#### Installation of Python
We recommend two possible methods for installation of Python.


1. [Anaconda](https://www.anaconda.com/products/individual) is a package manager that can install Python as well as a variety of common packages (e.g. NumPy, Pandas, scipy). If installed with Anaconda, the instructions below describing `conda install` will be most relevant.


2. Python can also be installed directly (see [here](https://www.python.org/downloads/)). This installation mode will require use of an outside text editor, but may offer more flexibility for installation. If installed via the Python documentation directly, please follow instructions for `pip`.

## Use of *boomdiff*
### Generalized autodifferentiation
In order to take advantage of the ability of *boomdiff*, all objects for optimization must be instantiated as automatic differentiation objects. This section provides a brief introduction and example to instantiation of AD objects. The following section, on objective functions, details how to combine these AD objects into an objective function.

#### Example 0: Instantiate a variable 

As a forward AD mode, we mostly have to instantiate input variables with value and derivative values as a starting point. This process can be easily done with `x1 = AD(*value,**derivate_dict)`. Value is a necessary argument and derivative dictionary are defaulted to be `{'x1': 1}`, you can set as you like. Currently, we only support single input, single scalar output.

The value of the variable will be stored in an attribute `func_val` and the partial derivative dictionary (for now, only one key-value pair) will be stored as `partial_dict`. Then you can use this defined varaible for constructing following complex functions.

```python
>>> from boomdiff import AD
>>> # Step1: Instantiate a variable called x1, with value 10, derivative to be default 1
>>> x1 = AD(10)
>>> # Step2: Demonstrate the information
>>> print(x1)
10 ({'x1': 1})
```

You can name the variable and set the derivative value as you like. There are two options for specifying the variable name and partial derivative. First, the desired name of the variable can be passed directly, and *boomdiff* will use a default seed value of 1 for the partial derivative.
```python
>>> x = AD(10, 'x')
>>> print(x)
10.0 ({'x': 1})
```
Second, the entire partial derivative dictionary can be passed through to the constructor:

```python
>>> # Step1: Instantiate a variable called a, with value 4.9, derivative to be 3.4
>>> a = AD(4.9, {'a': 3.4})

>>> # Step2: Demonstrate the information
>>> print(a)
4.9 ({'a': 3.4})
```
#### Example 1: Constructing an arbitrary function $f(x) = 3x + 4$
We use this example as a demonstration of overloaded primary function in our package. Currently, we support `+`(add), `-`(subtract & negation), `*`(multiply), `/`(divide) and `**`(power)

We will evaluate this function and its derivative at the point $x = 10$. From analytical derivation, we can ascertain that $f(10) = 34$ and $f'(10) = 3$. The tutorial below demonstrates how to calculate this result using *boomdiff*. Before beginning, one important note is that because *boomdiff* only supports scalar functions of one variable, the default behavior is to label the variable `x1`. While the variable name may be any variable name supported by Python, *boomdiff* will report a derivative associated with `x1` unless otherwise specified.

The value of the function will be stored in an attribute `func_val` and the partial derivative dictionary (for now, only one key-value pair) will be stored as `partial_dict`.

```python
>>> # Step 1: Instantiate AD object, as a variable x, at evaluation point (10)
>>> x = AD(10, {'x': 1})

>>> # Step 2: Construct function f(x) = 3x + 4
>>> f = 3*x + 4

>>> # Step 3: Evaluate function and derivative
>>>print(f.func_val)
34
>>> print(f.partial_dict)
{'x': 3}
```

#### Example 2: Static method functions, $f(x) = \sin(x^2)$
We use this example as a demonstration of static method primary functions in our package. Currently, we support `AD.sin()`(sine), `AD.cos()`(cosine), `AD.tan()`(tangent), `AD.log()`(natural log) and `AD.exp()`(exponential). 

We will evaluate this function and its derivative at the value 3. Analytically, the derivative of this function, $f'(x) = 2x \cos(2x)$. Unlike the first function, this one is not so easily evaluated simply by inspection. For *boomdiff*, however, this remains computationally easy. Additionally, in this example, we directly instantiate the function. Note that this functionality assumes that there is only one variable, and thus, $x + y$ will be treated as $x1 + x1$.

```python
>>> # Step 1: Instantiate AD object, as a variable x, at evaluation point (3)
>>> x = AD(10, {'x': 1})
>>> # Step 2: Instantiate function f = sin(x**2)
>>> f = AD.sin(x**2)
>>> # Step 2: Evaluate function and derivative
>>>print(f.func_val)
-0.5063656411097588
>>>print(f.partial_dict)
{'x': 17.24637744575368}
```

#### Example 3: Functions of many variables
```python
>>> from boomdiff import AD
  
>>> # Step1: instantiate multiple variables, x1, x2 here
>>> x1 = AD(10, {'x1': 1})
>>> x2 = AD(4.9, {'x2': 1})
    
>>> # Step2: Define a multivariant function, f = x1+x2
>>> f = x1 + x2
    
>>> # Step3: Show the information of f
>>> f.func_val
14.9   
    
>>> # Now we have multiple partial derivatives in the dictionary
>>> f.partial_dict
{'x1': 1, 'x2': 1}
```

### Optimization of objective functions
From these objects, users may instantiate more complex and arbitrary functions. Using the *boomdiff.optimize* framework, those functions can be efficiently minimzied to machine precision. In terms of structure, *boomdiff* relies on an `Optimizer()` superclass structure, with each gradient descent method taking being a subclass. More details regarding implementation and use can be found in the Implementation section below. In this section, we demonstrate the optimization of two functions: first, an arbirary function, $-x$, and second, an objective function given some data.

#### Example 1: Minimize $f(x) = -x^2$
We will minimize this function via Batch Gradient Descent
```python
>>> from boomdiff import AD, BGD
>>> import numpy as np
>>> x = AD(10, 'x')
>>> f = lambda: -(x **2)
>>> opt = BGD(learning_rate=0.1)
>>> opt.minimize(f, [x], steps=100)
>>> x.evaluate()
0.0 ({'x': 0.0})
```

#### Example 2: Optimization for regression


## Software organization
The software implementation for Version 2.0 of our software is presented below.  Thus, this organization is subject to change in future released versions of *boomdiff*.

```
cs107-FinalProject/
	LICENSE.txt
    setup.py
    requirements.txt
	boomdiff/
		__init__.py
		autodiff.py
        optimizer/
            __init__.py
            gradient_descent.py
	docs/
	    milestone1.ipynb
	    milestone2.ipynb
	    documentation.md
	tests/
		__init__.py
		test_unit.py
        test_suite.py
```

There are two main modules, both of which sit within the *boomdiff* package structure. The main functionality is encapsulated in the `optimize` subpackage, as this provides full access to our suite of optimization tools. Those tools, which are described in detail above and through the details on implementation below,, will not be reviewed here.  Additonally, the implemented module, `autodiff.py` provides support for automatic differentiation of a scalar functions of a many variables of elementary functions and operations (detailed in the 'Implementation' section below). 

Our test suite lives in the `tests` directory of the main directory structure. These unit tests currently cover 99% of the *boomdiff* functionality. Additionally, each of the overloaded operations and static methods implemented in the AD class have docstring tests to support additional testing and usability. Our repository is currently being tracked by Travis CI, integrated with CodeCov, to provide support for continuous integration of our library.
 
As described above, our package is distributed through two separate avenues. First, the package is installable via PyPi. Second, our packaging is distributed via a clone of this GitHub repository. We’re not planning to use the framework since our package is not going to be a web application. Also, the package should be basic enough and contain all required documentation. 

## Implementation

#### autodiff
- Core data structures:
    - *boomdiff*'s automatic differentiation' is primarily implemented through an object-oriented class, AD. This class is essentially the object or function to be evaluated and differentiated.
    - While the actual object must be separately called (e.g. AD(2)), this can be wrapped into a single function line, as the class will be returned and operated on by other methods of the AD class.
- Core classes:
    - AD class: `AD(eval_pt, der_dict)`
        - Constructor arguments:
            - `eval_pt`: Point to evaluate function at. Expected behavior as float or int; required.
            - `der_dict`: Optional. If supplied, must be either a string or dictionary. If string, the default seed vector will be set to 1. If specifying multiple variables in constructor, user must pass a dictionary, though this can be the arbitrary dictionary with each partial derivative assumed to be the default seed vector. Default behavior (if not passed at all) will be to assume that function is one variable (i.e. x1) and that its derivative is 1. Currently does not support multiple variables, though variable name passed to `der_dict` could, in theory, be any string.
        - Attributes:
            - `func_val`: value of function as a float; this will initally be the `eval_pt` passed at class initialization.
            - `partial_dict`: dictionary. This dictionary will store the partial derivatives. Each key corresponds to the variable (in a multiple variable function). Note that the multiple variable functionality has not been fully implemented and tested.
         - Elementary functions:
             - These methods are executed in two manners, depending on the function type. Functions which have Python built-ins have been overloaded. Other elementary functions have been implemented as class static methods. Please see below for greater details.
             - Overloaded:
                 - Addition (`+`)
                 - Subtraction (`-`)
                 - Multiplication (`*`)
                 - Power (`**`)
                 - Division (`/`)
                 - Negation (`-x`), where x is an instance of AD.
                 
             - Static methods:
                 - Sine (`AD.sin()`)
                 - Cosine (`AD.cos()`)
                 - Tangent (`AD.tan()`)
                 - Natural logarithm (`AD.log()`)
                 - Exponential (base e) (`AD.exp()`)
                 
        - Other methods:
            - `set_params(att, val)`: Sets parameter values outside of initializer. Note that if this is done after operations have been performed, may not be backwards compatible; preferable to set parameters with constructor. `att` argument must be one of `func_val` or `partial_dict` and must mirror behavior of `eval_pt` and `der_dict` arguments to constructor.
            - `name()`: This function returns the names of all variables in a given AD object. This is primarily used in *boomdiff.optimize*, but may be of interest to the user as well.
            - `value()`:  Returns the function value for an AD object. Equivalent to calling `x.func_val` attribute, but returnable.
            - `ders()`: Returns partial derivative dictionary. Similar to `name()` and `value()` in terms of intended functionaliy.
            - `evaluate()`: Combines `value()` and `ders()`, returning both the function value and partial derivative dictionary as a tupe (in that order).
                 
- External dependencies:
    - [NumPy](https://numpy.org/)
    - [itertools](https://docs.python.org/3/library/itertools.html)