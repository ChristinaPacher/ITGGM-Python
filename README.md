# ITGGM-Python

This is an implementation of the method *Graphical Granger Causality by Information-Theoretic Criteria* (ITGGM) [1] in Python 3.
This method uses information-theoretic criteria to find networks of Granger-causal relationships between time series.



## Installation

This project comes ready to be installed as a local `pip` module. To install, download the code, then navigate to the project's root folder (the folder containing the `setup.py` file) in the terminal.
Run the following command:

`pip install .`

This will install a pip module called "itgraphicalgranger". Once installed, it can be accessed from any local Python script with `import itgraphicalgranger`.

If you wish to make your own changes to the implementation, you can instead use `pip install -e .` to have the pip module automatically reflect any changes to the code base.

Note: If you encounter any errors during installation, please make sure that your pip version is up to date (use `pip install --upgrade pip`).


## Usage

All functionality is encapsuled within the `GrangerCausality` class (file `grangercausality.py`).
The user interface follows the *estimator* duck-typing interface that is defined in the API conventions of the *scikit-learn* library [2]: All user parameters are defined via optional constructor arguments, and the computation process is started by calling the `fit()` method and passing it the training data as an argument. 

The goal of the computations is to find the network of Granger-causal relationships between the time series in the training data.
Results are exposed in the form of public attributes of the `GrangerCausality` instance, which describe the computed network. 
All result attributes are explained in detail in the doc strings of the `GrangerCausality` class.

Basic usage example:

```
import itgraphicalgranger as itgg

# assumption : the data set is stored in the variable X, which is an array-like object of shape (n_timeseries, n_features)

# perform the Granger causality computations
gc = itgg.GrangerCausality()
gc.fit(X)

# print the result
print(gc.adjacency_matrix_)
```

In order to find the true Granger-causal network, the implementation essentially tries out different possible networks, evaluates how well each one fits the data set, and selects the network with the best fit as the true network. 
For this purpose, the method has two components:

- An information-theoretic criterion that is used to determine how well a given Granger-causal network fits the data set
- An optimization method that determines which Granger-causal networks are tried out

The criterion and optimization method can be selected via the `criterion` and `optimization` keyword arguments in the `GrangerCausality` constructor.
Available options are explained in the doc strings for this class (file `grangercausality.py`).

The behaviour of the chosen optimizer and criterion can be further fine-tuned by passing additional keyword arguments to the constructor. 
The `GrangerCausality` class internally passes them on to the constructors of the classes that implement the chosen optimizer and criterion.
All available options are explained in the doc strings for these classes (files `criterion.py` and `optimization.py`).

For example, to use a genetic algorithm optimization and let it run for 20 generations:

```
import itgraphicalgranger as itgg

# assumption : the training data is stored in the variable X, which is an array-like structure of shape (n_timeseries, n_features)

# perform the Granger causality computations
gc = itgg.GrangerCausality(optimization="genetic", n_generations=20)
gc.fit(X)

# print the result
print(gc.adjacency_matrix_)
```

## Extending the Implementation

The implementation can be extended with additional information-theoretic criteria (a number of possibilities are proposed in [1]) and optimization methods. To tie them in with the existing implementation, it suffices to make some small changes to the `GrangerCausality.fit()` method; the appropriate places are indicated in the comments.

New criterion classes need to follow the duck-typing interface defined by the `CriterionLike` protocol class (file `criterion.py`) in order to be compatible with the existing optimizers.

New optimizers need to follow the *estimator* duck-typing interface described in [2] and expose the results that are expected by the `GrangerCausality` class. The requirements for new optimizers are explained in more detail in the doc string for the `optimization` module (file `optimization.py`).


## References

[1] K. Hlaváčková-Schindler and C. Plant, “Graphical granger causality by information-theoretic
criteria,” in: *Proceedings of the 24th European Conference on Artificial Intelligence*, pp. 1459-1466, 2020.

[2] L. Buitinck, G. Louppe, M. Blondel, F. Pedregosa, A. Mueller, O. Grisel, V. Niculae,
P. Prettenhofer, A. Gramfort, J. Grobler, R. Layton, J. VanderPlas, A. Joly, B. Holt, and
G. Varoquaux, “API design for machine learning software: Experiences from the scikit-learn
project,” in: *ECML PKDD Workshop: Languages for Data Mining and Machine Learning*,
pp. 108–122, 2013.