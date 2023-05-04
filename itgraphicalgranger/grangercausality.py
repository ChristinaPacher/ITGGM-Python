import numpy as np
from typing import Union, Any, Optional
from numpy.typing import ArrayLike

from . import optimization as optim
from . import criterion as crit

class GrangerCausality(object):
    """User interface and wrapper around the internal logic.

    This class is responsible for accepting all user input and for coordinating the interaction between optimizer and criterion.
    User parameters should only be set via constructor arguments. If parameters are changed manually at any time after instantiation, there is no guarantee that the object's internal state will reflect these changes.

    Parameters
    ----------
    criterionname : str
        The name of the used information-theoretic criterion. Can be "SC" for the stochastic complexity, or "custom" if a user-defined criterion is used.
    criterion: object
        Instance of a class that implements the information-theoretic criterion.
    optimization: str
        The name of the used optimization method. Can be "genetic" for a genetic algorithm, or "exhaustive" for an exhaustive search.
    optimizer: object
        Instance of a class that implements the optimization method.
    
    Attributes
    ----------
    adjacency_matrix_: ndarray of shape (n_timeseries, n_timeseries)
        Binary matrix containing a representation of the Granger-causal relationships between the time series in the input data. An entry of 1 in row i and column j means that the j-th time series has a Granger-causal influence on the i-th time series. (Note that this is the transpose of the adjacency matrix of a directed graph.)
    beta_matrix_: ndarray of shape (n_timeseries, n_timeseries)
        If criterion supports the computation of connection strength values, beta_matrix_ has the same nonzero pattern as adjacency_matrix_, and the real values in beta_matrix_ indicate the strength of each Granger-causal connection in adjacency_matrix_. Otherwise, the contents of beta_matrix_ are meaningless.
    adjacency_matrix_scores_: ndarray of shape (n_timeseries,)
        Contains one score value for each row in adjacency_matrix_ to indicate how well that row fits the input data, according to criterion. Lower scores mean a better fit.
    """

    def __init__(self, criterion: Union[str, crit.CriterionLike] = "SC", optimization: str = "genetic", **kwargs: Optional[Any]) -> None:
        """Create an instance of the GrangerCausality class.

        Parameters
        ----------
        criterion : str or object, default="SC"
            Information-theoretic criterion:

            "SC": use the stochastic complexity criterion (implemented in `itgraphicalgranger.criterion.StochasticComplexity`).

            If criterion is an object that is not a string, then this object is used for the criterion computations in place of any of the predefined options. It has to follow the duck-typing interface for criterion classes (as defined by the protocol class `itgraphicalgranger.criterion.CriterionLike`).

        optimization : str, default "genetic"
            Optimization method:

            "genetic": use a genetic algorithm (implemented in `itgraphicalgranger.optimization.GeneticAlgorithm`).
            
            "exhaustive": use an exhaustive search (implemented in `itgraphicalgranger.optimization.ExhaustiveSearch`).
            
        **kwargs:
            Constructor keyword arguments for the chosen criterion and optimizer classes.
        """

        # save the optimization method
        self.optimization = optimization
        
        # save the criterion
        if type(criterion) == str: # "criterion" is the name of a provided criterion
            self.criterion = None
            self.criterionname = criterion
        else: # assuming that the provided object is a user-defined criterion implementation
            self.criterion = criterion
            self.criterionname = "custom"
            
        # save the keyword arguments so we can use them for the instantiations later on
        self.user_kwargs = kwargs
    
    def fit(self, X: ArrayLike) -> "GrangerCausality":
        """Compute the network of Granger-causal relationships between the time series in X.

        Parameters
        ----------
        X : array-like of shape (n_timeseries, n_features)
            Input time series.

        Returns
        -------
        self: object
            Fitted estimator.
        """
        
        # if the user did not provide a criterion instance, create one
        if self.criterion is None:
            if self.criterionname == "SC":
                criterion = crit.StochasticComplexity(X, **self.user_kwargs)
            # additional criteria can be added here
            else:
                raise ValueError("Invalid criterion name: " + self.criterionname)
            self.criterion = criterion
        
        # allocate the optimizer
        if self.optimization == "genetic":
            optimizer = optim.GeneticAlgorithm(self.criterion, **self.user_kwargs) 
        elif self.optimization == "exhaustive":
            optimizer = optim.ExhaustiveSearch(self.criterion, **self.user_kwargs)
        # additional optimizers can be added here
        else:
            raise ValueError("Unknown optimization method: " + self.optimization)
        self.optimizer = optimizer
        
        # make sure the data is in the form of a numpy array
        X = np.asarray(X)

        # run the optimizer
        optimizer.fit(X)

        # save the result (with trailing underscores to indicate computed values)
        self.adjacency_matrix_ = optimizer.adjacency_matrix_
        self.beta_matrix_ = optimizer.beta_matrix_
        self.adjacency_matrix_scores_ = optimizer.adjacency_matrix_scores_

        # scikit-like convention: fit() method should return self
        return self
            