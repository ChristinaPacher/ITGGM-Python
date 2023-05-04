"""This module contains implementations of information-theoretic criteria that can be used to determine how well a network of Granger-causal relationships fits a data set.

In addition, this module contains a protocol class that defines the duck-typing interface for criterion classes.

The following criteria are provided: stochastic complexity.

Classes
-------
CriterionLike
    Type protocol class that defines the duck-typing interface for all criterion classes (can be used for type annotations).
StochasticComplexity
    Implementation of the stochastic complexity criterion.
"""

import numpy as np
import scipy
import math
from typing import Protocol, Tuple, Optional, Any
from numpy.typing import ArrayLike

class CriterionLike(Protocol):
    """Typing protocol specifying the duck-typing interface for all information-theoretic criterion classes.
    
    This class can be used for type annotations.
    Any class that implements the method defined in this duck-typing interface can be used as criterion.
    """

    def score(self, i: int, assignment: ArrayLike) -> Tuple[float, ArrayLike]:
        """Compute a score indicating how well an assignment of Granger-causal relationships fits a data set.

        Lower scores indicate a better fit.

        Parameters
        ----------
        i : int
            Index of the target time series in the input data set.
        assignment : array-like of shape (n_timeseries,)
            Description of the Granger-causal influences on the target time series. Entries are binary. An entry of 1 at index j indicates that the j-th time series is Granger-causal for the target series.

        Returns
        -------
        criterion : float
            The criterion score.
        connections : array-like of shape (n_timeseries,)
            Array of real values with the same nonzero structure as assignment. The real values in connections indicate the strength of each Granger-causal connection in assignment.
            If the criterion does not provide a means to compute connection strength values, this return value may instead contain meaningless values. This choice must be documented by the criterion class.

        Raises
        ------
        ValueError
            If assignment is invalid for this criterion, i.e. it is not possible to compute a score for the given assignment vector.
        """

        ... 

class StochasticComplexity(object):
    """Implementation of the stochastic complexity criterion.

    Parameters
    ----------
    X : array-like of shape (n_timeseries, n_measurements)
        Input time series.
    lag : int, default=3
        Number of previous time points to consider for the Granger causality computations.
    rcond : float or None, default=None
        Cutoff ratio for small singular values in the least squares solver that is used to compute the regression coefficients (as defined by the rcond parameter in `numpy.linalg.lstsq`, or the cond parameter in `scipy.linalg.lstsq`). All singular values of the system matrix that are smaller than rcond time its largest singular value are treated as zeros.
    precompute_matrix_blocks : bool, default=True
        If True, building blocks for intermediate results are precomputed and stored. This increases the memory requirements, but can considerably reduce the runtime.
    save_lstsq_norms : bool, default=False
        If True, the solution norm and residual norm of the most recent least squares solution will be exposed as attributes of self. This option exists to allow the user to inspect the behaviour of the least squares solver, but should not be used as part of regular computations as it causes additional computation overhead.
    use_lapack_gelsy : bool, default=False
        Determines which function is used for solving the least squares system to compute the regression coefficients. By default, `numpy.linalg.lstsq` is used, which uses an SVD-based solution method. If use_lapack_gelsy is True, `scipy.linalg.lstsq` is used instead with its lapack_driver parameter set to 'gelsy', which uses a QR-decomposition-based solution method.
    n_timeseries: int
        Number of time series in the input data.
    n_measurements: int
        Number of features in each time series in the input data.
    
    Attributes
    ----------
    lstsq_solution_norm_ : float
        1-norm of the solution of the most recent least squares computation. Only available if save_lstsq_norms was set to True.
    lstsq_residual_norm_ : float
        1-norm of the residual of the most recent least squares computation. Only available if save_lstsq_norms was set to True.
    """

    def __init__(self, X: ArrayLike, lag: int = 3, rcond: float = None, precompute_matrix_blocks: bool = True, save_lstsq_norms: bool = False, use_lapack_gelsy: bool = False, **_: Optional[Any]) -> None:
        """Create an instance of the StochasticComplexity class.

        Parameters
        ----------
        X : array-like of shape (n_timeseries, n_measurements)
            Input time series.
        lag : int, default=3
            Number of previous time points to consider for the Granger causality computations.
        rcond : float or None, default=None
            Cutoff ratio for small singular values in the least squares solver that is used to compute the regression coefficients (as defined by the rcond parameter in `numpy.linalg.lstsq`, or the cond parameter in `scipy.linalg.lstsq`). All singular values of the system matrix that are smaller than rcond time its largest singular value are treated as zeros.
        precompute_matrix_blocks : bool, default=True
            If True, building blocks for intermediate results are precomputed and stored. This increases the memory requirements, but can considerably reduce the runtime.
        save_lstsq_norms : bool, default=False
            If True, the solution norm and residual norm of the most recent least squares solution will be exposed as attributes of self. This option exists to allow the user to inspect the behaviour of the least squares solver, but should not be used as part of regular computations as it causes additional computation overhead.
        use_lapack_gelsy : bool, default=False
            Determines which function is used for solving the least squares system to compute the regression coefficients. By default, `numpy.linalg.lstsq` is used, which uses an SVD-based solution method. If use_lapack_gelsy is True, `scipy.linalg.lstsq` is used instead with its lapack_driver parameter set to 'gelsy', which uses a QR-decomposition-based solution method.
        """
        
        # save user-provided input
        self.X = np.asarray(X)
        self.lag = lag
        self.rcond = rcond
        self.precompute_matrix_blocks = precompute_matrix_blocks
        self.save_lstsq_norms = save_lstsq_norms
        self.use_lapack_gelsy = use_lapack_gelsy
        
        self.n_timeseries, self.n_measurements = X.shape

        # some renaming of variables to shorten the code
        d = self.lag 
        n = self.n_measurements

        # sanity check: do our computations work with this data?
        if n/d <= self.n_timeseries + 1:
            raise ValueError("Time series are too short - the graphical Granger model is inconsistent")
        if n <= d + self.n_timeseries:
            # usually this should not happen if the first condition does not trigger, this is just to make sure we catch all edge cases
            raise ValueError("Time series are too short - cannot compute the stochastic criterion")

        # create the "building blocks" of the design matrix
        self._designmat_blocks = np.array([[[row[startval-offset] for offset in range(0,d)] for startval in range(d-1, n-1)] for row in X]) # array conversion to allow boolean indexing later on

        ## for future reference: the triple nested list comprehension is equivalent to the following code:
        #designmat_blocks = []
        #for row in X:
        #    block = []
        #    for startval in range(d-1, n-1):
        #        blockrow = []
        #        for offset in range(0,d):
        #            blockrow.append(row[startval-offset])
        #        block.append(blockrow)
        #    designmat_blocks.append(block)

        self._designmat_blocks_transposed = np.array([np.transpose(block) for block in self._designmat_blocks])

        # precompute the "building blocks" of the system matrix for beta_hat to avoid costly matrix-matrix multiplications later on
        if precompute_matrix_blocks:
            system_matrix_blocks = {}
            p = self.n_timeseries
            blocks_T = self._designmat_blocks_transposed
            blocks = self._designmat_blocks
            for i in range(p):
                for j in range(p):
                    block = blocks_T[i] @ blocks[j]
                    key = str(i) + "_" + str(j)
                    system_matrix_blocks.update({key: block})
            self._system_matrix_blocks = system_matrix_blocks
    
    def score(self, i: int, assignment: ArrayLike) -> Tuple[float, ArrayLike]:
        """Compute a score indicating how well an assignment of Granger-causal relationships fits the data set, according to the stochastic complexity criterion.

        Lower scores indicate a better fit.

        Parameters
        ----------
        i : int
            Index of the target time series in the data set.
        assignment : array-like of shape (n_timeseries,)
            Description of the Granger-causal influences on the target time series. Entries are binary. An entry of 1 at index j indicates that the j-th time series is Granger-causal for the target series.

        Returns
        -------
        criterion : float
            The stochastic complexity score.
        connections : array-like of shape (n_timeseries,)
            Array of real values with the same nonzero structure as assignment. The real values in connections indicate the strength of each Granger-causal connection in assignment. The connection strength values are computed from the regression coefficients of the Graphical Granger causality model.

        Raises
        ------
        ValueError
            If assignment is invalid. For the stochastic complexity, a vector of all zeros is invalid because in this case the score computation breaks down.
        """

        assignment = np.asarray(assignment)
        
        # check that the score can be computed for this assignment
        if not self._is_valid(assignment):
            raise ValueError("Invalid assignment")

        # define the parameters that can be extracted directly from the input data
        k_i = sum(assignment) # number of ones in the boolean assignment vector
        d = self.lag
        n = self.n_measurements

        # compute the remaining parameters
        sigma_squared_hat, R_hat, beta_hat = self._compute_parameters(i, assignment)

        # compute intermediate terms that occur multiple times in the formula
        term1 = (n - d - k_i) / 2
        term2 = k_i/2

        # compute the value of the stochastic criterion
        criterion = term1 * math.log(sigma_squared_hat) + term2 * math.log(R_hat) - math.lgamma(term1) - math.lgamma(term2)

        # create the vector that will hold the connection strength values
        connections = np.zeros(shape=self.n_timeseries, dtype=float)

        # for each nonzero entry in assignment, find the maximum absolute value in the corresponding section of beta_hat and write it to the connections vector
        beta_absvals = np.absolute(beta_hat) 
        nonzero_indices, = np.nonzero(assignment) # comma to unpack the tuple returned by nonzero()
        for idx_src, idx_dest in enumerate(nonzero_indices): # for 
            maxval = max(beta_absvals[(idx_src*d) : (idx_src+1)*d])
            connections[idx_dest] = maxval
        
        return criterion, connections

    def _is_valid(self, assignment: ArrayLike) -> bool:
        # check if an assignment vector has valid entries
        # invalid vectors:
        # - a vector with all zeros
        if sum(assignment) == 0:
            return False
        return True
    
    def _compute_parameters(self, i: int, assignment: ArrayLike) -> Tuple[float, float, np.ndarray]:
        # local handles
        p = self.n_timeseries
        k_i = sum(assignment)
        designmat_blocks = self._designmat_blocks
        designmat_blocks_transposed = self._designmat_blocks_transposed

        # extract the relevant indices from assignment
        assignment_indices = [idx for idx in range(p) if assignment[idx]==1] # list of indices where assignment has a 1

        # get the x_i vector
        x_i = self.X[i][self.lag:]

        # create the matrix of the system that we have to solve for beta 
        if self.precompute_matrix_blocks: 
            # assemble the system matrix from precomputed blocks
            system_matrix_blocks = self._system_matrix_blocks
            systemmatrix_blocked = []
            for idx_i in assignment_indices:
                block_row = []
                for idx_j in assignment_indices:
                    key = str(idx_i) + "_" + str(idx_j)
                    block_row.append(system_matrix_blocks[key])
                row = np.concatenate(block_row, axis=1)
                systemmatrix_blocked.append(row)
            systemmatrix = np.concatenate(systemmatrix_blocked)
        else: 
            # assemble the design matrix and its transpose, and compute the system matrix for beta via matrix-matrix multiplication
            assignment_bool = [val==1 for val in assignment] # like assignment, but as an array of True/False
            relevant_designmat_blocks_transposed = self._designmat_blocks_transposed[assignment_bool]
            assignmentmatrix_T = np.vstack(relevant_designmat_blocks_transposed)
            assignmentmatrix = np.transpose(assignmentmatrix_T)
            systemmatrix = assignmentmatrix_T @ assignmentmatrix

        # compute the right hand side vector for the system
        # equivalent to: rhs = assignmentmatrix_T @ x_i
        # because we don't want to have to assemble the matrix, we multiply x_i with each matrix block separately and stack the results
        rhs_blocks = [designmat_blocks_transposed[idx] @ x_i for idx in assignment_indices]
        rhs = np.concatenate(rhs_blocks)

        # compute beta_hat by solving the linear system
        if self.use_lapack_gelsy:
            beta_hat = scipy.linalg.lstsq(a=systemmatrix, b=rhs, cond=self.rcond, lapack_driver='gelsy')[0] # we're only interested in the first entry of the returned tuple   
        else:
            beta_hat = np.linalg.lstsq(a=systemmatrix, b=rhs, rcond=self.rcond)[0]

        # if required, compute and save the solution norm and residual norm of the least squares result
        if self.save_lstsq_norms:
            self.lstsq_solution_norm_ = np.linalg.norm(beta_hat, ord=1)
            residual = rhs - systemmatrix @ beta_hat
            self.lstsq_residual_norm_ = np.linalg.norm (residual, ord=1)

        # compute the estimate X_i @ beta_hat
        # blocked version to avoid the costly matrix assembling
        beta_blocks = np.split(beta_hat, k_i) # split beta_hat into k_i blocks of equal size
        
        x_estimate_components = [designmat_blocks[assignment_indices[idx]] @ beta_blocks[idx] for idx in range(k_i)]
        x_estimate = np.add.reduce(x_estimate_components) # sum up the results

        # compute sigma_squared_hat
        diffvec = x_i - x_estimate
        squared_norm = np.inner(diffvec, diffvec) # compute the squared norm without the unnecessary square-root-then-square step
        sigma_squared_hat = squared_norm / (self.n_measurements - self.lag - k_i) 

        # compute R_hat
        R_hat = np.inner(x_estimate, x_estimate) / (self.n_measurements - self.lag)

        # return the result
        return sigma_squared_hat, R_hat, beta_hat