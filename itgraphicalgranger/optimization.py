"""This module contains implementations of optimization methods that can be used to search for the best-fitting network of Granger-causal relationships in a data set of time series.

All optimizer classes need to fulfil the following requirements:
- All user-defined parameters can be set via optional keyword arguments in the constructor. Sensible defaults are provided for all user-defined parameters.
- The optimization process is started by calling the fit method, which accepts the training data as its only argument.
- Once the optimization process is finished, the optimizer exposes the following public attributes: adjacency_matrix_, beta_matrix_, and adjacency_matrix_scores_. These attributes contain the results in the same format as described in the documentation of the `itgraphicalgranger.GrangerCausality` class.

The following optimization methods are provided: exhaustive search and genetic algorithm.

Classes
-------
GeneticAlgorithm
    Implementation of the genetic algorithm optimization.
ExhaustiveSearch
    Implementation of the exhaustive search optimization.

"""

import numpy as np
import itertools
from typing import Any, Optional, Tuple
from numpy.typing import ArrayLike
from enum import Enum

from .criterion import CriterionLike

class GeneticAlgorithm(object):
    """Optimizer that uses a genetic algorithm to find the network of Granger-causal relationships in a data set.

    Optimization is carried out for each time series in the input data separately in order to find the Granger-causal influences on this time series.
    The results for all time series are assembled into an output matrix.

    To find the influences on one time series (the "target series"), the procedure is as follows:
    The algorithm maintains a population of binary assignment vectors that describe possible Granger-causal influences on the target time series. An entry of 1 at index j in an assignment vector means that the j-th time series is Granger-causal for the target time series. The starting population consists of randomly created individuals.
    For each assignment vector, a score is computed (with the help of a criterion object) that indicates how well this assignment fits the data set according to this criterion. Lower scores indicate a better fit.
    The optimizer then applies the following operators to the population to create a new generation:
        - Reproduction: Select the population members with the best scores (the "elite") and copy them into the next generation.
        - Crossover: Select two parents from the elite using a tournament selection. Create a new child by crossover of these parents: All entries in the child to the left of a random crossover location are copied from the first parent, all entries at and to the right of the crossover location are copied from the second parent. This process is repeated until the new generation contains the desired number of individuals.
        - Mutation: Flip some random bits in the new population.
    
    These steps are repeated to create the desired number of generations. The individual with the best score over all generations is reported as the best assignment vector for the target time series.
    

    Parameters
    ----------
    criterion : object
        Instance of a class that represents an information-theoretic criterion. This object will be used to compute the scores for each assignment vector. It must follow the duck-typing interface specified by the `itgraphicalgranger.criterion.CriterionLike` protocol.
    n_generations : int, default=10
        Number of generations to create.
    population_size : int, default=10
        Number of population members in each generation.
    elite_fraction : float, default=0.5
        Fraction of the total population size that is considered as elite for the reproduction stage. 
    elite_size : int, default=round(0.5*population_size)
        Number of population members that are copied into the next generation during the reproduction stage. The elite_size is computed as int(round(population_size * elite_fraction)). The remaining (population_size - elite_size) population members will be created via crossover.
    tournament_size : int, default=3
        Number of competitors in the tournament selection for the crossover parents. To find a single parent, tournament_size random members are drawn from the elite, and the one among them with the best score is chosen as parent.
    mutation_probability : float, default=0.1
        Probability that a single bit gets flipped during the mutation stage (must be between 0 and 1). The mutation probability should be kept low, otherwise the Genetic Algorithm can degenerate to a random search.
    remember_scores : bool, default=True
        If True, keep a history of all previously encountered population members and their scores. If new population members are duplicates of previous members, their scores can be retrieved from the history and do not have to be recomputed, which can lead to considerable runtime improvements. If False, no history is kept, and scores are computed from scratch for every population member.
    n_runs : int, default=1
        Number of times the optimization process is repeated with different random starting populations for each time series. The final result is the best result over all n_runs repetitions. Using multiple runs can be helpful in cases where the choice of starting population is supposed to have a large impact on the result.
    n_timeseries : int
        number of timeseries in the input data.
    n_measurements : int
        number of features in each time series in the input data.

    Attributes
    ----------
    adjacency_matrix_: ndarray of shape (n_timeseries, n_timeseries)
        Binary matrix containing a representation of the Granger-causal relationships between the time series in the input data. An entry of 1 in row i and column j means that the j-th time series has a Granger-causal influence on the i-th time series. (Note that this is the transpose of the adjacency matrix of a directed graph.)
    beta_matrix_: ndarray of shape (n_timeseries, n_timeseries)
        If criterion supports the computation of connection strength values, beta_matrix_ has the same nonzero pattern as adjacency_matrix_, and the real values in beta_matrix_ indicate the strength of each Granger-causal connection in adjacency_matrix_. Otherwise, the contents of beta_matrix_ are meaningless.
    adjacency_matrix_scores_: ndarray of shape (n_timeseries,)
        Contains one score value for each row in adjacency_matrix_ to indicate how well that row fits the input data, according to criterion. Lower scores mean a better fit.
    """

    class _ReplacementStrategy(Enum):
        RANDOM = 1
        OFFSPRING = 2

    def __init__(self, criterion: CriterionLike, n_generations: int = 10, population_size: int = 10, elite_fraction: float = 0.5, tournament_size: int = 3, mutation_probability: float = 0.1, remember_scores: bool = True, n_runs: int = 1, random_seed: int = None, **_: Optional[Any]) -> None:
        """Create an instance of the GeneticAlgorithm optimizer.

        Parameters
        ----------
        criterion : object
            Instance of a class that represents an information-theoretic criterion. This object will be used to compute the scores for each assignment vector. It must follow the duck-typing interface specified by the `itgraphicalgranger.criterion.CriterionLike` protocol.
        n_generations : int, default=10
            Number of generations to generate.
        population_size : int, default=10
            Number of population members in each generation.
        elite_fraction : float, default=0.5
            Fraction of the total population size that is considered as elite for the reproduction stage.
        tournament_size : int, default=3
            Number of competitors in the tournament selection for the crossover parents. To find a single parent, tournament_size random members are drawn from the elite, and the one among them with the best score is chosen as parent.
        mutation_probability : float, default=0.1
            Probability that a single bit gets flipped during the mutation stage (must be between 0 and 1). The mutation probability should be kept low, otherwise the Genetic Algorithm can degenerate to a random search.
        remember_scores : bool, default=True
            If True, keep a history of all previously encountered population members and their scores. If new population members are duplicates of previous members, their scores can be retrieved from the history and do not have to be recomputed, which can lead to considerable runtime improvements. If False, no history is kept, and scores are computed from scratch for every population member.
        n_runs : int, default=1
            Number of times the optimization process is repeated with different random starting populations for each time series. The final result is the best result over all n_runs repetitions. Using multiple runs can be helpful in cases where the choice of starting population is supposed to have a large impact on the result.
        random_seed : int or None, default=None
            Seed for the random generator (`numpy.random.default_rng`) that is used for all operations requiring randomness during the optimization process.
        """
        
        self.criterion = criterion
        
        # sanity checks for user input values
        if n_generations < 1:
            raise ValueError("Invalid number of generations")
        self.n_generations = n_generations

        if population_size < 4:
            raise ValueError("Invalid population size")
        self.population_size = population_size

        if elite_fraction <= 0 or elite_fraction > 1:
            raise ValueError("Invalid elite fraction")
        self.elite_fraction = elite_fraction
        elite_size = int(round(population_size * elite_fraction))
        self.elite_size = elite_size

        if tournament_size < 1:
            raise ValueError("Invalid tournament size")
        if tournament_size >= population_size:
            raise ValueError("Tournament size must be less than population size")
        self.tournament_size = tournament_size

        if mutation_probability < 0 or mutation_probability > 1:
            raise ValueError("Invalid mutation probability")
        self.mutation_probability = mutation_probability

        self.remember_scores = remember_scores

        if n_runs < 1:
            raise ValueError("Invalid number of experiment runs")
        self.n_runs = n_runs

        # initialize the random generator
        self._random_generator = np.random.default_rng(random_seed)

    def fit(self, X: ArrayLike) -> "GeneticAlgorithm":
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

        self.n_timeseries, self.n_measurements = X.shape

        adjmat = [] # output: the adjacency matrix of the causal graph
        betamat = [] # output: beta values corresponding to the adjacency matrix entries
        adjmat_scores = [] # output: the score that was computed for each row in the adjacency matrix

        for i in range(self.n_timeseries):
            # initialize the dictionary to remember all previously computed scores (saves computation time)
            # the score also depends on i, so we need to create a new history whenever the value of i changes
            if self.remember_scores:
                self._history = dict()
            
            # keep track of the overall best result
            overall_best_score = float('inf')

            # repeat the optimization n_runs times
            for _ in range(self.n_runs):
                # create the initial random population
                population = self._create_random_population()

                # compute the score for each population member
                scores, beta_vectors = self._compute_scores(population, i, replacement_strategy=self._ReplacementStrategy.RANDOM)

                # keep track of the overall best result
                best_member, best_beta, best_score = self._get_best_member(population, beta_vectors, scores)
                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_assignment = np.copy(best_member)
                    overall_best_beta = np.copy(best_beta)
            
                # iterate: keep performing crossover, mutation and selection to create new generations
                for _ in range(self.n_generations):
                    # create the new generation from the current elite and some crossovers of the current generation
                    population = self._create_next_generation(population, scores)

                    # randomly mutate some members of the population
                    population = self._apply_mutations(population)

                    # compute the score for each population member
                    scores, beta_vectors = self._compute_scores(population, i, replacement_strategy=self._ReplacementStrategy.OFFSPRING)

                    # keep track of the overall best result
                    best_member, best_beta, best_score = self._get_best_member(population, beta_vectors, scores)
                    if best_score < overall_best_score:
                        overall_best_score = best_score
                        overall_best_assignment = np.copy(best_member)
                        overall_best_beta = np.copy(best_beta)

            # pick the overall best member and score as result
            adjmat.append(overall_best_assignment)
            betamat.append(overall_best_beta)
            adjmat_scores.append(overall_best_score)
        
        # save results
        self.adjacency_matrix_ = np.asarray(adjmat)
        self.beta_matrix_ = np.asarray(betamat)
        self.adjacency_matrix_scores_ = np.asarray(adjmat_scores)
        
        return self
        
    def _create_random_population(self) -> np.ndarray:
        # create a random matrix filled with zeros and ones
        population = self._random_generator.integers(2, size=(self.population_size,self.n_timeseries))

        return np.asarray(population)
    
    def _replace_member(self, population: np.ndarray, memberidx: int, replacement_strategy: _ReplacementStrategy) -> None:
        # replace a "faulty" population member in a way that causes as little disturbance to the trajectory of the genetic algorithm as possible
        if replacement_strategy == self._ReplacementStrategy.RANDOM:
            # create a new random member
            population[memberidx, :] = self._random_generator.integers(2, size=self.n_timeseries)
        else:
            # this means that the error happened during the iterations
            # the entries in population are sorted: the first elite_size members are the parents, ordered by score
            if memberidx < self.elite_size:
                # in this case, the faulty member is a parent, which means that the error must have been caused by a mutation -> randomly flip a bit
                mutidx = self._random_generator.integers(0, self.n_timeseries-1, endpoint=True) # both endpoints inclusive
                population[memberidx, mutidx] = 0 if population[memberidx, mutidx] == 1 else 1
            else:
                # the faulty member is one of the new children -> replace by a new crossover version
                parent1idx = min(self._random_generator.choice(self.elite_size, size=self.tournament_size, replace=False))
                parent2idx = min(self._random_generator.choice(self.elite_size, size=self.tournament_size, replace=False)) 
                crossidx = self._random_generator.integers(1, high=self.n_timeseries-1, endpoint=True)
                population[memberidx, :crossidx] = population[parent1idx, :crossidx]
                population[memberidx, crossidx:] = population[parent2idx, crossidx:]

    def _compute_scores(self, population: np.ndarray, i: int, replacement_strategy: _ReplacementStrategy) -> Tuple[np.ndarray, np.ndarray]:
        # lists that will store the results
        scores = []
        beta_vectors = []
        
        # variable renaming for readability
        remember_scores = self.remember_scores
        if remember_scores:
            history = self._history
        
        for memberidx, member in enumerate(population):
            # if we're keeping a history, get the score from there if possible
            if remember_scores and tuple(member) in history:
                score, beta = history[tuple(member)]
            else: # either we don't keep a history, or we don't have the score in there yet
                while True: 
                    try: # try to compute the score
                        score, beta = self.criterion.score(i, member)
                        break
                    except ValueError: # if we encounter an error, replace the invalid member and try again
                        self._replace_member(population, memberidx, replacement_strategy)
                # if we're keeping a history, save the score there
                if remember_scores:
                    history.update({tuple(member): (score, beta)})
            # add the score and beta to the results
            scores.append(score)
            beta_vectors.append(beta)
            
        return np.asarray(scores), np.asarray(beta_vectors)
        
    def _create_next_generation(self, population: np.ndarray, scores: ArrayLike) -> np.ndarray:
        # sort both population and scores based on the scores
        sorting_indices = np.argsort(scores) # ascending order -> best scores go first
        population = population[sorting_indices]

        # select elite_size best members and put them into new_population
        new_population = np.empty((self.population_size, self.n_timeseries), dtype=np.int64)
        new_population[:self.elite_size, :] = population[:self.elite_size, :]

        # fill up the remaining spaces in the new population with children       
        for childidx in range(self.elite_size, self.population_size):
            # select two parents from the previous generation with a tournament selection
            # basically, draw tournament_size random indices, then take the smallest of them (works because population is already sorted by score)
            parent1idx = min(self._random_generator.choice(self.elite_size, size=self.tournament_size, replace=False))
            parent2idx = min(self._random_generator.choice(self.elite_size, size=self.tournament_size, replace=False)) # this could in theory be the same as parent1idx, but the probability is quite low so we just let that happen

            # select a random crossover location
            crossidx = self._random_generator.integers(1, high=self.n_timeseries-1, endpoint=True) # both endpoints inclusive; start at 1 to be sure we don't end up just creating a copy of parent2
            
            # create a child via crossover
            new_population[childidx, :crossidx] = population[parent1idx, :crossidx]
            new_population[childidx, crossidx:] = population[parent2idx, crossidx:]
        
        return new_population

    def _apply_mutations(self, population: np.ndarray) -> np.ndarray:
        # iterate over all population members
        for member in population:
            # iterate over each item in the current member
            for idx in range(len(member)):
                # draw a random number between 0 and 1
                randval = self._random_generator.random()
                # if the random value is below the mutation probability, flip this bit
                if randval < self.mutation_probability:
                    member[idx] = 0 if member[idx] == 1 else 1

        return population

    def _get_best_member(self, population: np.ndarray, beta_vectors: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # find the index of the lowest score
        min_idx = scores.argmin()
        return population[min_idx], beta_vectors[min_idx], scores[min_idx]


class ExhaustiveSearch(object):
    """Optimizer that uses an exhaustive search to find the network of Granger-causal relationships in a data set.

    A possible assignment of Granger-causal influences on a target time series is represented as a binary vector of length n_timeseries. An entry of 1 at index j in the assignment vector means that the j-th time series is Granger-causal for the target time series.
    For each time series, all 2^n_timeseries possible assignment vectors are evaluated. 
    A score is computed (using a criterion object) to determine how well each assignment fits the data set according to this criterion. The assignment vector with the best score is chosen as the result for this time series.
    The results for all time series are assembled into an output matrix.

    The runtime requirements grow exponentially with the number of time series in the data set. This optimizer is therefore only suitable for data sets where the number of time series is small.

    Parameters
    ----------
    criterion : object
        Instance of a class that represents an information-theoretic criterion. This object will be used to compute the scores for each assignment vector. It must follow the duck-typing interface specified by the `itgraphicalgranger.criterion.CriterionLike` protocol.
    verbose : bool, default=False
        If True, print information about which time series is currently being evaluated to the standard output (serves as a progress tracker).

    Attributes
    ----------
    adjacency_matrix_ : ndarray of shape (n_timeseries, n_timeseries)
        Binary matrix containing a representation of the Granger-causal relationships between the time series in the input data. An entry of 1 in row i and column j means that the j-th time series has a Granger-causal influence on the i-th time series. (Note that this is the transpose of the adjacency matrix of a directed graph.)
    beta_matrix_ : ndarray of shape (n_timeseries, n_timeseries)
        If criterion supports the computation of connection strength values, beta_matrix_ has the same nonzero pattern as adjacency_matrix_, and the real values in beta_matrix_ indicate the strength of each Granger-causal connection in adjacency_matrix_. Otherwise, the contents of beta_matrix_ are meaningless.
    adjacency_matrix_scores_ : ndarray of shape (n_timeseries,)
        Contains one score value for each row in adjacency_matrix_ to indicate how well that row fits the input data, according to criterion. Lower scores mean a better fit.
    current_i_ : int or None
        Index of the current target time series in the input data. Serves as a progress tracker (e.g. in the case of premature interruptions). Once the optimization procedure is finished for all time series, current_i_ is set to None.
    current_assignment_ : array-like of shape (n_timeseries,) or None
        Assignment vector that is currently being analyzed. Serves as a progress tracker (e.g. in the case of premature interruptions). Once the optimization procedure is finished for all time series, current_assignment_ is set to None.
    """

    def __init__(self, criterion: CriterionLike, verbose: bool = False, **_: Optional[Any]) -> None:
        """Create an instance of the ExhaustiveSearch optimizer.

        Parameters
        ----------
        criterion : object
            Instance of a class that represents an information-theoretic criterion. This object will be used to compute the scores for each assignment vector. It must follow the duck-typing interface specified by the `itgraphicalgranger.criterion.CriterionLike` protocol.
        verbose : bool, default=False
            If True, print information about which time series is currently being evaluated to the standard output (serves as a progress tracker).
        """

        self.criterion = criterion
        self.verbose = verbose

    def fit(self, X: ArrayLike) -> "ExhaustiveSearch":
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

        criterion = self.criterion

        # find the dimensionality of the data
        X = np.asarray(X)
        n_timeseries, n_measurements = X.shape

        # create all possible assignment vectors
        assignments = list(map(list, itertools.product([0,1], repeat=n_timeseries))) # convert to list of lists
        
        adjmat = [] # output: adjacency matrix
        betamat = [] # output: beta values corresponding to the adjacency matrix
        adjmat_scores = [] # output: the score that was computed for each row in the adjacency matrix

        # find the best fit for each time series
        for i in range(n_timeseries):

            # for easier user inspection in case of error, track the progress
            self.current_i_ = i

            if self.verbose:
                print(f"Optimizing time series {i+1} of {n_timeseries}")
                
            # initialize the current best value as infinity
            best_result = float('inf')

            # go through all possible assignments and evaluate them
            for assignment in assignments:
                # progress tracker
                self.current_assignment_ = assignment

                # check for validity of assignment
                try:
                    current_result, current_beta = criterion.score(i, assignment)

                    # if the new result is better than what we have found so far, save it
                    if current_result < best_result:
                        best_result = current_result
                        best_assignment = assignment
                        best_beta = current_beta
                except ValueError:
                    # if an assignment is considered invalid by the criterion, just ignore it and move on to the next one
                    pass
            
            # append the best assignment to the adjacency matrix
            adjmat.append(best_assignment)
            betamat.append(best_beta)
            adjmat_scores.append(best_result)
        
        # since optimization is done, delete the progress trackers
        self.current_i_ = None
        self.current_assignment_ = None

        # save the result
        self.adjacency_matrix_ = np.asarray(adjmat)
        self.beta_matrix_ = np.asarray(betamat)
        self.adjacency_matrix_scores_ = np.asarray(adjmat_scores)

        return self