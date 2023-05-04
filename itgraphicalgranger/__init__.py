"""This package provides an implementation of the "Graphical Granger Causality by Information-Theoretic Criteria" method [1].

The method uses information-theoretic criteria to find the network of Granger-causal relationships in a data set consisting of multiple time series.
For this purpose, two components are used: an information-theoretic criterion and an optimizer.
The criterion is used to determine how well a given network of Granger-causal relationships fits the data set.
The optimizer tries out different networks of Granger-causal relationships to find the one that best fits the data set according to the criterion.
Any optimizer can be combined with any criterion.

Classes
-------
GrangerCausality
    User interface and wrapper around the internal logic.

Modules
-------
criterion
    Contains implementations of all available information-theoretic criteria.
optimization
    Contains implementations of all available optimizers.

References
----------
[1] K. Hlaváčková-Schindler and C. Plant, “Graphical granger causality by information-theoretic
criteria,” in: Proceedings of the 24th European Conference on Artificial Intelligence pp. 1459-1466, 2020.
"""

from .grangercausality import GrangerCausality
from . import criterion, optimization