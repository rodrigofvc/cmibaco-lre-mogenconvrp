import numpy as np
from pymoo.core.crossover import Crossover

class MGCVRPCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            first, second = X[0, k, 0], X[1, k, 0]
            prob_cross = 0.80
            childs = []
            while len(childs) == 0:
                childs = first.crossover(second, prob_cross=prob_cross)
            child_a = childs[0]
            child_b = childs[1]
            Y[0, k, 0] = child_a
            Y[1, k, 0] = child_b
        return Y