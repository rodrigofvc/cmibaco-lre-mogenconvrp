import numpy as np
from pymoo.core.problem import ElementwiseProblem

class MGCVRProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=1, n_obj=3, n_ieq_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        x[0].get_fitness()
        out["F"] = np.array([x[0].f_1, x[0].f_2, x[0].f_3], dtype=float)