from pymoo.core.mutation import Mutation

class MGCVRPMutation(Mutation):

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            p_mut = 0.10
            solution = X[i, 0]
            mutation = solution.mutation(p_mut)
            X[i,0] = mutation
        return X
