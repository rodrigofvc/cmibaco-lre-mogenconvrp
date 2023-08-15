
class LNSolution():

    def __init__(self, solution, f_i):
        self.solution = solution
        self.f_i = f_i

    def get_dr(self, costumer, max_atd, epsilon=10e-4):
        z_i = costumer.get_max_vehicle_difference()
        l_i = costumer.get_max_arrival_diference()
        dr = z_i + l_i/(max_atd + epsilon)
        return dr

    def destroy_operator(self, n_removes):
        removed = []
        costumers = self.solution.assigments_costumers
        max_atd = self.solution.get_max_difference_arrive()
        dr = [(c,self.get_dr(c, max_atd)) for c in costumers]
        dr.sort(lambda x: x[1], reverse=True)
        for i in range(n_removes):
            c_rmv = dr[i]
            self.solution.remove_costumer(c_rmv)
            removed.append(c_rmv)
        return removed

    def repair_operator(self, costumers_to_add):
        pass








        pass
