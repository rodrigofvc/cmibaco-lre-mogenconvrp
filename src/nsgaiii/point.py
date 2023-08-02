import random
import numpy as np

class Point:
    counter = 0
    def __init__(self, solution, f_i):
        self.solution = solution
        self.f_i = np.copy(f_i)
        self.original_f_i = np.copy(f_i)
        self.closest_point = None
        self.distance_point = -1
        Point.counter += 1
        self.id = Point.counter

    def dominates(self, other):
        return (self.f_i <= other.f_i).all()

    def crossover(self, other, min_pheromone=10e-3, max_pheromone=10e5, prob_cross=0.80):
        solutions = self.solution.crossover(other.solution, min_pheromone, max_pheromone, prob_cross)
        if solutions == []:
            return []

        solution_1 = solutions[0]
        f_i_1 = np.array([solution_1.f_1, solution_1.f_2, solution_1.f_3])
        new_point_1 = Point(solution_1, f_i_1)

        solution_2 = solutions[1]
        f_i_2 = np.array([solution_2.f_1, solution_2.f_2, solution_2.f_3])
        new_point_2 = Point(solution_2, f_i_2)
        childs = [new_point_1, new_point_2]
        return childs

    def mutation(self, population, prob_mut):
        new_solution = self.solution.mutation(prob_mut)
        new_f_i = np.array([new_solution.f_1, new_solution.f_2, new_solution.f_3])
        repeated = [p for p in population if (p.f_i == new_f_i).all()]
        while len(repeated) != 0:
            new_solution = self.solution.mutation(prob_mut)
            new_f_i = np.array([new_solution.f_1, new_solution.f_2, new_solution.f_3])
            repeated = [p for p in population if (p.f_i == new_f_i).all()]
        self.solution = new_solution
        self.f_i = np.array([new_solution.f_1, new_solution.f_2, new_solution.f_3])

    def __eq__(self, other):
        if isinstance(other, Point):
            return (self.f_i == other.f_i).all()
        return False
