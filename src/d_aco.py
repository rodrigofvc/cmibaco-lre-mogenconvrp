from ant import Ant
from maco import build_solutions
from maco import initialize_multiple_matrix
import numpy as np

class SolutionDC(object):
    counter = 0
    def __init__(self, solution, vector):
        self.solution = solution
        self.vector = vector
        self.f_i = self.get_fitness()
        self.neighborhood_vectors = []
        SolutionDC.counter += 1
        self.id = SolutionDC.counter

    def get_fitness(self):
        self.f_i = self.vector[0] * self.solution.f_1 + self.vector[1] * self.solution.f_2 + self.vector[2] * self.solution.f_3

    def eval_vector(self, other_vector):
        other_f_i = other_vector[0] * self.solution.f_1 + other_vector[1] * self.solution.f_2 + other_vector[2] * self.solution.f_3

    def crossover(self, other):
        pass

def initialize_vectors(n_vectors, m,):
    vectors = []
    for _ in range(n_vectors):
        vectors.append(np.random.rand(m,))
    return vectors

def k_nearest_vectors(current_vector, vectors, k):
    vectors_exc = [v for v in vectors if not (v == current_vector).all()]
    distances_vector = [(np.linalg.norm(v-current_vector), v) for v in vectors_exc]
    distances_vector = distances_vector.sort(key=lambda x:x[0])
    k_near = distances_vector[:k]
    k_near = [p[1] for p in k_near]
    return k_near

def build_solutions_decomposition(vectors, n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)
    P = []
    n_costumers = len(costumers)
    for k in range(n_groups):
        s = Solution(timetables, days, vectors[k])
        for h in timetables:
            vehicles_timetable = copy.deepcopy(vehicles)
            costumers_h = get_costumers_day_timetable(costumers, h)
            costumers_timetable = copy.deepcopy(costumers_h)
            for d in range(days):
                depot = costumers[0]
                ant = Ant(depot, n_costumers, min_pheromone, max_pheromone)
                ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
                s.add_ant_timetable_day(ant, h)
            s.add_assigment_vehicles(vehicles_timetable, costumers_timetable, h)
        s.get_fitness()
        s.is_feasible()
        P.append(s)
    return P


def daco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    m = 3
    iterations = 10
    vectors = initialize_vectors(n_vectors, )
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    current_population = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)
    initial_population = []
    for j, s in enumerate(current_population):
        new_s = SolutionDC(current_population[j], vectors[j])
    current_population = initial_population
    for i in range(iterations):
        current_population = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)
        crossover_mutation = crossover_stage(current_population)
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>  crossover DACO {iteration}')
        """
        for pop in crossover_mutation:
            print (pop.f_i)
        """
        mutation_stage(crossover_mutation)
        

def repeated(iter, population):
    for ind in iter:
        for pop in population:
            if ind.id != pop.id and (ind.f_i == pop.f_i).all():
                return True
    return False

def crossover_stage(parent_population, prob_cross=0.8):
    n = len(parent_population)
    child_population = []
    while len(child_population) < n:
        sample = random.sample(parent_population, 2)
        first = sample[0]
        second = sample[1]
        childs = first.crossover(second, prob_cross=prob_cross)
        if len(childs) == 0:
            continue
        if repeated(childs, parent_population + child_population):
            continue
        if len(child_population) + len(childs) > n:
            child_population.append(childs[0])
        else:
            child_population = child_population + childs
    return child_population

def mutation_stage(population, prob_mut=0.1):
    for individual in population:
        individual.mutation(population, prob_mut)












def fname(arg):
    pass
