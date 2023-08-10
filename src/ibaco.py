from maco import global_update_pheromone
from maco import build_solutions
from maco import initialize_multiple_matrix
from maco import archive_update_pqedy
from pymoo.indicators.hv import HV

import time
import copy
import numpy as np
import random
import math

class SolutionIBACO():
    counter = 0
    def __init__(self, solution, f_i):
        self.solution = solution
        self.f_i = f_i
        self.fitness = 0
        SolutionIBACO.counter += 1
        self.id = SolutionIBACO.counter

    def dominates(self, other):
        return (self.f_i <= other.f_i).all()

    def crossover(self, other, min_pheromone=10e-3, max_pheromone=10e5, prob_cross=0.80):
        solutions = self.solution.crossover(other.solution, min_pheromone, max_pheromone, prob_cross)
        if solutions == []:
            return []

        solution_1 = solutions[0]
        f_i_1 = np.array([solution_1.f_1, solution_1.f_2, solution_1.f_3])
        new_point_1 = SolutionIBACO(solution_1, f_i_1)

        solution_2 = solutions[1]
        f_i_2 = np.array([solution_2.f_1, solution_2.f_2, solution_2.f_3])
        new_point_2 = SolutionIBACO(solution_2, f_i_2)
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
        self.f_i = new_f_i

    def __eq__(self, other):
        if isinstance(other, SolutionIBACO):
            return self.id == other.id
        return False

def wrap_ibaco(population):
    new_population = []
    for p in population:
        sol_ibaco = SolutionIBACO(p, np.array([p.f_1, p.f_2, p.f_3]))
        new_population.append(sol_ibaco)
    return new_population

def get_pheromone_delta_d_h_indicator(n, solutions_accepted, timetable, day, Q):
    delta_d_h = np.zeros(n)
    for s in solutions_accepted:
        for ant in s.solution.ants[timetable]:
            delta_d_h += ant.global_update * s.fitness
    return delta_d_h

def update_pheromone_indicator(pheromone_matrix, current_population, rho, Q, timetables, days):
    for timetable in timetables:
        for d in range(days):
            n = pheromone_matrix[timetable][d].shape
            delta_d_h = get_pheromone_delta_d_h_indicator(n, current_population, timetable, d, Q)
            pheromone_matrix[timetable][d] *= (1-rho)
            pheromone_matrix[timetable][d] += delta_d_h


def ibaco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    A = []
    k = 10e6
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    start = time.time()
    for i in range(max_iterations):
        current_population = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)
        n = len(current_population)
        current_population = wrap_ibaco(current_population)

        crossover_mutation = crossover_stage(current_population)
        mutation_stage(crossover_mutation, p_mut)

        current_population += crossover_mutation

        solutions_accepted = [c.solution for c in current_population]
        A, solutions_added = archive_update_pqedy(A, solutions_accepted, epsilon, dy)

        while len(current_population) > n:
            fitness_asigment(current_population, k, i)
            minimum = [(c.fitness, c) for c in current_population]
            minimum.sort(key=lambda x:x[0])
            minimum = minimum[0]
            current_population.remove(minimum[1])
            if len(current_population) % 50 == 0:
                print (f'iteration {i} - {len(current_population)} / {n}')
        for s in current_population:
            print (s.f_i)

        update_pheromone_indicator(pheromone_matrix, current_population, rho, Q, timetables, days)
        #global_update_pheromone(pheromone_matrix, solutions_accepted, rho, Q, timetables, days)
        print (f'>> Non dominated {i} | Added: {len(solutions_added)}')
        for a in A:
            print ((a.f_1, a.f_2, a.f_3))
        hyp = [(s.f_1, s.f_2, s.f_3) for s in A]
        hyp = np.array(hyp)
        hyp = ind(hyp)
        log_hypervolume.append(hyp)
        print (f'Hypervolume: {hyp}')
    duration = time.time() - start
    l = [a.f_1 for a in A]
    min_time_tour = min(l)
    max_time_tour = max(l)
    l1 = [a.f_2 for a in A]
    min_arrival_time = min(l1)
    max_arrival_time = max(l1)
    l2 = [a.f_3 for a in A]
    min_vehicle = min(l2)
    max_vehicle = max(l2)
    print (f'>>>> min time_tour {min_time_tour}, min arrival {min_arrival_time}, min vehicle {min_vehicle} - max time_tour {max_time_tour}, max arrival {max_arrival_time}, max vehicle {max_vehicle}')

    return A, log_hypervolume, duration

def indicator_hd(x_1,x_2):
    ref_point = np.array([20000, 1000, 100])
    ind = HV(ref_point=ref_point)
    h_x_1 = ind(x_1.f_i)
    if x_2.dominates(x_1):
        h_x_2 = ind(x_2.f_i)
        return h_x_2 - h_x_1
    h_diff = ind(x_1.f_i + x_2.f_i) - h_x_1
    return h_diff

def indicator_eps(x_1, x_2):
    return max(x_1.f_i - x_2.f_i)

def fitness_asigment(population, k, i):
    for j, p in enumerate(population):
        p_exc = [q for q in population if q.id != p.id]
        one_all_indicator = [-1*math.exp(-1*indicator_eps(q,p)/k) for q in p_exc]
        p.fitness += sum(one_all_indicator)


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

def repeated(iter, population):
    for ind in iter:
        for pop in population:
            if ind.id != pop.id and (ind.f_i == pop.f_i).all():
                return True
    return False
