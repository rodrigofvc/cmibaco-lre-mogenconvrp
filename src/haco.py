import random
import numpy as np
from pymoo.indicators.hv import HV
import time
from maco import initialize_multiple_matrix, build_solutions, archive_update_pqedy, global_update_pheromone


def crossover_stage(parent_population, min_pheromone, max_pheromone, prob_cross=0.8):
    n = len(parent_population)
    child_population = []
    while len(child_population) < n:
        sample = random.sample(parent_population, 2)
        first = sample[0]
        second = sample[1]
        childs = first.crossover(second, min_pheromone, max_pheromone, prob_cross)
        if len(childs) == 0:
            continue
        childs[0].get_fitness()
        childs[1].get_fitness()
        childs[0].is_feasible()
        childs[1].is_feasible()
        if len(child_population) + len(childs) > n:
            child_population.append(childs[0])
        else:
            child_population = child_population + childs
    return child_population

#TODO
def mutation_stage(population, prob_mut=0.10):
    new_population = []
    for p in population:
        mutation = p.mutation(prob_mut)
        new_population.append(mutation)
    return new_population


def haco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    A = []
    start = time.time()
    for i in range(max_iterations):
        P = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)
        C = crossover_stage(P, min_pheromone, max_pheromone)
        M = mutation_stage(C)
        R = M + P
        A, solutions_accepted = archive_update_pqedy(A, R, epsilon, dy)
        global_update_pheromone(pheromone_matrix, solutions_accepted, rho, Q, timetables, days)
        #update_pheromone(pheromone_matrix, delta_ant_matrix, P, rho, Q, timetables, days)
        #delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
        print (f'>> Non dominated {i} | Added: {len(solutions_accepted)}')
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
    avg_time_tour = sum(l)/len(l)
    l1 = [a.f_2 for a in A]
    min_arrival_time = min(l1)
    max_arrival_time = max(l1)
    avg_arrival_time = sum(l1)/len(l1)
    l2 = [a.f_3 for a in A]
    min_vehicle = min(l2)
    max_vehicle = max(l2)
    avg_vehicle = sum(l2)/len(l2)
    n_solutions = len(A)
    print (f'>>>> min time_tour {min_time_tour}, min arrival {min_arrival_time}, min vehicle {min_vehicle} - max time_tour {max_time_tour}, max arrival {max_arrival_time}, max vehicle {max_vehicle}')
    statistics = [min_time_tour, max_time_tour, avg_time_tour, min_arrival_time, max_arrival_time, avg_arrival_time, min_vehicle, max_vehicle, avg_vehicle, n_solutions, duration]
    return A, log_hypervolume, duration, statistics