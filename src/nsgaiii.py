import numpy as np
import time
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from nsgaiii_problem import MGCVRProblem
from nsgaiii_sampling import MGCVRPSampling
from nsgaiii_crossover import MGCVRPCrossover
from nsgaiii_mutation import MGCVRPMutation
from nsgaiii_eliminate import MGCVRPDuplicateElimination
from maco import archive_update_pqedy, global_update_pheromone, initialize_multiple_matrix, build_solutions


def nsgaiii_algorithm(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    A = []
    start = time.time()
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=15)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    current_population = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers,
                                         timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy,
                                         pheromone_matrix, delta_ant_matrix)
    algorithm = NSGA3(pop_size=len(current_population),
                      sampling=MGCVRPSampling(current_population),
                      crossover=MGCVRPCrossover(),
                      mutation=MGCVRPMutation(),
                      eliminate_duplicates=MGCVRPDuplicateElimination(),
                      ref_dirs=ref_dirs)

    res = minimize(MGCVRProblem(),
                   algorithm,
                   ('n_gen', 2),
                   seed=1,
                   verbose=False)
    solutions_accepted = [res.X[i][0] for i in range(res.X.shape[0])]
    for i in range(res.X.shape[0]):
        s = res.X[i][0]
        print (s.f_1, s.f_2, s.f_3)
    A, solutions_added = archive_update_pqedy(A, solutions_accepted, epsilon, dy)
    end = time.time()
    duration = end - start
    return A

def nsgaiii_aco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    A = []
    start = time.time()
    for i in range(max_iterations):
        current_population = nsgaiii_algorithm(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
        A, solutions_accepted = archive_update_pqedy(A, current_population, epsilon, dy)
        global_update_pheromone(pheromone_matrix, solutions_accepted, rho, Q, timetables, days)
        for a in A:
            print((a.f_1, a.f_2, a.f_3))
        hyp = [(s.f_1, s.f_2, s.f_3) for s in A]
        hyp = np.array(hyp)
        hyp = ind(hyp)
        log_hypervolume.append(hyp)
        print(f'Hypervolume: {hyp}')
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