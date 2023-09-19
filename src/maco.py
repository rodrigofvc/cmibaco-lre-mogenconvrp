from ant import Ant
from solution import Solution
import numpy as np
import random
import copy
import time
from pymoo.indicators.hv import HV

def initialize_multiple_matrix(days, n_costumers, ones):
    matrices = {'AM': [], 'PM': []}
    for d in range(days):
        if ones:
            m = np.ones((n_costumers,n_costumers))
            m_ = np.ones((n_costumers,n_costumers))
        else:
            m = np.zeros((n_costumers,n_costumers))
            m_ = np.zeros((n_costumers,n_costumers))
        matrices['AM'].append(m)
        matrices['PM'].append(m_)
    return matrices

def get_costumers_day_timetable(costumers, timetable_day):
    timetable = -1
    if timetable_day == 'AM':
        timetable = 0
    elif timetable_day == 'PM':
        timetable = 1
    else:
        raise()
    costumers_dh = []
    for costumer in costumers[1:]:
        if costumer.timetable == timetable:
            costumers_dh.append(costumer)
    if len(costumers_dh) == 0:
        raise()
    return costumers_dh

# True if x and y distance is less equal delta
def distance_hausforff_delta(x, y, delta):
    diff = [abs(x.get_fitness()[i] - y.get_fitness()[i]) for i in range(len(delta))]
    if diff[0] <= delta[0] and diff[1] <= delta[1] and diff[2] <= delta[2]:
        return True
    return False

def archive_update_pqedy(A, P, epsilon, delta):
    added = []
    for p in P:
        a_dominated_p = [a for a in A if a.epsilon_dominates(p, epsilon)]
        a_dominated_p += [a for a in A if distance_hausforff_delta(a, p, delta)]
        if len(a_dominated_p) == 0:
            A.append(p)
            added.append(p)

    for a in A[:]:
        p_dominated_a = [p for p in added if p.epsilon_dominates(a, epsilon)]
        if len(p_dominated_a) != 0:
            A.remove(a)
            added.remove(a)
    return A, added

def non_dominated(A, P):
    added = []
    for p in P:
        a_dominated_p = [a for a in A if a.dominates(p)]
        if len(a_dominated_p) == 0:
            A.append(p)
            added.append(p)

    for a in A[:]:
        p_dominated_a = [p for p in added if p.dominates(a)]
        if len(p_dominated_a) != 0:
            A.remove(a)
    return A

def update_pheromone(pheromone_matrix, delta_ant_matrix, P, rho, Q, timetables, days):
    for timetable in timetables:
        for d in range(days):
            pheromone_matrix[timetable][d] *= (1-rho)
            pheromone_matrix[timetable][d] += delta_ant_matrix[timetable][d]

def get_pheromone_delta_d_h(n, solutions_accepted, timetable, day, Q):
    delta_d_h = np.zeros(n)
    for s in solutions_accepted:
        ant = s.ants[timetable][day]
        vehicles_day = [v for v in s.assigments_vehicles if day in v.tour.keys()]
        for vehicle in vehicles_day:
            delta_d_h += ant.update_delta_matrix_global(vehicle, day, timetable, Q)
    return delta_d_h

#TODO
def global_update_pheromone(pheromone_matrix, solutions_accepted, rho, Q, timetables, days):
    for timetable in timetables:
        for d in range(days):
            n = pheromone_matrix[timetable][d].shape
            delta_d_h = get_pheromone_delta_d_h(n, solutions_accepted, timetable, d, Q)
            pheromone_matrix[timetable][d] *= (1-rho)
            pheromone_matrix[timetable][d] += delta_d_h


def build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix):
    P = []
    n_costumers = len(costumers)
    for k in range(n_groups):
        s = Solution(timetables, days)
        vehicles_timetable = copy.deepcopy(vehicles)
        costumers_timetable = copy.deepcopy(costumers[1:])
        depot = costumers[0]
        for h in timetables:
            for d in range(days):
                ant = Ant(depot, n_costumers, min_pheromone, max_pheromone)
                ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
                s.add_ant_timetable_day(ant, h)
        s.add_assigment_vehicles(vehicles_timetable, costumers_timetable)
        s.depot = depot
        s.get_fitness()
        s.is_feasible()
        P.append(s)
    return P



def maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
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
        A, solutions_accepted = archive_update_pqedy(A, P, epsilon, dy)
        update_pheromone(pheromone_matrix, delta_ant_matrix, P, rho, Q, timetables, days)
        delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
        print(f'>> non dominated {i}')
        for a in A:
            print ((a.f_1, a.f_2, a.f_3))
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
    print(f'>>>> min time_tour {min_time_tour}, min arrival {min_arrival_time}, min vehicle {min_vehicle} - max time_tour {max_time_tour}, max arrival {max_arrival_time}, max vehicle {max_vehicle}')
    statistics = [min_time_tour, max_time_tour, avg_time_tour, min_arrival_time, max_arrival_time, avg_arrival_time, min_vehicle, max_vehicle, avg_vehicle, n_solutions, duration]
    return A, log_hypervolume, duration, statistics
