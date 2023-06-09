from ant import Ant
from solution import Solution
import numpy as np
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
    for i, d in enumerate(diff):
        if d >= delta[i]:
            return False
    return True

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
    return A

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


def maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, epsilon, dy):
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    A = []
    start = time.time()
    for i in range(max_iterations):
        P = []
        for k in range(n_groups):
            s = Solution(timetables, days)
            for h in timetables:
                vehicles_timetable = copy.deepcopy(vehicles)
                costumers_h = get_costumers_day_timetable(costumers, h)
                costumers_timetable = copy.deepcopy(costumers_h)
                for d in range(days):
                    depot = costumers[0]
                    n = len(costumers)
                    ant = Ant(depot, n, min_pheromone, max_pheromone)
                    ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
                s.add_assigment_vehicles(vehicles_timetable, costumers_timetable, h)
            s.get_fitness()
            #s.is_feasible()
            P.append(s)
        update_pheromone(pheromone_matrix, delta_ant_matrix, P, rho, Q, timetables, days)
        delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
        A = archive_update_pqedy(A, P, epsilon, dy)
        print (f'>> non dominated {i}')
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
