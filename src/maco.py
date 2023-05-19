from ant import Ant
from solution import Solution
import numpy as np
import copy

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


def maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone):
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    A = []
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
                    ant = Ant(depot, n, min_pheromone)
                    ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
                s.add_assigment_vehicles(vehicles_timetable, costumers_timetable, h)
            s.get_fitness()
            #s.is_feasible()
            P.append(s)
        update_pheromone(pheromone_matrix, delta_ant_matrix, P, rho, Q, timetables, days)
        delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
        A = non_dominated(A, P)
        print (f'>> non dominated {i}')
        for a in A:
            print ((a.f_1, a.f_2, a.f_3))
    return A
