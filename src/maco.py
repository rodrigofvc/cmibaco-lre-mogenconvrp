from ant import Ant
from solution import Solution
import numpy as np
import copy

def initialize_multiple_matrix(days, n_costumers):
    matrices = {'AM': [], 'PM': []}
    for d in range(days):

        m = np.ones((n_costumers,n_costumers))
        matrices['AM'].append(m)

        m_ = np.ones((n_costumers,n_costumers))
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
        dominated = False
        for a in A:
            if a.dominates(p):
                dominated = True
        if not dominated:
            A.append(p)
            added.append(p)

    for a in A:
        for p in added:
            if p.dominates(a):
                A.remove(a)
                break
    return A

def ants_arc_ij(timetable, day, i, j, P, Q):
    delta_ijdh = 0
    for p in P:
        for vehicle in p.assigments_vehicles[timetable]:
            if vehicle.visited_costumer_ijdh(day, i, j):
                delta_ijdh += Q / vehicle.times_tour[day]
    return delta_ijdh

def update_arc_ij(timetable, day, matrix_dh, i, j, rho, Q, P):
    matrix_dh[i][j] = (1-rho) * matrix_dh[i][j] + ants_arc_ij(timetable, day, i, j, P, Q)
    matrix_dh[j][i] = matrix_dh[i][j]

def update_pheromone(pheromone_matrix, P, rho, Q):
    timetables = pheromone_matrix.keys()
    for h in timetables:
        matrix_h = pheromone_matrix[h]
        for d, matrix_dh in enumerate(matrix_h):
            n = matrix_dh.shape[0]
            for i in range(n):
                for j in range(i+1,n):
                    update_arc_ij(h, d, matrix_dh, i, j, rho, Q, P)
                    #print (f'{h}/day:{d} {i} {j}')

def maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles):
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers)
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
                    ant = Ant(depot)
                    ant.build_solution(pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable)
                s.add_assigment_vehicles(vehicles_timetable, costumers_timetable, h)
            P.append(s)
        update_pheromone(pheromone_matrix, P, rho, Q)
        A = non_dominated(A, P)
        print (f'>> non dominated {i}')
        for a in A:
            print (a.get_fitness())
