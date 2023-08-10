from ant import Ant
from solution import Solution
import numpy as np
import random
import copy
import time
from pymoo.indicators.hv import HV
from nsgaiii.nsgaiii_algorithm import nsgaiii
from nsgaiii.point import Point

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
        # deberia ser la unica hormiga para ese dia!!!!!!!
        # TODO
        for ant in s.ants[timetable]:
            vehicles_day = [v for v in s.assigments_vehicles[timetable] if day in v.tour.keys()]
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

#TODO
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

#TODO
def build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix):
    P = []
    n_costumers = len(costumers)
    for k in range(n_groups):
        s = Solution(timetables, days)
        vehicles_timetable = copy.deepcopy(vehicles)
        costumers_timetable = copy.deepcopy(costumers[1:])
        for h in timetables:
            for d in range(days):
                depot = costumers[0]
                ant = Ant(depot, n_costumers, min_pheromone, max_pheromone)
                ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
                s.add_ant_timetable_day(ant, h)
        s.add_assigment_vehicles(vehicles_timetable, costumers_timetable)
        s.get_fitness()
        s.is_feasible()
        P.append(s)
    return P

def solution_to_point(population):
    new_points = []
    for p in population:
        f_i = np.array([p.f_1, p.f_2, p.f_3])
        new_point = Point(p, f_i)
        new_points.append(new_point)
    return new_points

def point_to_solution(points):
    population = []
    for p in points:
        population.append(p.solution)
    return population


def maco_nsgaiii(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    A = []
    start = time.time()
    for i in range(10):
        current_population = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)
        print(f'>>>>>>>>>>>>>>>>>>>>>> ACO build_solutions {i}')
        current_population = solution_to_point(current_population)
        current_population = nsgaiii(current_population, 20)
        print(f'>>>>>>>>>>>>>>>>>>>>>> ACO NSGAIII {i}')
        current_population = point_to_solution(current_population)
        A, solutions_accepted = archive_update_pqedy(A, current_population, epsilon, dy)
        print(f'>>>>>>>>>>>>>>>>>>>>>> ACO ARCHIVE UPDATE {i}')
        global_update_pheromone(pheromone_matrix, solutions_accepted, rho, Q, timetables, days)
        print(f'>>>>>>>>>>>>>>>>>>>>>> ACO UPDATE PHEROMONE {i}')
        delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
        print (f'>> Non dominated {i} | Added: {len(solutions_accepted)}')
        for a in A:
            print ((a.f_1, a.f_2, a.f_3))
        hyp = [(s.f_1, s.f_2, s.f_3) for s in A]
        hyp = np.array(hyp)
        hyp = ind(hyp)
        log_hypervolume.append(hyp)
        print (f'Hypervolume: {hyp}')
    duration = time.time() - start
    return A, log_hypervolume, duration


def maco_genetic_algorithm(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
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
                    ant = Ant(depot, n_costumers, min_pheromone, max_pheromone)
                    ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
                    s.add_ant_timetable_day(ant, h)
                s.add_assigment_vehicles(vehicles_timetable, costumers_timetable, h)
            s.get_fitness()
            P.append(s)
        C = crossover_stage(P, min_pheromone, max_pheromone)
        M = mutation_stage(C)
        R = M + P
        A, solutions_accepted = archive_update_pqedy(A, R, epsilon, dy)
        global_update_pheromone(pheromone_matrix, solutions_accepted, rho, Q, timetables, days)
        #update_pheromone(pheromone_matrix, delta_ant_matrix, P, rho, Q, timetables, days)
        delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
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
    l1 = [a.f_2 for a in A]
    min_arrival_time = min(l1)
    max_arrival_time = max(l1)
    l2 = [a.f_3 for a in A]
    min_vehicle = min(l2)
    max_vehicle = max(l2)
    print (f'>>>> min time_tour {min_time_tour}, min arrival {min_arrival_time}, min vehicle {min_vehicle} - max time_tour {max_time_tour}, max arrival {max_arrival_time}, max vehicle {max_vehicle}')
    return A, log_hypervolume, duration


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
        P = []
        for k in range(n_groups):
            s = Solution(timetables, days)
            for h in timetables:
                vehicles_timetable = copy.deepcopy(vehicles)
                costumers_h = get_costumers_day_timetable(costumers, h)
                costumers_timetable = copy.deepcopy(costumers_h)
                for d in range(days):
                    depot = costumers[0]
                    ant = Ant(depot, n_costumers, min_pheromone, max_pheromone)
                    ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
                s.add_assigment_vehicles(vehicles_timetable, costumers_timetable, h)
            s.get_fitness()
            s.is_feasible()
            P.append(s)
        A, solutions_accepted = archive_update_pqedy(A, P, epsilon, dy)
        update_pheromone(pheromone_matrix, delta_ant_matrix, P, rho, Q, timetables, days)
        delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
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
