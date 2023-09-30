from maco import global_update_pheromone
from maco import build_solutions
from maco import initialize_multiple_matrix
from maco import archive_update_pqedy
from pymoo.indicators.hv import HV
from lns import mdls
from random import sample
import time
import numpy as np
import random
import math
from pymoo.indicators.hv import HV

from utils import get_statistics


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
        ant = s.solution.ants[timetable][day]
        delta_d_h += ant.global_update * s.fitness
    return delta_d_h

def update_pheromone_indicator(pheromone_matrix, current_population, rho, Q, timetables, days, min_pheromone=10e-3, max_pheromone=10e5):
    for timetable in timetables:
        for d in range(days):
            n = pheromone_matrix[timetable][d].shape
            delta_d_h = get_pheromone_delta_d_h_indicator(n, current_population, timetable, d, Q)
            pheromone_matrix[timetable][d] *= (1-rho)
            pheromone_matrix[timetable][d] += delta_d_h
            indices_less_ph = pheromone_matrix[timetable][d] < min_pheromone
            pheromone_matrix[timetable][d][indices_less_ph] = min_pheromone
            indices_max_ph = pheromone_matrix[timetable][d] > max_pheromone
            pheromone_matrix[timetable][d][indices_max_ph] = max_pheromone


def ibaco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, indicator, pheromone_matrix):
    A = []
    k = 10e6
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    start = time.time()
    epsilon_local = [0,0,0]
    dy_local = [0,0,0]
    for i in range(max_iterations):
        current_population = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)
        n = len(current_population)
        current_population = wrap_ibaco(current_population)

        crossover_mutation = crossover_stage(current_population)
        mutation_stage(crossover_mutation, p_mut)

        current_population += crossover_mutation

        solutions_accepted = [c.solution for c in current_population]
        A, solutions_added = archive_update_pqedy(A, solutions_accepted, epsilon_local, dy_local)

        while len(current_population) > n:
            fitness_asigment(current_population, k, i, indicator)
            minimum = [(c.fitness, c) for c in current_population]
            minimum.sort(key=lambda x:x[0])
            minimum = minimum[0]
            current_population.remove(minimum[1])
            if len(current_population) % 50 == 0:
                print (f'{indicator} | iteration {i} - {len(current_population)} / {n}')
            print (f'len {len(current_population)} {n}')
        #for s in current_population:
            #print (s.f_i)

        #update_pheromone_indicator(pheromone_matrix, current_population, rho, Q, timetables, days)

        #print (f'>> Non dominated {i} | Added: {len(solutions_added)}')
        #for a in A:
        #    print ((a.f_1, a.f_2, a.f_3))
        hyp = [(s.f_1, s.f_2, s.f_3) for s in A]
        hyp = np.array(hyp)
        hyp = ind(hyp)
        log_hypervolume.append(hyp)
        #print (f'Hypervolume: {hyp}')
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
    #print (f'>>>> min time_tour {min_time_tour}, min arrival {min_arrival_time}, min vehicle {min_vehicle} - max time_tour {max_time_tour}, max arrival {max_arrival_time}, max vehicle {max_vehicle}')
    statistics = [min_time_tour, max_time_tour, avg_time_tour, min_arrival_time, max_arrival_time, avg_arrival_time, min_vehicle, max_vehicle, avg_vehicle, n_solutions, duration]
    return A, log_hypervolume, duration, statistics

def indicator_hd(x_1,x_2):
    ref_point = np.array([20000, 1000, 100])
    ind = HV(ref_point=ref_point)
    h_x_1 = ind(x_1.f_i)
    if x_2.dominates(x_1):
        h_x_2 = ind(x_2.f_i)
        return h_x_2 - h_x_1
    h_diff = ind(np.array([x_1.f_i] + [x_2.f_i])) - h_x_1
    return h_diff

def indicator_eps(x_1, x_2):
    if (x_1.f_i <= x_2.f_i).all():
        return 0.0
    diff = x_1.f_i - x_2.f_i
    diff = [d for d in diff if d >= 0]
    return max(diff)

def distance_s_energy(i, j, s):
    return 1 / (np.linalg.norm(i-j, 2)**s)

def individual_contribution_s_energy(A, individual, s):
    Ac = [a for a in A if not np.allclose(a, individual)]
    if len(Ac) != len(A) - 1:
        raise('not found individual in A')
    EA = indicator_s_energy(A, s)
    EAC = indicator_s_energy(Ac, s)
    return 0.5 * (EA - EAC)

def indicator_s_energy(A, s):
    distances = [distance_s_energy(a, a_, s) for a in A for a_ in A if not np.allclose(a != a_)]
    return sum(distances)

def archive_riez_energy(archive, mu, s):
    A = [np.array([a.f_1, a.f_2, a.f_3]) for a in archive]
    while len(archive) > mu:
        contributions = [individual_contribution_s_energy(A, a, s) for a in A]
        a_worst = np.argmax(contributions)
        A.pop(a_worst)
        archive.pop(a_worst)
    return archive
def fitness_asigment(population, k, indicator):
    for j, p in enumerate(population):
        p_exc = [q for q in population if q.id != p.id]
        if indicator == 'epsilon':
            one_all_indicator = [-1 * math.exp(-1 * indicator_eps(q, p)/k) for q in p_exc]
        else:
            one_all_indicator = [-1 * math.exp(-1 * indicator_hd(q, p) / k) for q in p_exc]
        p.fitness = sum(one_all_indicator)

def crossover_stage(parent_population, prob_cross=0.8):
    n = len(parent_population)
    child_population = []
    # for tiny instances
    attempts = 2*n
    while len(child_population) < n and attempts > 0:
        attempts -= 1
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

def lns(current_population, params):
    solutions_unwrap = [s.solution for s in current_population]
    solutions = mdls(solutions_unwrap, params)
    for i, c in enumerate(current_population):
        c.solution = solutions[i]
        c.f_i = np.array([solutions[i].f_1, solutions[i].f_2, solutions[i].f_3])


def ibaco_indicator(params):
    n_groups = params['n_ants']
    rho = params['rho']
    days = params['days']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    Q = params['Q']
    max_iterations = params['max_iterations']
    costumers = params['costumers']
    timetables = params['timetables']
    vehicles = params['vehicles']
    q0 = params['q0']
    min_pheromone = params['min_pheromone']
    max_pheromone = params['max_pheromone']
    p_mut = params['p_mut']
    epsilon = params['epsilon']
    dy = params['dy']
    indicator = params['indicator']
    pheromone_matrix = params['pheromone_matrix']
    A = []
    k = params['k']
    log_hypervolume = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    delta_ant_matrix = initialize_multiple_matrix(days, n_costumers, False)
    start = time.time()
    solutions = []
    for i in range(max_iterations):
        current_population = build_solutions(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, pheromone_matrix, delta_ant_matrix)

        n = len(current_population)
        current_population = wrap_ibaco(current_population)

        crossover_mutation = crossover_stage(current_population)

        mutation_stage(crossover_mutation, p_mut)

        current_population += crossover_mutation

        lns(current_population, params['lns'])


        solutions_accepted = [c.solution for c in current_population]
        A, solutions_added = archive_update_pqedy(A, solutions_accepted, epsilon, dy)
        solutions += current_population

        while len(current_population) > n:
            fitness_asigment(current_population, k, indicator)
            minimum = [(c.fitness, c) for c in current_population]
            minimum.sort(key=lambda x:x[0])
            minimum = minimum[0]
            current_population.remove(minimum[1])
            if len(current_population) % 50 == 0:
                print (f'{indicator} | iteration {i} - {len(current_population)} / {n}')
        #for s in current_population:
        #    print (s.f_i)

        update_pheromone_indicator(pheromone_matrix, current_population, rho, Q, timetables, days)

        #print (f'>> Non dominated {i} | Added: {len(solutions_added)}')
        #for a in A:
        #    print ((a.f_1, a.f_2, a.f_3))
        hyp = [(s.f_1, s.f_2, s.f_3) for s in A]
        hyp = np.array(hyp)
        #print (hyp)
        hyp = ind(hyp)
        log_hypervolume.append(hyp)
        #print (f'Hypervolume: {hyp}')
    duration = time.time() - start
    statistics = get_statistics(A, duration)
    #print (f'>>>> min time_tour {statistics["min_time_tour"]}, min arrival {statistics["min_arrival_time"]}, min vehicle {statistics["min_vehicle"]} - max time_tour {statistics["max_time_tour"]}, max arrival {statistics["max_arrival_time"]}, max vehicle {statistics["max_vehicle"]}')
    A = [s for s in solutions if s.solution.id in [a.id for a in A]]
    return A, log_hypervolume, duration, statistics

def cooperative_ibaco(params):
    matrices = []
    days = params['days']
    rho = params['rho']
    Q = params['Q']
    timetables = params['timetables']
    costumers = params['costumers']
    vehicles = params['vehicles']
    indicators = params['cibaco']['indicators']
    k = len(indicators)
    epsilon = params['epsilon']
    dy = params['dy']
    n_costumers = len(costumers)
    max_iterations_cibaco = params['cibaco']['max_iterations_cibaco']
    nmig = params['cibaco']['nmig']
    mu = params['cibaco']['mu']
    s_energy = params['cibaco']['s_energy']
    log_hypervolume = []
    log_solutions_added = []
    ref_point = np.array([8000, 1000, len(vehicles)])
    ind = HV(ref_point=ref_point)
    for i in range(2):
        m = initialize_multiple_matrix(days, n_costumers, True)
        matrices.append(m)
    A = []
    A_indicator = []
    start = time.time()
    for i in range(max_iterations_cibaco):
        PS = []
        for j in range(k):
            indicator = indicators[j]
            params_indicator = {'n_ants': params['n_ants'], 'rho': params['rho'], 'days': params['days'],
                                'alpha': params['alpha'], 'beta': params['beta'], 'gamma': params['gamma'],
                                'delta': params['delta'], 'Q': params['Q'], 'max_iterations': params['max_iterations'],
                                'costumers': params['costumers'], 'timetables': params['timetables'], 'vehicles': params['vehicles'],
                                'q0': params['q0'], 'min_pheromone': params['min_pheromone'], 'max_pheromone': params['max_pheromone'],
                                'p_mut': params['p_mut'], 'epsilon': params['epsilon'], 'dy': params['dy'], 'indicator': indicator,
                                'k': params['cibaco']['k'], 'pheromone_matrix': matrices[j], 'lns': params['lns']}
            P,_,_,_, = ibaco_indicator(params_indicator)
            for p in P:
                p.solution.algorithm = indicator
            PS.append(P)
        for p in PS:
            solutions = [s.solution for s in p]
            A, solutions_added = archive_update_pqedy(A, solutions, epsilon, dy)
            A_indicator += [s for s in p if s.solution.id in [a.id for a in A]]
        #A = archive_riez_energy(A, mu, s_energy)
        ES = migration(A, A_indicator, nmig, k, params['cibaco']['k'], indicators)
        for j in range(k):
            solutions = [s for s in ES[j]]
            update_pheromone_indicator(matrices[j], solutions, rho, Q, timetables, days)
        for a in A:
            print((a.f_1, a.f_2, a.f_3))
        hyp = [(s.f_1, s.f_2, s.f_3) for s in A]
        hyp = np.array(hyp)
        hyp = ind(hyp)
        log_hypervolume.append(hyp)
        log_solutions_added.append(len(A))
        print(f'Hypervolume: {hyp}')
    duration = time.time() - start
    statistics = get_statistics(A, duration)
    print (f'>>>> min time_tour {statistics["min_time_tour"]}, min arrival {statistics["min_arrival_time"]}, min vehicle {statistics["min_vehicle"]} - max time_tour {statistics["max_time_tour"]}, max arrival {statistics["max_arrival_time"]}, max vehicle {statistics["max_vehicle"]}')
    return A, log_hypervolume, log_solutions_added, duration, statistics

def migration(A, A_indicator, nmig, k, k_indicator, indicators):
    ES = []
    for j in range(k):
        external_solutions = []
        random_nmig = [a for a in A if a.algorithm != indicators[j]]
        if len(random_nmig) >= nmig:
            external_solutions = [s for s in A_indicator if s.solution.id in [r.id for r in random_nmig]]
            external_solutions = sample(external_solutions, nmig)
        fitness_asigment(external_solutions, k_indicator, indicators[j])
        ES.append(external_solutions)
    return ES