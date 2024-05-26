import pickle

from pymoo.util.ref_dirs import get_reference_directions

from maco import build_solutions, initialize_multiple_matrix_rand, non_dominated
from maco import archive_update_pqedy
from lns import mdls
from random import sample
import time
import numpy as np
import random
import math
from pymoo.indicators.hv import HV

from solution import Solution
import utils

from utils import get_statistics

class SolutionIBACO():
    counter = 0
    fitness_eval = 0
    write_statisticals = False
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

    def mutation_deprecated(self, population, prob_mut):
        new_solution = self.solution.mutation(prob_mut)
        new_f_i = np.array([new_solution.f_1, new_solution.f_2, new_solution.f_3])
        repeated = [p for p in population if (p.f_i == new_f_i).all()]
        while len(repeated) != 0:
            new_solution = self.solution.mutation(prob_mut)
            new_f_i = np.array([new_solution.f_1, new_solution.f_2, new_solution.f_3])
            repeated = [p for p in population if (p.f_i == new_f_i).all()]
        self.solution = new_solution
        self.f_i = new_f_i

    def mutation(self, prob_mut):
        self.solution.mutation(prob_mut)
        self.f_i = np.array([self.solution.f_1, self.solution.f_2, self.solution.f_3])


    def __eq__(self, other):
        if isinstance(other, SolutionIBACO):
            return self.id == other.id
        return False

def wrap_ibaco(population, indicator):
    new_population = []
    for p in population:
        p.algorithm = indicator
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

def indicator_hd(x_1, x_2, ind_hv):
    h_x_1 = ind_hv(x_1.f_i)
    if x_2.dominates(x_1):
        h_x_2 = ind_hv(x_2.f_i)
        return h_x_2 - h_x_1
    h_diff = ind_hv(np.array([x_1.f_i] + [x_2.f_i])) - h_x_1
    return h_diff

def indicator_eps(x_1, x_2):
    if (x_1.f_i <= x_2.f_i).all():
        return 0.0
    diff = x_1.f_i - x_2.f_i
    diff = [d for d in diff if d >= 0]
    return max(diff)

def r2(A,W,z):
    diff = abs(A-z)
    d = np.array([d*W for d in diff])
    e = np.max(d, axis=-1)
    f = np.min(e, axis=0)
    f = f.sum()
    f = (1 / len(W)) * f.item()
    return f

def indicator_r2(x_1, x_2, w, z):
    r2_x1 = r2([x_1.f_i], w, z)
    r2_x1_x2 = r2([x_1.f_i, x_2.f_i], w, z)
    return r2_x1 - r2_x1_x2

def get_reference_point_r2(population):
    m = population[0].shape[0]
    z = np.zeros((m,))
    max_j = np.zeros((m,))
    min_j = np.zeros((m,))
    for j in range(m):
        val = [p[j] for p in population]
        min_j[j] = min(val)
        max_j[j] = max(val)
    for i in range(m):
        z[i] = max_j[i] - min_j[i]
    return z

def get_reference_point_hv_dynamic(population):
    z = np.zeros(3,)
    epsilon = np.array([1000, 1, 200])
    for i in range(3):
        z[i] = max([p.f_i[i] for p in population])
    z = z + epsilon
    return z

def get_reference_point_r2_dynamic(population):
    z = np.zeros(3,)
    for i in range(3):
        z[i] = min([p.f_i[i] for p in population])
    return z

def fitness_asigment(population, k, indicator, w_r2, weights=[]):
    if indicator == 'r2' or indicator == 'ws':
        z_r2 = get_reference_point_r2_dynamic(population)
        if w_r2.shape[0] != len(population):
            print(f'{w_r2.shape[0]} - {len(population)}')
            raise('no enought weights')
    if indicator == 'hv' or indicator == 'ws':
        ref_hv = get_reference_point_hv_dynamic(population)
        ind_hv = HV(ref_hv)
    for p in population:
        p_exc = [q for q in population if q.id != p.id]
        if len(p_exc) != len(population) - 1:
            raise('not excluding')
        if indicator == 'eps':
            one_all_indicator = [-1 * math.exp(-1 * indicator_eps(q, p)/k) for q in p_exc]
            p.fitness = sum(one_all_indicator)
        elif indicator == 'r2':
            one_all_indicator = [-1 * math.exp(-1 * indicator_r2(q, p, w_r2, z_r2) / k) for q in p_exc]
            p.fitness = sum(one_all_indicator)
        elif indicator == 'hv':
            one_all_indicator = [-1 * math.exp(-1 * indicator_hd(q, p, ind_hv) / k) for q in p_exc]
            p.fitness = sum(one_all_indicator)
        elif indicator == 'ws':
            one_all_indicator_eps = [-1 * math.exp(-1 * indicator_eps(q, p) / k[0]) for q in p_exc]
            fit_eps = sum(one_all_indicator_eps)
            one_all_indicator_hv = [-1 * math.exp(-1 * indicator_hd(q, p, ind_hv) / k[1]) for q in p_exc]
            fit_hv = sum(one_all_indicator_hv)
            one_all_indicator_r2 = [-1 * math.exp(-1 * indicator_r2(q, p, w_r2, z_r2) / k[2]) for q in p_exc]
            fit_r2 = sum(one_all_indicator_r2)
            p.fitness = weights[0] * fit_eps + weights[1] * fit_hv + weights[2] * fit_r2

    if len(population) != 0:
        min_fitness = min([p.fitness for p in population])
        for p in population:
            p.fitness += abs(min_fitness)

def selection_stage(current_population, k, indicator, w_r2_all, tournament_size=5, weights_ws=[]):
    w_r2 = []
    n = len(current_population)
    if indicator == 'r2' or indicator == 'ws':
        w_r2 = get_reference_directions("energy", 3, len(current_population), seed=1)
    fitness_asigment(current_population, k, indicator, w_r2, weights_ws)
    parent_population = []
    while len(parent_population) < n//2:
        tournament = random.choices(current_population, k=tournament_size)
        tournament.sort(key=lambda x: x.fitness)
        winner = tournament[-1]
        if not winner in parent_population:
            parent_population.append(winner)
    return parent_population



def crossover_stage(current_population, k, indicator, w_r2_all, tournament_size=5, weights_ws=[], prob_cross=0.8):
    n = len(current_population)
    parent_population = selection_stage(current_population, k, indicator, w_r2_all, tournament_size, weights_ws)
    child_population = []
    # for tiny instances
    attempts = 2*n
    while len(child_population) < n and attempts > 0:
        attempts -= 1
        parents = random.sample(parent_population, 2)
        first = parents[0]
        second = parents[1]
        childs = first.crossover(second, prob_cross=prob_cross)
        if len(childs) == 0:
            continue
        if len(child_population) + len(childs) > n:
            child_population.append(childs[0])
        else:
            child_population = child_population + childs
    return child_population

def mutation_stage(population, prob_mut=0.1):
    for individual in population:
        individual.mutation(prob_mut)


def lns(current_population, params, n_best):
    current_population.sort(key=lambda x: x.fitness, reverse=True)
    solutions_unwrap = [s.solution for s in current_population[:n_best]]
    solutions = mdls(solutions_unwrap, params)
    for i in range(n_best):
        c = current_population[i]
        #print(f'before LNS {c.f_i}')
        c.solution = solutions[i]
        c.f_i = np.array([solutions[i].f_1, solutions[i].f_2, solutions[i].f_3])
        #print(f'after LNS {c.f_i} {Solution.evaluations}')


def ibaco_indicator(params, pheromone_matrix, indicator, cooperative_mode, execution_n, apply_lns=False):
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    rho = params['rho']
    days = params['days']
    Q = params['Q']
    timetables = params['timetables']
    p_mut = params['p_mut']
    prob_cross = params['p_cross']
    epsilon = params['epsilon']
    dy = params['dy']
    max_iterations = params['ibaco-' + indicator]['max_iterations']
    A = []
    front = []
    k = params['ibaco-'+indicator]['k_fitness']
    weights_ws = []
    if indicator == 'ws':
        weights_ws = params['ibaco-ws']['weights']
    all_solutions = []
    log_hypervolume = []
    log_solutions_added = []
    log_pheromone = []
    ref_point_hv = utils.get_reference_point_file(params['file'])
    ind = HV(ref_point=ref_point_hv)
    w_r2 = []
    n_ants = params['n_ants']
    with open('references_r2/references_' + str(n_ants) + '.pkl', 'rb') as f:
        w_r2_all = pickle.load(f)
    start = time.time()
    log_evaluations = []

    for i in range(max_iterations):
        current_population = build_solutions(params, pheromone_matrix)
        current_population = wrap_ibaco(current_population, indicator)
        print (f'after built  {Solution.evals}')
        n = len(current_population)
        crossover_mutation = crossover_stage(current_population, k, indicator, w_r2_all, tournament_size=5, weights_ws=weights_ws, prob_cross=prob_cross)
        print(f'after cross  {Solution.evals}')
        mutation_stage(crossover_mutation, p_mut)
        print(f'after mut  {Solution.evals}')
        current_population += crossover_mutation
        if i == 0:
            current_evaluations = 0
            algorithm = 'ibaco-' + indicator
            if apply_lns:
                algorithm += '-lns'
            log_evaluations = [(current_evaluations, np.array([[s.solution.f_1, s.solution.f_2, s.solution.f_3] for s in current_population]))]
            utils.save_evaluations(algorithm, params['file'], execution_n, log_evaluations)
        if apply_lns:
            lns(current_population, params['lns'], n_best=len(current_population))

        all_solutions += [[c.solution.f_1, c.solution.f_2, c.solution.f_3] for c in current_population]

        A, solutions_added = archive_update_pqedy(A, current_population, epsilon, dy)
        front, _ = non_dominated(front, [c.solution for c in current_population])
        start1 = time.time()
        j = 0
        while len(current_population) > n:
            if indicator == 'r2' or indicator == 'ws':
                w_r2 = w_r2_all[j]
                j += 1
            fitness_asigment(current_population, k, indicator, w_r2, weights_ws)
            minimum = [(c.fitness, c) for c in current_population]
            minimum.sort(key=lambda x: x[0])
            minimum = minimum[0]
            current_population.remove(minimum[1])
            #if len(current_population) % 50 == 0:
            #    print (f'{indicator} | iteration {i}/{max_iterations} - {len(current_population)} / {n}')
        print (f'it {i} - fitness {Solution.evals} {time.time() - start1} | execution {execution_n}')

        update_pheromone_indicator(pheromone_matrix, current_population, rho, Q, timetables, days)
        #log_pheromone.append((np.copy(pheromone_matrix['AM'][0]), [(s.fitness, s.f_i) for s in current_population]))

        hyp = [(s.solution.f_1, s.solution.f_2, s.solution.f_3) for s in A]
        hyp = np.array(hyp)
        hyp = ind(hyp)
        print (f'{indicator} | it: {i} hypervolume: {hyp}')
        log_hypervolume.append(hyp)
        log_solutions_added.append(len(A))
        if not cooperative_mode:
            algorithm = 'ibaco-' + indicator
            if apply_lns:
                algorithm += '-lns'
            current_evaluations = Solution.evals
            log_evaluations = [(current_evaluations, np.array([[s.solution.f_1, s.solution.f_2, s.solution.f_3] for s in A]))]
            utils.save_evaluations(algorithm, params['file'], execution_n, log_evaluations)
    #utils.save_pheromone('ibaco-' + indicator, params['file'], execution_n, log_pheromone)
    duration = time.time() - start
    for a in A:
        a.solution.is_feasible()
    statistics = get_statistics([a.solution for a in A], log_hypervolume, log_solutions_added, duration)
    front = [[c.f_1, c.f_2, c.f_3] for c in front]
    front = np.array(front)
    all_solutions = np.array(all_solutions)
    return A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front


def cooperative_ibaco(params, n_execution=0, apply_lns=False):
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    matrices = []
    days = params['days']
    rho = params['rho']
    Q = params['Q']
    timetables = params['timetables']
    costumers = params['costumers']
    indicators = params['cibaco']['indicators']
    k = len(indicators)
    epsilon = params['epsilon']
    dy = params['dy']
    n_costumers = len(costumers)
    max_iterations_cibaco = params['cibaco']['max_iterations']
    nmig = params['cibaco']['nmig']
    all_solutions = np.empty((1, 3))
    log_hypervolume = []
    log_solutions_added = []
    log_evaluations = []
    reference_hv = utils.get_reference_point_file(params['file'])
    ind = HV(ref_point=reference_hv)
    for i in range(k):
        m = initialize_multiple_matrix_rand(days, n_costumers)
        matrices.append(m)
    A = []
    front = []
    start = time.time()
    for i in range(max_iterations_cibaco):
        PS = []
        for j in range(k):
            indicator = indicators[j]
            pheromone_matrix = matrices[j]
            P,_,_,_,_,_,all_solutions_p,_ = ibaco_indicator(params, pheromone_matrix, indicator, True, n_execution, apply_lns)
            all_solutions = np.concatenate((all_solutions, all_solutions_p), axis=0)
            for p in P:
                p.solution.algorithm = indicator
            PS.append(P)
        for P in PS:
            A, solutions_added = archive_update_pqedy(A, P, epsilon, dy)
            front, _ = non_dominated(front, [p.solution for p in P])
        ES = migration(A, nmig, k, params['cibaco']['k_fitness'], indicators)
        for j in range(k):
            solutions = [s for s in ES[j]]
            update_pheromone_indicator(matrices[j], solutions, rho, Q, timetables, days)
        hyp = [(s.solution.f_1, s.solution.f_2, s.solution.f_3) for s in A]
        hyp = np.array(hyp)
        hyp = ind(hyp)
        log_hypervolume.append(hyp)
        log_solutions_added.append(len(A))
        current_evaluations = Solution.evals
        log_evaluations = [(current_evaluations, np.array([[s.solution.f_1, s.solution.f_2, s.solution.f_3] for s in A]))]
        if apply_lns:
            alg = 'cmibaco-lns'
        else:
            alg = 'cmibaco'
        utils.save_evaluations(alg, params['file'], n_execution, log_evaluations)
        print (f'CMIBACO iteration {i} | evals {current_evaluations} hyp: {hyp}')
    all_solutions = all_solutions[1:,:]
    front = [[c.f_1, c.f_2, c.f_3] for c in front]
    front = np.array(front)
    duration = time.time() - start
    statistics = get_statistics([a.solution for a in A], log_hypervolume, log_solutions_added, duration)
    print (f'>>>> min time_tour {statistics["min_time_tour"]}, min arrival {statistics["min_arrival_time"]}, min vehicle {statistics["min_vehicle"]} - max time_tour {statistics["max_time_tour"]}, max arrival {statistics["max_arrival_time"]}, max vehicle {statistics["max_vehicle"]}')
    for a in A:
        a.solution.is_feasible()
    return A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front

def migration(A, nmig, k, k_indicator, indicators):
    ES = []
    for j in range(k):
        external_solutions = []
        random_nmig = [a for a in A if a.solution.algorithm != indicators[j]]
        if len(random_nmig) >= nmig:
            external_solutions = sample(random_nmig, nmig)
            w_r2 = get_reference_directions("energy", 3, nmig, seed=1)
            fitness_asigment(external_solutions, k_indicator[j], indicators[j], w_r2)
        ES.append(external_solutions)
    return ES