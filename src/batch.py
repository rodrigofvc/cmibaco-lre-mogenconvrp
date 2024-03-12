import os

from cibaco import cooperative_ibaco, ibaco_indicator
from lns import external_mdls
from maco import initialize_multiple_matrix_rand
from utils import get_execution_dir, plot_best_objective, plot_archive_3d, plot_archive_2d, \
    plot_log_hypervolume, plot_log_solutions_added, save_params, save_archive, save_statistics, save_evaluations, \
    save_all_solutions, save_front, plot_front_epsilon_front
from reader import read_dataset
from solution import Solution
import numpy as np
import random
import json
import sys

def exec_maco(params, algorithm, n_execution=0):
    dataset = params['file']
    dir = 'dataset/'
    costumers, vehicles, capacity, days, limit_time = read_dataset(dir + dataset)
    timetables = ['AM', 'PM']

    params['vehicles'] = vehicles
    params['costumers'] = costumers
    params['days'] = days
    params['timetables'] = timetables

    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    print (f'>>>>>>>>>>>>>>>> Results for {dataset} in results/{dataset} | {algorithm} | exec {n_execution}')

    exec_file(algorithm, params, dataset, n_execution)

def get_parameters(algorithm):
    if algorithm == 'ibaco-hv' or algorithm == 'ibaco-r2' or algorithm == 'ibaco-eps' or algorithm == 'ibaco-ws':
        params = ['params-ibaco-1.json', 'params-ibaco-2.json',
                  'params-ibaco-3.json', 'params-ibaco-4.json',
                  'params-ibaco-5.json', 'params-ibaco-6.json',
                  'params-ibaco-7.json', 'params-ibaco-8.json',
                  'params-ibaco-9.json', 'params-ibaco-10.json',
                  'params-ibaco-11.json', 'params-ibaco-12.json',
                  'params-ibaco-13.json', 'params-ibaco-14.json',
                  'params-ibaco-15.json', 'params-ibaco-16.json',
                  'params-ibaco-17.json', 'params-ibaco-18.json',
                  'params-ibaco-19.json', 'params-ibaco-20.json']
        return params
    elif algorithm == 'cmibaco' or algorithm == 'cmibaco-lns':
        params = ['params-cmibaco-1.json', 'params-cmibaco-2.json',
                  'params-cmibaco-3.json', 'params-cmibaco-4.json',
                  'params-cmibaco-5.json', 'params-cmibaco-6.json',
                  'params-cmibaco-7.json', 'params-cmibaco-8.json',
                  'params-cmibaco-9.json', 'params-cmibaco-10.json',
                  'params-cmibaco-11.json', 'params-cmibaco-12.json',
                  'params-cmibaco-13.json', 'params-cmibaco-14.json',
                  'params-cmibaco-15.json', 'params-cmibaco-16.json',
                  'params-cmibaco-17.json', 'params-cmibaco-18.json',
                  'params-cmibaco-19.json', 'params-cmibaco-20.json']
        return params

def exec_batch(algorithm, params_dir, dataset):
    if os.path.isfile(params_dir):
        f = open(params_dir)
        Solution.evaluations.get()
        Solution.evaluations.put(0)
        params = json.load(f)
        params['file'] = dataset
        exec_maco(params, algorithm)
        f.close()
        return
    elif os.path.isdir(params_dir):
        params_list = get_parameters(algorithm)
        for i, file in enumerate(params_list):
            f = open(params_dir + file)
            params = json.load(f)
            params['file'] = dataset
            Solution.evals = 0
            Solution.evaluations.get()
            Solution.evaluations.put(0)
            exec_maco(params, algorithm, i)
            f.close()
        return
    elif not os.path.exists(params_dir):
        raise('given path not exist')


def exec_algorithm(algorithm, params, n_execution):
    """"
    if algorithm == 'haco':
        A, log_hypervolume, log_solutions_added, duration, statistics = haco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    elif algorithm == 'maco':
        A, log_hypervolume, log_solutions_added, duration, statistics = maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    elif algorithm == 'ibaco-eps':
        A, log_hypervolume, log_solutions_added, duration, statistics = ibaco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, 'epsilon')
    elif algorithm == 'ibaco-hv':
        A, log_hypervolume, log_solutions_added, duration, statistics = ibaco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, 'hv')
    elif algorithm == 'nsgaiii-aco':
        A, log_hypervolume, log_solutions_added, duration, statistics = nsgaiii_aco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    elif algorithm == 'ribaco':
        A, log_hypervolume, log_solutions_added, duration, statistics = ibaco_lns(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, 'epsilon')
    elif algorithm == 'mdls':
        A, log_hypervolume, log_solutions_added, duration, statistics = external_mdls(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    """
    if algorithm == 'ibaco-eps' or algorithm == 'ibaco-hv' or algorithm == 'ibaco-r2' or algorithm == 'ibaco-ws':
        if algorithm == 'ibaco-eps':
            indicator = 'eps'
        elif algorithm == 'ibaco-hv':
            indicator = 'hv'
        elif algorithm == 'ibaco-r2':
            indicator = 'r2'
        elif algorithm == 'ibaco-ws':
            indicator = 'ws'
        days = params['days']
        costumers = params['costumers']
        n_costumers = len(costumers)
        pheromone_matrix = initialize_multiple_matrix_rand(days, n_costumers)
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = ibaco_indicator(params, pheromone_matrix, indicator, False, n_execution)
        A = [a.solution for a in A]
    elif algorithm == 'cmibaco':
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = cooperative_ibaco(params, n_execution)
        A = [a.solution for a in A]
    elif algorithm == 'cmibaco-lns':
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = cooperative_ibaco(params, n_execution, apply_lns=True)
        A = [a.solution for a in A]
    elif algorithm == 'mdlns':
        log_solutions_added = []
        log_evaluations = []
        all_solutions = []
        A, log_hypervolume, duration, statistics = external_mdls(params)
    return A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front


def exec_file(algorithm, params, file, execution_n=0):
    A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = exec_algorithm(algorithm, params, execution_n)
    execution_dir = get_execution_dir(file, algorithm)
    #plot_pheromone_matrices(matrices, file, execution_dir)
    plot_best_objective(A, file, 0, execution_dir)
    plot_best_objective(A, file, 1, execution_dir)
    plot_best_objective(A, file, 2, execution_dir)
    plot_archive_3d(A, file, execution_dir)
    plot_front_epsilon_front(front, A, all_solutions, file, execution_dir)
    plot_archive_2d(A, file, execution_dir)
    plot_log_hypervolume(log_hypervolume, file, execution_dir)
    if algorithm != 'mdls':
        plot_log_solutions_added(log_solutions_added, file, execution_dir)
    save_params(params, execution_dir)
    save_archive(A, execution_dir)
    save_front(front, execution_dir)
    save_all_solutions(all_solutions, execution_dir)
    save_statistics(statistics, execution_dir)
    print (f'>> Non Epsilon dominated for {file} / {execution_n} / {algorithm}')
    for a in A:
        print ((a.f_1, a.f_2, a.f_3))
    print (f'Elapsed time {duration} seconds')

def get_fitness_evals(iterations_cmib, iterations_ibaco, n, indicators=1):
    evaluations_fitness = 3*n*iterations_ibaco
    evaluations_fitness *= indicators
    evaluations_fitness *= iterations_cmib
    return evaluations_fitness


if __name__ == '__main__':
    #total = get_fitness_evals(100,1, 30, indicators=3)
    #print (f'total CIBACO fitness  {total}')
    #total = get_fitness_evals(1, 100, 90)
    #print (f'total IBACO-I fitness {total}')

    algorithm = sys.argv[1]
    params = sys.argv[2]
    dataset = sys.argv[3]
    exec_batch(algorithm, params, dataset)
