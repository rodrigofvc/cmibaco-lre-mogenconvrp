import os

from cibaco import cooperative_ibaco, ibaco_indicator
from cmibaco_components import cooperative_ibaco_components
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
    elif algorithm == 'cmibaco' or algorithm == 'cmibaco-lns' or algorithm == 'cmibaco-cross' or algorithm == 'cmibaco-mut' or algorithm == 'cmibaco-base':
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
    elif algorithm == 'ibaco-hv-lns' or algorithm == 'ibaco-r2-lns' or algorithm == 'ibaco-eps-lns' or algorithm == 'ibaco-ws-lns':
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
    elif algorithm == 'ibaco-eps-lns' or algorithm == 'ibaco-hv-lns' or algorithm == 'ibaco-r2-lns' or algorithm == 'ibaco-ws-lns':
        if algorithm == 'ibaco-eps-lns':
            indicator = 'eps'
        elif algorithm == 'ibaco-hv-lns':
            indicator = 'hv'
        elif algorithm == 'ibaco-r2-lns':
            indicator = 'r2'
        elif algorithm == 'ibaco-ws-lns':
            indicator = 'ws'
        days = params['days']
        costumers = params['costumers']
        n_costumers = len(costumers)
        pheromone_matrix = initialize_multiple_matrix_rand(days, n_costumers)
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = ibaco_indicator(params, pheromone_matrix, indicator, False, n_execution, apply_lns=True)
        A = [a.solution for a in A]
    elif algorithm == 'cmibaco':
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = cooperative_ibaco(params, n_execution)
        A = [a.solution for a in A]
    elif algorithm == 'cmibaco-lns':
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = cooperative_ibaco(params, n_execution, apply_lns=True)
        A = [a.solution for a in A]
    elif algorithm == 'cmibaco-cross':
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = cooperative_ibaco_components(params, n_execution, apply_lns=False, apply_crossover=True, apply_mutation=False)
        A = [a.solution for a in A]
    elif algorithm == 'cmibaco-mut':
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = cooperative_ibaco_components(params, n_execution, apply_lns=False, apply_crossover=False, apply_mutation=True)
        A = [a.solution for a in A]
    elif algorithm == 'cmibaco-base':
        A, log_hypervolume, log_solutions_added, duration, statistics, log_evaluations, all_solutions, front = cooperative_ibaco_components(params, n_execution, apply_lns=False, apply_crossover=False, apply_mutation=False, classic=True)
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

def check_dirs():
    if not os.path.isdir('results/'):
        os.makedirs('results')
    if not os.path.isdir('boxplot/'):
        os.makedirs('boxplot')
    if not os.path.isdir('boxplot-uvrp/'):
        os.makedirs('boxplot-uvrp')
    if not os.path.isdir('uncertainty/'):
        os.makedirs('uncertainty')
    if not os.path.isdir('fronts/'):
        os.makedirs('fronts')
    if not os.path.isdir('medians-iterations/'):
        os.makedirs('medians-iterations')
    if not os.path.isdir('medians-comparation/'):
        os.makedirs('medians-comparation')

if __name__ == '__main__':
    check_dirs()
    algorithm = sys.argv[1]
    params = sys.argv[2]
    dataset = sys.argv[3]
    exec_batch(algorithm, params, dataset)
