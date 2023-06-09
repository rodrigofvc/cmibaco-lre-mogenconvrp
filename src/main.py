from costumer import Costumer
from vehicle import Vehicle
from ant import Ant
from reader import read_dataset
from maco import maco
import os
import matplotlib.pyplot as plt
import random
import json
import sys

def save_result(dataset, A, path):
    if '.txt' in dataset:
        new_path = 'results/' + dataset.replace('.txt', '') + '/'
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        plt.savefig(new_path + dataset.replace('.txt', '') + '-' + path)
    elif '.vrp' in dataset:
        new_path = 'results/' + dataset.replace('.vrp', '') + '/'
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        plt.savefig(new_path + dataset.replace('.vrp', '') + '-' + path)
    else:
        raise()

def plot_archive_3d(A, dataset):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = []
    ys = []
    zs = []
    for s in A:
        xs.append(s.f_1)
        ys.append(s.f_2)
        zs.append(s.f_3)
    ax.scatter(xs, ys, zs, marker='o')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Diferencia máxima')
    ax.set_zlabel('Vehículos distintos')
    plt.title('Solutions in objective space for ' + dataset)
    save_result(dataset, A, '3d-archive.png')
    plt.close()

def plot_archive_2d(A, dataset):
    different_vehicles = [a.f_3 for a in A]
    different_vehicles = list(set(different_vehicles))
    different_vehicles.sort()
    for n_vehicle in different_vehicles:
        a_vehicle = [a for a in A if a.f_3 == n_vehicle]
        xs = [a.f_2 for a in a_vehicle]
        ys = [a.f_1 for a in a_vehicle]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(xs, ys, marker='o')
        ax.set_xlabel('Differencia máxima de llegada')
        ax.set_ylabel('Tiempo')
        plt.title('Soluciones e no dominadas con ' + str(n_vehicle) + ' vehiculos distintos ' + dataset)
        save_result(dataset, A, 'archive-' + str(n_vehicle) + '.png')
        plt.close()

def plot_log_hypervolume(log, dataset, params):
    plt.title('Indicador de hypervolumen ' + dataset)
    plt.xlabel('Iteración')
    plt.ylabel('Hypervolumen')
    plt.plot(log)
    if '.txt' in dataset:
        dataset = dataset.replace('.txt', '')
        plt.savefig('results/' + dataset + '/'+ dataset+ '-log-hyper.png')
    elif '.vrp' in dataset:
        dataset = dataset.replace('.vrp', '')
        plt.savefig('results/' + dataset + '/'+ dataset+ '-log-hyper.png')
    plt.close()

def exec_maco(params):
    dataset = params['file']
    dir = 'dataset/'
    costumers, capacity, days, limit_time = read_dataset(dir + dataset)
    depot = Costumer(0,0,0)
    depot.demands = [0] * days
    depot.arrival_times = [0] * days
    depot.vehicles_visit = [-1] * days
    depot.service_times = [0] * days

    costumers.insert(0, depot)
    vehicles = []
    for i in range(len(costumers)):
        vehicle = Vehicle(i, capacity, days, limit_time/2)
        vehicles.append(vehicle)

    seed = params['seed']
    n_groups = params['n_ants']
    rho = params['rho']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    Q = params['Q']
    q0 = params['q0']
    max_iterations = params['max_iterations']
    min_pheromone = 10e-3
    max_pheromone = 10e5
    timetables = ['AM', 'PM']
    epsilon = params['epsilon']
    dy = params['dy']
    random.seed(seed)
    print (f'>>>>>>>>>>>>>>>> Results for {dataset} in results/{dataset}')
    A, log_hypervolume, duration = maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, epsilon, dy)
    plot_archive_3d(A, dataset)
    plot_archive_2d(A, dataset)
    plot_log_hypervolume(log_hypervolume, dataset, params)
    dataset = dataset.replace('.txt', '')
    print (f'>> Non Epsilon dominated for {dataset}')
    for a in A:
        print ((a.f_1, a.f_2, a.f_3))
    print (f'Elapsed time {duration} seconds')


if __name__ == '__main__':
    params_file = sys.argv[1]
    f = open(params_file)
    params = json.load(f)
    exec_maco(params)
