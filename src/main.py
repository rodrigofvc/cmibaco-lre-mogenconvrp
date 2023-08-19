from costumer import Costumer
from vehicle import Vehicle
from ant import Ant
from reader import read_dataset
from maco import maco, maco_genetic_algorithm, maco_nsgaiii
from ibaco import ibaco
from lon import local_optimal_net
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import json
import sys
from ibaco import ibaco_lns

def save_result(dataset, path):
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
    save_result(dataset, '3d-archive.png')
    plt.close()

def plot_sub_vrp(solution, timetable, day, axs, vehicle_colors):
    tours = solution.get_vector_representation_dt(timetable, day)
    tour = []
    subtours = []
    for c in tours[1:]:
        if c.id == 0:
            subtours.append(tour)
            tour = []
        else:
            tour.append(c)
    subtours.append(tour)
    vehicles_used = []
    for t in subtours:
        vehicle_id = t[0].vehicles_visit[day]
        for c in t:
            if c.vehicles_visit[day] != vehicle_id:
                raise()
        vehicles_used.append(vehicle_id)
        x_tour = [0] + [c.x for c in t] + [0]
        y_tour = [0] + [c.y for c in t] + [0]
        x_tour = np.array(x_tour)
        y_tour = np.array(y_tour)
        vehicle_color = [v for v in vehicle_colors if v[0] == vehicle_id]
        axs.plot(x_tour, y_tour, c=vehicle_color[0][1])
        axs.plot(x_tour[1:-1], y_tour[1:-1], 'o')
        axs.plot([0],[0], '^', c='red')
    axs.set_title(timetable + ' ' + str(day))
    return vehicles_used

def plot_best_objective(A, dataset, objective):
    if objective == 0:
        best = [a.f_1 for a in A]
        title = 'Best total tours time '
        fig_name = 'best-solution-time-tour.png'
    elif objective == 1:
        best = [a.f_2 for a in A]
        title = 'Best maximum arrival time difference '
        fig_name = 'best-solution-arrival-diff.png'
    else:
        best = min([a.f_3 for a in A])
        best = [a.f_1 for a in A if a.f_3 == best]
        title = 'Best consistency driver '
        fig_name = 'best-solution-driver-diff.png'
    if objective == 2:
        best = min(best)
        best = [a for a in A if a.f_1 == best][0]
    else:
        ibest = np.argmin(best)
        best = A[ibest]
    title += '[' + str(best.f_1) + ', ' + str(best.f_2) + ', ' + str(best.f_3) + ']'
    fig, axs = plt.subplots(len(best.timetables), best.days)
    fig.set_figheight(18)
    fig.set_figwidth(20)
    vehicles_used = []
    np.random.seed(5)
    vehicle_colors = [(v.id, tuple(np.random.rand(3,))) for v in best.assigments_vehicles]
    for i, t in enumerate(best.timetables):
        for d in range(best.days):
            vehicles_used += plot_sub_vrp(best, t, d, axs[i, d], vehicle_colors)
    vehicles_used = list(set(vehicles_used))
    colors_patch = [mpatches.Patch(color=v[1], label='Vehicle ' + str(v[0])) for v in [vehicle_colors[i] for i in vehicles_used]]
    fig.legend(handles=colors_patch)
    fig.suptitle(title)
    save_result(dataset, fig_name)
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
        save_result(dataset, 'archive-' + str(n_vehicle) + '.png')
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
        # Paper says 1000
        vehicle = Vehicle(i, capacity, days, 1000)
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
    p_mut = params['p_mut']
    timetables = ['AM', 'PM']
    epsilon = params['epsilon']
    dy = params['dy']
    random.seed(seed)
    np.random.seed(seed)
    print (f'>>>>>>>>>>>>>>>> Results for {dataset} in results/{dataset}')
    #local_optimal_net(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    #A, log_hypervolume, duration = maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    #A, log_hypervolume, duration = ibaco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    A, log_hypervolume, duration = ibaco_lns(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    plot_best_objective(A, dataset, 0)
    plot_best_objective(A, dataset, 1)
    plot_best_objective(A, dataset, 2)
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
