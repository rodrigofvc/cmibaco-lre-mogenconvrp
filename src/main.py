from costumer import Costumer
from vehicle import Vehicle
from ant import Ant
from reader import read_dataset
from maco import *

import matplotlib.pyplot as plt

DATASET_DIR = 'dataset/Christofides_8_5_0.5.txt'
#DATASET_DIR = 'dataset/convrp_12_test_1.vrp'

def plot_archive_3d(A):
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
    plt.title('Soluciones en espacio objetivo')
    #plt.show()
    plt.savefig('3d-archive.png')
    plt.close()

def plot_archive_2d(A):
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
        plt.title('Soluciones no dominadas con ' + str(n_vehicle) + ' vehiculos distintos')
        plt.savefig('archive-' + str(n_vehicle) + '.png')
        plt.close()

if __name__ == '__main__':
    costumers, capacity, days, limit_time = read_dataset(DATASET_DIR)
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

    n_groups = 300
    rho = 0.6
    alpha = 4
    beta = 5
    gamma = 3
    delta = 3
    Q = 3
    q0 = 0.30
    max_iterations = 200
    min_pheromone = 10e-7
    timetables = ['AM', 'PM']

    A = maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone)
    plot_archive_3d(A)
    plot_archive_2d(A)
