from costumer import Costumer
from vehicle import Vehicle
from ant import Ant
from reader import read_dataset
from maco import *

DATASET_DIR = 'dataset/Christofides_7_5_0.5.txt'
DATASET_DIR = 'dataset/convrp_12_test_1.vrp'

if __name__ == '__main__':
    costumers, capacity, days = read_dataset(DATASET_DIR)
    depot = Costumer(0,0,0)
    depot.demands = [0] * days
    depot.arrival_times = [0] * days
    depot.vehicles_visit = [-1] * days
    depot.service_times = [0] * days

    costumers.insert(0, depot)
    vehicles = []
    for i in range(len(costumers)):
        vehicle = Vehicle(i, capacity, days)
        vehicles.append(vehicle)

    n_groups = 100
    rho = 0.5
    alpha = 0.4
    beta = 0.5
    gamma = 0.8
    delta = 0.4
    Q = 0.50
    max_iterations = 40
    timetables = ['AM', 'PM']

    maco(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles)
