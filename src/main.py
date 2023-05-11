from costumer import Costumer
from vehicle import Vehicle
from ant import Ant
from reader import read_dataset
from maco import *

DATASET_DIR = 'dataset/Christofides_7_5_0.5.txt'

if __name__ == '__main__':
    depot = Costumer(0,0,0)
    depot.demands = [0,0,0,0,0]
    depot.arrival_times = [0,0,0,0,0]
    depot.vehicles_visit = [-1,-1,-1,-1,-1]
    depot.service_times = [0,0,0,0,0]
    costumers, capacity, days = read_dataset(DATASET_DIR)
    costumers.insert(0, depot)
    vehicles = []
    for i in range(len(costumers)):
        vehicle = Vehicle(i, capacity, days)
        vehicles.append(vehicle)

    n_ants = 10
    rho = 0.4
    alpha = 0.4
    beta = 0.2
    gamma = 0.7
    delta = 0.3
    Q = 0.50
    max_iterations = 4
    timetables = ['AM', 'PM']

    maco(n_ants, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles)
