from costumer import Costumer
from vehicle import Vehicle
from reader import read_dataset
from maco import *


DATASET_DIR = 'dataset/Christofides_7_5_0.5.txt'

if __name__ == '__main__':
    depot = Costumer(0,0,0)
    costumers, capacity, days = read_dataset(DATASET_DIR)
    vehicles = []
    for i in range(len(costumers)):
        vehicle = Vehicle(i, capacity)
        vehicles.append(vehicle)

    n_ants = 10
    rho = 0.4
    days = 5
    alpha = 0.4
    beta = 0.2
    gamma = 0.7
    delta = 0.3
    Q = 0.50
    max_iterations = 10
    timetables = ['AM', 'PM']

    maco(n_ants, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables)
