from solution import Solution
from maco import get_costumers_day_timetable
from ant import Ant
import copy
import numpy as np

def random_multiple_matrix(days, n_costumers):
    matrices = {'AM': [], 'PM': []}
    for d in range(days):
        m = np.random.rand(n_costumers,n_costumers)
        m_ = np.random.rand(n_costumers,n_costumers)
        matrices['AM'].append(m)
        matrices['PM'].append(m_)
    return matrices


def get_solution(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    n_costumers = len(costumers)
    pheromone_matrix = random_multiple_matrix(days, n_costumers)
    delta_ant_matrix = random_multiple_matrix(days, n_costumers)
    s = Solution(timetables, days)
    for h in timetables:
        vehicles_timetable = copy.deepcopy(vehicles)
        costumers_h = get_costumers_day_timetable(costumers, h)
        costumers_timetable = copy.deepcopy(costumers_h)
        for d in range(days):
            depot = costumers[0]
            ant = Ant(depot, n_costumers, min_pheromone, max_pheromone)
            ant.build_solution(delta_ant_matrix, pheromone_matrix, d, h, alpha, beta, gamma, delta, Q, costumers_timetable, vehicles_timetable, q0)
            s.add_ant_timetable_day(ant, h)
        s.add_assigment_vehicles(vehicles_timetable, costumers_timetable, h)
    s.get_fitness()
    return s


def get_neighboorhod(solution, n, prob_mut):
    neighbors = []
    for _ in range(n):
        new_neighbor = solution.mutation(prob_mut)
        while new_neighbor == solution:
            new_neighbor = solution.mutation(prob_mut)
        neighbors.append(new_neighbor)
    return neighbors

def weight_sum(solution, weights):
    sum_1 = solution.f_1 * weights[0] + solution.f_2 * weights[1] + solution.f_3 * weights[2]
    return sum_1

def is_local_optima(solution, neighbors, weights):
    solution_weight = weight_sum(solution, weights)
    weight_sums = [weight_sum(s, weights) for s in neighbors]
    i_argmin = np.argmin(weight_sums)
    argmin = weight_sums[i_argmin]
    if argmin < solution_weight:
        return False, neighbors[i_argmin]
    return True, None

def get_local_optimal(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, weights):
    n_neighboors = 10
    solution = get_solution(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    neighbors = get_neighboorhod(solution, n_neighboors, p_mut)
    is_local = False
    while not is_local:
        is_local, new_local = is_local_optima(solution, neighbors, weights)
        if not is_local:
            solution = new_local
            neighbors = get_neighboorhod(solution, n_neighboors, p_mut)
    return solution

def get_n_nodes(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    n_nodes = 10000
    nodes = []
    weights = np.random.rand(3,)
    while len(nodes) != n_nodes:
        print (f'BUILDING NODES:   {len(nodes)} /{n_nodes}')
        local = get_local_optimal(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, weights)
        if not local in nodes:
            nodes.append(local)
    return nodes

def get_edges(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, nodes, attempts):
    edges = []
    for l, node in enumerate(nodes):
        current_node = node
        for k in range(attempts):
            next = current_node.mutation(p_mut)
            if next in nodes and next != node:
                is_contained = [p for p in edges if p[0] == node and p[1] == next]
                if len(is_contained) != 0:
                    edge = is_contained[0]
                    edge[2] += 1
                else:
                    new_edge = (node, next, 1)
                    edges.append(new_edge)
            current_node = next
            print(f'ADDING EDGES:  {len(edges)} {k} | node {l} ')
    return edges

def local_optimal_net(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy):
    nodes = get_n_nodes(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy)
    attempts = 900
    edges = get_edges(n_groups, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables, vehicles, q0, min_pheromone, max_pheromone, p_mut, epsilon, dy, nodes, attempts)
    print (f'EDGES >>>>>>>>>>>>>>>>>> {len(nodes)} - {len(edges)}')
    for edge in edges:
        print(f'{edge[0]} - {edge[1]} - {edge[2]}')










    pass
