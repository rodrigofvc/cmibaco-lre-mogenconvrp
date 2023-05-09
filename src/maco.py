import numpy as np

def initialize_multiple_matrix(days, n_costumers):
    matrices = {'AM': [], 'PM': []}
    for d in range(days):

        m = np.zeros((n_costumers,n_costumers))
        matrices['AM'].append(m)

        m_ = np.zeros((n_costumers,n_costumers))
        matrices['PM'].append(m_)
    return matrices

def get_costumers_day_timetable(costumers, day, timetables):
    pass


def maco(n_ants, rho, days, alpha, beta, gamma, delta, Q, max_iterations, costumers, timetables):
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers)
    for i in range(max_iterations):
        P = []
        for k in range(n_ants):
            S = []
            for d in range(days):
                for h in timetables:
                    costumers_dh = get_costumers_day_timetable(costumers, d, timetables)
                    #s = ant.build_solution(days, alpha, beta, gamma, delta, Q, )
                    #S.append(s)
        P.append(S)
    #A = non_dominated(P)
    #update_pheromone(pheromone_matrix, P)
