import argparse
import sys

sys.path.insert(0, "../")
from reader import read_dataset
from cibaco import cooperative_ibaco

def target_cmibaco(args):
    n_ants = 10
    min_pheromone = 10e-3
    max_pheromone = 10e5
    epsilon = [9000, 5, 550]
    dy = [100, 0, 50]
    cibaco_iterations = 100
    problem = args.instance
    dir = '../dataset/' + problem
    costumers, vehicles, capacity, days, _ = read_dataset(dir)
    params = {
        'n_ants': n_ants,
        'file': problem,
        'rho': args.rho,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'delta': args.delta,
        'Q': args.Q,
        'q0': args.q0,
        'p_mut': args.p_mut,
        'p_cross': args.p_cross,
        'seed' : args.seed,
        'min_pheromone' : min_pheromone,
        'max_pheromone' : max_pheromone,
        'epsilon' : epsilon,
        'dy' : dy,
        'cibaco' : {
            'max_iterations': cibaco_iterations,
            'k_fitness': [10e2, 10e6, 10e2],
            'indicators': ["eps", "hv", "r2"],
            'nmig': args.cmibaco_nmig
        },
        'ibaco-eps' : {
            "max_iterations": 1,
            "k_fitness": 10e2
        },
        'ibaco-hv': {
            "max_iterations": 1,
            "k_fitness": 10e6
        },
        'ibaco-r2': {
            "max_iterations": 1,
            "k_fitness": 10e2
        },
        'lns': {
            'max_iterations': 1,
            'n_removes': args.lns_removes,
            'eta': args.lns_eta,
            'delta': args.lns_delta,
            'ub_1': args.lns_ub_1,
            'ub_2': args.lns_ub_2
        }
    }
    params['vehicles'] = vehicles
    params['costumers'] = costumers
    params['days'] = days
    params['timetables'] = ['AM', 'PM']
    _, log_hypervolume, _, _, _, _, _, _ = cooperative_ibaco(params, -1, apply_lns=True, tuning=True)
    return -1 * log_hypervolume[-1]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Hiper parameter optimization for cMIBACO')
    ap.add_argument('--i', dest='instance', required=True)
    ap.add_argument('--seed', dest='seed', type=int, required=True)
    ap.add_argument('--rho', dest='rho', type=float, required=True)
    ap.add_argument('--alpha', dest='alpha', type=float, required=True)
    ap.add_argument('--beta', dest='beta', type=float, required=True)
    ap.add_argument('--gamma', dest='gamma', type=float, required=True)
    ap.add_argument('--delta', dest='delta', type=float, required=True)
    ap.add_argument('--Q', dest='Q', type=float, required=True)
    ap.add_argument('--q0', dest='q0', type=float, required=True)
    ap.add_argument('--p_mut', dest='p_mut', type=float, required=True)
    ap.add_argument('--p_cross', dest='p_cross', type=float, required=True)
    ap.add_argument('--cmibaco_nmig', dest='cmibaco_nmig', type=int, required=True)
    ap.add_argument('--lns_removes', dest='lns_removes', type=int, required=True)
    ap.add_argument('--lns_eta', dest='lns_eta', type=int, required=True)
    ap.add_argument('--lns_delta', dest='lns_delta', type=float, required=True)
    ap.add_argument('--lns_ub_1', dest='lns_ub_1', type=int, required=True)
    ap.add_argument('--lns_ub_2', dest='lns_ub_2', type=int, required=True)
    print ('Tunning cMIBACO')
    args = ap.parse_args()
    score = target_cmibaco(args)
    print (score)
