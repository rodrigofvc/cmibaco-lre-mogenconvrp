import sys

from uvrp import lightly_robust_solutions

if __name__ == '__main__':
    algorithm = sys.argv[1]
    dir_approx = sys.argv[2]
    dataset = sys.argv[3]
    lightly_robust_solutions(dataset, algorithm, dir_approx)

