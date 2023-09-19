from nsgaiii_algorithm import nsgaiii
from point import Point
import numpy as np
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions

if __name__ == '__main__':
    Point.counter = 0
    initial_population = []
    iterations = 500
    points = np.random.uniform(1,35,(30,3))
    for p in points:
        initial_population.append(Point(None, p))
        print (p)

    final_population = nsgaiii(initial_population, iterations)

    points = []
    print ('>>>>>>>>>>>>')
    for p in final_population:
        points.append(p.f_i)
        print (p.f_i)

    points = np.array(points)

    reference_points = get_reference_directions("energy", 3, 15, seed=1)
    plot = Scatter()
    plot.add(reference_points)
    plot.add(points)
    plot.show()
