import random
import copy
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

def nsgaiii(initial_population, iterations):
    current_population = initial_population
    while iterations != 0:
        for s in current_population:
            s.f_i = np.copy(s.original_f_i)
        reference_points = get_reference_directions("energy", 3, 40, seed=1)
        current_population = nsgaiii_step(current_population, reference_points, iterations)
        iterations -= 1

    points = []
    for p in current_population:
        points.append(p.f_i)
        print (p.f_i)
    points = np.array(points)
    reference_points = get_reference_directions("energy", 3, 40, seed=1)
    plot = Scatter()
    plot.add(reference_points)
    plot.add(points)
    #plot.show()
    return current_population

def nsgaiii_step(initial_population, reference_points, iteration):
    current_population = initial_population
    n = len(current_population)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>  initial population NSGAIII {iteration}')
    """
    for pop in current_population:
        print (pop.f_i)
    """
    crossover_mutation = crossover_stage(current_population)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>  crossover NSGAIII {iteration}')
    """
    for pop in crossover_mutation:
        print (pop.f_i)
    """
    mutation_stage(crossover_mutation)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>> mutation NSGAIII {iteration}')
    """
    for pop in crossover_mutation:
        print (pop.f_i)
    """
    result_pop = current_population + crossover_mutation
    if len(result_pop) != 2*n:
        raise(f'error {len(result_pop)} != 2n ({n})')
    fronts = fast_non_dominating_sorting(result_pop)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>> non dominated sort NSGAIII {iteration}')

    new_population = []
    i = 0
    j = 0
    i_front = fronts[i]
    while len(new_population) <= n:
        new_population += i_front
        j += 1
        if i + 1 != len(fronts):
            i += 1
            i_front = fronts[i]
        else:
            break
    if len(new_population) == n:
        current_population = new_population
        return current_population
    remaining_points = n - sum([len(f) for f in fronts[:j-1]])
    normalize(new_population)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>> normalize NSGAIII {iteration}')
    associate(new_population, reference_points)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>> associate NSGAIII {iteration}')
    fronts_no_last = [i for front in fronts[:j-1] for i in front]
    niche_count = get_niche_count(fronts_no_last, reference_points)
    remaining_population = niching(new_population, reference_points, niche_count, remaining_points, fronts[j-1], fronts_no_last)
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>> niching NSGAIII {iteration}')
    current_population = fronts_no_last + remaining_population
    if len(current_population) != n:
        print (f'current_population len :{len(current_population)} | n {n}')
        raise('no size required')
    return current_population


def niching(new_population, reference_points, niche_count, remaining_points, front, fronts_no_last):
    remaining_population = []

    while remaining_points != 0 and len(reference_points) > 0:
        for s in front:
            point = s.f_i
            distances = [distance_point_line(point, ref) for ref in reference_points]
            j = np.argmin(distances)
            s.closest_point = reference_points[j]
            s.distance_point = distances[j]

        p_i = np.argmin(niche_count)
        if p_i >= len(reference_points):
            print(f'niche count {niche_count}')
            print('reference ', reference_points)
        minimum_point = reference_points[p_i]

        # searching in F_{t+1} = S_t / F_l
        closest_indiv_list_current = [s for s in fronts_no_last if (s.closest_point == minimum_point).all()]
        p = len(closest_indiv_list_current)

        # searching in F_l
        closest_indiv_list_front = [s for s in front if (s.closest_point == minimum_point).all()]
        if len(closest_indiv_list_front) != 0:
            if p == 0:
                # add the point with minimum distance
                k = np.argmin([p.distance_point for p in closest_indiv_list_front])
                closest_indiv = closest_indiv_list_front[k]
                remaining_population.append(closest_indiv)
                front.remove(closest_indiv)
            else:
                # add random individual
                random_indv = random.choice(closest_indiv_list_front)
                remaining_population.append(random_indv)
                front.remove(random_indv)
            niche_count[p_i] += 1
            remaining_points -= 1
        else:
            reference_points = np.delete(reference_points, p_i, axis=0)
            niche_count = np.delete(niche_count, p_i, axis=0)
        if len(reference_points) != len(niche_count):
            print ('size ', len(reference_points), len(niche_count))
            print (reference_points)
            print (niche_count)
            raise('size dismatch')
    return remaining_population

def get_niche_count(population, reference_points):
    niche_count = []
    closest_point = [s.closest_point for s in population]
    for ref in reference_points:
        count = [(r == ref).all() for r in closest_point].count(True)
        niche_count.append(count)
    return np.array(niche_count)

def distance_point_line(point, reference_point):
    distance = point - np.dot(reference_point, point) / np.linalg.norm(reference_point)
    distance = np.linalg.norm(distance)
    return distance

def associate(new_population, reference_points):
    closest_point = []
    distance_point = []
    for s in new_population:
        point = s.f_i
        distances = [distance_point_line(point, ref) for ref in reference_points]
        i = np.argmin(distances)
        s.closest_point = reference_points[i]
        s.distance_point = distances[i]

def normalize(new_population):
    m = len(new_population[0].f_i)
    for s in new_population:
        s.original_f_i = np.copy(s.f_i)
    min_f = np.zeros((m,))
    max_f = np.zeros((m,))
    for i in range(m):
        min_f[i] = min([s.f_i[i] for s in new_population])
        tmp = sorted([s.f_i[i] for s in new_population])
        ids = [s.id for s in new_population]
        for s in new_population:
            s.f_i[i] -= min_f[i]
            if s.f_i[i] < 0:
                print (f'>>>>>>    {s.f_i} > {i} | min {min_f[i]} | original {len(tmp)} | new {len(new_population)}')
                print (f'get {sorted([n.f_i[i] for n in new_population])}')
                print (f'tmp: {tmp}')
                print (f'ids: {ids}')
                raise('minimum error')
        max_f[i] = max([s.f_i[i] for s in new_population])

    extreme_points_f = []
    for i in range(m):
        w = np.full((m,), 1e-6)
        w[i] = 1
        extreme_point_i = np.argmin([asf(w, s) for s in new_population])
        extreme_point = new_population[extreme_point_i]
        extreme_points_f.append(extreme_point)

    intercepts_f = np.zeros((m,))
    for i in range(m):
        intercepts_f[i] = max([ e.f_i[i] for e in extreme_points_f])
    for i in range(m):
        for j, s in enumerate(new_population):
            s.f_i[i] /= ((max_f[i] + min_f[i]) - min_f[i])
            #s.f_i[i] /= (intercepts_f[i] - min_f[i])
            if s.f_i[i] > 1 or s.f_i[i] < 0:
                print (f'>>>>>>>>>> error {i} {j}')
                print (f'extreme [{[s.f_i for s in extreme_points_f]}]')
                print (f'min {min_f}')
                print (f'max {max_f}')
                print (f'intercepts_f {intercepts_f}')
                print (f'normalized {s.f_i}')
                raise('normalized error')

def asf(w, individual):
    return max([individual.f_i[i] / w[i] for i in range(len(individual.f_i))])

def fast_non_dominating_sorting(population):
    fronts = []
    current_population = copy.deepcopy(population)
    while len(current_population) != 0:
        front = get_front(current_population)
        fronts.append(front)
        for f in front:
            current_population.remove(f)
    return fronts


def get_front(current_population):
    dominated = np.zeros((len(current_population),))
    for i,p in enumerate(current_population):
        for q in current_population:
            if p != q and q.dominates(p):
                dominated[i] += 1
    non_dominated = [j for j,v in enumerate(dominated) if v == 0]
    front = [current_population[i] for i in non_dominated]
    if front == []:
        print (f'error >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ')
        print (f'current {[c.id for c in current_population]}')
        print (f'dominated {dominated}')
        for c in current_population:
            print (c.f_i)
        dominated_1 = [j for j,v in enumerate(dominated) if v == 1]
        print (f'dominated 1 {[current_population[i].f_i for i in dominated_1]}')
        raise('empty front every solution is dominated')
    return front

def repeated(iter, population):
    for ind in iter:
        for pop in population:
            if ind.id != pop.id and (ind.f_i == pop.f_i).all():
                return True
    return False

def crossover_stage(parent_population, prob_cross=0.8):
    n = len(parent_population)
    child_population = []
    while len(child_population) < n:
        sample = random.sample(parent_population, 2)
        first = sample[0]
        second = sample[1]
        childs = first.crossover(second, prob_cross=prob_cross)
        if len(childs) == 0:
            continue
        if repeated(childs, parent_population + child_population):
            continue
        if len(child_population) + len(childs) > n:
            child_population.append(childs[0])
        else:
            child_population = child_population + childs
    return child_population

def mutation_stage(population, prob_mut=0.1):
    for individual in population:
        individual.mutation(population, prob_mut)
