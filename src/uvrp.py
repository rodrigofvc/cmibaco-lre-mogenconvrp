import copy
import os

import numpy as np

from ant import Ant
from reader import read_dataset
import pickle
from solution import Solution
from utils import plot_archive_uncertainty, get_execution_dir, plot_worst_escenarios, \
    plot_lightly_robust_escenarios
import lzma

def get_routes_uncertain(day, k_centers, costumers_scenario, max_capacity):
    remove_zeros = []
    for i in range(0, len(k_centers)):
        j = i + 1
        if j != len(k_centers):
            c_i = k_centers[i]
            c_j = k_centers[j]
            if c_i.id == 0 and c_j.id == 0:
                remove_zeros.append(j)
    k_centers = [k for i, k in enumerate(k_centers) if i not in remove_zeros]
    if k_centers[-1].id == 0:
        k_centers.pop()
    tours = []
    tour = []
    current_capacity = 0
    for i in range(0, len(k_centers)):
        k = k_centers[i]
        c = [n for n in costumers_scenario if n.id == k.id][0]
        if c.id == 0 and i > 0:
            tours.append(tour)
            current_capacity = 0
            tour = [c]
        elif current_capacity + c.demands[day] <= max_capacity:
            current_capacity += c.demands[day]
            tour.append(c)
        else:
            tours.append(tour)
            depot = [n for n in costumers_scenario if n.id == 0][0]
            tour = [depot, c]
            current_capacity = 0
    if len(tour) > 1:
        tours.append(tour)
    return tours

def get_available_tours_capacity(routes_uncertain, capacity, r, day):
    available_tours = []
    for tour in routes_uncertain:
        if len(tour) > 1:
            demand = sum([c.demands[day] for c in tour])
            if demand + r.demands[day] <= capacity:
                available_tours += tour
    return available_tours

def get_available_tours(routes_uncertain, capacity, r, day, limit_time, pos=-1):
    available_tours = []
    for tour in routes_uncertain:
        if len(tour) > 1:
            demand = sum([c.demands[day] for c in tour])
            time_tour = 0
            for i in range(len(tour)):
                j = i + 1
                c_i = tour[i]
                time_tour += c_i.service_times[day]
                if i == pos:
                    c_k = tour[i-1]
                    time_tour += c_k.distance_to(r)
                    time_tour += r.service_times[day]
                    time_tour += r.distance_to(c_i)
                elif j < len(tour):
                    c_j = tour[j]
                    time_tour += c_i.distance_to(c_j)
            if pos == -1:
                time_tour += tour[-1].distance_to(r) + r.service_times[day] + r.distance_to(tour[0])
            else:
                time_tour += tour[-1].distance_to(tour[0])
            if time_tour < limit_time and (demand + r.demands[day] <= capacity):
                available_tours += tour
    return available_tours


def get_distances_centers(depot, remaining, routes_uncertain, capacity, day):
    distances_cts = []
    for r in remaining:
        available_tours = get_available_tours_capacity(routes_uncertain, capacity, r, day)
        if len(available_tours) == 0:
            # no tour with available capacity
            distances_cts.append((r.distance_to(depot), depot))
            continue
        available_tours = [c for c in available_tours if c.id != 0]
        distances_r_k = [(r.distance_to(k), k) for k in available_tours]
        distances_r_k.sort(key=lambda x: x[0])
        found = False
        for (dist, k_center) in distances_r_k:
            route_center = [route for route in routes_uncertain if k_center in route][0]
            tmp_route = copy.deepcopy(route_center)
            index_k_center = tmp_route.index(k_center)
            tmp_route.insert(index_k_center, r)
            if valid_tour(tmp_route, r, day):
                distances_cts.append((dist, k_center))
                found = True
                break
        if not found:
            distances_cts.append((r.distance_to(depot), depot))
    if len(distances_cts) != len(remaining):
        raise
    return distances_cts

def insert_route(depot, node_insert, pair, routes_uncertain, day):
    (dist, center) = pair
    if center.id == 0:
        routes_uncertain.append([depot, node_insert])
        return
    for i, route in enumerate(routes_uncertain):
        if center in route:
            idx = route.index(center)
            route.insert(idx, node_insert)
            if not valid_tour(route, node_insert, day):
                raise('invalid tour')
            return
    raise('couldnt insert the node')

def valid_tour(route, node_insert, day):
    limit_time = 500 if node_insert.timetable == 0 else 1000
    if route[0].id != 0:
        raise()
    time = 0 if node_insert.timetable == 0 else 500
    for i in range(len(route)):
        j = i + 1
        c_i = route[i]
        time += c_i.service_times[day]
        if j < len(route):
            c_j = route[j]
            time += c_i.distance_to(c_j)
    time += route[-1].distance_to(route[0])
    if time > limit_time:
        return False
    return True

def assign_routes_vehicles(depot, timetable, day, routes_uncertain, vehicles_scenario, costumers_scenario):
    for i_vehicle, vehicle in enumerate(vehicles_scenario[:len(routes_uncertain)]):
        route = routes_uncertain[i_vehicle]
        vector_rep = [c for c in route if c.id != 0]
        if vector_rep[0].timetable == 0:
            limit_time = vehicles_scenario[0].limit_time / 2
        else:
            limit_time = vehicles_scenario[0].limit_time
        i = 0
        vehicle.times_tour[timetable][day] = 0
        vehicle.loads[timetable][day] = 0
        tour = [depot]
        vehicle.set_tour_day(timetable, day, tour)
        current_cos = vector_rep[i]
        current_time = vehicle.add_costumer_tour_day(timetable, day, current_cos)
        i += 1
        if i == len(vector_rep):
            vehicle.return_depot(timetable, day)
            continue
        current_cos = vector_rep[i]
        while current_time + vehicle.tour[timetable][day][-1].distance_to(current_cos) + current_cos.service_times[
            day] + current_cos.distance_to(
            vehicle.tour[timetable][day][0]) <= limit_time and vehicle.get_load(timetable, day) + current_cos.demands[day] <= vehicle.capacity:
            current_time = vehicle.add_costumer_tour_day(timetable, day, current_cos)
            i += 1
            if i == len(vector_rep):
                break
            current_cos = vector_rep[i]
        #else:
            #print (current_time + vehicle.tour[timetable][day][-1].distance_to(current_cos) + current_cos.service_times[
            #    day] + current_cos.distance_to(vehicle.tour[timetable][day][0]))
            #print (current_cos.distance_to(vehicle.tour[timetable][day][0]))
            #print(f'limit_time {limit_time}')
            #print(f'failed to add {current_cos.id} / {vehicle.get_load(timetable, day)} {vehicle.capacity} - {current_cos.demands[day]}')
        vehicle.return_depot(timetable, day)
        if len(vehicle.tour[timetable][day]) == 1:
            raise ('vehicle with only 1 costumer not allowed')
        if len(vehicle.tour[timetable][day]) != len(route):
            print ([c.id for c in vehicle.tour[timetable][day]])
            print ([c.id for c in route])
            print(current_time, vehicle.capacity)
            print(valid_tour(route, route[3], day))
            print(f'demands {[c.demands[day] for c in vehicle.tour[timetable][day]]} => {sum([c.demands[day] for c in vehicle.tour[timetable][day]])}')
            print(f'demands {[c.demands[day] for c in route]} => {sum([c.demands[day] for c in route])}')
            time = 0 if timetable == 0 else 500
            for i in range(len(vehicle.tour[timetable][day])):
                c_i = vehicle.tour[timetable][day][i]
                j = i+1
                time += c_i.service_times[day]
                if j < len(vehicle.tour[timetable][day]):
                    c_j = vehicle.tour[timetable][day][j]
                    time += c_i.distance_to(c_j)
            time += vehicle.tour[timetable][day][-1].distance_to(vehicle.tour[timetable][day][0])
            print (f'CALCULATED vehicle {time}')
            time = 0 if timetable == 0 else 500
            for i in range(len(route)):
                c_i = route[i]
                j = i+1
                time += c_i.service_times[day]
                if j < len(route):
                    c_j = route[j]
                    time += c_i.distance_to(c_j)
            time += route[-1].distance_to(route[0])
            print(f'CALCULATED tour {time}')
            raise('not planed tour in vehicle')
    for c in costumers_scenario:
        if c.vehicles_visit[day] == -1 and c.demands[day] > 0 and timetable == 'AM' and c.timetable == 0:
            print(f'UNVISITED')
            print(c)
            print(f'visited {len([c for c in costumers_scenario if c.vehicles_visit[day] != -1])}')
            raise ()
        if c.vehicles_visit[day] == -1 and c.demands[day] > 0 and timetable == 'PM' and c.timetable == 1:
            print(f'UNVISITED')
            print(c)
            raise ()

def eval_scenario_archive(dataset, archive, scenario='0.5'):
    dir = 'dataset/'
    ext = '.txt'
    if scenario == '0.7':
        ext = '.vrp'
    dataset_scenario = dataset[0:-7] + scenario + ext
    costumers_scenario, vehicles_scenario, capacity_scenario, days_scenario, _ = read_dataset(dir + dataset_scenario)
    population_scenario = []
    for solution in archive:
        uncertainty_solution = eval_scenario(solution, copy.deepcopy(costumers_scenario), copy.deepcopy(vehicles_scenario), capacity_scenario, days_scenario)
        population_scenario.append(uncertainty_solution)
    return population_scenario

def eval_scenario(solution, costumers_scenario, vehicles_scenario, capacity_scenario, days_scenario):
    n_costumers = len(costumers_scenario)
    depot = costumers_scenario[0]
    capacity_vehicle = vehicles_scenario[0].capacity
    timetables = ['AM', 'PM']
    solution_uncertainty = Solution(timetables, days_scenario)
    for h in timetables:
        for d in range(days_scenario):
            routes_h_d = solution.get_vector_representation_dt(h, d)
            t = 0 if h == 'AM' else 1
            costumers_scenario_h_d = [c for c in costumers_scenario if c.timetable == t and c.demands[d] > -1]
            k_centers = [c for c in routes_h_d if c in costumers_scenario_h_d + [depot]]
            routes_uncertain = get_routes_uncertain(d, k_centers, costumers_scenario, capacity_vehicle)
            remaining = [c for c in costumers_scenario_h_d if c not in routes_h_d]
            while len(remaining) > 0:
                # [(dist, node)]
                distances_centers = get_distances_centers(depot, remaining, routes_uncertain, capacity_vehicle, d)
                min_dist = min([d[0] for d in distances_centers])
                argmin = [d[0] for d in distances_centers].index(min_dist)
                insert_route(depot, remaining[argmin], distances_centers[argmin], routes_uncertain, d)
                remaining.pop(argmin)
            ant = Ant(depot, n_costumers, min_pheromone=10e-4, max_pheromone=10e3)
            solution_uncertainty.add_ant_timetable_day(ant, h)
            assign_routes_vehicles(depot, h, d, routes_uncertain, vehicles_scenario, costumers_scenario)
    solution_uncertainty.add_assigment_vehicles(vehicles_scenario, costumers_scenario[1:])
    solution_uncertainty.build_paths_ants()
    solution_uncertainty.depot = depot
    solution_uncertainty.get_fitness()
    solution_uncertainty.is_feasible()
    return solution_uncertainty

def is_contained_scenarios(x, y):
    if x == y:
        return False
    n = len(x)//2
    scenarios_x = np.array([[x[i * 2].f_1, x[i * 2].f_2, x[i * 2].f_3] for i in range(n)])
    m = len(y)//2
    scenarios_y = np.array([[y[i * 2].f_1, y[i * 2].f_2, y[i * 2].f_3] for i in range(m)])
    if np.array_equal(scenarios_x, scenarios_y):
        return False
    for x in scenarios_x:
        x_dominates_y = [y for y in scenarios_y if (x <= y).all()]
        if len(x_dominates_y) == 0:
            return False
    return True

def archive_update_re(P, A):
    for p in P:
        a_dominates_p = [a for a in A if is_contained_scenarios(a, p)]
        if len(a_dominates_p) == 0:
            A.append(p)
            to_remove = []
            for i, a in enumerate(A):
                if is_contained_scenarios(p, a):
                    to_remove.append(i)
            A = [a for j, a in enumerate(A) if j not in to_remove]
    return A

def get_lightly_robust_efficient(worst_escenarios):
    A = archive_update_re(worst_escenarios, [])
    return A


def get_worst_scenarios(scenarios_archive):
    worst_scenarios = []
    for scenario_solution in scenarios_archive:
        n = len(scenario_solution)
        points = []
        for i in range(n):
            solution_sn = scenario_solution[i]
            points.append([solution_sn.f_1, solution_sn.f_2, solution_sn.f_3])
        points = np.array(points)
        rank = np.zeros((n,))
        for i, p in enumerate(points):
            dominated = [q for q in points if (q != p).any() and (p <= q).all()]
            rank[i] = len(dominated)
        index_max = np.argwhere(rank == min(rank))
        worst_scenario = []
        for index in index_max:
            worst_scenario.append(scenario_solution[index[0]])
            worst_scenario.append(index[0])
        worst_scenario = tuple(worst_scenario)
        worst_scenarios.append(worst_scenario)
    return worst_scenarios


def lightly_robust_solutions(dataset, algorithm, dir):
    file_name = 'results/' + dataset[:-4] + '/' + algorithm + '/' + dir + '/archive-object'
    if os.path.exists(file_name + '.pkl'):
        file_name += '.pkl'
        file = open(file_name, 'rb')
    elif os.path.exists(file_name + '.xz'):
        file_name += '.xz'
        file = lzma.open(file_name, 'rb')
    else:
        raise('not found')

    archive = pickle.load(file)

    scenarios = ['0.5', '0.7', '0.9']
    scenarios_archive = [archive]

    for scenario in scenarios[1:]:
        scenario_archive = eval_scenario_archive(dataset, archive, scenario)
        scenarios_archive.append(scenario_archive)

    """
    for i, a in enumerate(archive):
        print (f'scenario (0.5) [{a.f_1, a.f_2, a.f_3}]')
        for j, scenario in enumerate(scenarios[1:]):
            s = scenarios_archive[j+1][i]
            print (f'scenario ({scenario}) [{s.f_1, s.f_2, s.f_3}]')
        print ('-------------------')
    """
    execution_dir = get_execution_dir(dataset, algorithm, run=True, uvrp=True)

    id_solutions = {}
    for archive in scenarios_archive:
        for i, s in enumerate(archive):
            id_solutions[s.id] = i

    plot_archive_uncertainty(scenarios_archive, scenarios, dataset, execution_dir, id_solutions=id_solutions)

    scenario_0_5 = scenarios_archive[0]
    scenario_0_7 = scenarios_archive[1]
    scenario_0_9 = scenarios_archive[2]
    scenarios_archive = list(zip(scenario_0_5, scenario_0_7, scenario_0_9))

    worst_escenarios = get_worst_scenarios(scenarios_archive)

    plot_worst_escenarios(worst_escenarios, scenarios, dataset, execution_dir + 'worst-', id_solutions=id_solutions)

    lre_solutions = get_lightly_robust_efficient(worst_escenarios)

    plot_lightly_robust_escenarios(lre_solutions, [], scenarios, dataset, execution_dir + 'lre-all-', id_solutions)

    worst_scenarios_no_lre = [t for t in worst_escenarios if t not in lre_solutions]

    if len(lre_solutions) + len(worst_scenarios_no_lre) != len(worst_escenarios):
        raise ('no filtering worst escenarios')

    size_group = 5
    n_groups = len(worst_scenarios_no_lre) // size_group
    for i in range(n_groups):
        #print(f'group {i}/{n_groups} - [{i*size_group},{(i+1)*size_group}] - n={len(worst_scenarios_no_lre)}')
        worst_scenarios_no_lre_sub = worst_scenarios_no_lre[i*size_group:(i+1)*size_group]
        plot_lightly_robust_escenarios(lre_solutions, worst_scenarios_no_lre_sub, scenarios, dataset, execution_dir + 'lre-' + str(i) + '-', id_solutions)
        if i == n_groups - 1 and len(worst_scenarios_no_lre) % size_group != 0:
            #print(f'group {i+1}/{n_groups} - [{(i + 1) * size_group}:] - n={len(worst_scenarios_no_lre)}')
            worst_scenarios_no_lre_sub = worst_scenarios_no_lre[(i+1)*size_group:]
            plot_lightly_robust_escenarios(lre_solutions, worst_scenarios_no_lre_sub, scenarios, dataset,
                                           execution_dir + 'lre-' + str(i+1) + '-', id_solutions)


if __name__ == '__main__':
    dataset = 'Christofides_2_5_0.5.txt'
    algorithm = 'cmibaco-lns'
    # lightly_robust_solutions(dataset, algorithm, '2024-03-24-19-10-46')
    dir = 'results/' + dataset.replace('.txt','') + '/' + algorithm + '/'
    dirs = os.listdir(dir)
    dirs = [(d, os.path.getmtime(os.path.join(dir, d))) for d in dirs]
    dirs.sort(key=lambda x: x[1])
    dirs = [d[0] for d in dirs if d[0].startswith('2023') or d[0].startswith('2024')]
    list_dir = os.listdir('results/' + dataset.replace('.txt','') + '/' + algorithm + '/')
    for dir in dirs:
        print (f'lightly robust solutions for {algorithm} - {dir}')
        lightly_robust_solutions(dataset, algorithm, dir)
