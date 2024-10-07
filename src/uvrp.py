import copy
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from autorank import autorank, plot_stats
from matplotlib import pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import wilcoxon

from ant import Ant
from reader import read_dataset
import pickle
from solution import Solution
from utils import plot_archive_uncertainty, plot_worst_escenarios, \
    plot_lightly_robust_escenarios, get_medians_files, get_reference_point_file, indicator_s_energy, r2, plot_ranks
import lzma

def get_execution_dir_uvrp(dataset, algorithm, dir):
    if '.txt' in dataset:
        name_dataset = dataset.replace('.txt', '')
    else:
        name_dataset = dataset.replace('.vrp', '')
    new_path = 'uncertainty/' + name_dataset + '/' + algorithm + '/' + dir + '/'
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path)
    return new_path

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
        vehicle.return_depot(timetable, day)
        if len(vehicle.tour[timetable][day]) == 1:
            raise ('vehicle with only 1 costumer not allowed')
        if len(vehicle.tour[timetable][day]) != len(route):
            time = 0 if timetable == 0 else 500
            for i in range(len(vehicle.tour[timetable][day])):
                c_i = vehicle.tour[timetable][day][i]
                j = i+1
                time += c_i.service_times[day]
                if j < len(vehicle.tour[timetable][day]):
                    c_j = vehicle.tour[timetable][day][j]
                    time += c_i.distance_to(c_j)
            time += vehicle.tour[timetable][day][-1].distance_to(vehicle.tour[timetable][day][0])
            time = 0 if timetable == 0 else 500
            for i in range(len(route)):
                c_i = route[i]
                j = i+1
                time += c_i.service_times[day]
                if j < len(route):
                    c_j = route[j]
                    time += c_i.distance_to(c_j)
            time += route[-1].distance_to(route[0])
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
    if os.path.exists(file_name + '.xz'):
        file_name += '.xz'
        file = lzma.open(file_name, 'rb')
    else:
        raise(f'not found archive object for {dataset} - {dir} in {algorithm}')

    archive = pickle.load(file)

    scenarios = ['0.5', '0.7', '0.9']
    scenarios_archive = [archive]

    for scenario in scenarios[1:]:
        scenario_archive = eval_scenario_archive(dataset, archive, scenario)
        scenarios_archive.append(scenario_archive)

    execution_dir = get_execution_dir_uvrp(dataset, algorithm, dir + '-lre')

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

def get_reference_point_file_uncertainty(dataset):
    if dataset == 'Christofides_8_5_0.5.txt':
        return np.array([15000, 8, 900])
    elif dataset == 'Christofides_6_5_0.5.txt':
        return np.array([16000, 9, 900])
    elif dataset == 'Christofides_4_5_0.5.txt':
        return np.array([12000, 7, 900])
    elif dataset == 'Christofides_2_5_0.5.txt':
        return np.array([10000, 6, 600])
    elif dataset == 'Christofides_1_5_0.5.txt':
        return np.array([10000, 7, 700])
    elif dataset == 'Christofides_3_5_0.5.txt':
        return np.array([10000, 6, 600])
    elif dataset == 'Christofides_7_5_0.5.txt':
        return np.array([12000, 8, 900])
    elif dataset == 'Christofides_5_5_0.5.txt':
        return np.array([16000, 9, 900])
    elif dataset == 'Christofides_9_5_0.5.txt':
        return np.array([15000, 9, 1000])
    elif dataset == 'Christofides_10_5_0.5.txt':
        return np.array([17000, 9, 1300])
    elif dataset == 'Christofides_11_5_0.5.txt':
        return np.array([17000, 9, 1300])
    elif dataset == 'Christofides_12_5_0.5.txt':
        return np.array([17000, 9, 1300])
    elif dataset == 'Christofides_1_5_0.9.txt':
        return np.array([10000, 7, 700])
    elif dataset == 'convrp_10_test_0.vrp':
        return np.array([10000, 7, 700])
    else:
        raise('not reference point available for dataset')

# Eval lightly robust solutions of archive
def eval_robust_solutions(problem, algorithm, dir):
    file_name = 'results/' + problem[:-4] + '/' + algorithm + '/' + dir + '/archive-object'
    if os.path.exists(file_name + '.xz'):
        file_name += '.xz'
        file = lzma.open(file_name, 'rb')
    else:
        print(problem, algorithm, file_name + '.pkl')
        raise ('not found file object')

    scenarios = ['0.5', '0.7', '0.9']
    archive = pickle.load(file)
    scenarios_archive = [archive]

    for scenario in scenarios[1:]:
        scenario_archive = eval_scenario_archive(problem, archive, scenario)
        scenarios_archive.append(scenario_archive)

    scenario_0_5 = scenarios_archive[0]
    scenario_0_7 = scenarios_archive[1]
    scenario_0_9 = scenarios_archive[2]
    scenarios_archive = list(zip(scenario_0_5, scenario_0_7, scenario_0_9))

    worst_escenarios = get_worst_scenarios(scenarios_archive)
    lre_solutions = get_lightly_robust_efficient(worst_escenarios)
    lre_solutions = [t for tpl in lre_solutions for t in tpl if isinstance(t, Solution)]
    lre_solutions = [[t.f_1, t.f_2, t.f_3] for t in lre_solutions]
    lre_solutions = np.array(lre_solutions)

    ref_point_hv = get_reference_point_file(problem)
    ind = HV(ref_point=ref_point_hv)
    w_r2 = get_reference_directions("energy", 3, 30, seed=1)
    z_r2 = np.array([0, 0, 0])
    hyp_lre_solutions = ind(lre_solutions)
    r2_lre_solutions = r2(lre_solutions, w_r2, z_r2)
    rs_lre_solutions = indicator_s_energy(lre_solutions, s=2)

    return lre_solutions, hyp_lre_solutions, r2_lre_solutions, rs_lre_solutions

def plot_difference_diagram_indicator_uvrp(problems, algorithms, file, indicator, output_file, labels):
    algorithms_columns = {}
    for i, a in enumerate(algorithms):
        algorithms_columns[i] = labels[a]
    df = pd.read_csv(file)
    data = pd.DataFrame()
    for problem in problems:
        data_sub = pd.DataFrame()
        for algorithm in algorithms:
            populations = df.query(f"algorithm == '{algorithm}' and problem == '{problem}' and indicator == '{indicator}'")
            populations.reset_index(drop=True, inplace=True)
            data_sub = pd.concat([data_sub, populations['value']], axis=1, ignore_index=True)
        data = pd.concat([data, data_sub], ignore_index=True)
    data.rename(columns=algorithms_columns, inplace=True)
    data.to_csv(output_file.replace('.pdf', '.csv'), index=False)
    if indicator != 'hv':
        data *= -1
    result = autorank(data, alpha=0.05, verbose=False, force_mode='nonparametric')
    fig, ax = plt.subplots(figsize=(15,25))
    ax = plot_stats(result, allow_insignificant=True)
    #ax.set_title(dataset + ' ' + indicator)
    fig.axes.append(ax)
    #fig.suptitle(dataset + ' ' + indicator)
    plt.savefig(output_file)
    plt.close()

def plot_general_diagram_uvrp(algorithms, labels):
    algorithms_columns = {}
    title = 'Critic difference diagram '
    for algorithm in algorithms:
        algorithms_columns[algorithm] = labels[algorithm]
        #title += '- ' + labels[algorithm] + ' '
    file_es = 'lre-critic-diff-es.csv'
    file_hv = 'lre-critic-diff-hv.csv'
    file_r2 = 'lre-critic-diff-r2.csv'
    file_times = 'total-evaluations-time.csv'
    df_es = pd.read_csv(file_es)
    df_hv = pd.read_csv(file_hv)
    df_r2 = pd.read_csv(file_r2)
    df_times = pd.read_csv(file_times)
    df_r2 *= -1
    df_es *= -1
    df_times *= -1
    total_df = pd.concat([df_es, df_hv, df_r2], axis=0)
    total_df.rename(columns=algorithms_columns, inplace=True)
    result = autorank(total_df, alpha=0.05, verbose=False, force_mode='nonparametric')
    fig, ax = plt.subplots(figsize=(15, 25))
    ax = plot_stats(result, allow_insignificant=True)
    fig.axes.append(ax)
    plt.title(title)
    output_file = 'lre-ranking-total.pdf'
    plt.savefig(output_file,  bbox_inches="tight", pad_inches=0.15)
    plt.close()


def get_scenarios_comparation(problems):
    table_scenarios_hv = {'Problema': problems, 'Indicador': ['HV']*len(problems), '0.5': [], '0.7': [], '0.9': []}
    table_scenarios_r2 = {'Problema': problems, 'Indicador': ['R2']*len(problems), '0.5': [], '0.7': [], '0.9': []}
    table_scenarios_es = {'Problema': problems, 'Indicador': ['$E_s$']*len(problems), '0.5': [], '0.7': [], '0.9': []}
    scenarios = ['0.5', '0.7', '0.9']
    algorithm = 'cmibaco-lns'
    for problem in problems:
        dir = 'results/' + problem.replace('.txt', '') + '/' + algorithm + '/'
        dirs = os.listdir(dir)
        dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
        dirs = [(d, datetime.strptime(d, '%Y-%m-%d-%H-%M-%S')) for d in dirs]
        dirs.sort(key=lambda x: x[1])
        dirs = [d[0] for d in dirs]
        if len(dirs) != 20:
            raise ('no enought dirs')
        indicators_scenarios = {}
        for scenario in scenarios:
            indicators_scenarios[scenario] = {'hv': [], 'r2': [], 'es': []}
        ref_point_hv = get_reference_point_file(problem)
        ind = HV(ref_point=ref_point_hv)
        w_r2 = get_reference_directions("energy", 3, 30, seed=1)
        z_r2 = np.array([0, 0, 0])
        for d in dirs:
            print(f'>>>>>> evaluating scenarios {problem}/{d}')
            file_name = 'results/' + problem[:-4] + '/' + algorithm + '/' + d + '/archive-object'
            if os.path.exists(file_name + '.xz'):
                file_name += '.xz'
                file = lzma.open(file_name, 'rb')
            else:
                print(problem, d, file_name + '.pkl')
                raise ('not found file object')

            archive = pickle.load(file)

            for scenario in scenarios[1:]:
                scenario_archive = eval_scenario_archive(problem, archive, scenario)
                scenario_archive = [[a.f_1, a.f_2, a.f_3] for a in scenario_archive]
                scenario_archive = np.array(scenario_archive)
                hyp_scenario = ind(scenario_archive)
                r2_scenario = r2(scenario_archive, w_r2, z_r2)
                es_scenario = indicator_s_energy(scenario_archive, s=2)
                indicators_scenarios[scenario]['hv'].append(hyp_scenario)
                indicators_scenarios[scenario]['r2'].append(r2_scenario)
                indicators_scenarios[scenario]['es'].append(es_scenario)

            archive = [[a.f_1, a.f_2, a.f_3] for a in archive]
            archive = np.array(archive)
            hyp_archive = ind(archive)
            r2_archive = r2(archive, w_r2, z_r2)
            es_archive = indicator_s_energy(archive, s=2)
            indicators_scenarios['0.5']['hv'].append(hyp_archive)
            indicators_scenarios['0.5']['r2'].append(r2_archive)
            indicators_scenarios['0.5']['es'].append(es_archive)


        for scenario in indicators_scenarios.keys():
            hipervolume_scenario = np.array(indicators_scenarios[scenario]['hv'])
            r2_scenario = np.array(indicators_scenarios[scenario]['r2'])
            es_scenario = np.array(indicators_scenarios[scenario]['es'])
            mean_hyp = np.mean(hipervolume_scenario)
            desv_hyp = np.std(hipervolume_scenario)
            mean_r2 = np.mean(r2_scenario)
            desv_r2 = np.std(r2_scenario)
            mean_es = np.mean(es_scenario)
            desv_es = np.std(es_scenario)
            table_scenarios_hv[scenario].append(f'{mean_hyp:.3e}' + ' (' + f'{desv_hyp:.3e}' + ')')
            table_scenarios_r2[scenario].append(f'{mean_r2:.3e}' + ' (' + f'{desv_r2:.3e}' + ')')
            table_scenarios_es[scenario].append(f'{mean_es:.3e}' + ' (' + f'{desv_es:.3e}' + ')')

    df_hv = pd.DataFrame.from_dict(table_scenarios_hv)
    df_hv.set_index('Problema')
    df_hv.to_latex('scenarios-comparation-hv-56.tex', column_format='cc' + len(scenarios) * 'r', index=False)

    df_r2 = pd.DataFrame.from_dict(table_scenarios_r2)
    df_r2.set_index('Problema')
    df_r2.to_latex('scenarios-comparation-r2-56.tex', column_format='cc' + len(scenarios) * 'r', index=False)

    df_es = pd.DataFrame.from_dict(table_scenarios_es)
    df_es.set_index('Problema')
    df_es.to_latex('scenarios-comparation-es-56.tex', column_format='cc' + len(scenarios) * 'r', index=False)

def get_medians_files_lre(problems, algorithms):
    medians = {}
    df = pd.read_csv('lre-executions.csv')
    for problem in problems:
        medians[problem] = {'hv': {}}
        for algorithm in algorithms:
            data = df.query(f"problem == '{problem}' and algorithm == '{algorithm}' and indicator == 'hv' ")
            data = data.sort_values(by=['value'])
            median_dir = data.iloc[data.shape[0] // 2]['dir']
            medians[problem]['hv'][algorithm] = median_dir
    return medians

def plot_medians_lre(problems):
    indicators = ['hv']
    algorithm = 'cmibaco-lns'
    scenarios = ['0.5', '0.7', '0.9']
    for indicator in indicators:
        problems_medians = get_medians_files_lre(problems, [algorithm])
        for p in problems_medians.keys():
            dir_median = problems_medians[p][indicator][algorithm]
            file_name = 'results/' + p[:-4] + '/' + algorithm + '/' + dir_median + '/archive-object'
            if os.path.exists(file_name + '.xz'):
                file_name += '.xz'
                file = lzma.open(file_name, 'rb')
            else:
                print(p, dir_median, file_name + '.pkl')
                raise('not found file object')

            archive = pickle.load(file)
            scenarios_archive = [archive]

            for scenario in scenarios[1:]:
                scenario_archive = eval_scenario_archive(p, archive, scenario)
                scenarios_archive.append(scenario_archive)

            execution_dir = get_execution_dir_uvrp(p, algorithm, dir_median + '-median')

            id_solutions = {}
            for archive in scenarios_archive:
                for i, s in enumerate(archive):
                    id_solutions[s.id] = i

            scenario_0_5 = scenarios_archive[0]
            scenario_0_7 = scenarios_archive[1]
            scenario_0_9 = scenarios_archive[2]
            scenarios_archive = list(zip(scenario_0_5, scenario_0_7, scenario_0_9))

            worst_escenarios = get_worst_scenarios(scenarios_archive)
            lre_solutions = get_lightly_robust_efficient(worst_escenarios)

            plot_lightly_robust_escenarios(lre_solutions, [], scenarios, p, execution_dir + 'median-' + indicator + '-',
                                           id_solutions)
def count_ranks(data_mean, ranks):
    data_mean_sorted = data_mean.sort_values()
    pos = 0
    for algorithm, _ in data_mean_sorted.items():
        ranks[algorithm][pos] += 1
        pos += 1


def get_table_mean_uvrp(problems, file, output_file, indicator, main_algorithm, other_algorithms):
    df = pd.read_csv(file)
    algorithms = [main_algorithm] + other_algorithms
    data_total = pd.DataFrame(columns=algorithms)
    ranks = {}
    for algorithm in algorithms:
        ranks[algorithm] = [0] * len(algorithms)
    algorithms_columns = {0: main_algorithm}
    for i, a in enumerate(other_algorithms):
        algorithms_columns[i+1] = a
    for problem in problems:
        data = pd.DataFrame()
        for algorithm in algorithms:
            algorithm_population = df.query(f"algorithm == '{algorithm}' and problem == '{problem}' and indicator == '{indicator}'")
            algorithm_population.reset_index(drop=True, inplace=True)
            algorithm_population = algorithm_population.sort_values(by=['execution'])
            data = pd.concat([data, algorithm_population['value']], axis=1, ignore_index=True)
        data.rename(columns=algorithms_columns, inplace=True)
        data_mean = data[algorithms].mean(numeric_only=True)
        count_ranks(data_mean, ranks)
        data_std = data[algorithms].std(numeric_only=True)
        rows_stats = {}
        for algorithm in algorithms:
            mean_alg = data_mean.loc[algorithm]
            std_alg = data_std.loc[algorithm]
            arrow = ''
            if algorithm != main_algorithm:
                x = data[main_algorithm]
                y = data[algorithm]
                rank = wilcoxon(x, y)
                p_value = rank.pvalue
                if p_value >= 0.05:
                    # no significant difference
                    arrow = ' $\\leftrightarrow$'
                else:
                    if indicator == 'hv':
                        if mean_alg > data_mean.loc[main_algorithm]:
                            arrow = " $\\uparrow$"
                        else:
                            arrow = " $\\downarrow$"
                    else:
                        if mean_alg < data_mean.loc[main_algorithm]:
                            arrow = " $\\uparrow$"
                        else:
                            arrow = " $\\downarrow$"
            stats = str(f'{mean_alg:.3e}') + ' (' + str(f'{std_alg:.3e}') + ')' + arrow
            rows_stats[algorithm] = stats
        data_total = data_total._append(rows_stats, ignore_index=True)
    problems_column = [[p.replace('_', '$\_$').replace('.txt', ''), indicator.upper()] for p in problems]
    problems_column = pd.DataFrame(problems_column, columns=['Problem', 'Indicator'])
    data_total = pd.concat([problems_column, data_total], axis=1)
    data_total.to_latex(output_file, column_format='cc' + len(algorithms) * 'r', index=False)
    return ranks

def get_comparation_lre_algorithms(problems, algorithms):
    columns_csv = ['problem','indicator','value','algorithm','execution','dir']
    for j, problem in enumerate(problems):
        for algorithm in algorithms:
            dir = 'results/' + problem.replace('.txt', '').replace('.vrp', '') + '/' + algorithm + '/'
            dirs = os.listdir(dir)
            dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
            dirs = [(d, datetime.strptime(d, '%Y-%m-%d-%H-%M-%S')) for d in dirs]
            dirs.sort(key=lambda x: x[1])
            dirs = [d[0] for d in dirs]
            if len(dirs) != 20:
                raise('no enought dirs')
            hypervolume_algorithm = []
            r2_algorithm = []
            es_algorithm = []
            for j, d in enumerate(dirs):
                print(f'>>>>>> evaluating lre solutions {problem}/{algorithm}/{d}')
                _, hyp_lre_solutions, r2_lre_solutions, rs_lre_solutions = eval_robust_solutions(problem, algorithm, d)
                hypervolume_algorithm.append(hyp_lre_solutions)
                r2_algorithm.append(r2_lre_solutions)
                es_algorithm.append(rs_lre_solutions)
                row_hv = [problem, 'hv', hyp_lre_solutions, algorithm, j, d]
                row_r2 = [problem, 'r2', r2_lre_solutions, algorithm, j, d]
                row_es = [problem, 'es', rs_lre_solutions, algorithm, j, d]
                df = pd.DataFrame(np.array([row_hv, row_r2, row_es]), columns=columns_csv)
                df.to_csv('lre-executions.csv', mode='a', index=False, header=False)

def compress_files():
    problems = ['Christofides_1_5_0.5.txt',
                'Christofides_2_5_0.5.txt',
                'Christofides_3_5_0.5.txt',
                'Christofides_4_5_0.5.txt',
                'Christofides_5_5_0.5.txt',
                'Christofides_6_5_0.5.txt',
                'Christofides_7_5_0.5.txt',
                'Christofides_8_5_0.5.txt',
                'Christofides_9_5_0.5.txt',
                'Christofides_10_5_0.5.txt',
                'Christofides_11_5_0.5.txt',
                'Christofides_12_5_0.5.txt']
    algorithms = ['cmibaco-lns', 'ibaco-eps-lns', 'ibaco-hv-lns', 'ibaco-r2-lns', 'ibaco-ws-lns']
    for problem in problems:
        for algorithm in algorithms:
            print(f'Compresing {problem}/{algorithm}')
            dir = 'results/' + problem.replace('.txt', '') + '/' + algorithm + '/'
            dirs = os.listdir(dir)
            dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
            dirs = [(d, datetime.strptime(d, '%Y-%m-%d-%H-%M-%S')) for d in dirs]
            dirs.sort(key=lambda x: x[1])
            dirs = [d[0] for d in dirs]
            if len(dirs) != 20:
                print(problem, algorithm, len(dirs))
                raise ('no enought dirs')

            for d in dirs:
                file_name = 'results/' + problem[:-4] + '/' + algorithm + '/' + d + '/archive-object'
                if os.path.exists(file_name + '.pkl') and not os.path.exists(file_name + '.xz'):
                    file = open(file_name + '.pkl', 'rb')
                elif os.path.exists(file_name + '.xz'):
                    print(f"alredy exist file .xz for {problem}/{algorithm} in {d}")
                    continue
                else:
                    print(f'ERROR: NOT EXIST FILE pkl FOR {problem}/{algorithm} in {d}')
                    continue
                archive = pickle.load(file)
                with lzma.open(file_name + '.xz', 'wb') as file:
                    pickle.dump(archive, file)
def get_boxplots(problems, file, indicator, algorithms, labels):
    algorithms_columns = {}
    for i, a in enumerate(algorithms):
        algorithms_columns[i] = labels[a]
    df = pd.read_csv(file)
    for problem in problems:
        data = pd.DataFrame()
        for algorithm in algorithms:
            populations = df.query(f"algorithm == '{algorithm}' and problem == '{problem}' and indicator == '{indicator}'")
            populations.reset_index(drop=True, inplace=True)
            data = pd.concat([data, populations['value']], axis=1, ignore_index=True)
        data.rename(columns=algorithms_columns, inplace=True)
        fig = plt.figure()
        plt.title(f"-Light robustness - {problem.replace('.txt', '')} - {indicator}")
        bplot = data.boxplot(column=list(data.columns))
        fig.axes.append(bplot)
        output_file = 'boxplot-uvrp/' + problem[13:].replace('0.5.txt', '0_5-boxplot.pdf').replace('0.9.txt',
                                                                                                   '0_9-boxplot.pdf').replace('0.7.vrp', '0_7-boxplot.pdf')
        plt.savefig(output_file)
        plt.close()
