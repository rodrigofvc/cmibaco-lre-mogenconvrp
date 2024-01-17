import os
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from datetime import datetime
import matplotlib.patches as mpatches
import json
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table

def save_result(dataset, fig_name, new_path):
    if '.txt' in dataset:
        name_dataset = dataset.replace('.txt', '')
    else:
        name_dataset = dataset.replace('.vrp', '')
    plt.savefig(new_path + name_dataset + '-' + fig_name)

def save_params(params, execution_dir):
    params_to_save = {}
    params_to_save['file'] = params['file']
    params_to_save['n_ants'] = params['n_ants']
    params_to_save['rho'] = params['rho']
    params_to_save['alpha'] = params['alpha']
    params_to_save['beta'] = params['beta']
    params_to_save['gamma'] = params['gamma']
    params_to_save['delta'] = params['delta']
    params_to_save['Q'] = params['Q']
    params_to_save['q0'] = params['q0']
    params_to_save['max_iterations'] = params['max_iterations']
    params_to_save['p_mut'] = params['p_mut']
    params_to_save['seed'] = params['seed']
    params_to_save['min_pheromone'] = params['min_pheromone']
    params_to_save['max_pheromone'] = params['max_pheromone']
    params_to_save['epsilon'] = params['epsilon']
    params_to_save['dy'] = params['dy']
    if 'cibaco' in params.keys():
        params_to_save['cibaco'] = params['cibaco']
    params_to_save['ibaco-eps'] = params['ibaco-eps']
    params_to_save['ibaco-hv'] = params['ibaco-hv']
    params_to_save['ibaco-r2'] = params['ibaco-r2']
    params_to_save['lns'] = params['lns']
    params = json.dumps(params_to_save)
    file = execution_dir + 'params.json'
    f = open(file, "w")
    f.write(params)
    f.close()

def save_archive(A, execution_dir):
    arr = [[a.f_1, a.f_2, a.f_3] for a in A]
    arr = np.array(arr)
    file_name = 'archive'
    np.save(execution_dir + file_name, arr)
    file_name += '-object.pkl'
    with open(execution_dir + file_name, 'wb') as file:
        pickle.dump(A, file)


def plot_log_hypervolume(log, dataset, execution_dir):
    plt.title('Hypervolume archive per iteration ' + dataset)
    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    plt.plot(log)
    fig_name = 'log-hyper.png'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_log_solutions_added(log, dataset, execution_dir):
    plt.title('Archive size per iteration ' + dataset)
    plt.xlabel('Iteration')
    plt.ylabel('# Archive size ')
    plt.plot(log)
    fig_name = 'log-solutions.png'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_best_objective(A, dataset, objective, execution_dir):
    if objective == 0:
        best = [a.f_1 for a in A]
        ibest = np.argmin(best)
        best = A[ibest]
        title = 'Best total tours time '
        fig_name = 'best-solution-time-tour.png'
    elif objective == 1:
        best = min([a.f_2 for a in A])
        best = min([a.f_1 for a in A if a.f_2 == best])
        best = [a for a in A if a.f_1 == best][0]
        title = 'Best consistency driver '
        fig_name = 'best-solution-driver-diff.png'
    elif objective == 2:
        best = [a.f_3 for a in A]
        ibest = np.argmin(best)
        best = A[ibest]
        title = 'Best maximum arrival time difference '
        fig_name = 'best-solution-arrival-diff.png'
    title += '[' + str(best.f_1) + ', ' + str(best.f_2) + ', ' + str(best.f_3) + ']'
    fig, axs = plt.subplots(len(best.timetables), best.days)
    fig.set_figheight(18)
    fig.set_figwidth(20)
    vehicles_used = []
    np.random.seed(5)
    vehicle_colors = [(v.id, tuple(np.random.rand(3,))) for v in best.assigments_vehicles]
    for i, t in enumerate(best.timetables):
        for d in range(best.days):
            vehicles_used += plot_sub_vrp(best, t, d, axs[i, d], vehicle_colors)
    vehicles_used = list(set(vehicles_used))
    colors_patch = [mpatches.Patch(color=v[1], label='Vehicle ' + str(v[0])) for v in [vehicle_colors[i] for i in vehicles_used]]
    fig.legend(handles=colors_patch)
    fig.suptitle(title)
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_sub_vrp(solution, timetable, day, axs, vehicle_colors):
    tours = solution.get_vector_representation_dt(timetable, day)
    tour = []
    subtours = []
    for c in tours[1:]:
        if c.id == 0:
            subtours.append(tour)
            tour = []
        else:
            tour.append(c)
    subtours.append(tour)
    vehicles_used = []
    for t in subtours:
        vehicle_id = t[0].vehicles_visit[day]
        for c in t:
            if c.vehicles_visit[day] != vehicle_id:
                raise()
        vehicles_used.append(vehicle_id)
        x_tour = [0] + [c.x for c in t] + [0]
        y_tour = [0] + [c.y for c in t] + [0]
        x_tour = np.array(x_tour)
        y_tour = np.array(y_tour)
        vehicle_color = [v for v in vehicle_colors if v[0] == vehicle_id]
        axs.plot(x_tour, y_tour, c=vehicle_color[0][1])
        axs.plot(x_tour[1:-1], y_tour[1:-1], 'o')
        axs.plot([0],[0], '^', c='red')
    axs.set_title(timetable + ' ' + str(day))
    return vehicles_used

def save_pheromone(algorithm, dataset, execution_n, log_pheromone):
    n = len(log_pheromone)
    figure, axis = plt.subplots(n, 2, figsize=(13, 3*n))

    for i in range(n):
        m = log_pheromone[i][0]
        mat_ = figure.axes[2*i].matshow(m)
        figure.colorbar(mat_)

        s_ = ''
        for p in log_pheromone[i][1]:
            fit = p[0]
            arr = p[1]
            s_ += str(fit) + ' ' + np.array_str(arr) + '\n'
        for side in ['top', 'right', 'bottom', 'left']:
            figure.axes[2*i+1].spines[side].set_visible(False)
        figure.axes[2*i+1].set_yticklabels([])
        figure.axes[2*i+1].set_xticklabels([])
        figure.axes[2*i+1].text(x=0, y=0, s=s_)
        print (f'it {i}: MAXX   {m.max()} {m.sum()}')
    figure.savefig('matrix_pheromone.svg')
    figure.savefig('matrix_pheromone.png')
    plt.close()


def test_log_lns(dataset, log_solutions_obj, params):
    alpha_1 = 1 / (1 + params['lns']['delta'] / params['lns']['ub_2'])
    alpha_3 = 1 / (1 + params['lns']['ub_1'] / params['lns']['delta'])
    alpha_2 = (alpha_1 + alpha_3) / 2
    alpha = [alpha_1, alpha_2, alpha_3]
    for i in range(3):
        fig, axs = plt.subplots(4, 1)
        fig.suptitle('LNS (' + str(i) + ') ' + dataset + ' f = ' + str(i) + ' alpha=' + str(alpha[i]))
        fig.set_figheight(18)
        fig.set_figwidth(20)
        log_solutions = log_solutions_obj[i]
        log_ws = [l[0] for l in log_solutions]
        log_f_1 = [l[1] for l in log_solutions]
        log_f_2 = [l[2] for l in log_solutions]
        log_f_3 = [l[3] for l in log_solutions]
        test_sub_log_lns(dataset, log_ws, 'weigthed sum function obj ' + str(i), axs[0])
        test_sub_log_lns(dataset, log_f_1, 'f_1', axs[1])
        test_sub_log_lns(dataset, log_f_2, 'f_2', axs[2])
        test_sub_log_lns(dataset, log_f_3, 'f_3', axs[3])
        save_result(dataset, 'lns-' + str(i) + '.png', 'lns', False)
        plt.close()

def test_sub_log_lns(dataset, log, title, axs):
    axs.set_title(title + ' ' + dataset)
    axs.set_xlabel('Iteración')
    axs.set_ylabel('Costo')
    for l in log:
        axs.plot(l)


def write_statistics(dataset, statistics, algorithm):
    if '.txt' in dataset:
        new_path = 'results/' + dataset.replace('.txt', '') + '/' + algorithm + '/'
        if not os.path.exists(new_path):
            os.mkdir(new_path)
    elif '.vrp' in dataset:
        new_path = 'results/' + dataset.replace('.vrp', '') + '/' + algorithm + '/'
        if not os.path.exists(new_path):
            os.mkdir(new_path)
    else:
        raise ()
    new_path += 'statistics.txt'
    f = open(new_path, "w")
    s = ' '.join([str(statistics[k]) for k in statistics])
    f.write(s)
    f.close()



def get_statistics(A, log_hypervolumen, log_solutions_added, duration):
    statistics = {}
    l = [a.f_1 for a in A]
    statistics['min_time_tour'] = min(l)
    statistics['max_time_tour']= max(l)
    statistics['avg_time_tour'] = sum(l)/len(l)
    l1 = [a.f_2 for a in A]
    statistics['min_arrival_time'] = min(l1)
    statistics['max_arrival_time'] = max(l1)
    statistics['avg_arrival_time'] = sum(l1)/len(l1)
    l2 = [a.f_3 for a in A]
    statistics['min_vehicle'] = min(l2)
    statistics['max_vehicle'] = max(l2)
    statistics['avg_vehicle'] = sum(l2)/len(l2)
    statistics['n_solutions_archive'] = len(A)
    statistics['duration_segs'] = duration
    statistics['log_hypervolumen'] = log_hypervolumen
    statistics['log_solutions_added'] = log_solutions_added
    return statistics

def save_statistics(statistics, execution_dir):
    params = json.dumps(statistics)
    file = execution_dir + 'statistics.json'
    f = open(file, "w")
    f.write(params)
    f.close()

# DEPRECATED
def save_one_evaluations(algorithm, file, execution_n, ref_point, archive, evals):
    ref_point_hv = np.array(ref_point)
    ind = HV(ref_point=ref_point_hv)
    w_r2 = np.array([[0, 0, 1],
                     [0, 0.33336843, 0.66663157],
                     [0, 0.66666837, 0.33333163],
                     [0, 1, 0],
                     [0.33336422, 0.66663578, 0],
                     [0.33336466, 0, 0.66663534],
                     [0.33398353, 0.33403976, 0.33197671],
                     [0.66663531, 0.33336469, 0],
                     [0.66667155, 0, 0.33332845],
                     [1, 0, 0]])
    columnas = ['Algoritmo', 'Problema', '#Ejecución', '#Evaluaciones', 'Indicador', 'Valor']
    rows = []
    hv_a = ind(archive)
    # TODO talvez no calcular el punto de referencia cada vez
    z_r2 = get_reference_point_r2(archive)
    r2_a = r2(archive, w_r2, z_r2)
    rs_a = indicator_s_energy(archive, s=2)
    row_hv = [algorithm, file, execution_n, evals, 'HV', hv_a]
    row_r2 = [algorithm, file, execution_n, evals, 'R2', r2_a]
    row_es = [algorithm, file, execution_n, evals, 'Es', rs_a]
    rows.append(row_hv)
    rows.append(row_r2)
    rows.append(row_es)
    df = pd.DataFrame(np.array(rows), columns=columnas)
    if algorithm == 'cibaco':
        file_stats = 'evaluations-cibaco.csv'
    elif algorithm == 'ibaco-eps':
        file_stats = 'evaluations-ibaco-eps.csv'
    elif algorithm == 'ibaco-hv':
        file_stats = 'evaluations-ibaco-hv.csv'
    elif algorithm == 'ibaco-r2':
        file_stats = 'evaluations-ibaco-r2.csv'
    df.to_csv(file_stats, mode='a', index=False, header=False)


def save_evaluations(algorithm, file, execution_n,  log_evaluations):
    ref_point_hv = get_reference_point_file(file)
    ind = HV(ref_point=ref_point_hv)
    w_r2 = get_reference_directions("energy", 3, 30, seed=1)
    z_r2 = np.array([0, 0, 0])
    columnas = ['Algoritmo', 'Problema', 'Ejecución', 'Evaluaciones', 'Indicador', 'Valor']
    rows = []
    for p in log_evaluations:
        evals = p[0]
        archive = p[1]
        if (archive >= ref_point_hv).any():
            print ('REBASED ------------------ ')
            print (archive)
            raise('Hypervolume point rebased')
        hv_a = ind(archive)
        r2_a = r2(archive, w_r2, z_r2)
        rs_a = indicator_s_energy(archive, s=2)
        row_hv = [algorithm, file, execution_n, evals, 'HV', hv_a]
        row_r2 = [algorithm, file, execution_n, evals, 'R2', r2_a]
        row_es = [algorithm, file, execution_n, evals, 'Es', rs_a]
        rows.append(row_hv)
        rows.append(row_r2)
        rows.append(row_es)
    """
    df = pd.DataFrame(np.array(rows), columns=columnas)
    if algorithm == 'cibaco':
        file_stats = 'evaluations-cibaco.csv'
    elif algorithm == 'ibaco-eps':
        file_stats = 'evaluations-ibaco-eps.csv'
    elif algorithm == 'ibaco-hv':
        file_stats = 'evaluations-ibaco-hv.csv'
    elif algorithm == 'ibaco-r2':
        file_stats = 'evaluations-ibaco-r2.csv'
    df.to_csv(file_stats, mode='a', index=False, header=False)
    """
    df = pd.DataFrame(np.array([row_hv]), columns=columnas)
    df.to_csv('evaluations-hv.csv', mode='a', index=False, header=False)
    df = pd.DataFrame(np.array([row_r2]), columns=columnas)
    df.to_csv('evaluations-r2.csv', mode='a', index=False, header=False)
    df = pd.DataFrame(np.array([row_es]), columns=columnas)
    df.to_csv('evaluations-es.csv', mode='a', index=False, header=False)

def plot_archive_3d(X, dataset, execution_dir):
    A = [[x.f_1, x.f_2, x.f_3] for x in X]
    A = np.array(A)
    plot = Scatter(figsize=(10, 6))
    plot.add(A)
    plot.title = 'Non epsilon dominated solutions (3D) ' + dataset
    plot.save(execution_dir + '3d-archive.png')
    plt.close()


def plot_archive_2d(A, dataset, execution_dir):
    different_vehicles = [a.f_2 for a in A]
    different_vehicles = list(set(different_vehicles))
    different_vehicles.sort()
    colors = ['blue', 'red', 'green', 'black', 'cyan', 'magenta', 'yellow']
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()
    plt.title('Non epsilon dominated solutions (2D) ' + dataset)
    ax.set_xlabel('$f_{3}$')
    ax.set_ylabel('$f_{1}$')
    for i, n_vehicle in enumerate(different_vehicles):
        a_vehicle = [a for a in A if a.f_2 == n_vehicle]
        xs = [a.f_3 for a in a_vehicle]
        ys = [a.f_1 for a in a_vehicle]
        ax.scatter(xs, ys, marker='o', c=colors[i], label='$f_{2}$='+str(n_vehicle))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_name = 'archive-2d.png'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def get_execution_dir(dataset, algorithm, run=True):
    if run:
        dir = 'results/'
    else:
        dir = 'test/'
    name_dir = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if '.txt' in dataset:
        name_dataset = dataset.replace('.txt', '')
    else:
        name_dataset = dataset.replace('.vrp', '')
    new_path = dir + name_dataset + '/' + algorithm + '/' + name_dir + '/'
    while os.path.exists(new_path):
        name_dir = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        new_path = dir + name_dataset + '/' + algorithm + '/' + name_dir + '/'
    os.makedirs(new_path)
    return new_path

def get_reference_point_r2(population):
    m = population[0].shape[0]
    z = np.zeros((m,))
    max_j = np.zeros((m,))
    min_j = np.zeros((m,))
    for j in range(m):
        val = [p[j] for p in population]
        min_j[j] = min(val)
        max_j[j] = max(val)
    max_diff = max([max_j[j] - min_j[j] for j in range(m)])
    for i in range(m):
        z[i] = min_j[i] - 2 * max_diff + (max_j[i] - min_j[i])
    return z

def r2_dep(A, W, z):
    acc = 0
    m = A[0].shape[0]
    for w in W:
        p = []
        for a in A:
            p.append(max([w[i] * abs(a[i] - z[i]) for i in range(m)]))
        acc += min(p)
    r = (1 / len(W)) * acc
    return r

def r2(A, W, z):
    diff = abs(A-z)
    d = np.array([d*W for d in diff])
    e = np.max(d, axis=-1)
    f = np.min(e, axis=0)
    f = f.sum()
    f = (1 / len(W)) * f.item()
    return f


def distance_s_energy(i, j, s):
    return 1 / (np.linalg.norm(i-j, 2)**s)


def indicator_s_energy(A, s):
    distances = [distance_s_energy(a, a_, s) for a in A for a_ in A if not np.allclose(a, a_)]
    return sum(distances)

def get_reference_point_file(dataset):
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
    else:
        raise('not reference point available for dataset')


def critic_diagram(file):
    data = pd.read_csv(file, index_col=0)
    result = autorank(data, alpha=0.05, verbose=False, force_mode='nonparametric')
    plot_stats(result, allow_insignificant=True)
    output_file = file.replace('.csv', '.png')
    plt.savefig(output_file)

def boxplot(file, dataset, output_file, title):
    algorithms = ['cmibaco', 'ibaco-eps', 'ibaco-hv', 'ibaco-r2', 'ibaco-ws']
    df = pd.read_csv(file)
    data = pd.DataFrame()
    for algorithm in algorithms:
        populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Evaluaciones == 27000")
        populations.reset_index(drop=True, inplace=True)
        data = pd.concat([data, populations['Valor']], axis=1, ignore_index=True)
    data.rename(columns={0: 'cmibaco', 1: 'ibaco-eps', 2: 'ibaco-hv', 3: 'ibaco-r2', 4: 'ibaco-ws'}, inplace=True)
    print ('boxplot ---- ')
    print (data)
    fig = plt.figure()
    plt.title(title)
    bplot = data.boxplot(column=list(data.columns))
    fig.axes.append(bplot)
    plt.savefig(output_file)
    plt.close()


def get_i_dir(dataset, algorithm, i):
    dir = 'results/' + dataset.replace('.txt', '') + '/' + algorithm + '/'
    dirs = os.listdir(dir)
    dirs = [(d, os.path.getmtime(os.path.join(dir, d))) for d in dirs]
    dirs.sort(key=lambda x: x[1])
    dirs = [d[0] for d in dirs if d[0].startswith('2023') or d[0].startswith('2024')]
    if len(dirs) != 20:
        print(f'{len(dirs)} - {dataset} - {algorithm}')
        raise()
    return dirs[i]

def get_medians_files(problems):
    problems_medians = {}
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        dataset = problem[0]
        problems_medians[dataset] = {}
        algorithms = ['cmibaco', 'ibaco-eps', 'ibaco-hv', 'ibaco-r2', 'ibaco-ws']
        for indicator in indicators:
            problems_medians[dataset][indicator] = {}
            file = 'evaluations-' + indicator + '.csv'
            df = pd.read_csv(file)
            data = pd.DataFrame({'Evaluacion': range(0,20)})
            for algorithm in algorithms:
                populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Evaluaciones == 27000")
                populations.reset_index(drop=True, inplace=True)
                data = pd.concat([data, populations['Valor']], axis=1, ignore_index=True)
            data.rename(columns={0: 'evaluacion', 1: 'cmibaco', 2: 'ibaco-eps', 3: 'ibaco-hv', 4: 'ibaco-r2', 5: 'ibaco-ws'}, inplace=True)
            for algorithm in algorithms:
                #print (data.sort_values(by=[algorithm]))
                n_eval_median_alg = int(data.sort_values(by=[algorithm]).iloc[9]['evaluacion'])
                #print (n_eval_median_alg)
                dir_median = get_i_dir(dataset, algorithm, n_eval_median_alg)
                #print (algorithm, n_eval_median_alg, dir_median)
                problems_medians[dataset][indicator][algorithm] = dir_median
                #print (problems_medians)
    return problems_medians

def plot_medians_log(problem_medians):
    problems = problem_medians.keys()
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        for indicator in indicators:
            plt.xlabel('iterations', fontsize=14)
            plt.ylabel('HV', fontsize=14)
            for algorithm in problem_medians[problem][indicator]:
                if algorithm == 'cmibaco':
                    label = 'cMIBACO'
                elif algorithm == 'ibaco-eps':
                    label = 'IBACO$_{\epsilon^+}$'
                elif algorithm == 'ibaco-hv':
                    label = 'IBACO$_{HV}$'
                elif algorithm == 'ibaco-r2':
                    label = 'IBACO$_{R2}$'
                elif algorithm == 'ibaco-ws':
                    label = 'IBACO$_{ws}$'
                dir = problem_medians[problem][indicator][algorithm]
                f = open('results/' + problem.replace('.txt', '') + '/' + algorithm + '/' + dir + '/statistics.json', 'r')
                data = json.load(f)
                serie = data['log_hypervolumen']
                f.close()
                plt.plot(serie, label=label)
            plt.legend()
            output_file = 'medians/' + problem[13:] + '-' + indicator + '.pdf'
            plt.suptitle(problem.replace('.txt', ' ') + '- Hypervolume', fontsize=14)
            plt.savefig(output_file)
            plt.close()



def plot_fronts(problem_medians):
    problems = problem_medians.keys()
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        for indicator in indicators:
            for algorithm in problem_medians[problem][indicator]:
                dir = problem_medians[problem][indicator][algorithm]
                archive = np.load('results/' + problem.replace('.txt', '') + '/' + algorithm + '/' + dir + '/archive.npy')
                plot = Scatter(figsize=(10, 7))
                plot.add(archive)
                if algorithm == 'cmibaco':
                    title = 'cMIBACO '
                elif algorithm == 'ibaco-eps':
                    title = 'IBACO$_{\epsilon^+}$ '
                elif algorithm == 'ibaco-r2':
                    title = 'IBACO$_{R2}$ '
                elif algorithm == 'ibaco-hv':
                    title = 'IBACO$_{HV}$ '
                elif algorithm == 'ibaco-ws':
                    label = 'IBACO$_{ws}$'
                #plot.title = (title, {'fontsize':22})
                plot.save('fronts/'+ problem[13:] + '-' + indicator + '-' + algorithm + '.pdf', bbox_inches="tight", pad_inches=0.25)
                plt.close()

def get_table_mean(problems, file, output_file, indicator):
    algorithms = ['cmibaco', 'ibaco-eps', 'ibaco-hv', 'ibaco-r2', 'ibaco-ws']
    df = pd.read_csv(file)
    data_total = pd.DataFrame(columns=algorithms)
    for problem in problems:
        dataset = problem[0]
        data = pd.DataFrame()
        for algorithm in algorithms:
            populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}'")
            populations.reset_index(drop=True, inplace=True)
            data = pd.concat([data, populations['Valor']], axis=1, ignore_index=True)
        data.rename(columns={0: 'cmibaco', 1: 'ibaco-eps', 2: 'ibaco-hv', 3: 'ibaco-r2', 4: 'ibaco-ws'}, inplace=True)
        data_mean = data[algorithms].mean(numeric_only=True)
        data_std = data[algorithms].std(numeric_only=True)
        # print (f'>>>>>>>>> PROBLEMS {problem}')
        # print (data_mean)
        # print (data_std)
        # print ('<<<<<<<<<<')
        row_stats = {}
        row_mean = {}
        row_std = {}
        for algorithm in algorithms:
            mean_alg = data_mean.loc[algorithm]
            std_alg = data_std.loc[algorithm]
            stats = str(f'{mean_alg:.2e}') + ' (' + str(f'{std_alg:.2e}') + ')'
            row_mean[algorithm] = mean_alg
            row_std[algorithm] = std_alg
            row_stats[algorithm] = stats
        data_total = data_total._append(row_stats, ignore_index=True)
        #print (data)
        #print (data_mean.loc['cibaco'])
        #print (data_total)
        #print (data_mean)
        #print (data_std)
        #raise()

    problems_column = [[p[0].replace('_', '$\_$').replace('.txt', ''), indicator] for p in problems]
    problems_column = pd.DataFrame(problems_column, columns=['dataset', 'indicator'])
    data_total = pd.concat([problems_column, data_total], axis=1)
    data_total.to_latex(output_file, column_format='ccrrrr', index=False)
    print(data_total)




def plot_general_table(problems, file, output_file):
    algorithms = ['cmibaco', 'ibaco-eps', 'ibaco-hv', 'ibaco-r2', 'ibaco-ws']
    df = pd.read_csv(file)
    data = pd.DataFrame()
    for problem in problems:
        dataset = problem[0]
        data_sub = pd.DataFrame()
        for algorithm in algorithms:
            populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}'")
            populations.reset_index(drop=True, inplace=True)
            data_sub = pd.concat([data_sub, populations['Valor']], axis=1, ignore_index=True)
        data = pd.concat([data, data_sub], ignore_index=True)
    data.rename(columns={0: 'cmibaco', 1: 'ibaco-eps', 2: 'ibaco-hv', 3: 'ibaco-r2', 4: 'ibaco-ws'}, inplace=True)
    data.to_csv(output_file.replace('.png', '.csv'), index=False)
    result = autorank(data, alpha=0.05, verbose=False, force_mode='nonparametric')
    fig, ax = plt.subplots(figsize=(15,25))
    ax = plot_stats(result, allow_insignificant=True)
    #ax.set_title(dataset + ' ' + indicator)
    fig.axes.append(ax)
    #fig.suptitle(dataset + ' ' + indicator)
    plt.savefig(output_file)
    plt.close()

def plot_general_diagram():
    file_es = 'total-evaluations-es.csv'
    file_hv = 'total-evaluations-hv.csv'
    file_r2 = 'total-evaluations-r2.csv'
    df_es = pd.read_csv(file_es)
    df_hv = pd.read_csv(file_hv)
    df_r2 = pd.read_csv(file_r2)
    df_r2 *= -1
    df_es *= -1
    total_df = pd.concat([df_es, df_hv, df_r2], axis=0)
    total_df.rename(columns={'cmibaco': 'cMIBACO', 'ibaco-eps': 'IBACO$_{\epsilon^+}$', 'ibaco-hv': 'IBACO$_{HV}$', 'ibaco-r2': 'IBACO$_{R2}$', 'ibaco-ws': 'IBACO$_{ws}$'}, inplace=True)
    result = autorank(total_df, alpha=0.05, verbose=False, force_mode='nonparametric')
    fig, ax = plt.subplots(figsize=(15, 25))
    ax = plot_stats(result, allow_insignificant=True)
    # ax.set_title(dataset + ' ' + indicator)
    fig.axes.append(ax)
    #fig.suptitle('Critic difference diagram')
    #plt.suptitle('Critic difference diagram')
    #plt.title('Critic difference diagram - cMIBACO - IBACO$_{\epsilon^+}$ - IBACO$_{HV}$ - IBACO$_{R2}$')
    output_file = 'ranking-total.pdf'
    #plt.savefig(output_file,  bbox_inches="tight", pad_inches=0.15)
    plt.savefig(output_file,  bbox_inches="tight")
    plt.close()

def foo_diagram():
    file = 'foo.csv'
    df = pd.read_csv(file)
    result = autorank(df, alpha=0.05,verbose=False, force_mode='nonparametric')
    fig, ax = plt.subplots(figsize=(15, 25))
    ax = plot_stats(result, allow_insignificant=True)
    fig.axes.append(ax)
    plt.suptitle('Critic difference diagram')
    output_file = 'ranking-foo.png'
    plt.savefig(output_file)
    plt.close()

def delete_alg_dt(file, dataset, algorithm):
    df = pd.read_csv(file)
    print(f'before drop {df.shape} {file}')
    index_r = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}'").index
    df.drop(index_r, inplace=True)
    print(f'after drop  {df.shape} {file}')
    df.to_csv(file, index=False)

def get_multiple_logs_hyp(dirs, dataset, algorithm):
    path = 'results/' + dataset + '/' + algorithm + '/'
    pairs = []
    for d in dirs:
        f = open(path + d + '/statistics.json')
        p = json.load(f)
        log = p['log_hypervolumen']
        pairs.append((log, d))
    plt.figure(figsize=(25, 7))
    for p in pairs:
        plt.plot(p[0], label=p[1])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('multi-logs-hv.png')




if __name__ == '__main__':
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


    cmibaco_1 = ['2023-12-27-19-04-47', '2023-12-27-19-16-30', '2023-12-27-19-28-12', '2023-12-27-19-39-52',
                 '2023-12-27-19-51-31', '2023-12-27-20-03-23', '2023-12-27-20-15-07', '2023-12-27-20-26-48',
                 '2023-12-27-20-39-41', '2023-12-27-20-51-58', '2023-12-27-21-03-38', '2023-12-27-21-15-24',
                 '2023-12-27-21-27-18', '2023-12-27-21-39-14', '2023-12-27-21-50-58', '2023-12-27-22-02-39',
                 '2023-12-27-22-14-19', '2023-12-27-22-26-03', '2023-12-27-22-37-47', '2023-12-27-22-49-34']

    #get_multiple_logs_hyp(cmibaco_1, 'Christofides_8_5_0.5', 'cmibaco')

    problems = [('Christofides_1_5_0.5.txt', 'ch1505-'),
                ('Christofides_2_5_0.5.txt', 'ch2505-'),
                ('Christofides_3_5_0.5.txt', 'ch3505-'),
                ('Christofides_4_5_0.5.txt', 'ch4505-'),
                ('Christofides_5_5_0.5.txt', 'ch5505-'),
                ('Christofides_6_5_0.5.txt', 'ch6505-'),
                ('Christofides_7_5_0.5.txt', 'ch7505-'),
                ('Christofides_8_5_0.5.txt', 'ch8505-'),
                ('Christofides_9_5_0.5.txt', 'ch9505-')]
    problems_medians = get_medians_files(problems)
    print(problems_medians)
    plot_medians_log(problems_medians)
    print(problems_medians['Christofides_2_5_0.5.txt']['hv'])
    plot_fronts(problems_medians)
    #raise ()

    indicators = ['hv', 'es', 'r2']
    # Tabla de promedio y desvet por indicador
    get_table_mean(problems, 'evaluations-hv.csv', 'table-hv-n.tex', 'HV')
    get_table_mean(problems, 'evaluations-r2.csv', 'table-r2-n.tex', 'R2')
    get_table_mean(problems, 'evaluations-es.csv', 'table-es-n.tex', '$E_s$')


    # Diagrama critico de cada indicador
    plot_general_table(problems, 'evaluations-hv.csv', 'total-evaluations-hv.png')
    plot_general_table(problems, 'evaluations-r2.csv', 'total-evaluations-r2.png')
    plot_general_table(problems, 'evaluations-es.csv', 'total-evaluations-es.png')
    # Diagrama critico general
    plot_general_diagram()

    raise ()

    # boxplots
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        dataset = problem[0]
        base = problem[1]
        for ind in indicators:
            file = 'evaluations-' + ind + '.csv'
            output_file = base + ind +'.png'
            output_file_box = base + ind +'-box.png'
            boxplot(file, dataset, output_file_box, dataset + ' ' + ind)