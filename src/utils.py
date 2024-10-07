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

from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats

from maco import non_dominated
import lzma
import seaborn as sns

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
    if 'ibaco-eps' in params.keys():
        params_to_save['ibaco-eps'] = params['ibaco-eps']
    if 'ibaco-hv' in params.keys():
        params_to_save['ibaco-hv'] = params['ibaco-hv']
    if 'ibaco-r2' in params.keys():
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
    file_name += '-object.xz'
    with lzma.open(execution_dir + file_name, 'wb') as file:
        pickle.dump(A, file)

def save_all_solutions(A, execution_dir):
    np.save(execution_dir + 'all-solutions', A)

def save_front(A, execution_dir):
    np.save(execution_dir + 'front', A)

def plot_log_hypervolume(log, dataset, execution_dir):
    plt.title('Hypervolume archive per iteration ' + dataset)
    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    plt.plot(log)
    fig_name = 'log-hyper.pdf'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_log_solutions_added(log, dataset, execution_dir):
    plt.title('Archive size per iteration ' + dataset)
    plt.xlabel('Iteration')
    plt.ylabel('# Archive size ')
    plt.plot(log)
    fig_name = 'log-solutions.pdf'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_solution_vrp(dataset, solution, title, fig_name, execution_dir):
    fig, axs = plt.subplots(len(solution.timetables), solution.days)
    fig.set_figheight(11) # 18
    fig.set_figwidth(20)
    vehicles_used = []
    np.random.seed(5)
    vehicle_colors = [(v.id, tuple(np.random.rand(3,))) for v in solution.assigments_vehicles]
    for i, t in enumerate(solution.timetables):
        for d in range(solution.days):
            vehicles_used += plot_sub_vrp(solution, t, d, axs[i, d], vehicle_colors)
    vehicles_used = list(set(vehicles_used))
    colors_patch = [mpatches.Patch(color=v[1], label='Vehicle ' + str(v[0])) for v in [vehicle_colors[i] for i in vehicles_used]]
    fig.legend(handles=colors_patch, fontsize=16, loc='center right', bbox_to_anchor=(1, 0.5))
    fig.suptitle(title, fontsize=20)
    save_result(dataset, fig_name, execution_dir)
    plt.close()


def plot_best_objective(A, dataset, objective, execution_dir):
    if objective == 0:
        best = [a.f_1 for a in A]
        ibest = np.argmin(best)
        best = A[ibest]
        title = 'Best total tours time '
        fig_name = 'best-solution-time-tour.pdf'
    elif objective == 1:
        best = min([a.f_2 for a in A])
        best = min([a.f_1 for a in A if a.f_2 == best])
        best = [a for a in A if a.f_1 == best][0]
        title = 'Best consistency driver '
        fig_name = 'best-solution-driver-diff.pdf'
    elif objective == 2:
        best = [a.f_3 for a in A]
        ibest = np.argmin(best)
        best = A[ibest]
        title = 'Best maximum arrival time difference '
        fig_name = 'best-solution-arrival-diff.pdf'
    title += '[' + str(best.f_1) + ', ' + str(best.f_2) + ', ' + str(best.f_3) + ']'
    plot_solution_vrp(dataset, best, title, fig_name, execution_dir)

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
        axs.plot(x_tour, y_tour, c=vehicle_color[0][1], linewidth=2)
        axs.plot(x_tour[1:-1], y_tour[1:-1], 'o', linewidth=2)
        axs.plot([0],[0], '^', c='red')
    axs.set_title(timetable + ' - day ' + str(day), fontsize=20)
    return vehicles_used


def save_pheromone(algorithm, dataset, execution_n, log_pheromone, labels):
    path_dataset = 'pheromone/' + dataset.replace('.vrp', '') + '/'
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)
    n = len(log_pheromone)
    algorithm_label = labels[algorithm]
    print (f'{algorithm} - {execution_n} saved to {path_dataset + algorithm}-{str(execution_n)}.pdf')
    for i in range(n):
        fitness = log_pheromone[i][1]
        for fit, f_i in fitness:
            print (f'fitness : {fit} | {f_i}')

        matrices = log_pheromone[i][0]
        days = len(matrices['AM'])
        figure, axis = plt.subplots(1, days*2, figsize=(13, 3))
        for i, t in enumerate(['AM', 'PM']):
            for d in range(days):
                m = matrices[t][d]
                color_mat = axis[2*i+d].matshow(m)
                axis[2*i+d].set_yticklabels([])
                axis[2*i+d].set_xticklabels([])
                axis[2*i+d].set_title(f'{t} - {d}')
                figure.colorbar(color_mat)
        figure.suptitle(f'{algorithm_label} - Iteration {execution_n}')
        plt.savefig(path_dataset + algorithm + '-' + str(execution_n) + '.pdf')
        plt.close()



def save_pheromone_(algorithm, dataset, execution_n, log_pheromone):
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
    figure.savefig('matrix_pheromone.pdf')
    plt.close()


def test_log_lns(params, dataset, log_solutions_obj, exectution_dir):
    alpha_1 = 1 / (1 + params['lns']['delta'] / params['lns']['ub_2'])
    alpha_3 = 1 / (1 + params['lns']['ub_1'] / params['lns']['delta'])
    alpha_2 = (alpha_1 + alpha_3) / 2
    alpha = [alpha_1, alpha_2, alpha_3]
    for i in range(3):
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("MDLNS - " + dataset)
        fig.subplots_adjust(hspace=0.4)
        fig.set_figheight(9)
        fig.set_figwidth(19)
        log_solutions = log_solutions_obj[i]
        log_ws = [l[0] for l in log_solutions]
        log_f_1 = [l[1] for l in log_solutions]
        log_f_2 = [l[2] for l in log_solutions]
        log_f_3 = [l[3] for l in log_solutions]
        test_sub_log_lns(dataset, log_ws, "$f'_" + str(i+1) + "$ = $f_1$ + " + r"$\alpha_"+str(i+1)+"$" + "(1- " + r"$\alpha_"+str(i+1)+"$)" + "$f_3$ - ", axs[0])
        test_sub_log_lns(dataset, log_f_1, '$f_1$ - ', axs[1])
        #test_sub_log_lns(dataset, log_f_2, '$\min f_2$', axs[2])
        test_sub_log_lns(dataset, log_f_3, '$f_3$ - ', axs[2])
        save_result(dataset, 'lns-' + str(i) + '.pdf', exectution_dir)
        plt.close()

def test_log_lns_weighted(params, dataset, log_solutions_obj, exectution_dir):
    alpha_1 = 1 / (1 + params['lns']['delta'] / params['lns']['ub_2'])
    alpha_3 = 1 / (1 + params['lns']['ub_1'] / params['lns']['delta'])
    alpha_2 = (alpha_1 + alpha_3) / 2
    alpha = [alpha_1, alpha_2, alpha_3]
    for i in range(3):
        fig= plt.figure(figsize=(10,5))
        ax = plt.gca()
        log_solutions = log_solutions_obj[i]
        log_ws = [l[0] for l in log_solutions]
        test_sub_log_lns(dataset, log_ws, "$f'_" + str(i+1) + "$ = $f_1$ + " + r"$\alpha_"+str(i+1)+"$" + "(1- " + r"$\alpha_"+str(i+1)+"$)" + "$f_3$ - ", ax)
        save_result(dataset, 'lns-' + str(i) + '-ws' + '.pdf', exectution_dir)
        plt.close()

        log_f_1 = [l[1] for l in log_solutions]
        fig = plt.figure(figsize=(10, 5))
        ax = plt.gca()
        test_sub_log_lns(dataset, log_f_1, '$f_1$ - ', ax)
        save_result(dataset, 'lns-' + str(i) + '-f_1' + '.pdf', exectution_dir)
        plt.close()

        log_f_2 = [l[2] for l in log_solutions]
        fig = plt.figure(figsize=(10, 5))
        ax = plt.gca()
        test_sub_log_lns(dataset, log_f_2, '$f_2$ - ', ax)
        save_result(dataset, 'lns-' + str(i) + '-f_2' + '.pdf', exectution_dir)
        plt.close()

        log_f_3 = [l[3] for l in log_solutions]
        fig = plt.figure(figsize=(10, 5))
        ax = plt.gca()
        test_sub_log_lns(dataset, log_f_3, '$f_3$ - ', ax)
        save_result(dataset, 'lns-' + str(i) + '-f_3' + '.pdf', exectution_dir)
        plt.close()


def test_sub_log_lns(dataset, log, title, axs):
    axs.set_title(title + ' ' + dataset)
    axs.set_xlabel('Iteration')
    axs.set_ylabel('Cost')
    for l in log:
        axs.plot(l)




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
    plot.save(execution_dir + '3d-archive.pdf')
    plt.close()

def plot_front_epsilon_front(front, A, all_solutions, dataset, execution_dir):
    front = np.unique(front, axis=0)
    p_epsilon = np.array([[x.f_1, x.f_2, x.f_3] for x in A])
    p_epsilon = np.unique(p_epsilon, axis=0)
    all_solutions = np.unique(all_solutions, axis=0)

    p_epsilon = [p for p in p_epsilon if not (front == p).all(axis=1) .any()]
    p_epsilon = np.array(p_epsilon)

    q = [p for p in all_solutions if not (front == p).all(axis=1).any()]
    q = np.array(q)
    q = [p for p in q if not (p_epsilon == p).all(axis=1).any()]
    q = np.array(q)

    if q.shape[0] + front.shape[0] + p_epsilon.shape[0] != all_solutions.shape[0]:
        print (q.shape[0])
        print (front.shape[0])
        print (p_epsilon.shape[0])
        print (all_solutions.shape[0])
        raise('solutions remaining')

    plot = Scatter(figsize=(3, 2.8))
    plot.add(p_epsilon, color="blue", label="$P_{Q,\epsilon}$", s=3.7)
    plot.add(front, color="black", label="$P_{Q}$", s=3.7)
    plot.add(q, color='red', s=3.7)

    plot.title = '$P_{Q}$ (black) - $P_{Q,\epsilon}$ (blue) - $Q$ (red) - ' + '\n' + dataset.replace('.txt', '')
    plot.save(execution_dir + 'p-q-and-p-qeps.pdf', bbox_inches="tight", pad_inches=0.35)
    plt.close()


def plot_archive_2d(A, dataset, execution_dir):
    different_vehicles = [a.f_2 for a in A]
    different_vehicles = list(set(different_vehicles))
    different_vehicles.sort()
    color_vehicles = {1: 'black', 2: 'blue', 3: 'red', 4: 'green', 5: 'cyan', 6: 'magenta', 7: 'yellow'}
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()
    plt.title('Non epsilon dominated solutions (2D) ' + dataset)
    ax.set_xlabel('$f_{3}$')
    ax.set_ylabel('$f_{1}$')
    for i, n_vehicle in enumerate(different_vehicles):
        a_vehicle = [a for a in A if a.f_2 == n_vehicle]
        xs = [a.f_3 for a in a_vehicle]
        ys = [a.f_1 for a in a_vehicle]
        ax.scatter(xs, ys, marker='o', c=color_vehicles[n_vehicle], label='$f_{2}$='+str(n_vehicle))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_name = 'archive-2d.pdf'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_archive_uncertainty(scenarios, scenarios_id, dataset, execution_dir, id_solutions):
    color_vehicles = {1: 'black', 2: 'blue', 3: 'red', 4: 'green', 5: 'cyan', 6: 'magenta', 7: 'yellow'}
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot()
    ax.set_xlabel('$f_{1}$')
    ax.set_ylabel('$f_{3}$')
    plt.title('Scenarios (2D) ' + dataset.replace('.txt', ''))
    labels_scenarios = ['o', '^', 's', 'X', 'P']

    for e, A in enumerate(scenarios):
        if len(A) == 0:
            continue
        different_vehicles = [a.f_2 for a in A]
        different_vehicles = list(set(different_vehicles))
        different_vehicles.sort()
        for i, n_vehicle in enumerate(different_vehicles):
            a_vehicle = [a for a in A if a.f_2 == n_vehicle and id_solutions[a.id] <= 30]
            xs = [a.f_1 for a in a_vehicle]
            ys = [a.f_3 for a in a_vehicle]
            ax.scatter(xs, ys, marker=labels_scenarios[e], c=color_vehicles[n_vehicle], label='$f_{2}$='+str(n_vehicle)+' ('+scenarios_id[e]+')')
            for j, a in enumerate(a_vehicle):
                txt = str(id_solutions[a.id])
                ax.annotate(txt, (a.f_1, a.f_3))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_name = 'scenarios.pdf'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_worst_escenarios(worst_escenarios, scenarios_id, dataset, execution_dir, id_solutions, front=False):
    color_vehicles = {1: 'black', 2: 'blue', 3: 'red', 4: 'green', 5: 'cyan', 6: 'magenta', 7: 'yellow'}
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot()
    ax.set_xlabel('$f_{1}$')
    ax.set_ylabel('$f_{3}$')
    if front:
        plt.title('$\min_{x \in P_{Q,\epsilon}} \sup_{\delta \in \mathbb{U}} \ F(x,\delta)$ - ' + dataset.replace('.txt', ''))
    else:
        plt.title('$\sup_{x \in P_{Q,\epsilon}, \ \delta \in \mathbb{U}} \ F(x,\delta)$ - ' + dataset.replace('.txt', ''))
    labels_scenarios = {0: 'o', 1: '^', 2: 's'}
    n = len(worst_escenarios)
    scenarios = {}
    for i, e in enumerate(scenarios_id):
        scenarios[i] = []
    for i in range(n):
        solution_scenario_tuple = worst_escenarios[i]
        m = len(solution_scenario_tuple)
        for j in range(m//2):
            solution_scenario = solution_scenario_tuple[j*2]
            id_scenario = solution_scenario_tuple[j*2+1]
            scenarios[id_scenario].append(solution_scenario)
    scenarios = [scenarios[k] for k in scenarios.keys()]
    if front:
        pareto_front, _ = non_dominated([], [c for scenario in scenarios for c in scenario])
    for e, A in enumerate(scenarios):
        if len(A) == 0:
            continue
        different_vehicles = [a.f_2 for a in A]
        different_vehicles = list(set(different_vehicles))
        different_vehicles.sort()
        for i, n_vehicle in enumerate(different_vehicles):
            if front:
                a_vehicle = [a for a in A if a.f_2 == n_vehicle and a in pareto_front]
            else:
                a_vehicle = [a for a in A if a.f_2 == n_vehicle]
            xs = [a.f_1 for a in a_vehicle]
            ys = [a.f_3 for a in a_vehicle]
            ax.scatter(xs, ys, marker=labels_scenarios[e], c=color_vehicles[n_vehicle],
                       label='$f_{2}$=' + str(n_vehicle) + ' (' + scenarios_id[e] + ')')
            for j, a in enumerate(a_vehicle):
                txt = str(id_solutions[a.id])
                ax.annotate(txt, (a.f_1, a.f_3))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_name = 'scenarios.pdf'
    save_result(dataset, fig_name, execution_dir)
    plt.close()

def plot_line_solutions(i, solution_scenario_tuple, color_vehicles):
    lre_solutions = [s for s in solution_scenario_tuple if not isinstance(s, np.int64)]
    line_styles = ['-', ':', '--', '-.']
    line_colours = color_vehicles
    # CASE 1: exists dominated solutions in two objectives (plot worst cases)
    solutions = [s for s in lre_solutions if len([q for q in lre_solutions if q != s and q.f_1 <= s.f_1 and q.f_3 <= s.f_3]) != 0]
    if len(solutions) == 0:
        # CASE 2: all lre solutions are non-dominated
        solutions = [s for s in lre_solutions if
                     len([q for q in lre_solutions if q != s and q.f_1 <= s.f_1 and q.f_3 <= s.f_3]) == 0]
    line_style = i % len(line_styles)
    line_style = line_styles[line_style]
    line_colour = i % len(line_colours)
    line_colour = line_colours[line_colour + 1]

    solutions.sort(key=lambda x: x.f_1)
    line_x1 = [0, solutions[0].f_1]
    line_y1 = [solutions[0].f_3, solutions[0].f_3]
    plt.plot(line_x1, line_y1, ls=line_style, color=line_colour)
    for j in range(1, len(solutions)):
        s_x1 = [solutions[j - 1].f_1, solutions[j].f_1]
        s_y1 = [solutions[j].f_3, solutions[j].f_3]
        plt.plot(s_x1, s_y1, ls=line_style, color=line_colour)

    solutions.sort(key=lambda x: x.f_3)
    line_x1 = [solutions[0].f_1, solutions[0].f_1]
    line_y1 = [0, solutions[0].f_3]
    plt.plot(line_x1, line_y1, ls=line_style, color=line_colour)
    for j in range(1, len(solutions)):
        s_x1 = [solutions[j].f_1, solutions[j].f_1]
        s_y1 = [solutions[j].f_3, solutions[j - 1].f_3]
        plt.plot(s_x1, s_y1, ls=line_style, color=line_colour)

def plot_lightly_robust_escenarios(lre_solutions, worst_scenarios_no_lre_sub, scenarios_id, dataset,
                                           execution_dir, id_solutions):
    color_vehicles = {1: 'black', 2: 'blue', 3: 'red', 4: 'green', 5: 'cyan', 6: 'magenta', 7: 'yellow'}

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    ax.set_xlabel('$f_{1}$', fontsize=14)
    ax.set_ylabel('$f_{3}$', fontsize=14)
    plt.title('$\min_{x \in P_{Q,\epsilon}} \sup_{\\xi \in \mathbb{U}} \ F(x,\\xi)$ - ' + dataset.replace('.txt', ''))
    labels_scenarios = {0: 'o', 1: '^', 2: 's'}

    n = len(lre_solutions)
    scenarios = {}
    for i, e in enumerate(scenarios_id):
        scenarios[i] = []
    for i in range(n):
        solution_scenario_tuple = lre_solutions[i]
        m = len(solution_scenario_tuple)
        # draw set-based mix-max area
        plot_line_solutions(i, solution_scenario_tuple, color_vehicles)
        for j in range(m//2):
            solution_scenario = solution_scenario_tuple[j*2]
            id_scenario = solution_scenario_tuple[j*2+1]
            scenarios[id_scenario].append(solution_scenario)
    scenarios = [scenarios[k] for k in scenarios.keys()]
    for e, A in enumerate(scenarios):
        if len(A) == 0:
            continue
        different_vehicles = [a.f_2 for a in A]
        different_vehicles = list(set(different_vehicles))
        different_vehicles.sort()
        for i, n_vehicle in enumerate(different_vehicles):
            a_vehicle = [a for a in A if a.f_2 == n_vehicle]
            xs = [a.f_1 for a in a_vehicle]
            ys = [a.f_3 for a in a_vehicle]
            ax.scatter(xs, ys, marker=labels_scenarios[e], c=color_vehicles[n_vehicle],
                       label='$f_{2}$=' + str(n_vehicle) + ' (' + scenarios_id[e] + ')')
            for j, a in enumerate(a_vehicle):
                #txt = str(id_solutions[a.id]) + '$_{lre}$'
                #txt = str(id_solutions[a.id])
                #ax.annotate(txt, (a.f_1, a.f_3))
                pass

    n = len(worst_scenarios_no_lre_sub)
    scenarios = {}
    for i, e in enumerate(scenarios_id):
        scenarios[i] = []
    for i in range(n):
        solution_scenario_tuple = worst_scenarios_no_lre_sub[i]
        m = len(solution_scenario_tuple)
        for j in range(m // 2):
            solution_scenario = solution_scenario_tuple[j * 2]
            id_scenario = solution_scenario_tuple[j * 2 + 1]
            scenarios[id_scenario].append(solution_scenario)

    scenarios = [scenarios[k] for k in scenarios.keys()]
    for e, A in enumerate(scenarios):
        if len(A) == 0:
            continue
        different_vehicles = [a.f_2 for a in A]
        different_vehicles = list(set(different_vehicles))
        different_vehicles.sort()
        for i, n_vehicle in enumerate(different_vehicles):
            a_vehicle = [a for a in A if a.f_2 == n_vehicle]
            xs = [a.f_1 for a in a_vehicle]
            ys = [a.f_3 for a in a_vehicle]
            ax.scatter(xs, ys, marker=labels_scenarios[e], c=color_vehicles[n_vehicle],
                       label='$f_{2}$=' + str(n_vehicle) + ' (' + scenarios_id[e] + ')')
            for j, a in enumerate(a_vehicle):
                txt = str(id_solutions[a.id])
                ax.annotate(txt, (a.f_1, a.f_3))

    ax.legend(loc='lower left')
    fig_name = 'scenarios.pdf'
    save_result(dataset, fig_name, execution_dir)
    plt.close()


def get_execution_dir(dataset, algorithm, run=True, uvrp=False):
    if uvrp:
        dir = 'uncertainty/'
    elif run:
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
    if 'Christofides_1_5' in dataset:
        return np.array([10000, 7, 700])
    elif 'Christofides_2_5' in dataset:
        return np.array([10000, 6, 600])
    elif 'Christofides_3_5' in dataset:
        return np.array([10000, 6, 600])
    elif 'Christofides_4_5' in dataset:
        return np.array([12000, 7, 900])
    elif 'Christofides_5_5' in dataset:
        return np.array([16000, 9, 900])
    elif 'Christofides_6_5' in dataset:
        return np.array([16000, 9, 900])
    elif 'Christofides_7_5' in dataset:
        return np.array([12000, 8, 900])
    elif 'Christofides_8_5' in dataset:
        return np.array([15000, 8, 900])
    elif 'Christofides_9_5' in dataset:
        return np.array([15000, 9, 1000])
    elif 'Christofides_10_5' in dataset:
        return np.array([17000, 9, 1300])
    elif 'Christofides_11_5' in dataset:
        return np.array([17000, 9, 1300])
    elif 'Christofides_12_5' in dataset:
        return np.array([17000, 9, 1300])
    else:
        raise('not reference point available for dataset')


def critic_diagram(file):
    data = pd.read_csv(file, index_col=0)
    result = autorank(data, alpha=0.05, verbose=False, force_mode='nonparametric')
    plot_stats(result, allow_insignificant=True)
    output_file = file.replace('.csv', '.pdf')
    plt.savefig(output_file)

def boxplot(file, dataset, output_file, title, algorithms, labels):
    algorithms_columns = {}
    for i, a in enumerate(algorithms):
        algorithms_columns[i] = labels[a]
    df = pd.read_csv(file)
    data = pd.DataFrame()
    for algorithm in algorithms:
        populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Evaluaciones == 27000")
        populations.reset_index(drop=True, inplace=True)
        data = pd.concat([data, populations['Valor']], axis=1, ignore_index=True)
    data.rename(columns=algorithms_columns, inplace=True)
    fig = plt.figure(figsize=(8,5))
    plt.title(title.replace('.txt', ''))
    bplot = data.boxplot(column=list(data.columns))
    fig.axes.append(bplot)
    plt.savefig(output_file)
    plt.close()

def get_index_dataset(dataset, algorithm, execution_dir):
    dir = 'results/' + dataset.replace('.txt', '').replace('.vrp', '') + '/' + algorithm + '/'
    dirs = os.listdir(dir)
    dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
    dirs = [(d, datetime.strptime(d, '%Y-%m-%d-%H-%M-%S')) for d in dirs]
    dirs.sort(key=lambda x: x[1])
    dirs = [d[0] for d in dirs]
    if len(dirs) != 20:
        print(f'{len(dirs)} - {dataset} - {algorithm}')
        raise()
    return dirs.index(execution_dir)


def get_i_dir(dataset, algorithm, i):
    dir = 'results/' + dataset.replace('.txt', '').replace('.vrp', '') + '/' + algorithm + '/'
    dirs = os.listdir(dir)
    dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
    dirs = [(d, datetime.strptime(d, '%Y-%m-%d-%H-%M-%S')) for d in dirs]
    dirs.sort(key=lambda x: x[1])
    dirs = [d[0] for d in dirs]
    #dirs = dirs[:20]
    if len(dirs) != 20:
        print(f'{len(dirs)} - {dataset} - {algorithm}')
        raise()
    return dirs[i]

# Median according aproximated solutions
def get_medians_files(problems, algorithms):
    problems_medians = {}
    indicators = ['hv', 'es', 'r2']
    algorithms_columns = {0: 'evaluacion'}
    for i, algorithm in enumerate(algorithms):
        algorithms_columns[i+1] = algorithm
    for dataset in problems:
        problems_medians[dataset] = {}
        for indicator in indicators:
            problems_medians[dataset][indicator] = {}
            file = 'evaluations-' + indicator + '.csv'
            df = pd.read_csv(file)
            data = pd.DataFrame({'Evaluacion': range(0,20)})
            for algorithm in algorithms:
                populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Evaluaciones == 27000")
                populations = populations.sort_values(by=['Evaluaciones'])
                populations.reset_index(drop=True, inplace=True)
                data = pd.concat([data, populations['Valor']], axis=1, ignore_index=True)
            data.rename(columns=algorithms_columns, inplace=True)
            for algorithm in algorithms:
                #print (data.sort_values(by=[algorithm]))
                n_eval_median_alg = int(data.sort_values(by=[algorithm]).iloc[9]['evaluacion'])
                #print (n_eval_median_alg)
                dir_median = get_i_dir(dataset, algorithm, n_eval_median_alg)
                #print (algorithm, n_eval_median_alg, dir_median)
                problems_medians[dataset][indicator][algorithm] = dir_median
                #print (problems_medians)
    return problems_medians


def get_table_time(problems, file_output, main_algorithm, other_algorithms, labels):
    algorithms = [main_algorithm] + other_algorithms
    algorithms_columns = {}
    for i, a in enumerate(algorithms):
        algorithms_columns[i] = labels[a]
    data_times = pd.DataFrame(columns=algorithms)
    data_total = pd.DataFrame()
    for (problem, _) in problems:
        row_times = {}
        data_sub = pd.DataFrame()
        for algorithm in algorithms:
            dir = 'results/' + problem.replace('.txt', '').replace('.vrp', '') + '/' + algorithm + '/'
            dirs = os.listdir(dir)
            dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
            if len(dirs) != 20:
                raise('should be 20 files')
            time = []
            for d in dirs:
                f = open(dir + d + '/statistics.json', 'r')
                data = json.load(f)
                time.append(data['duration_segs'])
            time = np.array(time)
            data_sub = pd.concat([data_sub, pd.DataFrame(time)], axis=1, ignore_index=True)
            if algorithm == main_algorithm:
                coop = time.copy()
            mean_time = time.mean()
            std_time = time.std()
            if algorithm == main_algorithm:
                row_times[algorithm] = str(f'{mean_time:.3e} ({std_time:.3e})')
            else:
                w = wilcoxon(coop, time)
                if w.pvalue > 0.05:
                    row_times[algorithm] = str(f'{mean_time:.3e} ({std_time:.3e})') + ' $\\leftrightarrow$'
                else:
                    if mean_time > coop.mean():
                        row_times[algorithm] = str(f'{mean_time:.3e} ({std_time:.3e})') + ' $\\downarrow$'
                    else:
                        row_times[algorithm] = str(f'{mean_time:.3e} ({std_time:.3e})') + ' $\\uparrow$'

        data_times = data_times._append(row_times, ignore_index=True)
        data_total = pd.concat([data_total, data_sub], ignore_index=True)
    problems_column = [[p[0].replace('_', '$\_$').replace('.txt', '').replace('.vrp', ''), 'time(secs)'] for p in problems]
    problems_column = pd.DataFrame(problems_column, columns=['dataset', 'mean time(secs)'])
    data_times = pd.concat([problems_column, data_times], axis=1)
    data_times.to_latex(file_output, column_format='ccrrrrr', index=False)
    data_total.rename(columns=algorithms_columns, inplace=True)
    data_total.to_csv('total-evaluations-time.csv', index=False)

# Compare the median of current algorithm against the other algorithms in the same parameters
def plot_median_algorithm_compare(problem_medians, main_algorithm, other_algorithms, labels):
    problems = problem_medians.keys()
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        for indicator in indicators:
            plt.xlabel('evaluations', fontsize=14)
            if indicator == 'hv':
                label_indicator = 'HV'
                plt.ylabel('HV', fontsize=14)
            elif indicator == 'es':
                label_indicator = '$E_s$'
                plt.ylabel('$E_s$', fontsize=14)
            elif indicator == 'r2':
                label_indicator = 'R2'
                plt.ylabel('R2', fontsize=14)
            data = pd.read_csv('evaluations-' + indicator + '.csv')
            median_algorithm_dir = problem_medians[problem][indicator][main_algorithm]
            index_problem = get_index_dataset(problem, main_algorithm, median_algorithm_dir)
            algorithms = [main_algorithm] + other_algorithms
            for algorithm in algorithms:
                label = labels[algorithm]
                populations = data.query(
                    f"Algoritmo == '{algorithm}' and Problema == '{problem}' and Ejecución == {index_problem}")
                populations = populations.sort_values(by=['Evaluaciones'])
                populations.reset_index(drop=True, inplace=True)
                x_log_iterations = populations['Evaluaciones'].to_numpy()
                y_hv = populations['Valor']
                plt.plot(x_log_iterations, y_hv, label=label)
            plt.legend()
            output_file = 'medians-comparation/' + problem[13:] + '-' + indicator + '.pdf'
            plt.suptitle(problem.replace('.txt', ' ').replace('.vrp', '') + ' - ' + label_indicator, fontsize=14)
            plt.savefig(output_file)
            plt.close()

def plot_medians_iterations_log(problem_medians, labels):
    problems = problem_medians.keys()
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        for indicator in indicators:
            plt.xlabel('evaluations', fontsize=14)
            if indicator == 'hv':
                label_indicator = 'HV'
                plt.ylabel('HV', fontsize=14)
            elif indicator == 'es':
                label_indicator = '$E_s$'
                plt.ylabel('$E_s$', fontsize=14)
            elif indicator == 'r2':
                label_indicator = 'R2'
                plt.ylabel('R2', fontsize=14)
            data = pd.read_csv('evaluations-' + indicator + '.csv')
            for algorithm in problem_medians[problem][indicator]:
                label = labels[algorithm]
                dir = problem_medians[problem][indicator][algorithm]
                index_problem = get_index_dataset(problem, algorithm, dir)
                populations = data.query(
                    f"Algoritmo == '{algorithm}' and Problema == '{problem}' and Ejecución == {index_problem}")
                populations = populations.sort_values(by=['Evaluaciones'])
                populations.reset_index(drop=True, inplace=True)
                x_log_iterations = populations['Evaluaciones'].to_numpy()
                y_hv = populations['Valor']
                plt.plot(x_log_iterations, y_hv, label=label)
                #print(f'showing {problem} {algorithm} {index_problem}')
            plt.legend()
            output_file = 'medians-iterations/' + problem[13:] + '-' + indicator + '.pdf'
            plt.suptitle(problem.replace('.txt', ' ') + ' - ' + label_indicator, fontsize=14)
            plt.savefig(output_file)
            plt.close()



def plot_medians_log(problem_medians, labels):
    problems = problem_medians.keys()
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        for indicator in indicators:
            plt.xlabel('iterations', fontsize=14)
            plt.ylabel('HV', fontsize=14)
            for algorithm in problem_medians[problem][indicator]:
                label = labels[algorithm]
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



def plot_fronts(problem_medians, labels):
    problems = problem_medians.keys()
    indicators = ['hv', 'es', 'r2']
    for problem in problems:
        for indicator in indicators:
            for algorithm in problem_medians[problem][indicator]:
                dir = problem_medians[problem][indicator][algorithm]
                archive_dir = 'results/' + problem.replace('.txt', '').replace('.vrp', '') + '/' + algorithm + '/' + dir + '/archive.npy'
                archive = np.load(archive_dir)
                # 10, 7
                fig = plt.figure(figsize=(2.5, 1.725))
                ax = fig.add_subplot(projection='3d')
                ax.scatter(archive[:,0], archive[:,1], archive[:,2], s=2.8, marker='o', depthshade=False)
                ax.set_xlabel('$f_1$')
                ax.set_ylabel('$f_2$')
                ax.set_zlabel('$f_3$')
                ax.view_init(45,45)
                title = labels[algorithm]
                plt.title(title, {'fontsize':12})
                plt.savefig('fronts/'+ problem[13:] + '-' + indicator + '-' + algorithm + '.pdf', bbox_inches="tight", pad_inches=0.25)
                plt.close()

def count_ranks(data_mean, ranks):
    data_mean_sorted = data_mean.sort_values()
    pos = 0
    for algorithm, _ in data_mean_sorted.items():
        ranks[algorithm][pos] += 1
        pos += 1

def get_table_mean(problems, file, output_file, indicator, algorithms_to_compare, main_algorithm):
    algorithms = [main_algorithm] + algorithms_to_compare
    fig = plt.figure(figsize=(10, 10))
    wilcoxon_map = np.zeros((len(problems), len(algorithms)-1))
    algorithms_columns = {0: main_algorithm}
    for i, a in enumerate(algorithms_to_compare):
        algorithms_columns[i+1] = a
    df = pd.read_csv(file)
    data_total = pd.DataFrame(columns=algorithms)
    ranks = {}
    for algorithm in algorithms:
        ranks[algorithm] = [0] * len(algorithms)
    for i, problem in enumerate(problems):
        dataset = problem[0]
        data = pd.DataFrame()
        for algorithm in algorithms:
            populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Evaluaciones == 27000")
            populations.reset_index(drop=True, inplace=True)
            data = pd.concat([data, populations['Valor']], axis=1, ignore_index=True)
        data.rename(columns=algorithms_columns, inplace=True)
        data_mean = data[algorithms].mean(numeric_only=True)
        count_ranks(data_mean, ranks)
        data_std = data[algorithms].std(numeric_only=True)
        row_stats = {}
        for j, algorithm in enumerate(algorithms):
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
                    if indicator == 'HV':
                        if mean_alg > data_mean.loc[main_algorithm]:
                            arrow = " $\\uparrow$"
                        else:
                            arrow = " $\\downarrow$"
                    else:
                        if mean_alg < data_mean.loc[main_algorithm]:
                            arrow = " $\\uparrow$"
                        else:
                            arrow = " $\\downarrow$"
                wilcoxon_map[i][j] = p_value
            stats = str(f'{mean_alg:.3e}') + ' (' + str(f'{std_alg:.3e}') + ')' + arrow
            row_stats[algorithm] = stats
        data_total = data_total._append(row_stats, ignore_index=True)
    problems_column = [[p[0].replace('_', '$\_$').replace('.txt', ''), indicator] for p in problems]
    problems_column = pd.DataFrame(problems_column, columns=['Problem', 'Indicator'])
    data_total = pd.concat([problems_column, data_total], axis=1)
    data_total.to_latex(output_file, column_format='cc' + len(algorithms)*'r', index=False)
    sns.heatmap(wilcoxon_map, annot=True, cmap="coolwarm", vmax=0.05)
    plt.savefig('heatmap-' + indicator + '.png')
    plt.close()
    return ranks


def plot_general_table(problems, file, output_file, algorithms, labels):
    algorithms_columns = {}
    for i, a in enumerate(algorithms):
        algorithms_columns[i] = labels[a]
    title = 'Critic difference diagram '
    for algorithm in algorithms:
        algorithms_columns[algorithm] = labels[algorithm]
        #title += '- ' + labels[algorithm] + ' '
    df = pd.read_csv(file)
    data = pd.DataFrame()
    for problem in problems:
        dataset = problem[0]
        data_sub = pd.DataFrame()
        for algorithm in algorithms:
            populations = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Evaluaciones == 27000")
            populations.reset_index(drop=True, inplace=True)
            data_sub = pd.concat([data_sub, populations['Valor']], axis=1, ignore_index=True)
        data = pd.concat([data, data_sub], ignore_index=True)
    data.rename(columns=algorithms_columns, inplace=True)
    data.to_csv(output_file.replace('.pdf', '.csv'), index=False)
    if 'hv' not in file:
        data *= -1
    result = autorank(data, alpha=0.05, verbose=False, force_mode='nonparametric')
    fig, ax = plt.subplots(figsize=(15,25))
    ax = plot_stats(result, allow_insignificant=True)
    #ax.set_title(dataset + ' ' + indicator)
    fig.axes.append(ax)
    #fig.suptitle(dataset + ' ' + indicator)
    plt.title(title)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.15)
    plt.close()

def plot_general_diagram(algorithms, labels):
    algorithms_columns = {}
    title = 'Critic difference diagram '
    for algorithm in algorithms:
        algorithms_columns[algorithm] = labels[algorithm]
        title += '- ' + labels[algorithm] + ' '
    file_es = 'total-evaluations-es.csv'
    file_hv = 'total-evaluations-hv.csv'
    file_r2 = 'total-evaluations-r2.csv'
    file_times = 'total-evaluations-time.csv'
    df_es = pd.read_csv(file_es)
    df_hv = pd.read_csv(file_hv)
    df_r2 = pd.read_csv(file_r2)
    df_times = pd.read_csv(file_times)
    df_r2 *= -1
    df_es *= -1
    df_times *= -1
    total_df = pd.concat([df_es, df_hv, df_r2, df_times], axis=0)
    total_df.rename(columns=algorithms_columns, inplace=True)
    result = autorank(total_df, alpha=0.05, verbose=False, force_mode='nonparametric')
    fig, ax = plt.subplots(figsize=(15, 25))
    ax = plot_stats(result, allow_insignificant=True)
    # ax.set_title(dataset + ' ' + indicator)
    fig.axes.append(ax)
    #fig.suptitle('Critic difference diagram')
    #plt.suptitle('Critic difference diagram')
    plt.title(title)
    output_file = 'ranking-total.pdf'
    plt.savefig(output_file,  bbox_inches="tight", pad_inches=0.15)
    #plt.savefig(output_file,  bbox_inches="tight")
    plt.close()

def plot_ranks(ranks, output_file, title, labels):
    if output_file == 'ranks-hv.pdf' or output_file == 'lre-ranks-hv.pdf':
        for algorithm in ranks.keys():
            ranks[algorithm].reverse()
    k = list(ranks.keys())
    for algorithm in k:
        ranks[labels[algorithm]] = ranks.pop(algorithm)
    index = ['first place', 'second place', 'third place', 'fourth place', 'fifth place', 'sixth place', 'seventh place']
    index = index[:len(k)]
    df = pd.DataFrame(ranks, index=index)
    df.plot(figsize=(30, 10))
    ax = df.plot.bar(rot=0, figsize=(30, 5), fontsize=18)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
    plt.title(title.replace('.txt', ''), fontsize=18)
    plt.savefig(output_file)
    plt.close()

def delete_by_evaluation(files, execution):
    for file in files:
        df = pd.read_csv(file)
        print(f'before drop {df.shape} {file}')
        index_r = df.query(f"Evaluaciones == {execution}").index
        df.drop(index_r, inplace=True)
        print(f'after drop  {df.shape} {file}')
        df.to_csv(file, index=False)

def delete_alg_dt(file, dataset, algorithm, execution):
    df = pd.read_csv(file)
    print(f'before drop {df.shape} {file}')
    index_r = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Ejecución == {execution}").index
    df.drop(index_r, inplace=True)
    print(f'after drop  {df.shape} {file}')
    df.to_csv(file, index=False)

def check_algorithms_execution_base():
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
    algorithms = ['cmibaco', 'cmibaco-lns', 'cmibaco-mut', 'cmibaco-cross',
                  'ibaco-hv', 'ibaco-r2', 'ibaco-eps', 'ibaco-ws',
                  'ibaco-hv-lns', 'ibaco-r2-lns', 'ibaco-eps-lns', 'ibaco-ws-lns']
    files = ['evaluations-hv.csv', 'evaluations-es.csv', 'evaluations-r2.csv']
    for file in files:
        df = pd.read_csv(file)
        for dataset in problems:
            for algorithm in algorithms:
                index_r = df.query(f"Algoritmo == '{algorithm}' and Problema == '{dataset}' and Evaluaciones == 0").index
                if index_r.size == 0:
                    print (f'no dataset {dataset} for {algorithm} - {file}')
                elif index_r.size != 20:
                    raise ('error, not enought origin')

def get_multiple_logs_hyp(dirs, dataset, algorithm, output_file):
    path = 'results/' + dataset + '/' + algorithm + '/'
    pairs = []
    for d in dirs:
        f = open(path + d + '/statistics.json')
        p = json.load(f)
        log = p['log_hypervolumen']
        pairs.append((log, d))
    plt.figure(figsize=(25, 7))
    plt.title(dataset)
    y_hv = [i for i in range(0, 27000) if i % 270 == 0]
    for p in pairs:
        plt.plot(y_hv, p[0], label=p[1])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_file)
    plt.close()

def get_multiple_logs_hyp_dir(dataset, algorithm, output):
    dir = 'results/' + dataset + '/' + algorithm + '/'
    dirs = os.listdir(dir)
    dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
    dirs = [(d, datetime.strptime(d, '%Y-%m-%d-%H-%M-%S')) for d in dirs]
    dirs.sort(key=lambda x: x[1])
    dirs = [d[0] for d in dirs]
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
    plt.savefig(output)
