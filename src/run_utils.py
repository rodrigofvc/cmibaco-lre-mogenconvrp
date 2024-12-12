from utils import *
import pandas as pd

def eval_archive(problem, algorithm, dir):
    archive_dir = 'results/' + problem.replace('.txt', '').replace('.vrp', '') + '/' + algorithm + '/' + dir + '/archive-object.xz'
    file = lzma.open(archive_dir, 'rb')
    archive = pickle.load(file)
    archive, _ = non_dominated([], archive)
    archive = [[a.f_1, a.f_2, a.f_3] for a in archive]
    archive = np.array(archive)
    ref_point_hv = get_reference_point_file(problem)
    ind = HV(ref_point=ref_point_hv)
    w_r2 = get_reference_directions("energy", 3, 90, seed=42)
    z_r2 = np.array([0, 0, 0])
    hv_a = ind(archive)
    r2_a = r2(archive, w_r2, z_r2)
    es_a = indicator_s_energy(archive, s=2)
    return hv_a, r2_a, es_a


def get_table_non_dominated(problems, algorithms):
    columns_csv = ['Algoritmo','Problema','Ejecución','Evaluaciones','Indicador','Valor']
    for j, p in enumerate(problems):
        problem = p[0]
        for algorithm in algorithms:
            dir = 'results/' + problem.replace('.txt', '').replace('.vrp', '') + '/' + algorithm + '/'
            dirs = os.listdir(dir)
            dirs = [d for d in dirs if d.startswith('2023') or d.startswith('2024')]
            dirs = [(d, datetime.strptime(d, '%Y-%m-%d-%H-%M-%S')) for d in dirs]
            dirs.sort(key=lambda x: x[1])
            dirs = [d[0] for d in dirs]
            if len(dirs) != 20:
                raise('no enought dirs')
            for j, d in enumerate(dirs):
                print(f'>>>>>> evaluating non-dominated solutions {problem}/{algorithm}/{d}')
                hypervolume, r2, es = eval_archive(problem, algorithm, d)

                row_hv = [algorithm, problem, j, '27000', 'HV', hypervolume]
                df_hv = pd.DataFrame(np.array([row_hv]), columns=columns_csv)
                df_hv.to_csv('non-evaluations-hv.csv', mode='a', index=False, header=False)
                df_hv = pd.DataFrame(np.array([[algorithm, problem + '-' + str(j), hypervolume]]))
                df_hv.to_csv('critic-hv.csv', mode='a', index=False, header=False)

                row_r2 = [algorithm, problem, j, '27000', 'R2', r2]
                df_r2 = pd.DataFrame(np.array([row_r2]), columns=columns_csv)
                df_r2.to_csv('non-evaluations-r2.csv', mode='a', index=False, header=False)
                df_r2 = pd.DataFrame(np.array([[algorithm, problem + '-' + str(j), r2]]))
                df_r2.to_csv('critic-r2.csv', mode='a', index=False, header=False)

                row_es = [algorithm, problem, j, '27000', 'Es', es]
                df_es = pd.DataFrame(np.array([row_es]), columns=columns_csv)
                df_es.to_csv('non-evaluations-es.csv', mode='a', index=False, header=False)
                df_es = pd.DataFrame(np.array([[algorithm, problem + '-' + str(j), es]]))
                df_es.to_csv('critic-es.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    algorithms_to_compare = ['ibaco-eps-lns', 'ibaco-hv-lns', 'ibaco-r2-lns', 'ibaco-ws-lns']
    main_algorithm = 'cmibaco-lns'

    algorithms = [main_algorithm] + algorithms_to_compare

    labels_algorithms = {'cmibaco': 'cMIBACO$_{hybrid}$', 'ibaco-hv-lns': 'IBACO$_{HV}$', 'ibaco-r2-lns': 'IBACO$_{R2}$',
                         'ibaco-eps-lns': 'IBACO$_{\epsilon^+}$', 'ibaco-ws-lns': 'IBACO$_{ws}$',
                         'cmibaco-lns': 'cMIBACO', 'cmibaco-cross': 'cMIBACO$_{crossover}$',
                         'cmibaco-mut': 'cMIBACO$_{mutation}$', 'cmibaco-base': 'cMIBACO$_{base}$'}


    problems = [('Christofides_1_5_0.5.txt', 'ch1505-'),
                ('Christofides_2_5_0.5.txt', 'ch2505-'),
                ('Christofides_3_5_0.5.txt', 'ch3505-'),
                ('Christofides_4_5_0.5.txt', 'ch4505-'),
                ('Christofides_5_5_0.5.txt', 'ch5505-'),
                ('Christofides_6_5_0.5.txt', 'ch6505-'),
                ('Christofides_7_5_0.5.txt', 'ch7505-'),
                ('Christofides_8_5_0.5.txt', 'ch8505-'),
                ('Christofides_9_5_0.5.txt', 'ch9505-'),
                ('Christofides_10_5_0.5.txt', 'ch10505-'),
                ('Christofides_11_5_0.5.txt', 'ch11505-'),
                ('Christofides_12_5_0.5.txt', 'ch12505-'),
                ('Christofides_1_5_0.7.vrp', 'ch1507-'),
                ('Christofides_2_5_0.7.vrp', 'ch2507-'),
                ('Christofides_3_5_0.7.vrp', 'ch3507-'),
                ('Christofides_4_5_0.7.vrp', 'ch4507-'),
                ('Christofides_5_5_0.7.vrp', 'ch5507-'),
                ('Christofides_6_5_0.7.vrp', 'ch6507-'),
                ('Christofides_7_5_0.7.vrp', 'ch7507-'),
                ('Christofides_8_5_0.7.vrp', 'ch8507-'),
                ('Christofides_9_5_0.7.vrp', 'ch9507-'),
                ('Christofides_10_5_0.7.vrp', 'ch10507-'),
                ('Christofides_11_5_0.7.vrp', 'ch11507-'),
                ('Christofides_12_5_0.7.vrp', 'ch12507-'),
                ('Christofides_1_5_0.9.txt', 'ch1509-'),
                ('Christofides_2_5_0.9.txt', 'ch2509-'),
                ('Christofides_3_5_0.9.txt', 'ch3509-'),
                ('Christofides_4_5_0.9.txt', 'ch4509-'),
                ('Christofides_5_5_0.9.txt', 'ch5509-'),
                ('Christofides_6_5_0.9.txt', 'ch6509-'),
                ('Christofides_7_5_0.9.txt', 'ch7509-'),
                ('Christofides_8_5_0.9.txt', 'ch8509-'),
                ('Christofides_9_5_0.9.txt', 'ch9509-'),
                ('Christofides_10_5_0.9.txt', 'ch10509-'),
                ('Christofides_11_5_0.9.txt', 'ch11509-'),
                ('Christofides_12_5_0.9.txt', 'ch12509-'),
                ]


    #problems_medians = get_medians_files([p[0] for p in problems], algorithms)
    #plot_median_algorithm_compare(problems_medians, main_algorithm, algorithms_to_compare, labels_algorithms)
    #plot_medians_iterations_log(problems_medians, labels_algorithms)
    #plot_fronts(problems_medians, labels_algorithms)

    #get_table_non_dominated(problems, algorithms)
    #get_table_time(problems, 'table-times.tex', main_algorithm, algorithms_to_compare, labels_algorithms)

    # Tabla de promedio y desv est por indicador
    ranks_hv = get_table_mean(problems, 'approx-non-evaluations-hv.csv', 'non-table-hv-n.tex', 'HV',
                              algorithms_to_compare=algorithms_to_compare, main_algorithm=main_algorithm)
    ranks_r2 = get_table_mean(problems, 'approx-non-evaluations-r2.csv', 'non-table-r2-n.tex', 'R2',
                              algorithms_to_compare=algorithms_to_compare, main_algorithm=main_algorithm)
    ranks_es = get_table_mean(problems, 'approx-non-evaluations-es.csv', 'non-table-es-n.tex', '$E_s$',
                              algorithms_to_compare=algorithms_to_compare, main_algorithm=main_algorithm)
    # get_table_time(problems, 'table-times.tex')

    # Grafica las puntuaciones de cada algoritmo
    plot_ranks(ranks_hv, 'ranks-hv.pdf', 'Performance rating indicator $HV$', labels_algorithms)
    plot_ranks(ranks_r2, 'ranks-r2.pdf', 'Performance rating indicator $R2$', labels_algorithms)
    plot_ranks(ranks_es, 'ranks-es.pdf', 'Performance rating indicator $E_s$', labels_algorithms)

    # Diagrama critico de cada indicador
    plot_general_table(problems, 'approx-non-evaluations-hv.csv', 'total-evaluations-hv.pdf', algorithms=algorithms,
                       labels=labels_algorithms)
    plot_general_table(problems, 'approx-non-evaluations-r2.csv', 'total-evaluations-r2.pdf', algorithms=algorithms,
                       labels=labels_algorithms)
    plot_general_table(problems, 'approx-non-evaluations-es.csv', 'total-evaluations-es.pdf', algorithms=algorithms,
                       labels=labels_algorithms)
    # Diagrama critico general
    plot_general_diagram(algorithms, labels_algorithms)

    # boxplots
    indicators = ['hv', 'es', 'r2']
    labels_indicators = {'hv': 'HV', 'es': '$E_s$', 'r2': 'R2'}
    for problem in problems:
        dataset = problem[0]
        # base = problem.replace('Christofides_', '').replace('.txt', '').replace('.vrp', '')
        base = problem[1]
        for ind in indicators:
            file = 'evaluations-' + ind + '.csv'
            output_file_box = 'boxplot/' + base + ind + '-box.pdf'
            title_file = dataset + ' - ' + labels_indicators[ind]
            boxplot(file, dataset, output_file_box, title_file, [main_algorithm] + algorithms_to_compare,
                    labels_algorithms)