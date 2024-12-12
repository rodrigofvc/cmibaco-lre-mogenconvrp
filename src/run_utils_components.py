from utils import *

if __name__ == '__main__':

    algorithms_to_compare = ['cmibaco-lns', 'cmibaco', 'cmibaco-cross', 'cmibaco-mut']
    main_algorithm = 'cmibaco-base'

    algorithms = [main_algorithm] + algorithms_to_compare

    labels_algorithms = {'cmibaco': 'cMIBACO$_{hybrid}$', 'ibaco-hv-lns': 'IBACO$_{HV}$', 'ibaco-r2-lns': 'IBACO$_{R2}$',
                         'ibaco-eps-lns': 'IBACO$_{\epsilon^+}$', 'ibaco-ws-lns': 'IBACO$_{ws}$',
                         'cmibaco-lns': 'cMIBACO$_{lns}$', 'cmibaco-cross': 'cMIBACO$_{crossover}$',
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
                ('Christofides_12_5_0.5.txt', 'ch12505-')
                ]

    # plot medians for each algorithm
    #problems_medians = get_medians_files([p[0] for p in problems], algorithms)
    #plot_median_algorithm_compare(problems_medians, main_algorithm, algorithms_to_compare, labels_algorithms)
    #plot_medians_iterations_log(problems_medians, labels_algorithms)
    #plot_fronts(problems_medians, labels_algorithms)

    indicators = ['hv', 'es', 'r2']
    # Tabla de promedio y desv.et por indicador
    ranks_hv = get_table_mean(problems, 'evaluations-hv.csv', 'table-hv-n.tex', 'HV', algorithms_to_compare=algorithms_to_compare, main_algorithm=main_algorithm)
    ranks_r2 = get_table_mean(problems, 'evaluations-r2.csv', 'table-r2-n.tex', 'R2', algorithms_to_compare=algorithms_to_compare, main_algorithm=main_algorithm)
    ranks_es = get_table_mean(problems, 'evaluations-es.csv', 'table-es-n.tex', '$E_s$', algorithms_to_compare=algorithms_to_compare, main_algorithm=main_algorithm)

    get_table_time(problems, 'table-times.tex', main_algorithm, algorithms_to_compare, labels_algorithms)

    # Grafica las puntuaciones de cada algoritmo
    plot_ranks(ranks_hv, 'ranks-hv.pdf', 'Performance rating indicator $HV$', labels_algorithms)
    plot_ranks(ranks_r2, 'ranks-r2.pdf', 'Performance rating indicator $R2$', labels_algorithms)
    plot_ranks(ranks_es, 'ranks-es.pdf', 'Performance rating indicator $E_s$', labels_algorithms)

    # Diagrama critico de cada indicador
    plot_general_table(problems, 'evaluations-hv.csv', 'total-evaluations-hv.pdf', algorithms=algorithms, labels=labels_algorithms)
    plot_general_table(problems, 'evaluations-r2.csv', 'total-evaluations-r2.pdf', algorithms=algorithms, labels=labels_algorithms)
    plot_general_table(problems, 'evaluations-es.csv', 'total-evaluations-es.pdf', algorithms=algorithms, labels=labels_algorithms)
    # Diagrama critico general
    plot_general_diagram(algorithms, labels_algorithms)

    
    # boxplots
    indicators = ['hv', 'es', 'r2']
    labels_indicators = {'hv': 'HV', 'es': '$E_s$', 'r2': 'R2'}
    for problem in problems:
        dataset = problem[0]
        base = problem[1]
        for ind in indicators:
            file = 'evaluations-' + ind + '.csv'
            output_file = base + ind +'.pdf'
            output_file_box = 'boxplot/' + base + ind +'-box.pdf'
            title_file = dataset + ' - ' + labels_indicators[ind]
            boxplot(file, dataset, output_file_box, title_file, [main_algorithm] + algorithms_to_compare, labels_algorithms)
