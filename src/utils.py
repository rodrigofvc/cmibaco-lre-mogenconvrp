import os


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



def get_statistics(A, duration):
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
    statistics['n_solutions'] = len(A)
    statistics['duration'] = duration
    return statistics