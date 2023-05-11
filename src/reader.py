from costumer import Costumer


def read_dataset(dataset_dir):
    f = open(dataset_dir, 'r')
    lines = f.readlines()
    f.close()
    #print (lines)

    NUM_DAYS = 4
    CAPACITY = 5
    NODE_COORD_SECTION = 10

    days = lines[NUM_DAYS]
    days = days.split()
    days = int(days[-1])

    capacity = lines[CAPACITY]
    capacity = capacity.split()
    capacity = int(capacity[-1])

    DEMAND_SECTION = [k for k, line in enumerate(lines) if 'DEMAND_SECTION' in line]
    DEMAND_SECTION = DEMAND_SECTION[0]

    SVC_TIME_SECTION = [k for k, line in enumerate(lines) if 'SVC_TIME_SECTION' in line]
    SVC_TIME_SECTION = SVC_TIME_SECTION[0]

    DEPOT_SECTION = [k for k, line in enumerate(lines) if 'DEPOT_SECTION' in line]
    DEPOT_SECTION = DEPOT_SECTION[0]

    costumers = get_nodes(lines, NODE_COORD_SECTION, DEMAND_SECTION)
    set_demand(lines, costumers, DEMAND_SECTION, SVC_TIME_SECTION)
    set_time_section(lines, costumers, SVC_TIME_SECTION, DEPOT_SECTION)
    return costumers, capacity, days


def get_nodes(lines, NODE_COORD_SECTION, DEMAND_SECTION):
    costumers = []
    for line in lines[NODE_COORD_SECTION+1:DEMAND_SECTION]:
        chunks = line.split()
        new_costumer = Costumer(int(chunks[0]), float(chunks[1]), float(chunks[2]))
        costumers.append(new_costumer)
    return costumers

def set_demand(lines, costumers, DEMAND_SECTION, SVC_TIME_SECTION):
    for line in lines[DEMAND_SECTION+1:SVC_TIME_SECTION]:
        chunks = line.split()
        id = int(chunks[0])
        demands = list(map(lambda x : int(x), chunks[1:]))
        costumers[id-1].demands = demands

def set_time_section(lines, costumers, SVC_TIME_SECTION, DEPOT_SECTION):
    for line in lines[SVC_TIME_SECTION+1:DEPOT_SECTION]:
        chunks = line.split()
        id = int(chunks[0])
        times = list(map(lambda x : float(x), chunks[1:]))
        costumers[id-1].service_times = times
        costumers[id-1].arrival_times = [-1] * len(times)
        costumers[id-1].vehicles_visit = [-1] * len(times)
