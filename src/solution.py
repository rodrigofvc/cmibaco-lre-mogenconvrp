import threading

import numpy as np
from ant import Ant
import copy
from threading import Event
import queue

class Solution:
    counter_id = 0
    # multi-threading case
    evaluations = queue.Queue(maxsize=1)
    evaluations.put(0)
    evals = 0
    lock_evaluations = threading.Lock()
    write_log = Event()

    def __init__(self, timetables, days):
        Solution.lock_evaluations.acquire()
        Solution.counter_id += 1
        Solution.lock_evaluations.release()
        self.id = Solution.counter_id
        self.algorithm = None
        self.days = days
        self.timetables = timetables
        self.assigments_vehicles = None
        self.assigments_costumers = None
        self.depot = None
        self.ants = {'AM': [], 'PM': []}
        self.f_1 = None
        self.f_2 = None
        self.f_3 = None

    def add_assigment_vehicles(self, vehicles, costumers):
        self.assigments_vehicles = vehicles
        self.assigments_costumers = costumers

    def get_vector_representation_dt(self, timetable, day):
        tmp = []
        for vehicle in self.assigments_vehicles:
            if not day in vehicle.tour[timetable].keys():
                continue
            tour_vehicle = vehicle.tour[timetable][day]
            tmp += tour_vehicle
        return tmp

    def print_vector(self, vec):
        s = '['
        for c in vec:
            s += str(c.id) + ', '
        s += ']'
        print (s)

    def shift(self, vector, point_1, point_2):
        current = vector[point_1]
        for i in range(point_1, point_2):
            if i + 1 != len(vector):
                next = vector[i+1]
                vector[i+1] = current
                current = next
        vector[point_1] = next

    def add_ant_timetable_day(self, ant, timetable):
        self.ants[timetable].append(ant)

    def get_assigments_crossover(self, even, min_pheromone, max_pheromone, other):
        vehicles_timetable = copy.deepcopy(self.assigments_vehicles)
        costumers_timetable = copy.deepcopy(self.assigments_costumers)
        crossover__ants = {'AM': [], 'PM': []}
        n = 1 + len(self.assigments_costumers)
        for c in costumers_timetable:
            c.arrival_times = [-1] * self.days
            c.vehicles_visit = [-1] * self.days
        for v in vehicles_timetable:
            v.tour = {'AM': {}, 'PM': {}}
            v.loads = {'AM': [0] * self.days, 'PM': [0] * self.days}
            v.times_tour = {'AM': [0] * self.days, 'PM': [0] * self.days}
        depot = self.assigments_vehicles[0].tour['AM'][0][0]
        if depot.id != 0:
            raise()
        for timetable in self.timetables:
            for day in range(self.days):
                if even:
                    if day % 2 == 0:
                        vehicles_day = [v for v in self.assigments_vehicles if day in v.tour[timetable].keys()]
                    else:
                        vehicles_day = [v for v in other.assigments_vehicles if day in v.tour[timetable].keys()]
                else:
                    if day % 2 == 0:
                        vehicles_day = [v for v in other.assigments_vehicles if day in v.tour[timetable].keys()]
                    else:
                        vehicles_day = [v for v in self.assigments_vehicles if day in v.tour[timetable].keys()]

                ant = Ant(depot, n, min_pheromone, max_pheromone)
                for v in vehicles_day:
                    v_new_i = vehicles_timetable.index(v)
                    v_new = vehicles_timetable[v_new_i]
                    depot = v.tour[timetable][day][0]
                    v_new.set_tour_day(timetable, day, [depot])
                    i = 0
                    j = 0
                    for c in v.tour[timetable][day]:
                        if c.id == 0:
                            continue
                        c_new_i = costumers_timetable.index(c)
                        c_new = costumers_timetable[c_new_i]
                        j = c_new.id
                        ant.global_update[i][j] = 1
                        ant.global_update[j][i] = 1
                        i = c_new.id
                        v_new.add_costumer_tour_day(timetable, day, c_new)
                    ant.global_update[i][0] = 1
                    ant.global_update[0][i] = 1
                    v_new.return_depot(timetable, day)
                crossover__ants[timetable].append(ant)

        return vehicles_timetable, costumers_timetable, crossover__ants

    def crossover(self, other, min_pheromone=10e-3, max_pheromone=10e5, prob_cross=.80):
        if self.days != other.days:
            print (f'one solution days {self.days}')
            print (f'other solution days {other.days}')
            raise('solutions not share days')
        childs = []
        prob = np.random.rand()
        if prob > prob_cross:
            return childs

        vehicles_child_1, costumers_child_1, ants_child_1 = self.get_assigments_crossover(True, min_pheromone, max_pheromone, other)
        vehicles_child_2, costumers_child_2, ants_child_2 = self.get_assigments_crossover(False, min_pheromone, max_pheromone, other)

        solution_1 = Solution(self.timetables, self.days)
        solution_1.add_assigment_vehicles(vehicles_child_1, costumers_child_1)
        solution_1.ants = ants_child_1
        solution_1.depot = self.depot

        solution_2 = Solution(self.timetables, self.days)
        solution_2.add_assigment_vehicles(vehicles_child_2, costumers_child_2)
        solution_2.ants = ants_child_2
        solution_2.depot = self.depot

        solution_1.get_fitness()
        solution_2.get_fitness()
        childs = [solution_1, solution_2]
        return childs

    def mutation(self, prob_mut):
        self.mutation_shift(prob_mut)
        self.get_fitness()

    def mutation_shift(self, prob_m):
        for timetable in self.timetables:
            for day in range(self.days):
                prob = np.random.rand()
                if prob <= prob_m:
                    vector_rep = self.get_vector_representation_dt(timetable, day)
                    point_1 = 0
                    point_2 = 0
                    while point_1 == point_2:
                        point_1 = np.random.choice([i for i in range(1,len(vector_rep)//2) if vector_rep[i].id != 0])
                        point_2 = np.random.choice([i for i in range(point_1+1,len(vector_rep)) if vector_rep[i].id != 0])
                        #print (f'!!!! MUTATE {timetable} - {day} {point_1} {point_2} {[c.id for c in vector_rep]}')
                    self.shift(vector_rep, point_1, point_2)
                    #print (f'shift {[c.id for c in vector_rep]}')
                    for c in self.assigments_costumers:
                        if c.timetable == 0 and timetable == 'AM':
                            c.arrival_times[day]  = -1
                            c.vehicles_visit[day] = -1
                        elif c.timetable == 1 and timetable == 'PM':
                            c.arrival_times[day]  = -1
                            c.vehicles_visit[day] = -1

                    self.ants[timetable][day].global_update = np.zeros(self.ants[timetable][day].global_update.shape)

                    depot = vector_rep[0]
                    i = 0
                    vector_rep = [c for c in vector_rep if c.id != 0]
                    if vector_rep[0].timetable == 0:
                        limit_time = self.assigments_vehicles[0].limit_time/2
                    else:
                        limit_time = self.assigments_vehicles[0].limit_time
                    for i_vehicle, vehicle in enumerate(self.assigments_vehicles):
                        vehicle.times_tour[timetable][day] = 0
                        vehicle.loads[timetable][day] = 0
                        last_vehicle = i_vehicle
                        tour = [depot]
                        vehicle.set_tour_day(timetable, day, tour)
                        current_cos = vector_rep[i]
                        current_time = vehicle.add_costumer_tour_day(timetable, day, current_cos)
                        self.ants[timetable][day].global_update[0][vehicle.tour[timetable][day][-1].id] = 1
                        self.ants[timetable][day].global_update[vehicle.tour[timetable][day][-1].id][0] = 1
                        i += 1
                        if i == len(vector_rep):
                            vehicle.return_depot(timetable, day)
                            break
                        current_cos = vector_rep[i]
                        while current_time + vehicle.tour[timetable][day][-1].distance_to(current_cos) + current_cos.service_times[day] + current_cos.distance_to(vehicle.tour[timetable][day][0]) <= limit_time and vehicle.get_total_load_day(day) + current_cos.demands[day] <= vehicle.capacity:
                            #print (f'adding {current_cos.id} / {vehicle.loads[day]} {vehicle.capacity} - {current_cos.demands[day]}')
                            before_costumer = vehicle.tour[timetable][day][-1]
                            self.ants[timetable][day].global_update[before_costumer.id][current_cos.id] = 1
                            self.ants[timetable][day].global_update[current_cos.id][before_costumer.id] = 1
                            current_time = vehicle.add_costumer_tour_day(timetable, day, current_cos)
                            i += 1
                            if i == len(vector_rep):
                                break
                            current_cos = vector_rep[i]
                        self.ants[timetable][day].global_update[0][vehicle.tour[timetable][day][-1].id] = 1
                        self.ants[timetable][day].global_update[vehicle.tour[timetable][day][-1].id][0] = 1
                        vehicle.return_depot(timetable, day)
                        if len(vehicle.tour[timetable][day]) == 1:
                            print (f'{[c.id for c in vehicle.tour[timetable][day]]}')
                            raise('vehicle with only 1 costumer not allowed')
                        if i >= len(vector_rep):
                            break

                    # borra la planeacion de los vehiculos que ya no se usaron
                    vehicles_excluded = [v for i, v in enumerate(self.assigments_vehicles) if i >= last_vehicle + 1 and day in v.tour[timetable].keys()]
                    for v in vehicles_excluded:
                        v.times_tour[timetable][day] = 0
                        v.loads[timetable][day] = 0
                        v.tour[timetable].pop(day)
                    for c in self.assigments_costumers:
                        if c.vehicles_visit[day] == -1 and c.demands[day] > 0 and timetable == 'AM' and c.timetable == 0:
                            print (f'UNVISITED')
                            print (c)
                            raise()
                        if c.vehicles_visit[day] == -1 and c.demands[day] > 0 and timetable == 'PM' and c.timetable == 1:
                            print (f'UNVISITED')
                            print (c)
                            raise()


    def get_total_time(self):
        total_time = 0
        for vehicle in self.assigments_vehicles:
            for timetable in self.timetables:
                for d in range(self.days):
                    total_time += vehicle.get_time(timetable, d)
        return total_time

    def get_max_difference_arrive(self):
        max_diff = 0
        for costumer in self.assigments_costumers:
            max_diff = max(max_diff, costumer.get_max_arrival_diference())
        return max_diff

    def max_costumer(self):
        c = None
        max_diff = 0
        for costumer in self.assigments_costumers:
            m = costumer.get_max_arrival_diference()
            if m >= max_diff:
                c = costumer
                max_diff = m
        return c

    def get_mean_arrive_difference(self):
        diff = 0
        n = 0
        for timetable in self.timetables:
            n += len(self.assigments_costumers[timetable])
            for costumer in self.assigments_costumers[timetable]:
                diff += costumer.get_max_arrival_diference()
        return diff / n

    def get_max_difference_drivers(self):
        max_driver_diff = 0
        for costumer in self.assigments_costumers:
            max_driver_diff = max(max_driver_diff, costumer.get_max_vehicle_difference())
        return max_driver_diff

    def get_fitness(self):
        self.f_1 = self.get_total_time()
        self.f_2 = self.get_max_difference_drivers()
        self.f_3 = self.get_max_difference_arrive()
        Solution.increase_counter_evals(1)
        return (self.f_1, self.f_2, self.f_3)

    def increase_counter_evals(evals):
        c = Solution.evaluations.get()
        c += evals
        Solution.evaluations.put(c)
        Solution.write_log.set()
        Solution.evals += evals

    def dominates(self, y):
        # F(X) == F(Y)
        if abs(self.f_1 - y.f_1) <= 10e-8 and abs(self.f_3 - y.f_3) <= 10e-8 and self.f_2 == y.f_2:
            return False
        # F(X) <= F(Y)
        if self.f_1 <= y.f_1 and self.f_2 <= y.f_2 and self.f_3 <= y.f_3:
            return True
        return False

    def epsilon_dominates(self, y, epsilon):
        # F(X) == F(y)
        if abs(self.f_1 - y.f_1) <= 10e-8 and abs(self.f_3 - y.f_3) <= 10e-8 and self.f_2 == y.f_2:
            return False
        # F(X) + e == F(Y)
        if abs((self.f_1 + epsilon[0]) - y.f_1) <= 10e-8 and abs((self.f_3 + epsilon[2]) - y.f_3) <= 10e-8 and (self.f_2 + epsilon[1]) == y.f_2:
            return False
        # F(X) + e <= F(Y)
        if self.f_1 + epsilon[0] <= y.f_1 and self.f_3 + epsilon[2] <= y.f_3 and self.f_2 + epsilon[1] <= y.f_2:
            return True
        return False


    def is_feasible(self):
        vehicles = self.assigments_vehicles
        for vehicle in vehicles:
            vehicle.is_feasible()
        customers = self.assigments_costumers
        for customer in customers:
            customer.is_feasible()
        for c in customers:
            if c.timetable == 0:
                t = 'AM'
            else:
                t = 'PM'
            if c.id == 0:
                continue
            days_service = [i for i, s in enumerate(c.demands) if s >= 0]
            for day in days_service:
                vehicles_visit_day = [v for v in vehicles if day in v.tour[t].keys() and c in v.tour[t][day]]
                if len(vehicles_visit_day) == 0:
                    print (c)
                    raise('costumer not visited')
                if len(vehicles_visit_day) > 1:
                    raise('costumer visited twice')
        for h in self.timetables:
            for d in range(self.days):
                ant_d_h = self.ants[h][d].global_update
                v_tours = [v for v in vehicles if d in v.tour[h].keys()]
                traces = 0
                for v in v_tours:
                    tour = [c.id for c in v.tour[h][d]]
                    tour_len = len(tour)
                    for i in range(tour_len):
                        c1 = tour[i]
                        j = i+1
                        if j != tour_len:
                            c2 = tour[j]
                            traces += 2
                            if ant_d_h[c1][c2] != 1 and ant_d_h[c2][c1] != 1:
                                print ('TOURRR _-------')
                                print (tour)
                                print(self)
                                raise()
                    if tour_len > 2:
                        traces += 2
                    if ant_d_h[0][tour[-1]] != 1 or ant_d_h[tour[-1]][0] != 1:
                        print (tour, ant_d_h[0][tour[-1]], ant_d_h[0][tour[-1]])
                        raise()
                ones = ant_d_h[ant_d_h == 1]
                if len(ones) != traces:
                    print (len(ones), traces)
                    raise()

        return True

    def __eq__(self, other):
        if isinstance(other, Solution):
            for timetable in self.timetables:
                for day in range(self.days):
                    vector = self.get_vector_representation_dt(timetable, day)
                    other_vector = other.get_vector_representation_dt(timetable, day)

                    vector = [str(c.id) for c in vector]
                    other_vector = [str(c.id) for c in other_vector]
                    """
                    if vector != other_vector:
                        return False
                    """
                    vector_str = ''.join(vector)
                    other_vector_str = ''.join(other_vector)

                    chunks_vector = vector_str.split('0')
                    chunks_other_vector = other_vector_str.split('0')

                    if len(chunks_vector) != len(chunks_other_vector):
                        return False

                    chunks_vector = chunks_vector[1:]
                    chunks_other_vector = chunks_other_vector[1:]

                    for chunk in chunks_vector:
                        if len(chunk) == 0:
                            continue
                        if not chunk in other_vector_str:
                            return False

            return True
        return False

    # For LNS
    def remove_costumer(self, costumer_rmv):
        vehicles = [v for v in self.assigments_vehicles if v.contains_costumer(costumer_rmv)]
        for v in vehicles:
            v.remove_costumer(costumer_rmv)

    # For LNS
    def get_cheapest_time_day_add(self, day, costumer):
        timetable = costumer.timetable
        if timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        cheapest_time = float('inf')
        cheapest_vehicle = None
        cheapest_position = -1
        for v in self.assigments_vehicles:
            if day in v.tour[timetable].keys() and costumer.demands[day] + v.loads[timetable][day] <= v.capacity:
                position, time_visit_costumer = v.can_visit_costumer_greedy(timetable, day, costumer)
                if position != -1 and time_visit_costumer < cheapest_time:
                    cheapest_time = time_visit_costumer
                    cheapest_vehicle = v
                    cheapest_position = position
        return cheapest_time, cheapest_vehicle, cheapest_position
    # For LNS
    def add_costumer_new_vehicle(self, day, costumer):
        timetable = costumer.timetable
        if timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        for v in self.assigments_vehicles:
            if v.loads[timetable][day] == 0:
                v.set_tour_day(timetable, day, [self.depot])
                v.add_costumer_tour_day(timetable, day, costumer)
                v.return_depot(timetable, day)
                break

    # For LNS
    def get_tour_costumer_day(self, j, day):
        if j.timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        vehicles_day = [v for v in self.assigments_vehicles if day in v.tour[timetable].keys()]
        for v in vehicles_day:
            if j in v.tour[timetable][day]:
                return v.tour[timetable][day]

    def can_push_front(self, day, costumer, pf):
        timetable = 'AM' if costumer.timetable == 0 else 'PM'
        vehicle = [v for v in self.assigments_vehicles if day in v.tour[timetable].keys() and costumer in v.tour[timetable][day]]
        vehicle = vehicle[0]
        return vehicle.can_push_front(timetable, day, pf)

    def can_push_back(self, day, costumer, pb):
        timetable = 'AM' if costumer.timetable == 0 else 'PM'
        vehicle = [v for v in self.assigments_vehicles if day in v.tour[timetable].keys() and costumer in v.tour[timetable][day]]
        vehicle = vehicle[0]
        return vehicle.can_push_back(timetable, day, pb)


    # For LNS
    def apply_pf(self, day, costumer, max_pf):
        if costumer.timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        vehicle = [v for v in self.assigments_vehicles if day in v.tour[timetable].keys() and costumer in v.tour[timetable][day]]
        vehicle = vehicle[0]
        vehicle.push_tour(timetable, day, max_pf)
        self.get_fitness()

    # For LNS
    def apply_pb(self, day, max_costumer, max_pb):
        if max_costumer.timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        vehicle = [v for v in self.assigments_vehicles if day in v.tour[timetable].keys() and max_costumer in v.tour[timetable][day]]
        vehicle = vehicle[0]
        vehicle.push_tour(timetable, day, max_pb)
        self.get_fitness()

    # For LNS
    def apply_2_opt(self, costumer, day):
        vehicle = [v for v in self.assigments_vehicles if v.contains_costumer_day(costumer, day)]
        vehicle = vehicle[0]
        new_tour = vehicle.apply_2_opt(costumer, day)
        if costumer.timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        new_matriz = np.zeros(self.ants[timetable][day].global_update.shape)
        n = len(new_tour)
        for i in range(n):
            c1 = new_tour[i]
            j = i+1
            if j != n:
                c2 = new_tour[j]
                new_matriz[c1][c2] = 1
                new_matriz[c2][c1] = 1
        new_matriz[0][new_tour[-1]] = 1
        new_matriz[new_tour[-1]][0] = 1
        self.ants[timetable][day].global_update = new_matriz
        self.get_fitness()

    # For LNS
    def build_paths_ants(self):
        for h in self.timetables:
            for d in range(self.days):
                ant_d_h = np.zeros_like(self.ants[h][d].global_update)
                v_tours = [v for v in self.assigments_vehicles if d in v.tour[h].keys()]
                for v in v_tours:
                    tour = [c.id for c in v.tour[h][d]]
                    tour_len = len(tour)
                    for i in range(tour_len):
                        c1 = tour[i]
                        j = i + 1
                        if j != tour_len:
                            c2 = tour[j]
                            ant_d_h[c1][c2] = 1
                            ant_d_h[c2][c1] = 1
                    ant_d_h[0][tour[-1]] = 1
                    ant_d_h[tour[-1]][0] = 1
                self.ants[h][d].global_update = ant_d_h
