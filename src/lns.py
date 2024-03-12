import copy
import math
import random
import threading
import time
import queue
import numpy as np
from pymoo.indicators.hv import HV

from maco import build_solutions, initialize_multiple_matrix, non_dominated
from utils import get_statistics


class LNSolution():

    def __init__(self, solution, index_f_i):
        self.solution = solution
        self.if_i = index_f_i

    def get_dr(self, costumer, max_atd, epsilon=10e-4):
        z_i = costumer.get_max_vehicle_difference()
        l_i = costumer.get_max_arrival_diference()
        dr = z_i + l_i/(max_atd + epsilon)
        return dr

    def destroy_operator(self, n_removes):
        removed = []
        costumers = [c for c in self.solution.assigments_costumers if c.id != 0]
        max_atd = self.solution.get_max_difference_arrive()
        dr = [(c, self.get_dr(c, max_atd)) for c in costumers]
        dr.sort(key=lambda x: x[1], reverse=True)
        for i in range(n_removes):
            c_rmv = dr[i]
            costumer = c_rmv[0]
            self.solution.remove_costumer(costumer)
            removed.append(costumer)
        return removed

    def repair_operator(self, costumers_to_add, eta):
        cost_time = []
        for day in range(self.solution.days):
            costumers_to_add_day = [c for c in costumers_to_add if c.demands[day] > 0]
            for c in costumers_to_add_day:
                time, v, p = self.solution.get_cheapest_time_day_add(day, c)
                cost_time.append((c, time + random.uniform(-1*eta, eta), v, p))
            n = len(costumers_to_add_day)
            for i in range(n):
                cost_time.sort(key=lambda x: x[1])
                cheapest_costumer = cost_time[0]
                costumer = cheapest_costumer[0]
                time = cheapest_costumer[1]
                vehicle = cheapest_costumer[2]
                position = cheapest_costumer[3]
                if vehicle == None:
                    # There is no avaible vehicle
                    self.solution.add_costumer_new_vehicle(day, costumer)
                else:
                    vehicle.add_costumer_day_cheapest_pos(day, costumer, position)
                cost_time = []
                costumers_to_add_day.remove(costumer)
                for c in costumers_to_add_day:
                    time, v, p = self.solution.get_cheapest_time_day_add(day, c)
                    cost_time.append((c, time, v, p))

    def task_max_atd(self):
        max_atd = self.solution.get_max_difference_arrive()
        c_max_atd = [c for c in self.solution.assigments_costumers if c.get_max_arrival_diference() == max_atd]
        return c_max_atd[0]

    def get_max_pb(self, j, bc, epsilon=10e-4):
        day_latest = j.get_day_latest_at()
        max_pb = 0
        tour_day_earlist_j = self.solution.get_tour_costumer_day(j, day_latest)
        for c in tour_day_earlist_j[1:]:
            push_j_c = self.push_back(j, c)
            #print(f'push_j_c {push_j_c}')
            if push_j_c < 0:
                continue
            if push_j_c < epsilon and c != j:
                if not c in bc:
                    bc.append(c)
            if push_j_c > max_pb and self.solution.can_push_back(day_latest, j, push_j_c):
                max_pb = push_j_c
        return max_pb

    def get_max_pf(self, j, bc, epsilon=10e-4):
        day_earliest = j.get_day_earliest_at()
        max_pf = epsilon
        tour_day_earlist_j = self.solution.get_tour_costumer_day(j, day_earliest)
        for c in tour_day_earlist_j[1:]:
            if c == j:
                continue
            push_j_c = self.push_front(j, c)
            #print (f'push_foward {push_j_c}')
            if push_j_c < epsilon:
                if not c in bc:
                    bc.append(c)
            if push_j_c > max_pf and self.solution.can_push_front(day_earliest, j, push_j_c):
                max_pf = push_j_c
        return max_pf

    def push_front(self, j, k):
        j_ear_day = j.get_day_earliest_at()
        k_ear_day = k.get_day_earliest_at()
        current = j_ear_day
        l_j = j.get_max_arrival_diference()
        if j_ear_day != k_ear_day:
            # case 1
            ak_current = k.arrival_times[current]
            day_ak_latest = k.get_day_latest_at()
            ak_latest = k.arrival_times[day_ak_latest]
            l_k = k.get_max_arrival_diference()
            p_jc = (l_j - l_k + (ak_latest - ak_current))/2
        else:
            # case 2
            sec_day_ak_earliest = k.get_day_second_earliest_at()
            ak_current = k.arrival_times[current]
            ak2_earliest = k.arrival_times[sec_day_ak_earliest]
            p_jc = (l_j + ak2_earliest - ak_current)/2
        return p_jc

    def push_back(self, j, k):
        j_last_day = j.get_day_latest_at()
        k_last_day = k.get_day_latest_at()
        current = j_last_day
        l_j = j.get_max_arrival_diference()
        if j_last_day != k_last_day:
            # case 1
            sec_day_ak_earliest = k.get_day_second_earliest_at()
            ak_current = k.arrival_times[current]
            ak2_earliest = k.arrival_times[sec_day_ak_earliest]
            l_k = k.get_max_arrival_diference()
            p_jc = (l_j - l_k + (ak_current - ak2_earliest))/2
        else:
            # case 2
            sec_day_ak_earliest = k.get_day_second_earliest_at()
            ak_current = k.arrival_times[current]
            ak2_earliest = k.arrival_times[sec_day_ak_earliest]
            p_jc = (l_j + (ak_current - ak2_earliest))/2
        return p_jc

    def apply_pb(self, max_costumer, max_pb, bc):
        day = max_costumer.get_day_latest_at()
        max_pb *= -1
        #past_diff = self.solution.get_max_difference_arrive()
        self.solution.apply_pb(day, max_costumer, max_pb)
        #new_diff = self.solution.get_max_difference_arrive()
        #print(f'<<<<<< PUSH BACK past_arrive {past_diff} new arrive {new_diff} {self.solution.f_2}')

    def apply_pf(self, max_costumer, max_pf, bc):
        day = max_costumer.get_day_earliest_at()
        #past_diff = self.solution.get_max_difference_arrive()
        self.solution.apply_pf(day, max_costumer, max_pf)
        #new_diff = self.solution.get_max_difference_arrive()
        #print(f'!!!!!! PUSH FRONT past_arrive {past_diff} new arrive {new_diff} {self.solution.f_2} {max_pf} {max_costumer.id}')


    def adjust_departure_times(self):
        epsilon = 10e-4
        max_pf = 1
        max_pb = 1
        attempts = 10
        while (max_pf > epsilon or max_pb > epsilon) and attempts != 0:
            i = self.task_max_atd()
            bc = [i]
            max_pf = epsilon
            max_pb = epsilon
            max_costumer = i
            for j in bc:
                #print ('bef get max pf')
                max_pf_j = self.get_max_pf(j, bc, epsilon)
                #print (f'aft get max pf bc {len(bc)}')
                if max_pf_j < max_pf or max_pf == epsilon:
                    max_pf = max_pf_j
                    max_costumer = j

            if max_pf > epsilon:
                #print ('bef apply pf')
                self.apply_pf(max_costumer, max_pf, bc)
                #print ('aft apply pf')
            else:
                bc = [i]
                max_pb = self.solution.get_max_difference_arrive()
                for j in bc:
                    #print('bef get max pb')
                    max_pb_j = self.get_max_pb(j, bc, epsilon)
                    #print('aft get max pb')
                    if max_pb_j < max_pb or max_pb == epsilon:
                        max_pb = max_pb_j
                        max_costumer = j
                if max_pb > epsilon:
                    #print('bef apply pb')
                    self.apply_pb(max_costumer, max_pb, bc)
                    #print('aft apply pb')
            attempts -= 1
    def improve_time_consistency(self, delta, ub_1, ub_2):
        new_f = 0.0
        old_f = 0.0
        attempts = 10
        while old_f <= new_f and attempts != 0:
            i = self.task_max_atd()
            old_f = self.get_f_i(delta, ub_1, ub_2)
            aat = i.get_average_at()
            eat = i.get_earliest_at()
            lat = i.get_latest_at()
            if aat - eat > lat - aat:
                d = i.get_day_earliest_at()
            else:
                d = i.get_day_latest_at()
            self.solution.apply_2_opt(i, d)
            new_f = self.get_f_i(delta, ub_1, ub_2)
            attempts -= 1

    def get_f_i(self, delta, ub_1, ub_2):
        alpha_1 = 1 / (1 + delta/ub_2)
        alpha_3 = 1 / (1 + ub_1/delta)
        alpha_2 = (alpha_1 + alpha_3)/2
        if self.if_i == 0:
            f_avg = alpha_1 * self.solution.f_1 + (1 - alpha_1) * self.solution.f_3
            return f_avg
        elif self.if_i == 1:
            f_avg = alpha_2 * self.solution.f_1 + (1 - alpha_2) * self.solution.f_3
            return f_avg
        else:
            f_avg = alpha_3 * self.solution.f_1 + (1 - alpha_3) * self.solution.f_3
            return f_avg

def wrap_solutions(population, index_f_i):
    wrapper = []
    for p in population:
        w = LNSolution(p, index_f_i)
        wrapper.append(w)
    return wrapper

def unwrap_solutions(population):
    unwrapper = [s.solution for s in population]
    return unwrapper

def external_mdls(params):
    days = params['days']
    epsilon = params['epsilon']
    dy = params['dy']
    costumers = params['costumers']
    vehicles = params['vehicles']
    log_hypervolume = []
    ref_point = np.array([15000, 8, 900])
    ind = HV(ref_point=ref_point)
    n_costumers = len(costumers)
    pheromone_matrix = initialize_multiple_matrix(days, n_costumers, True)
    solutions = build_solutions(params, pheromone_matrix)
    start = time.time()
    improved_solutions = mdls(solutions, params)
    duration = time.time() - start
    A = []
    #A, solutions_accepted = archive_update_pqedy(A, improved_solutions, epsilon, dy)
    A, solutions_accepted = non_dominated(A, improved_solutions)
    l = [a.f_1 for a in A]
    min_time_tour = min(l)
    max_time_tour = max(l)
    avg_time_tour = sum(l) / len(l)
    l1 = [a.f_2 for a in A]
    min_arrival_time = min(l1)
    max_arrival_time = max(l1)
    avg_arrival_time = sum(l1) / len(l1)
    l2 = [a.f_3 for a in A]
    min_vehicle = min(l2)
    max_vehicle = max(l2)
    avg_vehicle = sum(l2) / len(l2)
    n_solutions = len(A)
    print(f'>>>> min time_tour {min_time_tour}, min arrival {min_arrival_time}, min vehicle {min_vehicle} - max time_tour {max_time_tour}, max arrival {max_arrival_time}, max vehicle {max_vehicle}')
    archive = np.array([[a.f_1, a.f_2, a.f_3] for a in A])
    log_hypervolume.append(ind(archive))
    statistics = get_statistics(A, log_hypervolume, [], duration)
    print(f'Hypervolume: { ind(archive) }')
    return A, log_hypervolume, duration, statistics

def mdls(current_population, params):
    new_solutions = []
    for i, s in enumerate(current_population):
        current = s
        start = time.time()
        for f_i in range(3):
            wrapper_solution = wrap_solutions([current], f_i)
            lns_s = lns(wrapper_solution[0], params)
            #print(f'LNS {i}/{len(current_population)} {f_i} | {lns_s.solution.f_1, lns_s.solution.f_2, lns_s.solution.f_3} {time.time() - start}')
            current = lns_s.solution
        new_solutions.append(current)
    return new_solutions


def exec_lns_thread(params, s, new_solutions):
    current = s
    for f_i in range(3):
        wrapper_solution = wrap_solutions([current], f_i)
        lns_s = lns(wrapper_solution[0], params)
        current = lns_s.solution
    new_solutions.put(current)

def parallel_mdls(current_population, params):
    new_solutions = queue.Queue()
    threads = []
    for s in current_population:
        thread_i = threading.Thread(target=exec_lns_thread, args=(params, s, new_solutions))
        threads.append(thread_i)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    new_solutions = list(new_solutions.queue)
    return new_solutions

def lns_search(current_population, index_f_i, params):
    solutions = wrap_solutions(current_population, index_f_i)
    new_solutions = []
    for j, s in enumerate(solutions):
        lns_s = lns(s, params)
        new_solutions.append(lns_s)
    return unwrap_solutions(new_solutions)

def lns(s, params):
    current = s
    best = s
    max_iterations = params['max_iterations']
    n_removes = params['n_removes']
    eta = params['eta']
    delta = params['delta']
    ub_1 = params['ub_1']
    ub_2 = params['ub_2']
    for t in range(max_iterations):
        s_lns = copy.deepcopy(current)
        costumers_to_add = s_lns.destroy_operator(n_removes)
        s_lns.repair_operator(costumers_to_add, eta)
        s_lns.solution.get_fitness(count=False)
        s_lns.improve_time_consistency(delta, ub_1, ub_2)
        s_lns.adjust_departure_times()
        s_lns.solution.get_fitness()
        #print (f'LNS {t}/{max_iterations} current {current_f_i}: {current.get_f_i(delta, ub_1, ub_2):.4f} generated {s_lns_f_i}: {s_lns.get_f_i(delta, ub_1, ub_2):.4f} best {best_f_i}: {best.get_f_i(delta, ub_1, ub_2):.4f}')
        if s_lns.get_f_i(delta, ub_1, ub_2) < current.get_f_i(delta, ub_1, ub_2):
            s_lns.solution.build_paths_ants()
            current = s_lns
        if s_lns.get_f_i(delta, ub_1, ub_2) < best.get_f_i(delta, ub_1, ub_2):
            best = s_lns
    return best


def lns_sa(s, params):
    initial = s
    current = s
    best = s
    max_iterations = params['max_iterations']
    n_removes = params['n_removes']
    eta = params['eta']
    w = params['w_t']
    w_vehicles = params['w_vehicles']
    temperature_decrement = params['temperature_decrement']
    delta = params['delta']
    ub_1 = params['ub_1']
    ub_2 = params['ub_2']
    temperature = -1 * (w/math.log(0.5)) * s.get_f_i(delta, ub_1, ub_2)
    for t in range(max_iterations):
        s_lns = copy.deepcopy(current)
        costumers_to_add = s_lns.destroy_operator(n_removes)
        s_lns.repair_operator(costumers_to_add, eta)
        s_lns.solution.get_fitness(count=False)
        s_lns.adjust_departure_times()
        s_lns.improve_time_consistency(delta, ub_1, ub_2)
        s_lns.solution.get_fitness()
        if accepted_solution(initial, current, s_lns, max_iterations, temperature, w_vehicles, delta, ub_1, ub_2):
            s_lns.solution.build_paths_ants()
            current = s_lns
        if s_lns.get_f_i(delta, ub_1, ub_2) < best.get_f_i(delta, ub_1, ub_2):
            s_lns.solution.build_paths_ants()
            best = s_lns
        temperature *= temperature_decrement
    return best


def accepted_solution(initial, current, s_lns, max_iterations, temperature, w_vehicles, delta, ub_1, ub_2):
    g_s = get_g_s(initial, current, max_iterations, w_vehicles,delta, ub_1, ub_2)
    g_s_lns = get_g_s(initial, s_lns, max_iterations, w_vehicles, delta, ub_1, ub_2)
    if g_s_lns < g_s:
        return True
    prob_accept = math.exp((-1 * (g_s_lns - g_s)) / temperature)
    p = random.random()
    if p <= prob_accept:
        return True
    return False

def get_g_s(initial, s, max_iterations, w_vehicles, delta, ub_1, ub_2):
    penalty_s = get_penalty(initial, max_iterations, delta, ub_1, ub_2)
    z_i = s.solution.get_max_difference_drivers()
    tmp = math.exp(max([0, z_i - w_vehicles])/penalty_s)
    g_s = s.get_f_i(delta, ub_1, ub_2) + (tmp - 1) * max_iterations
    return g_s

def get_penalty(initial, max_iterations, delta, ub_1, ub_2):
    penalty = 1 / (math.log(initial.get_f_i(delta, ub_1, ub_2)/max_iterations) + 1)
    return penalty
