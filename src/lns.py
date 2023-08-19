
class LNSolution():

    def __init__(self, solution, f_i):
        self.solution = solution
        self.f_i = f_i

    def get_dr(self, costumer, max_atd, epsilon=10e-4):
        z_i = costumer.get_max_vehicle_difference()
        l_i = costumer.get_max_arrival_diference()
        dr = z_i + l_i/(max_atd + epsilon)
        return dr

    def destroy_operator(self, n_removes):
        removed = []
        costumers = [c for c in self.solution.assigments_costumers if c.id != 0]
        max_atd = self.solution.get_max_difference_arrive()
        dr = [(c,self.get_dr(c, max_atd)) for c in costumers]
        dr.sort(key=lambda x: x[1], reverse=True)
        for i in range(n_removes):
            c_rmv = dr[i]
            costumer = c_rmv[0]
            self.solution.remove_costumer(costumer)
            removed.append(costumer)
        return removed

    def repair_operator(self, costumers_to_add, eta_c):
        cost_time = []
        for day in range(self.solution.days):
            costumers_to_add_day = [c for c in costumers_to_add if c.demands[day] > 0]
            for c in costumers_to_add_day:
                time, v, p = self.solution.get_cheapest_time_day_add(day, c)
                cost_time.append((c, time, v, p))
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
            if push_j_c < epsilon and c != j:
                bc.add(c)
            if push_j_c > max_pb:
                max_pb = push_j_c
        return max_pb

    def get_max_pf(self, j, bc, epsilon=10e-4):
        day_earliest = j.get_day_earliest_at()
        max_pf = 0
        tour_day_earlist_j = self.solution.get_tour_costumer_day(j, day_earliest)
        for c in tour_day_earlist_j[1:]:
            push_j_c = self.push_front(j, c)
            if push_j_c < epsilon and c != j:
                bc.add(c)
            if push_j_c > max_pf:
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
        current = j_ear_day
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
        self.solution.apply_pb(day, max_costumer, max_pb)

    def apply_pf(self, max_costumer, max_pf, bc):
        day = max_costumer.get_day_earliest_at()
        past_diff = self.solution.get_max_difference_arrive()
        self.solution.apply_pf(day, max_costumer, max_pf)
        new_diff = self.solution.get_max_difference_arrive()

    def adjust_departure_times(self):
        epsilon=10e-4
        max_pf = epsilon
        max_pb = epsilon
        while max_pf <= epsilon and max_pb <= epsilon:
            i  = self.task_max_atd()
            bc = [i]
            max_pf = self.solution.get_max_difference_arrive()
            max_costumer = i
            for j in bc:
                max_pf_j = self.get_max_pf(j, bc, epsilon)
                if max_pf_j < max_pf:
                    max_pf = max_pf_j
                    max_costumer = j

            if max_pf > epsilon:
                self.apply_pf(max_costumer, max_pf, bc)
            else:
                bc = [i]
                max_pb = self.solution.get_max_difference_arrive()
                for j in bc:
                    max_pb_j = self.get_max_pb(j, bc, epsilon)
                    if max_pb_j < max_pb:
                        max_pb = max_pb_j
                        max_costumer = j
                if max_pb > epsilon:
                    self.apply_pb(max_costumer, max_pb, bc)


def wrap_solutions(population):
    wrapper = []
    for p in population:
        f_i = p.f_1
        w = LNSolution(p, f_i)
        wrapper.append(w)
    return wrapper

def unwrap_solutions(population):
    unwrapper = [s.solution for s in population]
    return unwrapper

def lns_search(current_population):
    solutions = wrap_solutions(current_population)
    n_removes = 20
    eta_c = 0.30
    for s in solutions:
        costumers_to_add = s.destroy_operator(n_removes)
        s.repair_operator(costumers_to_add, eta_c)
        s.adjust_departure_times()
        s.solution.is_feasible()
        s.solution.get_fitness()
         


        #print (f'{s.solution.f_1} {s.solution.f_2} {s.solution.f_3}')

    return unwrap_solutions(solutions)
