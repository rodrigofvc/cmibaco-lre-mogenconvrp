from shapely.geometry import LineString

class Vehicle():

    def __init__(self, id, capacity, days, limit_time):
        self.id = id
        self.capacity = capacity
        self.loads = {'AM': [0] * days, 'PM': [0] * days}
        self.tour = {'AM': {}, 'PM': {}}
        self.times_tour = {'AM': [0] * days, 'PM': [0] * days}
        self.limit_time = limit_time

    def set_tour_day(self, timetable, day, tour):
        self.tour[timetable][day] = tour

    def add_costumer_tour_day(self, timetable, day, costumer_j):
        costumer_i = self.tour[timetable][day][-1]
        if costumer_j.service_times[day] <= -1:
            raise()
        if costumer_i.arrival_times[day] <= 0 and costumer_i.id != 0:
            raise()
        if costumer_i.id == 0 and timetable == 'PM':
            # T/2 for PM costumers
            arrival_j = self.limit_time//2
        else:
            arrival_j = costumer_i.arrival_times[day]
        arrival_j += costumer_i.service_times[day]
        arrival_j += costumer_i.distance_to(costumer_j)
        costumer_j.arrival_times[day] = arrival_j
        self.tour[timetable][day].append(costumer_j)
        self.loads[timetable][day] += costumer_j.demands[day]
        self.times_tour[timetable][day] += costumer_i.distance_to(costumer_j)
        costumer_j.vehicles_visit[day] = self.id
        self.check_load(day)
        current_time = arrival_j + costumer_j.service_times[day]
        return current_time

    def return_depot(self, timetable, day):
        costumer_j = self.tour[timetable][day][-1]
        depot = self.tour[timetable][day][0]
        time = costumer_j.distance_to(depot)
        self.times_tour[timetable][day] += time

    def check_load(self, day):
        for timetable in self.loads:
            if self.loads[timetable][day] > self.capacity:
                print (f'Vehicle {self.id} overfull in day {day} with {self.loads[timetable][day]}/ {self.capacity}')
                raise(f'Vehicle {self.id} overfull in day {day} with {self.loads[timetable][day]}/ {self.capacity}')

    def get_time(self, timetable, day):
        time = 0
        if not day in self.tour[timetable].keys():
            return time
        tour_day = self.tour[timetable][day]
        for i in range(len(tour_day)):
            costumer_1 = tour_day[i]
            j = i+1
            if j >= len(tour_day):
                break
            costumer_2 = tour_day[j]
            time += costumer_1.distance_to(costumer_2)
        time += tour_day[0].distance_to(tour_day[-1])
        return time

    def get_load(self, timetable, day):
        load = 0
        if not day in self.tour[timetable].keys():
            return load
        tour_day = self.tour[timetable][day]
        load = [c.demands[day] for c in tour_day if c.id != 0]
        load = sum(load)
        return load

    def get_total_load_day(self, day):
        total_load = 0
        for timetable in self.loads:
            total_load += self.loads[timetable][day]
        return total_load

    def contains_costumer_day(self, costumer, day):
        timetable = costumer.timetable
        if timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        if day not in self.tour[timetable].keys():
            return False
        if costumer in self.tour[timetable][day]:
            return True
        return False

    # For LNS
    def contains_costumer(self, costumer):
        timetable = costumer.timetable
        if timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        if len(self.tour[timetable].keys()) == 0:
            return False
        for day in self.tour[timetable]:
            if costumer in self.tour[timetable][day]:
                return True
        return False

    # For LNS
    def remove_costumer(self, costumer):
        timetable = costumer.timetable
        if timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        days_remove = []
        for day in self.tour[timetable]:
            if costumer in self.tour[timetable][day]:
                i = self.tour[timetable][day].index(costumer)
                self.tour[timetable][day] = self.tour[timetable][day][:i] + self.tour[timetable][day][i+1:]
                self.adjust_times(timetable, day)
                costumer.arrival_times[day] = -1
                costumer.vehicles_visit[day] = -1
                if len (self.tour[timetable][day]) == 1:
                    days_remove.append(day)
        for day in days_remove:
            self.tour[timetable].pop(day)
            self.times_tour[timetable][day] = 0
            self.loads[timetable][day] = 0

    # For LNS
    def adjust_times(self, timetable, day):
        self.loads[timetable][day] = 0
        self.times_tour[timetable][day] = 0
        for i in range(len(self.tour[timetable][day])):
            costumer_i = self.tour[timetable][day][i]
            j = i + 1
            if j != len(self.tour[timetable][day]):
                costumer_j = self.tour[timetable][day][j]
                if costumer_j.service_times[day] <= -1:
                    raise()
                if costumer_i.arrival_times[day] <= 0 and costumer_i.id != 0:
                    raise()
                if costumer_i.id == 0 and timetable == 'PM':
                    # T/2 for PM costumers
                    arrival_j = self.limit_time/2
                else:
                    arrival_j = costumer_i.arrival_times[day]
                arrival_j += costumer_i.service_times[day]
                arrival_j += costumer_i.distance_to(costumer_j)
                costumer_j.arrival_times[day] = arrival_j
                self.loads[timetable][day] += costumer_j.demands[day]
                self.times_tour[timetable][day] += costumer_i.distance_to(costumer_j)
        self.times_tour[timetable][day] += self.tour[timetable][day][0].distance_to(self.tour[timetable][day][-1])


    # For LNS
    def get_time_sub_tour(self, timetable, day, expected_tour):
        arrival_time = 0
        if timetable == 'AM':
            arrival_time = 0
        else:
            arrival_time = self.limit_time/2
        time = 0
        for i in range(len(expected_tour)):
            costumer_i = expected_tour[i]
            j = i + 1
            if j != len(expected_tour):
                costumer_j = expected_tour[j]
                time += costumer_i.distance_to(costumer_j)
                arrival_time += costumer_i.service_times[day]
                arrival_time += costumer_i.distance_to(costumer_j)
        time += expected_tour[-1].distance_to(expected_tour[0])
        arrival_time += expected_tour[-1].service_times[day]
        arrival_time += expected_tour[-1].distance_to(expected_tour[0])
        if arrival_time > self.limit_time and timetable == 'PM':
            return float('inf')
        elif arrival_time > self.limit_time/2 and timetable == 'AM':
            return float('inf')
        return time

    # For LNS
    def can_visit_costumer_greedy(self, timetable, day, costumer):
        min_position = -1
        min_time = float('inf')
        tour = self.tour[timetable][day]
        for i in range(1, len(tour)):
            expected_tour = tour[:i] + [costumer] + tour[i:]
            expected_time = self.get_time_sub_tour(timetable, day, expected_tour)
            if expected_time < min_time and expected_time != float('inf'):
                min_time = expected_time
                min_position = i
        return min_position, min_time

    # For LNS
    def add_costumer_day_cheapest_pos(self, day, costumer, position):
        timetable = costumer.timetable
        if timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        tour = self.tour[timetable][day]
        costumer.vehicles_visit[day] = self.id
        self.tour[timetable][day] = tour[:position] + [costumer] + tour[position:]
        self.adjust_times(timetable, day)

    # For LNS 
    def push_tour(self, timetable, day, max_pf):
        depot = self.tour[timetable][day][0]
        first = self.tour[timetable][day][1]
        last = self.tour[timetable][day][-1]
        # case where max_pf is negative (because push back)
        if max_pf < 0 and first.arrival_times[day] + max_pf < depot.distance_to(first):
            return
        if timetable == 'AM':
            limit = self.limit_time/2
        else:
            limit = self.limit_time
        # case where max_pf is positive
        if max_pf > 0 and last.arrival_times[day] + last.service_times[day] + last.distance_to(depot) + max_pf > limit:
            return
        for c in self.tour[timetable][day]:
            if c.id == 0:
                continue
            c.arrival_times[day] += max_pf

    def can_push_front(self, timetable, day, max_pf):
        depot = self.tour[timetable][day][0]
        first = self.tour[timetable][day][1]
        last = self.tour[timetable][day][-1]
        if timetable == 'AM':
            limit = self.limit_time/2
        else:
            limit = self.limit_time
        if max_pf < 0:
            return False
        if last.arrival_times[day] + last.service_times[day] + last.distance_to(depot) + max_pf > limit:
            return False
        return True

    def can_push_back(self, timetable, day, max_pb):
        depot = self.tour[timetable][day][0]
        first = self.tour[timetable][day][1]
        last = self.tour[timetable][day][-1]
        if timetable == 'AM':
            limit = self.limit_time / 2
        else:
            limit = self.limit_time
        if max_pb > 0:
            # push back is negative
            return False
        if first.arrival_times[day] + max_pb < depot.distance_to(first):
            return False
        return True

    # For LNS
    def apply_2_opt(self, costumer, day):
        if costumer.timetable == 0:
            timetable = 'AM'
        else:
            timetable = 'PM'
        tour = self.tour[timetable][day]
        n = len(tour)
        lines = []
        for i in range(1, n):
            j = i + 1
            if j != n:
                c1 = tour[i]
                c2 = tour[j]
                lines.append((c1, c2))
        tuples = [(t1, t2) for t1 in lines for t2 in lines if t1[0] not in t2 and t1[1] not in t2 and self.intersects_route(t1, t2)]
        if len(tuples) > 0:
            paths = tuples[0]
            self.swap_tour(timetable, day, paths)
            self.adjust_times(timetable, day)
        ids = [c.id for c in self.tour[timetable][day]]
        return ids

    def swap_tour(self, timetable, day, paths):
        costumer_1 = paths[0][0]
        costumer_2 = paths[0][1]
        costumer_3 = paths[1][0]
        costumer_4 = paths[1][1]
        n = len(self.tour[timetable][day])
        i_1 = self.tour[timetable][day].index(costumer_1)
        i_2 = self.tour[timetable][day].index(costumer_2)
        i_3 = self.tour[timetable][day].index(costumer_3)
        i_4 = self.tour[timetable][day].index(costumer_4)
        indexes_c = [i_1, i_2, i_3, i_4]
        indexes_c = sorted(indexes_c)
        i_1 = indexes_c[0]
        i_2 = indexes_c[1]
        i_3 = indexes_c[2]
        i_4 = indexes_c[3]

        subtour_1 = self.tour[timetable][day][:i_2]
        subtour_2 = self.tour[timetable][day][i_2:i_3+1]
        subtour_2.reverse()
        subtour_4 = self.tour[timetable][day][i_4:]
        self.tour[timetable][day] = subtour_1 + subtour_2 + subtour_4
        m = len(self.tour[timetable][day])
        if n != m:
            print(n, m)
            raise('size mismatch after swap')


    def intersects_route(self, route_1, route_2):
        costumer_1 = route_1[0]
        costumer_2 = route_1[1]
        costumer_3 = route_2[0]
        costumer_4 = route_2[1]
        line_1 = LineString([(costumer_1.x, costumer_1.y), (costumer_2.x, costumer_2.y)])
        line_2 = LineString([(costumer_3.x, costumer_3.y), (costumer_4.x, costumer_4.y)])
        return line_1.intersects(line_2)

    def __eq__(self, other):
        if isinstance(other, Vehicle):
            return self.id == other.id
        return False

    # TODO
    def is_feasible(self):
        if list(self.tour['AM'].keys()) == [] and list(self.tour['PM'].keys()) == []:
            return

        for timetable in self.loads:
            for l in self.loads[timetable]:
                if l > self.capacity:
                    raise('capacity excedeed')

        for timetable in self.tour:
            days = self.tour[timetable].keys()
            for d in days:
                if not d in self.tour[timetable].keys():
                    continue
                if self.tour[timetable][d][0].id != 0:
                    raise('tour not start in depot')
                sum = 0
                load = 0
                if len(self.tour[timetable][d]) == 1:
                    print (f'>>>> {[c.id for c in self.tour[timetable][d]]}')
                    raise('tours of length 1 not allowed')
                for i in range(len(self.tour[timetable][d])):
                    costumer_i = self.tour[timetable][d][i]
                    if costumer_i in self.tour[timetable][d][i+1:]:
                        print (costumer_i, [c.id for c in self.tour[timetable][d]])
                        raise('costumers visited twice')
                    if costumer_i.id != 0 and costumer_i.vehicles_visit[d] != self.id:
                        raise('costumer in tour not visited by vehicle')
                    if costumer_i.service_times[d] < 0 and costumer_i.id != 0:
                        raise('costumer shouldnt appear in tour')
                    j = i + 1
                    if j < len(self.tour[timetable][d]):
                        costumer_j = self.tour[timetable][d][j]
                        load += costumer_j.demands[d]
                        sum += costumer_i.distance_to(costumer_j)
                        if costumer_i.id == 0 and timetable == 'PM':
                            if costumer_j.arrival_times[d] < self.limit_time//2 + costumer_i.distance_to(costumer_j):
                                raise('error in PM tour arrival time, not start in T/2')
                            continue
                        # minimum_arrival_ij is the minimum arrival time
                        minimum_arrival_ij = costumer_i.arrival_times[d] + costumer_i.service_times[d] + costumer_i.distance_to(costumer_j)
                        if costumer_j.arrival_times[d] < minimum_arrival_ij and abs(costumer_j.arrival_times[d] - minimum_arrival_ij) >= 1:
                            print (f'i-> j arrival_time j : {costumer_j.arrival_times[d]}, arrival_time i: {costumer_i.arrival_times[d]}, service_time i: {costumer_i.service_times[d]}  d_ij : {costumer_i.distance_to(costumer_j)}')
                            print (f'  estimated {minimum_arrival_ij} real {costumer_j.arrival_times[d]}')
                            raise()
                    else:
                        sum += self.tour[timetable][d][i].distance_to(self.tour[timetable][d][0])
                        if self.tour[timetable][d][i].id == 0 and len(self.tour[timetable][d]) > 1:
                            raise('last costumer shouldnt be depot')
                        if sum != self.times_tour[timetable][d]:
                            print(sum, self.times_tour[timetable][d])
                            raise('error in time tour')
                        if load != self.get_load(timetable, d):
                            raise('load wrong')
                        last_costumer = self.tour[timetable][d][-1]
                        if timetable == 'AM':
                            limit = self.limit_time/2
                        else:
                            limit = self.limit_time
                        if last_costumer.arrival_times[d] + last_costumer.service_times[d] + last_costumer.distance_to(self.tour[timetable][d][0]) > limit:
                            print (f'{last_costumer.arrival_times[d] + last_costumer.service_times[d] + last_costumer.distance_to(self.tour[timetable][d][0])} / {limit}')
                            raise('limit time excedeed')
