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

    # For LNS
    def contains_costumer(self, costumer):
        timetable = costumer.timetable
        for day in self.tour[timetable]:
            if costumer in self.tour[timetable][day]:
                return True
        return False

    # For LNS
    def remove_costumer(self, costumer):
        timetable = costumer.timetable
        for day in self.tour[timetable]:
            if costumer in self.tour[timetable][day]:
                i = self.tour[timetable][day].index(costumer)
                self.tour[timetable][day] = self.tour[timetable][day][:i] + self.tour[timetable][day][i+1:]
                self.adjust_times(timetable, day)
                costumer.arrival_times[day] = -1
                costumer.vehicles_visit[day] = -1

    # For LNS
    def adjust_times(self, timetable, day):
        self.loads[timetable][day] = 0
        self.time_tour[timetable][day] = 0
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
                    arrival_j = self.limit_time//2
                else:
                    arrival_j = costumer_i.arrival_times[day]
                arrival_j += costumer_i.service_times[day]
                arrival_j += costumer_i.distance_to(costumer_j)
                costumer_j.arrival_times[day] = arrival_j
                self.loads[timetable][day] += costumer_j.demands[day]
                self.times_tour[timetable][day] += costumer_i.distance_to(costumer_j)


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
                            if costumer_j.arrival_times[d] != self.limit_time//2 + costumer_i.distance_to(costumer_j):
                                raise('error in PM tour arrival time, not start in T/2')
                            continue
                        if costumer_j.arrival_times[d] != costumer_i.arrival_times[d] + costumer_i.service_times[d] + costumer_i.distance_to(costumer_j):
                            print (f'i-> j arrival_time j : {costumer_j.arrival_times[d]}, arrival_time i: {costumer_i.arrival_times[d]}, service_time i: {costumer_i.service_times[d]}  d_ij : {costumer_i.distance_to(costumer_j)}')
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
