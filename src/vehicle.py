class Vehicle():

    def __init__(self, id, capacity, days, limit_time):
        self.id = id
        self.capacity = capacity
        self.loads = [0] * days
        self.tour = {}
        self.times_tour = [0] * days
        self.limit_time = limit_time

    def set_tour_day(self, day, tour):
        self.tour[day] = tour

    def add_costumer_tour_day(self, day, costumer_j):
        costumer_i = self.tour[day][-1]
        if costumer_j.service_times[day] <= -1:
            raise()
        if costumer_i.arrival_times[day] <= 0 and costumer_i.id != 0:
            raise()
        arrival_j = costumer_i.arrival_times[day]
        arrival_j += costumer_i.service_times[day]
        arrival_j += costumer_i.distance_to(costumer_j)
        costumer_j.arrival_times[day] = arrival_j
        self.tour[day].append(costumer_j)
        self.loads[day] += costumer_j.demands[day]
        self.times_tour[day] += costumer_i.distance_to(costumer_j)
        costumer_j.vehicles_visit[day] = self.id
        self.check_load(day)
        current_time = arrival_j + costumer_j.service_times[day]
        return current_time

    def return_depot(self, day):
        costumer_j = self.tour[day][-1]
        depot = self.tour[day][0]
        time = costumer_j.distance_to(depot)
        self.times_tour[day] += time

    def check_load(self, day):
        if self.loads[day] > self.capacity:
            raise(f'Vehicle {self.id} overfull')

    def get_time(self, day):
        time = 0
        if not day in self.tour.keys():
            return time
        tour_day = self.tour[day]
        for i in range(len(tour_day)):
            costumer_1 = tour_day[i]
            j = i+1
            if j >= len(tour_day):
                break
            costumer_2 = tour_day[j]
            time += costumer_1.distance_to(costumer_2)
        time += tour_day[0].distance_to(tour_day[-1])
        return time

    def is_feasible(self):
        if list(self.tour.keys()) == []:
            # Vehicle not used
            return True
        for l in self.loads:
            if l > self.capacity:
                raise()

        days = self.tour.keys()
        for d in days:
            if self.tour[d][0].id != 0:
                raise()
            sum = 0
            load = 0
            for i in range(len(self.tour[d])):
                costumer_i = self.tour[d][i]
                if costumer_i in self.tour[d][:i]:
                    print (costumer_i, [c.id for c in self.tour[d]])
                    raise()
                if costumer_i.id != 0 and costumer_i.vehicles_visit[d] != self.id:
                    raise()
                if costumer_i.service_times[d] <= 0 and costumer_i.id != 0:
                    raise()
                j = i + 1
                if j < len(self.tour[d]):
                    costumer_j = self.tour[d][j]
                    sum += costumer_i.distance_to(costumer_j)
                    if costumer_j.arrival_times[d] != costumer_i.arrival_times[d] + costumer_i.service_times[d] + costumer_i.distance_to(costumer_j):
                        raise()
                    load += costumer_j.demands[d]
                else:
                    sum += self.tour[d][i].distance_to(self.tour[d][0])
                    if self.tour[d][i].id == 0:
                        raise()
            if sum != self.times_tour[d]:
                raise(sum, self.times_tour[d])
            if load != self.loads[d]:
                raise()
