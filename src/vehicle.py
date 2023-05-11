class Vehicle():

    def __init__(self, id, capacity, days):
        self.id = id
        self.capacity = capacity
        self.loads = [0] * days
        self.tour = {}

    def set_tour_day(self, day, tour):
        self.tour[day] = tour

    def add_costumer_tour_day(self, day, costumer_j):
        costumer_i = self.tour[day][-1]
        if costumer_j.service_times[day] <= 0:
            raise()
        if costumer_i.arrival_times[day] <= 0 and costumer_i.id != 0:
            raise()
        arrival_j = costumer_i.arrival_times[day]
        arrival_j += costumer_i.service_times[day]
        arrival_j += costumer_i.distance_to(costumer_j)
        costumer_j.arrival_times[day] = arrival_j
        self.tour[day].append(costumer_j)
        self.loads[day] += costumer_j.demands[day]
        costumer_j.vehicles_visit[day] = self.id
        self.check_load(day)

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

    def visited_costumer(self, costumer_j):
        days = self.tour.keys()
        for d in days:
            tour_day = self.tour[d]
            if costumer_j in tour_day:
                return True
        return False

    def visited_costumer_ijdh(self, day, costumer_i, costumer_j):
        if not day in self.tour.keys():
            return False
        tour_day = self.tour[day]
        for i in range(len(tour_day)):
            j = i+1
            if j >= len(tour_day):
                break
            if costumer_i == tour_day[i].id and costumer_j == tour_day[j].id:
                return True,
        if costumer_i == tour_day[-1].id and costumer_j == tour_day[0].id:
            return True
        return False

    """
    def set_arrive_time_costumers(self):
        days = self.tour.keys()
        for d in days:
            self.set_arrive_time_costumers(d)

    def set_arrive_time_costumers(self, day):
        tour_day = self.tour[day]
        depot = tour_day[0]
        costumer_1 = tour_day[1]
        time = depot.distance_to(costumer_1)
        costumer_1.set_arrive_time(time)
        for i in range(1,len(tour_day)):
            costumer_1 = tour_day[i]
            j = i+1
            if j >= len(tour_day):
                break
            costumer_2 = tour_day[j]
            arrive_time_2 = costumer_1.arrival_times[day]
            arrive_time_2 += costumer_1.service_times[day]
            arrive_time_2 += costumer_1.distance_to(costumer_2)
            costumer_2.set_arrive_time(arrive_time_2)
    """
