import math

class Costumer(object):

    def __init__(self, id, x, y, demands=[], service_times=[]):
        self.id = id
        self.x = x
        self.y = y
        # AM 0, PM 1
        self.timetable = int(id) % 2
        # [(day, demand), ...] how much costumer want
        self.demands = demands
        # [(day, service_time),....] how long takes the service per day
        self.service_times = service_times
        # [(day, arrival_time),...] at wich hour arrived the vehicle at costumer
        self.arrival_times = [-1] * len(self.demands)
        # drivers that visited costumer
        self.vehicles_visit = [-1] * len(self.demands)

    def is_feasible(self):
        if self.id == 0:
            if self.arrival_times.count(-1) != len(self.arrival_times):
                raise()
            if self.vehicles_visit.count(-1) != len(self.vehicles_visit):
                raise()
        for i in range(len(self.service_times)):
            if self.demands[i] != -1 and self.service_times[i] != 0 and self.arrival_times[i] != -1 and self.vehicles_visit[i] != -1:
                continue
            if self.demands[i] == -1 and self.service_times[i] == 0 and self.arrival_times[i] == -1 and self.vehicles_visit[i] == -1:
                continue
            raise()

    def distance_to(self, other):
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))

    def get_max_arrival_diference(self):
        if self.id == 0:
            return 0
        arrivals = [a for a in self.arrival_times if a > -1]
        if len(arrivals) < 2:
            # Required at least two visits to check arrivals
            return 0
        l = min(arrivals)
        g = max(arrivals)
        return g-l

    def get_worts_days(self):
        if self.id == 0:
            return []
        arrivals = [a for a in self.arrival_times if a > -1]
        if len(arrivals) < 2:
            # Required at least two visits to check arrivals
            return []
        l = min(arrivals)
        g = max(arrivals)
        day_1 = self.arrival_times.index(l)
        day_2 = self.arrival_times.index(g)
        worst = [day_1, day_2]
        return worst

    def get_max_vehicle_difference(self):
        if self.id == 0:
            return 0
        vehicles = [v for v in self.vehicles_visit if v > -1]
        max_diff = len(list(set(vehicles)))
        return max_diff

    def __eq__(self, other):
        if isinstance(other, Costumer):
            return self.id == other.id
        return False

    def __str__(self):
        to_string = 'id: ' + str(self.id) + ',\n'
        to_string += 'x: ' + str(self.x) +  ',\n'
        to_string += 'y: ' + str(self.y) + ',\n'
        to_string += 'timetable: ' + str(self.timetable) + ',\n'
        to_string += 'demands: ' + str(self.demands) + ',\n'
        to_string += 'service_times: ' + str(self.service_times) + ',\n'
        to_string += 'arrival_times: ' + str(self.arrival_times) + ',\n'
        to_string += 'vehicles_visit: ' + str(self.vehicles_visit) + ',\n'
        return to_string
