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
        # [(day, arrival_time),...] at wich hour arrived the vehicle at costumer ---- TODO
        self.arrival_times = [-1] * len(self.demands)
        # drivers that visited costumer
        self.vehicles_visit = [-1] * len(self.demands)

    def distance_to(self, other):
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))

    def get_max_arrival_diference(self):
        arrivals = [a for a in self.arrival_times if a > -1]
        max_diff = 0
        for a in arrivals:
            for a_ in arrivals:
                max_diff = max(max_diff, abs(a - a_))
        return max_diff

    def get_max_vehicle_difference(self):
        vehicles = [v for v in self.vehicles_visit if v > -1]
        max_diff = len(list(set(vehicles)))
        return max_diff

    def __eq__(self, other):
        return self.id == other.id

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
