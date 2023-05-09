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
        self.arrival_times = []

    def get_distance(self, other):
        return -1

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
        return to_string
