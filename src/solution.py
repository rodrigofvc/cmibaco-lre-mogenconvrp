import numpy as np

class Solution:
    def __init__(self, timetables, days):
        self.days = days
        self.timetables = timetables
        self.assigments_vehicles = {'AM': [], 'PM': []}
        self.assigments_costumers = {'AM': [], 'PM': []}

    def add_assigment_vehicles(self, vehicles, costumers, timetable):
        self.assigments_vehicles[timetable] = vehicles
        self.assigments_costumers[timetable] = costumers

    def get_total_time(self):
        total_time = 0
        for timetable in self.timetables:
            for vehicle in self.assigments_vehicles[timetable]:
                for d in range(self.days):
                    total_time += vehicle.get_time(d)
        return total_time

    def get_max_difference_arrive(self):
        max_diff = 0
        for timetable in self.timetables:
            for costumer in self.assigments_costumers[timetable]:
                    max_diff = max(max_diff, costumer.get_max_arrival_diference())
        return max_diff

    def get_max_difference_drivers(self):
        max_driver_diff = 0
        for timetable in self.timetables:
            for costumer in self.assigments_costumers[timetable]:
                max_driver_diff = max(max_driver_diff, costumer.get_max_vehicle_difference())
        return max_driver_diff

    def get_fitness(self):
        f_1 = self.get_total_time()
        f_2 = self.get_max_difference_arrive()
        f_3 = self.get_max_difference_drivers()
        return (f_1, f_2, f_3)

    def dominates(self, y):
        f_1, f_2, f_3 = self.get_fitness()
        f_1y, f_2y, f_3y = y.get_fitness()
        if f_1 <= f_1y and f_2 <= f_2y and f_3 <= f_3y:
            return True
        return False
