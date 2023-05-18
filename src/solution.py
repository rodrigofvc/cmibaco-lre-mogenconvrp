import numpy as np

class Solution:
    def __init__(self, timetables, days):
        self.days = days
        self.timetables = timetables
        self.assigments_vehicles = {'AM': [], 'PM': []}
        self.assigments_costumers = {'AM': [], 'PM': []}
        self.f_1 = None
        self.f_2 = None
        self.f_3 = None
        self.f_4 = None

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

    def max_costumer(self):
        c = None
        max_diff = 0
        for timetable in self.timetables:
            for costumer in self.assigments_costumers[timetable]:
                    m = costumer.get_max_arrival_diference()
                    if m >= max_diff:
                        c = costumer
                        max_diff = m
        return c

    def get_mean_arrive_difference(self):
        diff = 0
        n = 0
        for timetable in self.timetables:
            n += len(self.assigments_costumers[timetable])
            for costumer in self.assigments_costumers[timetable]:
                diff += costumer.get_max_arrival_diference()
        return diff / n

    def get_max_difference_drivers(self):
        max_driver_diff = 0
        for timetable in self.timetables:
            for costumer in self.assigments_costumers[timetable]:
                max_driver_diff = max(max_driver_diff, costumer.get_max_vehicle_difference())
        return max_driver_diff

    def get_fitness(self):
        self.f_1 = self.get_total_time()
        self.f_2 = self.get_max_difference_arrive()
        self.f_3 = self.get_max_difference_drivers()
        self.f_4 = self.get_mean_arrive_difference()
        return (self.f_1, self.f_2, self.f_3)

    def dominates(self, y):
        # F(X) == F(Y)
        if abs(self.f_1 - y.f_1) <= 10e-8 and abs(self.f_2 - y.f_2) <= 10e-8 and self.f_3 == y.f_3:
            return False
        # F(X) <= F(Y)
        if abs(self.f_1 - y.f_1) <= 10e-8 and abs(self.f_2 - y.f_2) <= 10e-8 and self.f_3 <= y.f_3:
            return True
        return False

    def is_feasible(self):
        for timetable in self.timetables:
            vehicles = self.assigments_vehicles[timetable]
            for vehicle in vehicles:
                vehicle.is_feasible()
            customers = self.assigments_costumers[timetable]
            for customer in customers:
                customer.is_feasible()
        return True
