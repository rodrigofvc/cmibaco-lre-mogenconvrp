class Vehicle():

    def __init__(self, capacity):
        self.capacity = capacity
        self.tour = {}

    def set_tour_day(self, tour, day):
        self.tour[day] = tour


    def get_time(self):
        return 0
