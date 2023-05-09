class Vehicle():

    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.tour = {}

    def set_tour_day(self, day, tour):
        self.tour[day] = tour


    def get_time(self):
        return 0
