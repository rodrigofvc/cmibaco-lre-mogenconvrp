class Costumer(object):

    def __init__(self, id, timetable):
        self.id = id
        # 0 AM client,  1 AM client
        self.timetable = timetable
        # [(day, demand), ...]
        self.demands = demands
        # [(day, service_time),....]
        self.service_times = service_times
        
