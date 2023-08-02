import random
import numpy as np

class Point:
    counter = 0
    def __init__(self, solution, f_i):
        self.solution = solution
        self.f_i = f_i
        self.original_f_i = f_i
        self.closest_point = None
        self.distance_point = -1
        Point.counter += 1
        self.id = Point.counter

    def dominates(self, other):
        return (self.f_i <= other.f_i).all()

    def crossover(self, other, prob_cross):
        childs = []
        p = random.random()
        if p <= prob_cross:
            new_f = np.zeros(self.f_i.shape)
            new_f[0] = other.f_i[0] + np.random.uniform(1,35)
            new_f[1] = self.f_i[1] + np.random.uniform(1,35)
            new_f[2] = other.f_i[2] + np.random.uniform(1,35)
            new = Point(None, new_f)
            childs.append(new)

            new_f1 = np.zeros(self.f_i.shape)
            new_f1[0] = self.f_i[0] + np.random.uniform(1,35)
            new_f1[1] = other.f_i[1] + np.random.uniform(1,35)
            new_f1[2] = self.f_i[2] + np.random.uniform(1,35)

            while (new_f1 == new_f).all():
                #los mismos padres vuelven a cruzarse
                new_f1 = np.zeros(self.f_i.shape)
                new_f1[0] = self.f_i[0] + np.random.uniform(1,35)
                new_f1[1] = other.f_i[1] + np.random.uniform(1,35)
                new_f1[2] = self.f_i[2] +  np.random.uniform(1,35)

            new1 = Point(None, new_f1)
            childs.append(new1)
        return childs

    def mutation(self, prob_mut):
        for i,_ in enumerate(self.f_i):
            p = random.random()
            if p <= prob_mut:
                self.f_i[i] = np.random.uniform(36,90)

    def __eq__(self, other):
        return self.id == other.id
        
