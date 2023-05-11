import numpy as np
import math

class Ant:
    def __init__(self, nest):
        self.nest = nest

    def get_remaining_costumers(self, costumers_dh, costumers_attended, day):
        remainig = []
        for costumer in costumers_dh:
            if not costumer in costumers_attended:
                remainig.append(costumer)
        return remainig

    def get_psi_ij(self, costumer_i, costumer_j, day):
        estimated_ij = costumer_i.arrival_times[day] + costumer_i.service_times[day] + costumer_i.distance_to(costumer_j)
        diffs = [(abs(a-estimated_ij)) for a in costumer_j.arrival_times if a > -1] + [0]
        wait_ij = max(diffs)
        psi_ij = 1 / max(1, wait_ij)
        return psi_ij

    def get_phi_j(self, costumer_j, current_vehicle, vehicles):
        different_vehicles = 0
        if current_vehicle.visited_costumer(costumer_j):
            return 1
        for v in vehicles:
            if v.visited_costumer(costumer_j) and v.id != current_vehicle.id:
                different_vehicles += 1
        phi_j = 1 / max(1, different_vehicles)
        return phi_j

    def get_eta_ij(self, costumer_i, costumer_j):
        eta_ij = 1 / costumer_i.distance_to(costumer_j)
        return eta_ij

    def get_probabilities_from_costumer(self, current_costumer, remaining_costumers, pheromone_matrix, day, timetable, alpha, beta, gamma, delta, Q, current_vehicle, vehicles):
        probabilities = []
        pheromone_matrix_day_h = pheromone_matrix[timetable][day]
        for remaining_costumer in remaining_costumers:
            i = current_costumer.id
            j = remaining_costumer.id
            if i == j:
                raise()
            pheromone_dh_ij = pheromone_matrix_day_h[i][j]
            eta_ij = self.get_eta_ij(current_costumer, remaining_costumer)
            psi_ij = self.get_psi_ij(current_costumer, remaining_costumer, day)
            phi_j = self.get_phi_j(remaining_costumer, current_vehicle, vehicles)
            prob_ij = math.pow(pheromone_dh_ij, alpha) * math.pow(eta_ij, beta) * math.pow(psi_ij,gamma) * math.pow(phi_j, delta)
            probabilities.append(prob_ij)
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        return probabilities


    def get_next_costumer(self, remaining_costumers, current_costumer, alpha, beta, gamma, delta, Q, current_vehicle, pheromone_matrix, day, timetable, vehicles):
        probabilities = self.get_probabilities_from_costumer(current_costumer, remaining_costumers, pheromone_matrix, day, timetable, alpha, beta, gamma, delta, Q, current_vehicle, vehicles)
        next_costumer = np.random.choice(remaining_costumers, 1, p=probabilities)[0]
        return next_costumer

    def get_costumers_day(self, costumers_dh, day):
        costumers = []
        for costumer in costumers_dh:
            if costumer.demands[day] > 0 and costumer.id != 0:
                costumers.append(costumer)
        return costumers

    # Create a solution for a planning in a specific day, timetable
    def build_solution(self, pheromone_matrix, day, timetable, alpha, beta, gamma, delta, Q, costumers_dh, vehicles):
        tour = [self.nest]
        current_costumer = tour[0]
        i = 0
        current_vehicle = vehicles[i]
        current_vehicle.set_tour_day(day, tour)
        costumers_attended = []
        costumers_day = self.get_costumers_day(costumers_dh, day)
        while len(costumers_attended) != len(costumers_day):
            remaining_costumers = self.get_remaining_costumers(costumers_day, costumers_attended, day)
            next_costumer = self.get_next_costumer(remaining_costumers, current_costumer, alpha, beta, gamma, delta, Q, current_vehicle, pheromone_matrix, day, timetable, vehicles)
            if current_vehicle.loads[day] + next_costumer.demands[day] <= current_vehicle.capacity:
                current_vehicle.add_costumer_tour_day(day, next_costumer)
                costumers_attended.append(next_costumer)
                current_costumer = next_costumer
            else:
                tour = [self.nest]
                current_costumer = tour[0]
                i += 1
                current_vehicle = vehicles[i]
                current_vehicle.set_tour_day(day, tour)
            #print (f'{len(costumers_attended)} / {len(costumers_dh)}')
