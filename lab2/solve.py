import csv
import os
from glob import glob
import numpy as np
from tqdm import tqdm


class Solver():
    def __init__(self, instancePath, outputPath):
        instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        print(instanceName)

    def readInstance(self, instancePath : str):
        instanceName = instancePath.split('/')[-1].split('.')[0]
        
        with open(instancePath, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            data = np.array(list(reader)).astype(int)

        cities = np.arange(data.shape[0])
        costs = data[:, 2]
        
        coords = data[:, :2]
        distances = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        distances = np.sqrt(distances)
        distances = np.round(distances).astype(float)
        return instanceName, cities, costs, distances
        
    def getTotalEdgeCost(self, node_i, node_j):
        return self.distances[node_i, node_j] + self.costs[node_j]
        
    def getTotalDistance(self, route):
        total = 0
        for i in range(len(route) - 1):
            total += self.getTotalEdgeCost(route[i], route[i + 1])
        # Return to the starting city
        total += self.getTotalEdgeCost(route[-1], route[0])
        return total
    
    def greedy_2_regret(self, startNode):        
        # Select 2nd initial city (greedily nearest)
        minTotalCost = np.inf
        for j in self.cities:
            if j != startNode:
                nodeCost = self.getTotalEdgeCost(startNode, j)
                if nodeCost < minTotalCost:
                    bestNode = j
        best_route = [startNode, bestNode]
        # Greedily build the TSP route
        unvisited_cities = list(set(self.cities) - set(best_route))
        while len(best_route) < self.targetSolutionSize:
            max_regret = -np.inf
            for city in unvisited_cities:
                tmp1 = range(len(best_route))
                tmp2 = np.roll(range(len(best_route)), 1)
                edges = np.stack([tmp1, tmp2]).T
                for i, j in edges:
                    totalCost_i = self.getTotalEdgeCost(best_route[i], city)
                    totalCost_j = self.getTotalEdgeCost(best_route[j], city)
                    minDiff = min(totalCost_i, totalCost_j)
                        
                    regret_i = totalCost_i - minDiff
                    regret_j = totalCost_j - minDiff
                    regret = regret_i + regret_j
                
                    if max_regret < regret:
                        if regret_i == 0:
                            prev_city_idx = i
                        else: 
                            prev_city_idx = j
                        max_regret = regret
                        next_city = city
            best_route.insert(prev_city_idx, next_city)
            unvisited_cities.remove(next_city)
        return best_route

    def get_closest(self, route, vertex):
        best_dist = np.inf
        for v in route:
            if self.distances[vertex, v] < best_dist and vertex not in route:
                best_dist = self.distances[vertex, v]
                best_city = v
        return best_city

    def get_closest_and_second_closest(self, best_route, vertex):
        best_city = self.get_closest(best_route, vertex)
        best_route2 = best_route.copy()
        best_route2.remove(best_city)
        second_best = self.get_closest(best_route2, vertex)
        return best_city, second_best

    def greedy_weighted(self, startNode, weight_obj_funct, weight_2_regret):
        # Select 2nd initial city (greedily nearest)
        minTotalCost = np.inf
        for j in self.cities:
            if j != startNode:
                nodeCost = self.getTotalEdgeCost(startNode, j)
                if nodeCost < minTotalCost:
                    bestNode = j
        best_route = [startNode, bestNode]
        # Greedily build the TSP route
        unvisited_cities = list(set(self.cities) - set(best_route))
        regrests = dict()
        while len(best_route) < self.targetSolutionSize:
        # while unvisited_cities:
            max_regret = 0
            for city in unvisited_cities:
                # Check if reagret has already been calculated
                if city in regrests:
                    regret = regrests[city]
                else:
                    # EDIT START
                    best_city, second_best = self.get_closest_and_second_closest(best_route, city)
                    first_loc_cost = self.getTotalEdgeCost(city, best_city)
                    second_loc_cost = self.getTotalEdgeCost(city, second_best)

                    best_index = best_route.index(best_city)
                    second_index = best_route.index(second_best)

                    temp_route = best_route.copy()
                    if best_index+1 < len(temp_route):
                        temp_route.insert(best_index+1, city)
                    else:
                        temp_route.append(city)
                    first_loc_change = self.getTotalDistance(temp_route)

                    temp_route.remove(city)
                    if second_index+1 < len(temp_route):
                        temp_route.insert(second_index+1, city)
                    else:
                        temp_route.append(city)
                    second_loc_change = self.getTotalDistance(temp_route)
                    
                    # best change in the objective function = the smallest change
                    regret = weight_2_regret * np.abs(first_loc_cost-second_loc_cost) - weight_obj_funct * np.abs(first_loc_change-second_loc_change)
                    if first_loc_change > second_loc_change:
                        location = second_index + 1
                    else:
                        location = best_index + 1
                if regret > max_regret:
                    max_regret = regret
                    next_city = city
                    best_location = location
                else:
                    regrests[city] = regret

            if best_location < len(best_route):
                best_route.insert(best_location, next_city)
            else:
                best_route.append(next_city)
            unvisited_cities.remove(next_city)
        return best_route

    def calculateStats(self, evaluations):
        best_sol_idx = np.argmin(evaluations)
        min_result = np.amin(evaluations)
        avg_result = np.mean(evaluations)
        max_result = np.amax(evaluations)
        return min_result, avg_result, max_result, best_sol_idx
    
    def writeRouteToCSV(self, route, outputPath):
        with open(outputPath, 'w') as f:
            write = csv.writer(f)
            write.writerows(np.array(route)[:, np.newaxis])

    def solve(self):
        algorithm = "greedy_2_regret"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        for startNode in tqdm(self.cities):
            solutions.append(self.greedy_2_regret(startNode))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        min_result, avg_result, max_result, best_sol_idx = self.calculateStats(evaluations)
        print(f"MIN {min_result} AVG {avg_result} MAX {max_result}")
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)

        algorithm = "greedy_weighted"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        for startNode in self.cities:
            solutions.append(self.greedy_weighted(startNode, weight_obj_funct=1, weight_2_regret=1))
            evaluations.append(self.getTotalDistance(solutions[-1])
        # Get and print stats
        min_result, avg_result, max_result, best_sol_idx = self.calculateStats(evaluations)
        print(f"MIN {min_result} AVG {avg_result} MAX {max_result}")
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        solver = Solver(instancePath, outputPath)
        solver.solve()
