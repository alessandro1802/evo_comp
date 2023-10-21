import csv
import os
from glob import glob
import numpy as np


class Solver():
    def __init__(instancePath, outputPath):
        instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        print(instanceName)

    def readInstance(self, instancePath : str):
        instanceName = instancePath.split('/')[-1].split('.')[0]
        with open(instancePath, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            data = list(reader)

        data = np.array(data).astype(int)

        coords = data[:, :2]
        cities = np.array(range(len(coords)))
        costs = data[:, 2]

        distances = np.round(np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))).astype(float)

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
        # Select 2nd initial city
        minTotalCost = np.inf
        for j in self.cities:
            if j != startNode:
                nodeCost = self.distances[startNode, j] + self.costs[j]
                if nodeCost < minTotalCost:
                    bestNode = j
        best_route = [startNode, bestNode]

        # Greedily build the TSP route
        min_distance = total_distance(best_route)
        unvisited_cities = list(set(self.cities) - set(initial_cities))
        regrests = dict{}
        while len(best_route) < self.targetSolutionSize:
        # while unvisited_cities:
            min_regret = np.inf
            for city in unvisited_cities:
                # Check if reagret has already been calculated
                if city in regrests:
                    regret = regrests[city]
                else:
                    totalCost_i = self.getTotalEdgeCost(best_route[0], city)
                    totalCost_j = self.getTotalEdgeCost(best_route[1], city)
                    minDiff = min(totalCost_i, totalCost_j)
                    
                    regret_i = totalCost_i - minDiff
                    regret_j = totalCost_j - minDiff
                    regret = regret_i + regret_j
                if regret < min_regret:
                    min_regret = regret
                    next_city = city
                # Store the regret
                else:
                    regrests[city] = regret
            best_route.append(next_city)
            unvisited_cities.remove(next_city)
        return best_route

    def greedy_weighted(self):
        # TODO
        pass

    def calculateStats(self, solutions):
        best_sol_idx = np.argmin(solutions)
        min_result = np.amin(solutions)
        avg_result = np.mean(solutions)
        max_result = np.amax(solutions)
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
        for startNode in self.cities:
            solutions.append(self.greedy_2_regret(startNode))
            evaluations.append(self.getTotalDistance(solution))
        # Get and print stats
        min_result, avg_result, max_result, best_sol_idx = self.calculateStats(solutions)
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
            solutions.append(self.greedy_weighted(startNode))
            evaluations.append(self.getTotalDistance(solution))
        # Get and print stats
        min_result, avg_result, max_result, best_sol_idx = self.calculateStats(solutions)
        print(f"MIN {min_result} AVG {avg_result} MAX {max_result}")
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"
    for instancePath in glob(os.path.join(instancesPath, "*.csv")):
        solver = Solver(instancePath, outputPath)
        solver.solve()
