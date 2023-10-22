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
        distances = np.round(distances).astype(int)
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
            tmp = np.roll(best_route, 1)
            edges = list(np.stack([best_route, tmp]).T)
            best_regret = -np.inf
            for city in unvisited_cities:
                # # Top 2 least costly insertions
                # # [1st best insertion cost, 2nd best insersion cost]
                # min_2_costs = [np.inf, np.inf]
                # for i, j in edges:
                # # Calculate cost of inserting the city between i and j
                #     insertionCost = self.distances[i][city] + self.distances[city][j] + self.costs[city] - self.distances[i][j]
                #     if insertionCost < min_2_costs[0]:
                #         min_2_costs[1] = min_2_costs[0]
                #         min_2_costs[0] = insertionCost
                #         # Insert new node instead of edge with min cost of insertion
                #         current_best_position = best_route.index(j)
                #     elif insertionCost < min_2_costs[1]:
                #         min_2_costs[1] = insertionCost

                # 2 subsequent edges
                for edge_idx in range(len(edges[:-1])):
                    i, j = edges[edge_idx]
                    insertionCost_0 = self.distances[i][city] + self.distances[city][j] + self.costs[city] - self.distances[i][j]

                    i, j = edges[edge_idx + 1]
                    insertionCost_1 = self.distances[i][city] + self.distances[city][j] + self.costs[city] - self.distances[i][j]
                    
                    regret = abs(insertionCost_0 - insertionCost_1)
                    
                    if regret > best_regret:
                        best_regret = regret
                        new_city = city
                        best_location = np.argmin([insertionCost_0, insertionCost_1])
                        best_position = edges[edge_idx + best_location][0]

                # # Top 2 least costly insertions
                # regret = min_2_costs[1] - min_2_costs[0]
                # if regret > best_regret:
                #     best_regret = regret
                #     new_city = city
                #     best_position = current_best_position
                    
            best_route.insert(best_position, new_city)
            unvisited_cities.remove(new_city)
        return best_route

    def greedy_weighted(self, startNode, weights = [0.5, 0.5]):
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
            tmp = np.roll(best_route, 1)
            edges = list(np.stack([best_route, tmp]).T)
            best_regret = -np.inf
            for city in unvisited_cities:
                # Top 2 least costly insertions
                # [1st best insertion cost, 2nd best insersion cost]
                # min_2_costs = [np.inf, np.inf]
                # for i, j in edges:
                # # Calculate cost of inserting the city between i and j
                #     insertionCost = self.distances[i][city] + self.distances[city][j] + self.costs[city] - self.distances[i][j]
                #     if insertionCost < min_2_costs[0]:
                #         min_2_costs[1] = min_2_costs[0]
                #         min_2_costs[0] = insertionCost
                #         current_best_position = best_route.index(j)
                #     elif insertionCost < min_2_costs[1]:
                #         min_2_costs[1] = insertionCost

                # 2 subsequent edges
                for edge_idx in range(len(edges[:-1])):
                    i, j = edges[edge_idx]
                    insertionCost_0 = self.distances[i][city] + self.distances[city][j] + self.costs[city] - self.distances[i][j]

                    i, j = edges[edge_idx + 1]
                    insertionCost_1 = self.distances[i][city] + self.distances[city][j] + self.costs[city] - self.distances[i][j]

                    min_cost = min(insertionCost_0, insertionCost_1)
                    regret = abs(insertionCost_0 - insertionCost_1)
                    regret = weights[0] * regret - weights[1]* min_cost
                    
                    if regret > best_regret:
                        best_regret = regret
                        new_city = city
                        best_location = np.argmin([insertionCost_0, insertionCost_1])
                        best_position = edges[edge_idx + best_location][0]

                # Top 2 least costly insertions
                # regret = min_2_costs[1] - min_2_costs[0]
                # if regret > best_regret:
                #     best_regret = regret
                #     new_city = city
                #     best_position = current_best_position
                    
            best_route.insert(best_position, new_city)
            unvisited_cities.remove(new_city)
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
        print(algorithm)
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
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.greedy_weighted(startNode))
            evaluations.append(self.getTotalDistance(solutions[-1]))
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
