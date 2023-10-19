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

    def readInstance(self, instancePath):
        instanceName = instancePath.split('/')[-1].split('.')[0]
        # TODO
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
        