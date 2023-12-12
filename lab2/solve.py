import csv
import os
from glob import glob

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from utils import Solver


class Greedy_Regret(Solver):
    def __init__(self, instancePath, outputPath):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)

    
    def getGredilyNearestCity(self, startNode):
        # Select 2nd initial city (greedily nearest)
        minTotalCost = np.inf
        for j in self.cities:
            if j != startNode:
                nodeCost = self.getTotalEdgeCost(startNode, j)
                if nodeCost < minTotalCost:
                    minTotalCost = nodeCost
                    bestNode = j
        return bestNode

    def greedy_2_regret(self, init_route, weights = []):
        best_route = init_route
        # Greedily build the TSP route
        unvisited_cities = list(set(self.cities) - set(best_route))
        while len(best_route) < self.targetSolutionSize:
            tmp = np.roll(best_route, -1)
            edges = list(np.stack([best_route, tmp]).T)
            best_regret = -np.inf
            for city in unvisited_cities:
                # Top 2 least costly insertions
                min_costs = [np.inf, np.inf]
                for node1, node2 in edges:
                    # Calculate cost of inserting the city between i and j
                    insertionCost = self.distances[node1][city] + self.distances[city][node2] + self.costs[city] - self.distances[node1][node2]
                    if insertionCost < min_costs[0]:
                        min_costs[1] = min_costs[0]
                        min_costs[0] = insertionCost
                        # Insert new node instead of edge with min cost of insertion
                        current_best_position = best_route.index(node2)
                    elif insertionCost < min_costs[1]:
                        min_costs[1] = insertionCost
                regret = min_costs[1] - min_costs[0]
                if weights:
                    regret = weights[0] * regret - weights[1] * min_costs[0]
                if regret > best_regret:
                    best_regret = regret
                    new_city = city
                    best_position = current_best_position
            best_route.insert(best_position, new_city)
            unvisited_cities.remove(new_city)
        return best_route

    
    def solve(self):
        def _solve(algorithm, weights):
            solutions = []
            evaluations = []
            # Get solutions and evaluations
            print(algorithm)
            for startNode in tqdm(self.cities):
                # Select 2nd initial city
                bestNode = self.getGredilyNearestCity(startNode)
                init_route = [startNode, bestNode]
                
                solutions.append(self.greedy_2_regret(init_route, weights = weights))
                evaluations.append(self.getTotalDistance(solutions[-1]))
            # Get and print stats
            min_result, avg_result, max_result, best_sol_idx = self.calculateStats(evaluations)
            print(f"MIN {min_result} AVG {avg_result} MAX {max_result}")
            # Save best solution
            best_sol = solutions[best_sol_idx]
            outputPath = os.path.join(self.outputPath, algorithm + ".csv")
            self.writeRouteToCSV(best_sol, outputPath)

        print(self.instanceName)
        _solve("greedy_2_regret", [])
        _solve("greedy_weighted", [0.5, 0.5])


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        solver = Greedy_Regret(instancePath, outputPath)
        solver.solve()
        print()
