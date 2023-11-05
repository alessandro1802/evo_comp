import csv
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import random

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
    
    def starting_solution(self, startNode, type):
        if type == "RANDOM":
            solution = random.sample(self.cities-startNode, self.targetSolutionSize-1)
            return [startNode] + solution
        # TO DO- second type

    def calculate_delta(self, solution, prev_edge, new_edge, start_edge):
        return self.getTotalDistance(solution) - self.getTotalEdgeCost(start_edge, prev_edge) + self.getTotalEdgeCost(start_edge, new_edge)
    
    # TO DO
    def generate_intra(self, current_route):
        intras = {}
        # change the order of nodes within the same set of nodes

        # change the order of edges within the same set of nodes
        return 0
    
    def generate_inter(self, current_route):
        inters = {}
        # change the set of selected nodes- one selected, one not selected
        # Are we changing the starting node???- now no
        for i in range(1, current_route):
            for node in list(set(self.cities) - set(current_route)):
                new_solution = current_route.copy()
                new_solution[i] = node
                delta = self.calculate_delta(current_route, prev_edge=current_route[i], new_edge=node, start_edge=current_route[i-1])
                inters[new_solution] = delta
        # return dictionary of nodes + their deltas
        return inters

    def get_neighbourhood(self, current_route):
        return self.generate_inter(current_route) + self.generate_intra(current_route)
    
    def search_neighbourhood(self, current_route):
        neighbourhood = self.get_neighbourhood(current_route)
        best_delta = np.inf
        best_neighbour = []

        # find neighbour with the smallest delta
        for neighbour in neighbourhood:
            current_delta = neighbourhood[neighbour]
            if current_delta < best_delta:
                best_delta = current_delta
                best_neighbour = neighbour
        return best_neighbour # return best neighbour

    def steepest_search(self, startNode, type):
        # Generate starting solution
        current_route = self.starting_solution(startNode, type=type)
        best_route = current_route.copy()
        best_dist = self.getTotalDistance(best_route)
        stop = 0
        # Find the best move in the neighbourhood
        while stop != 1:
            # Find the best move in the neighbourhood- using deltas
            current_route = self.search_neighbourhood(best_route)
            current_dist = self.getTotalDistance(current_route)
            # If the total distance of a new solution is smaller, change the current solution to the new solution
            if current_dist < best_dist:
                best_dist = current_dist
                best_route = current_route
            else:
                stop = 1
        return best_route
    
    # TO CHANGE
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

    # TO CHANGE
    def solve(self):        
        algorithm = "steepest"
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

        algorithm = "greedy"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.greedy_2_regret(startNode, weights = [0.5, 0.5]))
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
