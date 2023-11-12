import csv
import os
import sys
import random
from glob import glob
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from functools import partial

class Solver_LS():
    def __init__(self, instancePath, outputPath):#, heuristic_solver):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)


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
        
    def getTotalEdgeCost(self, node_j, node_i):
        return self.distances[node_i, node_j] + self.costs[node_j]
        
    def getTotalDistance(self, route):
        total = 0
        for i in range(len(route) - 1):
            total += self.getTotalEdgeCost(route[i], route[i + 1])
        # Return to the starting city
        total += self.getTotalEdgeCost(route[-1], route[0])
        return total
    
    def getNClosest(self, node_1, nodes):
        nodes_cpy = nodes.copy()
        if node_1 in nodes_cpy:
            nodes_cpy.remove(node_1)
        return sorted(nodes_cpy, key=partial(self.getTotalEdgeCost, node_i=node_1))[:10]

    
    ### Starting solutions
    def random_ss(self, startNode):
        return [startNode] + random.sample(list(set(self.cities) - {startNode}), self.targetSolutionSize - 1)

    def best_greedy_heuristic_ss(self, startNode):
        solution = self.heuristic_solver.greedy_2_regret(startNode, weights = [0.5, 0.5])
        return solution
    
    
    ### Neighbouhoods
    # Intra-route - Moves changing the order of nodes within the same set of selected nodes:
    #     two-nodes exchange;
    #     two-edges exchange;
    def getDeltaIntraNodes(self, prev_i, i, next_i, prev_j, j, next_j):
        # Adjacent nodes (i > j)
        if next_i == j:
            return (self.distances[i, next_j] + self.distances[prev_i, j]) \
             - (self.distances[prev_i, i] + self.distances[j, next_j])
        # Last - First
        elif next_j == i:
            return (self.distances[i, prev_j] + self.distances[j, next_i]) \
             - (self.distances[i, next_i] + self.distances[prev_j, j])
        return (self.distances[prev_i, j] + self.distances[j, next_i] + self.distances[prev_j, i] + self.distances[i, next_j]) \
            - (self.distances[prev_i, i] + self.distances[i, next_i] + self.distances[prev_j, j] + self.distances[j, next_j])
            
    def getDeltaIntraEdges(self, e1, e2):
        return self.distances[e1[0], e2[0]] + self.distances[e1[1], e2[1]] \
            - (self.distances[e1[0], e1[1]] + self.distances[e2[0], e2[1]])

    # Inter-route - Moves changing the set of selected nodes:
        # exchange of two nodes – one selected, one not selected;
    def getDeltaInter(self, prev, curr, next, new):
        return self.distances[prev, new] + self.distances[new, next] + self.costs[new] \
            - (self.distances[prev, curr] + self.distances[curr, next] + self.costs[curr])



    def steepest_ls(self, startNode, init_sol_f, intra_route_type):
        # Generate initial solution
        current_sol = init_sol_f(startNode)

        better_found = True
        while better_found:
            # Randomly wrap around current route
            start_idx = random.choice(range(len(current_sol)))
            current_sol = current_sol[start_idx:] + current_sol[:start_idx]

            best_delta = 0
            best_route = None
            better_found = False

            # Intra-route edges
            for i in range(self.targetSolutionSize):
                edge1_idx = [i, (i + 1) % self.targetSolutionSize]
                edge1 = [current_sol[edge1_idx[0]], current_sol[edge1_idx[1]]]

                # get 10 closest vertices to the current_node
                closest = self.getNClosest(edge1[0], current_sol)
                for j in closest:
                    if (current_sol.index(j)+1 % self.targetSolutionSize) == i:
                            continue
                    j_n = (current_sol.index(j)+1) % self.targetSolutionSize
                    if j_n == 100:
                        j_n = 0

                    next_j = current_sol[j_n]
                    edge2 = [j, next_j]

                    if current_sol.index(edge1[0]) > current_sol.index(edge2[0]) or current_sol.index(edge2[1]) == 0:
                        if current_sol.index(edge1[1]) != 0:
                            temp = edge1.copy()
                            edge1 = edge2.copy()
                            edge2 = temp.copy()
                    
                    delta = self.getDeltaIntraEdges(edge1, edge2)
                    if delta < best_delta:
                        best_delta = delta
                        # Using node indicies
                        # First part, Reversed middle part, Last part
                        best_route = current_sol.copy()
                        # Best route = current solution[until current_node] + current solution reversed[from the next_node to the closest_node] \
                        # + current solution[from closest_node + 1]
                        before = len(best_route)
                        best_route = best_route[:current_sol.index(edge1[1])] + best_route[current_sol.index(edge1[1]): current_sol.index(edge2[1])][::-1] + best_route[current_sol.index(edge2[1]):]
            
            # Inter-route
            # Get a list of not seleted nodes
            not_selected = list(set(self.cities) - set(current_sol))
            for i in range(self.targetSolutionSize):
                # get 10 closest neighbours to the current_node from the nodes that havent been selected yet
                closest_not_selected = self.getNClosest(current_sol[i], not_selected)
                for node_j in closest_not_selected:                    
                    delta = self.getDeltaInter(current_sol[i - 1], current_sol[i], current_sol[(i + 1) % self.targetSolutionSize], node_j)
                    if delta < best_delta:
                        best_delta = delta
                        best_route = current_sol.copy()
                        best_route[i] = node_j
            # If improving delta was found
            if best_route:
                current_sol = best_route.copy()
                better_found = True
        return current_sol

    
    def calculateStats(self, evaluations):
        best_sol_idx = np.argmin(evaluations)
        min_result = np.amin(evaluations)
        avg_result = np.mean(evaluations)
        max_result = np.amax(evaluations)
        return f"{avg_result}({min_result} - {max_result})", best_sol_idx
    
    def writeRouteToCSV(self, route, outputPath):
        with open(outputPath, 'w') as f:
            write = csv.writer(f)
            write.writerows(np.array(route)[:, np.newaxis])

    def solve(self):
        # steepest, two-edges exchange, random ss
        algorithm = "steepest_two-edges_random"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.steepest_ls(startNode, self.random_ss, "edges"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        heuristic_solver = Solver(instancePath, outputPath)
        solver = Solver_LS(instancePath, outputPath, heuristic_solver)
        solver.solve()
        print()
