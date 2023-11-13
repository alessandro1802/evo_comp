import csv
import os
import sys
import random
from glob import glob
from copy import deepcopy
from functools import partial

import numpy as np
from tqdm import tqdm


class Solver_LS():
    def __init__(self, instancePath, outputPath):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        nCandidates = 10
        self.candidates = self.getCandidates(nCandidates)
        
        print(self.instanceName)
        
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

    def getCandidates(self, nCandidates):
        candidates  = dict()
        for city in self.cities:
            # Get distances
            distances = list(self.distances[city, :])
            # Store the indices alongside the distances
            distances = [(i, distances[i]) for i in range(len(distances))]
            # Remove the distance to itself
            distances.pop(city)
            # Sort it (closest first)
            distances = sorted(distances, key = lambda x: x[1])[:nCandidates]
            # Store candidates
            candidates[city] = [distances[i][0] for i in range(len(distances))]
        return candidates
        
    def getTotalEdgeCost(self, node_j, node_i):
        return self.distances[node_i, node_j] + self.costs[node_j]
        
    def getTotalDistance(self, route):
        total = 0
        for i in range(len(route) - 1):
            total += self.getTotalEdgeCost(route[i], route[i + 1])
        # Return to the starting city
        total += self.getTotalEdgeCost(route[-1], route[0])
        return total

    
    ### Starting solutions
    def random_ss(self, startNode):
        return [startNode] + random.sample(list(set(self.cities) - {startNode}), self.targetSolutionSize - 1)

    
    ### Neighbouhoods
    # Intra-route - Moves changing the order of nodes within the same set of selected nodes:
    #     two-edges exchange;
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
            best_delta = 0
            best_route = None
            better_found = False

            # Intra-route edges
            for i in range(self.targetSolutionSize):
                current_edge1_idx = [i, (i + 1) % self.targetSolutionSize]
                current_edge1 = [current_sol[current_edge1_idx[0]], current_sol[current_edge1_idx[1]]]
                
                for candidate in self.candidates[current_edge1[0]]:
                    # Skip candidates not in the current colution
                    if candidate not in current_sol:
                        continue
                        
                    j = current_sol.index(candidate)
                    edge2_idx = [j, (j + 1) % self.targetSolutionSize]
                    edge2 = [candidate, current_sol[edge2_idx[1]]]

                    edge1_idx = current_edge1_idx
                    edge1 = current_edge1
                    if i > j:
                        # Swap edges order
                        edge1_idx, edge2_idx = edge2_idx, edge1_idx
                        edge1, edge2 = edge2, edge1
                        j = edge2_idx[0]
                    # Skip directly preceding cadidates 
                    if edge2_idx[1] == i:
                        continue
                    
                    delta = self.getDeltaIntraEdges(edge1, edge2)
                    if delta < best_delta:
                        best_delta = delta
                        # Using node indicies
                        best_route = deepcopy(current_sol)
                        # First part, Reversed middle part, Last part
                        best_route = best_route[:edge1_idx[1]] + best_route[edge1_idx[1]: (j + 1)][::-1] + best_route[(j + 1):]
            # Inter-route
            # Get a list of not seleted nodes
            not_selected = list(set(self.cities) - set(current_sol))
            for i in range(self.targetSolutionSize):
                
                for candidate in self.candidates[current_sol[i]]:
                    # Skip candidates that are in the current colution
                    if candidate in current_sol:
                        continue
                    # Current with candidate become connected, swap cadidate with either Previuos or Next
                    # Previous
                    delta = self.getDeltaInter(current_sol[i - 2], current_sol[i - 1], current_sol[i], 
                                               candidate)
                    if delta < best_delta:
                        best_delta = delta
                        best_route = deepcopy(current_sol)
                        best_route[i - 1] = candidate
                    # Next
                    next_i = (i + 1) % self.targetSolutionSize
                    delta = self.getDeltaInter(current_sol[i], current_sol[next_i], current_sol[(i + 2) % self.targetSolutionSize], 
                                               candidate)
                    if delta < best_delta:
                        best_delta = delta
                        best_route = deepcopy(current_sol)
                        best_route[next_i] = candidate
                # If improving delta was found
            if best_route:               
                current_sol = deepcopy(best_route)
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
        algorithm = "steepest_cm_two-edges_random"
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
        solver = Solver_LS(instancePath, outputPath)
        solver.solve()
        print()
