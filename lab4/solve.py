import os
import random
from glob import glob
from copy import deepcopy

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from lab3.solve import Local_Search


class Candidate_Local_Search(Local_Search):
    def __init__(self, instancePath, outputPath, nCandidates):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        self.candidates = self.getCandidates(nCandidates)
        

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

    def steepest(self, init_sol, intra_route_type):
        current_sol = init_sol

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


    def solve(self):
        def _solve(algorithmName, init_sol_f, algorithm, intra_route_type):
            solutions = []
            evaluations = []
            # Get solutions and evaluations
            print(algorithmName)
            for startNode in tqdm(self.cities):
                # Generate initial solution
                init_sol = init_sol_f(startNode)
                
                solutions.append(algorithm(init_sol, intra_route_type))
                evaluations.append(self.getTotalDistance(solutions[-1]))
            # Get and print stats
            stats, best_sol_idx = self.calculateStatsFormatted(evaluations)
            print(stats)
            # Save best solution
            best_sol = solutions[best_sol_idx]
            outputPath = os.path.join(self.outputPath, algorithmName + ".csv")
            self.writeRouteToCSV(best_sol, outputPath)
            
        print(self.instanceName)
        # steepest, two-edges exchange, random ss
        _solve("steepest_cm_two-edges_random", self.random_ss, self.steepest, "edges")


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        solver = Candidate_Local_Search(instancePath, outputPath, 10)
        solver.solve()
        print()
