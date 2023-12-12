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


class Dynamic_Local_Search(Local_Search):
    def __init__(self, instancePath, outputPath):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)


    def steepest(self, init_sol, intra_route_type):
        current_sol = init_sol

        evaluatedMovesIntra = dict()
        evaluatedMovesInter = dict()
        better_found = True
        while better_found:
            best_delta = 0
            best_route = None
            better_found = False
            improvingMovesIntra = dict()
            improvingMovesInter = dict()
            # Intra-route edges
            for i in range(self.targetSolutionSize):
                edge1_idx = [i, (i + 1) % self.targetSolutionSize]
                edge1 = (current_sol[edge1_idx[0]], current_sol[edge1_idx[1]])
                for j in range(i + 2, self.targetSolutionSize):
                    if (next_j := (j + 1) % self.targetSolutionSize) == i:
                        continue
                    edge2_idx = [j, next_j]
                    edge2 = (current_sol[edge2_idx[0]], current_sol[edge2_idx[1]])
                
                    move = (edge1, edge2)
                    if move in evaluatedMovesIntra.keys():
                        delta = evaluatedMovesIntra[move]
                    else:
                        # Evaluate all new moves
                        delta = self.getDeltaIntraEdges(edge1, edge2)
                        evaluatedMovesIntra[move] = delta
                    # Store improving moves
                    if delta < 0:
                        improvingMovesIntra[move] = delta
                        
            if improvingMovesIntra:
                # Sort improving moves by delta
                improvingMovesIntra = dict(sorted(improvingMovesIntra.items(), key=lambda item: item[1]))
                # Select best
                move, delta = list(improvingMovesIntra.keys())[0], list(improvingMovesIntra.values())[0]
                if delta < best_delta:
                    best_delta = delta
                    
                    edge1, edge2 = move
                    next_i = current_sol.index(edge1[1])
                    next_j = current_sol.index(edge2[1])
                    if next_j == 0:
                        next_j = self.targetSolutionSize
                    
                    best_route = deepcopy(current_sol)
                    # First part, Reversed middle part, Last part
                    best_route = best_route[:next_i] + best_route[next_i: next_j][::-1] + best_route[next_j:]
            # Inter-route
            # Get a list of not seleted nodes
            not_selected = list(set(self.cities) - set(current_sol))
            for i in range(self.targetSolutionSize):
                for node_j in not_selected:                    
                    move = (current_sol[i - 1], current_sol[i], current_sol[(i + 1) % self.targetSolutionSize], node_j)
                    if move in evaluatedMovesInter.keys():
                        delta = evaluatedMovesInter[move]
                    else:
                        # Evaluate new moves
                        delta = self.getDeltaInter(current_sol[i - 1], current_sol[i], current_sol[(i + 1) % self.targetSolutionSize], node_j)
                        evaluatedMovesInter[move] = delta
                    # Store improving moves
                    if delta < 0:
                        improvingMovesInter[move] = delta
            if improvingMovesInter:
                improvingMovesInter = dict(sorted(improvingMovesInter.items(), key=lambda item: item[1]))
                # Select best
                move, delta = list(improvingMovesInter.keys())[0], list(improvingMovesInter.values())[0]
                if delta < best_delta:
                    best_delta = delta
                    
                    node_i = move[1]
                    node_j = move[-1]
                    node_i_idx = current_sol.index(node_i)
                    
                    best_route = deepcopy(current_sol)
                    best_route[node_i_idx] = node_j
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
        _solve("steepest_dynamic_two-edges_random", self.random_ss, self.steepest, "edges")
        

if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        solver = Dynamic_Local_Search(instancePath, outputPath)
        solver.solve()
        print()
