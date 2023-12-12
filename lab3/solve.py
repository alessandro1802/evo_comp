import os
import random
from glob import glob
from copy import deepcopy

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from utils import Solver
from lab2.solve import Greedy_Regret


class Local_Search(Solver):
    def __init__(self, instancePath, outputPath, heuristic_solver):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)

        self.heuristic_solver = heuristic_solver

    
    ### Starting solutions
    def random_ss(self, startNode):
        return [startNode] + random.sample(list(set(self.cities) - {startNode}), self.targetSolutionSize - 1)

    def best_greedy_heuristic_ss(self, startNode):
        bestNode = self.heuristic_solver.getGredilyNearestCity(startNode)
        init_route = [startNode, bestNode]
        
        solution = self.heuristic_solver.greedy_2_regret(init_route, weights = [0.5, 0.5])
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

    
    ### Type of search
    def greedy(self, init_sol, intra_route_type):
        current_sol = init_sol
        
        better_found = True
        while better_found:
            # Randomly choose neighbourhood
            if random.random() > 0.5:
                neighbourhood_idx = 1
            else:
                neighbourhood_idx = 0
            # Randomly wrap around current route
            start_idx = random.choice(range(len(current_sol)))
            current_sol = current_sol[start_idx:] + current_sol[:start_idx]
        
            best_route = None
            better_found = False

            if neighbourhood_idx == 0:
                # Intra-route nodes
                if intra_route_type == "nodes":
                    for i in range(self.targetSolutionSize - 1):
                        for j in range(i + 1, self.targetSolutionSize):
                            delta = self.getDeltaIntraNodes(current_sol[i - 1], current_sol[i], current_sol[i + 1], 
                                                            current_sol[j - 1], current_sol[j], current_sol[(j + 1) % self.targetSolutionSize])
                            if delta < 0:
                                best_route = deepcopy(current_sol)
                                best_route[i], best_route[j] = best_route[j], best_route[i]
                                break
                        if best_route:
                            current_sol = best_route
                            better_found = True
                            break
                # Intra-route edges
                else:
                    for i in range(self.targetSolutionSize):
                        edge1_idx = [i, (i + 1) % self.targetSolutionSize]
                        edge1 = [current_sol[edge1_idx[0]], current_sol[edge1_idx[1]]]
                        for j in range(i + 2, self.targetSolutionSize):
                            if (next_j := (j + 1) % self.targetSolutionSize) == i:
                                    continue
                            edge2_idx = [j, next_j]
                            edge2 = [current_sol[edge2_idx[0]], current_sol[edge2_idx[1]]]
                            # Using nodes themselves
                            delta = self.getDeltaIntraEdges(edge1, edge2)
                            if delta < 0:
                                # Using node indicies
                                # First part, Reversed middle part, Last part
                                best_route = deepcopy(current_sol)
                                best_route = best_route[:edge1_idx[1]] + best_route[edge1_idx[1]: (j + 1)][::-1] + best_route[(j + 1):]
                                break
                        if best_route:                    
                            current_sol = best_route
                            better_found = True
                            break
            # Inter-route
            else:
                # Get a list of not seleted nodes
                not_selected = list(set(self.cities) - set(current_sol))
                for i in range(self.targetSolutionSize):
                    for node_j in not_selected:                    
                        delta = self.getDeltaInter(current_sol[i - 1], current_sol[i], current_sol[(i + 1) % self.targetSolutionSize], node_j)
                        if delta < 0:
                            best_route = deepcopy(current_sol)
                            best_route[i] = node_j
                            break
                    if best_route:
                        current_sol = best_route
                        better_found = True
                        break
        return current_sol

    def steepest(self, init_sol, intra_route_type):
        current_sol = init_sol

        better_found = True
        while better_found:
            # Randomly wrap around current route
            start_idx = random.choice(range(len(current_sol)))
            current_sol = current_sol[start_idx:] + current_sol[:start_idx]

            best_delta = 0
            best_route = None
            better_found = False

            # Intra-route nodes
            if intra_route_type == "nodes":
                for i in range(self.targetSolutionSize - 1):
                    for j in range(i + 1, self.targetSolutionSize):
                        delta = self.getDeltaIntraNodes(current_sol[i - 1], current_sol[i], current_sol[i + 1], 
                                                        current_sol[j - 1], current_sol[j], current_sol[(j + 1) % self.targetSolutionSize])
                        if delta < best_delta:
                            best_delta = delta
                            best_route = deepcopy(current_sol)
                            best_route[i], best_route[j] = best_route[j], best_route[i]
            # Intra-route edges
            else:
                for i in range(self.targetSolutionSize):
                    edge1_idx = [i, (i + 1) % self.targetSolutionSize]
                    edge1 = [current_sol[edge1_idx[0]], current_sol[edge1_idx[1]]]
                    for j in range(i + 2, self.targetSolutionSize):
                        if (next_j := (j + 1) % self.targetSolutionSize) == i:
                                continue
                        edge2_idx = [j, next_j]
                        edge2 = [current_sol[edge2_idx[0]], current_sol[edge2_idx[1]]]
                        # Using nodes themselves
                        delta = self.getDeltaIntraEdges(edge1, edge2)
                        if delta < best_delta:
                            best_delta = delta
                            # Using node indicies
                            # First part, Reversed middle part, Last part
                            best_route = deepcopy(current_sol)
                            best_route = best_route[:edge1_idx[1]] + best_route[edge1_idx[1]: (j + 1)][::-1] + best_route[(j + 1):]
            # Inter-route
            # Get a list of not seleted nodes
            not_selected = list(set(self.cities) - set(current_sol))
            for i in range(self.targetSolutionSize):
                for node_j in not_selected:                    
                    delta = self.getDeltaInter(current_sol[i - 1], current_sol[i], current_sol[(i + 1) % self.targetSolutionSize], node_j)
                    if delta < best_delta:
                        best_delta = delta
                        best_route = deepcopy(current_sol)
                        best_route[i] = node_j
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
        # 1. greedy, two-nodes exchange, random ss
        _solve("greedy_two-nodes_random", self.random_ss, self.greedy, "nodes")
        # 2. greedy, two-nodes exchange, best greedy heuristic ss
        _solve("greedy_two-nodes_best-heuristic", self.best_greedy_heuristic_ss, self.greedy, "nodes")
        # 3. greedy, two-edges exchange, random ss
        _solve("greedy_two-edges_random", self.random_ss, self.greedy, "edges")
        # 4. greedy, two-edges exchange, best greedy heuristic ss
        _solve("greedy_two-edges_best-heuristic", self.best_greedy_heuristic_ss, self.greedy, "edges")
        # 5. steepest, two-nodes exchange, random ss
        _solve("steepest_two-nodes_random", self.random_ss, self.steepest, "nodes")
        # 6. steepest, two-nodes exchange, best greedy heuristic ss
        _solve("steepest_two-nodes_best-heuristic", self.best_greedy_heuristic_ss, self.steepest, "nodes")
        # 7. steepest, two-edges exchange, random ss
        _solve("steepest_two-edges_random", self.random_ss, self.steepest, "edges")
        # 8. steepest, two-edges exchange, best greedy heuristic ss
        _solve("steepest_two-edges_best-heuristic", self.best_greedy_heuristic_ss, self.steepest, "edges")


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        heuristic_solver = Greedy_Regret(instancePath, outputPath)
        solver = Local_Search(instancePath, outputPath, heuristic_solver)
        solver.solve()
        print()
