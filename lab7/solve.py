import csv
import os
import random
from glob import glob
from copy import deepcopy
from time import time

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from lab2.solve import Solver


class Solver_LS():
    def __init__(self, instancePath, outputPath, heuristic_solver):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        # print(self.instanceName)
        self.heuristic_solver = heuristic_solver

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

    
    def steepest_ls(self, init_sol):
        current_sol = init_sol

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

    
    #TODO
    def destroy(self, solution):
        pass

    #TODO
    def repair(self, solution):
        pass
    

    def lsns(self, initial_sol, max_time_seconds, ls, init_ls = False):
        current_sol = initial_sol
        # Initial LS (in case of random intial solution)
        if init_ls:
            current_sol = self.steepest_ls(current_sol)

        mainLoopRuns = 0
        best_obj = np.inf
        best_sol = None
        
        start_time = time()
        while time() - start_time < max_time_seconds:
            mainLoopRuns += 1
            
            rebuilt_sol = self.destroy(current_sol)
            rebuilt_sol = self.repair(rebuilt_sol)
            
            if ls:
                rebuilt_sol = self.steepest_ls(rebuilt_sol)
            # Calculate objective
            obj = self.getTotalDistance(rebuilt_sol)
            # Assign best solution
            if obj < best_obj:
                best_obj = obj
                best_sol = rebuilt_sol
        return best_sol, best_obj, mainLoopRuns

    
    def calculateStats(self, evaluations, runs):
        best_sol_idx = np.argmin(evaluations)
        min_result = np.amin(evaluations)
        avg_result = np.mean(evaluations)
        max_result = np.amax(evaluations)
        
        min_runs = np.amin(runs)
        avg_runs = np.mean(runs)
        max_runs = np.amax(runs)
        return f"{avg_result} ({min_result} - {max_result})", best_sol_idx, f"{avg_runs} ({min_runs} - {max_runs})"
    
    def writeRouteToCSV(self, route, outputPath):
        with open(outputPath, 'w') as f:
            write = csv.writer(f)
            write.writerows(np.array(route)[:, np.newaxis])
            

    def solve(self):
        max_time_seconds = 320
        
        startNode = np.random.randint(len(self.cities))
        initial_sol = self.heuristic_solver.greedy_2_regret(startNode, weights = [0.5, 0.5])
        
        algorithm = "lsns_with_ls"
        solutions, evaluations, runs = [], [], []
        # Get solutions and evaluations
        print(algorithm)
        for _ in tqdm(range(20)):
            sol, eval, lsRuns = self.lsns(initial_sol, max_time_seconds, True)
            solutions.append(sol)
            evaluations.append(eval)
            runs.append(mainLoopRuns)
        # Get and print stats
        stats_results, best_sol_idx, stats_runs = self.calculateStats(evaluations, runs)
        print("Results:", stats_results)
        print("Runs:", stats_runs)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)

        algorithm = "lsns_without_ls"
        solutions, evaluations, runs = [], [], []
        # Get solutions and evaluations
        print(algorithm)
        for _ in tqdm(range(20)):
            sol, eval, lsRuns = self.lsns(initial_sol, max_time_seconds, False)
            solutions.append(sol)
            evaluations.append(eval)
            runs.append(mainLoopRuns)
        # Get and print stats
        stats_results, best_sol_idx, stats_runs = self.calculateStats(evaluations, runs)
        print("Results:", stats_results)
        print("Runs:", stats_runs)
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