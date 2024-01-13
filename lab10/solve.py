import os
import random
from glob import glob
from time import time

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from lab2.solve import Greedy_Regret
from lab3.solve import Local_Search
from lab9.solve import Hybrid_Evolutionary_alg


class CustomSolver(Hybrid_Evolutionary_alg):
    def __init__(self, instancePath, outputPath, ls_solver, greedy_solver):
        super().__init__(instancePath, outputPath, ls_solver)
        self.greedy_solver = greedy_solver

    # Locate all common nodes and edges.
    # Fill the rest of solution using Greedy 2-regret wighted.
    def constructOffspring(self, x, y):   
        child = [None for _ in range(len(x))]
        
        x_idx = {node: idx for idx, node in enumerate(x)}
        y_idx = {node: idx for idx, node in enumerate(y)}
        
        tmp = np.roll(x, -1)
        edges_x = [list(edge) for edge in np.stack([x, tmp]).T]
        
        tmp = np.roll(y, -1)
        edges_y = [list(edge) for edge in np.stack([y, tmp]).T]
        edges_y_rev = [[edge[1], edge[0]] for edge in edges_y]
        edges_y += edges_y_rev
        
        for edge in edges_x:
            if edge in edges_y:
                child[x_idx[edge[0]]] = edge[0]
                child[x_idx[edge[1]]] = edge[1]
        
        for node in x:
            if node in y:
                child[x_idx[node]] = node
        # Remvoe Nones
        child = [node for node in child if node]
        # Fill the solution up
        child = self.greedy_solver.greedy_2_regret(child, weights = [0.5, 0.5])
        return child

    
    def solve(self):
        def _solve(algorithmName, max_time_seconds, ls):
            solutions, evaluations, runs = [], [], []
            # Get solutions and evaluations
            print(algorithmName)
            for _ in tqdm(range(20)): 
                init_pop = self.generatePopulation()
                sol, eval, mainLoopRuns = self.evolve(init_pop, max_time_seconds, ls)
                solutions.append(sol)
                evaluations.append(eval)
                runs.append(mainLoopRuns)
            # Get and print stats
            stats_results, best_sol_idx, stats_runs = self.calculateStatsFormattedWithRuns(evaluations, runs)
            print("Results:", stats_results)
            print("Runs:", stats_runs)
            # Save best solution
            best_sol = solutions[best_sol_idx]
            outputPath = os.path.join(self.outputPath, algorithmName + ".csv")
            self.writeRouteToCSV(best_sol, outputPath)

        print(self.instanceName)

        max_time_seconds = 320

        _solve("custom", max_time_seconds, True)
        
        
if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        ls_solver = Local_Search(instancePath, outputPath, None)
        greedy_solver = Greedy_Regret(instancePath, outputPath)
        
        solver = CustomSolver(instancePath, outputPath, ls_solver, greedy_solver)
        solver.solve()
        print()
