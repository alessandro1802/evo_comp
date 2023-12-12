import os
import random
from glob import glob
from time import time

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from lab3.solve import Local_Search


class Extended_Local_Search(Local_Search):
    def __init__(self, instancePath, outputPath):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)


    def msls(self, nIterations = 200):
        solutions, evaluations = [], []
        for _ in range(nIterations):
            startNode = np.random.randint(len(self.cities))
            init_sol = self.random_ss(startNode)
            
            solutions.append(self.steepest(init_sol, "edges"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        best_sol_idx = np.argmin(evaluations)
        return solutions[best_sol_idx], evaluations[best_sol_idx], nIterations

    
    def new_vertex(self, solution):
        new_sol = solution.copy()
        not_selected = list(set(self.cities) - set(new_sol))
        edge1 = random.randint(0, len(new_sol)-1)
        new_vertex = random.choice(not_selected)
        new_sol[edge1] = new_vertex
        return new_sol

    def perturb(self, solution):        
        new_sol1 = self.new_vertex(solution)
        new_sol2 = self.new_vertex(new_sol1)
        new_sol3 = self.new_vertex(new_sol2)
        new_sol4 = self.new_vertex(new_sol3)
        return new_sol4
        
    def ils(self, max_time_seconds):
        startNode = np.random.randint(len(self.cities))
        current_sol = self.random_ss(startNode)

        best_obj = np.inf
        best_sol = None

        lsRuns = 0
        start_time = time()
        while time() - start_time < max_time_seconds:
            # LS
            improved_solution = self.steepest(current_sol, "edges")
            current_obj = self.getTotalDistance(improved_solution)
            # LS on perturbed solution
            perturbed_solution = self.perturb(improved_solution)
            intensified_solution = self.steepest(perturbed_solution, "edges")
            new_obj = self.getTotalDistance(intensified_solution)

            lsRuns += 1
            # Acceptance criterion
            if new_obj < current_obj:
                current_sol = intensified_solution
            # Assign best solution
            if new_obj < best_obj:
                best_obj = new_obj
                best_sol = intensified_solution
        return best_sol, best_obj, lsRuns
        

    def solve(self):
        def _solve(algorithmName, algorithm, max_time_seconds = None):
            solutions, evaluations, runs = [], [], []
            # Get solutions and evaluations
            print(algorithmName)    
            for _ in tqdm(range(20)):
                if max_time_seconds:
                    sol, eval, lsRuns = algorithm(max_time_seconds)
                else:
                    sol, eval, lsRuns = algorithm()
                solutions.append(sol)
                evaluations.append(eval)
                runs.append(lsRuns)
            # Get and print stats
            stats_results, best_sol_idx, stats_runs = self.calculateStatsFormattedWithRuns(evaluations, runs)
            print("Results:", stats_results)
            print("Runs:", stats_runs)
            # Save best solution
            best_sol = solutions[best_sol_idx]
            outputPath = os.path.join(self.outputPath, algorithmName + ".csv")
            self.writeRouteToCSV(best_sol, outputPath)

        print(self.instanceName)
        _solve("msls", self.msls)
        _solve("ils", self.ils, max_time_seconds = 320)
        

if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        solver = Extended_Local_Search(instancePath, outputPath)
        solver.solve()
        print()
