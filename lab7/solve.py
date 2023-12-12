import os
import random
from glob import glob
from time import time

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from utils import Solver
from lab2.solve import Greedy_Regret
from lab3.solve import Local_Search


class Large_Scale_Neighbourhood_search(Solver):
    def __init__(self, instancePath, outputPath, greedy_solver, ls_solver):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        
        self.greedy_solver = greedy_solver
        self.ls_solver = ls_solver

    
    def destroy(self, sol):
        # Remove subpaths of length from 1 to 5 (20-30% of nodes)
        new_sol = sol.copy()
        n_rem = np.floor(0.3 * self.targetSolutionSize)
        # Vertices removed at random or as several or one subpath
        while n_rem > 0:
            if n_rem >= 5:
                length = random.randint(1, 5)
            else:
                length = 1
            n_rem -= length
            i = random.randint(0, len(sol)-1-length)
            new_sol = new_sol[:i] + new_sol[i+length:]
        return new_sol
    
    # Repair the solution using the weighted Greedy 2-regret heuristic
    def repair(self, sol):
        new_sol = self.greedy_solver.greedy_2_regret(sol, weights = [0.5, 0.5])
        return new_sol

    def lsn(self, initial_sol, max_time_seconds, ls, init_ls = False):
        current_sol = initial_sol
        # Initial LS (in case of random intial solution)
        if init_ls:
            current_sol = self.ls_solver.steepest(current_sol, "edges")

        mainLoopRuns = 0
        current_obj = np.inf
        
        start_time = time()
        while time() - start_time < max_time_seconds:
            mainLoopRuns += 1
            
            rebuilt_sol = self.destroy(current_sol)
            rebuilt_sol = self.repair(rebuilt_sol)
            
            if ls:
                rebuilt_sol = self.ls_solver.steepest(rebuilt_sol, "edges")
            # Calculate objective
            obj = self.getTotalDistance(rebuilt_sol)
            # Assign best solution
            if obj < current_obj:
                current_obj = obj
                current_sol = rebuilt_sol
        return current_sol, current_obj, mainLoopRuns


    def solve(self):
        def _solve(algorithmName, initial_sol, max_time_seconds, ls):
            solutions, evaluations, runs = [], [], []
            # Get solutions and evaluations
            print(algorithmName)
            for _ in tqdm(range(20)):
                sol, eval, mainLoopRuns = self.lsn(initial_sol, max_time_seconds, ls)
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
        
        startNode = np.random.randint(len(self.cities))
        bestNode = self.greedy_solver.getGredilyNearestCity(startNode)
        init_route = [startNode, bestNode]
        initial_sol = self.greedy_solver.greedy_2_regret(init_route, weights = [0.5, 0.5])

        max_time_seconds = 320

        _solve("lsn_with_ls", initial_sol, max_time_seconds, True)
        _solve("lsn_without_ls", initial_sol, max_time_seconds, False)
        
        
if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        greedy_solver = Greedy_Regret(instancePath, outputPath)
        ls_solver = Local_Search(instancePath, outputPath, greedy_solver)
        
        solver = Large_Scale_Neighbourhood_search(instancePath, outputPath, greedy_solver, ls_solver)
        solver.solve()
        print()
