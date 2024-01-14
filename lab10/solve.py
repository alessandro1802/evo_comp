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
from lab7.solve import Large_Scale_Neighbourhood_search
from lab9.solve import Hybrid_Evolutionary_alg


class CustomSolver(Hybrid_Evolutionary_alg):
    def __init__(self, instancePath, outputPath, ls_solver, greedy_solver, lsn_solver):
        super().__init__(instancePath, outputPath, ls_solver)
        self.greedy_solver = greedy_solver
        self.lsn_solver = lsn_solver

    # Locate all common nodes and edges.
    # Fill the rest of solution using Greedy 2-regret wighted.
    def constructOffspring(self, x, y, destructionProb = 0.3):   
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
        # Remove Nones
        child = [node for node in child if node]
        # Randomly destroy solution with probability
        if np.random.random() < destructionProb:
            self.lsn_solver.destroy(child)
        # Fill the solution up
        child = self.greedy_solver.greedy_2_regret(child, weights = [0.5, 0.5])
        return child

    def evolve(self, population, max_time_seconds, ls = False, noImprovementLimit = 20):
        fitnesses = [self.getTotalDistance(sol) for sol in population]
        
        mainLoopRuns = 0
        noImprovement = 0
        
        start_time = time()
        while time() - start_time < max_time_seconds:
            mainLoopRuns += 1

            parent1, parent2 = self.parentSelection(population)
            child = self.constructOffspring(parent1, parent2)
            if ls:
                child = self.ls_solver.steepest(child, "edges")
                
            fitness = self.getTotalDistance(child)
            if fitness not in fitnesses:
                worstSolIdx = np.argmax(fitnesses)
                if fitness < fitnesses[worstSolIdx]:
                    population[worstSolIdx] = child
                    fitnesses[worstSolIdx] = fitness
                    noImprovement = 0
                else:
                    noImprovement += 1
                if noImprovement == noImprovementLimit:
                    break

        bestSolIdx = np.argmin(fitnesses)
        return population[bestSolIdx], fitnesses[bestSolIdx], mainLoopRuns

    
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
        greedy_solver = Greedy_Regret(instancePath, outputPath)
        ls_solver = Local_Search(instancePath, outputPath, greedy_solver)
        lsn_solver = Large_Scale_Neighbourhood_search(instancePath, outputPath, greedy_solver, ls_solver)
        
        solver = CustomSolver(instancePath, outputPath, ls_solver, greedy_solver, lsn_solver)
        solver.solve()
        print()
