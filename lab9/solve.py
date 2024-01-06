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
from lab3.solve import Local_Search


class Hybrid_Evolutionary_alg(Solver):
    def __init__(self, instancePath, outputPath, ls_solver):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        
        self.ls_solver = ls_solver

    def generatePopulation(self, popSize = 20):
        population = [self.ls_solver.random_ss(np.random.randint(len(self.cities))) for _ in range(popSize)]
        population = [self.ls_solver.steepest(sol, "edges") for sol in population]
        return population

    # Uniformly select 2 parents
    def parentSelection(self, population):
        idx = np.random.choice(len(population), 2, replace=False)
        return population[idx[0]], population[idx[1]]

    # Locate all common nodes and edges.
    # Fill the rest of the solution at random.
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

        not_selected = list(set(self.cities) - set(child))
        for i, node in enumerate(child):
            if not node:
                rand_idx = np.random.randint(len(not_selected))
                child[i] = not_selected.pop(rand_idx)
        return child

    def evolve(self, population, max_time_seconds, ls = False):
        fitnesses = [self.getTotalDistance(sol) for sol in population]
        
        mainLoopRuns = 0
        
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

        _solve("hea_with_ls", max_time_seconds, True)
        _solve("hea_without_ls", max_time_seconds, False)
        
        
if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        ls_solver = Local_Search(instancePath, outputPath, None)
        
        solver = Hybrid_Evolutionary_alg(instancePath, outputPath, ls_solver)
        solver.solve()
        print()
