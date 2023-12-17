from glob import glob
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from utils import Solver


class Similarity_Calculator(Solver):
    def __init__(self, instancePath, outputPath, solutionsPath):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.solutionsPath = os.path.join(solutionsPath, self.instanceName)

    def readSolution(self, path):
        return pd.read_csv(path, header=None).to_numpy().squeeze()
        
    def getSolutonName(self, path):
        return path.split('/')[-1].split('.')[0]

    def getEdges(self, solution):
        tmp = np.roll(solution, -1)
        return np.stack([solution, tmp]).T

    
    # Number of common edges
    def similarityEdges(self, edges1, edges2):
        return len([x for x in set(tuple(x) for x in edges1) & set(tuple(x) for x in edges2)])

    # Average number of common edges
    def similarityEdgesAvg(self, target_edges):
        sim = 0
        for solPath in (solutionsPaths := glob(os.path.join(self.solutionsPath, "*.csv"))):
            solution = self.readSolution(solPath)
            edges = self.getEdges(solution)
            sim += self.similarityEdges(target_edges, edges)
        return sim / len(solutionsPaths)
        

    # Number of common nodes
    def similarityNodes(self, solution1, solution2):
        return len(np.intersect1d(solution1, solution2))

    # Average number of common nodes
    def similarityNodesAvg(self, target_solution):
        sim = 0
        for solPath in (solutionsPaths := glob(os.path.join(self.solutionsPath, "*.csv"))):
            solution = self.readSolution(solPath)
            sim += self.similarityNodes(target_solution, solution)
        return sim / len(solutionsPaths)

    
    def getBestSolutionAndObjectives(self):
        best_obj = np.inf
        best_sol = None
        objectives = dict()
        for solPath in glob(os.path.join(self.solutionsPath, "*.csv")):
            solutionName = self.getSolutonName(solPath)
            solution = self.readSolution(solPath)
            obj = self.getTotalDistance(solution)
            objectives[solutionName] = obj
            if obj < best_obj:
                best_obj = obj
                best_sol = solution
        return best_sol, objectives
    

    def calculateSimilarities(self):
        print(self.instanceName)
        
        best_sol, objectives = self.getBestSolutionAndObjectives()
        best_edges = self.getEdges(best_sol)
        
        edges_best, edges_avg, nodes_best, nodes_avg = [], [], [], []
        for solPath in tqdm(sorted(glob(os.path.join(self.solutionsPath, "*.csv")))):
            solutionName = self.getSolutonName(solPath)
            solution = self.readSolution(solPath)
            edges = self.getEdges(solution)
            
            edges_best.append((solutionName, objectives[solutionName], self.similarityEdges(best_edges, edges)))
            edges_avg.append((solutionName, objectives[solutionName], self.similarityEdgesAvg(edges)))

            nodes_best.append((solutionName, objectives[solutionName], self.similarityNodes(best_sol, solution)))
            nodes_avg.append((solutionName, objectives[solutionName], self.similarityNodesAvg(solution)))
        
        columns = ["Solution name", "Objective", "Similarity"]
        pd.DataFrame(edges_best, columns=columns).to_csv(os.path.join(self.outputPath, 'edges_best.csv'), index=False)
        pd.DataFrame(edges_avg, columns=columns).to_csv(os.path.join(self.outputPath, 'edges_avg.csv'), index=False)
        pd.DataFrame(nodes_best, columns=columns).to_csv(os.path.join(self.outputPath, 'nodes_best.csv'), index=False)
        pd.DataFrame(nodes_avg, columns=columns).to_csv(os.path.join(self.outputPath, 'nodes_avg.csv'), index=False)


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./similarities/"
    solutionsPath = "./solutions"

    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        calculator = Similarity_Calculator(instancePath, outputPath, solutionsPath)
        calculator.calculateSimilarities()