import csv

import numpy as np
import pandas as pd

import os

class Solver():
    def __init__(self, instancePath, outputPath):
        self.instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, self.instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)

    def readInstance(self, instancePath: str):
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
    
    def getBestSolution(self, os_path):
        best_obj = np.inf
        best_f = None
        best_sol = None
        for file in os.listdir(os_path):
            if file.endswith(".csv"):
                file = os.path.join(os_path, file)
                with open(file, 'r') as f:
                    reader = csv.reader(f)
                    solution = [int(node[0]) for node in list(reader)]
                    obj = self.getTotalDistance(solution)
                    if obj < best_obj:
                        best_sol = solution
                        best_obj = obj
                        best_f = file
        return best_f, best_sol, best_obj
    
    def getNodesBestSim(self, best_sol, sol):
        return len(np.intersect1d(best_sol, sol))
    
    def getNodesAvgSim(self, sol1, os_path):
        sim = 0
        for file in os.listdir(os_path):
            if file.endswith(".csv"):
                file = os.path.join(os_path, file)
                with open(file, 'r') as f:
                    reader = csv.reader(f)
                    solution = [int(node[0]) for node in list(reader)]
                    sim += len(np.intersect1d(sol1, solution))
        return sim / len(os.listdir(os_path))

    def getEdgesBestSim(self, best_sol, sol):
        edges_best = np.array([[best_sol[i], best_sol[(i + 1) % self.targetSolutionSize]] for i in range(self.targetSolutionSize)])
        edges = np.array([[sol[i], sol[(i + 1) % self.targetSolutionSize]] for i in range(self.targetSolutionSize)])
        return len(np.array([x for x in set(tuple(x) for x in edges) & set(tuple(x) for x in edges_best)]))
    
    def getEdgesAvgSim(self, sol1, os_path):
        sim = 0
        for file in os.listdir(os_path):
            if file.endswith(".csv"):
                file = os.path.join(os_path, file)
                with open(file, 'r') as f:
                    reader = csv.reader(f)
                    solution = [int(node[0]) for node in list(reader)]
                    edges_sol = np.array([[sol1[i], sol1[(i + 1) % self.targetSolutionSize]] for i in range(self.targetSolutionSize)])
                    edges = np.array([[solution[i], solution[(i + 1) % self.targetSolutionSize]] for i in range(self.targetSolutionSize)])
                    sim += len(np.array([x for x in set(tuple(x) for x in edges) & set(tuple(x) for x in edges_sol)]))
        return sim / len(os.listdir(os_path))
    
    def getSimilarities(self, os_path):
        best_f, best_sol, best_obj = self.getBestSolution(os_path)
        e_b = []
        e_a = []
        n_b = []
        n_a = []

        for file in os.listdir(os_path):
            if file.endswith(".csv"):
                ff = os.path.join(os_path, file)
                with open(ff, 'r') as f:
                    reader = csv.reader(f)
                    solution = [int(node[0]) for node in list(reader)]

                    # Num of common edges, similarity to best
                    e_best = self.getEdgesBestSim(best_sol, solution)
                    e_b.append([file, e_best])
                    
                    # Num of common edges, average similarity to all
                    e_avg = self.getEdgesAvgSim(solution, os_path)
                    e_a.append([file, e_avg])

                    # Num of common nodes, similarity to best
                    n_best = self.getNodesBestSim(best_sol, solution)
                    n_b.append([file, n_best])
                    
                    # Num of common nodes, average similarity to all
                    n_avg = self.getNodesAvgSim(solution, os_path)
                    n_a.append([file, n_avg])
                print(len(e_b), len(e_a), len(n_b), len(n_a))
        pd.DataFrame(e_a).to_csv('../lab8/solutions/TSPD/edges_avg.csv', index=False)
        pd.DataFrame(n_a).to_csv('../lab8/solutions/TSPD/nodes_avg.csv', index=False)
        pd.DataFrame(e_b).to_csv('../lab8/solutions/TSPD/edges_best.csv', index=False)
        pd.DataFrame(n_b).to_csv('../lab8/solutions/TSPD/nodes_best.csv', index=False)
    
