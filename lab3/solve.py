import csv
import os
import sys
import random
from glob import glob
from copy import deepcopy

import numpy as np
from tqdm import tqdm

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

    
    ### Starting solutions
    def random_ss(self, startNode):
        return [startNode] + random.sample(list(set(self.cities) - {startNode}), self.targetSolutionSize - 1)

    def best_greedy_heuristic_ss(self, startNode):
        solution = self.heuristic_solver.greedy_2_regret(startNode, weights = [0.5, 0.5])
        return solution
    
    
    ### Neighbouhoods
    # Moves changing the order of nodes within the same set of selected nodes:
        # two-nodes exchange;
    def getDeltaIntraNodes(self, prev_i, i, j, next_j):
        return self.distances[prev_i, j] - self.distances[prev_i, i] + self.distances[next_j, i] - self.distances[j, next_j]
        # two-edges exchange;
    def getDeltaIntraEdges(self, e1, e2):
        return self.distances[e1[0], e2[0]] + self.distances[e1[1], e2[1]] - self.distances[e1[0], e1[1]] - self.distances[e2[0], e2[1]] 

    # Moves changing the set of selected nodes:
        # exchange of two nodes – one selected, one not selected;
    def getDeltaInter(self, prev, curr, next, new):
        return self.distances[prev, new] - self.distances[prev, curr] + self.distances[new, next] - self.distances[curr, next] + self.costs[new] - self.costs[curr]

    
    ### Type of search
    def greedy_ls(self, startNode, init_sol_f, intra_route_type):
        # Generate initial solution
        current_sol = init_sol_f(startNode)
        
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
                if intra_route_type == "nodes":
                    # Intra nodes
                    for node_i_idx in range(len(current_sol[:-1])):
                        if node_i_idx == 0:
                            exclude = 1
                        else:
                            exclude = 0
                        
                        for node_j_idx in range(node_i_idx + 1, len(current_sol) - exclude):
                            if (next_j:= node_j_idx + 1) == len(current_sol):
                                next_j = 0
                            delta = self.getDeltaIntraNodes(current_sol[node_i_idx - 1], current_sol[node_i_idx], current_sol[node_j_idx], current_sol[next_j])
                            if delta < 0:
                                best_route = current_sol[:node_i_idx] + current_sol[node_i_idx: node_j_idx + 1][::-1]
                                if next_j != 0 :
                                    best_route += current_sol[next_j:]
                                break
                        if best_route:
                            current_sol = best_route
                            better_found = True
                            break
                else:
                    # Intra edges
                    # Generate edges with node indicies [[node1, node2], [node1_idx, node2_idx]]
                    edges = [[[node_i, current_sol[i + 1]], [i, i + 1]] for i, node_i in enumerate(current_sol[:-1])]
                    edges.append([[current_sol[-1], current_sol[0]], [len(current_sol), 0]])
                    
                    for edge1_idx, edge1 in enumerate(edges[:-2]):
                        if edge1_idx == 0:
                            exclude = -1
                        else:
                            exclude = len(edges)
                        
                        for edge2 in edges[edge1_idx + 2: exclude]:
                            # Using nodes themselves
                            delta = self.getDeltaIntraEdges(edge1[0], edge2[0])
                            if delta < 0:
                                # First part, Reversed middle part, Last part, using node indicies
                                e_idx1, e_idx2 = edge1[1], edge2[1]
                                best_route = deepcopy(current_sol)
                                best_route = best_route[:e_idx1[1]] + best_route[e_idx1[1]: e_idx2[1]][::-1] + best_route[e_idx2[1]:]
                                break
                        if best_route:                    
                            current_sol = best_route
                            better_found = True
                            break
            else:
                # Inter
                # Get a list of not seleted nodes
                not_selected = list(set(self.cities) - set(current_sol))
                for i, node_i in enumerate(current_sol[:-1]):
                    for node_j in not_selected:
                        delta = self.getDeltaInter(current_sol[i - 1], node_i, current_sol[i + 1], node_j)
                        if delta < 0:
                            best_route = deepcopy(current_sol)
                            best_route[i] = node_j
                            break
                    if best_route:
                        current_sol = best_route
                        better_found = True
                        break
        return current_sol

    def steepest_ls(self, startNode, init_sol_f, intra_route_type):
        # Generate initial solution
        current_sol = init_sol_f(startNode)

        better_found = True
        while better_found:
            # Randomly wrap around current route
            start_idx = random.choice(range(len(current_sol)))
            current_sol = current_sol[start_idx:] + current_sol[:start_idx]

            best_delta = 0
            best_route = None
            better_found = False

            if intra_route_type == "nodes":
                # Intra-route nodes
                for node_i_idx in range(len(current_sol[:-1])):
                    if node_i_idx == 0:
                        exclude = 1
                    else:
                        exclude = 0
                    
                    for node_j_idx in range(node_i_idx + 1, len(current_sol) - exclude):
                        if (next_j:= node_j_idx + 1) == len(current_sol):
                            next_j = 0
                        delta = self.getDeltaIntraNodes(current_sol[node_i_idx - 1], current_sol[node_i_idx], current_sol[node_j_idx], current_sol[next_j])
                        if delta < best_delta:
                            best_delta = delta
                            best_route = current_sol[:node_i_idx] + current_sol[node_i_idx: node_j_idx + 1][::-1]
                            if next_j != 0 :
                                best_route += current_sol[next_j:]
                            # break
                if best_route:
                    current_sol = best_route
                    better_found = True
                    # break
            else:
                # Intra-route edges
                # Generate edges with node indicies [[node1, node2], [node1_idx, node2_idx]]
                edges = [[[node_i, current_sol[i + 1]], [i, i + 1]] for i, node_i in enumerate(current_sol[:-1])]
                edges.append([[current_sol[-1], current_sol[0]], [len(current_sol), 0]])
                
                for edge1_idx, edge1 in enumerate(edges[:-2]):
                    if edge1_idx == 0:
                        exclude = -1
                    else:
                        exclude = len(edges)
                    
                    for edge2 in edges[edge1_idx + 2: exclude]:
                        # Using nodes themselves
                        delta = self.getDeltaIntraEdges(edge1[0], edge2[0])
                        if delta < best_delta:
                            best_delta = delta
                            # First part, Reversed middle part, Last part, using node indicies
                            e_idx1, e_idx2 = edge1[1], edge2[1]
                            best_route = deepcopy(current_sol)
                            best_route = best_route[:e_idx1[1]] + best_route[e_idx1[1]: e_idx2[1]][::-1] + best_route[e_idx2[1]:]
                if best_route:                    
                    current_sol = best_route
                    better_found = True
            # Inter-route
            # Get a list of not seleted nodes
            not_selected = list(set(self.cities) - set(current_sol))
            for i, node_i in enumerate(current_sol[:-1]):
                for node_j in not_selected:
                    delta = self.getDeltaInter(current_sol[i - 1], node_i, current_sol[i + 1], node_j)
                    if delta < best_delta:
                        best_delta = delta
                        best_route = deepcopy(current_sol)
                        best_route[i] = node_j
                        # break
            if best_route:
                current_sol = best_route
                better_found = True
        return current_sol

    
    def calculateStats(self, evaluations):
        best_sol_idx = np.argmin(evaluations)
        min_result = np.amin(evaluations)
        avg_result = np.mean(evaluations)
        max_result = np.amax(evaluations)
        return f"{avg_result}({min_result} - {max_result})", best_sol_idx
    
    def writeRouteToCSV(self, route, outputPath):
        with open(outputPath, 'w') as f:
            write = csv.writer(f)
            write.writerows(np.array(route)[:, np.newaxis])

    def solve(self):
        # 1. greedy, two-nodes exchange, random ss
        algorithm = "greedy_two-nodes_random"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.greedy_ls(startNode, self.random_ss, "nodes"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)
        
        # 2. greedy, two-nodes exchange, best greedy heuristic ss
        algorithm = "greedy_two-nodes_best-heuristic"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.greedy_ls(startNode, self.best_greedy_heuristic_ss, "nodes"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)
        
        # 3. greedy, two-edges exchange, random ss
        algorithm = "greedy_two-edges_random"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.greedy_ls(startNode, self.random_ss, "edges"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)

        # 4. greedy, two-edges exchange, best greedy heuristic ss
        algorithm = "greedy_two-edges_best-heuristic"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.greedy_ls(startNode, self.best_greedy_heuristic_ss, "edges"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)

        # 5. steepest, two-nodes exchange, random ss
        algorithm = "steepest_two-nodes_random"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.steepest_ls(startNode, self.random_ss, "nodes"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)

        # 6. steepest, two-nodes exchange, best greedy heuristic ss
        algorithm = "steepest_two-nodes_best-heuristic"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.steepest_ls(startNode, self.best_greedy_heuristic_ss, "nodes"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)
        
        # 7. steepest, two-edges exchange, random ss
        algorithm = "steepest_two-edges_random"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.steepest_ls(startNode, self.random_ss, "edges"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
        # Save best solution
        best_sol = solutions[best_sol_idx]
        outputPath = os.path.join(self.outputPath, algorithm + ".csv")
        self.writeRouteToCSV(best_sol, outputPath)

        # 8. steepest, two-edges exchange, best greedy heuristic ss
        algorithm = "steepest_two-edges_best-heuristic"
        solutions = []
        evaluations = []
        # Get solutions and evaluations
        print(algorithm)
        for startNode in tqdm(self.cities):
            solutions.append(self.steepest_ls(startNode, self.best_greedy_heuristic_ss, "edges"))
            evaluations.append(self.getTotalDistance(solutions[-1]))
        # Get and print stats
        stats, best_sol_idx = self.calculateStats(evaluations)
        print(stats)
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
