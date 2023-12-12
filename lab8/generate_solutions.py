import os
from glob import glob

import numpy as np
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_dir)
from lab3.solve import Local_Search


class Generator():
    def __init__(self, instancePath, outputPath, ls_solver):
        self.instanceName = ls_solver.instanceName
        self.outputPath = os.path.join(outputPath, self.instanceName)        
        self.ls_solver = ls_solver


    def generate(self):
        print(self.instanceName)
        for i in tqdm(range(1000)):
            # Generate initial solution
            startNode = np.random.randint(len(ls_solver.cities))
            init_sol = self.ls_solver.random_ss(startNode)
            # Run algorithm 
            solution = self.ls_solver.greedy(init_sol, "edges")
            # Save solution
            outputPath = os.path.join(self.outputPath, f"{i + 1}.csv")
            ls_solver.writeRouteToCSV(solution, outputPath)


if __name__ == "__main__":
    instancesPath = "../instances/"
    outputPath = "./solutions/"

    np.random.seed(123)
    for instancePath in sorted(glob(os.path.join(instancesPath, "*.csv"))):
        ls_solver = Local_Search(instancePath, outputPath, None)
        
        generator = Generator(instancePath, outputPath, ls_solver)
        generator.generate()
        print()
