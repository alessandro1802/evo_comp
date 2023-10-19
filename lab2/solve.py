import csv
import os
from glob import glob
import numpy as np


class Solver():
    def __init__(instancePath, outputPath):
        instanceName, self.cities, self.costs, self.distances = self.readInstance(instancePath)
        self.outputPath = os.path.join(outputPath, instanceName)
        self.targetSolutionSize = round(len(self.cities) / 2)
        print(instanceName)

    def readInstance(self, instancePath):
        instanceName = instancePath.split('/')[-1].split('.')[0]
        # TODO
        return instanceName, cities, costs, distances
        
    