import csv


solutionPaths = ["./lab6/solutions/TSPA/ils.csv", 
                 "./lab6/solutions/TSPB/ils.csv", 
                 "./lab6/solutions/TSPC/ils.csv", 
                 "./lab6/solutions/TSPD/ils.csv"]


for solPath in solutionPaths:
    instanceName = solPath.split('/')[-2]
    with open(solPath, 'r') as f:
        reader = csv.reader(f)
        solution = [int(node[0]) for node in list(reader)]
    print(f"{instanceName}: {', '.join(map(str, solution))}\n")
    