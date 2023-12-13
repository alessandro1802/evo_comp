import csv


solutionPaths = ["../lab7/solutions/TSPA/lsn_with_ls.csv", 
                 "../lab7/solutions/TSPB/lsn_with_ls.csv", 
                 "../lab7/solutions/TSPC/lsn_with_ls.csv", 
                 "../lab6/solutions/TSPD/ils.csv"]


for solPath in solutionPaths:
    instanceName = solPath.split('/')[-2]
    with open(solPath, 'r') as f:
        reader = csv.reader(f)
        solution = [int(node[0]) for node in list(reader)]
    print(f"{instanceName}: {', '.join(map(str, solution))}\n")
    