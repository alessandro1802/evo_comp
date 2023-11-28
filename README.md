# Evolutionary Computation

## Set-up
```shell
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Problem description
We are given three columns of integers with a row for each node (`intances/`). The first two columns contain *x* and *y* coordinates of the node positions in a plane. The third column contains node costs. The goal is to select exactly 50% of the nodes (if the number of nodes is odd we round the number of nodes to be selected up) and form a Hamiltonian cycle (closed path) through this set of nodes such that the sum of the total length of the path plus the total cost of the selected nodes is minimized.

The distances between nodes are calculated as Euclidean distances rounded **mathematically** to integer values. The distance matrix should be calculated just after reading an instance and then only the distance matrix (no nodes coordinates) should be accessed by optimization methods to allow instances defined only by distance matrices.