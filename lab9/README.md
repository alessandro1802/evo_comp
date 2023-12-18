# Hybrid evolutionary algorithm

The goal is to implement a hybrid evolutionary algorithm and compare it with the MSLS, ILS, and LNS methods implemented in the previous assignments.

Proposed algorithm parameters:
- Elite population of 20.
- Steady state algorithm.
- Parents selected from the population with the uniform probability.
- There must be no copies of the same solution in the population (you can compare the entire solution or the value of the objective function).

Proposed recombination operators:
- Operator 1.
  We locate in the offspring all common nodes and edges and fill the rest of the solution at random.
- Operator 2.
  We choose one of the parents as the starting solution.
  We remove from this solution all edges and nodes that are not present in the other parent.
  The solution is repaired using the heuristic method in the same way as in the LNS method.
  We also test the version of the algorithm without local search after recombination (we still use local search for the initial population).

If the algorithm described above would cause premature convergence, it can be modified, e.g. additional diversification preservation mechanisms.

Additionally, another custom recombination operator can be proposed.

Experiment parameters same as ILS/LNS.

Report – analogous to before.