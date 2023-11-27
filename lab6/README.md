# Multiple start local search (MSLS) and iterated local search (ILS)

The goal of the task is to implement two simple extensions of local search:

- Multiple start local search (MSLS) – we will use steepest local search starting from random
solutions.
- Iterated local search (ILS).

You can use basic steepest local search with edge exchange as intra-route move or version with list of
moves (if it was implemented successfully).

The perturbation for ILS should be designed by you and precisely described in the report. You should
aim at obtaining better results with ILS than with MSLS.

**Computational experiment:** Run each of the methods (MSLS and ILS) 20 times for each instance. In
MSLS perform 200 iterations of basic local search. For ILS as the stopping condition use the average
running time of MSLS. For ILS report also the number of runs of basic local search. For ILS as the
starting solution (one for each run of ILS) use random solution.

**Reporting results:** Use tables as in the previous assignment. For ILS add a table with the number of
runs of basic LS. Report the best solutions for each instance as a lists of nodes.

The outline of the report as previously.