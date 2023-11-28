# The use of move evaluations (deltas) from previous iterations in local search

The goal of the task is to improve the time efficiency of the steepest local search with the use move evaluations (deltas) from previous iterations (list of improving moves) using the neighborhood, which turned out to be the best in assignment 3. Both inter-route and intra-route moves should be included in the list. In the case of inter-route moves of the exchange of two edges, you should carefully read the description of the lectures on the traveling salesman problem.

This mechanism should be used separately from candidate moves. Optionally, you can try to implement them both together.

As starting solutions use random solutions.

As baseline report also results of the steepest local search with random starting solutions without these mechanisms.

**Computational experiment:** Run each of the methods 200 times.

**Reporting results:** Use tables as in the previous assignment.

The outline of the report as previously.