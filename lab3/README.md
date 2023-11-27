# Local search

Implement local search. Local search should use deltas (changes) of the objective function. Do not
use trivial implementation with neighbor solution constructed explicitly and the objective function
calculated from the scratch. On the other hand, do not use more advanced techniques like deltas
from previous iterations or candidate moves.

**Type of local search** : Implement both steepest and greedy version of local search. In greedy version
the neighborhood should be browsed in random/randomized order. In the report please describe
what kind of randomization was used.

**Type of neighborhood:** We will need two kinds of moves: intra-route moves – moves changing the
order of nodes within the same set of selected nodes, inter-route moves – moves changing the set of
selected nodes. As inter-route move use exchange of two nodes – one selected one not selected. For
intra-route moves use two options:

- two-nodes exchange,
- two-edges exchange.

Note that the whole neighborhood is composed of moves of two kinds inter- and intra-route moves.
In the steepest version we should select the best move from the whole neighborhood composed of moves of two types. In the greedy version ideally we should browse moves of two kinds in a random order. If we use a simpler randomization, we cannot, for example, start always from moves of a given kind (e.g. always from intra-route moves).

**Type of starting solutions:** Use two options:

- Random starting solutions.
- The best greedy construction heuristic (including regret heuristics) from the previous assignments with a random starting node.

Summarizing we have 8 combinations based on three binary options: type of local search, type of neighborhood, type of starting solutions.

**Computational experiment:** Run each of the eight methods 200 times. For random starting solutions use 200 randomly generated solutions. For greedy starting solutions use each of the 200 nodes as the starting node for the greedy heuristic.

**Reporting results:** Use table with the following outline

|          | Instance 1     | Instance 2     |
|----------|----------------|----------------|
| Method 1 | av (min – max) | av (min – max) |
| Method 2 | av (min – max) | av (min – max) |
| ...      |                |                |

Include also results for all previous methods.

Report two tables one for the values of the objective function and for the running times.

The outline of the report as previously.