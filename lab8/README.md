# Global convexity (fitness-distance/similarity correlations) tests

For each instance generate 1000 random local optima obtained from random solutions using greedy local search. 
For each solution calculate its similarity to the best solution (could the best out of the 1000 local optima or an even better solution generated by another method) and average similarity to all other local optima. 
Make charts, x-axis – value of the objective function, y-axis (average) similarity.

Use (separately) two measures of similarity:
- The number of common edges.
- The number of common selected nodes.

Finally we have 16 charts: 4 instances, 2 versions of similarity (to the best or average), 2 similarity measures.

For each chart calculate also the correlation coefficient.