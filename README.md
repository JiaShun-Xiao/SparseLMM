# SparseLMM
the implement of “varbvs: Fast Variable Selection for Large-scale Regression” in pytho

### Description
$Y = X*\beta$

Compute fully-factorized variational approximation for Bayesian variable selection in linear (family = 'gaussian') or logistic regression (family = 'binomial'). More precisely, find the "best" fully-factorized approximation to the posterior distribution of the coefficients, with spike-and-slab priors on the coefficients. By "best", we mean the approximating distribution that locally minimizes the Kullback-Leibler divergence between the approximating distribution and the exact posterior.
