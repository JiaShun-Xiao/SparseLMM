# SparseLMM
the implement of “varbvs: Fast Variable Selection for Large-scale Regression” in python

### Description
Consider a linear model that relates covariates Z (Z1,...,Zm) and variables X (X1,...,Xp) to the response Y : <br>
Y = Z*W + X*U + e <br>
where W are fixed effects, U are random effects, e is random noise and e ~ N(0,Sigmae^2) <br>
Let Rj be the variable indicating whether Uj is zeros ot not. We assume the following spike-slab prior: <br>
Uj ~ N(0,Sigmau^2) if Rj = 1,<br>
Uj = 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if Rj = 0,<br>
Where Pr(Rj = 1) = pi and Pr(Rj = 0) = 1-pi. <br>
The main idea is deriving a variational expectation-maximization algorithm based on the mean-field approximation to estimate parameters {Sigmae, Sigmau, pi} and latent variales {W,U,R}. More precisely, find the "best" fully-factorized approximation to the posterior distribution of the coefficients, with spike-and-slab priors on the coefficients. By "best", we mean the approximating distribution that locally minimizes the Kullback-Leibler divergence between the approximating distribution and the exact posterior.

Here I only implement the fully-factorized variational approximation for Bayesian variable selection in linear regression.  

### Dependence
Python 3.6.8  <br>
numpy 1.16.2

### Reference 
http://pcarbo.github.io/varbvs  <br>
Carbonetto, P., Zhou, X., & Stephens, M. (2017). varbvs: Fast Variable Selection for Large-scale Regression. arXiv preprint arXiv:1709.06597.  <br>
Carbonetto, P., & Stephens, M. (2012). Scalable variational inference for Bayesian variable selection in regression, and its accuracy in genetic association studies. Bayesian analysis, 7(1), 73-108.
