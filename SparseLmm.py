import numpy as np
import logging

logging.basicConfig(filename='spaselmm.log', level=logging.DEBUG)

###
# Input Arguments
###
# X: n x p input matrix, where n is the number of samples, and p is the number of variables. X cannot be sparse.
# Z: n x m covariate data matrix
# y: Vector of length n containing phenotype

###
# Output Arguments
###
# w: normalized weights compute from logw.
# alpha: variational estimates of posterior inclusion probs.
# pip: "Averaged" posterior inclusion probabilities.
# mu:  variational estimates of posterior mean coefficients.
# mu_cov: posterior estimates of coefficients for covariates.
# heri: estimated heritability
# logodd: approximate marginal log-likelihood for each setting of hyperparameters.


def varbvs(X,y,Z=None,tol=1e-5,maxiter=1e5,verbose=False):
    logging.info("X,y,Z shape: {},{},{}".format(str(X.shape),str(y.shape),str(Z.shape)))
    X = X.astype(np.float32)
    n,p = X.shape
    if Z.all() != None:
        Z = Z.astype(np.float64)
        Z = np.concatenate((np.ones((n,1)),Z),axis=1)
    else:
        Z = np.ones((n,1))
    ncov = Z.shape[1]-1
    y = y.astype(np.float32)
    ## 1. PROCESS OPTIONS
    ## initial sigma, sa, logodds
    ns = 20
    sigma = np.array([y.var() for _ in range(ns)])
    sa = np.array([1. for _ in range(ns)])
    logodds = np.linspace(-np.log10(3500),-1,ns)
    ## initial estimates of variational parameter alpha.
    logging.info("initial parameter")
    alpha = np.random.rand(p,ns)
    alpha = alpha/alpha.sum(axis=0)
    mu = np.random.randn(p,ns)
    ## 2.PREPROCESSING STEPS
    ## Adjust the genotypes and phenotypes
    if ncov == 0:
        X = X - X.mean(axis=0)
        y = y - y.mean()
    else:
        ZTZ_ = np.linalg.inv(Z.T.dot(Z))
        SZy = ZTZ_.dot(Z.T).dot(y)
        SZX = ZTZ_.dot(Z.T).dot(X)
        X = X - Z.dot(SZX)
        y = y - Z.dot(SZy)
    d = np.linalg.norm(X,axis=0)**2
    xy = (y.T.dot(X)).T
    ## 3. FIT BAYESIAN VARIABLE SELECTION MODE
    logging.info("fit bayesian variable selection model")
    logging.info("number of tried log odds: {}".format(str(ns)))
    logw = np.zeros(ns) # log likelihood for each hyperparameter setting (logw)
    s = np.zeros((p,ns)) # variances of the regression coefficients (s)
    mu_cov = np.zeros((ncov+1,ns)) # posterior mean estimates of the coefficients for the covariates
    ## 4. find a good initialization for the variational parameters
    logging.info("find a good initialization for the variational parameters")
    for i in range(ns):
        logging.info("itreration for {} log odds".format(str(i)))
        logw[i], sigma[i], sa[i], alpha[:,i], mu[:,i], s[:,i], mu_cov[:,i] = \
            outerloop(d,xy,X,Z,y,SZy,SZX,sigma[i],sa[i],logodds[i],alpha[:,i],mu[:,i],tol,maxiter)
    ## Choose an initialization common to all the runs of the coordinate ascent algorithm
    i = np.argmax(logw)
    alpha = np.repeat(alpha[:,i], ns).reshape(p,ns)
    mu = np.repeat(mu[:,i], ns).reshape(p,ns)
    sigma = np.full(ns,sigma[i])
    sa = np.full(ns,sa[i])
    ## 5. optimazition
    logging.info("Main loop for computing a variational approximation")
    for i in range(ns):
        logging.info("itreration for {} log odds".format(str(i)))
        logw[i], sigma[i], sa[i], alpha[:,i], mu[:,i], s[:,i], mu_cov[:,i] = \
            outerloop(d,xy,X,Z,y,SZy,SZX,sigma[i],sa[i],logodds[i],alpha[:,i],mu[:,i],tol,maxiter)
    ## 6. CREATE FINAL OUTPUT
    w = normalizelogw(logw)
    pip = alpha.dot(w)
    #beta = mu.dot(w)
    #cov_beta = mu_cov.dot(w)
    sigma = sigma.dot(w)
    sa = sa.dot(w)
    #print(cov_beta)
    #print(pip,beta)
    #print(np.exp(logw.max()))
    return w,alpha,pip,mu,mu_cov,sa/(sigma+sa),np.exp(logw.max())

def outerloop(d,xy,X,Z,y,SZy,SZX,sigma,sa,logodds,alpha,mu,tol,maxiter):
    n,p = X.shape
    if np.isscalar(logodds):
        logodds = np.full(p,logodds)
    #print(logodds)
    logw,err,sigma,sa,alpha,mu,s = varbvsnorm(d,xy,X,Z,y,sigma,sa,logodds,alpha,mu,tol,maxiter)
    (sign, logdet) = np.linalg.slogdet(Z.T.dot(Z))
    logw = logw - sign*logdet/2
    mu_cov = SZy - SZX.dot(alpha*mu)
    logw = logw[-1]
    return logw, sigma, sa, alpha, mu, s, mu_cov

def varbvsnorm(d,xy,X,Z,y,sigma,sa,logodds,alpha,mu,tol,maxiter):
    maxiter = int(maxiter)
    X = X.astype(np.float32)
    n,p = X.shape
    Xr = X.dot(alpha*mu)
    #print(sigma,sa,logodds,alpha,mu,tol,maxiter,sa0,n0)
    s = (sa*sigma)/(sa*d+1)
    logw = np.zeros(maxiter)
    err = np.zeros(maxiter)
    ## main loop
    for it in range(maxiter):
        alpha0 = alpha
        mu0    = mu
        s0     = s
        sigma0 = sigma
        sa0    = sa
        ## COMPUTE CURRENT VARIATIONAL LOWER BOUND
        logw0 = int_linear(Xr,d,y,sigma,alpha,mu,s)
        ## UPDATE VARIATIONAL APPROXIMATION
        #print(alpha,mu,Xr)
        if it%2 == 0:
            order = range(p)
        else:
            order = range(p-1,-1,-1)
        alpha,mu,Xr = varbvsnormupdate(X,sigma,sa,logodds,xy,d,alpha,mu,Xr,order)
        #print(alpha,mu)
        ## COMPUTE UPDATED VARIATIONAL LOWER BOUND
        logw[it] = int_linear(Xr,d,y,sigma,alpha,mu,s)
        ## UPDATE RESIDUAL VARIANCE
        betavar = alpha*(sigma+(mu**2))-(alpha*mu)**2
        sigma = (np.linalg.norm(y - Xr)**2+d.dot(betavar)+alpha.dot(s+mu**2)/sa)/(n+alpha.sum())
        s = (sa*sigma)/(sa*d+1)
        #sa = (sa0*n0+alpha.dot(s+mu**2))/(n0+sigma*alpha.sum())
        sa = (alpha.dot(s+mu**2))/(sigma*alpha.sum())
        s = (sa*sigma)/(sa*d+1)
        ## check convergence
        err[it] = np.absolute(alpha - alpha0).max()
        #print(err[it],logw[it])
        if logw[it] < logw0:
            logw[it] = logw0
            err[it]  = 0
            sigma      = sigma0
            sa         = sa0
            alpha      = alpha0
            mu         = mu0
            s          = s0
            break
        elif err[it] < tol:
            break
    logw = logw[:it+1]
    err = err[:it+1]
    return logw,err,sigma,sa,alpha,mu,s
        
def int_linear(Xr, d, y, sigma, alpha, mu, s):
    n = y.shape[0]
    betavar = alpha*(sigma+(mu**2))-(alpha*mu)**2
    return (-n/2)*np.log(2*np.pi*sigma) - np.linalg.norm(y - Xr)**2/(2*sigma) - d.T.dot(betavar)/(2*sigma)

def varbvsnormupdate(X,sigma,sa,logodds,xy,d,alpha,mu,Xr,order):
    n,p = X.shape
    #print(mu)
    s = np.zeros(p)
    for i in order:
        s[i] = (sa*sigma)/(sa*d[i]+1)
        r = alpha[i]*mu[i]
        mu[i] = (s[i]/sigma)*(xy[i]+d[i]*r-X[:,i].T.dot(Xr))
        #print(mu**2/s)
        SSR = mu[i]**2/s[i]
        alpha_tmp = logodds[i]+(np.log(s[i]/(sigma*sa))+SSR)/2
        alpha[i] = 1/(1+np.exp(-alpha_tmp))
        rnew = alpha[i]*mu[i]
        Xr = Xr + X[:,i]*(rnew-r)
    return alpha,mu,Xr

def normalizelogw(logw):
    c = logw.max()
    w = np.exp(logw-c)
    w = w/w.sum()
    return w

def varbvspredict(X,Z,w,alpha,mu_,mu_cov_):
    X = X.astype(np.float32)
    Z = np.concatenate((np.ones((Z.shape[0],1)),Z),axis=1)
    return (Z.dot(mu_cov_)+X.dot(alpha*mu_)).dot(w)