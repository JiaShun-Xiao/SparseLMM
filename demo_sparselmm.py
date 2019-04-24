import numpy as np
from sklearn.metrics import r2_score
from sklearn import linear_model
from SparseLmm import *

n = 1000 # number of sample
p = 100000 # number of SNP
m = 3 # number of covariate
na = 20 # number of causal snp
r = 0.5 # SNP heritability
se = 4 # variance of residual

maf = 0.05+0.45*np.random.rand(p)
maf_mat = np.repeat(maf,n).reshape(p,n).T
# latent variable gamma
gamma = np.zeros(p)
index_gamma = np.random.choice(list(range(p)),na,replace=False)
beta = np.zeros(p)
beta[index_gamma] = np.random.randn(na)

X = (np.random.rand(n,p) < maf_mat)*1 + (np.random.rand(n,p) < maf_mat)*1
beta = np.sqrt((r/(1-r))*se/X.dot(beta).var())*beta

intercept = np.random.randn(1)
Z = np.random.randn(n,m)
u = np.random.randn(m)
y = intercept + X.dot(beta) + Z.dot(u) + np.sqrt(se)*np.random.randn(n)

sep = 800
train_X = X[:sep,:]
train_y = y[:sep]
train_Z = Z[:sep,:]
test_X = X[sep:,:]
test_y = y[sep:]
test_Z = Z[sep:,:]
print(train_X.shape,train_y.shape,train_Z.shape,test_X.shape,test_y.shape,test_Z.shape)

w,alpha,pip,mu_,mu_cov_,heri,logodd = varbvs(train_X,train_y,train_Z)

y_pre = varbvspredict(test_X,test_Z,w,alpha,mu_,mu_cov_)
print(r2_score(test_y,y_pre))

clf = linear_model.Lasso(alpha=.1)
clf.fit(np.concatenate((train_X,train_Z),axis=1),train_y)
y_pre_lasso = clf.predict(np.concatenate((test_X,test_Z),axis=1))
print(r2_score(test_y,y_pre_lasso))