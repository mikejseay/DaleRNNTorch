import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

plt.close('all')


# N        = 200
# sigm     = 2.
# mu       = 0.
# angs     = linspace(0,2*np.pi,100)

# rnondale = 2*sigm*(sc.rand(N,N)-0.5)+mu
# mm,sm    = np.mean(rnondale),np.var(rnondale)

# lam,vs   = np.linalg.eig(rnondale)
# rho      = np.sqrt(N*sm)

# mp.figure()
# mp.plot(np.real(lam),np.imag(lam),'o',ms=3,color='0.7')
# mp.plot(np.sin(angs)*rho,np.cos(angs)*rho,'r-',lw=3)


N = 300
frac = 0.8  # frac of exc
Ne = int(frac * N)
Ni = N - Ne
scale, scali = 0.1, 0.3  # scale of uniform intervals
mue = 0.5  # mean of exc weights
mui = -4.5 * mue  # mean of inh weights, for singular eigenvalue <0: mui>Ne/Ni*mu

print('Dale')
emat = scale * (sc.rand(N, Ne)) + (mue - scale / 2.)  # choose such that right mean
imat = -scali * (sc.rand(N, Ni)) + (mui + scali / 2.)  # choose such that right mean
rdale = np.append(emat, imat, axis=1)  # glue mats together
lamd, vsd = np.linalg.eig(rdale)  # compute spectrum
mmd = np.mean(rdale)
smde, smdi = np.var(emat), np.var(imat)
print('pop means: ', N * mmd, Ne * mue + Ni * mui)
print('subpop std, exc: ', np.std(emat),
      scale / np.sqrt(12))  # note: for uniform distr in [a,b]: mean = (b-a)/2, var=(b-a)^2/12
print('subpop std, inh: ', np.std(imat), scali / np.sqrt(12))
angs = np.linspace(0, 2 * np.pi, 100)
rhod = np.sqrt(Ne * smde + Ni * smdi)  # for linear stab: rhod<1
print('rhod: ', rhod, max(np.real(lamd)))
plt.figure()
plt.plot(np.sin(angs) * rhod, np.cos(angs) * rhod, 'r-', lw=3)
plt.plot(np.real(lamd), np.imag(lamd), 'o', ms=3, color='0.7')

# run linear dynamics
T, dt = 500, 0.1
ts = np.arange(0, T, dt)
tau = 1.
x = np.zeros((N, len(ts)))
inp = np.ones(N)
for i in range(1, len(ts)):
    x[:, i] = x[:, i - 1] + dt / tau * (-x[:, i - 1] + np.dot(rdale, x[:, i - 1]) + inp)

# plot dynamics    
plt.figure()
plt.plot(ts, x.T)

# check if at fixed point activations
# tau*dx/dt = 0 = -x + rdale*x + inp
# ==> x0 = (identity-rdale)^-1 * inp = dot(inv(eye(N)-rdale),inp) 
plt.figure()
plt.plot(x[:, -1], 'b', lw=4)
plt.plot(np.dot(np.linalg.inv(np.eye(N) - rdale), inp), 'r')

print('')
print('Hybrid')

rndale = rdale.copy()
[np.random.shuffle(rndale[_]) for _ in range(N)]
# mp.matshow(rndale)
print('pop std: ', np.std(rndale), end=' ')
smnd = np.var(rndale)
rhond = np.sqrt(N * smnd)  # for linear stab: rhod<1
print('rhod,rhond: ', rhod, rhond)
lamnd, vsnd = np.linalg.eig(rndale)
print('rhond estimate: ', max(np.real(lamnd)))
plt.figure()
plt.plot(np.sin(angs) * rhond, np.cos(angs) * rhond, 'r-', lw=3)
plt.plot(np.real(lamnd), np.imag(lamnd), 'o', ms=3, color='0.7')

### make sure, all elements are really simply shuffled, not changed otherwise
# mp.figure()
# [mp.plot(np.sort(rdale[_]),np.sort(rndale[_]),'b.') for _ in range(N)] 


# rndale=rndale/25.
# run linear dynamics
T, dt = 500, 0.1
ts = np.arange(0, T, dt)
tau = 1.
x = np.zeros((N, len(ts)))
inp = np.ones(N)
for i in range(1, len(ts)):
    x[:, i] = x[:, i - 1] + dt / tau * (-x[:, i - 1] + np.dot(rndale, x[:, i - 1]) + inp)

# plot dynamics    
plt.figure()
plt.plot(ts, x.T)

# check if at fixed point activations
# tau*dx/dt = 0 = -x + rdale*x + inp
# ==> x0 = (identity-rdale)^-1 * inp = dot(inv(eye(N)-rdale),inp) 
plt.figure()
plt.plot(np.dot(np.linalg.inv(np.eye(N) - rndale), inp), 'b', lw=4)  # note: solution only stable, if max(real(lam))<1
plt.plot(x[:, -1], 'r')

'''
### Variance computation for hybrid network:
ai = mui-scali/2.             # lower interval bound inh
bi = mui+scali/2.             # upper interval bound inh
vi = 1./3*(ai**2+ai*bi+bi**2) # raw variance inh (for uniform distr)
mi = 1./2*(ai+bi)             # mean inh
ae = mue-scale/2.             # lower interval bound exc
be = mue+scale/2.             # upper interval bound exc
ve = 1./3*(ae**2+ae*be+be**2) # raw variance exc
me = 1./2*(ae+be)             # mean exc
# rhond = np.sqrt(total var):
print np.sqrt((Ne*ve+Ni*vi)-(Ne/N*me+Ni/N*mi)**2), np.sqrt(N*np.mean(rndale**2)-N*np.mean(rndale)**2)
'''
plt.show()
