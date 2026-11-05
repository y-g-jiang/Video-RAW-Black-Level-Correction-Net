"""Numerical and support checks against the original scalar Gaussian implementation."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys
from pathlib import Path
import numpy as np
from filter_comparison import apply, matrices, windows, gains
from reference_smoothers import kalman_center,jump_center

rng=np.random.default_rng(714)
w=rng.normal(size=(4,7,4));r=np.array([.003,.006,.002,.009])
for cfg,ref in [({'family':'kalman','q':.1},lambda x:kalman_center(x,.1,r)),
                ({'family':'jump','q':.01,'p':.1,'v':16.},lambda x:jump_center(x,.01,.1,16.,r))]:
 out=apply(w,cfg,r)[:,3]
 expected=np.stack([ref(v) for v in w])
 assert np.max(abs(out-expected))<2e-7,(cfg,np.max(abs(out-expected)))
cfgs=[{'family':'mean'},{'family':'median'},{'family':'exponential','rho':.55},
      {'family':'kalman','q':.1},{'family':'jump','q':.01,'p':.1,'v':16.}]
for cfg in cfgs:
 constant=np.full((3,7,4),.61)
 assert np.max(abs(apply(constant,cfg,r)-constant))<2e-6
 assert np.isfinite(apply(w,cfg,r)).all()
 # Individual windows are isolated; perturbing one cannot change another.
 x=apply(w,cfg,r);changed=w.copy();changed[0]+=1
 assert np.array_equal(x[1:],apply(changed,cfg,r)[1:])
# A symmetric exponential with rho=1 exactly reproduces the seven-frame mean.
assert np.max(abs(apply(w,{'family':'exponential','rho':1.},r)-apply(w,{'family':'mean'},r)))<1e-14
print('PASS: scalar Gaussian reference, constant preservation, finite outputs, isolated support, exponential limit')

# A fixed segment mean and a rolling center mean have different relative gain.
# This guards the specific operator mismatch corrected in this revision.
paired=np.zeros((2,3,114,4))
paired[:,1]=.3
paired[:,2]=(.3*(-1.)**np.arange(114))[None,:,None]
win=np.stack([windows(paired[:,k]) for k in range(3)],axis=1)
filtered=apply(win,{'family':'mean'},r)[...,3,:]
truth=paired[:,:,3:-3]
gc,gr=gains(filtered,truth)
assert np.allclose(gc,1.) and np.allclose(gr,-1/7)
gc,gr=gains(truth,truth)
assert np.allclose(gc,1.) and np.allclose(gr,1.)
print('PASS: response gains use rolling centers; known unit and -1/7 gains recovered')
