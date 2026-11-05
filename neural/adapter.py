import torch
from torch.nn import functional as F
from model import metadata
from estimator_input import estimator_view
BOUND=3.
BLIND_RN=3.078100092249427
BLIND_K=.5991577980453828

def inputs(raw,q):
 b,t=raw.shape[:2];dev=raw.device
 rn=torch.full((b,t,4),BLIND_RN,device=dev);one=torch.ones(b,t,1,device=dev)
 # Radiance templates are linear RGB proxies, no D810 colour calibration transplant.
 wb=torch.ones(b,t,3,device=dev);valid=torch.zeros(b,t,3,device=dev)
 meta=metadata(rn,torch.full_like(rn,BLIND_K),wb,one,one,one*25,valid)
 starts=torch.zeros(b,t,dtype=torch.bool,device=dev);starts[:,0]=True
 return dict(raw=estimator_view(raw),rn=rn,wb=wb,ccm=torch.eye(3,device=dev).view(1,1,3,3).expand(b,t,3,3),
  meta=meta,saturation=torch.full_like(rn,3583),dt=one*(1001/30000),segment_start=starts)

def loss(out,y):
 assert bool(torch.isfinite(y).all() and (y.abs()<BOUND).all()),'Never clip labels'
 q=(y/BOUND+1)/2
 def part(s):return BOUND**2*F.binary_cross_entropy_with_logits(2*s/BOUND,q)
 return .25*part(out['frame_score'])+.375*part(out['single_score'])+.375*part(out['offline_score'])
