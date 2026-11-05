"""Full spatial-grid cross-attention, no per64-cell feature pooling/statistics.
Reuses learned projection columns for CNN features, metadata and position.
All original model tensors remain; no diagnostic weights are persisted.
"""
import torch
from torch.nn import functional as F
from model import input_planes

def grid_encode(m,raw,rn,wb,ccm,meta,saturation,tile=None):
    wb=torch.where(meta[:,14:15].bool(),wb,torch.ones_like(wb))
    h,w=raw.shape[-2:];x=input_planes(raw,rn,wb,ccm)
    x=F.pad(x,(0,(-w)%64,0,(-h)%64),mode='replicate')
    f=m.spatial_features(x,tile)
    gh,gw=(h+7)//8,(w+7)//8;f=f[...,:gh,:gw]
    ys=torch.arange(gh,device=f.device,dtype=f.dtype)*8
    xs=torch.arange(gw,device=f.device,dtype=f.dtype)*8
    ys=(ys+torch.minimum(ys+8,ys.new_tensor(h)))/h-1
    xs=(xs+torch.minimum(xs+8,xs.new_tensor(w)))/w-1
    yy,xx=torch.meshgrid(ys,xs,indexing='ij');pos=torch.stack([yy,xx])[None].expand(raw.shape[0],-1,-1,-1)
    desc=torch.cat([f,meta[...,None,None].expand(-1,-1,gh,gw),pos],1).flatten(2).transpose(1,2)
    # Original333: featuremean128,std128,stats58,metadata17,xy2.
    # Selecting columns preserves model/Adam tensor shapes. Unused columns get zero current gradient.
    weight=torch.cat([m.frame.project.weight[:,:128],m.frame.project.weight[:,314:]],1)
    t=F.linear(desc,weight,m.frame.project.bias)
    q=m.frame.queries.expand(t.shape[0],-1,-1)
    a,_=m.frame.attention(q,t,t,need_weights=False)
    mean=t.mean(1);var=(t-mean[:,None]).square().mean(1)
    feature=m.frame.out(torch.cat([a.flatten(1),mean,(var+1e-6).sqrt()],-1))
    return feature,m.frame.point(feature)
