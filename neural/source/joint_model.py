"""Full spatial-grid rereading with offline temporal evidence exchange."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from dense_model import DenseOfflineDirectBLE
from model import input_planes

class TemporalExchange(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(192)
        self.attention=nn.MultiheadAttention(192,4,batch_first=True)
        self.ffnorm=nn.LayerNorm(192)
        self.ff=nn.Sequential(nn.Linear(192,384),nn.GELU(),nn.Linear(384,192))
        self.gate=nn.Parameter(torch.full((192,),.1))
        self.ffgate=nn.Parameter(torch.full((192,),.1))
    def forward(self,q,position):
        t,n,d=q.shape
        z=self.norm(q).reshape(1,t*n,d)
        p=position[:,None].expand(t,n,d).reshape(1,t*n,d)
        a,_=self.attention(z+p,z+p,z,need_weights=False)
        q=q+a.reshape(t,n,d)*self.gate
        return q+self.ff(self.ffnorm(q))*self.ffgate

class JointGridBLE(DenseOfflineDirectBLE):
    def __init__(self):
        super().__init__()
        # Appended after all127 inherited tensors; deterministic fresh tensors.
        self.joint=nn.ModuleList([TemporalExchange() for _ in range(4)])
        self.joint_blend=0.
        self.grid_blend=1.
    def spatial_tokens(self,raw,rn,wb,ccm,meta,saturation,tile=None):
        wb=torch.where(meta[:,14:15].bool(),wb,torch.ones_like(wb))
        h,w=raw.shape[-2:]
        f=self.spatial_features(F.pad(input_planes(raw,rn,wb,ccm),(0,(-w)%64,0,(-h)%64),mode='replicate'),tile)
        gh,gw=(h+7)//8,(w+7)//8;f=f[...,:gh,:gw]
        ys=torch.arange(gh,device=f.device,dtype=f.dtype)*8
        xs=torch.arange(gw,device=f.device,dtype=f.dtype)*8
        ys=(ys+torch.minimum(ys+8,ys.new_tensor(h)))/h-1
        xs=(xs+torch.minimum(xs+8,xs.new_tensor(w)))/w-1
        yy,xx=torch.meshgrid(ys,xs,indexing='ij')
        pos=torch.stack([yy,xx])[None].expand(raw.shape[0],-1,-1,-1)
        desc=torch.cat([f,meta[...,None,None].expand(-1,-1,gh,gw),pos],1).flatten(2).transpose(1,2)
        weight=torch.cat([self.frame.project.weight[:,:128],self.frame.project.weight[:,314:]],1)
        return F.linear(desc,weight,self.frame.project.bias)
    def read(self,q,t):
        return self.frame.attention(q,t,t,need_weights=False)[0]
    def position(self,dt):
        p=torch.cumsum(dt[:,0],0)-dt[0,0]
        freq=torch.exp(torch.arange(0,192,2,device=p.device,dtype=p.dtype)*(-math.log(10000.)/192))
        z=p[:,None]*freq[None];return torch.stack([z.sin(),z.cos()],-1).flatten(1)*.1
    def finish(self,q,means,stds):
        return self.frame.out(torch.cat([q.flatten(1),means,stds],-1))
    def infer_tokens(self,tokens,dt,segment_start,checkpoint=False):
        from torch.utils.checkpoint import checkpoint as ck
        use=checkpoint and self.training and torch.is_grad_enabled()
        def run(fn,*args):return ck(fn,*args,use_reentrant=False) if use else fn(*args)
        qs=[];ms=[];ss=[]
        for tok in tokens:
            qs.append(run(self.read,self.frame.queries.expand(tok.shape[0],-1,-1),tok))
            mean=tok.mean(1);ms.append(mean);ss.append(((tok-mean[:,None]).square().mean(1)+1e-6).sqrt())
        q=torch.stack(qs,1);means=torch.stack(ms,1);stds=torch.stack(ss,1)
        b,t,n,d=q.shape
        base_feature=self.finish(q.reshape(b*t,n,d),means.reshape(b*t,d),stds.reshape(b*t,d)).reshape(b,t,-1)
        base=self.frame.point(base_feature)
        result=[]
        for bi in range(b):
            starts=segment_start[bi].nonzero().flatten().tolist()+[t];parts=[]
            for a,z in zip(starts[:-1],starts[1:]):
                v=q[bi,a:z];position=self.position(dt[bi,a:z])
                for ri in range(3):
                    v=run(self.joint[ri],v,position)
                    # Context updates the queries; each reread sees the FULL own-frame grid.
                    rereads=[run(self.read,v[j:j+1],tokens[a+j][bi:bi+1]) for j in range(z-a)]
                    v=v+torch.cat(rereads,0)*.25
                v=run(self.joint[3],v,position)
                feature=self.finish(v,means[bi,a:z],stds[bi,a:z])
                parts.append(self.frame.point(feature))
            result.append(torch.cat(parts,0))
        joint=torch.stack(result)
        return dict(frame=base,offline=base+self.joint_blend*(joint-base),features=base_feature)
    def forward(self,raw,rn,wb,ccm,meta,saturation,dt,segment_start,tile=None,checkpoint_frames=False,frame_batch=1):
        assert raw.shape[1]<=29 and bool(segment_start[:,0].all())
        assert bool(torch.isfinite(dt).all() and (dt>0).all())
        from torch.utils.checkpoint import checkpoint
        tokens=[]
        for j in range(raw.shape[1]):
            args=tuple(v[:,j] for v in (raw,rn,wb,ccm,meta,saturation))
            if checkpoint_frames and self.training and torch.is_grad_enabled():
                tok=checkpoint(lambda *a:self.spatial_tokens(*a,tile=tile),*args,use_reentrant=False)
            else:tok=self.spatial_tokens(*args,tile=tile)
            tokens.append(tok)
        return self.infer_tokens(tokens,dt,segment_start,checkpoint_frames)
