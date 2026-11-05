"""OFFLINE_DIRECT_BLE_V1. Canonical planes [R,G1,G2,B], signed physical DN.

Untrained preparation. RAW adapters must supply measured units and explicit
metadata; normalized public TIFFs are not silently treated as S5II sensor DN.
"""
import torch
from torch import nn
from torch.nn import functional as F
from components import SpatialEncoder, FrameAggregator, TemporalHead


def input_planes(raw, rn, wb, ccm):
    """raw B,4,H,W; rn B,4; WB B,3; camera RGB -> output RGB CCM B,3,3."""
    assert raw.ndim == 4 and raw.shape[1] == 4
    assert rn.shape == raw.shape[:2] and bool((rn > 0).all())
    assert wb.shape == (raw.shape[0], 3) and bool((wb > 0).all())
    assert ccm.shape == (raw.shape[0], 3, 3)
    packing = raw.new_tensor([[1,0,0,0],[0,.5,.5,0],[0,0,0,1]])
    matrix = (ccm * wb[:, None, :]) @ packing
    rgb = torch.einsum('bij,bjhw->bihw', matrix, raw)
    # Independent CFA read noise is an explicit model assumption.
    variance = (matrix.square() * rn[:, None, :].square()).sum(-1)
    shared_scale = variance.mean(-1).clamp_min(1e-12).sqrt()[:, None, None, None]
    gd = (raw[:, 1:2]-raw[:, 2:3]) / (rn[:, 1].square()+rn[:, 2].square()).sqrt()[:, None, None, None]
    return torch.cat([raw/64, torch.asinh(raw/rn[...,None,None]),
                      torch.asinh(rgb/shared_scale), torch.asinh(gd)], 1)


def metadata(rn, k, wb, exposure, gain, temperature, valid):
    """17 fields: log RN4, log k4, log WB3, log exposure, log analog gain,
    (temperature-25)/20, validity(WB/exposure/temperature).
    Missing WB/exposure/temp become identity/1/25 with explicit validity=0.
    RN/k/gain require an adapter contract; unknown calibration must not be guessed.
    """
    assert rn.shape[-1] == k.shape[-1] == 4 and wb.shape[-1] == 3
    assert valid.shape[-1] == 3
    assert bool((rn > 0).all() and (k > 0).all() and (gain > 0).all())
    wb = torch.where(valid[...,0:1].bool(), wb, torch.ones_like(wb))
    exposure = torch.where(valid[...,1:2].bool(), exposure, torch.ones_like(exposure))
    temperature = torch.where(valid[...,2:3].bool(), temperature, torch.full_like(temperature,25))
    assert bool((wb > 0).all() and (exposure > 0).all())
    return torch.cat([rn.log(), k.log(), wb.log(), exposure.log(), gain.log(),
                      (temperature-25)/20, valid.to(rn.dtype)], -1)


def masked_statistics(raw, rn, saturation, region=64):
    """58 fields; edge padding never counts as negative pixels or evidence."""
    b,_,h,w=raw.shape
    ph=(-h)%region; pw=(-w)%region
    pad=lambda x:F.pad(x,(0,pw,0,ph))
    mask=pad(torch.ones_like(raw[:,:1]))
    summ=lambda x:F.avg_pool2d(x,region,region)*(region*region)
    count=summ(mask)
    avg=lambda x:summ(pad(x))/count.clamp_min(1)
    mean=avg(raw)
    var=(avg(raw.square())-mean.square()).clamp_min(0)
    neg=avg((raw<0).to(raw.dtype))
    sat=avg((raw>=saturation[...,None,None]).to(raw.dtype))
    # Each real horizontal/vertical pair counted once, no padded differences.
    gx=F.pad((raw[...,1:]-raw[...,:-1]).abs(),(0,1))
    gy=F.pad((raw[...,1:,:]-raw[...,:-1,:]).abs(),(0,0,0,1))
    mx=torch.ones_like(raw[:,:1]);mx[...,-1]=0
    my=torch.ones_like(mx);my[...,-1,:]=0
    grad=summ(pad(gx+gy))/summ(pad(mx+my)).clamp_min(1)
    z=raw/rn[...,None,None]
    cdf=[avg(torch.sigmoid((t-z)/.25)) for t in (-4.,-2.,-1.,-.5,0.,.5,1.,2.,4.)]
    return torch.cat([mean/64,torch.log1p(var),neg,sat,torch.log1p(grad),*cdf,
                      count.clamp_min(1).log(),count/(region*region)],1),mask


class OfflineDirectBLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=SpatialEncoder()
        self.frame=FrameAggregator()
        self.temporal=TemporalHead()
        # This preparation version trains only the selected offline task.
        del self.temporal.causal

    def spatial_features(self, x, tile=256):
        """Same local encoder; aligned cores, halo64; no image-wide pooling per tile."""
        if tile is None:return self.spatial(x)
        assert tile%64==0 and tile>=64
        h,w=x.shape[-2:]; rows=[]
        for y in range(0,h,tile):
            cols=[]
            for z in range(0,w,tile):
                y1=min(y+tile,h);z1=min(z+tile,w)
                ya=max(0,y-64);za=max(0,z-64)
                yb=min(h,y1+64);zb=min(w,z1+64)
                f=self.spatial(x[...,ya:yb,za:zb])
                cols.append(f[...,(y-ya)//8:(y1-ya)//8,(z-za)//8:(z1-za)//8])
            rows.append(torch.cat(cols,-1))
        return torch.cat(rows,-2)

    def encode_frame(self, raw, rn, wb, ccm, meta, saturation, tile=256):
        assert min(raw.shape[-2:])>=2 and meta.shape==(raw.shape[0],17)
        wb=torch.where(meta[:,14:15].bool(),wb,torch.ones_like(wb))
        stats,mask=masked_statistics(raw,rn,saturation)
        h,w=raw.shape[-2:]
        x=input_planes(raw,rn,wb,ccm)
        x=F.pad(x,(0,(-w)%64,0,(-h)%64),mode='replicate')
        f=self.spatial_features(x,tile)
        fw=F.avg_pool2d(mask,8,8)
        den=F.avg_pool2d(fw,8,8).clamp_min(1e-12)
        mean=F.avg_pool2d(f*fw,8,8)/den
        var=(F.avg_pool2d(f.square()*fw,8,8)/den-mean.square()).clamp_min(0)
        gh,gw=stats.shape[-2:]
        ys=(torch.arange(gh,device=raw.device,dtype=raw.dtype)*64)
        xs=(torch.arange(gw,device=raw.device,dtype=raw.dtype)*64)
        ys=(ys+torch.minimum(ys+64,ys.new_tensor(h)))/h-1
        xs=(xs+torch.minimum(xs+64,xs.new_tensor(w)))/w-1
        yy,xx=torch.meshgrid(ys,xs,indexing='ij')
        pos=torch.stack([yy,xx],0)[None].expand(raw.shape[0],-1,-1,-1)
        desc=torch.cat([mean,(var+1e-6).sqrt(),stats,
                        meta[...,None,None].expand(-1,-1,gh,gw),pos],1)
        return self.frame(desc.flatten(2).transpose(1,2))

    def aggregate(self, features, point, dt, segment_start):
        """B,T. dt in seconds from previous frame, or frame units under declared
        synthetic sampling contract. Real timestamps must not be invented.
        Segment boundaries block both forward and reverse information.
        """
        b,t,_=features.shape
        assert dt.shape==(b,t,1) and segment_start.shape==(b,t)
        assert bool(torch.isfinite(dt).all() and (dt>0).all())
        assert bool(segment_start[:,0].all())
        outputs=[]
        for i in range(b):
            starts=segment_start[i].nonzero().flatten().tolist()+[t]
            pieces=[]
            for a,z in zip(starts[:-1],starts[1:]):
                pieces.append(self.temporal.sequence(features[i:i+1,a:z],point[i:i+1,a:z],
                                                     dt[i:i+1,a:z],offline=True))
            outputs.append(torch.cat(pieces,1))
        return torch.cat(outputs,0)

    def forward(self, raw, rn, wb, ccm, meta, saturation, dt, segment_start, tile=256,
                checkpoint_frames=False,frame_batch=1):
        fs=[];ps=[]
        assert frame_batch>=1
        for j in range(0,raw.shape[1],frame_batch):
            n=min(frame_batch,raw.shape[1]-j)
            args=tuple(v[:,j:j+n].reshape(-1,*v.shape[2:]) for v in (raw,rn,wb,ccm,meta,saturation))
            if checkpoint_frames and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                f,p=checkpoint(lambda *a:self.encode_frame(*a,tile=tile),*args,use_reentrant=False)
            else:
                f,p=self.encode_frame(*args,tile)
            fs.append(f.reshape(raw.shape[0],n,-1));ps.append(p.reshape(raw.shape[0],n,4))
        f=torch.cat(fs,1);p=torch.cat(ps,1)
        return dict(frame=p,offline=self.aggregate(f,p,dt,segment_start),features=f)


def offline_loss(output, truth, valid, segment_start):
    """Supervise absolute residual and true increments; never force constant black."""
    frame,offline=output['frame'],output['offline']
    assert truth.shape==frame.shape and valid.shape==truth.shape
    def mean_loss(a,b,m):
        a=torch.where(m,a,torch.zeros_like(a))
        b=torch.where(m,b,torch.zeros_like(b))
        v=F.smooth_l1_loss(a,b,beta=.05,reduction='none')
        return (v*m).sum()/m.sum().clamp_min(1)
    point=.25*mean_loss(frame,truth,valid)+.75*mean_loss(offline,truth,valid)
    if truth.shape[1]<2:return point
    m=valid[:,1:] & valid[:,:-1] & ~segment_start[:,1:,None]
    return point+.1*mean_loss(offline[:,1:]-offline[:,:-1],truth[:,1:]-truth[:,:-1],m)
