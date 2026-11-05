"""Every native CFA sample retained. Exact global softmax, streamed recompute backward.

One token is a lossless 2x2 Bayer packing, with all four signed values retained.
No pooling, quantization, top-k or dark threshold. All real cells read once/pass.
Only first derivatives supported; this layer is not a second-order optimizer.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def descriptor(raw,rn,wb,ccm,meta,start,end):
    b,_,h,w=raw.shape
    x=raw.flatten(2)[:,:,start:end].transpose(1,2)
    packing=raw.new_tensor([[1,0,0,0],[0,.5,.5,0],[0,0,0,1]])
    matrix=(ccm*wb[:,None,:])@packing
    rgb=x@matrix.transpose(1,2)
    scale=(matrix.square()*rn[:,None,:].square()).sum(-1).mean(-1).clamp_min(1e-12).sqrt()
    gd=(x[...,1:2]-x[...,2:3])/(rn[:,1].square()+rn[:,2].square()).sqrt()[:,None,None]
    physical=torch.cat([x/64,torch.asinh(x/rn[:,None,:]),torch.asinh(rgb/scale[:,None,None]),torch.asinh(gd)],-1)
    ix=torch.arange(start,end,device=raw.device)
    pos=torch.stack([(ix//w+.5)*(2./h)-1,(ix%w+.5)*(2./w)-1],-1).to(raw.dtype)
    # Explicit calibrated noise context, not scene normalization.
    d=torch.cat([physical,pos[None].expand(b,-1,-1),meta[:,:8,None].transpose(1,2).expand(-1,end-start,-1)],-1)
    return d,physical

def kv(raw,rn,wb,ccm,meta,start,end,params):
    w,b,wk,bk,wv,bv=params
    d,physical=descriptor(raw,rn,wb,ccm,meta,start,end)
    hidden=F.silu(F.linear(d,w,b))
    key=torch.tanh(F.linear(hidden,wk,bk))
    value=torch.cat([physical,F.linear(hidden,wv,bv)],-1)
    return key,value

class StreamRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,raw,rn,wb,ccm,meta,w,b,wk,bk,wv,bv,chunk):
        params=(w,b,wk,bk,wv,bv);n=raw.shape[-2]*raw.shape[-1];scale=q.shape[-1]**-.5
        maximum=q.new_full(q.shape[:2]+(1,),-torch.inf);den=q.new_zeros(maximum.shape);num=q.new_zeros(q.shape[:2]+(28,))
        for a in range(0,n,chunk):
            k,v=kv(raw,rn,wb,ccm,meta,a,min(a+chunk,n),params)
            logits=(q@k.transpose(1,2))*scale;m=torch.maximum(maximum,logits.amax(-1,keepdim=True))
            decay=torch.exp(maximum-m);p=torch.exp(logits-m)
            num=num*decay+p@v;den=den*decay+p.sum(-1,keepdim=True);maximum=m
        out=num/den;lse=maximum+den.log()
        ctx.save_for_backward(q,raw,rn,wb,ccm,meta,*params,out,lse);ctx.chunk=chunk
        return out
    @staticmethod
    def backward(ctx,dy):
        q,raw,rn,wb,ccm,meta,*rest=ctx.saved_tensors;params=rest[:6];out,lse=rest[6:]
        n=raw.shape[-2]*raw.shape[-1];scale=q.shape[-1]**-.5;dq=torch.zeros_like(q)
        # No large feature graph retained: reconstruct each chunk and differentiate
        # its key/value map using the exact full-softmax adjoint.
        dr=torch.zeros_like(raw) if ctx.needs_input_grad[1] else None
        dp=[torch.zeros_like(p) for p in params]
        center=(dy*out).sum(-1,keepdim=True)
        for a in range(0,n,ctx.chunk):
            z=min(a+ctx.chunk,n)
            with torch.enable_grad():
                rr=raw.detach();pp=[p.detach().requires_grad_(True) for p in params]
                # RAW gradient probe uses a small chunk as a synthetic 1-row image
                # only for value calculus; retain original coordinates separately.
                d,physical=descriptor(rr,rn,wb,ccm,meta,a,z)
                d=d.detach().requires_grad_(True)
                hidden=F.silu(F.linear(d,pp[0],pp[1]));k=torch.tanh(F.linear(hidden,pp[2],pp[3]));v=torch.cat([d[...,:12],F.linear(hidden,pp[4],pp[5])],-1)
                with torch.no_grad():
                    prob=torch.exp((q@k.transpose(1,2))*scale-lse)
                    dv=prob.transpose(1,2)@dy
                    ds=prob*(dy@v.transpose(1,2)-center)
                    dq.add_((ds@k)*scale);dk=(ds.transpose(1,2)@q)*scale
                grads=torch.autograd.grad((k,v),(d,*pp),(dk,dv))
                if dr is not None:
                    # Differentiate descriptor only for input-gradient diagnostics.
                    rc=raw.flatten(2)[:,:,a:z].detach().requires_grad_(True)
                    packing=raw.new_tensor([[1,0,0,0],[0,.5,.5,0],[0,0,0,1]])
                    mat=(ccm*wb[:,None,:])@packing;xx=rc.transpose(1,2);rgb=xx@mat.transpose(1,2)
                    sc=(mat.square()*rn[:,None,:].square()).sum(-1).mean(-1).clamp_min(1e-12).sqrt()
                    gg=(xx[...,1:2]-xx[...,2:3])/(rn[:,1].square()+rn[:,2].square()).sqrt()[:,None,None]
                    ph=torch.cat([xx/64,torch.asinh(xx/rn[:,None,:]),torch.asinh(rgb/sc[:,None,None]),torch.asinh(gg)],-1)
                    dr.flatten(2)[:,:,a:z]=torch.autograd.grad(ph,rc,grads[0][...,:12])[0]
            for target,g in zip(dp,grads[1:]):target.add_(g)
        return (dq,dr,None,None,None,None,*dp,None)

class NativePixelMemory(nn.Module):
    def __init__(self,chunk=262144):
        super().__init__();self.norm=nn.LayerNorm(192);self.query=nn.Linear(192,16)
        self.hidden=nn.Linear(22,32);self.key=nn.Linear(32,16);self.value=nn.Linear(32,16);self.output=nn.Linear(28,192)
        nn.init.normal_(self.output.weight,std=.002);nn.init.zeros_(self.output.bias)
        self.chunk=chunk
    def forward(self,q,cache,dense=False):
        raw,rn,wb,ccm,meta=cache;qq=self.query(self.norm(q));params=(self.hidden.weight,self.hidden.bias,self.key.weight,self.key.bias,self.value.weight,self.value.bias)
        if dense:
            k,v=kv(raw,rn,wb,ccm,meta,0,raw.shape[-2]*raw.shape[-1],params)
            y=torch.softmax(qq@k.transpose(1,2)/4,-1)@v
        else:y=StreamRead.apply(qq,raw,rn,wb,ccm,meta,*params,self.chunk)
        return self.output(y)
