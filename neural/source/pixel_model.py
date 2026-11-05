import torch
from joint_model import JointGridBLE
from pixel_memory import NativePixelMemory
from bounded_head import decode_dn
class PixelMemoryBLE(JointGridBLE):
    def __init__(self):
        super().__init__();self.pixel=NativePixelMemory();self.pixel_blend=0.
    def read_native(self,q,cache):
        if self.pixel_blend==0.:return torch.zeros_like(q)
        return self.pixel_blend*.1*self.pixel(q,cache)
    def spatial_tokens(self,raw,rn,wb,ccm,meta,saturation,tile=None):
        grid=super().spatial_tokens(raw,rn,wb,ccm,meta,saturation,tile)
        wb=torch.where(meta[:,14:15].bool(),wb,torch.ones_like(wb))
        return dict(grid=grid,native=(raw,rn,wb,ccm,meta))
    def infer_tokens(self,tokens,dt,segment_start,checkpoint=False):
        from torch.utils.checkpoint import checkpoint as ck
        use=checkpoint and self.training and torch.is_grad_enabled()
        def run(fn,*args):return ck(fn,*args,use_reentrant=False) if use else fn(*args)
        qs=[];ms=[];ss=[]
        for memory in tokens:
            tok=memory['grid']
            initial=self.frame.queries.expand(tok.shape[0],-1,-1)
            qs.append(run(self.read,initial,tok)+self.read_native(initial,memory['native']))
            mean=tok.mean(1);ms.append(mean);ss.append(((tok-mean[:,None]).square().mean(1)+1e-6).sqrt())
        q=torch.stack(qs,1);means=torch.stack(ms,1);stds=torch.stack(ss,1)
        b,t,n,d=q.shape
        base_feature=self.finish(q.reshape(b*t,n,d),means.reshape(b*t,d),stds.reshape(b*t,d)).reshape(b,t,-1)
        base_score=self.frame.point.logits(base_feature)
        base=decode_dn(base_score)
        result=[]
        for bi in range(b):
            starts=segment_start[bi].nonzero().flatten().tolist()+[t];parts=[]
            for a,z in zip(starts[:-1],starts[1:]):
                v=q[bi,a:z];position=self.position(dt[bi,a:z])
                for ri in range(3):
                    v=run(self.joint[ri],v,position)
                    # Context updates the queries; each reread sees the FULL own-frame grid.
                    rereads=[run(self.read,v[j:j+1],tokens[a+j]['grid'][bi:bi+1])+self.read_native(v[j:j+1],tuple(x[bi:bi+1] for x in tokens[a+j]['native'])) for j in range(z-a)]
                    v=v+torch.cat(rereads,0)*.25
                v=run(self.joint[3],v,position)
                feature=self.finish(v,means[bi,a:z],stds[bi,a:z])
                parts.append(self.frame.point.logits(feature))
            result.append(torch.cat(parts,0))
        joint_score=torch.stack(result)
        assert 0. <= self.joint_blend <= 1.
        offline_score=base_score+self.joint_blend*(joint_score-base_score)
        return dict(frame=base,offline=decode_dn(offline_score),frame_score=base_score,offline_score=offline_score,features=base_feature)
