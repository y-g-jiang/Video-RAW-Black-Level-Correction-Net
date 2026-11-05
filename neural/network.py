"""No new parameters: original PTC37524 backbone with explicit T1/T7 tasks."""
import torch
from pixel_model import PixelMemoryBLE
CONFIG=dict(schema='PTC37524_T1_T7_V1',grid=True,joint=True,native=True,bound_dn=3.,shared_spatial=True,checkpoint=True)

class PosttrainBLE(PixelMemoryBLE):
 def __init__(self):
  super().__init__();self.grid_blend=self.joint_blend=self.pixel_blend=1.
 def verify_execution(self):
  assert self.grid_blend==self.joint_blend==self.pixel_blend==1.
 def forward_inputs(self,x,tile=None,checkpoint=None):
  self.verify_execution()
  if checkpoint is None:checkpoint=self.training and torch.is_grad_enabled()
  from torch.utils.checkpoint import checkpoint as ck
  tokens=[]
  for j in range(x['raw'].shape[1]):
   args=tuple(x[k][:,j] for k in ['raw','rn','wb','ccm','meta','saturation'])
   fn=lambda *a:self.spatial_tokens(*a,tile=tile)
   tokens.append(ck(fn,*args,use_reentrant=False) if checkpoint else fn(*args))
  # Every frame is a separate segment for full offline T1, no neighbour leakage.
  single=self.infer_tokens(tokens,x['dt'],torch.ones_like(x['segment_start']),checkpoint)
  video=self.infer_tokens(tokens,x['dt'],x['segment_start'],checkpoint)
  decode=lambda z:3*torch.tanh(z/3)
  return dict(frame=decode(video['frame_score']),single=decode(single['offline_score']),offline=decode(video['offline_score']),
   frame_score=video['frame_score'],single_score=single['offline_score'],offline_score=video['offline_score'])

def load_model(state,config,device='cuda'):
 assert config==CONFIG
 m=PosttrainBLE().to(device);m.load_state_dict(state,strict=True);m.verify_execution();return m
