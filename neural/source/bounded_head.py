"""Native-DN bounded mean regression; no display clamp or straight-through gradient."""
import torch
from torch import nn
from torch.nn import functional as F
BOUND_DN=1.0
class BoundedDNHead(nn.Linear):
    # Identical learned tensor names/shapes; retained Adam moments are explicit warm-start state.
    def logits(self,features):return F.linear(features,self.weight,self.bias)
    def forward(self,features):return decode_dn(self.logits(features))
def decode_dn(score):return BOUND_DN*torch.tanh(score)
def dn_loss(score,target_dn):
    if not bool(torch.isfinite(score).all()):raise ValueError('Non-finite head score')
    if not bool(torch.isfinite(target_dn).all()) or bool((target_dn.abs()>BOUND_DN).any()):
        raise ValueError('Target outside native DN contract; never silently clip labels')
    # q=(y+1)/2 is a continuous target, NOT a sign class. For each scalar,
    # d loss/d score = tanh(score)-y. Wrong saturated predictions retain gradient.
    return F.binary_cross_entropy_with_logits(2*score,(target_dn/BOUND_DN+1)/2)
