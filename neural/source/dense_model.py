from model import OfflineDirectBLE
from dense_grid import grid_encode
class DenseOfflineDirectBLE(OfflineDirectBLE):
    def __init__(self):
        super().__init__();self.grid_blend=0.
    def encode_frame(self,raw,rn,wb,ccm,meta,saturation,tile=None):
        b=self.grid_blend
        if b<=0:return super().encode_frame(raw,rn,wb,ccm,meta,saturation,tile)
        dense=grid_encode(self,raw,rn,wb,ccm,meta,saturation,tile)
        if b>=1:return dense
        old=super().encode_frame(raw,rn,wb,ccm,meta,saturation,tile)
        return tuple(a+b*(z-a) for a,z in zip(old,dense))
