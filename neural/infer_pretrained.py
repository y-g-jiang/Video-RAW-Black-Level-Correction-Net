"""Pretrained T1 used for the PTC and exposure-ladder results.

Input .npz: raw=(T,4,H/2,W/2) signed reference-subtracted DN;
optional wb=(T,3), ccm=(T,3,3). Output: offsets=(T,4) DN.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from pixel_model import PixelMemoryBLE
from model import metadata
from estimator_input import estimator_view

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
    path=HERE.parent/'weights/nafble_pretrained_37524.pt'
    provenance=json.loads((HERE/'PROVENANCE.json').read_text('utf8'))['pretrained_37524']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['file_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['phase_updates']==37524 and state['head_bound_dn']==1.
    torch.set_num_threads(2)
    model=PixelMemoryBLE();model.grid_blend=model.joint_blend=model.pixel_blend=1.
    model.load_state_dict(state['model'],strict=True);model=model.to(a.device).eval().requires_grad_(False)
    with np.load(a.input,allow_pickle=False) as data:
        raw=data['raw'].astype(np.float32);has_wb='wb' in data
        wb=data['wb'] if has_wb else np.ones((len(raw),3))
        ccm=data['ccm'] if 'ccm' in data else np.tile(np.eye(3),(len(raw),1,1))
    assert raw.ndim==4 and raw.shape[1]==4 and np.isfinite(raw).all()
    offsets=[]
    with torch.inference_mode():
        for t in range(len(raw)):
            x=torch.from_numpy(raw[t])[None,None].to(a.device)
            w=torch.as_tensor(wb[t],dtype=torch.float32,device=a.device).reshape(1,1,3)
            c=torch.as_tensor(ccm[t],dtype=torch.float32,device=a.device).reshape(1,1,3,3)
            rn=torch.full((1,1,4),3.078100092249427,device=a.device);k=torch.full_like(rn,.5991577980453828)
            one=torch.ones(1,1,1,device=a.device)
            valid=torch.tensor([float(has_wb),0.,0.],device=a.device).reshape(1,1,3)
            meta=metadata(rn,k,w,one,one,one*25,valid)
            out=model(raw=estimator_view(x),rn=rn,wb=w,ccm=c,meta=meta,
                saturation=torch.full_like(rn,3583),dt=one,
                segment_start=torch.ones(1,1,dtype=torch.bool,device=a.device),tile=256,checkpoint_frames=False)
            offsets.append(out['offline'][0,0].cpu().numpy())
    np.savez_compressed(a.output,offsets=offsets)
    print('Wrote',len(offsets),'pretrained T1 estimates in DN')

if __name__=='__main__':main()
