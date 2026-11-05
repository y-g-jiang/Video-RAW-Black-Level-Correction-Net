"""Frozen NAFBLE 105k: signed, reference-subtracted Bayer-cell input in DN.

Example: python neural/infer.py --input packed_raw.npy --output offsets.npz
Input shape is (T, 4, H/2, W/2), channel order R,G1,G2,B; T >= 7.
The output centers omit three frames on each boundary. No RAW decoder or ISP
is implied by this array interface.
"""
from pathlib import Path
import argparse, hashlib, sys, json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
from network import CONFIG,load_model
from adapter import inputs
EXPECTED='cc42ebe94662b5e3f16987bc51f982623dcef5fb7d65b035192dcdc8d2eb964a'

def load(device='cpu', checkpoint='105k'):
    provenance=json.loads((HERE/'PROVENANCE.json').read_text('utf8'))
    path=HERE.parent/('weights/nafble_105000.pt' if checkpoint=='105k' else 'weights/nafble_temporal_46000.pt')
    expected=EXPECTED if checkpoint=='105k' else provenance['temporal_46000']['file_sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['phase_updates']==(105000 if checkpoint=='105k' else 46000) and state['model_config']==CONFIG
    return load_model(state['model'],CONFIG,device).eval().requires_grad_(False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',choices=['105k','46k'],default='105k')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    raw=np.load(a.input,allow_pickle=False)
    assert raw.ndim==4 and raw.shape[1]==4 and len(raw)>=7
    assert np.isfinite(raw).all()
    torch.set_num_threads(2)
    model=load(a.device,a.checkpoint);one=[];seven=[];centers=np.arange(3,len(raw)-3)
    with torch.inference_mode():
        for c in centers:
            x=torch.from_numpy(np.ascontiguousarray(raw[c-3:c+4],dtype=np.float32))[None].to(a.device)
            out=model.forward_inputs(inputs(x,None),checkpoint=False)
            one.append(out['single'][0,3].cpu().numpy())
            seven.append(out['offline'][0,3].cpu().numpy())
    np.savez_compressed(a.output,centers=centers,t1=one,t7=seven)
    print('Wrote',len(centers),'center estimates in DN to',a.output)

if __name__=='__main__':main()
