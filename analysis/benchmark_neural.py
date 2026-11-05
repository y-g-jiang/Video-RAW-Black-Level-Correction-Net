"""Inference-only seven-frame timing fixture; frozen model and no source images."""
import os
os.environ.update(TEMP='E:/Temp/bcm18',TMP='E:/Temp/bcm18',PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_HOME='E:/Caches/torch')
from pathlib import Path
import sys,time,json,hashlib,platform,subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'neural'
sys.path[:0]=[str(RUN/'source'),str(RUN)]
from network import CONFIG,load_model
from adapter import inputs
SNAP=ROOT/'weights/nafble_105000.pt'
assert hashlib.sha256(SNAP.read_bytes()).hexdigest()=='cc42ebe94662b5e3f16987bc51f982623dcef5fb7d65b035192dcdc8d2eb964a'
torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
s=torch.load(SNAP,map_location='cpu',weights_only=False)
m=load_model(s['model'],CONFIG,'cuda').eval().requires_grad_(False);del s
torch.manual_seed(20261104)
raw=(torch.randn(1,7,4,160,240,device='cuda')*1.5+2).round()
x=inputs(raw,None)
cached_tokens=[]
with torch.inference_mode():
    for j in range(6):
        cached_tokens.append(m.spatial_tokens(*(x[k][:,j] for k in ['raw','rn','wb','ccm','meta','saturation']),tile=None))

def branch(mode):
    tokens=list(cached_tokens) if mode=='T7_cached' else []
    indices=([6] if mode=='T7_cached' else [3] if mode=='T1_frame' else range(7))
    for j in indices:
        args=tuple(x[k][:,j] for k in ['raw','rn','wb','ccm','meta','saturation'])
        tokens.append(m.spatial_tokens(*args,tile=None))
    dt=x['dt'][:,3:4] if mode=='T1_frame' else x['dt']
    segments=torch.ones_like(x['segment_start'][:,3:4]) if mode=='T1_frame' else torch.ones_like(x['segment_start']) if mode=='T1' else x['segment_start']
    out=m.infer_tokens(tokens,dt,segments,False)
    return 3*torch.tanh(out['offline_score']/3)

with torch.inference_mode():
    reference=m.forward_inputs(x,checkpoint=False)
    for mode,key in [('T1','single'),('T7','offline')]:
        assert torch.equal(branch(mode),reference[key]),mode
    assert torch.equal(branch('T1_frame'),reference['single'][:,3:4])
    assert torch.equal(branch('T7_cached'),reference['offline'])
    del reference
    modes=['T1','T7','T1_frame','T7_cached']
    for mode in modes:
        for _ in range(10):branch(mode)
    torch.cuda.synchronize()
    records={mode:[] for mode in modes};peaks={}
    for repeat in range(5):
        for mode in (modes if repeat%2==0 else modes[::-1]):
            torch.cuda.reset_peak_memory_stats()
            for _ in range(20):
                torch.cuda.synchronize();start=time.perf_counter_ns()
                branch(mode);torch.cuda.synchronize()
                records[mode].append((time.perf_counter_ns()-start)/1e6)
            peaks[mode]=max(peaks.get(mode,0),torch.cuda.max_memory_allocated())
        print('timing block',repeat+1,'/5',flush=True)
info={'status':'INFERENCE_ONLY_WARM_TIMING','gpu':torch.cuda.get_device_name(0),
      'cpu':platform.processor(),'platform':platform.platform(),'pytorch':torch.__version__,
      'cuda':torch.version.cuda,'dtype':'float32','batch':1,'packed_input_shape':[1,7,4,160,240],
      'bayer_frame_shape':[320,480],'fixture':'rounded Gaussian numerical fixture, no scene/test pixels',
      'output':'one valid center per independently recomputed seven-frame window; all seven branch outputs computed',
      'scope':'GPU-resident inputs; includes 7 spatial encodings and one inference branch, excludes adapter, decoding, ISP and CPU postfilter',
      'streaming_modes':{'T1_frame':'One new RAW frame encoded and estimated; prior T1 outputs cached for the seven-frame postfilter.',
                         'T7_cached':'Six previous spatial tokens cached; one new RAW frame encoded, then seven-frame joint inference. Native pixels retained for every reread.'},
      'warmup_per_mode':10,'timed_per_mode':100,'alternating_blocks':5,
      'equivalence_to_original_forward':'bitwise identical T1 and T7 outputs on fixture',
      'modes':{mode:{'median_ms_per_center':float(np.median(v)),
          'p10_p90_ms':np.quantile(v,[.1,.9]).tolist(),'all_ms':v,
          'peak_allocated_bytes':peaks[mode]} for mode,v in records.items()},
      'model_file_sha256':hashlib.sha256(SNAP.read_bytes()).hexdigest(),
      'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'training_updates':0,'test_pixels_read':0}
(ROOT/'evidence/NEURAL_RUNTIME_105K.json').write_text(json.dumps(info,indent=2),'utf-8')
print(json.dumps({m:{k:v for k,v in row.items() if k!='all_ms'} for m,row in info['modes'].items()},indent=2),flush=True)
