"""Frozen 105k, paired long-sequence probes; no optimizer or training writes.

The local source generator is required only for neural re-inference. Frozen
predictions allow the subsequent filter comparison to run without source images.
"""
import os
os.environ.update(TEMP='E:/Temp/bcm18', TMP='E:/Temp/bcm18',
    TORCH_HOME='E:/Caches/torch', PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / 'naf_z63_legacy_posttrain_20261030'
sys.path[:0] = [str(ROOT/'neural'), str(ROOT/'neural/source'), str(RUN),
               str(RUN/'synthetic_video_20261030')]
import build_video as video
from runtime import configure
from network import CONFIG, load_model
from adapter import inputs
from sampler import JointMotionSampler

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    protocol = json.loads((ROOT/'analysis/CENTER_RESPONSE_PROTOCOL.json').read_text('utf8'))
    weight = ROOT/'weights/nafble_105000.pt'
    assert sha(weight) == protocol['model_file_sha256']
    rows = json.loads((RUN/'DEV_SOURCES.json').read_text('utf8'))
    original = np.load(ROOT/'evidence/Z63_CONTINUOUS_105K.npz')
    centers, valid = np.arange(3,117), np.arange(6,114)
    configure()
    ckpt = torch.load(weight, map_location='cpu', weights_only=False)
    assert ckpt['phase_updates'] == 105000 and ckpt['model_config'] == CONFIG
    model = load_model(ckpt['model'], CONFIG, 'cuda').eval().requires_grad_(False)
    del ckpt
    all_p1, all_p7, all_y, provenance = [], [], [], []
    original_observe = JointMotionSampler.observe
    start = time.perf_counter()
    with torch.inference_mode():
        for source_index, row in enumerate(rows):
            seed = 2026110300 + source_index
            q = JointMotionSampler().sample(120, seed, mode='analog')
            assert np.array_equal(q['delta_dn'], original['truth'][source_index])
            r = q['fluctuation_dn'].astype(np.float64)
            h = r-r[valid].mean(0)
            h *= .3/np.max(np.abs(h), axis=0, keepdims=True)
            changes = [np.zeros_like(h), np.full_like(h,.3), h]
            assert max(np.max(np.abs(q['delta_dn']+d)) for d in changes) < 3
            captured = [[], [], []]
            def observe(self, clean, sample, noise_seed):
                latent, codes = original_observe(self, clean, sample, noise_seed)
                t = len(captured[0])
                captured[0].append((codes[0]-sample['reference_dn'][:,None,None]).astype(np.float32))
                for k in (1,2):
                    plus = np.floor(latent + changes[k][t][None,:,None,None] + .5).astype(np.int32)
                    captured[k].append((plus[0]-sample['reference_dn'][:,None,None]).astype(np.float32))
                return latent, codes
            def render_source(row=row, seed=seed):
                assert sha(row['path']) == row['sha256']
                a = np.asarray(Image.open(row['path']).convert('RGB'), dtype=np.float32)/255
                if a.shape[0]>a.shape[1]: a=np.rot90(a)
                a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
                rng=np.random.default_rng(seed+555)
                sy=int(rng.integers(a.shape[0]-3*video.H+1));sx=int(rng.integers(a.shape[1]-3*video.W+1))
                canvas=a[sy:sy+3*video.H,sx:sx+3*video.W].copy()
                canvas*=2./max(float(np.median(canvas)),.001)
                return canvas,row,[sy,sx],rng
            JointMotionSampler.observe = observe
            video.SEED=seed; video.source=render_source
            try:
                raw, clean, actual, _, crop, *_ = video.simulate()
            finally:
                JointMotionSampler.observe = original_observe
            del clean
            frames = [np.stack(x) for x in captured]
            del captured
            assert np.array_equal(frames[0],raw)
            del raw
            p1, p7 = [], []
            for k, data in enumerate(frames):
                one, seven = [], []
                for c in centers:
                    x=torch.from_numpy(np.ascontiguousarray(data[c-3:c+4]))[None].to('cuda')
                    out=model.forward_inputs(inputs(x,None), checkpoint=False)
                    one.append(out['single'][0,3].cpu().numpy())
                    seven.append(out['offline'][0,3].cpu().numpy())
                p1.append(np.asarray(one,dtype=np.float64));p7.append(np.asarray(seven,dtype=np.float64))
                print('source',source_index+1,'/8 probe',k+1,'/3',flush=True)
            discrepancy=max(np.max(abs(p1[0]-original['t1'][source_index])),
                            np.max(abs(p7[0]-original['t7'][source_index])))
            assert discrepancy<2e-6, discrepancy
            all_p1.append(p1);all_p7.append(p7)
            all_y.append([q['delta_dn']+d for d in changes])
            provenance.append(dict(source=row['name'],source_sha256=row['sha256'],seed=seed,
                profile=q['recording'],crop_yx=crop,baseline_max_prediction_difference=float(discrepancy),
                raw_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in frames]))
            del frames
    np.savez_compressed(ROOT/'evidence/CENTER_RESPONSE_105K.npz',
        t1=np.asarray(all_p1),t7=np.asarray(all_p7),truth=np.asarray(all_y),centers=centers,valid=valid)
    result=dict(status='PAIRED_SLIDING_CENTER_DEVELOPMENT_RESPONSE',checkpoint_updates=105000,
        model_file_sha256=sha(weight),protocol_sha256=sha(ROOT/'analysis/CENTER_RESPONSE_PROTOCOL.json'),
        code_sha256=sha(__file__),predictions_sha256=sha(ROOT/'evidence/CENTER_RESPONSE_105K.npz'),
        provenance=provenance,wall_seconds=time.perf_counter()-start,neural_updates=0,sealed_test_pixels_read=0)
    (ROOT/'evidence/CENTER_RESPONSE_105K.json').write_text(json.dumps(result,indent=2),'utf8')
    print('COMPLETE',result['wall_seconds'],flush=True)

if __name__=='__main__': main()
