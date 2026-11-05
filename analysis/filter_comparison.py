"""Finite seven-frame filters; source-held-out development evaluation at 105k.

Run with Python 3.10+, NumPy and SciPy. Inputs and all outputs are revision-relative.
No neural inference, training, raw-source access, or sealed-test access occurs here.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  TEMP='E:/Temp/bcm18', TMP='E:/Temp/bcm18')
from pathlib import Path
from functools import lru_cache
from itertools import product
import hashlib, json, time, platform
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence'
N, C = 7, 3

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def windows(x):
    return np.stack([x[:, i:i+7] for i in range(x.shape[1]-6)], axis=1)

@lru_cache(None)
def matrices(q, probability=0., jump=0.):
    pats = np.array(list(product((0, 1), repeat=6))) if jump else np.zeros((1, 6))
    cumulative = np.concatenate([np.zeros((len(pats), 1)),
                                  np.cumsum(q + pats*jump, axis=1)], axis=1)
    k = 1e6 + cumulative[:, np.minimum.outer(np.arange(7), np.arange(7))]
    cov = k + np.eye(7)[None]
    inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    assert np.all(sign > 0)
    # K(C^-1) = I - C^-1 avoids multiplying the diffuse 1e6 terms.
    weights = np.eye(7)[None] - inv
    counts = pats.sum(1)
    prior = (counts*np.log(probability) + (6-counts)*np.log1p(-probability)
             if jump else np.zeros(1))
    return inv, weights, logdet, prior

def apply(w, cfg, variance):
    """Input (..., 7, 4); return all seven states under exactly this segment."""
    family = cfg['family']
    if family == 'mean':
        return np.broadcast_to(w.mean(-2, keepdims=True), w.shape).copy()
    if family == 'median':
        return np.broadcast_to(np.median(w, axis=-2, keepdims=True), w.shape).copy()
    if family == 'exponential':
        k = cfg['rho'] ** np.abs(np.arange(7)[:, None]-np.arange(7)[None])
        return np.einsum('ij,...jc->...ic', k/k.sum(1, keepdims=True), w)
    inv, weights, logdet, prior = matrices(cfg['q'], cfg.get('p', 0.), cfg.get('v', 0.))
    if family == 'kalman':
        return np.einsum('ij,...jc->...ic', weights[0], w)
    # The posterior is evaluated separately for each CFA and each seven-frame window.
    shape = w.shape
    z = np.moveaxis(w, -1, -2).reshape(-1, 4, 7)
    quad = np.einsum('bci,pij,bcj->bcp', z, inv, z, optimize=True)
    logw = prior[None, None] - .5*(logdet[None, None] + quad/variance[None, :, None])
    posterior = np.exp(logw-logsumexp(logw, axis=-1, keepdims=True))
    path_mean = np.einsum('pij,bcj->bcpi', weights, z, optimize=True)
    y = np.einsum('bcp,bcpi->bci', posterior, path_mean)
    return np.moveaxis(y.reshape(shape[:-2]+(4, 7)), -2, -1)

def metrics(x, y):
    err = x-y
    return {'A_dn': np.abs(err).mean((1, 2)),
            'J_dn': np.abs(np.diff(err, axis=1)).mean((1, 2)),
            'S_dn': np.abs(np.diff(x, axis=1)).mean((1, 2))}

def gains(pred, truth):
    # All arrays are (scene, paired condition, sliding center, CFA).
    common = pred[:, 1]-pred[:, 0]
    dy = truth[:, 1]-truth[:, 0]
    gc = common.mean(1)/dy.mean(1)
    relative = pred[:, 2]-pred[:, 0]
    yr = truth[:, 2]-truth[:, 0]
    relative = relative-relative.mean(1, keepdims=True)
    yr = yr-yr.mean(1, keepdims=True)
    gr = (relative*yr).sum(1)/(yr*yr).sum(1)
    return gc, gr

def fit_calibration(gc, gr, base_gc, base_gr, target_gc, target_gr, train):
    """Shared-control extension: residual mixing plus per-CFA shrinkage.

    Fit two coefficients per CFA on other scenes to match T7 common/relative gains.
    This calibration is frozen before applying it to the held-out scene.
    """
    fc,fr,bc,br,tc,tr=[v[train].mean(0) for v in (gc,gr,base_gc,base_gr,target_gc,target_gr)]
    denominator=tc*(fr-br)-tr*(fc-bc)
    alpha=np.divide(tr*bc-tc*br,denominator,out=np.zeros(4),where=abs(denominator)>1e-12)
    alpha=np.clip(alpha,0.,1.)
    scale=np.clip(tc/(bc+alpha*(fc-bc)),0.,1.5)
    return alpha,scale

def main():
    protocol = json.loads((ROOT/'analysis/FILTER_PROTOCOL.json').read_text('utf-8'))
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--uncalibrated',action='store_true');args=parser.parse_args()
    protocol['response_calibration']=not args.uncalibrated
    output_stem='FILTER_UNCALIBRATED' if args.uncalibrated else 'FILTER_COMPARISON'
    with np.load(E/'Z63_CONTINUOUS_105K.npz') as z:
        t1, t7, y, valid = (z[k].copy() for k in ['t1', 't7', 'truth', 'valid'])
    assert t1.shape == t7.shape == (8, 114, 4) and y.shape == (8, 120, 4)
    assert np.array_equal(valid, np.arange(6, 114))
    cw = windows(t1)
    y = y[:, valid]
    dev = json.loads((E/'DEV_105K_FULL.json').read_text('utf-8'))
    rows = dev['rows']
    with np.load(E/'CENTER_RESPONSE_105K.npz') as probes:
        raw_probe1 = probes['t1'].copy()
        probe1 = raw_probe1[:,:,3:-3]
        probe7 = probes['t7'][:,:,3:-3].copy()
        probey = probes['truth'][:,:,valid].copy()
    probe_windows = np.stack([windows(raw_probe1[:,k]) for k in range(3)], axis=1)
    zero = np.array([[[r['single_prediction'] for r in rows if r['source_index']==s and r['recording']==rec and r['group']=='zero' and r['brightness_dn']==2.][0] for rec in ['DSC4194','DSC4200']] for s in range(8)])
    assert probe1.shape == probe7.shape == probey.shape == (8,3,108,4)
    target_gc, target_gr = gains(probe7, probey)
    base_gc, base_gr = gains(probe1, probey)
    cfgs = [{'family':'mean'}, {'family':'median'}]
    cfgs += [{'family':'exponential','rho':v} for v in protocol['grid']['exponential_rho']]
    cfgs += [{'family':'kalman','q':v} for v in protocol['grid']['kalman_q_over_r']]
    cfgs += [{'family':'jump','q':q,'p':p,'v':v} for q,p,v in product(
        protocol['grid']['jump_q_over_r'],protocol['grid']['jump_probability'],protocol['grid']['jump_variance_over_r'])]
    families = ['mean', 'median', 'exponential', 'kalman', 'jump']
    selected = {f: np.empty_like(t1[:, 3:-3]) for f in families}
    selected_probes = {f: np.empty_like(probe1) for f in families}
    fold_results, candidate_oof = [], {str(i):[] for i in range(len(cfgs))}
    timings = {f:[] for f in families}
    for holdout in range(8):
        train = np.array([i for i in range(8) if i != holdout])
        variance = np.var(zero[train].reshape(-1, 4), axis=0, ddof=1)
        assert np.all(variance > 0)
        candidates = []
        for ci, cfg in enumerate(cfgs):
            pred = apply(cw, cfg, variance)[:, :, C]
            pr = apply(probe_windows, cfg, variance)[..., C, :]
            gc, gr = gains(pr, probey)
            calibration={}
            if protocol.get('response_calibration',False):
                alpha,scale=fit_calibration(gc,gr,base_gc,base_gr,target_gc,target_gr,train)
                pred=scale*((1-alpha)*t1[:,3:-3]+alpha*pred)
                pr=scale*((1-alpha)*probe1+alpha*pr)
                gc,gr=gains(pr,probey)
                calibration={'alpha_per_cfa':alpha.tolist(),'scale_per_cfa':scale.tolist()}
            m = metrics(pred, y)
            distance = float(max(np.max(np.abs(gc[train].mean(0)-target_gc[train].mean(0))),
                                 np.max(np.abs(gr[train].mean(0)-target_gr[train].mean(0)))))
            score = {'id':ci,'config':cfg,'response_distance':distance,
                     'calibration':calibration,'train_A_dn':float(m['A_dn'][train].mean()),'train_J_dn':float(m['J_dn'][train].mean())}
            candidates.append((score, pred, pr))
            candidate_oof[str(ci)].append({'source':holdout,'A_dn':float(m['A_dn'][holdout]),
                'J_dn':float(m['J_dn'][holdout]),'g_common':gc[holdout].tolist(),'g_relative':gr[holdout].tolist()})
        picked = {}
        for family in families:
            options = [x for x in candidates if x[0]['config']['family']==family]
            match = [x for x in options if x[0]['response_distance'] <= protocol['response_tolerance']]
            if match:
                chosen = min(match,key=lambda x:(x[0]['train_J_dn'],x[0]['train_A_dn'],x[0]['id']))
            else:
                chosen = min(options,key=lambda x:(x[0]['response_distance'],x[0]['train_J_dn'],x[0]['train_A_dn'],x[0]['id']))
            s,pred,pr = chosen
            selected[family][holdout] = pred[holdout]
            selected_probes[family][holdout] = pr[holdout]
            picked[family] = dict(s,training_response_matched=bool(match))
            # One-thread, warm, batched application; covariance setup is excluded.
            fixture = cw[holdout]
            def timed_filter():
                output=apply(fixture,s['config'],variance)
                if s['calibration']:
                    aa=np.array(s['calibration']['alpha_per_cfa']);bb=np.array(s['calibration']['scale_per_cfa'])
                    output=bb*((1-aa)*fixture+aa*output)
                return output
            for _ in range(2):timed_filter()
            elapsed=[]
            for _ in range(7):
                start=time.perf_counter_ns();timed_filter()
                elapsed.append((time.perf_counter_ns()-start)/1e3/len(fixture))
            timings[family].append(float(np.median(elapsed)))
        fold_results.append({'held_out_source':holdout,'measurement_variance_dn2':variance.tolist(),'selection':picked})
        print('completed source',holdout+1,'/8',flush=True)
    allpred={'T1':t1[:,3:-3],'T7':t7[:,3:-3],**selected}
    allprobe={'T1':probe1,'T7':probe7,**selected_probes}
    scores={}
    for method,pred in allpred.items():
        m=metrics(pred,y);gc,gr=gains(allprobe[method],probey)
        scores[method]={**{k:float(v.mean()) for k,v in m.items()},
            'source_metrics':{k:v.tolist() for k,v in m.items()},
            'g_common':gc.mean(0).tolist(),'g_relative':gr.mean(0).tolist(),
            'source_g_common':gc.tolist(),'source_g_relative':gr.tolist(),
            'filter_us_per_center':float(np.mean(timings[method])) if method in timings else 0.,
            'source_filter_us_per_center':timings.get(method,[])}
    rng=np.random.default_rng(20261104);indices=rng.integers(0,8,(20000,8));paired={}
    for method in allpred:
        if method=='T7':continue
        paired[method]={}
        for metric in ['A_dn','J_dn']:
            delta=np.array(scores['T7']['source_metrics'][metric])-np.array(scores[method]['source_metrics'][metric])
            paired[method][metric]={'T7_minus_control_dn':float(delta.mean()),
                'exploratory_source_bootstrap95_dn':np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
                'T7_improved_sources':int((delta<0).sum())}
    frontier=[]
    for i,cfg in enumerate(cfgs):
        rr=candidate_oof[str(i)]
        a=np.mean([v['A_dn'] for v in rr]);j=np.mean([v['J_dn'] for v in rr])
        gc=np.mean([v['g_common'] for v in rr],0);gr=np.mean([v['g_relative'] for v in rr],0)
        distance=float(max(abs(gc-target_gc.mean(0)).max(),abs(gr-target_gr.mean(0)).max()))
        frontier.append({'config':cfg,'A_dn':float(a),'J_dn':float(j),'g_common':gc.tolist(),
                         'g_relative':gr.tolist(),'response_distance':distance})
    result={'status':'LEAVE_ONE_SCENE_OUT_DEVELOPMENT_ANALYSIS','checkpoint_updates':105000,
        'model_state_sha256':dev['model_sha'],'source_count':8,'continuous_frames_per_source':120,
        'scored_frames_inclusive':[6,113],'continuous_window_frames':7,'lookahead_frames':3,
        'probe_suite':'Paired 120-frame common and empirical-shaped relative probes; all methods measured only at the same sliding-seven-frame centers 6..113 used for A/J.',
        'response_calibration':protocol.get('response_calibration',False),
        'scores':scores,'folds':fold_results,'candidate_frontier':frontier,'T7_vs_controls':paired,
        'filter_timing':{'unit':'microseconds per center','statistic':'mean across 8 folds of median of 7 warm batched timings, batch=108',
            'scope':'CPU filtering only; covariance precomputation excluded; includes all-state posterior though center only is scored',
            'platform':platform.platform(),'processor':platform.processor(),'numpy':np.__version__,'threads':1},
        'inputs':{n:sha(E/n) for n in ['Z63_CONTINUOUS_105K.npz','DEV_105K_FULL.json','CENTER_RESPONSE_105K.npz']},
        'protocol_sha256':sha(ROOT/'analysis/FILTER_PROTOCOL.json'),'code_sha256':sha(__file__),
        'sealed_test_pixels_read':0,'neural_updates':0}
    (E/(output_stem+'_105K.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    np.savez_compressed(E/('FILTER_UNCALIBRATED_PREDICTIONS_105K.npz' if args.uncalibrated else 'FILTER_PREDICTIONS_105K.npz'),truth=y,valid=valid,**allpred)
    np.savez_compressed(E/('FILTER_UNCALIBRATED_RESPONSE_105K.npz' if args.uncalibrated else 'FILTER_RESPONSE_PREDICTIONS_105K.npz'),truth=probey,valid=valid,**allprobe)
    print(json.dumps({m:{k:v[k] for k in ['A_dn','J_dn','g_common','g_relative','filter_us_per_center']} for m,v in scores.items()},indent=2),flush=True)

if __name__=='__main__':main()
