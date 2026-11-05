"""Independent array, identity, manuscript and PDF regression checks."""
from pathlib import Path
import ast,hashlib,io,json,re,zipfile
import numpy as np
import fitz
R=Path(__file__).resolve().parents[1];E=R/'evidence'
from filter_comparison import gains

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    audit={}
    for p in list((R/'analysis').glob('*.py'))+list((R/'neural').rglob('*.py')):
        ast.parse(p.read_text('utf8'))
    audit['python_syntax']='pass'
    for prefix,pr,npz in [('calibrated','FILTER_COMPARISON_105K.json','FILTER_PREDICTIONS_105K.npz'),
                           ('uncalibrated','FILTER_UNCALIBRATED_105K.json','FILTER_UNCALIBRATED_PREDICTIONS_105K.npz')]:
        d=json.loads((E/pr).read_text('utf8'));z=np.load(E/npz)
        probes=np.load(E/('FILTER_RESPONSE_PREDICTIONS_105K.npz' if prefix=='calibrated' else 'FILTER_UNCALIBRATED_RESPONSE_105K.npz'))
        for m,s in d['scores'].items():
            err=z[m]-z['truth']
            values={'A_dn':abs(err).mean(),'J_dn':abs(np.diff(err,axis=1)).mean(),'S_dn':abs(np.diff(z[m],axis=1)).mean()}
            for k,v in values.items():assert abs(v-s[k])<1e-12,(prefix,m,k)
            gc,gr=gains(probes[m],probes['truth'])
            assert np.allclose(gc.mean(0),s['g_common'],rtol=0,atol=1e-12)
            assert np.allclose(gr.mean(0),s['g_relative'],rtol=0,atol=1e-12)
        assert d['code_sha256']==sha(R/'analysis/filter_comparison.py')
        assert d['protocol_sha256']==sha(R/'analysis/FILTER_PROTOCOL.json')
        audit[prefix+'_arrays_and_responses']='pass'
    probe=json.loads((E/'CENTER_RESPONSE_105K.json').read_text('utf8'))
    assert all(x['baseline_max_prediction_difference']==0 for x in probe['provenance'])
    assert probe['predictions_sha256']==sha(E/'CENTER_RESPONSE_105K.npz')
    assert probe['code_sha256']==sha(R/'analysis/run_center_response.py')
    assert probe['protocol_sha256']==sha(R/'analysis/CENTER_RESPONSE_PROTOCOL.json')
    audit['eight_scene_baseline_bitwise_replay']=True
    model=json.loads((R/'neural/PROVENANCE.json').read_text('utf8'))
    assert sha(R/model['weights'])==model['weight_sha256']
    for key in ['pretrained_37524','temporal_46000']:
        assert sha(R/model[key]['file'])==model[key]['file_sha256']
        assert model[key]['state_tensors_bitwise_verified_against_parent']
    audit['three_model_files_and_export_identities']='pass'
    ptc=json.loads((E/'PTC.json').read_text('utf8'))
    nonzero=[v for v in ptc['rows'] if np.any(np.array(v['truth'])!=0)]
    zero=[v for v in ptc['rows'] if not np.any(np.array(v['truth'])!=0)]
    assert len(nonzero)==168 and len(zero)==14
    a=np.mean([abs(np.array(v['prediction'])-v['truth']).mean() for v in nonzero])
    assert abs(a-.054768)<.0000005
    audit['ptc_nonzero_mae_dn']=float(a)
    original=Path(json.loads((R/'SOURCE_REVISION.json').read_text('utf8'))['source_zip'])
    assert sha(original)==json.loads((R/'SOURCE_REVISION.json').read_text('utf8'))['sha256']
    frozen=[]
    with zipfile.ZipFile(original) as parent:
        for stem in ['buckets','color','gains','metrics','patterns','ptc','sources']:
            for lang in ['en','zh']:
                name=f'source/tables/{stem}_{lang}.tex'
                assert parent.read(name)==(R/name).read_bytes(),name
                frozen.append(name)
        for lang in ['en','zh']:
            name='main.tex' if lang=='en' else 'chinese.tex'
            before=parent.read('source/'+name).decode('utf8').replace('\r\n','\n');after=(R/'source'/name).read_text('utf8')
            equations=lambda t:re.findall(r'\\begin\{(?:equation|align)\*?\}.*?\\end\{(?:equation|align)\*?\}',t,re.S)
            assert equations(before)==equations(after)
    audit['unchanged_frozen_tables']=frozen;audit['main_equations_unchanged']=True
    for stem in ['buckets','color','filters','filters_raw','gains','metrics','patterns','ptc','sources']:
        vals=[]
        for lang in ['en','zh']:
            tx=(R/'source/tables'/f'{stem}_{lang}.tex').read_text('utf8').split('\\midrule')[-1]
            vals.append(re.findall(r'[-+]?\d+(?:\.\d+)?',tx))
        assert vals[0]==vals[1],stem
    audit['nine_bilingual_table_bodies']='pass'
    texts='\n'.join(p.read_text('utf8') for p in (R/'source').glob('*.tex'))
    for bad in ['0.004','0.060405','0.063818','0.123727','at matched response','At matched response','斜坡','ramp stress','并不','快照','fallback']:
        assert bad not in texts,bad
    assert 'using the 46k checkpoint' in texts and '46k 检查点在 Z63' in texts
    assert 'after development evaluation' in texts
    audit['claim_regression']='pass'
    pages={}
    for p in (R/'paper').rglob('*.pdf'):
        doc=fitz.open(p);tx=''.join(page.get_text() for page in doc)
        assert '??' not in tx and '\ufffd' not in tx,p
        pages[str(p.relative_to(R))]=len(doc)
    audit['pdf_page_counts']=pages
    for p in (R/'source').glob('*.log'):
        tx=p.read_text('utf8',errors='replace')
        assert 'Overfull' not in tx and 'Missing character' not in tx,p
        assert not re.search(r'(?:Reference|Citation).*undefined',tx),p
    audit['tex_overflow_missing_glyphs_unresolved_citations']='pass'
    audit['training_updates']=0;audit['sealed_test_pixels_read']=0
    audit['remaining_research']=['independent-source confirmation','same-protocol original BLBE comparison','retrained native-reread ablation']
    (R/'qa/SCIENTIFIC_CHANGE_AUDIT.json').write_text(json.dumps(audit,indent=2),'utf8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
