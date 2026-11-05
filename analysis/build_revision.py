"""Build both manuscripts and supplements, retaining cross-document aux files.

Uses Tectonic or XeLaTeX from PATH. TECTONIC_BINARY, TECTONIC_BUNDLE, and
TEX_SEARCH_PATHS override local convenience defaults.
"""
from pathlib import Path
import os,subprocess,shutil,sys
R=Path(__file__).resolve().parents[1]
Q=R/'qa';Q.mkdir(exist_ok=True)
os.environ.update(LOCALAPPDATA='E:/Caches/below_zero_tectonic',TEMP='E:/Temp/bcm18',TMP='E:/Temp/bcm18')
local=Path('E:/Tools/tectonic-0.15.0/tectonic.exe')
tectonic=os.environ.get('TECTONIC_BINARY') or shutil.which('tectonic') or (str(local) if local.exists() else None)
if tectonic:
    command=[tectonic,'--keep-logs','--keep-intermediates']
    bundle=os.environ.get('TECTONIC_BUNDLE','E:/Caches/below_zero_v5_resource_bundle')
    if Path(bundle).exists():command+=['--bundle',bundle]
    paths=os.environ.get('TEX_SEARCH_PATHS','').split(os.pathsep) if os.environ.get('TEX_SEARCH_PATHS') else [
        'E:/Caches/below_zero_v11_bundle_exact',
        'E:/Workspaces/single_frame_signed_raw_completion_20261025/outputs/SPACE_V2_ISO12800_JOT_LOCKED/V11_EDITORIAL_RESTRUCTURE_20261030/paper_zh']
    for p in paths:
        if Path(p).exists():command+=['-Z','search-path='+p]
else:
    engine=shutil.which('xelatex')
    if not engine:raise SystemExit('Install Tectonic or XeLaTeX, or set TECTONIC_BINARY.')
    command=[engine,'-interaction=nonstopmode','-halt-on-error']
for name in ['main','chinese','supplement_en','supplement_zh']:
    for repeat in range(1 if tectonic else 2):
        result=subprocess.run(command+[name+'.tex'],cwd=R/'source',capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        (Q/('build_'+name+'.txt')).write_bytes(result.stdout+result.stderr)
        if result.returncode:
            print((result.stdout+result.stderr).decode('utf8',errors='replace')[-6000:]);sys.exit(result.returncode)
    target=R/'paper'/('zh' if name in ['chinese','supplement_zh'] else 'en')/('supplement.pdf' if name.startswith('supplement') else name+'.pdf')
    shutil.copy2(R/'source'/(name+'.pdf'),target)
    print('compiled',name,flush=True)
