"""Create the corrected release with a complete SHA-256 file manifest."""
from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parents[1]
DEST=Path('E:/UserData/bcm18/Downloads/BLE_TCI_Bilingual_Corrected_20261104.zip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    root_files=['LICENSE','SOURCE_REVISION.json','EDITORIAL_CHANGELOG.md',
        'CONTENT_MIGRATION.csv','CLAIM_EVIDENCE_MAP.md','RESEARCH_STATUS.md','LAYOUT_AND_LANGUAGE_AUDIT.json']
    files=[R/n for n in root_files]
    for folder in ['analysis','evidence','neural','weights','paper','source']:
        for p in (R/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.log','.aux','.out']:continue
            if folder=='source' and p.parent==R/'source' and p.suffix!='.tex':continue
            files.append(p)
    files += [p for p in (R/'qa').glob('*.json')]
    files=sorted(set(files));rows=[]
    for p in files:rows.append({'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest={'name':'Video-RAW-Black-Level-Correction-Net','version':'2026-11-04-corrected-sliding-center',
        'total_files':len(rows),'source_revision':json.loads((R/'SOURCE_REVISION.json').read_text('utf8')),
        'files':rows}
    (R/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),'utf8')
    tmp_dest=DEST.parent/(DEST.stem+'.tmp.zip')
    if tmp_dest.exists():tmp_dest.unlink()
    with zipfile.ZipFile(tmp_dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[R/'MANIFEST.json']:z.write(p,p.relative_to(R).as_posix())
    with zipfile.ZipFile(tmp_dest) as z:
        assert z.testzip() is None
        for row in rows:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    if DEST.exists():
        backup=DEST.parent/(DEST.stem+'_prev.zip')
        if backup.exists():backup.unlink()
        DEST.rename(backup)
    tmp_dest.replace(DEST)
    result={'zip':str(DEST),'bytes':DEST.stat().st_size,'sha256':sha(DEST),
        'entries':len(rows)+1,'manifest_files_verified':len(rows),'zip_crc':'pass'}
    (R/'PACKAGE_VERIFIED.json').write_text(json.dumps(result,indent=2),'utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
